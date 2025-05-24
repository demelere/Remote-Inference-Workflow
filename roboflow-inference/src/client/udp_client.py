import cv2
import time
import socket
import logging
import numpy as np
from typing import Optional, Tuple, Dict
from collections import deque

class UDPClient:
    def __init__(
        self, 
        host: str = 'localhost', 
        port: int = 5000, 
        buffer_size: int = 65507,
        initial_jpeg_quality: int = 60,
        min_jpeg_quality: int = 30,
        target_size: Optional[Tuple[int, int]] = None
    ):
        """Initialize UDP client for sending frames.
        
        Args:
            host: Server hostname or IP
            port: Server UDP port
            buffer_size: Maximum UDP packet size (default is max UDP packet size - header)
            initial_jpeg_quality: Starting JPEG quality (0-100)
            min_jpeg_quality: Minimum JPEG quality before resizing
            target_size: Optional (width, height) to resize frames to
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.initial_jpeg_quality = initial_jpeg_quality
        self.min_jpeg_quality = min_jpeg_quality
        self.target_size = target_size
        self.socket = None
        self.logger = logging.getLogger(__name__)
        
        # Monitoring metrics
        self.frame_sizes = deque(maxlen=100)
        self.compression_rates = deque(maxlen=100)
        self.frame_times = deque(maxlen=100)
        self.last_metrics_time = time.time()
        self.frames_sent = 0
        self.bytes_sent = 0

    def connect(self) -> bool:
        """Create UDP socket."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create UDP socket: {e}")
            return False

    def _resize_frame(self, frame: np.ndarray, scale_factor: float) -> np.ndarray:
        """Resize frame by a scale factor."""
        if scale_factor >= 1.0:
            return frame
        new_size = (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def _compress_frame(self, frame: np.ndarray, quality: int) -> Tuple[bool, bytes]:
        """Compress frame with given JPEG quality."""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            data = buffer.tobytes()
            return True, data
        except Exception as e:
            self.logger.error(f"Compression error: {e}")
            return False, b""

    def send_frame(self, frame: np.ndarray) -> bool:
        """Send a frame over UDP with adaptive compression."""
        if self.socket is None:
            self.logger.error("Socket not initialized")
            return False

        start_time = time.time()
        original_size = frame.shape[0] * frame.shape[1] * frame.shape[2]
        current_frame = frame.copy()
        
        try:
            # If target size is set, resize first
            if self.target_size:
                current_frame = cv2.resize(current_frame, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Start with initial quality
            current_quality = self.initial_jpeg_quality
            scale_factor = 1.0
            success = False
            
            while not success and scale_factor > 0.3:
                # Try current quality
                while current_quality >= self.min_jpeg_quality:
                    success, data = self._compress_frame(current_frame, current_quality)
                    if not success:
                        break
                        
                    if len(data) <= self.buffer_size:
                        # Successfully compressed to acceptable size
                        success = True
                        break
                    
                    # Reduce quality and try again
                    current_quality -= 5
                
                if success:
                    break
                    
                # If quality reduction didn't work, scale down the image
                scale_factor *= 0.8
                current_frame = self._resize_frame(current_frame, scale_factor)
                current_quality = self.initial_jpeg_quality  # Reset quality for new size
            
            if not success:
                self.logger.warning("Could not reduce frame to fit UDP packet")
                return False
            
            # Send the frame
            self.socket.sendto(data, (self.host, self.port))
            
            # Update metrics
            process_time = time.time() - start_time
            self.frame_sizes.append(len(data))
            self.compression_rates.append(len(data) / original_size)
            self.frame_times.append(process_time)
            self.frames_sent += 1
            self.bytes_sent += len(data)
            
            # Log metrics every second
            if time.time() - self.last_metrics_time > 1.0:
                self._log_metrics()
                self.last_metrics_time = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending frame: {e}")
            return False

    def _log_metrics(self):
        """Log current performance metrics."""
        if not self.frame_times:
            return
            
        avg_frame_size = sum(self.frame_sizes) / len(self.frame_sizes)
        avg_compression = sum(self.compression_rates) / len(self.compression_rates)
        avg_process_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
        
        # Calculate min/max frame sizes
        min_frame_size = min(self.frame_sizes) / 1024  # KB
        max_frame_size = max(self.frame_sizes) / 1024  # KB
        avg_frame_size_kb = avg_frame_size / 1024
        
        self.logger.info(
            f"Performance Metrics:\n"
            f"  - FPS: {fps:.1f}\n"
            f"  - Frame Sizes (KB): min={min_frame_size:.1f}, avg={avg_frame_size_kb:.1f}, max={max_frame_size:.1f}\n"
            f"  - Compression Ratio: {avg_compression:.3f}\n"
            f"  - JPEG Quality: {self.initial_jpeg_quality}\n"
            f"  - Buffer Usage: {(avg_frame_size/self.buffer_size)*100:.1f}%"
        )

    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        if not self.frame_times:
            return {}
            
        return {
            'avg_frame_size': sum(self.frame_sizes) / len(self.frame_sizes),
            'avg_compression': sum(self.compression_rates) / len(self.compression_rates),
            'avg_process_time': sum(self.frame_times) / len(self.frame_times),
            'current_quality': self.initial_jpeg_quality,
            'frames_sent': self.frames_sent,
            'bytes_sent': self.bytes_sent
        }

    def receive_prediction(self) -> Tuple[bool, Optional[bytes]]:
        """Receive prediction data from server."""
        if self.socket is None:
            return False, None

        try:
            data, _ = self.socket.recvfrom(self.buffer_size)
            return True, data
        except socket.timeout:
            self.logger.debug("Receive timeout")
            return False, None
        except Exception as e:
            self.logger.error(f"Error receiving prediction: {e}")
            return False, None

    def close(self):
        """Close the UDP socket."""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
