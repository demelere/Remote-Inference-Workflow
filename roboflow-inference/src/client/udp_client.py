import cv2
import socket
import logging
import numpy as np
from typing import Optional, Tuple

class UDPClient:
    def __init__(self, host: str = 'localhost', port: int = 5000, buffer_size: int = 65507):
        """Initialize UDP client for sending frames.
        
        Args:
            host: Server hostname or IP
            port: Server UDP port
            buffer_size: Maximum UDP packet size (default is max UDP packet size - header)
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        """Create UDP socket.
        
        Returns:
            bool: True if socket created successfully
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create UDP socket: {e}")
            return False

    def send_frame(self, frame: np.ndarray) -> bool:
        """Send a frame over UDP.
        
        Args:
            frame: OpenCV/numpy image array
            
        Returns:
            bool: True if frame sent successfully
        """
        if self.socket is None:
            self.logger.error("Socket not initialized")
            return False

        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            data = buffer.tobytes()
            
            # Check if frame exceeds buffer size
            if len(data) > self.buffer_size:
                self.logger.warning(f"Frame size ({len(data)}) exceeds UDP buffer size ({self.buffer_size})")
                return False
            
            # Send the frame
            self.socket.sendto(data, (self.host, self.port))
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending frame: {e}")
            return False

    def receive_prediction(self) -> Tuple[bool, Optional[bytes]]:
        """Receive prediction data from server.
        
        Returns:
            Tuple[bool, Optional[bytes]]: (success, data) pair
        """
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
