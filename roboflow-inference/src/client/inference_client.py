import cv2
import logging
from typing import Optional, Tuple, Dict
from .video_capture import VideoCapture
from .udp_client import UDPClient
from ..utils.prediction_handler import PredictionHandler

class InferenceClient:
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5000,
        frame_size: Tuple[int, int] = (320, 240),
        jpeg_quality: int = 40,
        camera_id: int = 0
    ):
        """Initialize the inference client.
        
        Args:
            host: Inference server hostname/IP
            port: Server UDP port
            frame_size: Target frame size (width, height)
            jpeg_quality: JPEG compression quality (1-100)
            camera_id: Camera device ID (default: 0 for primary camera)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.video_capture = VideoCapture(camera_id)
        self.udp_client = UDPClient(
            host=host,
            port=port,
            initial_jpeg_quality=jpeg_quality,
            target_size=frame_size
        )
        self.prediction_handler = PredictionHandler()
        
        # State tracking
        self.is_running = False
        self.latest_predictions = {}
        self.latest_metrics = {}

    def start(self) -> bool:
        """Start the inference client."""
        if self.is_running:
            self.logger.warning("Client is already running")
            return True

        # Start video capture
        if not self.video_capture.start():
            self.logger.error("Failed to start video capture")
            return False

        # Connect UDP client
        if not self.udp_client.connect():
            self.logger.error("Failed to connect UDP client")
            self.video_capture.release()
            return False

        self.is_running = True
        self.logger.info("Inference client started successfully")
        return True

    def process_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Process a single frame and return the annotated result."""
        if not self.is_running:
            self.logger.error("Client is not running")
            return False, None

        # Capture frame
        success, frame = self.video_capture.read_frame()
        if not success:
            self.logger.error("Failed to read frame")
            return False, None

        # Send frame and get prediction
        if self.udp_client.send_frame(frame):
            success, data = self.udp_client.receive_prediction()
            if success:
                self.latest_predictions = self.prediction_handler.parse_prediction(data)
                frame = self.prediction_handler.draw_predictions(frame, self.latest_predictions)
            else:
                self.logger.debug("No predictions received")
        else:
            self.logger.warning("Failed to send frame")

        # Update metrics
        self.latest_metrics = self.udp_client.get_metrics()

        return True, frame

    def get_metrics(self) -> Dict:
        """Get current performance metrics and prediction summary."""
        metrics = self.latest_metrics.copy()
        if self.latest_predictions:
            metrics['prediction_summary'] = self.prediction_handler.get_prediction_summary(self.latest_predictions)
        return metrics

    def stop(self):
        """Stop the inference client and release resources."""
        if not self.is_running:
            return

        self.video_capture.release()
        self.udp_client.close()
        self.is_running = False
        self.logger.info("Inference client stopped")
