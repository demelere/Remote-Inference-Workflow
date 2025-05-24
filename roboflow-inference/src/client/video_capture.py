import cv2
import logging
from typing import Optional, Tuple, Union

class VideoCapture:
    def __init__(self, source: Union[int, str] = 0):
        """Initialize video capture from webcam or video file.
        
        Args:
            source: Camera index (int) or video file path (str). Default is 0 (primary webcam).
        """
        self.source = source
        self.cap = None
        self.logger = logging.getLogger(__name__)

    def start(self) -> bool:
        """Start video capture.
        
        Returns:
            bool: True if capture started successfully, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video source: {self.source}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error starting video capture: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Read a frame from the video source.
        
        Returns:
            Tuple[bool, Optional[cv2.Mat]]: (success, frame) pair
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Failed to read frame")
            return False, None
        
        return True, frame

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the video frames.
        
        Returns:
            Tuple[int, int]: (width, height) of frames
        """
        if self.cap is None:
            return 0, 0
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def release(self):
        """Release the video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
