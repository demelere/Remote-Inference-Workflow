import cv2
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.client.video_capture import VideoCapture

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize video capture (using default webcam)
    cap = VideoCapture()
    
    if not cap.start():
        print("Failed to start video capture")
        return
    
    print(f"Video dimensions: {cap.get_frame_dimensions()}")
    
    try:
        while True:
            success, frame = cap.read_frame()
            if not success:
                print("Failed to read frame")
                break
            
            # Display the frame
            cv2.imshow('Video Test', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 