import cv2
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.client.inference_client import InferenceClient

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create inference client
    client = InferenceClient(
        host='localhost',  # Change to your inference server address
        port=5000,
        frame_size=(320, 240),
        jpeg_quality=40
    )
    
    # Start the client
    if not client.start():
        print("Failed to start inference client")
        return
    
    print("Press 'q' to quit")
    
    try:
        while True:
            # Process a frame
            success, frame = client.process_frame()
            if not success:
                print("Failed to process frame")
                break
            
            # Get current metrics
            metrics = client.get_metrics()
            if metrics:
                # Display metrics on frame
                y_pos = 30
                for key, value in metrics.items():
                    text = f"{key}: {value}"
                    cv2.putText(frame, text, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_pos += 20
            
            # Display the frame
            cv2.imshow('Inference Stream', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        client.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 