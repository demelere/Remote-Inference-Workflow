import cv2
import sys
import socket
import logging
import threading
import time
import json
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.client.video_capture import VideoCapture
from src.client.udp_client import UDPClient
from src.utils.prediction_handler import PredictionHandler

def run_test_server(host='localhost', port=5000):
    """Run a simple UDP echo server for testing."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((host, port))
    print(f"Test server listening on {host}:{port}")
    
    try:
        while True:
            data, addr = server_socket.recvfrom(65507)
            print(f"Received {len(data)} bytes")
            
            # Send back a mock prediction
            mock_prediction = {
                "predictions": [
                    {
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.3,
                        "height": 0.4,
                        "class": "person",
                        "confidence": 0.95
                    }
                ]
            }
            response = json.dumps(mock_prediction).encode('utf-8')
            server_socket.sendto(response, addr)
    except KeyboardInterrupt:
        print("Server stopping...")
    finally:
        server_socket.close()

def display_metrics(frame, metrics, prediction_summary="No predictions"):
    """Display metrics on the frame."""
    if not metrics:
        return frame
    
    # Create a copy of the frame
    frame_with_metrics = frame.copy()
    
    # Add black background for text
    cv2.rectangle(frame_with_metrics, (10, 10), (400, 170), (0, 0, 0), -1)
    
    # Add metrics text
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 30
    white = (255, 255, 255)
    
    fps = 1.0 / metrics['avg_process_time'] if metrics['avg_process_time'] > 0 else 0
    metrics_text = [
        f"FPS: {fps:.1f}",
        f"Frame Size: {metrics['avg_frame_size']/1024:.1f}KB",
        f"Buffer Usage: {(metrics['avg_frame_size']/65507)*100:.1f}%",
        f"Compression: {metrics['avg_compression']:.3f}",
        f"Quality: {metrics['current_quality']}",
        f"Total Frames: {metrics['frames_sent']}",
        f"Predictions: {prediction_summary}"
    ]
    
    for text in metrics_text:
        cv2.putText(frame_with_metrics, text, (20, y_pos), font, 0.6, white, 1)
        y_pos += 20
    
    return frame_with_metrics

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Start test server in a separate thread
    server_thread = threading.Thread(target=run_test_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Initialize components
    cap = VideoCapture()
    if not cap.start():
        print("Failed to start video capture")
        return
    
    width, height = cap.get_frame_dimensions()
    print(f"Original frame dimensions: {width}x{height}")
    
    client = UDPClient(
        initial_jpeg_quality=40,
        min_jpeg_quality=20,
        target_size=(320, 240)
    )
    
    if not client.connect():
        print("Failed to create UDP socket")
        cap.release()
        return

    prediction_handler = PredictionHandler()
    
    try:
        while True:
            # Capture frame
            success, frame = cap.read_frame()
            if not success:
                print("Failed to read frame")
                break
            
            # Send frame
            if client.send_frame(frame):
                # Receive prediction
                success, data = client.receive_prediction()
                if success:
                    predictions = prediction_handler.parse_prediction(data)
                    frame = prediction_handler.draw_predictions(frame, predictions)
                    prediction_summary = prediction_handler.get_prediction_summary(predictions)
                else:
                    prediction_summary = "No predictions received"
            else:
                prediction_summary = "Failed to send frame"
            
            # Get and display metrics
            metrics = client.get_metrics()
            frame_with_metrics = display_metrics(frame, metrics, prediction_summary)
            
            # Display the frame
            cv2.imshow('UDP Streaming Test', frame_with_metrics)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        client.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 