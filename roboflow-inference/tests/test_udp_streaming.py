import cv2
import sys
import socket
import logging
import threading
import time
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.client.video_capture import VideoCapture
from src.client.udp_client import UDPClient

def run_test_server(host='localhost', port=5000):
    """Run a simple UDP echo server for testing."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((host, port))
    print(f"Test server listening on {host}:{port}")
    
    try:
        while True:
            data, addr = server_socket.recvfrom(65507)
            print(f"Received {len(data)} bytes")
            # Echo back a simple acknowledgment
            server_socket.sendto(b"ACK", addr)
    except KeyboardInterrupt:
        print("Server stopping...")
    finally:
        server_socket.close()

def display_metrics(frame, metrics):
    """Display metrics on the frame."""
    if not metrics:
        return frame
    
    # Create a copy of the frame
    frame_with_metrics = frame.copy()
    
    # Add black background for text
    cv2.rectangle(frame_with_metrics, (10, 10), (400, 150), (0, 0, 0), -1)
    
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
        f"Total Frames: {metrics['frames_sent']}"
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
    
    # Initialize video capture
    cap = VideoCapture()
    if not cap.start():
        print("Failed to start video capture")
        return
    
    # Get frame dimensions
    width, height = cap.get_frame_dimensions()
    print(f"Original frame dimensions: {width}x{height}")
    
    # Initialize UDP client with target size
    client = UDPClient(
        initial_jpeg_quality=40,
        min_jpeg_quality=20,
        target_size=(320, 240)
    )
    
    if not client.connect():
        print("Failed to create UDP socket")
        cap.release()
        return
    
    try:
        while True:
            # Capture frame
            success, frame = cap.read_frame()
            if not success:
                print("Failed to read frame")
                break
            
            # Send frame
            if client.send_frame(frame):
                # Receive acknowledgment
                success, data = client.receive_prediction()
                if success:
                    print(f"Received: {data.decode()}")
            
            # Get and display metrics
            metrics = client.get_metrics()
            frame_with_metrics = display_metrics(frame, metrics)
            
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