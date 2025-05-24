import cv2
import sys
import socket
import logging
import threading
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
    
    # Initialize UDP client
    client = UDPClient()
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
            
            # Display the frame
            cv2.imshow('UDP Streaming Test', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        client.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 