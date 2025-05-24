# Roboflow Video Inference Client

A client application for streaming video/camera input to a Roboflow UDP inference server and receiving real-time predictions.

## Architecture
- **Client** (This Repository): MacBook-based video capture and streaming client
- **Server**: Cloud-hosted Roboflow UDP inference server (to be deployed on AWS/Lightning.ai)

## Requirements
- macOS (Tested on 2019 Intel MacBook)
- Python 3.9+
- Docker Desktop for Mac (for local testing)

## Setup

### 1. Clone this repository
```bash
git clone git@github.com:demelere/Remote-Inference-Workflow.git
cd Remote-Inference-Workflow
```

### 2. Set up virtual environment
```bash
# Create and activate virtual environment
cd roboflow-inference
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings:
# - Add your Roboflow API key
# - Configure server host/port
# - Set camera/video source
```

### 4. Testing

#### Test Video Capture
To test your camera/video input:
```bash
python tests/test_video_capture.py
```
This will open a window showing your camera feed. Press 'q' to quit.

#### Test UDP Streaming
To test video streaming with compression and metrics:
```bash
python tests/test_udp_streaming.py
```
This will:
- Start a local test UDP server
- Stream compressed video frames (320x240)
- Display real-time metrics including:
  - FPS
  - Frame sizes
  - Compression ratio
  - Buffer usage
Press 'q' to quit.

### 5. Run the client (coming soon)

## Project Status
🚧 Under Development

## TODO

### Configuration Management
- [ ] Move hardcoded values to configuration
- [ ] Add configuration validation
- [ ] Support different environments (dev/prod)
- [ ] Add configuration for frame processing parameters

### Video Capture Enhancements
- [ ] Add frame processing capabilities:
  - [ ] Resize frames
  - [ ] Format conversion (BGR to RGB)
  - [ ] Frame rate control
  - [ ] Frame buffering
- [ ] Support iPhone camera streaming:
  - [ ] Integrate with Continuity Camera
  - [ ] Add NSCameraUseContinuityCameraDeviceType to Info.plist
  - [ ] Handle AVCaptureDeviceTypeContinuityCamera
- [ ] Add support for multiple camera sources
- [ ] Add camera hot-swapping

### Performance Optimizations
- [ ] Implement frame skipping for performance
- [ ] Add async frame capture
- [ ] Optimize memory usage for continuous streaming
- [ ] Add performance monitoring

### Error Handling & Logging
- [ ] Improve error messages for camera permissions
- [ ] Add reconnection logic for lost connections
- [ ] Add detailed logging for debugging
- [ ] Add telemetry for monitoring

## License
MIT License (coming soon) 