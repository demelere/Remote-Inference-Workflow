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

### 4. Run the client (coming soon)

## Project Status
🚧 Under Development

## License
MIT License (coming soon) 