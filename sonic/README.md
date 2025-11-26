# SONIC - Smart Rodent Detection System

YOLOv8-powered rodent detection with modular tracking, configurable alerts, and visualization.

## Features

- **Real-time Detection**: YOLOv8 inference on video streams, images, or camera feeds
- **Multi-Object Tracking**: Distance-based association with configurable thresholds
- **Flexible Alerts**: Console, file, and log handlers with cooldown management
- **Visualization**: Bounding boxes, confidence scores, and session statistics overlay
- **Type-Safe Config**: JSON-based configuration with dataclass validation

## Installation

```bash
# From project root
pip install -e .

# Or just sonic dependencies
pip install ultralytics opencv-python numpy
```

## Usage

### Quick Start

```bash
# Copy example config
cp config.example.json config.json

# Edit config.json to set:
# - model_path: path to your YOLO weights
# - target_class: class name to detect (e.g., "mouse", "rat")
# - confidence_threshold: minimum confidence (0.0-1.0)

# Run on camera
python -m sonic.src.cli --camera

# Process video
python -m sonic.src.cli --video input.mp4 --output output.mp4

# Analyze single image
python -m sonic.src.cli --image frame.jpg
```

### Configuration Options

See `config.example.json` for full schema. Key parameters:

- `confidence_threshold`: Minimum detection confidence (default: 0.7)
- `track_distance_threshold`: Max pixel distance for track association (default: 120.0)
- `max_track_age`: Frames before track expiration (default: 30)
- `alert_cooldown`: Seconds between alerts (default: 5.0)
- `save_detections`: Export session results to JSON (default: true)

### Output

Results saved to `detections/session_<timestamp>.json`:

```json
{
  "session_info": {
    "total_frames": 1234,
    "total_detections": 56,
    "active_tracks": 3
  },
  "detections": [...],
  "tracks": [...]
}
```

## Architecture

```
sonic/src/
├── core/
│   ├── detector.py      # YOLO inference
│   ├── tracker.py       # Multi-object tracking
│   └── models.py        # Detection & Track dataclasses
├── alerts/
│   ├── base.py          # AlertHandler interface
│   └── handlers.py      # Console, File, Log implementations
├── visualization/
│   └── overlay.py       # OpenCV rendering
├── config.py            # DetectorConfig management
└── cli.py               # Main entry point
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Lint
ruff check sonic/

# Format
black sonic/
```

## Model Training

To train custom YOLOv8 weights:

1. Prepare dataset in YOLO format
2. Train: `yolo detect train data=dataset.yaml model=yolov8n.pt epochs=100`
3. Export: `yolo export model=runs/detect/train/weights/best.pt format=onnx`
4. Update `config.json` with new `model_path`

## Assets

- **Pitch Decks**: `assets/docs/` - EN/AZ presentations, evaluation reports
- **UI Mockups**: `assets/mockups/` - HTML prototypes for farmer interface
- **Dataset**: `assets/dataset/` - Field captures and sample images

## License

See project root LICENSE file.

## Contact

For technical support: qurbanelifeyzullayev@gmail.com
