# Architecture Overview

## System Components

### SONIC Rat Detection System

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Entry Point                        │
│                    (sonic.src.cli)                          │
└───────────────┬─────────────────────────────────────────────┘
                │
                ├──> DetectionSession
                │    ├─> DetectorConfig (config.py)
                │    ├─> Detector (core/detector.py)
                │    ├─> Tracker (core/tracker.py)
                │    ├─> OverlayRenderer (visualization/overlay.py)
                │    └─> AlertHandlers[] (alerts/handlers.py)
                │
                ▼
         Video/Image Input
                │
                ├──> Frame Processing Loop
                │    ├─> detect() → List[Detection]
                │    ├─> update_tracks() → List[Track]
                │    ├─> check_alerts()
                │    └─> draw_visualizations()
                │
                ▼
         Output (video/JSON/alerts)
```

## Data Flow

1. **Input Stage**: Video/camera/image loaded via OpenCV
2. **Detection Stage**: YOLOv8 inference on each frame
3. **Tracking Stage**: Associate detections to tracks (nearest-neighbor)
4. **Alert Stage**: Trigger handlers for new tracks (with cooldown)
5. **Visualization Stage**: Overlay boxes, labels, stats
6. **Output Stage**: Save video, JSON session summary

## Core Components

### `Detector` (core/detector.py)
- **Purpose**: Encapsulate YOLO model and inference
- **Key Methods**:
  - `detect(frame, frame_id)` → `List[Detection]`
- **Responsibilities**: Model loading, inference, confidence filtering

### `Tracker` (core/tracker.py)
- **Purpose**: Multi-object tracking using distance-based association
- **Key Methods**:
  - `update(detections, frame_id)` → `List[Track]`
  - `reset()` - Clear all tracks
- **Algorithm**: Nearest-neighbor with distance threshold

### `Detection` & `Track` (core/models.py)
- **Purpose**: Immutable detection and mutable track data structures
- **Key Properties**:
  - `Detection.center` - Bounding box center
  - `Track.latest_detection` - Most recent detection
  - `Track.length` - Number of detections

### `AlertHandler` (alerts/base.py)
- **Purpose**: Abstract interface for alert dispatchers
- **Implementations**:
  - `ConsoleAlertHandler` - Print to stdout
  - `FileAlertHandler` - Append to file
  - `LogAlertHandler` - Write to application log
- **Extension**: Subclass `AlertHandler` for custom notifications

### `OverlayRenderer` (visualization/overlay.py)
- **Purpose**: Draw bounding boxes and session info
- **Key Methods**:
  - `draw_detections(frame, detections)`
  - `draw_info_panel(frame, stats)`

### `DetectorConfig` (config.py)
- **Purpose**: Type-safe configuration management
- **Key Methods**:
  - `from_file(path)` - Load from JSON
  - `save(path)` - Write to JSON
  - `to_dict()` - Serialize

## Design Principles

### 1. Separation of Concerns
Each module has single responsibility:
- Detection ≠ Tracking ≠ Alerts ≠ Visualization

### 2. Dependency Injection
Components receive dependencies via constructor:
```python
detector = Detector(model_path="...", confidence_threshold=0.7)
tracker = Tracker(distance_threshold=120.0)
session = DetectionSession(config)  # Composes detector + tracker
```

### 3. Interface-Based Design
`AlertHandler` abstract base enables extensibility without modifying core

### 4. Immutability Where Possible
`Detection` is frozen dataclass; `Track` is mutable for incremental updates

### 5. Explicit Configuration
No side effects during init; config loaded/saved explicitly

## Testing Strategy

- **Unit Tests**: Individual components (detector, tracker, alerts)
- **Fixtures**: Reusable sample data (detection, track, frame)
- **Coverage**: Core logic, edge cases (stale tracks, empty detections)

## Extension Points

### Custom Alerts
```python
from sonic.src.alerts import AlertHandler

class WebhookAlert(AlertHandler):
    def send_alert(self, track: Track):
        requests.post("https://api.example.com/alert", json=track.to_dict())
```

### Custom Tracking
Replace `Tracker` with Hungarian algorithm or Kalman filter:
```python
from sonic.src.core.tracker import Tracker

class KalmanTracker(Tracker):
    def update(self, detections, frame_id):
        # Custom tracking logic
        pass
```

### Custom Visualizations
```python
from sonic.src.visualization import OverlayRenderer

class HeatmapRenderer(OverlayRenderer):
    def draw_detections(self, frame, detections):
        # Add heatmap visualization
        return super().draw_detections(frame, detections)
```

## Performance Considerations

- **Inference**: YOLOv8 is GPU-accelerated when available
- **Tracking**: O(N*M) nearest-neighbor; consider spatial indexing for >50 objects
- **Alerts**: Cooldown prevents spam; file I/O is non-blocking

## Future Improvements

- Async video processing (multi-threading for I/O)
- Spatial indexing (KD-tree) for tracking
- Hungarian algorithm for better multi-object association
- Model quantization for edge deployment
