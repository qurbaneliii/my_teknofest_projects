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
Kalman filter tracker with Hungarian algorithm is now available:
```python
from sonic.src.core.kalman_tracker import KalmanTracker

tracker = KalmanTracker(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3,
)
tracks = tracker.update(detections)
```

Enable via configuration:
```json
{
    "tracker_type": "kalman",
    "kalman_iou_threshold": 0.3,
    "kalman_min_hits": 3
}
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
- **Tracking**: Kalman tracker uses Hungarian algorithm O(N³) for optimal assignment
- **Alerts**: Cooldown prevents spam; file I/O is non-blocking

## AgroScan Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AgroScan Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Multispectral Input    →    Vegetation Index    →   Output │
│  (Red, NIR, Blue)            Computation              Maps   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   agroscan.src                          │ │
│  │  ┌─────────────┐  ┌───────────┐  ┌──────────────────┐  │ │
│  │  │preprocessing│  │   ndvi    │  │     stress       │  │ │
│  │  │             │  │           │  │                  │  │ │
│  │  │load_bands() │──│compute_   │──│classify_stress() │  │ │
│  │  │normalize()  │  │ndvi/savi/ │  │stress_to_rgb()   │  │ │
│  │  │             │  │evi/ndwi   │  │compute_stats()   │  │ │
│  │  └─────────────┘  └───────────┘  └──────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Entry Points:                                               │
│  • CLI: agroscan ndvi --red r.png --nir n.png               │
│  • CLI: agroscan stress --red r.png --nir n.png             │
│  • API: POST /api/v1/ndvi                                    │
│  • Demo: streamlit run agroscan/demo/app.py                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Vegetation Indices

| Index | Formula | Use Case |
|-------|---------|----------|
| NDVI | (NIR-R)/(NIR+R) | General vegetation health |
| SAVI | ((NIR-R)/(NIR+R+L))*(1+L) | Sparse vegetation, soil correction |
| EVI | G*(NIR-R)/(NIR+C1*R-C2*B+L) | High biomass, atmospheric correction |
| NDWI | (NIR-SWIR)/(NIR+SWIR) | Water content in vegetation |

### Stress Classification

```
NDVI Value    →    Stress Level    →    Color
≥ 0.6              Healthy              Green
0.4 - 0.6          Mild Stress          Yellow-Green
0.25 - 0.4         Moderate Stress      Gold
0.1 - 0.25         Severe Stress        Orange
0 - 0.1            Critical             Red
< 0                No Vegetation        Brown
```

## Future Improvements

- Async video processing (multi-threading for I/O)
- Spatial indexing (KD-tree) for tracking  
- Model quantization for edge deployment
- GeoTIFF support for real orthomosaic tiles
- Time-series analysis for crop monitoring
