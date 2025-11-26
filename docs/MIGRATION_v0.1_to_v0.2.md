# Migration Guide: v0.1 → v0.2

## Overview

Version 0.2 introduces a complete architectural refactor with modular components, comprehensive testing, and improved configuration management.

## Breaking Changes

### 1. File Structure

**Old:**
```
sonic/ExportBlock-7ea5f3ff-1aa7-4ba4-99f5-782a170eb01b-Part-1/
  _Mouse_Detection_using_YOLOv5_Object_De.py
```

**New:**
```
sonic/
  src/
    core/{detector.py, tracker.py, models.py}
    alerts/{base.py, handlers.py}
    visualization/overlay.py
    config.py
    cli.py
  assets/
    docs/
    mockups/
    dataset/
```

### 2. Entry Point

**Old:**
```bash
python sonic/ExportBlock-.../\_Mouse_Detection_using_YOLOv5_Object_De.py --camera
```

**New:**
```bash
python -m sonic.src.cli --camera
```

### 3. Configuration

**Old:**
- Config written on every detector initialization
- Side effects during object creation

**New:**
- Explicit config loading: `DetectorConfig.from_file("config.json")`
- Immutable dataclass with validation
- Example template: `sonic/config.example.json`

### 4. Imports

**Old:**
```python
from _Mouse_Detection_using_YOLOv5_Object_De import RatDetector, Detection
```

**New:**
```python
from sonic.src.core import Detector, Tracker, Detection, Track
from sonic.src.alerts import ConsoleAlertHandler, FileAlertHandler
from sonic.src.config import DetectorConfig
```

### 5. Alert Callbacks

**Old:**
```python
def my_alert(track: RatTrack):
    print(track.track_id)

detector.add_alert_callback(my_alert)
```

**New:**
```python
from sonic.src.alerts import AlertHandler

class CustomAlert(AlertHandler):
    def send_alert(self, track: Track):
        print(track.track_id)

session = DetectionSession(config)
session.alert_handlers.append(CustomAlert())
```

## Migration Steps

### For Users

1. **Update imports:**
   ```bash
   pip install -e .  # or pip install -r requirements.txt
   ```

2. **Create config file:**
   ```bash
   cp sonic/config.example.json config.json
   # Edit config.json with your settings
   ```

3. **Update CLI commands:**
   ```bash
   # Replace old script path with module invocation
   python -m sonic.src.cli --camera
   ```

### For Developers

1. **Install dev dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Update custom integrations:**
   - Replace `RatDetector` with `DetectionSession`
   - Implement `AlertHandler` interface for custom alerts
   - Use `DetectorConfig` for parameters

3. **Run tests:**
   ```bash
   pytest tests/
   ```

4. **Lint and format:**
   ```bash
   ruff check sonic/
   black sonic/
   ```

## Deprecated Features

- ❌ `_Mouse_Detection_using_YOLOv5_Object_De.py` (removed)
- ❌ Auto-writing config on init (use explicit save)
- ❌ Mixed alert callbacks as raw functions (use AlertHandler)

## New Features

- ✅ Modular architecture (core/alerts/visualization)
- ✅ Comprehensive test suite (pytest + fixtures)
- ✅ Type-safe configuration management
- ✅ Clean separation of concerns
- ✅ Pluggable alert system
- ✅ Project packaging support (`pyproject.toml`)

## Performance

No performance regressions. Refactor focuses on maintainability without sacrificing speed.

## Support

Questions? Open an issue or contact: qurbanelifeyzullayev@gmail.com
