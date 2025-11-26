"""Concrete implementations of alert handlers."""

import logging
from pathlib import Path
from datetime import datetime
from .base import AlertHandler
from sonic.src.core.models import Track

logger = logging.getLogger(__name__)


class ConsoleAlertHandler(AlertHandler):
    """Print alerts to console/stdout."""

    def send_alert(self, track: Track) -> None:
        latest = track.latest_detection
        if not latest:
            return
        print(f"\n🚨 ALERT | Track {track.track_id} | "
              f"Confidence: {latest.confidence:.2f} | "
              f"Time: {latest.timestamp.strftime('%H:%M:%S')}")


class LogAlertHandler(AlertHandler):
    """Write alerts to application log."""

    def __init__(self, level: int = logging.WARNING):
        self.level = level

    def send_alert(self, track: Track) -> None:
        latest = track.latest_detection
        if not latest:
            return
        logger.log(
            self.level,
            f"Detection alert - Track {track.track_id}, "
            f"Class: {latest.class_name}, "
            f"Confidence: {latest.confidence:.2f}",
        )


class FileAlertHandler(AlertHandler):
    """Append alerts to a persistent file."""

    def __init__(self, filepath: str | Path = "alerts.txt"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def send_alert(self, track: Track) -> None:
        latest = track.latest_detection
        if not latest:
            return
        timestamp = datetime.now().isoformat()
        message = f"{timestamp} | Track {track.track_id} | {latest.class_name} | Conf: {latest.confidence:.2f}\n"
        
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(message)
        except IOError as e:
            logger.error(f"Failed to write alert to {self.filepath}: {e}")
