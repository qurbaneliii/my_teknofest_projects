"""Data models for detections and tracks."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class Detection:
    """Immutable detection result from a single frame."""
    
    x: int
    y: int
    width: int
    height: int
    confidence: float
    class_name: str
    timestamp: datetime
    frame_id: int

    @property
    def center(self) -> tuple[int, int]:
        """Return center coordinates of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "timestamp": self.timestamp.isoformat(),
            "frame_id": self.frame_id,
        }


@dataclass
class Track:
    """Mutable track representing temporal sequence of detections."""
    
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    alert_sent: bool = False

    @property
    def first_detection(self) -> Detection | None:
        """Get earliest detection in track."""
        return self.detections[0] if self.detections else None

    @property
    def latest_detection(self) -> Detection | None:
        """Get most recent detection in track."""
        return self.detections[-1] if self.detections else None

    @property
    def length(self) -> int:
        """Number of detections in track."""
        return len(self.detections)

    def to_dict(self) -> dict:
        """Serialize track metadata."""
        first = self.first_detection
        return {
            "track_id": self.track_id,
            "is_active": self.is_active,
            "detection_count": self.length,
            "first_seen": first.timestamp.isoformat() if first else None,
            "last_seen": self.last_seen.isoformat(),
            "alert_sent": self.alert_sent,
        }
