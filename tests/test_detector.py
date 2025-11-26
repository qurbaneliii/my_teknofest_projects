"""Tests for core detection models."""

from datetime import datetime
from sonic.src.core.models import Detection, Track


def test_detection_creation():
    """Test Detection dataclass instantiation."""
    det = Detection(
        x=10, y=20, width=30, height=40,
        confidence=0.9, class_name="mouse",
        timestamp=datetime.now(), frame_id=1
    )
    assert det.x == 10
    assert det.confidence == 0.9


def test_detection_center(sample_detection):
    """Test Detection.center property."""
    cx, cy = sample_detection.center
    assert cx == 100 + 50 // 2
    assert cy == 150 + 60 // 2


def test_detection_to_dict(sample_detection):
    """Test Detection serialization."""
    data = sample_detection.to_dict()
    assert data["x"] == 100
    assert data["confidence"] == 0.85
    assert "timestamp" in data


def test_track_properties(sample_track, sample_detection):
    """Test Track accessor properties."""
    assert sample_track.first_detection == sample_detection
    assert sample_track.latest_detection == sample_detection
    assert sample_track.length == 1


def test_track_to_dict(sample_track):
    """Test Track serialization."""
    data = sample_track.to_dict()
    assert data["track_id"] == 1
    assert data["is_active"] is True
    assert data["detection_count"] == 1
