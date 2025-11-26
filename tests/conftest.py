"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from datetime import datetime
from sonic.src.core.models import Detection, Track


@pytest.fixture
def sample_detection():
    """Create a sample detection for testing."""
    return Detection(
        x=100,
        y=150,
        width=50,
        height=60,
        confidence=0.85,
        class_name="mouse",
        timestamp=datetime.now(),
        frame_id=42,
    )


@pytest.fixture
def sample_track(sample_detection):
    """Create a sample track with one detection."""
    return Track(
        track_id=1,
        detections=[sample_detection],
        last_seen=sample_detection.timestamp,
    )


@pytest.fixture
def dummy_frame():
    """Create a blank test frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)
