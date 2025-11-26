"""Tests for tracking logic."""

import pytest
from datetime import datetime
from sonic.src.core.tracker import Tracker
from sonic.src.core.models import Detection


def test_tracker_initialization():
    """Test Tracker instantiation."""
    tracker = Tracker(distance_threshold=100.0, max_age=20)
    assert tracker.distance_threshold == 100.0
    assert tracker.max_age == 20
    assert len(tracker.tracks) == 0


def test_tracker_creates_new_track(sample_detection):
    """Test that tracker creates track for first detection."""
    tracker = Tracker()
    tracks = tracker.update([sample_detection], frame_id=1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 0


def test_tracker_associates_close_detections():
    """Test that nearby detections are associated to same track."""
    tracker = Tracker(distance_threshold=50.0)
    
    # First detection
    det1 = Detection(x=100, y=100, width=50, height=50, confidence=0.9,
                     class_name="mouse", timestamp=datetime.now(), frame_id=1)
    tracker.update([det1], frame_id=1)
    
    # Close detection (within threshold)
    det2 = Detection(x=110, y=105, width=50, height=50, confidence=0.9,
                     class_name="mouse", timestamp=datetime.now(), frame_id=2)
    tracks = tracker.update([det2], frame_id=2)
    
    assert len(tracks) == 1
    assert tracks[0].length == 2


def test_tracker_creates_separate_track_for_distant_detection():
    """Test that far detections create new tracks."""
    tracker = Tracker(distance_threshold=50.0)
    
    det1 = Detection(x=100, y=100, width=50, height=50, confidence=0.9,
                     class_name="mouse", timestamp=datetime.now(), frame_id=1)
    tracker.update([det1], frame_id=1)
    
    # Far detection (beyond threshold)
    det2 = Detection(x=300, y=300, width=50, height=50, confidence=0.9,
                     class_name="mouse", timestamp=datetime.now(), frame_id=2)
    tracks = tracker.update([det2], frame_id=2)
    
    assert len(tracker.tracks) == 2


def test_tracker_deactivates_stale_tracks():
    """Test that old tracks are deactivated."""
    tracker = Tracker(max_age=3)
    
    det = Detection(x=100, y=100, width=50, height=50, confidence=0.9,
                    class_name="mouse", timestamp=datetime.now(), frame_id=1)
    tracker.update([det], frame_id=1)
    
    # Skip many frames with no detections
    tracker.update([], frame_id=10)
    
    assert len(tracker.active_tracks) == 0
