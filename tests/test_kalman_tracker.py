"""Tests for Kalman filter tracker."""

import numpy as np
import pytest
from datetime import datetime

from sonic.src.core.kalman_tracker import KalmanTracker, KalmanState


class MockDetection:
    """Mock detection for testing."""

    def __init__(self, bbox, confidence=0.9, label="mouse"):
        self.bbox = bbox
        self.confidence = confidence
        self.label = label


class TestKalmanState:
    """Test KalmanState class."""

    def test_from_bbox(self):
        """Test state initialization from bounding box."""
        bbox = (10, 20, 50, 60)  # x1, y1, x2, y2
        state = KalmanState.from_bbox(bbox)

        assert state.state.shape == (8,)
        # Check center
        assert state.state[0] == 30  # cx = (10+50)/2
        assert state.state[1] == 40  # cy = (20+60)/2
        # Check size
        assert state.state[2] == 40  # w = 50-10
        assert state.state[3] == 40  # h = 60-20
        # Velocities should be zero
        assert np.all(state.state[4:] == 0)

    def test_to_bbox(self):
        """Test converting state back to bounding box."""
        bbox = (10, 20, 50, 60)
        state = KalmanState.from_bbox(bbox)
        result = state.to_bbox()

        np.testing.assert_array_almost_equal(result, bbox)

    def test_initial_covariance(self):
        """Test initial covariance matrix."""
        state = KalmanState.from_bbox((0, 0, 10, 10))

        assert state.covariance.shape == (8, 8)
        # Position uncertainty should be lower than velocity
        assert state.covariance[0, 0] < state.covariance[4, 4]


class TestKalmanTracker:
    """Test KalmanTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = KalmanTracker(max_age=30, min_hits=3, iou_threshold=0.3)

        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert len(tracker.tracks) == 0

    def test_create_track(self):
        """Test track creation from detection."""
        tracker = KalmanTracker(min_hits=1)
        det = MockDetection((10, 10, 30, 30))

        tracks = tracker.update([det])

        assert len(tracker.tracks) == 1
        assert tracker.next_id == 2

    def test_track_association(self):
        """Test detection-to-track association."""
        tracker = KalmanTracker(min_hits=1, iou_threshold=0.1)

        # First frame
        det1 = MockDetection((10, 10, 30, 30))
        tracker.update([det1])

        # Second frame - same position
        det2 = MockDetection((11, 11, 31, 31))
        tracks = tracker.update([det2])

        # Should still have only 1 track (associated)
        assert len(tracker.tracks) == 1

    def test_multiple_tracks(self):
        """Test tracking multiple objects."""
        tracker = KalmanTracker(min_hits=1, iou_threshold=0.1)

        # Two detections far apart
        det1 = MockDetection((10, 10, 30, 30))
        det2 = MockDetection((100, 100, 120, 120))

        tracker.update([det1, det2])

        assert len(tracker.tracks) == 2

    def test_track_removal(self):
        """Test removal of stale tracks."""
        tracker = KalmanTracker(max_age=2, min_hits=1)

        det = MockDetection((10, 10, 30, 30))
        tracker.update([det])

        # Update without detections - track should age
        tracker.update([])
        assert len(tracker.tracks) == 1

        tracker.update([])
        assert len(tracker.tracks) == 1

        # After max_age+1 frames, track should be removed
        tracker.update([])
        assert len(tracker.tracks) == 0

    def test_prediction(self):
        """Test Kalman prediction step."""
        tracker = KalmanTracker(min_hits=1)

        # Detection moving right
        det1 = MockDetection((10, 10, 30, 30))
        tracker.update([det1])

        det2 = MockDetection((20, 10, 40, 30))
        tracker.update([det2])

        # Get track state - should have learned velocity
        track_id = list(tracker.tracks.keys())[0]
        state = tracker.tracks[track_id]

        # Velocity should be positive in x direction
        assert state.state[4] > 0  # vx > 0

    def test_iou_calculation(self):
        """Test IOU calculation."""
        tracker = KalmanTracker()

        # Perfect overlap
        iou = tracker._iou((0, 0, 10, 10), (0, 0, 10, 10))
        assert iou == 1.0

        # No overlap
        iou = tracker._iou((0, 0, 10, 10), (20, 20, 30, 30))
        assert iou == 0.0

        # Partial overlap
        iou = tracker._iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0 < iou < 1

    def test_reset(self):
        """Test tracker reset."""
        tracker = KalmanTracker(min_hits=1)

        det = MockDetection((10, 10, 30, 30))
        tracker.update([det])
        assert len(tracker.tracks) > 0

        tracker.reset()

        assert len(tracker.tracks) == 0
        assert tracker.next_id == 1
        assert tracker.frame_count == 0

    def test_confirmed_tracks(self):
        """Test that tracks need min_hits to be confirmed."""
        tracker = KalmanTracker(min_hits=3)

        det = MockDetection((10, 10, 30, 30))

        # First update - not confirmed
        tracks = tracker.update([det])
        assert len(tracks) == 0

        # Second update - still not confirmed
        tracks = tracker.update([det])
        assert len(tracks) == 0

        # Third update - now confirmed
        tracks = tracker.update([det])
        assert len(tracks) == 1


class TestKalmanIntegration:
    """Integration tests for Kalman tracker."""

    def test_linear_motion(self):
        """Test tracking object with linear motion."""
        tracker = KalmanTracker(min_hits=1, iou_threshold=0.1)

        # Object moving diagonally
        for i in range(10):
            det = MockDetection((10 + i * 5, 10 + i * 5, 30 + i * 5, 30 + i * 5))
            tracks = tracker.update([det])

        # Should have one track
        assert len(tracker.tracks) == 1

        # Track should have velocity
        track_id = list(tracker.tracks.keys())[0]
        state = tracker.tracks[track_id]
        assert state.state[4] > 0  # vx > 0
        assert state.state[5] > 0  # vy > 0

    def test_occlusion_handling(self):
        """Test handling temporary occlusions."""
        tracker = KalmanTracker(max_age=5, min_hits=1)

        # Object visible
        for _ in range(3):
            det = MockDetection((10, 10, 30, 30))
            tracker.update([det])

        track_count_before = len(tracker.tracks)

        # Object occluded for 2 frames
        tracker.update([])
        tracker.update([])

        # Track should still exist
        assert len(tracker.tracks) == track_count_before

        # Object reappears
        det = MockDetection((12, 12, 32, 32))
        tracker.update([det])

        # Should associate with existing track
        assert len(tracker.tracks) == track_count_before
