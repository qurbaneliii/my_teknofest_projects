"""Core detection and tracking logic for SONIC rat detection system."""

from .models import Detection, Track
from .detector import Detector
from .tracker import Tracker
from .kalman_tracker import KalmanTracker, KalmanState, TrackResult

__all__ = [
    "Detection",
    "Track",
    "TrackResult",
    "Detector",
    "Tracker",
    "KalmanTracker",
    "KalmanState",
]
