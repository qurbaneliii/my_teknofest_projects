"""Kalman Filter-based object tracker for improved tracking accuracy.

This module implements a Kalman filter tracker that provides smoother
tracking and better prediction of object positions, especially useful
for handling occlusions and fast-moving objects.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackResult:
    """Lightweight track result from Kalman tracker.
    
    This is a simpler output format than Track, designed for real-time
    tracking results without storing detection history.
    """
    
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]
    label: str
    confidence: float
    age: int
    velocity: Tuple[float, float]  # (vx, vy)


@dataclass
class KalmanState:
    """State vector for Kalman filter tracking.

    State: [x, y, w, h, vx, vy, vw, vh]
    where (x, y) is center, (w, h) is size, and v* are velocities.
    """

    state: np.ndarray  # 8D state vector
    covariance: np.ndarray  # 8x8 covariance matrix
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    @classmethod
    def from_bbox(cls, bbox: Tuple[float, float, float, float]) -> "KalmanState":
        """Create initial state from bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Initial state: position + size, zero velocities
        state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)

        # Initial covariance (high uncertainty for velocities)
        covariance = np.diag([10, 10, 10, 10, 1000, 1000, 1000, 1000]).astype(np.float64)

        return cls(state=state, covariance=covariance, hits=1)

    def to_bbox(self) -> Tuple[float, float, float, float]:
        """Convert state to bounding box [x1, y1, x2, y2]."""
        cx, cy, w, h = self.state[:4]
        w = max(w, 1)
        h = max(h, 1)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class KalmanTracker:
    """Multi-object tracker using Kalman filters and Hungarian algorithm.

    This tracker provides:
    - Smooth position estimation using Kalman filtering
    - Optimal detection-to-track assignment using Hungarian algorithm
    - Robust handling of missed detections and new objects
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
    ):
        """Initialize Kalman tracker.

        Args:
            max_age: Maximum frames to keep alive a track without detections.
            min_hits: Minimum hits before track is considered confirmed.
            iou_threshold: Minimum IOU for detection-track association.
            process_noise: Process noise multiplier for state transition.
            measurement_noise: Measurement noise multiplier.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.tracks: Dict[int, KalmanState] = {}
        self.track_labels: Dict[int, str] = {}
        self.track_confidences: Dict[int, float] = {}
        self.next_id = 1
        self.frame_count = 0

        # Kalman matrices (constant velocity model)
        dt = 1.0  # time step

        # State transition matrix
        self.F = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0, 0],
                [0, 1, 0, 0, 0, dt, 0, 0],
                [0, 0, 1, 0, 0, 0, dt, 0],
                [0, 0, 0, 1, 0, 0, 0, dt],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Measurement matrix (observe position and size only)
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )

        # Process noise covariance
        self.Q = np.eye(8, dtype=np.float64) * process_noise
        self.Q[4:, 4:] *= 0.01  # Lower noise for velocities

        # Measurement noise covariance
        self.R = np.eye(4, dtype=np.float64) * measurement_noise

    def _predict(self, state: KalmanState) -> None:
        """Predict next state using Kalman filter."""
        state.state = self.F @ state.state
        state.covariance = self.F @ state.covariance @ self.F.T + self.Q
        state.age += 1
        state.time_since_update += 1

    def _update(self, state: KalmanState, measurement: np.ndarray) -> None:
        """Update state with measurement using Kalman filter."""
        # Innovation
        y = measurement - self.H @ state.state

        # Innovation covariance
        S = self.H @ state.covariance @ self.H.T + self.R

        # Kalman gain
        K = state.covariance @ self.H.T @ np.linalg.inv(S)

        # Update state
        state.state = state.state + K @ y

        # Update covariance
        I = np.eye(8)
        state.covariance = (I - K @ self.H) @ state.covariance

        state.hits += 1
        state.time_since_update = 0

    def _iou(
        self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
    ) -> float:
        """Compute IOU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _associate(
        self, detections: List[Detection], track_ids: List[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using Hungarian algorithm.

        Returns:
            matches: List of (detection_idx, track_id) pairs
            unmatched_detections: Detection indices without tracks
            unmatched_tracks: Track IDs without detections
        """
        if len(detections) == 0:
            return [], [], list(track_ids)

        if len(track_ids) == 0:
            return [], list(range(len(detections))), []

        # Compute IOU cost matrix
        cost_matrix = np.zeros((len(detections), len(track_ids)))

        for d_idx, det in enumerate(detections):
            for t_idx, track_id in enumerate(track_ids):
                track_bbox = self.tracks[track_id].to_bbox()
                iou = self._iou(det.bbox, track_bbox)
                cost_matrix[d_idx, t_idx] = 1 - iou  # Convert to cost

        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(track_ids)

        for d_idx, t_idx in zip(row_indices, col_indices):
            track_id = track_ids[t_idx]

            if cost_matrix[d_idx, t_idx] > (1 - self.iou_threshold):
                # IOU too low, don't match
                continue

            matches.append((d_idx, track_id))
            unmatched_detections.discard(d_idx)
            unmatched_tracks.discard(track_id)

        return matches, list(unmatched_detections), list(unmatched_tracks)

    def update(self, detections: List[Detection]) -> List[TrackResult]:
        """Process new detections and return active tracks.

        Args:
            detections: List of Detection objects for current frame.

        Returns:
            List of confirmed Track objects.
        """
        self.frame_count += 1

        # Predict all existing tracks
        for track_id in list(self.tracks.keys()):
            self._predict(self.tracks[track_id])

        # Associate detections to tracks
        track_ids = list(self.tracks.keys())
        matches, unmatched_dets, unmatched_tracks = self._associate(detections, track_ids)

        # Update matched tracks
        for det_idx, track_id in matches:
            det = detections[det_idx]
            measurement = self._bbox_to_measurement(det.bbox)
            self._update(self.tracks[track_id], measurement)
            self.track_confidences[track_id] = det.confidence
            self.track_labels[track_id] = det.label

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._create_track(det)

        # Remove dead tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].time_since_update > self.max_age:
                self._delete_track(track_id)

        # Return confirmed tracks
        return self._get_confirmed_tracks()

    def _bbox_to_measurement(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Convert bbox to measurement vector [cx, cy, w, h]."""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float64)

    def _create_track(self, detection: Detection) -> int:
        """Create new track from detection."""
        track_id = self.next_id
        self.next_id += 1

        self.tracks[track_id] = KalmanState.from_bbox(detection.bbox)
        self.track_labels[track_id] = detection.label
        self.track_confidences[track_id] = detection.confidence

        logger.debug(f"Created track {track_id} at {detection.bbox}")
        return track_id

    def _delete_track(self, track_id: int) -> None:
        """Remove track."""
        if track_id in self.tracks:
            del self.tracks[track_id]
            del self.track_labels[track_id]
            del self.track_confidences[track_id]
            logger.debug(f"Deleted track {track_id}")

    def _get_confirmed_tracks(self) -> List[TrackResult]:
        """Get list of confirmed tracks (enough hits, recently updated)."""
        confirmed = []

        for track_id, state in self.tracks.items():
            if state.hits >= self.min_hits and state.time_since_update <= 1:
                bbox = state.to_bbox()
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                track = TrackResult(
                    track_id=track_id,
                    bbox=bbox,
                    center=center,
                    label=self.track_labels.get(track_id, "unknown"),
                    confidence=self.track_confidences.get(track_id, 0.0),
                    age=state.age,
                    velocity=(state.state[4], state.state[5]),  # vx, vy
                )
                confirmed.append(track)

        return confirmed

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.track_labels.clear()
        self.track_confidences.clear()
        self.next_id = 1
        self.frame_count = 0
        logger.info("Tracker reset")
