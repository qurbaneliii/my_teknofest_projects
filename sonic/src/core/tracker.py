"""Object tracking logic using distance-based association."""

import logging
from typing import List, Dict
from .models import Detection, Track

logger = logging.getLogger(__name__)


class Tracker:
    """Manages multi-object tracking using nearest-neighbor association."""

    def __init__(
        self,
        distance_threshold: float = 120.0,
        max_age: int = 30,
    ):
        """Initialize tracker with association parameters.
        
        Args:
            distance_threshold: Max pixel distance for detection-track matching
            max_age: Max frames without detection before track deactivation
        """
        self.distance_threshold = distance_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0

    def update(self, detections: List[Detection], frame_id: int) -> List[Track]:
        """Associate detections to existing tracks and create new ones.
        
        Args:
            detections: List of detections from current frame
            frame_id: Current frame number
            
        Returns:
            List of active tracks after update
        """
        if not detections and not self.tracks:
            return []

        unmatched_detections = detections.copy()

        # Match detections to existing tracks
        for track in list(self.tracks.values()):
            if not track.is_active or not track.detections:
                continue

            best_detection = self._find_best_match(track, unmatched_detections)
            
            if best_detection:
                track.detections.append(best_detection)
                track.last_seen = best_detection.timestamp
                unmatched_detections.remove(best_detection)
            else:
                # Check if track expired
                last_frame = track.latest_detection.frame_id if track.latest_detection else 0
                if frame_id - last_frame > self.max_age:
                    track.is_active = False
                    logger.debug(f"Track {track.track_id} expired after {self.max_age} frames")

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track = Track(
                track_id=self.next_track_id,
                detections=[detection],
                last_seen=detection.timestamp,
            )
            self.tracks[self.next_track_id] = new_track
            logger.debug(f"Created track {self.next_track_id}")
            self.next_track_id += 1

        return self.active_tracks

    def _find_best_match(
        self,
        track: Track,
        detections: List[Detection],
    ) -> Detection | None:
        """Find closest detection to track's last position.
        
        Args:
            track: Track to match
            detections: Candidate detections
            
        Returns:
            Best matching detection or None if no match within threshold
        """
        if not track.latest_detection:
            return None

        last_cx, last_cy = track.latest_detection.center
        best_detection = None
        best_distance = self.distance_threshold

        for detection in detections:
            curr_cx, curr_cy = detection.center
            distance = ((curr_cx - last_cx) ** 2 + (curr_cy - last_cy) ** 2) ** 0.5

            if distance < best_distance:
                best_distance = distance
                best_detection = detection

        return best_detection

    @property
    def active_tracks(self) -> List[Track]:
        """Get all currently active tracks."""
        return [t for t in self.tracks.values() if t.is_active]

    def reset(self) -> None:
        """Clear all tracks and reset counter."""
        self.tracks.clear()
        self.next_track_id = 0
        logger.info("Tracker reset")
