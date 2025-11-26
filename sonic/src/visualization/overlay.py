"""Visualization utilities for detection overlays."""

from typing import List
import cv2
import numpy as np
from sonic.src.core.models import Detection


class OverlayRenderer:
    """Handles drawing detection boxes and labels on frames."""

    def __init__(
        self,
        high_conf_color: tuple[int, int, int] = (0, 200, 0),
        low_conf_color: tuple[int, int, int] = (0, 200, 200),
        box_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        self.high_conf_color = high_conf_color
        self.low_conf_color = low_conf_color
        self.box_thickness = box_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        confidence_threshold: float = 0.7,
    ) -> np.ndarray:
        """Overlay bounding boxes and labels on frame.
        
        Args:
            frame: Input image (modified in-place)
            detections: List of detections to visualize
            confidence_threshold: Threshold for color coding
            
        Returns:
            Modified frame with overlays
        """
        for det in detections:
            color = self.high_conf_color if det.confidence >= confidence_threshold else self.low_conf_color
            
            # Draw bounding box
            cv2.rectangle(
                frame,
                (det.x, det.y),
                (det.x + det.width, det.y + det.height),
                color,
                self.box_thickness,
            )
            
            # Draw label background and text
            label = f"{det.class_name}:{det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
            cv2.rectangle(
                frame,
                (det.x, det.y - label_h - 6),
                (det.x + label_w, det.y),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (det.x, det.y - 4),
                self.font,
                self.font_scale,
                (0, 0, 0),
                1,
            )

        return frame

    def draw_info_panel(
        self,
        frame: np.ndarray,
        frame_count: int,
        detection_count: int,
        active_track_count: int,
    ) -> np.ndarray:
        """Draw session information overlay.
        
        Args:
            frame: Input image
            frame_count: Current frame number
            detection_count: Number of detections in current frame
            active_track_count: Number of active tracks
            
        Returns:
            Frame with info overlay
        """
        info_text = f"Frame {frame_count} | Det {detection_count} | Tracks {active_track_count}"
        cv2.putText(
            frame,
            info_text,
            (10, 24),
            self.font,
            0.6,
            (255, 255, 255),
            2,
        )
        return frame
