"""YOLOv8-based object detector for rat/mouse detection."""

import logging
from pathlib import Path
from datetime import datetime
from typing import List
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics") from e

from .models import Detection

logger = logging.getLogger(__name__)


class Detector:
    """Encapsulates YOLO model inference and detection filtering."""

    def __init__(
        self,
        model_path: str | Path = "models/best.pt",
        class_name: str = "mouse",
        confidence_threshold: float = 0.7,
    ):
        """Initialize detector with model and thresholds.
        
        Args:
            model_path: Path to YOLOv8 weights file
            class_name: Target class to detect (e.g., "mouse", "rat")
            confidence_threshold: Minimum confidence for valid detections
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.class_name = class_name
        self.confidence_threshold = confidence_threshold
        logger.info(f"Loaded model from {self.model_path}, target class: {class_name}")

    def detect(self, frame: np.ndarray, frame_id: int) -> List[Detection]:
        """Run inference on a single frame and return filtered detections.
        
        Args:
            frame: Input image as numpy array (BGR format)
            frame_id: Sequential frame identifier
            
        Returns:
            List of Detection objects above confidence threshold
        """
        try:
            results = self.model(frame, verbose=False)
        except Exception as e:
            logger.warning(f"Inference failed on frame {frame_id}: {e}")
            return []

        detections: List[Detection] = []
        timestamp = datetime.now()

        for result in results:
            boxes = getattr(result, "boxes", [])
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                except (IndexError, ValueError) as e:
                    logger.debug(f"Skipping malformed box: {e}")
                    continue

                class_name = self._get_class_name(cls)
                
                if class_name == self.class_name and conf >= self.confidence_threshold:
                    detections.append(
                        Detection(
                            x=x1,
                            y=y1,
                            width=x2 - x1,
                            height=y2 - y1,
                            confidence=conf,
                            class_name=class_name,
                            timestamp=timestamp,
                            frame_id=frame_id,
                        )
                    )

        return detections

    def _get_class_name(self, cls_idx: int) -> str:
        """Resolve class index to name from model metadata."""
        if hasattr(self.model, "names"):
            return self.model.names.get(cls_idx, f"class_{cls_idx}")
        return f"class_{cls_idx}"
