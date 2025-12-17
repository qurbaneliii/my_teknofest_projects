import numpy as np

from datetime import datetime

from sonic.src.core.models import Detection
from sonic.src.visualization import OverlayRenderer


def test_overlay_draws():
    renderer = OverlayRenderer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        Detection(
            x=10,
            y=10,
            width=20,
            height=20,
            confidence=0.9,
            class_name="mouse",
            timestamp=datetime.now(),
            frame_id=1,
        )
    ]
    out = renderer.draw_detections(frame, detections, confidence_threshold=0.5)
    assert out.shape == frame.shape
