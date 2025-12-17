import json
from pathlib import Path

import cv2
import numpy as np

from sonic.src.cli import DetectionSession
from sonic.src.config import DetectorConfig


def make_config(tmp_path: Path) -> DetectorConfig:
    cfg = DetectorConfig(
        model_path="models/best.pt",  # path not checked by validation beyond non-empty
        show_preview=False,
        save_detections=False,
        dataset_dir=str(tmp_path / "dataset"),
        preprocess_output_dir=str(tmp_path / "out"),
    )
    cfg.validate()
    return cfg


def test_preprocess_dataset(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(dataset_dir / "sample.png"), img)

    cfg = make_config(tmp_path)
    session = DetectionSession(cfg)
    session.preprocess_dataset()

    manifest = Path(cfg.preprocess_output_dir) / "dataset_manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["count"] == 1


def test_simulate_alert(tmp_path: Path):
    cfg = make_config(tmp_path)
    session = DetectionSession(cfg)
    session.simulate_alert()
    # Track should be recorded in tracker
    assert len(session.tracker.tracks) >= 1
