import json
from pathlib import Path

import pytest

from sonic.src.config import DetectorConfig


def test_config_validation_pass(tmp_path: Path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"confidence_threshold": 0.8, "model_path": "model.pt"}))
    cfg = DetectorConfig.from_file(cfg_path)
    assert cfg.confidence_threshold == 0.8
    cfg.validate()


def test_config_validation_fail(tmp_path: Path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"confidence_threshold": 2.0}))
    with pytest.raises(Exception):
        DetectorConfig.from_file(cfg_path)
