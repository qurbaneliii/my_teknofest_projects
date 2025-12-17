"""Configuration management for SONIC detector with validation."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Configuration parameters for rat detection system."""
    
    # Detection parameters
    confidence_threshold: float = 0.7
    model_path: str = "models/best.pt"
    target_class: str = "mouse"
    
    # Tracking parameters
    track_distance_threshold: float = 120.0
    max_track_age: int = 30
    tracker_type: str = "simple"  # "simple" or "kalman"
    kalman_iou_threshold: float = 0.3
    kalman_min_hits: int = 3
    
    # Alert parameters
    alert_cooldown: float = 5.0
    enable_console_alerts: bool = True
    enable_file_alerts: bool = True
    enable_log_alerts: bool = True
    
    # Output parameters
    save_detections: bool = True
    output_dir: str = "detections"
    video_output: bool = True
    show_preview: bool = True
    dataset_dir: Optional[str] = None
    preprocess_output_dir: str = "outputs"
    preview_width: Optional[int] = None
    allow_missing_model: bool = True

    @classmethod
    def from_file(cls, path: str | Path) -> "DetectorConfig":
        """Load configuration from JSON file with validation."""
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            instance = cls()
            instance.validate()
            return instance

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded config from {config_path}")
            instance = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            instance.validate()
            return instance
        except ValidationError as exc:
            logger.error(f"Config validation failed for {config_path}: {exc}")
            raise
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file.
        
        Args:
            path: Target file path
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
            logger.info(f"Saved config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def validate(self) -> None:
        """Validate current configuration using pydantic rules."""

        class _ConfigModel(BaseModel):
            confidence_threshold: float = Field(default=0.7, ge=0, le=1)
            model_path: str = Field(default="models/best.pt", min_length=1)
            target_class: str = Field(default="mouse", min_length=1)
            track_distance_threshold: float = Field(default=120.0, gt=0)
            max_track_age: int = Field(default=30, gt=0)
            tracker_type: str = Field(default="simple", pattern=r"^(simple|kalman)$")
            kalman_iou_threshold: float = Field(default=0.3, ge=0, le=1)
            kalman_min_hits: int = Field(default=3, ge=1)
            alert_cooldown: float = Field(default=5.0, ge=0)
            enable_console_alerts: bool = True
            enable_file_alerts: bool = True
            enable_log_alerts: bool = True
            save_detections: bool = True
            output_dir: str = Field(default="detections", min_length=1)
            video_output: bool = True
            show_preview: bool = True
            dataset_dir: Optional[str] = None
            preprocess_output_dir: str = Field(default="outputs", min_length=1)
            preview_width: Optional[int] = Field(default=None, gt=0)
            allow_missing_model: bool = True

        _ConfigModel(**self.to_dict())
