"""Configuration management for SONIC detector."""

import json
import logging
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict

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

    @classmethod
    def from_file(cls, path: str | Path) -> "DetectorConfig":
        """Load configuration from JSON file.
        
        Args:
            path: Path to config file
            
        Returns:
            DetectorConfig instance
        """
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded config from {config_path}")
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls()

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
