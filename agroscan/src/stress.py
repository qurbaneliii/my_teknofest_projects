"""Crop stress classification based on vegetation indices.

This module provides functions for classifying plant stress levels
from computed vegetation indices like NDVI.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class StressLevel(Enum):
    """Crop stress severity levels based on vegetation health."""

    HEALTHY = "healthy"
    MILD_STRESS = "mild_stress"
    MODERATE_STRESS = "moderate_stress"
    SEVERE_STRESS = "severe_stress"
    CRITICAL = "critical"
    NO_VEGETATION = "no_vegetation"


@dataclass
class StressThresholds:
    """Configurable thresholds for stress classification.

    Default values are calibrated for typical agricultural crops.
    Adjust based on crop type, growth stage, and regional conditions.
    """

    healthy_min: float = 0.6
    mild_min: float = 0.4
    moderate_min: float = 0.25
    severe_min: float = 0.1
    no_veg_max: float = 0.0

    def __post_init__(self) -> None:
        """Validate threshold ordering."""
        if not (
            self.healthy_min > self.mild_min > self.moderate_min > self.severe_min
        ):
            raise ValueError("Thresholds must be in decreasing order")


def classify_stress(
    ndvi: NDArray[np.floating],
    thresholds: Optional[StressThresholds] = None,
) -> NDArray[np.int32]:
    """Classify stress levels from NDVI values.

    Args:
        ndvi: NDVI array with values in range [-1, 1].
        thresholds: Custom stress thresholds. Uses defaults if None.

    Returns:
        Integer array where each value corresponds to a StressLevel:
            0 = HEALTHY
            1 = MILD_STRESS
            2 = MODERATE_STRESS
            3 = SEVERE_STRESS
            4 = CRITICAL
            5 = NO_VEGETATION
    """
    if thresholds is None:
        thresholds = StressThresholds()

    ndvi = np.asarray(ndvi, dtype=np.float64)
    stress = np.zeros(ndvi.shape, dtype=np.int32)

    # Classify from healthy to critical
    stress[ndvi >= thresholds.healthy_min] = 0  # HEALTHY
    stress[
        (ndvi >= thresholds.mild_min) & (ndvi < thresholds.healthy_min)
    ] = 1  # MILD_STRESS
    stress[
        (ndvi >= thresholds.moderate_min) & (ndvi < thresholds.mild_min)
    ] = 2  # MODERATE_STRESS
    stress[
        (ndvi >= thresholds.severe_min) & (ndvi < thresholds.moderate_min)
    ] = 3  # SEVERE_STRESS
    stress[
        (ndvi >= thresholds.no_veg_max) & (ndvi < thresholds.severe_min)
    ] = 4  # CRITICAL
    stress[ndvi < thresholds.no_veg_max] = 5  # NO_VEGETATION

    return stress


def stress_to_label(stress_value: int) -> StressLevel:
    """Convert integer stress value to StressLevel enum."""
    mapping = {
        0: StressLevel.HEALTHY,
        1: StressLevel.MILD_STRESS,
        2: StressLevel.MODERATE_STRESS,
        3: StressLevel.SEVERE_STRESS,
        4: StressLevel.CRITICAL,
        5: StressLevel.NO_VEGETATION,
    }
    return mapping.get(stress_value, StressLevel.NO_VEGETATION)


def compute_stress_statistics(
    stress: NDArray[np.int32],
) -> dict[str, float]:
    """Compute statistics for stress distribution.

    Args:
        stress: Integer stress classification array.

    Returns:
        Dictionary with percentage of each stress level and summary stats.
    """
    total_pixels = stress.size
    if total_pixels == 0:
        return {}

    stats: dict[str, float] = {}

    for level in StressLevel:
        level_value = list(StressLevel).index(level)
        count = np.sum(stress == level_value)
        percentage = (count / total_pixels) * 100
        stats[level.value] = round(percentage, 2)

    # Compute vegetation coverage (exclude NO_VEGETATION)
    veg_pixels = np.sum(stress < 5)
    stats["vegetation_coverage"] = round((veg_pixels / total_pixels) * 100, 2)

    # Compute average stress index (0=healthy, 4=critical, exclude no_veg)
    veg_stress = stress[stress < 5]
    if len(veg_stress) > 0:
        stats["average_stress_index"] = round(np.mean(veg_stress), 2)
    else:
        stats["average_stress_index"] = 0.0

    return stats


def get_stress_colormap() -> dict[int, Tuple[int, int, int]]:
    """Get RGB colormap for stress visualization.

    Returns:
        Dictionary mapping stress level to RGB tuple.
    """
    return {
        0: (34, 139, 34),  # Forest green - healthy
        1: (154, 205, 50),  # Yellow-green - mild stress
        2: (255, 215, 0),  # Gold - moderate stress
        3: (255, 140, 0),  # Dark orange - severe stress
        4: (220, 20, 60),  # Crimson - critical
        5: (139, 69, 19),  # Saddle brown - no vegetation
    }


def stress_to_rgb(stress: NDArray[np.int32]) -> NDArray[np.uint8]:
    """Convert stress classification to RGB image for visualization.

    Args:
        stress: Integer stress classification array.

    Returns:
        RGB image array with shape (*stress.shape, 3).
    """
    colormap = get_stress_colormap()
    rgb = np.zeros((*stress.shape, 3), dtype=np.uint8)

    for level, color in colormap.items():
        mask = stress == level
        rgb[mask] = color

    return rgb
