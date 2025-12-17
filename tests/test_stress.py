"""Tests for AgroScan stress classification functions."""

import numpy as np
import pytest

from agroscan.src.stress import (
    classify_stress,
    compute_stress_statistics,
    stress_to_rgb,
    stress_to_label,
    get_stress_colormap,
    StressLevel,
    StressThresholds,
)


class TestStressThresholds:
    """Test StressThresholds dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = StressThresholds()

        assert thresholds.healthy_min == 0.6
        assert thresholds.mild_min == 0.4
        assert thresholds.moderate_min == 0.25
        assert thresholds.severe_min == 0.1

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = StressThresholds(
            healthy_min=0.7,
            mild_min=0.5,
            moderate_min=0.3,
            severe_min=0.15,
        )

        assert thresholds.healthy_min == 0.7
        assert thresholds.mild_min == 0.5

    def test_invalid_threshold_order(self):
        """Invalid threshold order should raise ValueError."""
        with pytest.raises(ValueError, match="decreasing order"):
            StressThresholds(
                healthy_min=0.3,  # Lower than mild
                mild_min=0.5,
            )


class TestClassifyStress:
    """Test stress classification function."""

    def test_classify_healthy(self):
        """High NDVI should classify as healthy."""
        ndvi = np.array([[0.8, 0.7], [0.65, 0.9]])
        stress = classify_stress(ndvi)

        assert np.all(stress == 0)  # All healthy

    def test_classify_mild_stress(self):
        """Medium-high NDVI should classify as mild stress."""
        ndvi = np.array([[0.5, 0.45]])
        stress = classify_stress(ndvi)

        assert np.all(stress == 1)  # All mild stress

    def test_classify_moderate_stress(self):
        """Medium NDVI should classify as moderate stress."""
        ndvi = np.array([[0.3, 0.35]])
        stress = classify_stress(ndvi)

        assert np.all(stress == 2)  # All moderate stress

    def test_classify_severe_stress(self):
        """Low NDVI should classify as severe stress."""
        ndvi = np.array([[0.15, 0.2]])
        stress = classify_stress(ndvi)

        assert np.all(stress == 3)  # All severe stress

    def test_classify_critical(self):
        """Very low NDVI should classify as critical."""
        ndvi = np.array([[0.05, 0.08]])
        stress = classify_stress(ndvi)

        assert np.all(stress == 4)  # All critical

    def test_classify_no_vegetation(self):
        """Negative NDVI should classify as no vegetation."""
        ndvi = np.array([[-0.1, -0.5]])
        stress = classify_stress(ndvi)

        assert np.all(stress == 5)  # All no vegetation

    def test_classify_mixed(self):
        """Test mixed stress levels."""
        ndvi = np.array([[0.8, 0.5, 0.3, 0.15, 0.05, -0.1]])
        stress = classify_stress(ndvi)

        expected = np.array([[0, 1, 2, 3, 4, 5]])
        np.testing.assert_array_equal(stress, expected)

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        ndvi = np.array([[0.5]])  # Would be mild with defaults
        # Lower all thresholds to make 0.5 healthy
        thresholds = StressThresholds(
            healthy_min=0.4,
            mild_min=0.3,
            moderate_min=0.2,
            severe_min=0.1,
        )
        stress = classify_stress(ndvi, thresholds)

        assert stress[0, 0] == 0  # Healthy


class TestStressStatistics:
    """Test stress statistics computation."""

    def test_statistics_uniform(self):
        """Test statistics with uniform stress."""
        stress = np.zeros((10, 10), dtype=np.int32)  # All healthy
        stats = compute_stress_statistics(stress)

        assert stats["healthy"] == 100.0
        assert stats["mild_stress"] == 0.0
        assert stats["vegetation_coverage"] == 100.0

    def test_statistics_mixed(self):
        """Test statistics with mixed stress levels."""
        stress = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
        stats = compute_stress_statistics(stress)

        # Each level is ~16.67%
        assert abs(stats["healthy"] - 16.67) < 0.1
        assert abs(stats["no_vegetation"] - 16.67) < 0.1
        # Vegetation coverage excludes no_vegetation
        assert abs(stats["vegetation_coverage"] - 83.33) < 0.1

    def test_statistics_empty(self):
        """Test statistics with empty array."""
        stress = np.array([], dtype=np.int32).reshape(0, 0)
        stats = compute_stress_statistics(stress)

        assert stats == {}


class TestStressVisualization:
    """Test stress visualization functions."""

    def test_stress_to_rgb_shape(self):
        """RGB output should have correct shape."""
        stress = np.array([[0, 1], [2, 3]], dtype=np.int32)
        rgb = stress_to_rgb(stress)

        assert rgb.shape == (2, 2, 3)
        assert rgb.dtype == np.uint8

    def test_stress_to_rgb_colors(self):
        """Each stress level should have distinct color."""
        colormap = get_stress_colormap()
        stress = np.array([[0]], dtype=np.int32)
        rgb = stress_to_rgb(stress)

        expected = colormap[0]
        np.testing.assert_array_equal(rgb[0, 0], expected)

    def test_stress_to_label(self):
        """Test integer to enum conversion."""
        assert stress_to_label(0) == StressLevel.HEALTHY
        assert stress_to_label(1) == StressLevel.MILD_STRESS
        assert stress_to_label(5) == StressLevel.NO_VEGETATION
        assert stress_to_label(99) == StressLevel.NO_VEGETATION  # Invalid


class TestStressLevel:
    """Test StressLevel enum."""

    def test_enum_values(self):
        """Test enum string values."""
        assert StressLevel.HEALTHY.value == "healthy"
        assert StressLevel.CRITICAL.value == "critical"

    def test_all_levels(self):
        """All expected levels should exist."""
        expected = {"healthy", "mild_stress", "moderate_stress", "severe_stress", "critical", "no_vegetation"}
        actual = {level.value for level in StressLevel}
        assert expected == actual
