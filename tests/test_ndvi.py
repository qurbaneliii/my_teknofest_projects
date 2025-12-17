"""Tests for AgroScan NDVI computation functions."""

import numpy as np
import pytest

from agroscan.src.ndvi import compute_ndvi, compute_savi, compute_evi, compute_ndwi


class TestComputeNDVI:
    """Test cases for NDVI computation."""

    def test_ndvi_basic(self):
        """Test NDVI with simple arrays."""
        nir = np.array([[0.8, 0.7], [0.6, 0.5]])
        red = np.array([[0.1, 0.2], [0.3, 0.4]])

        ndvi = compute_ndvi(nir, red)

        assert ndvi.shape == (2, 2)
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0

    def test_ndvi_healthy_vegetation(self):
        """High NIR, low red should give high NDVI."""
        nir = np.array([[0.9]])
        red = np.array([[0.1]])

        ndvi = compute_ndvi(nir, red)

        assert ndvi[0, 0] > 0.5  # Healthy vegetation

    def test_ndvi_bare_soil(self):
        """Similar NIR and red should give low NDVI."""
        nir = np.array([[0.3]])
        red = np.array([[0.3]])

        ndvi = compute_ndvi(nir, red)

        assert abs(ndvi[0, 0]) < 0.1  # Near zero

    def test_ndvi_water(self):
        """High red, low NIR should give negative NDVI."""
        nir = np.array([[0.1]])
        red = np.array([[0.5]])

        ndvi = compute_ndvi(nir, red)

        assert ndvi[0, 0] < 0  # Negative for water

    def test_ndvi_shape_mismatch(self):
        """Should raise ValueError for mismatched shapes."""
        nir = np.array([[0.8, 0.7]])
        red = np.array([[0.1, 0.2, 0.3]])

        with pytest.raises(ValueError, match="Band shapes must match"):
            compute_ndvi(nir, red)

    def test_ndvi_clipping(self):
        """NDVI should be clipped to [-1, 1]."""
        nir = np.array([[1.0, 0.0]])
        red = np.array([[0.0, 1.0]])

        ndvi = compute_ndvi(nir, red)

        assert ndvi[0, 0] <= 1.0
        assert ndvi[0, 1] >= -1.0


class TestComputeSAVI:
    """Test cases for SAVI computation."""

    def test_savi_basic(self):
        """Test SAVI with default L factor."""
        nir = np.array([[0.8, 0.6]])
        red = np.array([[0.2, 0.3]])

        savi = compute_savi(nir, red)

        assert savi.shape == (1, 2)
        assert savi.min() >= -1.0
        assert savi.max() <= 1.0

    def test_savi_with_l_factor(self):
        """Test SAVI with different L values."""
        nir = np.array([[0.8]])
        red = np.array([[0.2]])

        savi_low = compute_savi(nir, red, L=0.25)
        savi_high = compute_savi(nir, red, L=1.0)

        # Different L should produce different results
        assert savi_low[0, 0] != savi_high[0, 0]

    def test_savi_shape_mismatch(self):
        """Should raise ValueError for mismatched shapes."""
        nir = np.zeros((2, 2))
        red = np.zeros((3, 3))

        with pytest.raises(ValueError):
            compute_savi(nir, red)


class TestComputeEVI:
    """Test cases for EVI computation."""

    def test_evi_basic(self):
        """Test EVI with three bands."""
        nir = np.array([[0.8]])
        red = np.array([[0.2]])
        blue = np.array([[0.1]])

        evi = compute_evi(nir, red, blue)

        assert evi.shape == (1, 1)
        assert -1.0 <= evi[0, 0] <= 1.0

    def test_evi_shape_mismatch(self):
        """Should raise ValueError for mismatched shapes."""
        nir = np.zeros((2, 2))
        red = np.zeros((2, 2))
        blue = np.zeros((3, 3))

        with pytest.raises(ValueError, match="Band shapes must match"):
            compute_evi(nir, red, blue)


class TestComputeNDWI:
    """Test cases for NDWI computation."""

    def test_ndwi_basic(self):
        """Test NDWI with NIR and SWIR bands."""
        nir = np.array([[0.6, 0.4]])
        swir = np.array([[0.3, 0.5]])

        ndwi = compute_ndwi(nir, swir)

        assert ndwi.shape == (1, 2)
        assert ndwi.min() >= -1.0
        assert ndwi.max() <= 1.0

    def test_ndwi_high_water_content(self):
        """High NIR, low SWIR should give positive NDWI."""
        nir = np.array([[0.8]])
        swir = np.array([[0.2]])

        ndwi = compute_ndwi(nir, swir)

        assert ndwi[0, 0] > 0


class TestIntegration:
    """Integration tests for NDVI module."""

    def test_large_array(self):
        """Test with large arrays for performance."""
        np.random.seed(42)
        nir = np.random.uniform(0.2, 0.9, (1024, 1024))
        red = np.random.uniform(0.1, 0.6, (1024, 1024))

        ndvi = compute_ndvi(nir, red)

        assert ndvi.shape == (1024, 1024)
        assert not np.any(np.isnan(ndvi))
        assert not np.any(np.isinf(ndvi))
