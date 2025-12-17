"""Tests for FastAPI endpoints."""

import io
import numpy as np
import pytest
from PIL import Image

# Skip if fastapi not installed
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_red_band():
    """Create sample red band image."""
    arr = np.random.uniform(0.1, 0.5, (64, 64))
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def sample_nir_band():
    """Create sample NIR band image."""
    arr = np.random.uniform(0.3, 0.9, (64, 64))
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health(self, client):
        """Test health endpoint returns OK."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "sonic_available" in data


class TestNDVIEndpoint:
    """Test NDVI computation endpoint."""

    def test_ndvi_success(self, client, sample_red_band, sample_nir_band):
        """Test successful NDVI computation."""
        response = client.post(
            "/api/v1/ndvi",
            files={
                "red_band": ("red.png", sample_red_band, "image/png"),
                "nir_band": ("nir.png", sample_nir_band, "image/png"),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "ndvi_min" in data
        assert "ndvi_max" in data
        assert "ndvi_mean" in data
        assert "statistics" in data
        assert -1 <= data["ndvi_min"] <= 1
        assert -1 <= data["ndvi_max"] <= 1

    def test_ndvi_with_threshold(self, client, sample_red_band, sample_nir_band):
        """Test NDVI with custom healthy threshold."""
        response = client.post(
            "/api/v1/ndvi?healthy_threshold=0.7",
            files={
                "red_band": ("red.png", sample_red_band, "image/png"),
                "nir_band": ("nir.png", sample_nir_band, "image/png"),
            },
        )

        assert response.status_code == 200

    def test_ndvi_missing_band(self, client, sample_red_band):
        """Test NDVI with missing NIR band."""
        response = client.post(
            "/api/v1/ndvi",
            files={
                "red_band": ("red.png", sample_red_band, "image/png"),
            },
        )

        assert response.status_code == 422  # Validation error


class TestNDVIImageEndpoint:
    """Test NDVI image generation endpoint."""

    def test_ndvi_image(self, client, sample_red_band, sample_nir_band):
        """Test NDVI image generation."""
        response = client.post(
            "/api/v1/ndvi/image?output_type=ndvi",
            files={
                "red_band": ("red.png", sample_red_band, "image/png"),
                "nir_band": ("nir.png", sample_nir_band, "image/png"),
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_stress_image(self, client, sample_red_band, sample_nir_band):
        """Test stress map image generation."""
        response = client.post(
            "/api/v1/ndvi/image?output_type=stress",
            files={
                "red_band": ("red.png", sample_red_band, "image/png"),
                "nir_band": ("nir.png", sample_nir_band, "image/png"),
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


class TestSAVIEndpoint:
    """Test SAVI computation endpoint."""

    def test_savi_success(self, client, sample_red_band, sample_nir_band):
        """Test successful SAVI computation."""
        response = client.post(
            "/api/v1/savi?soil_factor=0.5",
            files={
                "red_band": ("red.png", sample_red_band, "image/png"),
                "nir_band": ("nir.png", sample_nir_band, "image/png"),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "savi_min" in data
        assert "savi_max" in data
        assert "savi_mean" in data
        assert data["soil_factor"] == 0.5


class TestDetectionEndpoint:
    """Test detection endpoint (may fail without model)."""

    def test_detection_no_model(self, client, sample_red_band):
        """Detection should handle missing model gracefully."""
        response = client.post(
            "/api/v1/detect",
            files={
                "image": ("test.png", sample_red_band, "image/png"),
            },
        )

        # Should return error about missing model/module
        assert response.status_code in (200, 500, 503)
