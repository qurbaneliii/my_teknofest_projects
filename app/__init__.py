"""FastAPI service exposing SONIC detection and AgroScan NDVI endpoints."""

import io
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

# AgroScan imports
from agroscan.src.ndvi import compute_ndvi, compute_savi
from agroscan.src.stress import (
    classify_stress,
    compute_stress_statistics,
    stress_to_rgb,
    StressThresholds,
)
from agroscan.src.preprocessing import normalize_band

# SONIC imports (lazy to avoid ultralytics requirement)
try:
    from sonic.src.core.detector import Detector
    from sonic.src.config import DetectorConfig

    SONIC_AVAILABLE = True
except ImportError:
    SONIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TEKNOFEST AI Services",
    description="API for SONIC rodent detection and AgroScan crop health analysis",
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.3.0"
    sonic_available: bool = SONIC_AVAILABLE


class Detection(BaseModel):
    """Single detection result."""

    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    label: str = Field(..., description="Detection label")


class DetectionResponse(BaseModel):
    """Response for detection endpoint."""

    detections: List[Detection]
    count: int
    image_width: int
    image_height: int


class NDVIRequest(BaseModel):
    """Request model for NDVI computation (when using base64)."""

    healthy_threshold: float = Field(0.6, ge=0, le=1, description="NDVI threshold for healthy")


class StressStatistics(BaseModel):
    """Crop stress statistics."""

    healthy: float = Field(..., description="Percentage of healthy vegetation")
    mild_stress: float = Field(..., description="Percentage with mild stress")
    moderate_stress: float = Field(..., description="Percentage with moderate stress")
    severe_stress: float = Field(..., description="Percentage with severe stress")
    critical: float = Field(..., description="Percentage in critical condition")
    no_vegetation: float = Field(..., description="Percentage with no vegetation")
    vegetation_coverage: float = Field(..., description="Total vegetation coverage")
    average_stress_index: float = Field(..., description="Average stress index (0=healthy)")


class NDVIResponse(BaseModel):
    """Response for NDVI/stress analysis."""

    ndvi_min: float
    ndvi_max: float
    ndvi_mean: float
    statistics: StressStatistics


# ============== Global State ==============

detector: Optional["Detector"] = None


def get_detector() -> "Detector":
    """Get or initialize SONIC detector."""
    global detector
    if not SONIC_AVAILABLE:
        raise HTTPException(503, "SONIC module not available (ultralytics not installed)")
    if detector is None:
        try:
            detector = Detector(
                model_path="models/best.pt",
                class_name="mouse",
                confidence_threshold=0.7,
                allow_missing_model=True,
            )
        except ImportError as e:
            raise HTTPException(503, f"SONIC detector failed to initialize: {e}")
    return detector


# ============== Endpoints ==============


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health status."""
    return HealthResponse(sonic_available=SONIC_AVAILABLE)


@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_objects(
    image: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: float = 0.5,
) -> DetectionResponse:
    """Detect rodents in uploaded image using SONIC/YOLOv8."""
    det = get_detector()

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img_array = np.array(img)

        detections = det.detect(img_array)

        results = []
        for d in detections:
            if d.confidence >= confidence_threshold:
                results.append(
                    Detection(
                        x1=d.bbox[0],
                        y1=d.bbox[1],
                        x2=d.bbox[2],
                        y2=d.bbox[3],
                        confidence=d.confidence,
                        label=d.label,
                    )
                )

        return DetectionResponse(
            detections=results,
            count=len(results),
            image_width=img.width,
            image_height=img.height,
        )
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(500, f"Detection failed: {str(e)}")


@app.post("/api/v1/ndvi", response_model=NDVIResponse)
async def compute_ndvi_endpoint(
    red_band: UploadFile = File(..., description="Red band image"),
    nir_band: UploadFile = File(..., description="NIR band image"),
    healthy_threshold: float = 0.6,
) -> NDVIResponse:
    """Compute NDVI and stress classification from multispectral bands."""
    try:
        # Load images
        red_contents = await red_band.read()
        nir_contents = await nir_band.read()

        red_img = np.array(Image.open(io.BytesIO(red_contents)).convert("L")) / 255.0
        nir_img = np.array(Image.open(io.BytesIO(nir_contents)).convert("L")) / 255.0

        if red_img.shape != nir_img.shape:
            raise HTTPException(400, "Red and NIR bands must have same dimensions")

        # Compute NDVI
        ndvi = compute_ndvi(nir_img, red_img)

        # Classify stress
        thresholds = StressThresholds(healthy_min=healthy_threshold)
        stress = classify_stress(ndvi, thresholds)
        stats = compute_stress_statistics(stress)

        return NDVIResponse(
            ndvi_min=float(ndvi.min()),
            ndvi_max=float(ndvi.max()),
            ndvi_mean=float(ndvi.mean()),
            statistics=StressStatistics(**stats),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NDVI computation error: {e}")
        raise HTTPException(500, f"NDVI computation failed: {str(e)}")


@app.post("/api/v1/ndvi/image")
async def compute_ndvi_image(
    red_band: UploadFile = File(..., description="Red band image"),
    nir_band: UploadFile = File(..., description="NIR band image"),
    output_type: str = "ndvi",
) -> StreamingResponse:
    """Generate NDVI or stress map image.

    Args:
        output_type: "ndvi" for NDVI grayscale, "stress" for stress RGB map
    """
    try:
        red_contents = await red_band.read()
        nir_contents = await nir_band.read()

        red_img = np.array(Image.open(io.BytesIO(red_contents)).convert("L")) / 255.0
        nir_img = np.array(Image.open(io.BytesIO(nir_contents)).convert("L")) / 255.0

        ndvi = compute_ndvi(nir_img, red_img)

        if output_type == "stress":
            stress = classify_stress(ndvi)
            output_array = stress_to_rgb(stress)
            mode = "RGB"
        else:
            # NDVI grayscale: map [-1, 1] to [0, 255]
            output_array = ((ndvi + 1) / 2 * 255).astype(np.uint8)
            mode = "L"

        img = Image.fromarray(output_array, mode=mode)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"NDVI image generation error: {e}")
        raise HTTPException(500, f"Image generation failed: {str(e)}")


@app.post("/api/v1/savi", response_model=dict)
async def compute_savi_endpoint(
    red_band: UploadFile = File(..., description="Red band image"),
    nir_band: UploadFile = File(..., description="NIR band image"),
    soil_factor: float = 0.5,
) -> dict:
    """Compute Soil Adjusted Vegetation Index (SAVI)."""
    try:
        red_contents = await red_band.read()
        nir_contents = await nir_band.read()

        red_img = np.array(Image.open(io.BytesIO(red_contents)).convert("L")) / 255.0
        nir_img = np.array(Image.open(io.BytesIO(nir_contents)).convert("L")) / 255.0

        savi = compute_savi(nir_img, red_img, L=soil_factor)

        return {
            "savi_min": float(savi.min()),
            "savi_max": float(savi.max()),
            "savi_mean": float(savi.mean()),
            "soil_factor": soil_factor,
        }
    except Exception as e:
        logger.error(f"SAVI computation error: {e}")
        raise HTTPException(500, f"SAVI computation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
