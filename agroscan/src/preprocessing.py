"""Preprocessing utilities for multispectral imagery.

This module provides functions for loading, normalizing, and processing
multispectral bands from various sources including GeoTIFF files.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image

logger = logging.getLogger(__name__)


def load_bands(
    image_path: Union[str, Path],
    band_indices: Optional[Tuple[int, ...]] = None,
) -> NDArray[np.floating]:
    """Load image bands from file.

    Supports common image formats (PNG, JPEG, TIFF). For multi-band
    GeoTIFF files, use specialized libraries like rasterio.

    Args:
        image_path: Path to image file.
        band_indices: Tuple of band indices to extract. If None, returns all bands.

    Returns:
        Array of shape (height, width, num_bands) or (height, width) for grayscale.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    arr = np.asarray(img, dtype=np.float64)

    # Normalize to 0-1 range if needed
    if arr.max() > 1.0:
        arr = arr / 255.0

    if band_indices is not None and arr.ndim == 3:
        arr = arr[:, :, list(band_indices)]

    return arr


def normalize_band(
    band: NDArray[np.floating],
    method: str = "minmax",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> NDArray[np.floating]:
    """Normalize band values to [0, 1] range.

    Args:
        band: Input band array.
        method: Normalization method:
            - "minmax": Scale by min/max values.
            - "percentile": Scale by percentile values (robust to outliers).
            - "zscore": Z-score normalization, then scale to [0, 1].
        percentile_low: Lower percentile for percentile method.
        percentile_high: Upper percentile for percentile method.

    Returns:
        Normalized band array with values in [0, 1].
    """
    band = np.asarray(band, dtype=np.float64)

    if method == "minmax":
        vmin, vmax = band.min(), band.max()
    elif method == "percentile":
        vmin = np.percentile(band, percentile_low)
        vmax = np.percentile(band, percentile_high)
    elif method == "zscore":
        mean, std = band.mean(), band.std()
        if std > 0:
            band = (band - mean) / std
        vmin, vmax = band.min(), band.max()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if vmax - vmin > 0:
        normalized = (band - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(band)

    return np.clip(normalized, 0.0, 1.0)


def create_rgb_composite(
    red: NDArray[np.floating],
    green: NDArray[np.floating],
    blue: NDArray[np.floating],
    normalize: bool = True,
) -> NDArray[np.uint8]:
    """Create RGB composite image from individual bands.

    Args:
        red: Red band array.
        green: Green band array.
        blue: Blue band array.
        normalize: Whether to normalize each band before compositing.

    Returns:
        RGB image array with shape (height, width, 3) and dtype uint8.
    """
    if not (red.shape == green.shape == blue.shape):
        raise ValueError("All bands must have the same shape")

    if normalize:
        red = normalize_band(red, method="percentile")
        green = normalize_band(green, method="percentile")
        blue = normalize_band(blue, method="percentile")

    rgb = np.stack([red, green, blue], axis=-1)
    return (rgb * 255).astype(np.uint8)


def create_false_color_composite(
    nir: NDArray[np.floating],
    red: NDArray[np.floating],
    green: NDArray[np.floating],
    normalize: bool = True,
) -> NDArray[np.uint8]:
    """Create false-color composite (NIR-R-G) for vegetation analysis.

    In false-color composites, vegetation appears bright red/pink,
    making it easy to identify healthy vegetation areas.

    Args:
        nir: Near-infrared band array.
        red: Red band array.
        green: Green band array.
        normalize: Whether to normalize each band before compositing.

    Returns:
        False-color RGB image array.
    """
    return create_rgb_composite(nir, red, green, normalize=normalize)


def compute_histogram(
    band: NDArray[np.floating],
    bins: int = 256,
    range: Optional[Tuple[float, float]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.floating]]:
    """Compute histogram of band values.

    Args:
        band: Input band array.
        bins: Number of histogram bins.
        range: Value range (min, max). If None, uses data range.

    Returns:
        Tuple of (histogram counts, bin edges).
    """
    if range is None:
        range = (float(band.min()), float(band.max()))

    hist, edges = np.histogram(band.flatten(), bins=bins, range=range)
    return hist, edges


def apply_mask(
    data: NDArray[np.floating],
    mask: NDArray[np.bool_],
    fill_value: float = np.nan,
) -> NDArray[np.floating]:
    """Apply binary mask to data array.

    Args:
        data: Input data array.
        mask: Boolean mask where True = valid pixels.
        fill_value: Value to use for masked pixels.

    Returns:
        Masked data array with invalid pixels set to fill_value.
    """
    result = data.copy()
    result[~mask] = fill_value
    return result


def generate_synthetic_bands(
    height: int = 256,
    width: int = 256,
    seed: Optional[int] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Generate synthetic multispectral bands for testing.

    Creates realistic-looking patterns with vegetation-like NDVI response.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (red, nir, blue) band arrays.
    """
    if seed is not None:
        np.random.seed(seed)

    # Create base pattern with gradients and noise
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)

    # Vegetation-like pattern
    pattern = 0.5 + 0.3 * np.sin(xx) * np.cos(yy)
    noise = np.random.uniform(-0.1, 0.1, (height, width))
    veg_mask = pattern + noise

    # Generate bands with realistic reflectance relationships
    red = 0.1 + 0.3 * (1 - veg_mask) + np.random.uniform(0, 0.1, (height, width))
    nir = 0.3 + 0.5 * veg_mask + np.random.uniform(0, 0.1, (height, width))
    blue = 0.05 + 0.2 * (1 - veg_mask) + np.random.uniform(0, 0.05, (height, width))

    red = np.clip(red, 0, 1)
    nir = np.clip(nir, 0, 1)
    blue = np.clip(blue, 0, 1)

    return red, nir, blue
