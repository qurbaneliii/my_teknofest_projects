"""Vegetation index computation functions.

This module provides functions for computing various vegetation indices
from multispectral imagery, including NDVI, SAVI, and EVI.
"""

import numpy as np
from numpy.typing import NDArray


def compute_ndvi(
    nir: NDArray[np.floating],
    red: NDArray[np.floating],
    epsilon: float = 1e-10,
) -> NDArray[np.floating]:
    """Compute Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        nir: Near-infrared band array (values typically 0-1 or 0-255).
        red: Red band array (same shape as nir).
        epsilon: Small value to prevent division by zero.

    Returns:
        NDVI array with values in range [-1, 1].
        Healthy vegetation typically shows NDVI > 0.3.

    Example:
        >>> nir = np.array([[0.8, 0.7], [0.6, 0.5]])
        >>> red = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> ndvi = compute_ndvi(nir, red)
        >>> ndvi.shape
        (2, 2)
    """
    nir = np.asarray(nir, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)

    if nir.shape != red.shape:
        raise ValueError(f"Band shapes must match: NIR={nir.shape}, Red={red.shape}")

    numerator = nir - red
    denominator = nir + red + epsilon

    ndvi = numerator / denominator
    return np.clip(ndvi, -1.0, 1.0)


def compute_savi(
    nir: NDArray[np.floating],
    red: NDArray[np.floating],
    L: float = 0.5,
    epsilon: float = 1e-10,
) -> NDArray[np.floating]:
    """Compute Soil Adjusted Vegetation Index (SAVI).

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

    SAVI minimizes soil brightness influences, useful for areas
    with sparse vegetation cover.

    Args:
        nir: Near-infrared band array.
        red: Red band array.
        L: Soil brightness correction factor (0=high veg, 1=low veg).
           Default 0.5 works well for intermediate vegetation cover.
        epsilon: Small value to prevent division by zero.

    Returns:
        SAVI array with values typically in range [-1, 1].
    """
    nir = np.asarray(nir, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)

    if nir.shape != red.shape:
        raise ValueError(f"Band shapes must match: NIR={nir.shape}, Red={red.shape}")

    numerator = (nir - red) * (1 + L)
    denominator = nir + red + L + epsilon

    savi = numerator / denominator
    return np.clip(savi, -1.0, 1.0)


def compute_evi(
    nir: NDArray[np.floating],
    red: NDArray[np.floating],
    blue: NDArray[np.floating],
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
    epsilon: float = 1e-10,
) -> NDArray[np.floating]:
    """Compute Enhanced Vegetation Index (EVI).

    EVI = G * ((NIR - Red) / (NIR + C1*Red - C2*Blue + L))

    EVI is an optimized index designed to enhance the vegetation signal
    with improved sensitivity in high biomass regions and improved
    vegetation monitoring through a de-coupling of the canopy background
    signal and a reduction in atmosphere influences.

    Args:
        nir: Near-infrared band array.
        red: Red band array.
        blue: Blue band array.
        G: Gain factor (default 2.5).
        C1: Coefficient for red band atmospheric correction (default 6.0).
        C2: Coefficient for blue band atmospheric correction (default 7.5).
        L: Canopy background adjustment (default 1.0).
        epsilon: Small value to prevent division by zero.

    Returns:
        EVI array with values typically in range [-1, 1].
    """
    nir = np.asarray(nir, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)
    blue = np.asarray(blue, dtype=np.float64)

    if not (nir.shape == red.shape == blue.shape):
        raise ValueError(
            f"Band shapes must match: NIR={nir.shape}, Red={red.shape}, Blue={blue.shape}"
        )

    numerator = G * (nir - red)
    denominator = nir + C1 * red - C2 * blue + L + epsilon

    evi = numerator / denominator
    return np.clip(evi, -1.0, 1.0)


def compute_ndwi(
    nir: NDArray[np.floating],
    swir: NDArray[np.floating],
    epsilon: float = 1e-10,
) -> NDArray[np.floating]:
    """Compute Normalized Difference Water Index (NDWI).

    NDWI = (NIR - SWIR) / (NIR + SWIR)

    NDWI is used for monitoring water content in vegetation.
    Higher values indicate higher water content.

    Args:
        nir: Near-infrared band array.
        swir: Short-wave infrared band array.
        epsilon: Small value to prevent division by zero.

    Returns:
        NDWI array with values in range [-1, 1].
    """
    nir = np.asarray(nir, dtype=np.float64)
    swir = np.asarray(swir, dtype=np.float64)

    if nir.shape != swir.shape:
        raise ValueError(f"Band shapes must match: NIR={nir.shape}, SWIR={swir.shape}")

    numerator = nir - swir
    denominator = nir + swir + epsilon

    ndwi = numerator / denominator
    return np.clip(ndwi, -1.0, 1.0)
