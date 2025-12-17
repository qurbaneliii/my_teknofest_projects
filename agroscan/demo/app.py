"""AgroScan Streamlit Demo - Crop Health Analysis using Vegetation Indices.

This demo allows users to:
1. Upload multispectral images (Red and NIR bands)
2. Compute NDVI and other vegetation indices
3. Visualize crop stress classification
4. Download analysis results
"""

import io
import json
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agroscan.src.ndvi import compute_ndvi, compute_savi, compute_evi
from agroscan.src.stress import (
    classify_stress,
    compute_stress_statistics,
    stress_to_rgb,
    StressThresholds,
    StressLevel,
)
from agroscan.src.preprocessing import generate_synthetic_bands, normalize_band


# Page configuration
st.set_page_config(
    page_title="AgroScan - Crop Health Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_image_as_band(uploaded_file) -> np.ndarray:
    """Load uploaded file as normalized grayscale band."""
    img = Image.open(uploaded_file).convert("L")
    return np.array(img, dtype=np.float64) / 255.0


def create_ndvi_colormap(ndvi: np.ndarray) -> np.ndarray:
    """Create colorful NDVI visualization."""
    # Normalize to 0-1 range
    normalized = (ndvi + 1) / 2

    # Create RGB using a vegetation colormap
    rgb = np.zeros((*ndvi.shape, 3), dtype=np.uint8)

    # Red channel: high for low NDVI (stressed)
    rgb[:, :, 0] = np.clip((1 - normalized) * 255, 0, 255).astype(np.uint8)
    # Green channel: high for high NDVI (healthy)
    rgb[:, :, 1] = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    # Blue channel: low overall
    rgb[:, :, 2] = 50

    return rgb


def main():
    st.title("🌾 AgroScan - Crop Health Analysis")
    st.markdown(
        """
        Analyze crop health using **vegetation indices** computed from multispectral imagery.
        Upload Red and NIR band images, or use synthetic data for demonstration.
        """
    )

    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")

    data_source = st.sidebar.radio(
        "Data Source",
        ["🎲 Synthetic Demo", "📁 Upload Images"],
        index=0,
    )

    st.sidebar.subheader("Stress Thresholds")
    healthy_min = st.sidebar.slider("Healthy (min NDVI)", 0.3, 0.8, 0.6, 0.05)
    mild_min = st.sidebar.slider("Mild Stress (min NDVI)", 0.2, 0.6, 0.4, 0.05)
    moderate_min = st.sidebar.slider("Moderate Stress (min NDVI)", 0.1, 0.4, 0.25, 0.05)
    severe_min = st.sidebar.slider("Severe Stress (min NDVI)", 0.0, 0.2, 0.1, 0.05)

    index_type = st.sidebar.selectbox(
        "Vegetation Index",
        ["NDVI", "SAVI"],
        index=0,
    )

    if index_type == "SAVI":
        soil_factor = st.sidebar.slider("Soil Factor (L)", 0.0, 1.0, 0.5, 0.1)

    # Main content
    col1, col2 = st.columns(2)

    red_band = None
    nir_band = None
    blue_band = None

    if "🎲" in data_source:
        # Synthetic data
        st.info("Using synthetic multispectral data for demonstration.")

        size = st.sidebar.slider("Image Size", 64, 512, 256, 64)
        seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)

        red_band, nir_band, blue_band = generate_synthetic_bands(size, size, seed=int(seed))

        with col1:
            st.subheader("Red Band")
            st.image((red_band * 255).astype(np.uint8), use_container_width=True)

        with col2:
            st.subheader("NIR Band")
            st.image((nir_band * 255).astype(np.uint8), use_container_width=True)

    else:
        # Upload images
        with col1:
            st.subheader("Red Band")
            red_file = st.file_uploader(
                "Upload Red band image",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="red",
            )
            if red_file:
                red_band = load_image_as_band(red_file)
                st.image((red_band * 255).astype(np.uint8), use_container_width=True)

        with col2:
            st.subheader("NIR Band")
            nir_file = st.file_uploader(
                "Upload NIR band image",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="nir",
            )
            if nir_file:
                nir_band = load_image_as_band(nir_file)
                st.image((nir_band * 255).astype(np.uint8), use_container_width=True)

    # Analysis
    if red_band is not None and nir_band is not None:
        st.markdown("---")
        st.header("📊 Analysis Results")

        # Compute vegetation index
        if index_type == "NDVI":
            vi = compute_ndvi(nir_band, red_band)
            vi_name = "NDVI"
        else:
            vi = compute_savi(nir_band, red_band, L=soil_factor)
            vi_name = "SAVI"

        # Classify stress
        try:
            thresholds = StressThresholds(
                healthy_min=healthy_min,
                mild_min=mild_min,
                moderate_min=moderate_min,
                severe_min=severe_min,
            )
        except ValueError as e:
            st.error(f"Invalid thresholds: {e}")
            return

        stress = classify_stress(vi, thresholds)
        stats = compute_stress_statistics(stress)

        # Display results
        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.subheader(f"{vi_name} Map")
            ndvi_rgb = create_ndvi_colormap(vi)
            st.image(ndvi_rgb, use_container_width=True)
            st.caption(f"Range: [{vi.min():.3f}, {vi.max():.3f}], Mean: {vi.mean():.3f}")

        with result_col2:
            st.subheader("Stress Classification")
            stress_rgb = stress_to_rgb(stress)
            st.image(stress_rgb, use_container_width=True)

            # Legend
            st.markdown(
                """
                **Legend:**
                - 🟢 Healthy
                - 🟡 Mild Stress
                - 🟠 Moderate Stress
                - 🔴 Severe Stress
                - ⚫ Critical/No Vegetation
                """
            )

        with result_col3:
            st.subheader("Statistics")

            # Key metrics
            st.metric("Vegetation Coverage", f"{stats.get('vegetation_coverage', 0):.1f}%")
            st.metric("Average Stress Index", f"{stats.get('average_stress_index', 0):.2f}")

            # Distribution chart
            import pandas as pd

            dist_data = {
                "Level": ["Healthy", "Mild", "Moderate", "Severe", "Critical", "No Veg"],
                "Percentage": [
                    stats.get("healthy", 0),
                    stats.get("mild_stress", 0),
                    stats.get("moderate_stress", 0),
                    stats.get("severe_stress", 0),
                    stats.get("critical", 0),
                    stats.get("no_vegetation", 0),
                ],
            }
            df = pd.DataFrame(dist_data)
            st.bar_chart(df.set_index("Level"))

        # Download section
        st.markdown("---")
        st.subheader("📥 Download Results")

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            # NDVI image
            ndvi_img = Image.fromarray(ndvi_rgb, mode="RGB")
            buf = io.BytesIO()
            ndvi_img.save(buf, format="PNG")
            st.download_button(
                f"Download {vi_name} Map",
                buf.getvalue(),
                f"{vi_name.lower()}_map.png",
                "image/png",
            )

        with dl_col2:
            # Stress map
            stress_img = Image.fromarray(stress_rgb, mode="RGB")
            buf = io.BytesIO()
            stress_img.save(buf, format="PNG")
            st.download_button(
                "Download Stress Map",
                buf.getvalue(),
                "stress_map.png",
                "image/png",
            )

        with dl_col3:
            # Statistics JSON
            st.download_button(
                "Download Statistics",
                json.dumps(stats, indent=2),
                "statistics.json",
                "application/json",
            )

    else:
        st.info("👆 Upload Red and NIR band images or select synthetic demo to begin analysis.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>AgroScan v0.3.0 | Part of TEKNOFEST AI Projects</p>
            <p>🌱 Empowering precision agriculture with AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
