"""Command-line interface for AgroScan crop health analysis."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np

from .ndvi import compute_ndvi, compute_savi, compute_evi
from .preprocessing import load_bands, normalize_band, generate_synthetic_bands
from .stress import classify_stress, compute_stress_statistics, stress_to_rgb, StressThresholds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.3.0", prog_name="AgroScan")
def main() -> None:
    """AgroScan - Drone-based crop health analytics using vegetation indices."""
    pass


@main.command()
@click.option(
    "--red",
    "-r",
    type=click.Path(exists=True),
    help="Path to red band image.",
)
@click.option(
    "--nir",
    "-n",
    type=click.Path(exists=True),
    help="Path to NIR band image.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="ndvi_output.png",
    help="Output path for NDVI image.",
)
@click.option(
    "--synthetic",
    "-s",
    is_flag=True,
    help="Use synthetic test data instead of input files.",
)
@click.option(
    "--size",
    type=int,
    default=256,
    help="Size for synthetic data (height=width).",
)
def ndvi(
    red: Optional[str],
    nir: Optional[str],
    output: str,
    synthetic: bool,
    size: int,
) -> None:
    """Compute NDVI from red and NIR bands."""
    try:
        if synthetic:
            logger.info(f"Generating synthetic bands ({size}x{size})")
            red_band, nir_band, _ = generate_synthetic_bands(size, size, seed=42)
        else:
            if not red or not nir:
                raise click.UsageError("Must provide --red and --nir paths or use --synthetic")
            logger.info(f"Loading bands: red={red}, nir={nir}")
            red_band = load_bands(red)
            nir_band = load_bands(nir)
            if red_band.ndim == 3:
                red_band = red_band[:, :, 0]
            if nir_band.ndim == 3:
                nir_band = nir_band[:, :, 0]

        logger.info("Computing NDVI...")
        ndvi_result = compute_ndvi(nir_band, red_band)

        # Save as image
        from PIL import Image

        # Map NDVI [-1, 1] to [0, 255] using green colormap
        ndvi_normalized = ((ndvi_result + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(ndvi_normalized, mode="L")
        img.save(output)

        logger.info(f"NDVI saved to {output}")
        logger.info(f"NDVI range: [{ndvi_result.min():.3f}, {ndvi_result.max():.3f}]")
        logger.info(f"NDVI mean: {ndvi_result.mean():.3f}")

    except Exception as e:
        logger.error(f"Error computing NDVI: {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--red",
    "-r",
    type=click.Path(exists=True),
    help="Path to red band image.",
)
@click.option(
    "--nir",
    "-n",
    type=click.Path(exists=True),
    help="Path to NIR band image.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="stress_output.png",
    help="Output path for stress classification image.",
)
@click.option(
    "--json-output",
    "-j",
    type=click.Path(),
    help="Output path for statistics JSON.",
)
@click.option(
    "--synthetic",
    "-s",
    is_flag=True,
    help="Use synthetic test data.",
)
@click.option(
    "--healthy-threshold",
    type=float,
    default=0.6,
    help="NDVI threshold for healthy classification.",
)
def stress(
    red: Optional[str],
    nir: Optional[str],
    output: str,
    json_output: Optional[str],
    synthetic: bool,
    healthy_threshold: float,
) -> None:
    """Classify crop stress from NDVI values."""
    try:
        if synthetic:
            logger.info("Generating synthetic bands...")
            red_band, nir_band, _ = generate_synthetic_bands(256, 256, seed=42)
        else:
            if not red or not nir:
                raise click.UsageError("Must provide --red and --nir paths or use --synthetic")
            red_band = load_bands(red)
            nir_band = load_bands(nir)
            if red_band.ndim == 3:
                red_band = red_band[:, :, 0]
            if nir_band.ndim == 3:
                nir_band = nir_band[:, :, 0]

        logger.info("Computing NDVI and stress classification...")
        ndvi_result = compute_ndvi(nir_band, red_band)

        thresholds = StressThresholds(healthy_min=healthy_threshold)
        stress_result = classify_stress(ndvi_result, thresholds)
        stats = compute_stress_statistics(stress_result)

        # Save RGB stress map
        from PIL import Image

        stress_rgb = stress_to_rgb(stress_result)
        img = Image.fromarray(stress_rgb, mode="RGB")
        img.save(output)
        logger.info(f"Stress map saved to {output}")

        # Save statistics
        if json_output:
            with open(json_output, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Statistics saved to {json_output}")

        # Print statistics
        click.echo("\nStress Distribution:")
        for level, pct in stats.items():
            if level not in ("vegetation_coverage", "average_stress_index"):
                click.echo(f"  {level}: {pct:.1f}%")
        click.echo(f"\nVegetation coverage: {stats.get('vegetation_coverage', 0):.1f}%")
        click.echo(f"Average stress index: {stats.get('average_stress_index', 0):.2f}")

    except Exception as e:
        logger.error(f"Error classifying stress: {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="sample_data",
    help="Directory to save sample data.",
)
@click.option(
    "--size",
    type=int,
    default=256,
    help="Size of generated images.",
)
def generate_samples(output_dir: str, size: int) -> None:
    """Generate sample multispectral data for testing."""
    from PIL import Image

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {size}x{size} sample bands...")
    red, nir, blue = generate_synthetic_bands(size, size, seed=42)

    # Save individual bands
    for name, band in [("red", red), ("nir", nir), ("blue", blue)]:
        band_uint8 = (band * 255).astype(np.uint8)
        img = Image.fromarray(band_uint8, mode="L")
        img.save(output_path / f"{name}_band.png")
        logger.info(f"Saved {name}_band.png")

    # Compute and save NDVI
    ndvi_result = compute_ndvi(nir, red)
    ndvi_uint8 = ((ndvi_result + 1) / 2 * 255).astype(np.uint8)
    Image.fromarray(ndvi_uint8, mode="L").save(output_path / "ndvi.png")

    # Compute and save stress
    stress_result = classify_stress(ndvi_result)
    stress_rgb = stress_to_rgb(stress_result)
    Image.fromarray(stress_rgb, mode="RGB").save(output_path / "stress.png")

    # Save statistics
    stats = compute_stress_statistics(stress_result)
    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"All samples saved to {output_path}")


if __name__ == "__main__":
    main()
