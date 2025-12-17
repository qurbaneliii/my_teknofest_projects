# SONIC (Smart Rodent Intelligence & Control)

## Architecture
- **Core**: YOLOv8 detector + tracker (`sonic/src/core/`)
- **Alerts**: Pluggable handlers (`sonic/src/alerts/`)
- **Visualization**: OpenCV overlay (`sonic/src/visualization/`)
- **CLI**: Session orchestration (`sonic/src/cli.py`)
- **Config**: Validated dataclass (`sonic/src/config.py`)

## Quickstart
```bash
pip install -e .
cp sonic/config.example.json config.json
python -m sonic.src.cli --config config.json --image sonic/assets/dataset/sample_images/image1.png
```

## CLI Commands
- `--video <path>` / `--image <path>` / `--camera`
- `--preprocess-dataset` (writes manifest from `dataset_dir`)
- `--simulate-alert` (triggers alert handlers)
- `--no-preview` (disable OpenCV window)

## Configuration Fields
See `docs/config.md` for full table. Validation enforces sensible ranges (confidence 0-1, positive distances/ages).

## Tests
Run `pytest --cov=sonic --cov=tools` for coverage.
