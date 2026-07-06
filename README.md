# TEKNOFEST AI Projects

Applied AI systems for TEKNOFEST-style robotics, agriculture, and computer-vision experimentation.

## Overview

This repository collects two applied AI project areas:

- `sonic/`: smart rodent detection and control workflow with object detection, tracking, alerts, and visualization modules.
- `agroscan/`: drone/crop-health analytics workflow with vegetation-index computation, stress classification, and demo tooling.

The repository includes Python packages, CLI entry points, FastAPI service code, tests, Docker configuration, MkDocs documentation, and Streamlit demo paths.

## Problem

Agriculture and field-monitoring workflows often need fast visual inspection, early risk detection, and lightweight decision support. These projects explore practical AI building blocks for pest monitoring and crop-stress analysis without claiming production deployment.

## Features

- SONIC rodent-detection workflow with YOLO-oriented detection modules
- Tracking support including Kalman-style tracking logic
- Configurable alert handlers and OpenCV visualization helpers
- AgroScan vegetation-index utilities for NDVI, SAVI, and EVI-style analysis
- Crop-stress classification and statistics helpers
- FastAPI endpoints for detection and vegetation-index workflows
- Streamlit demo entry points
- Docker and Docker Compose setup
- Pytest-based test suite and Python package configuration

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Language | Python 3.10+ |
| Computer Vision | OpenCV, Ultralytics / YOLO-oriented workflow |
| Data / Imaging | NumPy, Pillow |
| API | FastAPI, Uvicorn |
| Demo | Streamlit |
| Docs | MkDocs |
| DevOps | Docker, Docker Compose, GitHub Actions |
| Quality | pytest, ruff, black, mypy |

## Architecture

```text
.
  sonic/        Rodent detection, tracking, alerts, visualization, and demo code
  agroscan/     Vegetation-index and crop-stress analytics code
  app/          FastAPI service layer
  tools/        Utility scripts, including text extraction
  tests/        Pytest test suite
  docs/         MkDocs documentation
```

## Getting Started

Install the base package:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Install all optional extras:

```bash
pip install -e ".[all]"
```

## Usage

### SONIC Detection

```bash
sonic-detect --camera
sonic-detect --video input.mp4
sonic-detect --image frame.jpg
```

### AgroScan CLI

```bash
agroscan ndvi --red red.png --nir nir.png --output ndvi.png
agroscan stress --red red.png --nir nir.png --output stress.png
agroscan generate-samples --output-dir samples/
```

### FastAPI Service

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Useful endpoints include:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Health check |
| `/api/v1/detect` | POST | Image detection workflow |
| `/api/v1/ndvi` | POST | NDVI and stress statistics |
| `/api/v1/ndvi/image` | POST | NDVI/stress map image output |
| `/api/v1/savi` | POST | SAVI computation |

### Docker

```bash
docker compose up --build
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
black .
mypy sonic/src agroscan/src
```

Build local docs:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Environment Variables

No required secrets are documented for the default local package/test workflow. API/demo deployments may need local paths or service-specific configuration depending on the command being run.

## Status

Status: Supporting applied-AI portfolio repository.

The project has a real package structure, tests, API code, Docker configuration, and documentation. It is best presented as a competition/prototype repository, not as a deployed production agriculture system.

## Roadmap

- Add screenshots or demo clips for SONIC and AgroScan
- Document expected input image formats for each CLI/API path
- Add model-weight handling instructions without committing large model files
- Expand hardware integration notes only where supported by implementation
- Keep generated analysis artifacts out of the main documentation path

## Known Limitations

- Real-world field performance is not documented in this repository.
- Some workflows require local images, video, camera access, or optional model assets.
- The included pitch PDF is large and should be moved to release assets or compressed if repository size becomes a concern.

## License

This repository includes an MIT [LICENSE](LICENSE).
