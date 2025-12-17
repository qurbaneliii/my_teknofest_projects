# TEKNOFEST AI Projects

[![CI](https://github.com/qurbaneliii/my_teknofest_projects/actions/workflows/ci.yml/badge.svg)](https://github.com/qurbaneliii/my_teknofest_projects/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Applied AI systems for agricultural monitoring and sustainable farming, developed for TEKNOFEST competitions.

## 🚀 Quick Links

| Resource | Link |
|----------|------|
| **SONIC Docs** | [sonic/README.md](sonic/README.md) |
| **AgroScan Docs** | [agroscan/docs/agroscan_pitch.md](agroscan/docs/agroscan_pitch.md) |
| **API Docs** | Run locally: `uvicorn app:app` → http://localhost:8000/docs |
| **MkDocs Site** | `pip install -e .[docs] && mkdocs serve` |

---

## 📊 Project Overview

| Project | Description | Key Features | Status |
|---------|-------------|--------------|--------|
| **SONIC** | Smart rodent detection & control | YOLOv8 detection, Kalman tracking, configurable alerts | ✅ v0.3.0 |
| **AgroScan** | Drone-based crop health analytics | NDVI/SAVI computation, stress classification, Streamlit demo | ✅ v0.3.0 |

---

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/qurbaneliii/my_teknofest_projects.git
cd my_teknofest_projects

# Install base package
pip install -e .

# Install with all extras (dev, docs, demo, api)
pip install -e .[all]
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# Access:
# - API: http://localhost:8000/docs
# - SONIC Demo: http://localhost:8501
# - AgroScan Demo: http://localhost:8502
# - Docs: http://localhost:8080
```

---

## 🐀 SONIC · Smart Rodent Intelligence & Control

AI-powered pest detection combining YOLOv8 vision with modular tracking and configurable alert systems.

### Architecture

```
sonic/src/
├── core/           # Detector, Tracker, KalmanTracker, models
├── alerts/         # Pluggable handlers (console, file, log)
├── visualization/  # OpenCV overlay rendering
├── config.py       # Type-safe Pydantic configuration
└── cli.py          # Main CLI entry point
```

### Usage

```bash
# Run detection
sonic-detect --camera                    # Live camera
sonic-detect --video input.mp4           # Video file
sonic-detect --image frame.jpg           # Single image

# With Kalman tracking
sonic-detect --video input.mp4 --config config.json
# config.json: {"tracker_type": "kalman"}
```

### Demo

```bash
pip install -e .[demo]
streamlit run sonic/demo/app.py
```

---

## 🌾 AgroScan · Precision Crop Intelligence

Vegetation index computation and crop stress analysis for precision agriculture.

### Features

- **NDVI/SAVI/EVI** vegetation index computation
- **Stress classification** with configurable thresholds
- **Synthetic data generation** for testing
- **Streamlit demo** for interactive analysis

### Usage

```bash
# CLI commands
agroscan ndvi --red red.png --nir nir.png --output ndvi.png
agroscan stress --red red.png --nir nir.png --output stress.png
agroscan generate-samples --output-dir samples/

# Demo
streamlit run agroscan/demo/app.py
```

### Python API

```python
from agroscan.src import compute_ndvi, classify_stress, compute_stress_statistics

ndvi = compute_ndvi(nir_band, red_band)
stress = classify_stress(ndvi)
stats = compute_stress_statistics(stress)
print(f"Healthy: {stats['healthy']:.1f}%")
```

---

## 🔌 REST API

FastAPI service for programmatic access to detection and analysis.

```bash
# Start API
uvicorn app:app --host 0.0.0.0 --port 8000

# Or via Docker
docker-compose up api
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/detect` | POST | Detect rodents in image |
| `/api/v1/ndvi` | POST | Compute NDVI + stress statistics |
| `/api/v1/ndvi/image` | POST | Generate NDVI/stress map image |
| `/api/v1/savi` | POST | Compute SAVI index |

---

## 🧪 Development

```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest --cov=sonic --cov=agroscan

# Lint & format
ruff check .
black .
mypy sonic/src agroscan/src

# Build docs
mkdocs serve
```

---

## 📁 Repository Structure

```
my_teknofest_projects/
├── sonic/                  # SONIC detection package
│   ├── src/                # Core modules
│   ├── demo/               # Streamlit demo
│   └── assets/             # Docs, mockups, dataset
├── agroscan/               # AgroScan analytics package
│   ├── src/                # NDVI, stress, preprocessing
│   ├── demo/               # Streamlit demo
│   └── notebooks/          # Jupyter notebooks
├── app/                    # FastAPI service
├── tools/                  # Utilities (text_extractor)
├── tests/                  # Pytest test suite
├── docs/                   # MkDocs documentation
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Service orchestration
└── pyproject.toml          # Package configuration
```

---

## ✅ Roadmap

- [x] Modular architecture (core/alerts/visualization)
- [x] Comprehensive test suite (pytest + fixtures)
- [x] Pydantic configuration management
- [x] Kalman filter tracking with Hungarian algorithm
- [x] AgroScan NDVI/stress module
- [x] FastAPI REST service
- [x] Docker containerization
- [x] CI/CD with GitHub Actions
- [ ] Hardware integration notes (ultrasonic array, drone specs)
- [ ] GeoTIFF support for real orthomosaic tiles
- [ ] Model quantization for edge deployment
- [ ] Time-series crop monitoring

---

## 📬 Contact & Collaboration

- **Email**: qurbanelifeyzullayev@gmail.com
- **LinkedIn**: [linkedin.com/in/gurbanalifeyzullayev](https://linkedin.com/in/gurbanalifeyzullayev/)

Open to technical feedback, data partnerships, and field-test collaborations around sustainable agriculture and applied AI.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

