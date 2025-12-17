# Live Demos

Interactive demonstrations of SONIC and AgroScan capabilities.

## 🖥️ UI Mockups (GitHub Pages)

Static HTML mockups showcasing the planned user interfaces:

- [SONIC Mobile App](assets/mockups/sonic_app.html) - Mobile-first detection interface
- [SONIC Website](assets/mockups/sonic_website.html) - Web dashboard for monitoring

## 🐀 SONIC Streamlit Demo

Real-time rodent detection using YOLOv8.

### Local Installation

```bash
# Install demo dependencies
pip install -e .[demo]

# Run SONIC demo
streamlit run sonic/demo/app.py
```

### Features
- Upload images or video files
- Real-time detection with bounding boxes
- Confidence threshold adjustment
- Detection statistics and export

### Docker Deployment

```bash
docker-compose up sonic-demo
# Access at http://localhost:8501
```

## 🌾 AgroScan Streamlit Demo

Crop health analysis using vegetation indices.

### Local Installation

```bash
# Install demo dependencies
pip install -e .[demo]

# Run AgroScan demo
streamlit run agroscan/demo/app.py
```

### Features
- Upload Red and NIR band images (or use synthetic data)
- Compute NDVI, SAVI vegetation indices
- Interactive stress classification thresholds
- Stress distribution statistics and charts
- Export NDVI maps, stress maps, and JSON reports

### Docker Deployment

```bash
docker-compose up agroscan-demo
# Access at http://localhost:8502
```

## 📓 Jupyter Notebooks

Interactive notebooks for exploration and learning:

| Notebook | Description |
|----------|-------------|
| `agroscan/notebooks/ndvi_demo.ipynb` | NDVI calculation and visualization |

## 🔌 REST API

FastAPI service for programmatic access:

```bash
# Start API server
docker-compose up api
# Or locally:
uvicorn app:app --host 0.0.0.0 --port 8000

# Access docs at http://localhost:8000/docs
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/detect` | POST | Detect rodents in image |
| `/api/v1/ndvi` | POST | Compute NDVI and stress statistics |
| `/api/v1/ndvi/image` | POST | Generate NDVI/stress map image |
| `/api/v1/savi` | POST | Compute SAVI index |

### Example: NDVI via API

```python
import requests

with open("red_band.png", "rb") as red, open("nir_band.png", "rb") as nir:
    response = requests.post(
        "http://localhost:8000/api/v1/ndvi",
        files={"red_band": red, "nir_band": nir},
    )
    print(response.json())
```

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TEKNOFEST AI Services                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │   SONIC      │     │   AgroScan   │     │   FastAPI    │ │
│  │   Demo       │     │   Demo       │     │   Service    │ │
│  │  :8501       │     │  :8502       │     │  :8000       │ │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘ │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                     Python Packages                       ││
│  │  ┌────────────┐  ┌─────────────┐  ┌────────────────────┐ ││
│  │  │sonic.src   │  │agroscan.src │  │tools               │ ││
│  │  │ - detector │  │ - ndvi      │  │ - text_extractor   │ ││
│  │  │ - tracker  │  │ - stress    │  │                    │ ││
│  │  │ - alerts   │  │ - preproc   │  │                    │ ││
│  │  └────────────┘  └─────────────┘  └────────────────────┘ ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```
