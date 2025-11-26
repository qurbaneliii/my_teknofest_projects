# TEKNOFEST Field Projects
Applied AI, robotics, and sustainability prototypes prepared for TEKNOFEST submissions. Each folder contains the research collateral, pitch material, and—when available—production code that powered our latest iterations.

---

## Project Snapshot
| Project | What it tackles | Key assets (repo paths) | Status |
| --- | --- | --- | --- |
| SONIC | Smart pest mitigation that pairs drone scouting, ultrasonic deterrents, and on-edge vision | Modular detector: `sonic/src/{core,alerts,visualization}`, CLI: `sonic/src/cli.py`, Assets: `sonic/assets/` | Production-ready v0.2 |
| AgroScan | Drone-based crop-health analytics with NDVI-style insights and mission planning | `agroscan/docs/agroscan_pitch.pdf` | Concept validation |
| Future entries | Additional TEKNOFEST concepts (aerial autonomy, smart irrigation, carbon monitoring) | Placeholder | Scoping |

---

## SONIC · Smart Rodent Intelligence & Control
SONIC combines YOLOv8-based detection with modular tracking, configurable alerts, and farmer-facing interfaces to keep storage silos and fields rodent-free without chemical interventions.

**Architecture (v0.2 - Refactored)**
- **Core Detection**: `sonic/src/core/` - Detector, Tracker, and data models (Detection, Track)
- **Alert System**: `sonic/src/alerts/` - Pluggable handlers (console, file, log)
- **Visualization**: `sonic/src/visualization/` - OpenCV overlay rendering
- **Configuration**: `sonic/src/config.py` - Type-safe config management with JSON schema
- **CLI**: `sonic/src/cli.py` - Main entry point with video/image/camera modes

**Assets**
- Pitch decks: `sonic/assets/docs/{sonic_pitch_en.pdf, sonic_pitch_az.pdf, sonic_2023_evaluation.pdf}`
- UI mockups: `sonic/assets/mockups/{sonic_app.html, sonic_website.html}`
- Dataset: `sonic/assets/dataset/{field_captures/, sample_images/}`

**Quickstart**
```bash
# Install
pip install -e .  # or: pip install -r requirements.txt

# Setup config
cp sonic/config.example.json config.json
# Edit config.json to point to your model weights

# Run detection
python -m sonic.src.cli --camera              # Live camera
python -m sonic.src.cli --video input.mp4     # Video file
python -m sonic.src.cli --image frame.jpg     # Single image
```

**Development**
```bash
pip install -r requirements-dev.txt
pytest                    # Run tests
ruff check sonic/         # Lint
black sonic/              # Format
```

---

## AgroScan · Precision Crop Intelligence
AgroScan focuses on affordable aerial scouting. The current assets capture the mission narrative, KPIs, and deployment plan while code and datasets are being curated for release.

- `agroscan/AgroScan proje sunumu.pdf` outlines the drone platform, orthomosaic workflow, and data-to-advice pipeline
- Planned deliverables: dataset schemas, preprocessing notebooks, and an inference service for fast vegetation indexing
- Near-term work: port NDVI and stress classification notebooks, publish bill of materials, and document field test protocol

---

## Repository Layout
```
my_teknofest_projects/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── tests/
│   ├── conftest.py
│   ├── test_detector.py
│   ├── test_tracker.py
│   └── test_alerts.py
├── tools/
│   ├── text_extractor.py
│   └── README.md
├── agroscan/
│   └── docs/
│       └── agroscan_pitch.pdf
└── sonic/
    ├── config.example.json
    ├── src/
    │   ├── core/           (Detector, Tracker, models)
    │   ├── alerts/         (AlertHandler interface + implementations)
    │   ├── visualization/  (OverlayRenderer)
    │   ├── config.py
    │   └── cli.py
    └── assets/
        ├── docs/           (pitch decks, evaluation reports)
        ├── mockups/        (HTML prototypes)
        └── dataset/        (field captures, sample images)
```

---

## Roadmap
- [x] Modularize detector into core/alerts/visualization components
- [x] Add comprehensive test suite (pytest + fixtures)
- [x] Implement proper configuration management
- [x] Reorganize assets with semantic naming
- [ ] Publish SONIC hardware integration notes (ultrasonic array, drone payload specs)
- [ ] Add AgroScan preprocessing notebooks + sample orthomosaic tiles
- [ ] Deploy containerized inference service (Docker + FastAPI)
- [ ] CI/CD pipeline with automated testing

---

## Collaboration & Contact
- Email: qurbanelifeyzullayev@gmail.com
- LinkedIn: https://linkedin.com/in/gurbanalifeyzullayev/

I’m always open to technical feedback, data partnerships, or field-test collaborations around sustainable agriculture and applied AI.

