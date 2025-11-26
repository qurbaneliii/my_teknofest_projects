# TEKNOFEST Field Projects
Applied AI, robotics, and sustainability prototypes prepared for TEKNOFEST submissions. Each folder contains the research collateral, pitch material, and—when available—production code that powered our latest iterations.

---

## Project Snapshot
| Project | What it tackles | Key assets (repo paths) | Status |
| --- | --- | --- | --- |
| SONIC | Smart pest mitigation that pairs drone scouting, ultrasonic deterrents, and on-edge vision | `sonic/src/rat_detector.py`, UI mockups (`sonic_app.html`, `sonic_website_mockup.html` planned), image evidence set, pitch PDFs | Field testing & software hardening |
| AgroScan | Drone-based crop-health analytics with NDVI-style insights and mission planning | `agroscan/AgroScan proje sunumu.pdf` (deck + KPIs) | Concept validation |
| Future entries | Additional TEKNOFEST concepts (aerial autonomy, smart irrigation, carbon monitoring) | Placeholder | Scoping |

---

## SONIC · Smart Rodent Intelligence & Control
Sonic combines a YOLOv8-based detector with custom alerting, drone scouting plans, and farmer-facing interfaces to keep storage silos and open fields rodent-free without chemical interventions.

**Highlights**
- YOLOv8 inference + multi-target tracking implemented in `_Mouse_Detection_using_YOLOv5_Object_De.py` (OpenCV UI overlays, alert callbacks, JSON logging)
- Low-bandwidth telemetry and alert fan-out (console/log/file callbacks) tuned for rural connectivity
- UI mockups (`SONC_APP.html`, `SONC_vebsayt.html`) and multilingual collateral (`sonic azərbaycanca.pdf`, `sonic_pitch.pdf`, `sonic_teknofest_2023_proje_deerlendirme_raporu.pdf`)
- Visual dataset snapshots (`image*.png`, WhatsApp exports) documenting field captures and misclassification cases

**Quickstart (detector script)**
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install ultralytics opencv-python numpy`
3. Drop your custom YOLO weights into `models/best.pt` (configurable via CLI)
4. Run camera mode: `python sonic/.../_Mouse_Detection_using_YOLOv5_Object_De.py --camera`
5. Run inference on footage: `python ...py --video data/test.mp4 --output runs/test.avi`

Useful flags: `--image path/to/frame.jpg`, `--no-preview` for headless servers, `--config custom_config.json` to persist thresholds/output paths.

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
├── agroscan/
│   └── AgroScan proje sunumu.pdf
└── sonic/
    ├── sonic azərbaycanca.pdf
    └── ExportBlock-7ea5f3ff-1aa7-4ba4-99f5-782a170eb01b-Part-1/
        ├── _Mouse_Detection_using_YOLOv5_Object_De.py
        ├── SONC_APP.html
        ├── SONC_vebsayt.html
        ├── image*.png, WhatsApp_kil*.jpg
        └── sonic_pitch.pdf · sonic_teknofest_2023_proje_deerlendirme_raporu.pdf
```

---

## Roadmap
- [ ] Publish SONIC hardware integration notes (ultrasonic array, drone payload specs)
- [ ] Add AgroScan preprocessing notebooks + sample orthomosaic tiles
- [ ] Capture short demo videos and host them via release artifacts
- [ ] Localize SONIC app copy (EN/TR/AZ) inside the HTML prototypes

---

## Collaboration & Contact
- Email: qurbanelifeyzullayev@gmail.com
- LinkedIn: https://linkedin.com/in/gurbanalifeyzullayev/

I’m always open to technical feedback, data partnerships, or field-test collaborations around sustainable agriculture and applied AI.

