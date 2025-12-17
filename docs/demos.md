# Demos

## UI Mockups (hosted via GitHub Pages)
- [SONIC App](assets/mockups/sonic_app.html)
- [SONIC Website](assets/mockups/sonic_website.html)

## Streamlit Demo (local)
```bash
pip install -e .[demo]
streamlit run sonic/demo/app.py
```

The app loads a YOLOv8 model, lets users upload images or videos, and renders detections with bounding boxes.

## AgroScan Notebook
Open `agroscan/notebooks/ndvi_demo.ipynb` to see NDVI and simple stress scoring on sample imagery.
