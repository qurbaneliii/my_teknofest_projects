import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import cv2

from sonic.src.config import DetectorConfig
from sonic.src.core import Detector
from sonic.src.visualization import OverlayRenderer

st.set_page_config(page_title="SONIC YOLOv8 Demo", layout="wide")

st.title("SONIC • YOLOv8 Rodent Detection")
st.write("Upload an image to see detections rendered with bounding boxes.")

@st.cache_resource(show_spinner=False)
def load_detector(model_path: str):
    cfg = DetectorConfig(model_path=model_path)
    cfg.validate()
    return Detector(model_path=cfg.model_path, class_name=cfg.target_class, confidence_threshold=cfg.confidence_threshold)

model_path = st.text_input("Model path", "models/best.pt")
detector = load_detector(model_path)
renderer = OverlayRenderer()

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
    temp.write(uploaded.getvalue())
    temp.flush()

    frame = cv2.imread(temp.name)
    if frame is None:
        st.error("Could not read image")
    else:
        detections = detector.detect(frame, frame_id=0)
        annotated = renderer.draw_detections(frame.copy(), detections, detector.confidence_threshold)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated, caption=f"Detections: {len(detections)}", use_column_width=True)
else:
    st.info("Upload an image to begin.")
