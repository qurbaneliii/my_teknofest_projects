#!/usr/bin/env python3
"""Rat / Mouse detector entrypoint using YOLOv8.

Moved from legacy path `_Mouse_Detection_using_YOLOv5_Object_De.py` to
`sonic/src/rat_detector.py` for clearer naming and structure.
"""

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import List
import argparse
import logging
import json
import time
import warnings
import cv2
import numpy as np

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics") from e

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("rat_detector.log"), logging.StreamHandler()],
)


@dataclass
class Detection:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    class_name: str
    timestamp: datetime
    frame_id: int

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "timestamp": self.timestamp.isoformat(),
            "frame_id": self.frame_id,
        }


@dataclass
class RatTrack:
    track_id: int
    detections: List[Detection]
    last_seen: datetime
    is_active: bool = True
    alert_sent: bool = False


class RatDetector:
    def __init__(self, model_path: str = "models/best.pt", config_path: str = "config.json", class_name: str = "mouse"):
        self.config = self._load_config(config_path)
        self.model = YOLO(model_path)
        self.class_name = class_name
        self.frame_count = 0
        self.detection_history: List[Detection] = []
        self.tracks: dict[int, RatTrack] = {}
        self.track_id_counter = 0
        self.alert_callbacks: List = []
        self.last_alert_time = 0.0
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.7))
        self.max_track_age = int(self.config.get("max_track_age", 30))
        self.alert_cooldown = float(self.config.get("alert_cooldown", 5.0))

    def _load_config(self, path: str) -> dict:
        base = {
            "confidence_threshold": 0.7,
            "max_track_age": 30,
            "alert_cooldown": 5.0,
            "save_detections": True,
            "output_dir": "detections",
            "video_output": True,
            "show_preview": True,
        }
        p = Path(path)
        if p.exists():
            try:
                user = json.loads(p.read_text())
                base.update(user)
            except Exception as e:
                logging.warning(f"Failed reading config {path}: {e}")
        p.write_text(json.dumps(base, indent=2))
        return base

    def detect(self, frame: np.ndarray) -> List[Detection]:
        try:
            results = self.model(frame)
        except Exception as e:
            logging.warning(f"Inference failed: {e}")
            return []
        out: List[Detection] = []
        for r in results:
            for box in getattr(r, "boxes", []):
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                except Exception:
                    continue
                name = self.model.names.get(cls, str(cls)) if hasattr(self.model, "names") else str(cls)
                if name == self.class_name and conf >= self.confidence_threshold:
                    out.append(
                        Detection(
                            x=x1,
                            y=y1,
                            width=x2 - x1,
                            height=y2 - y1,
                            confidence=conf,
                            class_name=name,
                            timestamp=datetime.now(),
                            frame_id=self.frame_count,
                        )
                    )
        return out

    def update_tracks(self, detections: List[Detection]) -> None:
        if not detections and not self.tracks:
            return
        unmatched = detections.copy()
        for track in list(self.tracks.values()):
            if not track.is_active or not track.detections:
                continue
            last = track.detections[-1]
            lx = last.x + last.width // 2
            ly = last.y + last.height // 2
            best = None
            best_dist = 120.0
            for d in unmatched:
                cx = d.x + d.width // 2
                cy = d.y + d.height // 2
                dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = d
            if best:
                track.detections.append(best)
                track.last_seen = best.timestamp
                unmatched.remove(best)
                if not track.alert_sent:
                    self._send_alert(track)
                    track.alert_sent = True
            else:
                if self.frame_count - last.frame_id > self.max_track_age:
                    track.is_active = False
        for d in unmatched:
            self.tracks[self.track_id_counter] = RatTrack(track_id=self.track_id_counter, detections=[d], last_seen=d.timestamp)
            self.track_id_counter += 1

    def _send_alert(self, track: RatTrack) -> None:
        now = time.time()
        if now - self.last_alert_time > self.alert_cooldown:
            for cb in self.alert_callbacks:
                try:
                    cb(track)
                except Exception as e:
                    logging.warning(f"Alert callback error: {e}")
            self.last_alert_time = now

    def add_alert_callback(self, cb):
        self.alert_callbacks.append(cb)

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        for d in detections:
            ok = d.confidence >= self.confidence_threshold
            color = (0, 200, 0) if ok else (0, 200, 200)
            cv2.rectangle(frame, (d.x, d.y), (d.x + d.width, d.y + d.height), color, 2)
            label = f"{d.class_name}:{d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (d.x, d.y - th - 6), (d.x + tw, d.y), color, -1)
            cv2.putText(frame, label, (d.x, d.y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        active = sum(1 for t in self.tracks.values() if t.is_active)
        info = f"frame {self.frame_count} | det {len(detections)} | tracks {active}"
        cv2.putText(frame, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return frame

    def _save_results(self) -> None:
        if not self.config.get("save_detections", True):
            return
        out_dir = Path(self.config["output_dir"]) ; out_dir.mkdir(exist_ok=True)
        summary = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "total_frames": self.frame_count,
                "total_detections": len(self.detection_history),
                "total_tracks": len(self.tracks),
                "active_tracks": sum(1 for t in self.tracks.values() if t.is_active),
            },
            "detections": [d.to_dict() for d in self.detection_history],
            "tracks": [
                {
                    "track_id": tid,
                    "is_active": t.is_active,
                    "detection_count": len(t.detections),
                    "first_seen": t.detections[0].timestamp.isoformat() if t.detections else None,
                    "last_seen": t.last_seen.isoformat(),
                    "alert_sent": t.alert_sent,
                }
                for tid, t in self.tracks.items()
            ],
        }
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"rat_detection_results_{stamp}.json"
        out_file.write_text(json.dumps(summary, indent=2))
        logging.info(f"Saved results {out_file}")

    def run_video(self, video_path: str | None = None, output_path: str | None = None) -> None:
        while True:
            cap = cv2.VideoCapture(video_path if video_path and Path(video_path).exists() else 0)
            if not cap.isOpened():
                logging.error("Video source unavailable. Retrying...")
                time.sleep(2)
                continue
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = None
            if output_path and self.config.get("video_output", True):
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            start = time.time()
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    self.frame_count += 1
                    dets = self.detect(frame)
                    self.update_tracks(dets)
                    self.detection_history.extend(dets)
                    frame = self.draw(frame, dets)
                    if self.config.get("show_preview", True):
                        try:
                            cv2.imshow("rat-detector", frame)
                        except cv2.error:
                            pass
                    if writer:
                        writer.write(frame)
                    if self.frame_count % 120 == 0:
                        elapsed = time.time() - start
                        logging.info(f"fps {(self.frame_count/max(elapsed,1e-3)):.1f}")
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord("s"):
                        snap = Path(f"frame_{self.frame_count}.jpg")
                        cv2.imwrite(str(snap), frame)
                        logging.info(f"Saved {snap}")
                self._save_results()
            finally:
                cap.release()
                if writer:
                    writer.release()
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            break


# Alert callbacks
def console_alert(track: RatTrack):
    print(f"\nALERT track={track.track_id} conf={track.detections[-1].confidence:.2f}")

def log_alert(track: RatTrack):
    logging.warning(f"TRACK {track.track_id} alert conf={track.detections[-1].confidence:.2f}")

def file_alert(track: RatTrack):
    Path("alerts.txt").write_text(f"{datetime.now().isoformat()} track {track.track_id}\n", append=True) if hasattr(Path("alerts.txt"), 'append') else open("alerts.txt","a").write(f"{datetime.now().isoformat()} track {track.track_id}\n")


def main():
    p = argparse.ArgumentParser(description="Rat / Mouse detector")
    p.add_argument("--video", type=str, help="Input video path", default=None)
    p.add_argument("--output", type=str, help="Output video path", default=None)
    p.add_argument("--config", type=str, default="config.json")
    p.add_argument("--camera", action="store_true", help="Force camera source")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--image", type=str, help="Single image path")
    args = p.parse_args()

    det = RatDetector(config_path=args.config)
    if args.no_preview:
        det.config["show_preview"] = False
    det.add_alert_callback(console_alert)
    det.add_alert_callback(log_alert)
    det.add_alert_callback(file_alert)

    if args.image:
        ip = Path(args.image)
        if not ip.exists():
            logging.error(f"Image not found: {ip}")
            return
        frame = cv2.imread(str(ip))
        if frame is None:
            logging.error("Failed to load image")
            return
        ds = det.detect(frame)
        frame = det.draw(frame, ds)
        if det.config.get("show_preview", True):
            try:
                cv2.imshow("rat-image", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        out = ip.with_name(f"detected_{ip.name}")
        cv2.imwrite(str(out), frame)
        logging.info(f"Saved {out} detections={len(ds)}")
        return

    source_video = None if args.camera else args.video
    det.run_video(video_path=source_video, output_path=args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unhandled: {e}")
