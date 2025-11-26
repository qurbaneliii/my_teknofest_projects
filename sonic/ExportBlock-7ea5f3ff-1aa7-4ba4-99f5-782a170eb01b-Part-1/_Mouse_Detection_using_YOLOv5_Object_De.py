#!/usr/bin/env python3
"""
Complete Functional Rat Detector Model
A ready-to-run rat detection system with built-in fallback methods
"""

import cv2
import numpy as np
import time
import json
import logging
from datetime import datetime
from typing import List
from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
import warnings

warnings.filterwarnings('ignore')

# Add YOLOv8 import
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install the 'ultralytics' package: pip install ultralytics")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rat_detector.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class Detection:
    """Detection result data structure"""
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
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence,
            'class_name': self.class_name,
            'timestamp': self.timestamp.isoformat(),
            'frame_id': self.frame_id
        }

@dataclass
class RatTrack:
    """Rat tracking data structure"""
    track_id: int
    detections: List[Detection]
    last_seen: datetime
    is_active: bool = True
    alert_sent: bool = False

class RatDetector:
    """Mouse/Rat detection using YOLOv8 with lightweight tracking and alerting."""
    def __init__(self, model_path: str = "models/best.pt", config_path: str = "config.json", class_name: str = "mouse"):
        self.config = self._load_config(config_path)
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            logging.error(f"Failed to load YOLO model at {model_path}: {e}")
            raise
        self.class_name = class_name
        self.frame_count = 0
        self.detection_history: List[Detection] = []
        self.tracks: dict[int, RatTrack] = {}
        self.track_id_counter = 0
        self.alert_callbacks: List = []
        self.last_alert_time = 0.0
        self.confidence_threshold = float(self.config.get('confidence_threshold', 0.7))
        self.max_track_age = int(self.config.get('max_track_age', 30))
        self.alert_cooldown = float(self.config.get('alert_cooldown', 5.0))

    def _load_config(self, config_path: str) -> dict:
        default_config = {
            "confidence_threshold": 0.7,
            "max_track_age": 30,
            "alert_cooldown": 5.0,
            "save_detections": True,
            "output_dir": "detections",
            "video_output": True,
            "show_preview": True,
        }
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logging.warning(f"Error loading config: {e}")
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config

    def detect_rats(self, frame: np.ndarray) -> List[Detection]:
        """Run inference and collect detections above confidence threshold."""
        try:
            results = self.model(frame)
        except Exception as e:
            logging.warning(f"Model inference failed: {e}")
            return []
        detections: List[Detection] = []
        for r in results:
            boxes = getattr(r, 'boxes', [])
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                except Exception:
                    continue
                name = self.model.names.get(cls, "unknown") if hasattr(self.model, 'names') else str(cls)
                if name == self.class_name and conf >= self.confidence_threshold:
                    detections.append(Detection(
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                        confidence=conf,
                        class_name=name,
                        timestamp=datetime.now(),
                        frame_id=self.frame_count
                    ))
        return detections
    
    def update_tracks(self, detections: List[Detection]) -> None:
        """Associate detections to existing tracks using nearest-neighbor distance."""
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
            best_dist = 120.0  # distance threshold
            for det in unmatched:
                cx = det.x + det.width // 2
                cy = det.y + det.height // 2
                d = ((cx - lx)**2 + (cy - ly)**2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best = det
            if best:
                track.detections.append(best)
                track.last_seen = best.timestamp
                unmatched.remove(best)
                if not track.alert_sent:
                    self._send_alert(track)
                    track.alert_sent = True
            else:
                # deactivate stale tracks
                if self.frame_count - last.frame_id > self.max_track_age:
                    track.is_active = False
        for det in unmatched:
            new_track = RatTrack(track_id=self.track_id_counter, detections=[det], last_seen=det.timestamp)
            self.tracks[self.track_id_counter] = new_track
            self.track_id_counter += 1
    
    def _send_alert(self, track: RatTrack):
        """Send alert for rat detection"""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            for callback in self.alert_callbacks:
                callback(track)
            self.last_alert_time = current_time
    
    def add_alert_callback(self, callback):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Overlay boxes, labels, confidence bars, and session info."""
        for d in detections:
            color = (0, 200, 0) if d.confidence >= self.confidence_threshold else (0, 200, 200)
            cv2.rectangle(frame, (d.x, d.y), (d.x + d.width, d.y + d.height), color, 2)
            label = f"{d.class_name}:{d.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (d.x, d.y - lh - 6), (d.x + lw, d.y), color, -1)
            cv2.putText(frame, label, (d.x, d.y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # Confidence bar
            bar_w = int(80 * min(max(d.confidence, 0.0), 1.0))
            bar_y = d.y + d.height + 6
            cv2.rectangle(frame, (d.x, bar_y), (d.x + 80, bar_y + 8), (140, 140, 140), 1)
            cv2.rectangle(frame, (d.x, bar_y), (d.x + bar_w, bar_y + 8), (0, 180, 0), -1)
        active = sum(1 for t in self.tracks.values() if t.is_active)
        info = f"YOLOv8 | Frame {self.frame_count} | Det {len(detections)} | Tracks {active}"
        cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame
    
    def process_video(self, video_path: str | None = None, output_path: str | None = None):
        while True:
            if video_path and Path(video_path).exists():
                cap = cv2.VideoCapture(video_path)
                logging.info(f"Processing video: {video_path}")
            else:
                cap = cv2.VideoCapture(0)
                logging.info("Using camera feed")
            
            if not cap.isOpened():
                logging.error("Cannot open video source. Retrying in 3 seconds...")
                time.sleep(3)
                continue  # Try again

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer
            writer = None
            if output_path and self.config.get('video_output', True):
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            logging.info(f"Video properties: {width}x{height} @ {fps} FPS")
            logging.info(f"Detection method: YOLOv8")
            
            # Set up window with resizable and position
            window_name = 'Rat Detector'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(1280, width), min(720, height))
            # Move window to right side of the screen (e.g., x=1000, y=100)
            cv2.moveWindow(window_name, 1000, 100)
            # Try to set window always on top and focused (platform dependent)
            try:
                import ctypes
                hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
                if hwnd:
                    ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception:
                pass
            
            start_time = time.time()
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logging.warning("Frame read failed, retrying...")
                        time.sleep(0.1)
                        continue
                    
                    self.frame_count += 1
                    
                    detections = self.detect_rats(frame)
                    self.update_tracks(detections)
                    self.detection_history.extend(detections)
                    frame_with_detections = self.draw_detections(frame, detections)

                    # Draw a scale bar (e.g., 100 pixels = 10 cm)
                    bar_length = 100
                    bar_height = 8
                    bar_x = 30
                    bar_y = height - 40
                    cv2.rectangle(frame_with_detections, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), (255,255,255), -1)
                    cv2.putText(frame_with_detections, '10 cm', (bar_x + bar_length + 10, bar_y + bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    # Draw settings/info panel
                    panel_x, panel_y, panel_w, panel_h = width - 260, 20, 240, 120
                    cv2.rectangle(frame_with_detections, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (40,40,40), -1)
                    cv2.putText(frame_with_detections, 'Settings:', (panel_x + 10, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(frame_with_detections, f"Confidence: {self.confidence_threshold}", (panel_x + 10, panel_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 1)
                    cv2.putText(frame_with_detections, f"Tracks: {len(self.tracks)}", (panel_x + 10, panel_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 1)
                    cv2.putText(frame_with_detections, f"Frame: {self.frame_count}", (panel_x + 10, panel_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 1)

                    # Draw a large accuracy/confidence bar at the top
                    top_bar_x, top_bar_y = 50, 20
                    top_bar_w, top_bar_h = width - 100, 40
                    cv2.rectangle(frame_with_detections, (top_bar_x, top_bar_y), (top_bar_x + top_bar_w, top_bar_y + top_bar_h), (50,50,50), -1)
                    # Show the highest confidence in this frame
                    if detections:
                        max_conf = max([d.confidence for d in detections])
                        fill_w = int(top_bar_w * max_conf)
                        cv2.rectangle(frame_with_detections, (top_bar_x, top_bar_y), (top_bar_x + fill_w, top_bar_y + top_bar_h), (0,220,0), -1)
                        conf_text = f"Detection Confidence: {int(max_conf*100)}%"
                    else:
                        conf_text = "Detection Confidence: ---"
                    cv2.putText(frame_with_detections, conf_text, (top_bar_x + 20, top_bar_y + int(top_bar_h*0.7)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                    
                    # Show preview
                    try:
                        if self.config.get('show_preview', True):
                            cv2.imshow(window_name, frame_with_detections)
                    except cv2.error as e:
                        logging.warning(f"cv2.imshow failed: {e}")
                    
                    # Write output video
                    if writer:
                        writer.write(frame_with_detections)
                    
                    # Log progress
                    if self.frame_count % 120 == 0:
                        elapsed = time.time() - start_time
                        fps_calc = self.frame_count / max(elapsed, 1e-3)
                        logging.info(f"Frames: {self.frame_count} | FPS: {fps_calc:.1f} | Detections: {len(self.detection_history)}")
                    
                    # Check for quit
                    try:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('l'):
                            logging.info("'l' pressed - image mode disabled in loop; use --image CLI argument instead.")
                        elif key == ord('c'):
                            # Return to camera feed
                            cv2.destroyAllWindows()
                            cap.release()
                            cap = cv2.VideoCapture(0)
                            if not cap.isOpened():
                                logging.error("Failed to reopen camera feed")
                                break
                            self.frame_count = 0 # Reset frame count for new camera feed
                            self.detection_history = [] # Clear history for new camera feed
                            self.tracks = {} # Clear tracks for new camera feed
                            self.track_id_counter = 0 # Reset track ID counter
                            self.last_alert_time = 0 # Reset alert cooldown
                            logging.info("Switched back to camera feed")
                        elif key == ord('s') and self.config.get('show_preview', True):
                            out_path = Path(f"frame_{self.frame_count}.jpg")
                            cv2.imwrite(str(out_path), frame_with_detections)
                            logging.info(f"Saved snapshot {out_path}")
                    except Exception as e:
                        logging.warning(f"Error handling key press: {e}")
            except KeyboardInterrupt:
                logging.info("Interrupted by user")
            finally:
                cap.release()
                if writer:
                    writer.release()
                try:
                    cv2.destroyAllWindows()
                except cv2.error as e:
                    logging.warning(f"cv2.destroyAllWindows failed: {e}")
                # Save results
                self._save_results()
    
    def _save_results(self):
        """Save detection results"""
        if not self.config.get('save_detections', True):
            return
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Prepare results
        results = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'total_frames': self.frame_count,
                'detection_method': "YOLOv8",
                'total_detections': len(self.detection_history),
                'total_tracks': len(self.tracks),
                'active_tracks': sum(1 for track in self.tracks.values() if track.is_active)
            },
            'detections': [detection.to_dict() for detection in self.detection_history],
            'tracks': []
        }
        
        # Add track information
        for track_id, track in self.tracks.items():
            track_info = {
                'track_id': track_id,
                'is_active': track.is_active,
                'detection_count': len(track.detections),
                'first_seen': track.detections[0].timestamp.isoformat() if track.detections else None,
                'last_seen': track.last_seen.isoformat(),
                'alert_sent': track.alert_sent
            }
            results['tracks'].append(track_info)
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'rat_detection_results_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total detections: {len(self.detection_history)}")
        print(f"Total tracks: {len(self.tracks)}")
        print(f"Active tracks: {sum(1 for track in self.tracks.values() if track.is_active)}")
        print(f"Detection method: YOLOv8")
        print(f"Results saved to: {output_file}")
        print("="*50)

# Alert callback functions
def console_alert(track: RatTrack):
    """Console alert"""
    print(f"\n🚨 RAT DETECTED! Track ID: {track.track_id}")
    print(f"   First seen: {track.detections[0].timestamp.strftime('%H:%M:%S')}")
    print(f"   Confidence: {track.detections[-1].confidence:.2f}")
    print(f"   Location: ({track.detections[-1].x}, {track.detections[-1].y})")

def log_alert(track: RatTrack):
    """Log alert"""
    logging.warning(f"RAT ALERT - Track {track.track_id}: {track.detections[-1].class_name} "
                   f"detected with {track.detections[-1].confidence:.2f} confidence")

def file_alert(track: RatTrack):
    """File alert"""
    alert_file = Path("alerts.txt")
    with open(alert_file, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - RAT DETECTED - Track {track.track_id}\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Functional Rat Detector')
    parser.add_argument('--video', type=str, help='Input video file path')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--camera', action='store_true', help='Use camera feed')
    parser.add_argument('--no-preview', action='store_true', help='Disable preview window')
    parser.add_argument('--image', type=str, help='Detect mouse/rat in a single image file')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RatDetector(args.config)
    
    # Override config if needed
    if args.no_preview:
        detector.config['show_preview'] = False
    
    # Add alert callbacks
    detector.add_alert_callback(console_alert)
    detector.add_alert_callback(log_alert)
    detector.add_alert_callback(file_alert)
    
    # Image detection mode
    if args.image:
        if not Path(args.image).exists():
            logging.error(f"Image file {args.image} does not exist.")
            return
        frame = cv2.imread(args.image)
        if frame is None:
            logging.error(f"Failed to load image: {args.image}")
            return
        detections = detector.detect_rats(frame)
        frame_with_detections = detector.draw_detections(frame, detections)
        if detector.config.get('show_preview', True):
            try:
                cv2.imshow('Rat Detector - Image', frame_with_detections)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error as e:
                logging.warning(f"cv2.imshow or destroyAllWindows failed: {e}")
        # Optionally save the result
        out_path = Path(args.image).with_name(f"detected_{Path(args.image).name}")
        cv2.imwrite(str(out_path), frame_with_detections)
        logging.info(f"Detection result saved to {out_path}")
        # Print detection summary
        print("\n" + "="*50)
        print("IMAGE DETECTION SUMMARY")
        print("="*50)
        print(f"Detections: {len(detections)}")
        for det in detections:
            print(f"Class: {det.class_name}, Confidence: {det.confidence:.2f}, Location: ({det.x}, {det.y}, {det.width}, {det.height})")
        print("="*50)
        return
    
    # Process video
    if args.camera:
        detector.process_video(output_path=args.output)
    else:
        detector.process_video(args.video, args.output)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)