"""Main CLI entry point for SONIC rat detection system."""

import argparse
import logging
import time
from pathlib import Path
from typing import List
import cv2

from sonic.src.config import DetectorConfig
from sonic.src.core import Detector, Tracker, Detection
from sonic.src.alerts import ConsoleAlertHandler, FileAlertHandler, LogAlertHandler
from sonic.src.visualization import OverlayRenderer

logger = logging.getLogger(__name__)


class DetectionSession:
    """Manages a complete detection session with video processing."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.detector = Detector(
            model_path=config.model_path,
            class_name=config.target_class,
            confidence_threshold=config.confidence_threshold,
        )
        self.tracker = Tracker(
            distance_threshold=config.track_distance_threshold,
            max_age=config.max_track_age,
        )
        self.renderer = OverlayRenderer()
        self.alert_handlers = self._setup_alerts()
        self.frame_count = 0
        self.detection_history: List[Detection] = []
        self.last_alert_time = 0.0

    def _setup_alerts(self) -> list:
        """Initialize alert handlers based on config."""
        handlers = []
        if self.config.enable_console_alerts:
            handlers.append(ConsoleAlertHandler())
        if self.config.enable_file_alerts:
            handlers.append(FileAlertHandler())
        if self.config.enable_log_alerts:
            handlers.append(LogAlertHandler())
        return handlers

    def process_frame(self, frame):
        """Process single frame: detect, track, visualize."""
        self.frame_count += 1
        
        # Run detection
        detections = self.detector.detect(frame, self.frame_count)
        self.detection_history.extend(detections)
        
        # Update tracking
        tracks = self.tracker.update(detections, self.frame_count)
        
        # Send alerts for new tracks
        self._check_alerts(tracks)
        
        # Visualize
        frame = self.renderer.draw_detections(
            frame, detections, self.config.confidence_threshold
        )
        frame = self.renderer.draw_info_panel(
            frame, self.frame_count, len(detections), len(tracks)
        )
        
        return frame

    def _check_alerts(self, tracks):
        """Send alerts for tracks that haven't been alerted yet."""
        current_time = time.time()
        if current_time - self.last_alert_time < self.config.alert_cooldown:
            return
        
        for track in tracks:
            if not track.alert_sent and track.length > 0:
                for handler in self.alert_handlers:
                    try:
                        handler(track)
                    except Exception as e:
                        logger.error(f"Alert handler failed: {e}")
                track.alert_sent = True
                self.last_alert_time = current_time

    def run_video(self, video_path: str | None = None, output_path: str | None = None):
        """Run detection on video file or camera stream."""
        source = video_path if video_path and Path(video_path).exists() else 0
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {source}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps} FPS")
        
        # Setup video writer
        writer = None
        if output_path and self.config.video_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = self.process_frame(frame)
                
                if self.config.show_preview:
                    try:
                        cv2.imshow("SONIC Detector", frame)
                    except cv2.error:
                        pass
                
                if writer:
                    writer.write(frame)
                
                # Log progress
                if self.frame_count % 120 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = self.frame_count / max(elapsed, 1e-3)
                    logger.info(f"Frame {self.frame_count} | FPS: {fps_actual:.1f}")
                
                # Handle keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    snapshot = Path(f"snapshot_{self.frame_count}.jpg")
                    cv2.imwrite(str(snapshot), frame)
                    logger.info(f"Saved {snapshot}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self._save_results()

    def process_image(self, image_path: str):
        """Process a single image file."""
        img_path = Path(image_path)
        if not img_path.exists():
            logger.error(f"Image not found: {img_path}")
            return
        
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.error(f"Failed to load image: {img_path}")
            return
        
        frame = self.process_frame(frame)
        
        if self.config.show_preview:
            cv2.imshow("SONIC Detector", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        output = img_path.with_name(f"detected_{img_path.name}")
        cv2.imwrite(str(output), frame)
        logger.info(f"Saved result to {output}")

    def _save_results(self):
        """Save session results to JSON."""
        if not self.config.save_detections:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        import json
        
        summary = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "total_frames": self.frame_count,
                "total_detections": len(self.detection_history),
                "total_tracks": len(self.tracker.tracks),
                "active_tracks": len(self.tracker.active_tracks),
            },
            "detections": [d.to_dict() for d in self.detection_history],
            "tracks": [t.to_dict() for t in self.tracker.tracks.values()],
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"session_{timestamp}.json"
        output_file.write_text(json.dumps(summary, indent=2))
        logger.info(f"Results saved to {output_file}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SONIC - Smart rodent detection system"
    )
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--video", type=str, help="Input video file")
    parser.add_argument("--output", type=str, help="Output video file")
    parser.add_argument("--image", type=str, help="Single image to process")
    parser.add_argument("--camera", action="store_true", help="Use camera feed")
    parser.add_argument("--no-preview", action="store_true", help="Disable video preview")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("sonic_detector.log"),
            logging.StreamHandler(),
        ],
    )
    
    # Load config
    config = DetectorConfig.from_file(args.config)
    if args.no_preview:
        config.show_preview = False
    
    # Create session
    session = DetectionSession(config)
    
    # Run appropriate mode
    if args.image:
        session.process_image(args.image)
    else:
        video_source = None if args.camera else args.video
        session.run_video(video_path=video_source, output_path=args.output)


if __name__ == "__main__":
    main()
