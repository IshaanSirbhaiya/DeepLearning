"""
detector.py — SafeEdge Detection Layer
Main fire detection engine.

- Ingests RTSP / webcam / video file via OpenCV
- Runs YOLOv8n with pre-trained fire/smoke weights
- Multi-frame confirmation via RiskScorer
- Face-blurred snapshots via PrivacyFilter
- Structured JSON alerts via AlertGenerator
- Exposes a FastAPI endpoint for communication layer
- Logs FPS, CPU %, memory MB to prove edge viability

Run modes:
  python detector.py --input testbench/sample_fire.mp4
  python detector.py --input 0                          (webcam)
  python detector.py --input rtsp://192.168.1.10/stream
  python detector.py --serve                            (FastAPI server only)
"""

import os
import sys
import time
import queue
import logging
import argparse
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Local imports ─────────────────────────────────────────────────────────────
from risk_scorer    import RiskScorer, ScorerConfig, FrameDetection, ScoreResult
from privacy_filter import PrivacyFilter
from alert_generator import AlertGenerator, LocationConfig

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SafeEdge.Detector")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class DetectorConfig:
    # Model
    MODEL_PATH          = Path("models/fire_smoke.pt")
    MODEL_REPO          = "keremberke/yolov8n-fire-smoke-detection"   # HuggingFace
    CONFIDENCE_THRESHOLD = 0.45       # lower than scorer ignore — YOLO pre-filter
    IOU_THRESHOLD        = 0.45       # NMS IoU
    IMG_SIZE             = 640        # inference resolution
    DEVICE               = "cpu"      # "cuda" if GPU available

    # Stream
    TARGET_FPS          = 15
    FRAME_SKIP          = 2           # process every Nth frame (1 = every frame)
    DISPLAY             = True        # show live preview window

    # Alert
    CAMERA_ID           = os.getenv("CAMERA_ID",   "CAM_01")
    BUILDING            = os.getenv("BUILDING",    "Block 4A")
    FLOOR               = int(os.getenv("FLOOR",   "1"))
    ZONE                = os.getenv("ZONE",        "unknown")

    # API
    API_HOST            = "0.0.0.0"
    API_PORT            = 8001

    # Claude Vision (optional enrichment — requires ANTHROPIC_API_KEY)
    USE_CLAUDE_VISION   = os.getenv("USE_CLAUDE_VISION", "true").lower() == "true"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    """
    Load YOLOv8n fire/smoke model.

    Priority:
      1. Local weights at models/fire_smoke.pt
      2. Download from HuggingFace via ultralytics hub
      3. Fall back to base YOLOv8n (generic COCO — less accurate for fire)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.critical(
            "ultralytics not installed. Run: pip install ultralytics"
        )
        sys.exit(1)

    cfg = DetectorConfig

    # ── Try local weights first ───────────────────────────────────────────────
    if cfg.MODEL_PATH.exists():
        logger.info(f"Loading local weights: {cfg.MODEL_PATH}")
        model = YOLO(str(cfg.MODEL_PATH))
        logger.info(f"✓ Model loaded from disk — classes: {model.names}")
        return model

    # ── Download from HuggingFace ─────────────────────────────────────────────
    logger.info(f"Local weights not found. Downloading from HuggingFace: {cfg.MODEL_REPO}")
    try:
        from huggingface_hub import hf_hub_download
        pt_file = hf_hub_download(
            repo_id=cfg.MODEL_REPO,
            filename="best.pt",
            local_dir="models",
        )
        # Rename to canonical name
        os.rename(pt_file, str(cfg.MODEL_PATH))
        model = YOLO(str(cfg.MODEL_PATH))
        logger.info(f"✓ Downloaded & loaded. Classes: {model.names}")
        return model

    except Exception as e:
        logger.warning(f"HuggingFace download failed: {e}")

    # ── Fallback: base YOLOv8n (COCO — no fire class) ────────────────────────
    logger.warning(
        "Falling back to base YOLOv8n (COCO weights). "
        "Fire/smoke detection accuracy will be LOWER. "
        "Place fire-specific weights at models/fire_smoke.pt for best results."
    )
    model = YOLO("yolov8n.pt")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EDGE METRICS
# ══════════════════════════════════════════════════════════════════════════════

class EdgeMetrics:
    """Rolling average of FPS, CPU, and memory — logged per detection."""

    def __init__(self, window: int = 30):
        self._fps_buf:  List[float] = []
        self._cpu_buf:  List[float] = []
        self._mem_buf:  List[float] = []
        self._window    = window
        self._prev_time = time.time()
        self._process   = psutil.Process()

    def tick(self) -> Dict[str, float]:
        now  = time.time()
        fps  = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now

        cpu  = self._process.cpu_percent()
        mem  = self._process.memory_info().rss / (1024 * 1024)  # MB

        self._fps_buf.append(fps)
        self._cpu_buf.append(cpu)
        self._mem_buf.append(mem)

        if len(self._fps_buf) > self._window:
            self._fps_buf.pop(0)
            self._cpu_buf.pop(0)
            self._mem_buf.pop(0)

        return {
            "fps_current":  round(fps, 1),
            "fps_avg":      round(sum(self._fps_buf) / len(self._fps_buf), 1),
            "cpu_pct":      round(cpu, 1),
            "mem_mb":       round(mem, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# FIRE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class FireDetector:
    """
    Core detection engine. Processes frames from any OpenCV-compatible source.

    Thread-safe: detection loop runs on a background thread.
    Alerts are placed into self.alert_queue for the API layer to consume.
    """

    FIRE_CLASSES  = {"fire", "smoke", "flame", "Fire", "Smoke"}  # adapt to model labels

    def __init__(self):
        cfg = DetectorConfig

        self.model         = load_model()
        self.scorer        = RiskScorer(ScorerConfig(
            window_size  = 8,
            confirm_min  = 5,
            cooldown_sec = 30.0,
        ))
        self.generator     = AlertGenerator(
            camera_id         = cfg.CAMERA_ID,
            location          = LocationConfig(cfg.BUILDING, cfg.FLOOR, cfg.ZONE),
            use_claude_vision = cfg.USE_CLAUDE_VISION,
        )
        self.metrics       = EdgeMetrics()
        self.alert_queue:  queue.Queue = queue.Queue(maxsize=100)

        self._running      = False
        self._thread:      Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_alert: Optional[Dict]       = None
        self._stats        = {"total_frames": 0, "total_alerts": 0, "fps_avg": 0}

        logger.info("FireDetector ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, source: str, display: bool = True):
        """
        Blocking run — processes video source until end or KeyboardInterrupt.
        source: file path, RTSP URL, or '0'/'1' for webcam index.
        """
        cap = self._open_source(source)
        logger.info(f"Stream opened: {source}")

        frame_idx   = 0
        cfg         = DetectorConfig

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if source not in ("0", "1"):
                        logger.info("Stream ended.")
                    break

                frame_idx += 1
                self._stats["total_frames"] = frame_idx

                # Skip frames for performance
                if frame_idx % cfg.FRAME_SKIP != 0:
                    continue

                # ── Inference ─────────────────────────────────────────────────
                detections = self._infer(frame, frame_idx)
                score      = self.scorer.update(detections)
                m          = self.metrics.tick()
                self._stats["fps_avg"] = m["fps_avg"]

                # ── Log every 30 frames ───────────────────────────────────────
                if frame_idx % 30 == 0:
                    logger.info(
                        f"Frame {frame_idx} | FPS {m['fps_current']} "
                        f"(avg {m['fps_avg']}) | CPU {m['cpu_pct']}% "
                        f"| MEM {m['mem_mb']} MB | {score.summary()}"
                    )

                # ── Alert ─────────────────────────────────────────────────────
                if score.should_alert:
                    logger.warning(
                        f"🔥 ALERT [{score.risk_level}] "
                        f"conf={score.best_confidence:.2f} "
                        f"frames={score.positive_frames}/{score.window_size}"
                    )
                    alert = self.generator.generate(frame, score, m)
                    self._latest_alert = alert.to_dict()
                    self._stats["total_alerts"] += 1

                    if not self.alert_queue.full():
                        self.alert_queue.put(alert.to_dict())

                    print("\n" + "═" * 60)
                    print(alert.to_json())
                    print("═" * 60 + "\n")

                # ── Display ───────────────────────────────────────────────────
                if display:
                    vis = self._draw(frame, detections, score, m)
                    cv2.imshow("SafeEdge — Fire Detection", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            logger.info(
                f"Finished — {frame_idx} frames | "
                f"{self._stats['total_alerts']} alerts | "
                f"avg {self._stats['fps_avg']} FPS"
            )

    def get_status(self) -> Dict:
        return {**self._stats, "latest_alert": self._latest_alert}

    def drain_alerts(self) -> List[Dict]:
        """Drain all pending alerts from queue (non-blocking)."""
        alerts = []
        while not self.alert_queue.empty():
            try:
                alerts.append(self.alert_queue.get_nowait())
            except queue.Empty:
                break
        return alerts

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer(self, frame: np.ndarray, frame_idx: int) -> List[FrameDetection]:
        """Run YOLOv8n and return FrameDetection list for this frame."""
        cfg      = DetectorConfig
        results  = self.model(
            frame,
            conf    = cfg.CONFIDENCE_THRESHOLD,
            iou     = cfg.IOU_THRESHOLD,
            imgsz   = cfg.IMG_SIZE,
            device  = cfg.DEVICE,
            verbose = False,
        )

        detections = []
        h, w = frame.shape[:2]

        for r in results:
            for box in r.boxes:
                cls_id  = int(box.cls[0])
                label   = self.model.names[cls_id].lower()
                conf    = float(box.conf[0])

                # Only pass fire/smoke classes
                if not self._is_fire_class(label):
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(FrameDetection(
                    timestamp   = time.time(),
                    confidence  = conf,
                    label       = label,
                    bbox        = [x1 / w, y1 / h, x2 / w, y2 / h],
                    frame_index = frame_idx,
                ))

        return detections

    def _is_fire_class(self, label: str) -> bool:
        return any(kw in label for kw in ("fire", "smoke", "flame"))

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _draw(
        self,
        frame:  np.ndarray,
        dets:   List[FrameDetection],
        score:  ScoreResult,
        metrics: Dict,
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw detection boxes
        for det in dets:
            x1 = int(det.bbox[0] * w); y1 = int(det.bbox[1] * h)
            x2 = int(det.bbox[2] * w); y2 = int(det.bbox[3] * h)
            color = (0, 0, 255) if "fire" in det.label else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis, f"{det.label} {det.confidence:.2f}",
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        # HUD overlay
        risk_color = {
            "CRITICAL": (0, 0, 255),
            "HIGH":     (0, 100, 255),
            "WARNING":  (0, 200, 255),
            "IGNORE":   (0, 200, 0),
        }.get(score.risk_level, (200, 200, 200))

        hud = [
            f"RISK: {score.risk_level}  conf={score.best_confidence:.2f}",
            f"Pos: {score.positive_frames}/{score.window_size}  ALERT={'YES' if score.should_alert else 'NO'}",
            f"FPS: {metrics['fps_current']}  CPU: {metrics['cpu_pct']}%  MEM: {metrics['mem_mb']} MB",
        ]
        for i, line in enumerate(hud):
            cv2.putText(vis, line, (10, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)

        # Alert banner
        if score.should_alert:
            cv2.rectangle(vis, (0, h - 50), (w, h), (0, 0, 200), -1)
            cv2.putText(vis, f"🔥 {score.risk_level} — FIRE/SMOKE CONFIRMED",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return vis

    # ── Stream ────────────────────────────────────────────────────────────────

    def _open_source(self, source: str) -> cv2.VideoCapture:
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.critical(f"Cannot open source: {source}")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI SERVER
# ══════════════════════════════════════════════════════════════════════════════

def create_api(detector: FireDetector) -> FastAPI:
    app = FastAPI(title="SafeEdge Detection API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "SafeEdge Detection Layer"}

    @app.get("/status")
    def status():
        return detector.get_status()

    @app.get("/alerts/latest")
    def latest_alert():
        alert = detector._latest_alert
        if not alert:
            raise HTTPException(status_code=404, detail="No alerts yet")
        return alert

    @app.get("/alerts/drain")
    def drain_alerts():
        """
        Drain all pending alerts from queue.
        Communication layer polls this endpoint.
        """
        return {"alerts": detector.drain_alerts()}

    return app


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SafeEdge Fire Detector")
    parser.add_argument("--input",   default="0",     help="Video source: file path, RTSP URL, or webcam index (0)")
    parser.add_argument("--no-display", action="store_true", help="Disable live preview (headless mode)")
    parser.add_argument("--serve",   action="store_true", help="Also run FastAPI server for communication layer")
    parser.add_argument("--port",    default=8001, type=int, help="API server port")
    args = parser.parse_args()

    detector = FireDetector()

    if args.serve:
        api = create_api(detector)
        # Run API in background thread
        def run_api():
            uvicorn.run(api, host=DetectorConfig.API_HOST, port=args.port, log_level="warning")
        t = threading.Thread(target=run_api, daemon=True)
        t.start()
        logger.info(f"API running at http://localhost:{args.port}")

    detector.run(
        source  = args.input,
        display = not args.no_display,
    )


if __name__ == "__main__":
    main()
