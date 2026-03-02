"""
detector.py — SafeEdge Detection Layer
Main fire detection engine.

- Ingests RTSP / webcam / video file via OpenCV
- Runs YOLOv8n with pre-trained fire/smoke weights
- Multi-frame confirmation via RiskScorer
- Face-blurred snapshots via PrivacyFilter
- Structured JSON alerts via AlertGenerator
- PRE-FIRE early warning via EarlyFireDetector (optical flow + bg sub + texture)
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
from dotenv import load_dotenv

load_dotenv()                                                # cwd .env
load_dotenv(Path(__file__).parent.parent / ".env")           # project root .env

# ── Local imports (works from both project root and detection/ directory) ─────
try:
    from detection.risk_scorer     import RiskScorer, ScorerConfig, FrameDetection, ScoreResult
    from detection.privacy_filter  import PrivacyFilter
    from detection.alert_generator import AlertGenerator, LocationConfig
    from detection.fire_event      import fire_event, register_routes
    from detection.early_detector  import EarlyFireDetector, EarlyWarning
    from detection.supabase_publisher import publish as supabase_publish
except ImportError:
    from risk_scorer     import RiskScorer, ScorerConfig, FrameDetection, ScoreResult
    from privacy_filter  import PrivacyFilter
    from alert_generator import AlertGenerator, LocationConfig
    from fire_event      import fire_event, register_routes
    from early_detector  import EarlyFireDetector, EarlyWarning
    from supabase_publisher import publish as supabase_publish

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
    MODEL_REPO          = "keremberke/yolov8n-fire-smoke-detection"
    CONFIDENCE_THRESHOLD = 0.45
    IOU_THRESHOLD        = 0.45
    IMG_SIZE             = 640
    DEVICE               = "cpu"

    # Stream
    TARGET_FPS          = 15
    FRAME_SKIP          = 2
    DISPLAY             = True

    # Alert
    CAMERA_ID           = os.getenv("CAMERA_ID",   "CAM_01")
    BUILDING            = os.getenv("BUILDING",    "Block 4A")   # overridden per-alert
    FLOOR               = int(os.getenv("FLOOR",   "1"))
    ZONE                = os.getenv("ZONE",        "unknown")

    # API
    API_HOST            = "0.0.0.0"
    API_PORT            = 8001

    # Claude Vision (optional enrichment — requires ANTHROPIC_API_KEY)
    USE_CLAUDE_VISION   = os.getenv("USE_CLAUDE_VISION", "true").lower() == "true"


# ══════════════════════════════════════════════════════════════════════════════
# NTU CAMPUS LOCATIONS
# Each entry is a full location dict sent to teammates via fire_event payload.
# lat/lng are used by the outdoor evacuation / Google Maps routing module.
# ══════════════════════════════════════════════════════════════════════════════

NTU_LOCATIONS = [
    {
        "building": "The Hive",
        "floor":    2,
        "zone":     "Collaboration Studio",
        "campus":   "NTU",
        "lat":      1.34321,
        "lng":      103.68275,
    },
    {
        "building": "Northspine",
        "floor":    1,
        "zone":     "Food Court",
        "campus":   "NTU",
        "lat":      1.3431,
        "lng":      103.6805,
    },
    {
        "building": "School of Chemical and Biomedical Engineering",
        "floor":    3,
        "zone":     "Laboratory Wing",
        "campus":   "NTU",
        "lat":      1.34572,
        "lng":      103.67855,
    },
    {
        "building": "Hall of Residence 2",
        "floor":    4,
        "zone":     "Common Kitchen",
        "campus":   "NTU",
        "lat":      1.3547,
        "lng":      103.6853,
    },
]


def pick_random_ntu_location() -> dict:
    """
    Randomly select one NTU campus location for demo purposes.
    Called once per confirmed fire alert — teammates receive whichever
    location is chosen in the fire_event payload.
    """
    import random
    loc = random.choice(NTU_LOCATIONS)
    logger.info(
        f"[Demo] Location picked → {loc['building']} "
        f"Floor {loc['floor']} ({loc['zone']})"
    )
    return loc


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.critical("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    cfg = DetectorConfig

    if cfg.MODEL_PATH.exists():
        logger.info(f"Loading local weights: {cfg.MODEL_PATH}")
        model = YOLO(str(cfg.MODEL_PATH))
        logger.info(f"✓ Model loaded from disk — classes: {model.names}")
        return model

    logger.info(f"Local weights not found. Downloading from HuggingFace: {cfg.MODEL_REPO}")
    try:
        from huggingface_hub import hf_hub_download
        pt_file = hf_hub_download(
            repo_id=cfg.MODEL_REPO,
            filename="best.pt",
            local_dir="models",
        )
        os.rename(pt_file, str(cfg.MODEL_PATH))
        model = YOLO(str(cfg.MODEL_PATH))
        logger.info(f"✓ Downloaded & loaded. Classes: {model.names}")
        return model
    except Exception as e:
        logger.warning(f"HuggingFace download failed: {e}")

    logger.warning(
        "Falling back to base YOLOv8n (COCO weights). "
        "Fire/smoke detection accuracy will be LOWER."
    )
    model = YOLO("yolov8n.pt")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EDGE METRICS
# ══════════════════════════════════════════════════════════════════════════════

class EdgeMetrics:
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
        mem  = self._process.memory_info().rss / (1024 * 1024)

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
    Core detection engine. Now runs TWO parallel detection tracks:
      Track A: YOLOv8n — detects VISIBLE fire and smoke (unchanged)
      Track B: EarlyFireDetector — detects PRE-FIRE heat anomalies
    """

    FIRE_CLASSES = {"fire", "smoke", "flame", "Fire", "Smoke"}

    def __init__(self):
        cfg = DetectorConfig

        self.model         = load_model()
        self.scorer        = RiskScorer(ScorerConfig(
            window_size  = 8,
            confirm_min  = 5,
            cooldown_sec = 30.0,
        ))
        # Location is set per-alert (random NTU location) — initialise with
        # first entry as default so AlertGenerator has something to start with
        _default_loc = NTU_LOCATIONS[0]
        self.generator     = AlertGenerator(
            camera_id         = cfg.CAMERA_ID,
            location          = LocationConfig(
                _default_loc["building"],
                _default_loc["floor"],
                _default_loc["zone"],
            ),
            use_vision_api    = cfg.USE_CLAUDE_VISION,
        )
        self.metrics       = EdgeMetrics()

        # ── NEW: Early pre-fire anomaly detector ──────────────────────────────
        self.early_detector = EarlyFireDetector()
        self._last_early_warning: Optional[EarlyWarning] = None
        # ─────────────────────────────────────────────────────────────────────

        self.alert_queue:  queue.Queue = queue.Queue(maxsize=100)
        self._running      = False
        self._thread:      Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_alert: Optional[Dict]        = None
        self._stats = {
            "total_frames":         0,
            "total_alerts":         0,
            "total_early_warnings": 0,   # ← NEW counter
            "fps_avg":              0,
        }

        logger.info("FireDetector ready (YOLO + EarlyFireDetector).")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, source: str, display: bool = True):
        cap       = self._open_source(source)
        logger.info(f"Stream opened: {source}")
        frame_idx = 0
        cfg       = DetectorConfig

        if display:
            cv2.namedWindow("SafeEdge — Fire Detection", cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if source not in ("0", "1"):
                        logger.info("Stream ended.")
                    break

                frame_idx += 1
                self._stats["total_frames"] = frame_idx

                if frame_idx % cfg.FRAME_SKIP != 0:
                    continue

                # ════════════════════════════════════════════════════════════
                # TRACK B — Early pre-fire detection (runs FIRST, no model needed)
                # ════════════════════════════════════════════════════════════
                early_warning = self.early_detector.update(frame, frame_idx)
                self._last_early_warning = early_warning   # store for _draw()

                if early_warning and early_warning.should_alert:
                    self._stats["total_early_warnings"] += 1
                    early_loc = pick_random_ntu_location()
                    logger.warning(
                        f"⚠️  EARLY_WARNING [{early_warning.anomaly_type}] "
                        f"score={early_warning.anomaly_score:.2f} "
                        f"signals={early_warning.active_signals} "
                        f"@ {early_loc['building']}"
                    )
                    # Publish to fire_event bus at EARLY_WARNING severity
                    # (teammates can filter on risk_level == "EARLY_WARNING")
                    fire_event.publish(
                        building   = early_loc["building"],
                        floor      = early_loc["floor"],
                        zone       = early_loc["zone"],
                        confidence = early_warning.anomaly_score,
                        risk_level = "EARLY_WARNING",
                        camera_id  = DetectorConfig.CAMERA_ID,
                        latitude   = early_loc["lat"],
                        longitude  = early_loc["lng"],
                    )
                    supabase_publish(early_loc["lat"], early_loc["lng"], early_loc["building"])

                # ════════════════════════════════════════════════════════════
                # TRACK A — YOLOv8n fire/smoke detection (UNCHANGED)
                # ════════════════════════════════════════════════════════════
                detections = self._infer(frame, frame_idx)
                score      = self.scorer.update(detections)
                m          = self.metrics.tick()
                self._stats["fps_avg"] = m["fps_avg"]

                if frame_idx % 30 == 0:
                    ew_score = early_warning.anomaly_score if early_warning else 0.0
                    logger.info(
                        f"Frame {frame_idx} | FPS {m['fps_current']} "
                        f"(avg {m['fps_avg']}) | CPU {m['cpu_pct']}% "
                        f"| MEM {m['mem_mb']} MB | {score.summary()} "
                        f"| early_score={ew_score:.2f}"
                    )

                if score.should_alert:
                    logger.warning(
                        f"🔥 ALERT [{score.risk_level}] "
                        f"conf={score.best_confidence:.2f} "
                        f"frames={score.positive_frames}/{score.window_size}"
                    )
                    # Pick a random NTU location for this alert
                    loc = pick_random_ntu_location()
                    self.generator.update_location(
                        building = loc["building"],
                        floor    = loc["floor"],
                        zone     = loc["zone"],
                    )

                    alert = self.generator.generate(frame, score, m)
                    self._latest_alert = alert.to_dict()
                    self._stats["total_alerts"] += 1

                    if not self.alert_queue.full():
                        self.alert_queue.put(alert.to_dict())

                    # ── Formatted terminal output ────────────────────
                    va = alert.vision_analysis
                    w = 57
                    conf_str = f"{score.best_confidence:.2f}"
                    det_label = score.best_detection.label if score.best_detection else "fire"
                    frames_str = f"{score.positive_frames}/{score.window_size}"

                    if va and va.get("false_positive_likely"):
                        # 2FA caught false positive — don't publish alert
                        reason = va.get("false_positive_reason") or "No real fire detected"
                        print("\n" + "=" * w)
                        print(f"  YOLO Detection: {det_label} (conf={conf_str})")
                        print(f"  Vision 2FA:     FALSE POSITIVE")
                        print(f"  Reason:         {reason}")
                        print(f"  Action:         No alert sent - false alarm")
                        print("=" * w + "\n")
                    elif va:
                        # 2FA confirmed fire — publish alert
                        fire_event.publish(
                            building   = loc["building"],
                            floor      = loc["floor"],
                            zone       = loc["zone"],
                            confidence = score.best_confidence,
                            risk_level = score.risk_level,
                            camera_id  = DetectorConfig.CAMERA_ID,
                            latitude   = loc["lat"],
                            longitude  = loc["lng"],
                        )
                        supabase_publish(loc["lat"], loc["lng"], loc["building"])
                        risk = va.get("risk_level", score.risk_level).upper()
                        print("\n" + "=" * w)
                        print(f"  YOLO Detection: {det_label} (conf={conf_str})")
                        print(f"  Vision 2FA:     CONFIRMED FIRE")
                        print(f"  Risk Level:     {risk}")
                        print(f"  Action:         Alert published - evacuate now")
                        print("=" * w + "\n")
                    else:
                        # No Vision API — clean output, no 2FA mention
                        fire_event.publish(
                            building   = loc["building"],
                            floor      = loc["floor"],
                            zone       = loc["zone"],
                            confidence = score.best_confidence,
                            risk_level = score.risk_level,
                            camera_id  = DetectorConfig.CAMERA_ID,
                            latitude   = loc["lat"],
                            longitude  = loc["lng"],
                        )
                        supabase_publish(loc["lat"], loc["lng"], loc["building"])
                        print("\n" + "=" * w)
                        print(f"  YOLO Detection: {det_label} (conf={conf_str})")
                        print(f"  Confidence:     {frames_str} frames confirmed")
                        print(f"  Risk Level:     {score.risk_level}")
                        print(f"  Action:         Alert published to evacuation system")
                        print("=" * w + "\n")

                # ── Display ───────────────────────────────────────────────
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
                f"{self._stats['total_alerts']} fire alerts | "
                f"{self._stats['total_early_warnings']} early warnings | "
                f"avg {self._stats['fps_avg']} FPS"
            )

    def get_status(self) -> Dict:
        return {**self._stats, "latest_alert": self._latest_alert}

    def drain_alerts(self) -> List[Dict]:
        alerts = []
        while not self.alert_queue.empty():
            try:
                alerts.append(self.alert_queue.get_nowait())
            except queue.Empty:
                break
        return alerts

    # ── Inference (UNCHANGED) ─────────────────────────────────────────────────

    def _infer(self, frame: np.ndarray, frame_idx: int) -> List[FrameDetection]:
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

    # ── Visualisation (extended with early warning overlay) ───────────────────

    def _draw(
        self,
        frame:   np.ndarray,
        dets:    List[FrameDetection],
        score:   ScoreResult,
        metrics: Dict,
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw YOLO detection boxes (UNCHANGED)
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

        # HUD overlay (UNCHANGED)
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

        # YOLO fire alert banner (UNCHANGED)
        if score.should_alert:
            cv2.rectangle(vis, (0, h - 50), (w, h), (0, 0, 200), -1)
            cv2.putText(vis, f"🔥 {score.risk_level} — FIRE/SMOKE CONFIRMED",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # ── NEW: Early warning overlay (orange, sits above YOLO banner) ───────
        vis = self.early_detector.draw_overlay(vis, self._last_early_warning)
        # ─────────────────────────────────────────────────────────────────────

        return vis

    # ── Stream (UNCHANGED) ────────────────────────────────────────────────────

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
        return {"alerts": detector.drain_alerts()}

    # ── NEW: early warning endpoint for teammates ─────────────────────────────
    @app.get("/early-warning")
    def early_warning():
        """
        Returns the latest early warning state.
        Teammates can poll this alongside /fire to get advance notice.
        """
        ew = detector._last_early_warning
        if not ew:
            return {"early_warning_active": False}
        return {
            "early_warning_active": True,
            "anomaly_type":   ew.anomaly_type,
            "anomaly_score":  ew.anomaly_score,
            "active_signals": ew.active_signals,
            "confirm_count":  ew.confirm_count,
            "timestamp":      ew.timestamp,
        }
    # ─────────────────────────────────────────────────────────────────────────

    register_routes(app)
    return app


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SafeEdge Fire Detector")
    parser.add_argument("--input",      default="0",   help="Video source")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--serve",      action="store_true")
    parser.add_argument("--port",       default=8001, type=int)
    parser.add_argument("--no-vision",  action="store_true", help="Disable OpenAI Vision 2FA")
    args = parser.parse_args()

    if args.no_vision:
        DetectorConfig.USE_CLAUDE_VISION = False

    detector = FireDetector()

    if args.serve:
        api = create_api(detector)
        def run_api():
            uvicorn.run(api, host=DetectorConfig.API_HOST, port=args.port, log_level="warning")
        t = threading.Thread(target=run_api, daemon=True)
        t.start()
        logger.info(f"API running at http://localhost:{args.port}")

    detector.run(source=args.input, display=not args.no_display)


if __name__ == "__main__":
    main()
