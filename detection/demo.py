"""
demo.py — SafeEdge Live Detection Demo
=======================================
Plays a real video file through the full detection pipeline with a
professional HUD overlay. No synthetic frames — real YOLO + EarlyDetector
running on every frame.

Run:
    python demo.py                                     # default: sample_fire.mp4
    python demo.py --input ../testbench/sample_fire.mp4  # custom video

Press Q to quit. Video loops automatically.
"""

import cv2
import argparse
import math
import time
import random
import sys
from pathlib import Path

# ── Import detection components ──────────────────────────────────────────────

try:
    from early_detector import EarlyFireDetector
    EARLY_AVAILABLE = True
except ImportError:
    EARLY_AVAILABLE = False

try:
    from ultralytics import YOLO
    MODEL_PATH = Path("models/fire_smoke.pt")
    if MODEL_PATH.exists():
        yolo_model = YOLO(str(MODEL_PATH))
        YOLO_AVAILABLE = True
    else:
        YOLO_AVAILABLE = False
        yolo_model = None
except ImportError:
    YOLO_AVAILABLE = False
    yolo_model = None

# ── NTU Locations ────────────────────────────────────────────────────────────

NTU_LOCATIONS = [
    {"building": "The Hive",          "floor": 2, "zone": "Collaboration Studio"},
    {"building": "Northspine",        "floor": 1, "zone": "Food Court"},
    {"building": "SCBE",              "floor": 3, "zone": "Laboratory Wing"},
    {"building": "Hall of Residence 2", "floor": 4, "zone": "Common Kitchen"},
]

FIRE_CLASSES = {"fire", "smoke", "flame", "Fire", "Smoke"}

# ── HUD Drawing ──────────────────────────────────────────────────────────────

def draw_hud(frame, loc, risk, signals, yolo_detections, early_result,
             fps, frame_idx, elapsed):
    """Draw professional overlay on the video frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── Colour scheme ────────────────────────────────────────────────────
    risk_colors = {
        "CLEAR":         (50, 200, 50),
        "EARLY_WARNING": (50, 165, 255),
        "HIGH":          (50, 100, 255),
        "CRITICAL":      (50, 50, 255),
    }
    rc = risk_colors.get(risk, (180, 180, 180))

    # ── Top-left status panel ────────────────────────────────────────────
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (520, 115), (15, 15, 15), -1)
    vis = cv2.addWeighted(overlay, 0.75, vis, 0.25, 0)

    cv2.putText(vis, f"SafeEdge  |  NTU  |  {risk}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, rc, 2)

    loc_text = f"{loc['building']}  |  Floor {loc['floor']}  |  {loc['zone']}"
    cv2.putText(vis, loc_text, (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv2.putText(vis, f"FPS: {fps:.0f}  |  Frame: {frame_idx}  |  {elapsed:.1f}s",
                (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    # Early detector score
    if early_result:
        score_text = f"Anomaly Score: {early_result.anomaly_score:.2f}  |  Signals: {early_result.active_signals}"
        cv2.putText(vis, score_text, (12, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)

    # ── Top-right signal indicators ──────────────────────────────────────
    sig_names = ["OPTICAL FLOW", "BACKGROUND SUB", "TEXTURE VAR", "YOLO FIRE"]
    overlay2 = vis.copy()
    cv2.rectangle(overlay2, (w - 240, 0), (w, 26 * len(sig_names) + 12), (15, 15, 15), -1)
    vis = cv2.addWeighted(overlay2, 0.75, vis, 0.25, 0)

    for i, name in enumerate(sig_names):
        active = signals.get(name, False)
        dot_color = (0, 220, 50) if active else (60, 60, 60)
        text_color = (50, 220, 50) if active else (100, 100, 100)
        sy = 22 + i * 26
        cv2.circle(vis, (w - 225, sy), 7, dot_color, -1)
        cv2.putText(vis, name, (w - 210, sy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1)

    # ── YOLO bounding boxes ──────────────────────────────────────────────
    for det in yolo_detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]
        cls = det["class"]
        box_color = (0, 0, 255) if "fire" in cls.lower() else (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # ── Bottom banners ───────────────────────────────────────────────────
    if risk == "EARLY_WARNING":
        pulse = 0.6 + 0.4 * abs(math.sin(time.time() * 3))
        bcolor = (int(20 * pulse), int(100 * pulse), int(220 * pulse))
        cv2.rectangle(vis, (0, h - 45), (w, h - 18), bcolor, -1)
        cv2.putText(vis,
                    f"EARLY WARNING — {loc['building']} Floor {loc['floor']}  |  Monitoring...",
                    (12, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    elif risk in ("HIGH", "CRITICAL"):
        pulse = 0.7 + 0.3 * abs(math.sin(time.time() * 5))
        cv2.rectangle(vis, (0, h - 45), (w, h - 18),
                      (int(30 * pulse), int(30 * pulse), int(220 * pulse)), -1)
        cv2.putText(vis,
                    f"FIRE @ {loc['building']} Floor {loc['floor']} ({loc['zone']}) — EVACUATE",
                    (12, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        # Flashing red border
        if int(time.time() * 4) % 2 == 0:
            cv2.rectangle(vis, (2, 2), (w - 2, h - 2), (0, 0, 220), 3)

    return vis


# ── Main demo loop ───────────────────────────────────────────────────────────

def run_demo(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{video_path}'")
        sys.exit(1)

    # Init detectors
    early = EarlyFireDetector() if EARLY_AVAILABLE else None

    loc = random.choice(NTU_LOCATIONS)
    start_time = time.time()
    frame_idx = 0
    prev_time = time.time()
    fps = 0.0

    print("=" * 60)
    print("  SafeEdge — Live Detection Demo")
    print(f"  Video: {video_path}")
    print(f"  YOLO:  {'LOADED' if YOLO_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  Early: {'LOADED' if EARLY_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  Location: {loc['building']} Floor {loc['floor']}")
    print("  Press Q to quit")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop the video, pick new location
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            loc = random.choice(NTU_LOCATIONS)
            frame_idx = 0
            start_time = time.time()
            if early:
                early.reset()
            print(f"\n[Demo] Loop restart -> {loc['building']} Floor {loc['floor']}")
            continue

        elapsed = time.time() - start_time
        frame_idx += 1

        # FPS calculation
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        prev_time = now

        # ── Track B: EarlyFireDetector ───────────────────────────────────
        early_result = None
        early_triggered = False
        if early:
            early_result = early.update(frame, frame_idx)
            if early_result and early_result.should_alert:
                early_triggered = True

        # ── Track A: YOLO inference ──────────────────────────────────────
        yolo_detections = []
        yolo_fire = False
        if YOLO_AVAILABLE and frame_idx % 2 == 0:  # run every other frame
            results = yolo_model.predict(frame, conf=0.45, iou=0.45,
                                         imgsz=640, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_name = r.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if cls_name in FIRE_CLASSES:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        yolo_detections.append({
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "class": cls_name,
                        })
                        yolo_fire = True

        # ── Determine risk level ─────────────────────────────────────────
        if yolo_fire:
            max_conf = max(d["conf"] for d in yolo_detections)
            risk = "CRITICAL" if max_conf >= 0.7 else "HIGH"
        elif early_triggered:
            risk = "EARLY_WARNING"
        else:
            risk = "CLEAR"

        # ── Signal status ────────────────────────────────────────────────
        signals = {
            "OPTICAL FLOW":   early_result is not None and early_result.anomaly_score > 0.2,
            "BACKGROUND SUB": early_result is not None and early_result.anomaly_score > 0.3,
            "TEXTURE VAR":    early_result is not None and early_result.anomaly_score > 0.4,
            "YOLO FIRE":      yolo_fire,
        }

        # ── Draw HUD ────────────────────────────────────────────────────
        vis = draw_hud(frame, loc, risk, signals, yolo_detections,
                       early_result, fps, frame_idx, elapsed)

        cv2.imshow("SafeEdge — Fire Detection Demo", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeEdge Live Detection Demo")
    parser.add_argument("--input", type=str,
                        default="../testbench/sample_fire.mp4",
                        help="Path to video file (default: sample_fire.mp4)")
    args = parser.parse_args()

    try:
        run_demo(args.input)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nDemo ended.")
