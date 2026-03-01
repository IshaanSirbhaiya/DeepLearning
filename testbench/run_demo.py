"""
SafeEdge - End-to-End Demo Script
Runs the full pipeline: YOLO detection -> risk scoring -> alert generation -> backend storage -> AI summary

Two modes:
  python testbench/run_demo.py              # real detection on sample video
  python testbench/run_demo.py --simulated  # simulated alerts (no model needed)
"""

import sys
import os
import io
import time
import json
import asyncio
import argparse
import subprocess
import threading
from pathlib import Path

# Fix Windows console encoding for Unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(num, total, text):
    print(f"[{num}/{total}] {text}\n")


# ═══════════════════════════════════════════════════════════════
# REAL DETECTION MODE
# ═══════════════════════════════════════════════════════════════

def run_real_demo():
    """Run actual YOLO detection on sample fire video with backend."""
    import requests

    print_header("SafeEdge - REAL End-to-End Demo")
    print("This runs the full pipeline with real YOLO fire/smoke detection.\n")

    video_path = PROJECT_ROOT / "testbench" / "sample_fire.mp4"
    model_path = PROJECT_ROOT / "detection" / "models" / "fire_smoke.pt"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run: python detection/models/download_model.py")
        return

    if not video_path.exists():
        print(f"WARNING: Sample video not found at {video_path}")
        print("Falling back to simulated mode.\n")
        asyncio.run(run_simulated_demo())
        return

    total_steps = 6

    # Step 1: Start backend
    print_step(1, total_steps, "Starting backend server (port 8000)...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "backend.server"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(4)

    try:
        resp = requests.get("http://localhost:8000/", timeout=3)
        print(f"  Backend running: {resp.json().get('service', 'OK')}\n")
    except Exception:
        print("  WARNING: Backend may not have started cleanly, continuing...\n")

    # Step 2: Run detection
    print_step(2, total_steps, "Running YOLO fire/smoke detection on sample video...")
    print(f"  Video: {video_path.name}")
    print(f"  Model: {model_path.name}")
    print(f"  Mode: headless (no display), forwarding alerts to backend\n")

    detector = subprocess.Popen(
        [sys.executable, "-m", "detection.detector",
         "--input", str(video_path), "--no-display"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    # Read detector stderr in real time for logging
    def stream_logs(proc, label):
        for line in proc.stderr:
            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded:
                # Only show important lines
                if any(kw in decoded for kw in ["ALERT", "WARNING", "CRITICAL", "HIGH", "loaded", "ready", "Stream"]):
                    print(f"  [{label}] {decoded}")

    log_thread = threading.Thread(target=stream_logs, args=(detector, "DETECT"), daemon=True)
    log_thread.start()

    print("  Processing video (this runs the actual YOLO model on each frame)...")
    print("  Waiting for detections...\n")

    # Wait for detection to process enough for alerts
    time.sleep(60)

    detector.terminate()
    try:
        detector.wait(timeout=10)
    except subprocess.TimeoutExpired:
        detector.kill()

    print("\n  Detection complete.\n")

    # Step 3: Query backend for results
    print_step(3, total_steps, "Querying backend for received alerts...")
    try:
        resp = requests.get("http://localhost:8000/alerts", timeout=5)
        alerts = resp.json()
        print(f"  Backend received {len(alerts)} alert(s):\n")
        for a in alerts:
            risk = a.get("risk_score", "?")
            indicator = {"WARNING": "[!]", "HIGH": "[!!]", "CRITICAL": "[!!!]"}.get(risk, "[?]")
            print(f"  {indicator} Alert #{a['id']}: {a['event'].replace('_', ' ').upper()}")
            print(f"     Camera: {a['camera_id']} | Confidence: {a['confidence']:.0%} | Risk: {risk}")
            loc = a.get("location", {})
            if loc:
                print(f"     Location: {loc.get('building','?')}, Floor {loc.get('floor','?')}, {loc.get('zone','?')}")
            if a.get("summary"):
                print(f"     Summary: \"{a['summary']}\"")
            print()
    except Exception as e:
        print(f"  Error querying backend: {e}\n")

    # Step 4: Check alert snapshots
    print_step(4, total_steps, "Checking alert snapshots (face-blurred)...")
    alerts_dir = PROJECT_ROOT / "alerts"
    jpgs = list(alerts_dir.glob("snap_*.jpg"))
    jsons = list(alerts_dir.glob("snap_*.json"))
    print(f"  Blurred snapshots saved: {len(jpgs)}")
    print(f"  JSON sidecar files:      {len(jsons)}\n")

    # Step 5: Backend stats
    print_step(5, total_steps, "System statistics...")
    try:
        resp = requests.get("http://localhost:8000/stats", timeout=5)
        stats = resp.json()
        print(f"  Total alerts:      {stats.get('total_alerts', 0)}")
        print(f"  CRITICAL:          {stats.get('critical', 0)}")
        print(f"  HIGH:              {stats.get('high', 0)}")
        print(f"  WARNING:           {stats.get('warning', 0)}")
        print(f"  Pending forward:   {stats.get('pending_forward', 0)}\n")
    except Exception as e:
        print(f"  Error getting stats: {e}\n")

    # Step 6: Done
    print_step(6, total_steps, "Pipeline summary")
    print("  Detection  -> YOLO fire/smoke model ran on CCTV video")
    print("  Scoring    -> Multi-frame confirmation (5/8 frames positive)")
    print("  Privacy    -> Faces auto-blurred in alert snapshots")
    print("  Alerts     -> Structured JSON with metadata + blurred images")
    print("  Backend    -> SQLite storage, AI-powered incident summaries")
    print("  Forwarding -> Store-and-forward for offline resilience")

    print_header("Demo Complete")
    print("In production, alerts are sent to:")
    print("  - Telegram bot -> residents with evacuation directions")
    print("  - Command dashboard -> authorities with AI summaries")
    print("  - Google Maps -> dynamic evacuation routes avoiding fire zone\n")

    # Cleanup
    backend.terminate()
    try:
        backend.wait(timeout=5)
    except subprocess.TimeoutExpired:
        backend.kill()


# ═══════════════════════════════════════════════════════════════
# SIMULATED MODE (no model / video needed)
# ═══════════════════════════════════════════════════════════════

DEMO_ALERTS = [
    {
        "camera_id": "CAM_01",
        "event": "fire_detected",
        "confidence": 0.55,
        "risk_score": "WARNING",
        "location": {"building": "Block 4A", "floor": 1, "zone": "lobby"},
    },
    {
        "camera_id": "CAM_03",
        "event": "smoke_detected",
        "confidence": 0.78,
        "risk_score": "HIGH",
        "location": {"building": "Block 4A", "floor": 3, "zone": "kitchen"},
    },
    {
        "camera_id": "CAM_03",
        "event": "fire_detected",
        "confidence": 0.94,
        "risk_score": "CRITICAL",
        "location": {"building": "Block 4A", "floor": 3, "zone": "kitchen"},
    },
]


async def run_simulated_demo():
    """Run simulated demo with pre-defined alerts (no model needed)."""
    from backend.database import init_db, store_alert, get_alerts, get_unforwarded_alerts, mark_forwarded
    from backend.summarizer import generate_summary

    print_header("SafeEdge - Simulated Demo (no model/video required)")
    print("This demo simulates the full detection -> alert -> summary pipeline.\n")

    db_path = "testbench/demo.db"
    init_db(db_path)
    print("[1/5] Database ready.\n")

    print("[2/5] Simulating fire detection from CCTV cameras...\n")
    for i, alert in enumerate(DEMO_ALERTS):
        from datetime import datetime
        alert["timestamp"] = datetime.now().isoformat()

        risk = alert["risk_score"]
        indicator = {"WARNING": "[!]", "HIGH": "[!!]", "CRITICAL": "[!!!]"}.get(risk, "[?]")
        print(f"  {indicator} Alert {i+1}: {alert['event'].replace('_', ' ').upper()}")
        print(f"     Camera: {alert['camera_id']} | Confidence: {alert['confidence']:.0%} | Risk: {risk}")
        loc = alert["location"]
        print(f"     Location: {loc['building']}, Floor {loc['floor']}, {loc['zone']}")
        alert_id = store_alert(alert, db_path)
        print(f"     Stored as alert #{alert_id}\n")
        time.sleep(1)

    print("[3/5] Generating AI incident summaries...\n")
    alerts = get_alerts(db_path=db_path)
    for alert in alerts:
        summary = await generate_summary(alert)
        print(f"  Alert #{alert['id']}: \"{summary}\"\n")
        time.sleep(0.5)

    print("[4/5] Store-and-forward demo...\n")
    pending = get_unforwarded_alerts(db_path=db_path)
    print(f"  {len(pending)} alerts pending forward")
    for p in pending:
        mark_forwarded(p["id"], db_path)
    print(f"  All marked as forwarded.\n")

    print("[5/5] Statistics\n")
    all_alerts = get_alerts(db_path=db_path)
    print(f"  Total alerts: {len(all_alerts)}")
    print(f"  CRITICAL:     {sum(1 for a in all_alerts if a.get('risk_score') == 'CRITICAL')}")
    print(f"  HIGH:         {sum(1 for a in all_alerts if a.get('risk_score') == 'HIGH')}")
    print(f"  WARNING:      {sum(1 for a in all_alerts if a.get('risk_score') == 'WARNING')}")

    print_header("Simulated Demo Complete")

    if os.path.exists(db_path):
        os.remove(db_path)
        for ext in ["-wal", "-shm"]:
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeEdge End-to-End Demo")
    parser.add_argument("--simulated", action="store_true", help="Use simulated alerts (no model/video needed)")
    args = parser.parse_args()

    if args.simulated:
        asyncio.run(run_simulated_demo())
    else:
        run_real_demo()
