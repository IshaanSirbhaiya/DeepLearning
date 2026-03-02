"""
SafeEdge — Single-Command End-to-End Demo
==========================================
Runs the FULL pipeline in ONE terminal, with ZERO API keys required:

  Phase 1: Fire Detection   (YOLO + EarlyDetector on video, CV2 window)
  Phase 2: Communication    (simulated Telegram alerts + evacuation routing)
  Phase 3: Dashboard        (Streamlit Command Centre in browser)

Run:
    python testbench/run_demo.py
    python testbench/run_demo.py --input testbench/my_fire_video.mp4
    python testbench/run_demo.py --no-display   # headless (no CV2 window)

Press Q in the video window to skip to Phase 2.
Press Ctrl+C to exit at any time.
"""

import sys
import os
import io
import time
import json
import math
import subprocess
import argparse
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

# Disable Vision 2FA (no API key needed for judges)
os.environ["USE_CLAUDE_VISION"] = "false"

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ═════════════════════════════════════════════════════════════
# UTILITY
# ═════════════════════════════════════════════════════════════

def header(text):
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"{'='*65}\n")


def phase(num, text):
    print(f"\n{'─'*65}")
    print(f"  PHASE {num}: {text}")
    print(f"{'─'*65}\n")


# Haversine distance (copied from mesh_router.py to avoid import issues)
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))


# 9 NTU Assembly Areas (same as mesh_router.py)
SAFE_ZONES = {
    "North Spine Plaza":          (1.3468, 103.6810),
    "South Spine Plaza":          (1.3425, 103.6832),
    "Sports & Rec Centre (SRC)":  (1.3496, 103.6835),
    "Yunnan Garden (Open Field)": (1.3458, 103.6858),
    "Hall 1-3 Field":             (1.3540, 103.6855),
    "Innovation Centre Carpark":  (1.3448, 103.6785),
    "The Arc (North Spine CP-E)": (1.3475, 103.6800),
    "CCEB Assembly (CW4)":        (1.3435, 103.6870),
    "CCDS Assembly (N3 Carpark)": (1.3460, 103.6790),
}


# ═════════════════════════════════════════════════════════════
# PHASE 1: DETECTION
# ═════════════════════════════════════════════════════════════

def run_detection(video_path, display=True):
    """Run YOLO + EarlyDetector on video. Returns alert dict when fire confirmed."""
    import cv2

    phase(1, "FIRE DETECTION")
    print(f"  Video:  {video_path}")
    print(f"  Vision 2FA: DISABLED (no API key needed)")
    print(f"    -> In production, OpenAI GPT-4o confirms each detection")
    print(f"    -> Catches false positives (e.g. warm skin tones)")
    print(f"  Display: {'ON' if display else 'OFF'}\n")

    # Load YOLO model
    try:
        from ultralytics import YOLO
        model_path = PROJECT_ROOT / "detection" / "models" / "fire_smoke.pt"
        if not model_path.exists():
            print("  [INFO] YOLO model not found locally, downloading from HuggingFace...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="keremberke/yolov8n-fire-smoke-detection",
                            filename="best.pt", local_dir=str(model_path.parent))
            os.rename(str(model_path.parent / "best.pt"), str(model_path))
        model = YOLO(str(model_path))
        print(f"  [INFO] YOLO model loaded: {model.names}\n")
    except ImportError:
        print("  [ERROR] ultralytics not installed. Run: pip install -r requirements.txt")
        return None

    # Load EarlyDetector
    try:
        from early_detector import EarlyFireDetector
        early = EarlyFireDetector()
        print("  [INFO] EarlyFireDetector loaded (optical flow + bg sub + texture)\n")
    except ImportError:
        early = None
        print("  [WARN] EarlyFireDetector not available\n")

    FIRE_CLASSES = {"fire", "smoke", "flame", "Fire", "Smoke"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return None

    frame_idx = 0
    fire_frames = 0
    fire_confirmed = False
    confirm_time = None
    alert_data = None
    start = time.time()

    print("  Scanning video for fire/smoke...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Early detector (lightweight — run on every frame)
        early_result = None
        if early and frame_idx > 5:
            early_result = early.update(frame, frame_idx)
            if early_result and early_result.should_alert:
                print(f"  [EARLY]  EARLY WARNING — anomaly score {early_result.anomaly_score:.2f}, "
                      f"signals: {early_result.active_signals}")

        # YOLO (every 5th frame for smooth playback)
        yolo_fire = False
        detections = []
        run_yolo = (frame_idx % 5 == 0)
        if run_yolo:
            results = model.predict(frame, conf=0.45, iou=0.45, imgsz=640, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_name = r.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if cls_name in FIRE_CLASSES:
                        yolo_fire = True
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        detections.append((cls_name, conf, x1, y1, x2, y2))

            # Only update fire counter on YOLO frames
            if yolo_fire:
                fire_frames += 1
                max_conf = max(d[1] for d in detections)
                risk = "CRITICAL" if max_conf >= 0.7 else "HIGH" if max_conf >= 0.5 else "WARNING"
                print(f"  [YOLO]   {detections[0][0]} detected — conf={max_conf:.2f} "
                      f"({fire_frames}/8 frames) — risk: {risk}")

                if fire_frames >= 5 and not fire_confirmed:
                    fire_confirmed = True
                    confirm_time = time.time()
                    elapsed = time.time() - start
                    fire_lat, fire_lng = 1.34321, 103.68275
                    alert_data = {
                        "fire_detected": True,
                        "latitude": fire_lat,
                        "longitude": fire_lng,
                        "location": {"building": "The Hive", "floor": 2, "zone": "Collaboration Studio"},
                        "confidence": round(max_conf, 4),
                        "risk_level": risk,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "camera_id": "CAM_01",
                        "detection_time_sec": round(elapsed, 1),
                    }
                    print(f"\n  [ALERT]  FIRE CONFIRMED! {fire_frames}/8 frames positive")
                    print(f"  [ALERT]  Location: The Hive, Floor 2, Collaboration Studio")
                    print(f"  [ALERT]  Detection time: {elapsed:.1f}s\n")
                    print(f"  [ALERT]  JSON output:")
                    print(f"  {json.dumps(alert_data, indent=2)}\n")

                    if display:
                        print("  Press Q in video window to continue to Phase 2...\n")
            else:
                fire_frames = max(0, fire_frames - 1)

        # Display with bounding boxes
        if display:
            vis = frame.copy()
            for cls_name, conf, x1, y1, x2, y2 in detections:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Status bar
            status = "FIRE DETECTED" if fire_confirmed else "SCANNING..."
            color = (0, 0, 255) if fire_confirmed else (0, 200, 0)
            cv2.putText(vis, f"SafeEdge | {status} | Frame {frame_idx}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("SafeEdge Detection", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        # Auto-advance 3 seconds after fire confirmed
        if confirm_time and (time.time() - confirm_time) > 3:
            break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    if not fire_confirmed:
        print("  [INFO] Video ended without confirmed fire detection.")
        print("  [INFO] Creating simulated alert for demo purposes...\n")
        alert_data = {
            "fire_detected": True,
            "latitude": 1.34321,
            "longitude": 103.68275,
            "location": {"building": "The Hive", "floor": 2, "zone": "Collaboration Studio"},
            "confidence": 0.85,
            "risk_level": "HIGH",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "camera_id": "CAM_01",
            "detection_time_sec": round(time.time() - start, 1),
        }

    return alert_data


# ═════════════════════════════════════════════════════════════
# PHASE 2: COMMUNICATION (SIMULATED)
# ═════════════════════════════════════════════════════════════

def run_communication(alert):
    """Simulate Telegram alerts + evacuation routing. No API keys needed."""
    phase(2, "COMMUNICATION & EVACUATION ROUTING")

    fire_lat = alert["latitude"]
    fire_lng = alert["longitude"]
    building = alert["location"]["building"]

    # Simulate Telegram broadcast
    users = [
        ("Abhishek Vulla", "8112299954"),
        ("Ishaan Sirbhaiya", "5808527465"),
        ("Naman Kumar", "1674734061"),
        ("Abhishek Vulla", "5574113481"),
    ]

    print(f"  [TELEGRAM] Broadcasting CRITICAL ALARM to {len(users)} registered users...")
    for name, uid in users:
        time.sleep(0.3)
        print(f"  [TELEGRAM] Alert sent to: {name} (ID: {uid})")

    print()

    # Simulate user responses with real routing math
    simulated_users = [
        ("Abhishek Vulla", 1.3445, 103.6830),      # ~180m from Hive = ENDANGERED
        ("Ishaan Sirbhaiya", 1.3520, 103.6800),     # ~980m from Hive = SAFE
        ("Naman Kumar", 1.3440, 103.6815),           # ~150m from Hive = ENDANGERED
    ]

    evacuees = []

    for name, u_lat, u_lng in simulated_users:
        time.sleep(0.5)
        dist = calculate_distance(fire_lat, fire_lng, u_lat, u_lng)
        status = "endangered" if dist <= 350 else "secure"

        print(f"  [USER]    {name} sends location: ({u_lat}, {u_lng})")
        print(f"  [ROUTING] Distance to fire: {dist:.0f}m — {'ENDANGERED' if status == 'endangered' else 'SAFE'}")

        if status == "endangered":
            # Find nearest safe zone
            best_zone = ""
            best_dist = float("inf")
            for zone_name, (z_lat, z_lng) in SAFE_ZONES.items():
                d = calculate_distance(u_lat, u_lng, z_lat, z_lng)
                if d < best_dist:
                    best_dist = d
                    best_zone = zone_name
                    best_coords = (z_lat, z_lng)

            gmaps = f"https://www.google.com/maps/dir/?api=1&origin={u_lat},{u_lng}&destination={best_coords[0]},{best_coords[1]}&travelmode=walking"
            print(f"  [ROUTING] Nearest safe zone: {best_zone} ({best_dist:.0f}m)")
            print(f"  [ROUTING] Google Maps: {gmaps}")

        evacuees.append({
            "name": name,
            "status": status,
            "location_link": f"https://www.google.com/maps?q={u_lat},{u_lng}",
        })
        print()

    # Simulate button taps
    time.sleep(0.5)
    print(f"  [USER]    Abhishek taps 'I have reached Safety'")
    evacuees[0]["status"] = "secure"
    print(f"  [STATUS]  Abhishek Vulla -> SECURE\n")

    time.sleep(0.5)
    print(f"  [USER]    Naman taps 'EMERGENCY RESCUE'")
    evacuees[2]["status"] = "emergency help"
    print(f"  [STATUS]  Naman Kumar -> EMERGENCY HELP (SOS)\n")

    # Write local state file for dashboard
    state = {
        "hazards": [{"name": f"{building} Fire", "latitude": fire_lat, "longitude": fire_lng, "status": "active"}],
        "evacuees": evacuees,
    }
    state_path = PROJECT_ROOT / "testbench" / "demo_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"  [STATE]  Local state written to {state_path.name}")

    return state


# ═════════════════════════════════════════════════════════════
# PHASE 2b: INTERACTIVE COMMUNICATION (JUDGE PLAYS AS EVACUEE)
# ═════════════════════════════════════════════════════════════

def find_nearest_safe_zone(lat, lng):
    """Find nearest NTU assembly zone. Returns (name, dist_m, coords, gmaps_url)."""
    best_zone, best_dist, best_coords = "", float("inf"), (0, 0)
    for zone_name, (z_lat, z_lng) in SAFE_ZONES.items():
        d = calculate_distance(lat, lng, z_lat, z_lng)
        if d < best_dist:
            best_dist, best_zone, best_coords = d, zone_name, (z_lat, z_lng)
    gmaps = (f"https://www.google.com/maps/dir/?api=1"
             f"&origin={lat},{lng}&destination={best_coords[0]},{best_coords[1]}"
             f"&travelmode=walking")
    return best_zone, best_dist, best_coords, gmaps


def write_demo_state(hazards, evacuees):
    """Write demo_state.json for dashboard to read."""
    state_path = PROJECT_ROOT / "testbench" / "demo_state.json"
    with open(state_path, "w") as f:
        json.dump({"hazards": hazards, "evacuees": evacuees}, f, indent=2)


def run_communication_interactive(alert):
    """Interactive Phase 2: Judge experiences the Telegram evacuation flow as an evacuee."""
    phase(2, "COMMUNICATION & EVACUATION ROUTING")

    fire_lat = alert["latitude"]
    fire_lng = alert["longitude"]
    building = alert["location"]["building"]

    hazards = [{"name": f"{building} Fire", "latitude": fire_lat,
                "longitude": fire_lng, "status": "active"}]

    w = 53  # box width for terminal UI

    # ── Telegram Broadcast (automated) ──────────────────────────
    print(f"  [TELEGRAM] CRITICAL ALARM — Fire detected at {building}!")
    print(f"  [TELEGRAM] Broadcasting to 4 registered users...")
    for name in ["You (Judge)", "Ishaan Sirbhaiya", "Naman Kumar", "Abhishek Vulla"]:
        time.sleep(0.3)
        print(f"  [TELEGRAM] Alert sent to: {name}")

    print(f"\n  {'='*w}")
    print(f"  | {'CRITICAL ALARM':<{w-3}}|")
    print(f"  | {'Fire detected at ' + building + ', Floor 2!':<{w-3}}|")
    print(f"  |{' '*(w-2)}|")
    print(f"  | {'[ SEND MY LOCATION ]':^{w-3}}|")
    print(f"  {'='*w}\n")

    # ── Prompt 1: Share location? ───────────────────────────────
    share = input("  Do you share your location? [Y/n]: ").strip().lower()
    shared = share not in ("n", "no")

    # Judge's assumed location (near the fire)
    judge_lat, judge_lng = 1.3440, 103.6815  # ~150m from The Hive
    judge_status = "endangered"
    judge_zone_name = None
    judge_gmaps = None

    if shared:
        dist = calculate_distance(fire_lat, fire_lng, judge_lat, judge_lng)
        print(f"\n  [LOCATION] Your location received: ({judge_lat}, {judge_lng})")
        print(f"  [ROUTING]  Distance to fire: {dist:.0f}m — ENDANGERED!\n")

        # Route to nearest safe zone
        judge_zone_name, zone_dist, _, judge_gmaps = find_nearest_safe_zone(judge_lat, judge_lng)
        print(f"  [ROUTING]  Nearest safe zone: {judge_zone_name} ({zone_dist:.0f}m)")
        print(f"  [ROUTING]  Google Maps: {judge_gmaps}\n")
    else:
        print(f"\n  [STATUS]   You chose not to share. You remain untracked.\n")

    # ── Background users auto-process ───────────────────────────
    print(f"  --- Meanwhile, other users are responding ---\n")
    time.sleep(0.5)

    # Ishaan (safe)
    i_dist = calculate_distance(fire_lat, fire_lng, 1.3520, 103.6800)
    print(f"  [USER]     Ishaan sends location (North Spine area)")
    print(f"  [ROUTING]  Distance: {i_dist:.0f}m — SAFE. No evacuation needed.\n")
    time.sleep(0.3)

    # Naman (endangered)
    n_dist = calculate_distance(fire_lat, fire_lng, 1.3440, 103.6815)
    naman_zone, naman_zone_dist, _, naman_gmaps = find_nearest_safe_zone(1.3440, 103.6815)
    print(f"  [USER]     Naman sends location (near {building})")
    print(f"  [ROUTING]  Distance: {n_dist:.0f}m — ENDANGERED")
    print(f"  [ROUTING]  Naman routed to {naman_zone} ({naman_zone_dist:.0f}m)\n")

    # Build evacuees list & write state #1
    evacuees = [
        {"name": "You (Judge)", "status": judge_status,
         "location_link": f"https://www.google.com/maps?q={judge_lat},{judge_lng}"},
        {"name": "Ishaan Sirbhaiya", "status": "secure",
         "location_link": "https://www.google.com/maps?q=1.3520,103.6800"},
        {"name": "Naman Kumar", "status": "endangered",
         "location_link": "https://www.google.com/maps?q=1.3440,103.6815"},
    ]
    write_demo_state(hazards, evacuees)

    # ── Prompt 2: Status button (only if endangered & shared) ───
    if shared and judge_status == "endangered":
        print(f"  {'='*w}")
        print(f"  | {'You have been routed to: ' + judge_zone_name:<{w-3}}|")
        print(f"  |{' '*(w-2)}|")
        print(f"  | {'[1]  I have reached Safety':<{w-3}}|")
        print(f"  | {'[2]  EMERGENCY RESCUE (SOS)':<{w-3}}|")
        print(f"  {'='*w}\n")

        btn = input("  Your status? [1/2]: ").strip()
        if btn == "2":
            evacuees[0]["status"] = "emergency help"
            print(f"\n  [STATUS]  You -> EMERGENCY HELP (SOS)")
            print(f"  [STATUS]  Your SOS is now flashing red on the Command Dashboard.")
            print(f"  [STATUS]  Rescue teams notified of your exact location.\n")
        else:
            evacuees[0]["status"] = "secure"
            print(f"\n  [STATUS]  You -> SECURE")
            print(f"  [STATUS]  You have been marked safe on the Command Dashboard.\n")
        write_demo_state(hazards, evacuees)  # state write #2

    # ── Background resolution ───────────────────────────────────
    time.sleep(0.5)
    print(f"  --- Meanwhile ---\n")
    time.sleep(0.3)
    print(f"  [USER]     Naman taps 'EMERGENCY RESCUE'")
    evacuees[2]["status"] = "emergency help"
    print(f"  [STATUS]   Naman Kumar -> EMERGENCY HELP (SOS)\n")
    write_demo_state(hazards, evacuees)  # state write #3 (final)

    # ── Summary table ───────────────────────────────────────────
    print(f"  +-------------------------+------------------+")
    print(f"  | Evacuee                 | Status           |")
    print(f"  +-------------------------+------------------+")
    for e in evacuees:
        s = e["status"].upper()
        print(f"  | {e['name']:<23} | {s:<16} |")
    print(f"  +-------------------------+------------------+")
    print(f"\n  [STATE]  demo_state.json updated — Dashboard refreshing...")

    return {"hazards": hazards, "evacuees": evacuees}


# ═════════════════════════════════════════════════════════════
# PHASE 3: DASHBOARD
# ═════════════════════════════════════════════════════════════

def run_dashboard():
    """Launch Streamlit dashboard. Reads from local state file if no Supabase."""
    phase(3, "COMMAND CENTRE DASHBOARD")

    print("  [DASHBOARD] Starting Sentinel-Mesh Command Centre...")
    print("  [DASHBOARD] Open http://localhost:8501 in your browser\n")

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py",
             "--server.headless", "true"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        # Try to open browser
        import webbrowser
        time.sleep(3)
        webbrowser.open("http://localhost:8501")

        print("  [DASHBOARD] Dashboard running. Press Ctrl+C to exit.\n")
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n  [DASHBOARD] Stopped.")
    except Exception as e:
        print(f"  [DASHBOARD] Could not start: {e}")
        print("  [DASHBOARD] Run manually: streamlit run app.py\n")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SafeEdge — Full E2E Demo")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to fire video (default: testbench/sample_fire.mp4)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no CV2 window)")
    parser.add_argument("--skip-dashboard", action="store_true",
                        help="Skip launching Streamlit dashboard")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip interactive prompts (auto-simulate Phase 2)")
    args = parser.parse_args()

    # Find video
    if args.input:
        video = Path(args.input)
    else:
        video = PROJECT_ROOT / "testbench" / "sample_fire.mp4"

    if not video.exists():
        print(f"ERROR: Video not found: {video}")
        print("Place a fire video in testbench/ or specify --input path")
        sys.exit(1)

    header("SafeEdge — End-to-End Fire Detection Demo")
    print("  This demo runs the FULL pipeline with ZERO API keys.")
    print("  All external services (Telegram, Supabase) are simulated.\n")
    print(f"  Video:     {video}")
    print(f"  Display:   {'ON' if not args.no_display else 'OFF'}")
    print(f"  Dashboard: {'ON' if not args.skip_dashboard else 'SKIP'}")

    try:
        # Phase 1: Detection
        alert = run_detection(str(video), display=not args.no_display)

        # Phase 2: Communication
        if args.no_interactive:
            run_communication(alert)
        else:
            run_communication_interactive(alert)

        # Phase 3: Dashboard
        if not args.skip_dashboard:
            run_dashboard()
        else:
            header("Demo Complete")
            print("  Detection -> Alert -> Communication -> Routing")
            print("  All phases completed successfully.\n")

    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")


if __name__ == "__main__":
    main()
