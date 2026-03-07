"""
run_live.py — Single-command launcher for the REAL SafeEdge E2E flow.

Launches detection + Telegram bot + Streamlit dashboard as parallel
subprocesses. For the live pitch demo — requires API keys in .env.

Usage:
    python testbench/run_live.py                          # webcam + all
    python testbench/run_live.py --input fire.mp4         # fire video + all
    python testbench/run_live.py --skip-detection         # telegram + dashboard only
"""

import subprocess
import sys
import os
import time
import signal
import argparse
import atexit
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

children = []

def cleanup():
    for p in children:
        try:
            p.terminate()
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

atexit.register(cleanup)

def signal_handler(sig, frame):
    print("\n[run_live] Shutting down all services...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def check_env():
    missing = []
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        missing.append("TELEGRAM_BOT_TOKEN")
    if not os.getenv("SUPABASE_URL"):
        missing.append("SUPABASE_URL")
    if not os.getenv("SUPABASE_KEY"):
        missing.append("SUPABASE_KEY")
    if missing:
        print(f"[run_live] WARNING: Missing env vars: {', '.join(missing)}")
        print(f"[run_live] Some features may not work. Check your .env file.")
    else:
        print("[run_live] All API keys found in .env")


def wait_for_api(url, timeout=30):
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description="SafeEdge Live E2E Launcher")
    parser.add_argument("--input", default="0", help="Video file or webcam index (default: 0 = webcam)")
    parser.add_argument("--skip-detection", action="store_true", help="Skip detection, only run Telegram + Dashboard")
    args = parser.parse_args()

    print("=" * 60)
    print("  SafeEdge Live E2E Demo")
    print("=" * 60)

    check_env()

    # --- 1. Detection Layer ---
    if not args.skip_detection:
        print("\n[1/3] Starting Detection Layer...")
        det_cmd = [sys.executable, str(ROOT / "detection" / "detector.py"),
                   "--input", args.input, "--serve"]
        det_proc = subprocess.Popen(det_cmd, cwd=str(ROOT))
        children.append(det_proc)

        print("[1/3] Waiting for detection API (localhost:8001)...")
        if wait_for_api("http://localhost:8001/health", timeout=30):
            print("[1/3] Detection API ready!")
        else:
            print("[1/3] WARNING: Detection API not responding after 30s, continuing anyway...")
    else:
        print("\n[1/3] Detection SKIPPED (--skip-detection)")

    # --- 2. Telegram Bot + Evacuation ---
    print("\n[2/3] Starting Telegram Bot + Evacuation Router...")
    bot_proc = subprocess.Popen(
        [sys.executable, str(ROOT / "mesh_router.py")],
        cwd=str(ROOT)
    )
    children.append(bot_proc)
    time.sleep(3)  # let bot send mass alert before dashboard opens

    # --- 3. Streamlit Dashboard ---
    print("\n[3/3] Starting Streamlit Dashboard...")
    dash_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.headless", "true"],
        cwd=str(ROOT)
    )
    children.append(dash_proc)

    print("\n" + "=" * 60)
    print("  All services running!")
    if not args.skip_detection:
        print("  - Detection:  http://localhost:8001")
    print("  - Dashboard:  http://localhost:8501")
    print("  - Telegram:   Check your phone for alerts")
    print("  Press Ctrl+C to stop all services")
    print("=" * 60)

    # Wait for any child to exit
    try:
        while True:
            for p in children:
                if p.poll() is not None:
                    print(f"\n[run_live] Process exited with code {p.returncode}")
                    cleanup()
                    sys.exit(p.returncode)
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
