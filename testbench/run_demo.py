"""
SafeEdge - One-Click Demo Script
Simulates a full fire detection -> alert -> summary pipeline for judges to test.
Run: python testbench/run_demo.py
"""

import sys
import os
import io
import time
import json
import asyncio
from pathlib import Path

# Fix Windows console encoding for Unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# --- Demo Alert Scenarios ---

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


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


async def run_demo():
    """Run the full demo pipeline without needing the server running."""
    from backend.database import init_db, store_alert, get_alerts, get_unforwarded_alerts, mark_forwarded
    from backend.summarizer import generate_summary

    print_header("SafeEdge - Fire Safety Intelligence System Demo")
    print("This demo simulates the full detection -> alert -> summary pipeline.\n")

    # Step 1: Initialize database
    print("[1/5] Initializing database...")
    db_path = "testbench/demo.db"
    init_db(db_path)
    print("       Database ready.\n")

    # Step 2: Simulate fire detection alerts
    print("[2/5] Simulating fire detection from CCTV cameras...\n")

    for i, alert in enumerate(DEMO_ALERTS):
        from datetime import datetime
        alert["timestamp"] = datetime.now().isoformat()

        risk = alert["risk_score"]
        conf = alert["confidence"]
        loc = alert["location"]
        cam = alert["camera_id"]

        # Risk indicator
        indicator = {"WARNING": "[!]", "HIGH": "[!!]", "CRITICAL": "[!!!]"}.get(risk, "[?]")

        print(f"  {indicator} Alert {i+1}: {alert['event'].replace('_', ' ').upper()}")
        print(f"     Camera: {cam} | Confidence: {conf:.0%} | Risk: {risk}")
        print(f"     Location: {loc['building']}, Floor {loc['floor']}, {loc['zone']}")

        alert_id = store_alert(alert, db_path)
        print(f"     Stored as alert #{alert_id}")
        print()
        time.sleep(1)

    # Step 3: Generate AI summaries
    print("[3/5] Generating AI incident summaries...\n")

    alerts = get_alerts(db_path=db_path)
    for alert in alerts:
        summary = await generate_summary(alert)
        print(f"  Alert #{alert['id']} summary:")
        print(f"     \"{summary}\"\n")
        time.sleep(0.5)

    # Step 4: Store-and-forward demo
    print("[4/5] Demonstrating store-and-forward (offline capability)...\n")

    pending = get_unforwarded_alerts(db_path=db_path)
    print(f"  {len(pending)} alerts pending forward to communication layer")
    for p in pending:
        print(f"    - Alert #{p['id']}: {p['event']} ({p['risk_score']})")
        mark_forwarded(p["id"], db_path)
    print(f"  All {len(pending)} alerts marked as forwarded.\n")

    pending_after = get_unforwarded_alerts(db_path=db_path)
    print(f"  Pending after forward: {len(pending_after)} (should be 0)\n")

    # Step 5: Summary stats
    print("[5/5] System Statistics\n")

    all_alerts = get_alerts(db_path=db_path)
    print(f"  Total alerts:    {len(all_alerts)}")
    print(f"  CRITICAL:        {sum(1 for a in all_alerts if a.get('risk_score') == 'CRITICAL')}")
    print(f"  HIGH:            {sum(1 for a in all_alerts if a.get('risk_score') == 'HIGH')}")
    print(f"  WARNING:         {sum(1 for a in all_alerts if a.get('risk_score') == 'WARNING')}")

    print_header("Demo Complete")
    print("In production, these alerts would be sent to:")
    print("  - Telegram bot -> residents with evacuation directions")
    print("  - Command dashboard -> authorities with AI summaries")
    print("  - Google Maps -> dynamic evacuation routes avoiding fire zone\n")

    # Cleanup demo db
    if os.path.exists(db_path):
        os.remove(db_path)
        # Also remove WAL/SHM files if they exist
        for ext in ["-wal", "-shm"]:
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    asyncio.run(run_demo())
