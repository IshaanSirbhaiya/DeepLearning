"""
fire_event.py — SafeEdge Detection Layer
===========================================
THE ONLY FILE YOUR TEAMMATES NEED TO IMPORT.

This module exposes two things:
  1. A shared `fire_event` object they can subscribe to
  2. A polling endpoint at GET http://localhost:8001/fire

When fire is confirmed, teammates get exactly:
  {
    "fire_detected": true,
    "latitude": 1.34321,
    "longitude": 103.68275,
    "location": {
      "building": "The Hive",
      "floor": 2,
      "zone": "Collaboration Studio"
    },
    "confidence": 0.92,
    "risk_level": "CRITICAL",
    "timestamp": "2026-03-02T14:23:01+00:00",
    "camera_id": "CAM_01",
    "event_id": "EVT_001"
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEAMMATE USAGE — Option A: HTTP Polling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import requests, time

while True:
    r = requests.get("http://localhost:8001/fire")
    data = r.json()
    if data["fire_detected"]:
        lat  = data["latitude"]
        lng  = data["longitude"]
        risk = data["risk_level"]
        print(f"🔥 FIRE at lat={lat}, lng={lng} — {risk}")
        # → run Dijkstra, send Telegram, etc.
    time.sleep(2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEAMMATE USAGE — Option B: Direct Import
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from fire_event import fire_event

def on_fire(event):
    print(f"FIRE at lat={event['latitude']}, lng={event['longitude']}")
    # → run Dijkstra, send Telegram, etc.

fire_event.subscribe(on_fire)   # your callback fires instantly on detection
"""

import threading
import time
import os
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

# ── Shared Fire Event Object ──────────────────────────────────────────────────

class FireEventBus:
    """
    Shared singleton. Detection layer writes to it.
    Teammates read from it via subscribe() or get_latest().
    """

    def __init__(self):
        self._lock         = threading.Lock()
        self._latest:  Optional[Dict] = None
        self._history: List[Dict]     = []
        self._subscribers: List[Callable] = []
        self._event_counter = 0

    # ── Write (called by detector.py) ─────────────────────────────────────────

    def publish(
        self,
        building:   str,
        floor:      int,
        zone:       str,
        confidence: float,
        risk_level: str,
        camera_id:  str,
        latitude:   float = 0.0,
        longitude:  float = 0.0,
    ) -> Dict:
        """Called internally by detector when fire is confirmed."""
        self._event_counter += 1
        event = {
            "fire_detected": True,
            "latitude":    latitude,
            "longitude":   longitude,
            "location": {
                "building": building,
                "floor":    floor,
                "zone":     zone,
            },
            "confidence":  round(confidence, 4),
            "risk_level":  risk_level,           # "WARNING" | "HIGH" | "CRITICAL"
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "camera_id":   camera_id,
            "event_id":    f"EVT_{self._event_counter:04d}",
        }

        with self._lock:
            self._latest = event
            self._history.append(event)
            if len(self._history) > 50:
                self._history.pop(0)

        # Notify all subscribers (non-blocking)
        for cb in self._subscribers:
            threading.Thread(target=cb, args=(event,), daemon=True).start()

        return event

    # ── Read (called by teammates) ────────────────────────────────────────────

    def get_latest(self) -> Dict:
        """
        Returns latest fire event, or {"fire_detected": false} if none yet.
        Poll this every 2 seconds from your Telegram / mapping code.
        """
        with self._lock:
            if self._latest:
                return self._latest
        return {"fire_detected": False}

    def get_history(self, n: int = 10) -> List[Dict]:
        """Last N fire events."""
        with self._lock:
            return list(self._history[-n:])

    def subscribe(self, callback: Callable[[Dict], None]):
        """
        Register a callback — fires instantly when fire is detected.
        callback receives the fire event dict.

        Example:
            def on_fire(event):
                send_telegram(event["location"]["building"])
            fire_event.subscribe(on_fire)
        """
        self._subscribers.append(callback)

    def clear(self):
        """Reset after incident resolved."""
        with self._lock:
            self._latest = None


# ── Module-level singleton ─────────────────────────────────────────────────────
fire_event = FireEventBus()


# ── FastAPI routes (mounted into detector.py's app) ───────────────────────────

def register_routes(app):
    """
    Call this from detector.py to add /fire endpoints to the FastAPI app.
    Already wired in — teammates just hit http://localhost:8001/fire
    """
    from fastapi import HTTPException

    @app.get("/fire", summary="Get latest fire event")
    def get_fire():
        """
        Returns current fire state.
        fire_detected=false  → no active fire
        fire_detected=true   → fire confirmed, includes location
        """
        return fire_event.get_latest()

    @app.get("/fire/history", summary="Get last N fire events")
    def get_fire_history(n: int = 10):
        return {"events": fire_event.get_history(n)}

    @app.post("/fire/clear", summary="Clear fire state after incident resolved")
    def clear_fire():
        fire_event.clear()
        return {"status": "cleared"}

    return app


# ── Standalone demo (run directly to test without detector) ───────────────────

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="SafeEdge Fire Event API")
    register_routes(app)

    print("━" * 50)
    print("SafeEdge Fire Event Bus — Demo Mode")
    print("━" * 50)
    print("Simulating a fire detection in 3 seconds...")
    print("Poll:  GET http://localhost:8001/fire\n")

    def simulate():
        time.sleep(3)
        event = fire_event.publish(
            building   = "The Hive",
            floor      = 2,
            zone       = "Collaboration Studio",
            confidence = 0.92,
            risk_level = "CRITICAL",
            camera_id  = "CAM_01",
            latitude   = 1.34321,
            longitude  = 103.68275,
        )
        print(f"\n🔥 PUBLISHED: {event}\n")

        # Also push to Supabase for integration testing
        try:
            from supabase_publisher import publish as supabase_publish
            supabase_publish(1.34321, 103.68275, "The Hive")
            print("✅ Pushed to Supabase hazards table")
        except Exception as e:
            print(f"⚠️  Supabase push skipped: {e}")

    threading.Thread(target=simulate, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8001)
