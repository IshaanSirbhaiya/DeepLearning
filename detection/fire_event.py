"""
fire_event.py — SafeEdge Detection Layer
===========================================
THE ONLY FILE YOUR TEAMMATES NEED TO IMPORT.

This module exposes two things:
  1. A shared `fire_event` object they can subscribe to
  2. A polling endpoint at GET http://localhost:8001/fire

When fire is confirmed, teammates get exactly:
  {
    "latitude": 1.34321,
    "longitude": 103.68275
  }

If no fire is detected:
  {
    "fire_detected": false
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEAMMATE USAGE — Option A: HTTP Polling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import requests, time

while True:
    r = requests.get("http://localhost:8001/fire")
    data = r.json()
    if "latitude" in data:
        lat = data["latitude"]
        lng = data["longitude"]
        print(f"🔥 FIRE at lat={lat}, lng={lng}")
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
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional


# ── Reference: NTU hardcoded locations (selection happens in detector.py) ─────
# These are the only 4 valid fire locations. detector.py picks one randomly
# and passes the coordinates into publish(). Defined here for documentation.
#
#   The Hive                          lat=1.34321   lng=103.68275
#   Northspine                        lat=1.3431    lng=103.6805
#   School of Chemical & Biomed Eng   lat=1.34572   lng=103.67855
#   Hall of Residence 2               lat=1.3547    lng=103.6853


# ── Shared Fire Event Object ──────────────────────────────────────────────────

class FireEventBus:
    """
    Shared singleton. Detection layer writes to it.
    Teammates read from it via subscribe() or get_latest().

    Output on fire: {"latitude": float, "longitude": float}
    Output on no fire: {"fire_detected": False}
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
        latitude:   float,
        longitude:  float,
    ) -> Dict:
        """
        Called internally by detector when fire is confirmed.
        detector.py does the random NTU location selection and passes
        the chosen lat/lng here. Only those two values are stored and
        returned to teammates — nothing else.
        """
        self._event_counter += 1

        # Output payload — only lat/lng for teammates
        event = {
            "latitude":  latitude,
            "longitude": longitude,
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
        Returns latest fire event with only latitude + longitude,
        or {"fire_detected": false} if no fire has been detected yet.
        Poll this every 2 seconds from your Telegram / mapping code.
        """
        with self._lock:
            if self._latest:
                return self._latest
        return {"fire_detected": False}

    def get_history(self, n: int = 10) -> List[Dict]:
        """Last N fire events (each containing only latitude + longitude)."""
        with self._lock:
            return list(self._history[-n:])

    def subscribe(self, callback: Callable[[Dict], None]):
        """
        Register a callback — fires instantly when fire is detected.
        Callback receives {"latitude": float, "longitude": float}.

        Example:
            def on_fire(event):
                run_dijkstra(event["latitude"], event["longitude"])
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

    @app.get("/fire", summary="Get latest fire event")
    def get_fire():
        """
        Returns current fire state.
        No fire  → {"fire_detected": false}
        Fire     → {"latitude": float, "longitude": float}
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

    threading.Thread(target=simulate, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8001)
