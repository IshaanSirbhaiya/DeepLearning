"""
SafeEdge - FastAPI Backend Server
Receives fire detection alerts, stores them, generates AI summaries,
and exposes API endpoints for the communication layer (Telegram bot).
"""

import os
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.database import (
    init_db,
    store_alert,
    get_alerts,
    get_alert_by_id,
    get_unforwarded_alerts,
    mark_forwarded,
    update_alert_summary,
    register_camera,
    get_cameras,
)
from backend.summarizer import generate_summary


# --- Pydantic Models ---

class LocationModel(BaseModel):
    building: str = "Unknown"
    floor: int = 0
    zone: str = "Unknown"


class AlertCreate(BaseModel):
    camera_id: str
    event: str = "fire_detected"
    confidence: float = Field(ge=0.0, le=1.0)
    risk_score: str = Field(pattern="^(WARNING|HIGH|CRITICAL)$")
    timestamp: str | None = None
    location: LocationModel = LocationModel()
    snapshot_path: str | None = None


class AlertResponse(BaseModel):
    id: int
    camera_id: str
    event: str
    confidence: float
    risk_score: str
    timestamp: str
    location: dict
    snapshot_path: str | None
    summary: str | None
    forwarded: int
    created_at: str


class CameraRegister(BaseModel):
    id: str
    name: str | None = None
    building: str | None = None
    floor: int | None = None
    zone: str | None = None
    rtsp_url: str | None = None


# --- App Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Ensure alerts directory exists
    Path("alerts").mkdir(exist_ok=True)
    yield


app = FastAPI(
    title="SafeEdge API",
    description="Edge-based fire safety intelligence system - backend API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve alert snapshots as static files
alerts_dir = Path("alerts")
alerts_dir.mkdir(exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=str(alerts_dir)), name="snapshots")


# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "running", "service": "SafeEdge API", "timestamp": datetime.now().isoformat()}


@app.post("/alert", response_model=dict)
async def receive_alert(alert: AlertCreate):
    """
    Receive a fire detection alert from the detection layer.
    Stores it in SQLite and generates an AI summary.
    """
    alert_dict = alert.model_dump()
    alert_dict["location"] = alert.location.model_dump()

    if not alert_dict.get("timestamp"):
        alert_dict["timestamp"] = datetime.now().isoformat()

    # Store alert
    alert_id = store_alert(alert_dict)

    # Generate AI summary (non-blocking for MVP, but we await it here)
    summary = await generate_summary(alert_dict)
    if summary:
        update_alert_summary(alert_id, summary)

    return {
        "alert_id": alert_id,
        "status": "received",
        "risk_score": alert.risk_score,
        "summary": summary,
    }


@app.post("/alert/with-snapshot", response_model=dict)
async def receive_alert_with_snapshot(
    camera_id: str = Form(...),
    event: str = Form("fire_detected"),
    confidence: float = Form(...),
    risk_score: str = Form(...),
    timestamp: str = Form(None),
    building: str = Form("Unknown"),
    floor: int = Form(0),
    zone: str = Form("Unknown"),
    snapshot: UploadFile = File(None),
):
    """
    Receive alert with an image snapshot (multipart form upload).
    Used when the detection layer sends a blurred snapshot along with the alert.
    """
    snapshot_path = None
    if snapshot:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snap_{camera_id}_{ts}.jpg"
        snapshot_path = f"alerts/{filename}"
        content = await snapshot.read()
        with open(snapshot_path, "wb") as f:
            f.write(content)

    alert_dict = {
        "camera_id": camera_id,
        "event": event,
        "confidence": confidence,
        "risk_score": risk_score,
        "timestamp": timestamp or datetime.now().isoformat(),
        "location": {"building": building, "floor": floor, "zone": zone},
        "snapshot_path": snapshot_path,
    }

    alert_id = store_alert(alert_dict)

    summary = await generate_summary(alert_dict)
    if summary:
        update_alert_summary(alert_id, summary)

    return {
        "alert_id": alert_id,
        "status": "received",
        "risk_score": risk_score,
        "summary": summary,
        "snapshot_path": snapshot_path,
    }


@app.get("/alerts", response_model=list[dict])
def list_alerts(limit: int = 50, risk_score: str = None):
    """Get recent alerts, optionally filtered by risk score."""
    return get_alerts(limit=limit, risk_score=risk_score)


@app.get("/alerts/{alert_id}", response_model=dict)
def get_single_alert(alert_id: int):
    """Get a single alert by ID."""
    alert = get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@app.get("/alerts/pending/forward", response_model=list[dict])
def get_pending_alerts():
    """
    Get alerts not yet forwarded to the communication layer.
    Used by the Telegram bot to poll for new alerts (store-and-forward pattern).
    """
    return get_unforwarded_alerts()


@app.post("/alerts/{alert_id}/forwarded")
def mark_alert_forwarded(alert_id: int):
    """Mark an alert as forwarded to the communication layer."""
    alert = get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    mark_forwarded(alert_id)
    return {"status": "marked_forwarded", "alert_id": alert_id}


@app.post("/cameras", response_model=dict)
def add_camera(camera: CameraRegister):
    """Register a new camera in the system."""
    register_camera(camera.model_dump())
    return {"status": "registered", "camera_id": camera.id}


@app.get("/cameras", response_model=list[dict])
def list_cameras():
    """Get all registered cameras."""
    return get_cameras()


@app.get("/stats")
def get_stats():
    """Get alert statistics for the dashboard."""
    all_alerts = get_alerts(limit=1000)
    total = len(all_alerts)
    critical = sum(1 for a in all_alerts if a.get("risk_score") == "CRITICAL")
    high = sum(1 for a in all_alerts if a.get("risk_score") == "HIGH")
    warning = sum(1 for a in all_alerts if a.get("risk_score") == "WARNING")
    pending = len(get_unforwarded_alerts())

    return {
        "total_alerts": total,
        "critical": critical,
        "high": high,
        "warning": warning,
        "pending_forward": pending,
        "cameras": len(get_cameras()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
