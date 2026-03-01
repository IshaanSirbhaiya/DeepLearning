"""
SafeEdge - SQLite Database Layer
Stores fire alerts with store-and-forward capability for offline operation.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "safeedge.db"


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(db_path or str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    return conn


def init_db(db_path: str = None):
    """Initialize database tables."""
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            event TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_score TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            location TEXT,
            snapshot_path TEXT,
            summary TEXT,
            forwarded INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            name TEXT,
            building TEXT,
            floor INTEGER,
            zone TEXT,
            rtsp_url TEXT,
            status TEXT DEFAULT 'active'
        );

        CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
        CREATE INDEX IF NOT EXISTS idx_alerts_forwarded ON alerts(forwarded);
        CREATE INDEX IF NOT EXISTS idx_alerts_risk ON alerts(risk_score);
    """)
    conn.commit()
    conn.close()


def store_alert(alert: dict, db_path: str = None) -> int:
    """
    Store a fire detection alert. Returns the alert ID.

    Expected alert schema:
    {
        "camera_id": "CAM_01",
        "event": "fire_detected",
        "confidence": 0.92,
        "risk_score": "CRITICAL",
        "timestamp": "2026-03-02T14:23:01",
        "location": {"building": "Block 4A", "floor": 3, "zone": "kitchen"},
        "snapshot_path": "alerts/snap_001_blurred.jpg"
    }
    """
    conn = get_connection(db_path)
    location_str = json.dumps(alert.get("location", {}))
    cursor = conn.execute(
        """INSERT INTO alerts
           (camera_id, event, confidence, risk_score, timestamp, location, snapshot_path, summary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            alert["camera_id"],
            alert["event"],
            alert["confidence"],
            alert["risk_score"],
            alert.get("timestamp", datetime.now().isoformat()),
            location_str,
            alert.get("snapshot_path"),
            alert.get("summary"),
        ),
    )
    conn.commit()
    alert_id = cursor.lastrowid
    conn.close()
    return alert_id


def get_alerts(limit: int = 50, risk_score: str = None, db_path: str = None) -> list[dict]:
    """Fetch recent alerts, optionally filtered by risk score."""
    conn = get_connection(db_path)
    query = "SELECT * FROM alerts"
    params = []

    if risk_score:
        query += " WHERE risk_score = ?"
        params.append(risk_score)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    alerts = []
    for row in rows:
        alert = dict(row)
        if alert.get("location"):
            alert["location"] = json.loads(alert["location"])
        alerts.append(alert)
    return alerts


def get_alert_by_id(alert_id: int, db_path: str = None) -> dict | None:
    """Fetch a single alert by ID."""
    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    conn.close()
    if row:
        alert = dict(row)
        if alert.get("location"):
            alert["location"] = json.loads(alert["location"])
        return alert
    return None


def get_unforwarded_alerts(db_path: str = None) -> list[dict]:
    """Get alerts that haven't been forwarded to communication layer yet (store-and-forward)."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM alerts WHERE forwarded = 0 ORDER BY created_at ASC"
    ).fetchall()
    conn.close()

    alerts = []
    for row in rows:
        alert = dict(row)
        if alert.get("location"):
            alert["location"] = json.loads(alert["location"])
        alerts.append(alert)
    return alerts


def mark_forwarded(alert_id: int, db_path: str = None):
    """Mark an alert as forwarded to communication layer."""
    conn = get_connection(db_path)
    conn.execute("UPDATE alerts SET forwarded = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def update_alert_summary(alert_id: int, summary: str, db_path: str = None):
    """Update an alert with an AI-generated summary."""
    conn = get_connection(db_path)
    conn.execute("UPDATE alerts SET summary = ? WHERE id = ?", (summary, alert_id))
    conn.commit()
    conn.close()


def register_camera(camera: dict, db_path: str = None):
    """Register a camera in the system."""
    conn = get_connection(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO cameras (id, name, building, floor, zone, rtsp_url, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            camera["id"],
            camera.get("name", camera["id"]),
            camera.get("building"),
            camera.get("floor"),
            camera.get("zone"),
            camera.get("rtsp_url"),
            camera.get("status", "active"),
        ),
    )
    conn.commit()
    conn.close()


def get_cameras(db_path: str = None) -> list[dict]:
    """Get all registered cameras."""
    conn = get_connection(db_path)
    rows = conn.execute("SELECT * FROM cameras").fetchall()
    conn.close()
    return [dict(row) for row in rows]


# Initialize DB on import
init_db()
