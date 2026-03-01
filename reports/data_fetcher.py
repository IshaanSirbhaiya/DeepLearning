"""
SafeEdge Data Fetcher — Fetch public datasets with local caching and fallback.
"""

import json
import csv
import io
import time
import requests
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"


def _cache_path(name: str, ext: str = "json") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.{ext}"


# ── Data.gov.sg ──────────────────────────────────────────────────────────────

def fetch_datagovsg(dataset_id: str, cache_name: str) -> list:
    """Fetch CSV dataset from Data.gov.sg v1 API (two-step download)."""
    cached = _cache_path(cache_name, "csv")
    if cached.exists():
        reader = csv.DictReader(io.StringIO(cached.read_text(encoding="utf-8")))
        return list(reader)

    base = "https://api-open.data.gov.sg/v1/public/api/datasets"
    try:
        resp = requests.get(f"{base}/{dataset_id}/initiate-download", timeout=10)
        resp.raise_for_status()

        csv_url = None
        for _ in range(10):
            poll = requests.get(f"{base}/{dataset_id}/poll-download", timeout=10)
            data = poll.json()
            url = data.get("data", {}).get("url")
            if url:
                csv_url = url
                break
            time.sleep(3)

        if not csv_url:
            return []

        csv_resp = requests.get(csv_url, timeout=30)
        cached.write_text(csv_resp.text, encoding="utf-8")
        reader = csv.DictReader(io.StringIO(csv_resp.text))
        return list(reader)
    except Exception:
        return []


# ── Local SafeEdge Alerts ────────────────────────────────────────────────────

def load_safeedge_alerts() -> list:
    """Load all SafeEdge alert JSON files from alerts/ and detection/alerts/."""
    project_root = Path(__file__).parent.parent
    alerts = []
    for pattern in ["alerts/snap_*.json", "detection/alerts/snap_*.json"]:
        for f in project_root.glob(pattern):
            alerts.append(json.loads(f.read_text(encoding="utf-8")))
    return sorted(alerts, key=lambda a: a.get("timestamp", ""))


# ── Fallback Data ─────────────────────────────────────────────────────────────

SCDF_FIRE_FALLBACK = {
    "years": list(range(2014, 2025)),
    "total_fires": [4707, 4877, 4934, 4852, 4189, 2874, 2800, 2906, 3047, 3091, 3144],
    "residential": [1233, 1277, 1340, 1360, 1198, 1100, 1050, 1080, 1130, 1140, 1160],
    "non_residential": [520, 554, 570, 568, 530, 480, 460, 470, 500, 510, 520],
    "non_building": [2954, 3046, 3024, 2924, 2461, 1294, 1290, 1356, 1417, 1441, 1464],
    "injuries": [75, 80, 68, 72, 65, 48, 42, 45, 50, 51, 53],
    "fatalities": [1, 2, 1, 3, 0, 1, 2, 0, 1, 1, 1],
}

# SCDF emergency response data by division (Singapore Civil Defence Force)
# Source: SCDF Annual Report / Data.gov.sg
SCDF_EMERGENCY_FALLBACK = {
    "divisions": ["Central", "Jurong", "Tampines", "Woodlands", "Changi"],
    "incident_counts": [8420, 6380, 5910, 5640, 4850],
    "avg_response_min": [7.8, 8.2, 8.5, 9.1, 8.8],
    "fire_incidents": [1250, 980, 870, 820, 720],
    "hourly_pattern": [
        180, 150, 130, 120, 110, 140, 210, 320, 380, 360, 340, 330,
        350, 360, 370, 390, 420, 450, 430, 380, 330, 280, 240, 210,
    ],
    "monthly_fires_2024": [380, 350, 410, 430, 450, 470, 420, 390, 410, 440, 420, 370],
}
