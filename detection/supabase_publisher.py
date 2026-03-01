"""
supabase_publisher.py — Bridge between detection layer and Supabase.

When fire is detected, this module writes the event to the Supabase
`hazards` table so the Sentinel-Mesh dashboard and mesh_router can
read it without polling the /fire endpoint directly.
"""

import os
import logging
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger("SafeEdge.SupabasePublisher")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def publish(latitude: float, longitude: float, building_name: str = "Unknown"):
    """Insert a fire hazard into Supabase hazards table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set — skipping publish")
        return

    url = f"{SUPABASE_URL}/rest/v1/hazards"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    payload = {
        "name": f"{building_name} Fire",
        "latitude": latitude,
        "longitude": longitude,
        "status": "active",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        if resp.status_code in (200, 201):
            logger.info(f"Published fire to Supabase: {building_name} ({latitude}, {longitude})")
        else:
            logger.warning(f"Supabase publish failed ({resp.status_code}): {resp.text}")
    except Exception as e:
        logger.warning(f"Supabase publish error: {e}")
