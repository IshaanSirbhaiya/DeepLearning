"""
alert_generator.py — SafeEdge Detection Layer
Creates structured JSON alerts with blurred snapshot, metadata, and severity.
Optionally calls OpenAI Vision API for a second-opinion description.
"""

import os
import json
import base64
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

from detection.risk_scorer import ScoreResult, RiskLevel
from detection.privacy_filter import PrivacyFilter

logger = logging.getLogger("SafeEdge.AlertGenerator")

ALERTS_DIR = Path("alerts")
ALERTS_DIR.mkdir(exist_ok=True)


# ─── Alert Schema ─────────────────────────────────────────────────────────────

@dataclass
class FireAlert:
    alert_id:        str
    camera_id:       str
    event:           str              # "fire_detected" | "smoke_detected" | "fire_and_smoke"
    confidence:      float
    risk_score:      str              # RiskLevel string
    timestamp:       str              # ISO-8601 UTC
    location:        Dict[str, Any]   # building, floor, zone
    snapshot_path:   str              # path to blurred image on disk
    snapshot_b64:    Optional[str]    # base64 of blurred image (for API transport)
    positive_frames: int
    window_size:     int
    dual_class:      bool
    vision_analysis: Optional[Dict]   # filled if OpenAI Vision enabled
    edge_metrics:    Dict[str, float] # fps, cpu_pct, mem_mb

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("snapshot_b64", None)   # strip b64 from file-saved version
        return json.dumps(d, indent=2)

    def to_dict(self) -> Dict:
        return asdict(self)


# ─── Location Config ──────────────────────────────────────────────────────────

@dataclass
class LocationConfig:
    building: str  = "Block 4A"
    floor:    int  = 1
    zone:     str  = "unknown"


# ─── Alert Generator ──────────────────────────────────────────────────────────

class AlertGenerator:
    """
    Creates, saves, and optionally enriches fire alerts with OpenAI Vision.

    Usage:
        gen = AlertGenerator(camera_id="CAM_01", location=LocationConfig(...))
        alert = gen.generate(frame, score_result, edge_metrics)
    """

    def __init__(
        self,
        camera_id:          str = "CAM_01",
        location:           Optional[LocationConfig] = None,
        save_snapshots:     bool = True,
        include_b64:        bool = True,
        use_vision_api:     bool = False,   # requires OPENAI_API_KEY env var
        openai_api_key:     Optional[str] = None,
    ):
        self.camera_id        = camera_id
        self.location         = location or LocationConfig()
        self.save_snapshots   = save_snapshots
        self.include_b64      = include_b64
        self.use_vision_api   = use_vision_api
        self._api_key         = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._privacy         = PrivacyFilter()
        self._alert_counter   = 0

        if use_vision_api and not self._api_key:
            logger.warning(
                "OPENAI_API_KEY not set — Vision API disabled. "
                "Set env var to enable AI-powered alert enrichment."
            )
            self.use_vision_api = False

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        frame: np.ndarray,
        score: ScoreResult,
        edge_metrics: Optional[Dict[str, float]] = None,
    ) -> FireAlert:
        """
        Generate a complete FireAlert from a confirmed detection.
        Applies face blur, saves snapshot, optionally calls Claude Vision.
        """
        self._alert_counter += 1
        alert_id = f"ALERT_{self.camera_id}_{int(time.time())}_{self._alert_counter:04d}"

        # 1. Blur faces
        blurred_frame, face_count = self._privacy.apply(frame)
        if face_count > 0:
            logger.info(f"[{alert_id}] Blurred {face_count} face(s) from snapshot")

        # 2. Determine event type
        det  = score.best_detection
        label = det.label if det else "fire"
        if score.dual_class:
            event = "fire_and_smoke"
        else:
            event = "fire_detected" if "fire" in label else "smoke_detected"

        # 3. Save snapshot
        snapshot_path = ""
        if self.save_snapshots:
            snapshot_path = self._save_snapshot(blurred_frame, alert_id, score.risk_level)

        # 4. Base64 encode for API transport
        snapshot_b64 = None
        if self.include_b64:
            snapshot_b64 = self._encode_b64(blurred_frame)

        # 5. OpenAI Vision enrichment
        vision_analysis = None
        if self.use_vision_api and snapshot_b64:
            vision_analysis = self._call_openai_vision(snapshot_b64, score)

        alert = FireAlert(
            alert_id        = alert_id,
            camera_id       = self.camera_id,
            event           = event,
            confidence      = round(score.best_confidence, 4),
            risk_score      = score.risk_level,
            timestamp       = datetime.now(timezone.utc).isoformat(),
            location        = {
                "building": self.location.building,
                "floor":    self.location.floor,
                "zone":     self.location.zone,
            },
            snapshot_path   = snapshot_path,
            snapshot_b64    = snapshot_b64,
            positive_frames = score.positive_frames,
            window_size     = score.window_size,
            dual_class      = score.dual_class,
            vision_analysis = vision_analysis,
            edge_metrics    = edge_metrics or {},
        )

        # Save JSON sidecar
        if self.save_snapshots:
            json_path = snapshot_path.replace(".jpg", ".json")
            Path(json_path).write_text(alert.to_json())
            logger.info(f"[{alert_id}] Alert saved → {snapshot_path}")

        return alert

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def _save_snapshot(self, frame: np.ndarray, alert_id: str, risk: str) -> str:
        filename = f"{ALERTS_DIR}/snap_{alert_id}_{risk}_blurred.jpg"
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        cv2.imwrite(filename, frame, encode_params)
        return filename

    def _encode_b64(self, frame: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode("utf-8")

    # ── OpenAI Vision ──────────────────────────────────────────────────────────

    def _call_openai_vision(self, b64_image: str, score: ScoreResult) -> Optional[Dict]:
        """
        Call OpenAI Vision API to get a second-opinion description of the snapshot.
        Returns structured JSON analysis or None on failure.
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key)

            prompt = (
                "You are a fire safety analyst reviewing a CCTV snapshot from an automated "
                "fire detection system. The local YOLOv8 model flagged this frame as potentially "
                f"containing fire or smoke (confidence: {score.best_confidence:.0%}, "
                f"risk level: {score.risk_level}).\n\n"
                "Analyse the image and respond with ONLY a JSON object -- no markdown, no preamble:\n"
                "{\n"
                '  "fire_visible": true|false,\n'
                '  "smoke_visible": true|false,\n'
                '  "risk_level": "none|low|medium|high|critical",\n'
                '  "confidence": 0.0-1.0,\n'
                '  "description": "1-2 sentence plain English description of what you see",\n'
                '  "location_in_frame": "e.g. bottom-left corner, kitchen counter area",\n'
                '  "recommended_action": "monitor|alert_residents|evacuate_immediately",\n'
                '  "false_positive_likely": true|false,\n'
                '  "false_positive_reason": "null or brief explanation"\n'
                "}"
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            raw = response.choices[0].message.content.strip()
            # Strip any accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            analysis = json.loads(raw)
            logger.info(f"OpenAI Vision: risk={analysis.get('risk_level')} "
                        f"fp={analysis.get('false_positive_likely')}")
            return analysis

        except ImportError:
            logger.warning("openai package not installed -- skipping Vision API")
        except json.JSONDecodeError as e:
            logger.error(f"OpenAI Vision returned non-JSON: {e}")
        except Exception as e:
            logger.error(f"OpenAI Vision call failed: {e}")

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def update_location(self, building: str = None, floor: int = None, zone: str = None):
        """Dynamically update location context (e.g. per-camera config)."""
        if building: self.location.building = building
        if floor:    self.location.floor    = floor
        if zone:     self.location.zone     = zone
