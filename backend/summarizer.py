"""
SafeEdge - OpenAI Incident Summarizer
Generates natural language incident summaries for first responders using GPT-4o-mini.
Falls back to template-based summaries when API is unavailable.
"""

import os
import json
from openai import AsyncOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """You are a fire safety incident summarizer for the SafeEdge early detection system.
Given structured alert data from a fire detection system, generate a concise 2-3 sentence incident summary
for first responders. Be direct, factual, and actionable. Include:
- What was detected and where
- Confidence/severity level
- Recommended immediate action
Do NOT use markdown. Output plain text only."""


def _template_summary(alert: dict) -> str:
    """Fallback template-based summary when OpenAI API is unavailable."""
    location = alert.get("location", {})
    building = location.get("building", "Unknown building")
    floor = location.get("floor", "Unknown floor")
    zone = location.get("zone", "Unknown zone")
    confidence = alert.get("confidence", 0)
    risk = alert.get("risk_score", "UNKNOWN")
    event = alert.get("event", "fire_detected").replace("_", " ").title()
    camera = alert.get("camera_id", "Unknown camera")

    pct = int(confidence * 100)

    if risk == "CRITICAL":
        action = "Immediate dispatch recommended. Evacuate all occupants in the affected zone."
    elif risk == "HIGH":
        action = "Dispatch recommended. Alert building management and prepare evacuation."
    else:
        action = "Monitor closely. Verify with on-site personnel."

    return (
        f"{event} at {building}, Floor {floor} ({zone}). "
        f"Detected by camera {camera} with {pct}% confidence. Severity: {risk}. "
        f"{action}"
    )


async def generate_summary(alert: dict) -> str:
    """
    Generate an AI-powered incident summary using OpenAI GPT-4o-mini.
    Falls back to template if API key is missing or call fails.
    """
    if not OPENAI_API_KEY:
        return _template_summary(alert)

    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        location = alert.get("location", {})
        user_message = json.dumps({
            "camera_id": alert.get("camera_id"),
            "event": alert.get("event"),
            "confidence": alert.get("confidence"),
            "risk_score": alert.get("risk_score"),
            "timestamp": alert.get("timestamp"),
            "building": location.get("building"),
            "floor": location.get("floor"),
            "zone": location.get("zone"),
        }, indent=2)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate incident summary:\n{user_message}"},
            ],
            max_tokens=200,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return _template_summary(alert)
