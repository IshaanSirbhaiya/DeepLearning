"""
SafeEdge AI Narrator — OpenAI GPT-4o-mini narrative generation for report sections.
Synchronous API (batch report generation, not real-time).
Falls back to plain data summary if API fails.
"""

import os
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def generate_narrative(section: str, data_summary: str, audience: str) -> str:
    """Generate a professional narrative paragraph for a report section."""
    if not os.getenv("OPENAI_API_KEY"):
        return data_summary

    system = (
        f"You are a fire safety intelligence analyst writing for {audience}. "
        "Write clear, data-driven narrative paragraphs. Be concise but insightful. "
        "Include specific numbers. Do NOT use markdown. Write 2-4 sentences in "
        "professional report style."
    )

    try:
        resp = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Section: {section}\n\nData:\n{data_summary}\n\nWrite a professional narrative paragraph."},
            ],
            max_tokens=300,
            temperature=0.4,
        )
        text = resp.choices[0].message.content.strip()
        # Sanitize unicode characters that fpdf2 Helvetica can't render
        text = text.replace("\u2014", "-").replace("\u2013", "-")   # em/en dash
        text = text.replace("\u2018", "'").replace("\u2019", "'")   # smart quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2026", "...").replace("\u2022", "-") # ellipsis, bullet
        return text
    except Exception as e:
        return data_summary
