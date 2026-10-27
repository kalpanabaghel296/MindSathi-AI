"""Risk level helpers shared by the trainer and the dashboard."""
from __future__ import annotations

ORDER = {"low": 0, "moderate": 1, "high": 2}


def level_from_prob(p: float, t_moderate: float, t_high: float) -> str:
    return "high" if p >= t_high else "moderate" if p >= t_moderate else "low"


def apply_safety_override(level: str, phq9_item9: int | None = None, crisis_text: bool = False) -> str:
    """A positive PHQ-9 item 9 (thoughts of self-harm) or crisis wording is ALWAYS high, whatever the model says."""
    if crisis_text or (phq9_item9 is not None and phq9_item9 > 0):
        return "high"
    return level


SUPPORT = {
    "low": {"title": "Keep your routine steady",
            "actions": ["Re-check in 2 to 4 weeks.", "Small habits that tend to help: regular sleep time, a daily walk, time with people you like."]},
    "moderate": {"title": "Worth some extra care",
                 "actions": ["Talk with Saathi or someone you trust this week.", "Consider booking a chat with a counsellor or doctor.", "Do a daily check-in so changes are easy to spot."]},
    "high": {"title": "Please involve a real person",
             "actions": ["Talk to a counsellor or doctor soon.", "Tele-MANAS: 14416 (free, 24x7). Emergency: 112.", "With your consent, a trusted person can be told."]},
}
