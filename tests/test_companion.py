"""Safety and companion tests. They need no data files and no network."""
import os
from datetime import date, timedelta

import pytest

os.environ.pop("ANTHROPIC_API_KEY", None)  # tests must use the offline templates
from app import companion, risk, safety, storage  # noqa: E402

CRISIS = ["I want to kill myself", "I don't want to live anymore", "i feel like i should end my life", "I want to die",
          "mujhe marna chahta hoon", "jeene ka mann nahi hai", "main khud ko khatam kar lunga", "मैं आत्महत्या करना चाहता हूँ", "मुझे जीने का मन नहीं है",
          "I keep thinking about self harm", "everyone is better off without me"]
NORMAL = ["I had a good day", "I can't sleep at night", "mujhe neend nahi aa rahi", "I miss my family", "hello", "मुझे अकेलापन लगता है"]


@pytest.mark.parametrize("t", CRISIS)
def test_crisis_recall(t):
    assert safety.classify(t) == "crisis"
    r = companion.respond(t)
    assert r["level"] == "crisis" and r["source"] == "safety_template" and "14416" in r["reply"] and "112" in r["reply"]


@pytest.mark.parametrize("t", NORMAL)
def test_normal_is_not_crisis(t):
    assert safety.classify(t) != "crisis"


def test_careful_mode_persists_after_crisis():
    hist = [{"role": "user", "text": "I want to die"}, {"role": "bot", "text": "x"}]
    r = companion.respond("I don't know", hist)
    assert r["level"] == "crisis" and "14416" in r["reply"]


def test_medical_boundary_no_diagnosis():
    r = companion.respond("Do I have depression? Which medicine should I take?")
    assert r["level"] == "medical" and not companion.BAD_OUTPUT.search(r["reply"])


def test_language_detection_and_reply_language():
    assert companion.detect_lang("मुझे अकेलापन लगता है") == "hi"
    assert companion.detect_lang("mujhe bahut akela lag raha hai") == "hinglish"
    assert companion.detect_lang("I feel lonely") == "en"
    assert "Akelapan" in companion.respond("mujhe bahut akela lag raha hai")["reply"]


def test_templates_never_give_diagnosis_or_medicine_advice():
    for theme, langs in companion.REPLIES.items():
        for lang, texts in langs.items():
            for t in texts:
                assert not companion.BAD_OUTPUT.search(t), (theme, lang)
                assert t.count("?") <= 1


def test_safety_override_to_high():
    assert risk.apply_safety_override("low", phq9_item9=1) == "high"
    assert risk.apply_safety_override("low", crisis_text=True) == "high"
    assert risk.apply_safety_override("low", phq9_item9=0) == "low"
    assert risk.level_from_prob(0.2, 0.3, 0.6) == "low" and risk.level_from_prob(0.7, 0.3, 0.6) == "high"


def test_trend_uses_own_baseline():
    today = date(2026, 10, 21)
    rows = [{"day": (today - timedelta(days=i)).isoformat(), "mood": 4 if i > 6 else 2, "loneliness": 2 if i > 6 else 4} for i in range(20)]
    assert storage.trend(rows, today)["nudge"] == "down"
    assert storage.trend(rows[:2], today)["nudge"] is None  # too little data: no nudge
