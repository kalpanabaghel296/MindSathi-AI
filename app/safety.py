"""Safety layer for the companion. Runs BEFORE any reply is generated.

Design rule: over-trigger rather than miss. A false alarm costs one extra kind message;
a missed crisis costs far more. The word lists are a first draft for English, Hindi
(Devanagari) and Hinglish (Hindi in Roman letters) and MUST be reviewed by a counsellor
before any real use.
"""
from __future__ import annotations

import re

HELPLINES = {
    "name": "Tele-MANAS",
    "numbers": ["14416", "1-800-891-4416"],
    "note": "Free, 24x7, many languages. Number checked online on 2026-10-21; re-verify before real deployment.",
    "emergency": "112",
}

# --- crisis: self-harm / suicide wording (regex, lowercase text) -------------------------
CRISIS = [
    r"\bkill (myself|me)\b", r"\bsuicid", r"\bend (my|it all|my own) life\b", r"\bwant(ed)? to die\b",
    r"\bwish i (was|were) dead\b", r"\bbetter off (dead|without me)\b", r"\bno reason to (live|go on)\b",
    r"\b(hurt|harm|cut) myself\b", r"\bself[- ]?harm\b", r"\btake my own life\b", r"\bdon'?t want to (live|be alive|exist)\b",
    r"\bcan'?t go on\b", r"\bend (everything|this all)\b",
    # Hinglish
    r"\bmar(na| jana| jaana) (chahta|chahti|hai)\b", r"\bmarna chah", r"\bjeene ka (mann|man|mood) nahi\b",
    r"\bjeena nahi (chahta|chahti)\b", r"\bkhud ko (khatam|maar|nuksan)\b", r"\bapni jaan\b", r"\bkhatam kar (lu|lun|dun|du)\b",
    r"\baatmahatya\b", r"\bsuicide karna\b", r"\bmar jaun\b", r"\bzindagi khatam\b",
    # Devanagari
    "आत्महत्या", "मरना चाहता", "मरना चाहती", "जीने का मन नहीं", "जीना नहीं चाहता", "जीना नहीं चाहती",
    "खुद को खत्म", "अपनी जान", "जिंदगी खत्म", "ज़िंदगी खत्म", "मर जाऊं", "मर जाऊँ",
]
# --- distress: heavy feelings, not (yet) a crisis -------------------------------------------
DISTRESS = [
    r"\bhopeless\b", r"\bworthless\b", r"\bno one (cares|loves)\b", r"\bnobody (cares|loves)\b", r"\bcan'?t take (it|this)\b",
    r"\bgive up\b", r"\bburden\b", r"\bempty inside\b", r"\bcry(ing)? (all|every)\b",
    r"\bkoi nahi (hai )?(mere|meri)\b", r"\bbekar\b", r"\bthak gaya\b", r"\bthak gayi\b", r"\bha+r gay[ai]\b", r"\bha+r gaye\b",
    r"\bkisi ko (meri )?parwah nahi\b", r"\bbojh\b",
    "निराश", "बेकार", "कोई नहीं है", "थक गया", "थक गई", "हार गया", "हार गई",
]
# --- boundary topics the companion must not answer ---------------------------------------
MEDICAL = [
    r"\bdo i have depression\b", r"\bam i depressed\b", r"\bdiagnos", r"\bmedicine\b", r"\bmedication\b", r"\bantidepressant",
    r"\bdose\b", r"\btablet", r"\bpills?\b", r"\bprescri", r"\bdawai\b", r"\bdawa\b", r"\bdavai\b", r"\bgoli\b", r"\bdepression hai\b", r"\banxiety hai\b", r"\bmujhe (depression|anxiety)\b",
    r"\bkya mujhe .{0,30}(depression|anxiety)", r"\bdo i have (anxiety|bipolar|adhd|ocd)\b",
    "दवा", "दवाई", "गोली", "डिप्रेशन है",
]


def _hit(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def classify(text: str) -> str:
    """Return 'crisis', 'distress', 'medical' or 'normal'. Order matters: crisis wins."""
    t = text.lower()
    if _hit(CRISIS, t):
        return "crisis"
    if _hit(DISTRESS, t):
        return "distress"
    if _hit(MEDICAL, t):
        return "medical"
    return "normal"


def crisis_in_history(history: list[dict], window: int = 6) -> bool:
    """If the person said something crisis-level recently, stay in careful mode."""
    users = [m.get("text", "") for m in history if m.get("role") == "user"][-window:]
    return any(classify(u) == "crisis" for u in users)


CRISIS_REPLY = {
    "en": ("I'm really glad you told me. What you're feeling matters, and you deserve support from a real person right now. "
           "Please call Tele-MANAS (free, 24x7): 14416 or 1-800-891-4416. If you are in immediate danger, call 112. "
           "If someone is near you, please go to them or ask them to stay with you. I'm an AI, so I can't be there in person, "
           "but I'll stay here with you. Are you safe right now?"),
    "hinglish": ("Mujhe bahut accha laga ki aapne mujhe bataya. Aap jo feel kar rahe ho woh bahut important hai, aur abhi aapko "
                 "kisi insaan ka saath milna chahiye. Please Tele-MANAS par call karein (free, 24x7): 14416 ya 1-800-891-4416. "
                 "Agar aap abhi khatre mein ho to 112 dial karein. Agar koi paas hai to unke paas chale jaayein ya unhe apne saath "
                 "rukne ko bolein. Main ek AI hoon, isliye paas nahi aa sakta, par main yahin hoon. Kya aap abhi safe ho?"),
    "hi": ("आपने मुझे बताया, यह बहुत अच्छा हुआ। आप जो महसूस कर रहे हैं वह बहुत मायने रखता है, और अभी आपको किसी इंसान का साथ मिलना चाहिए। "
           "कृपया Tele-MANAS पर कॉल करें (मुफ़्त, 24x7): 14416 या 1-800-891-4416। अगर आप अभी खतरे में हैं तो 112 डायल करें। "
           "अगर कोई पास है तो उनके पास जाएँ या उन्हें अपने साथ रुकने को कहें। मैं एक AI हूँ, इसलिए पास नहीं आ सकता, पर मैं यहीं हूँ। क्या आप अभी सुरक्षित हैं?"),
}
