"""Saathi: a listen-first companion. Not a therapist, never diagnoses, never advises on medicine.

Reply path:  safety check -> (crisis: fixed template, nothing else)
             -> optional LLM (only if ANTHROPIC_API_KEY is set) with guardrails + output check
             -> otherwise built-in listen-first templates (works offline, no key needed).
"""
from __future__ import annotations

import os
import random
import re

from . import rag, safety

HINGLISH_WORDS = {"hai", "hain", "nahi", "nahin", "mujhe", "main", "mai", "kya", "bahut", "bohot", "akela", "akeli", "mera", "meri",
                  "mere", "aap", "tum", "kuch", "kyun", "kyu", "raat", "neend", "ghar", "yaar", "thoda", "bas", "lag", "raha", "rahi",
                  "hoon", "hu", "toh", "to", "aur", "se", "ko", "ka", "ki", "ke", "accha", "theek", "dukhi", "udaas", "tension"}


def detect_lang(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    words = re.findall(r"[a-z']+", text.lower())
    return "hinglish" if sum(w in HINGLISH_WORDS for w in words) >= 2 else "en"


THEMES = {
    "lonely": [r"\blonel", r"\balone\b", r"\bno one to talk", r"\bmiss (my|home|family)", r"\bakela\b", r"\bakeli\b", r"अकेला", r"अकेली", r"ghar ki yaad"],
    "sleep": [r"\bsleep", r"\binsomnia\b", r"\bawake\b", r"\bneend\b", r"नींद"],
    "anxious": [r"\banxi", r"\bworr", r"\bpanic", r"\bnervous", r"\bstress", r"\btension\b", r"\bghabra", r"घबरा", r"चिंता"],
    "sad": [r"\bsad\b", r"\bdown\b", r"\bunhappy", r"\bcry", r"\blow\b", r"\budaas\b", r"\bdukhi\b", r"उदास", r"दुखी"],
    "tired": [r"\btired\b", r"\bexhaust", r"\bno energy", r"\bthak", r"थक"],
    "family": [r"\bfamily\b", r"\bparents?\b", r"\bson\b", r"\bdaughter\b", r"\bfriends?\b", r"\bghar\b", r"परिवार", r"दोस्त"],
    "positive": [r"\bgood\b", r"\bhappy\b", r"\bbetter\b", r"\bgreat\b", r"\bthanks?\b", r"\baccha\b", r"\btheek\b", r"अच्छा", r"शुक्रिया", r"धन्यवाद"],
    "greeting": [r"^(hi|hello|hey|namaste|hii+)\b", r"^नमस्ते"],
}

# One reflection + ONE gentle question. Keep short and plain.
REPLIES = {
    "greeting": {
        "en": ["Hello, I'm Saathi. I'm an AI companion, here to listen. How has your day been so far?"],
        "hinglish": ["Namaste, main Saathi hoon. Main ek AI companion hoon, aapki baat sunne ke liye. Aaj ka din kaisa raha?"],
        "hi": ["नमस्ते, मैं साथी हूँ। मैं एक AI साथी हूँ, आपकी बात सुनने के लिए। आज का दिन कैसा रहा?"]},
    "lonely": {
        "en": ["Feeling lonely can be heavy, and it takes courage to say it out loud. Thank you for telling me. When did you last talk with someone you feel close to?"],
        "hinglish": ["Akelapan bahut bhaari lag sakta hai, aur use bolna himmat ka kaam hai. Batane ke liye shukriya. Aakhri baar kab kisi apne se baat hui thi?"],
        "hi": ["अकेलापन बहुत भारी लग सकता है, और उसे कहना हिम्मत का काम है। बताने के लिए शुक्रिया। आखिरी बार किसी अपने से कब बात हुई थी?"]},
    "sleep": {
        "en": ["Poor sleep makes everything feel harder. I'm sorry it's been rough. What does your evening usually look like before bed?"],
        "hinglish": ["Neend poori na ho to sab kuch mushkil lagta hai. Sun kar bura laga. Sone se pehle aapki shaam aam taur par kaisi hoti hai?"],
        "hi": ["नींद पूरी न हो तो सब कुछ मुश्किल लगता है। सुनकर बुरा लगा। सोने से पहले आपकी शाम आम तौर पर कैसी होती है?"]},
    "anxious": {
        "en": ["That sounds stressful, and it makes sense that your mind feels busy. I'm here. What is the worry that keeps coming back the most?"],
        "hinglish": ["Yeh stressful lagta hai, aur dimaag ka busy rehna samajh aata hai. Main yahin hoon. Sabse zyada kaun si baat baar baar aa rahi hai?"],
        "hi": ["यह तनावपूर्ण लगता है, और मन का व्यस्त रहना समझ आता है। मैं यहीं हूँ। कौन सी बात सबसे ज़्यादा बार-बार आ रही है?"]},
    "sad": {
        "en": ["I'm sorry you're feeling low. You don't have to explain it perfectly, I'm happy to just listen. What has been weighing on you most?"],
        "hinglish": ["Sun kar bura laga ki aap low feel kar rahe ho. Perfectly samjhana zaroori nahi, main bas sunna chahta hoon. Sabse zyada kya pareshaan kar raha hai?"],
        "hi": ["सुनकर बुरा लगा कि आप उदास हैं। सब कुछ ठीक से समझाना ज़रूरी नहीं, मैं बस सुनना चाहता हूँ। सबसे ज़्यादा क्या परेशान कर रहा है?"]},
    "tired": {
        "en": ["It sounds like you're carrying a lot and running low on energy. That's real. Have you been able to rest a little today?"],
        "hinglish": ["Lagta hai aap bahut kuch utha rahe ho aur energy kam hai. Yeh real hai. Aaj thoda aaram kar paaye?"],
        "hi": ["लगता है आप बहुत कुछ उठा रहे हैं और ऊर्जा कम है। यह असली बात है। क्या आज थोड़ा आराम कर पाए?"]},
    "family": {
        "en": ["The people in our lives matter a lot. Tell me a little about them, and how things feel between you right now?"],
        "hinglish": ["Hamare apne logon ka bahut matlab hota hai. Unke baare mein thoda batao, aur abhi aapke beech kaisa lag raha hai?"],
        "hi": ["हमारे अपने लोग बहुत मायने रखते हैं। उनके बारे में थोड़ा बताइए, और अभी आपके बीच कैसा महसूस होता है?"]},
    "positive": {
        "en": ["I'm glad to hear that. Small good moments count. What made today a little better?"],
        "hinglish": ["Yeh sun kar accha laga. Chhote achhe pal bhi matter karte hain. Aaj kya cheez thodi better lagi?"],
        "hi": ["यह सुनकर अच्छा लगा। छोटे अच्छे पल भी मायने रखते हैं। आज क्या चीज़ थोड़ी बेहतर लगी?"]},
    "default": {
        "en": ["Thank you for sharing that. I'm listening. Can you tell me a bit more about how it feels for you?",
               "I hear you. Take your time. What part of this is on your mind the most right now?"],
        "hinglish": ["Batane ke liye shukriya. Main sun raha hoon. Thoda aur batao, aapko kaisa lag raha hai?",
                     "Main samajh raha hoon. Aaram se batao. Is mein se sabse zyada kya dimaag mein chal raha hai?"],
        "hi": ["बताने के लिए शुक्रिया। मैं सुन रहा हूँ। थोड़ा और बताइए, आपको कैसा लग रहा है?",
               "मैं समझ रहा हूँ। आराम से बताइए। इसमें से सबसे ज़्यादा क्या मन में चल रहा है?"]},
}

REPLIES["default"]["en"] += ["That sounds like a lot to hold. I'm listening, and there's no rush. What feels hardest about it right now?",
                             "Thanks for putting that into words. If you think of the last few days, when did it feel heaviest?"]
REPLIES["default"]["hinglish"] += ["Yeh kaafi bhaari lagta hai. Main sun raha hoon, koi jaldi nahi hai. Abhi sabse mushkil kya lag raha hai?",
                                   "Shabdon mein batane ke liye shukriya. Pichhle kuch dinon mein kab sabse zyada bhaari laga?"]
REPLIES["default"]["hi"] += ["यह काफ़ी भारी लगता है। मैं सुन रहा हूँ, कोई जल्दी नहीं है। अभी सबसे मुश्किल क्या लग रहा है?",
                             "शब्दों में बताने के लिए शुक्रिया। पिछले कुछ दिनों में कब सबसे ज़्यादा भारी लगा?"]

DISTRESS_EXTRA = {
    "en": " If these feelings get heavier, you can call Tele-MANAS any time (14416, free, 24x7). Talking to someone you trust can help too.",
    "hinglish": " Agar yeh feelings aur bhaari ho jaayein, to aap kabhi bhi Tele-MANAS (14416, free, 24x7) par baat kar sakte ho. Kisi apne se baat karna bhi madad karta hai.",
    "hi": " अगर ये भावनाएँ और भारी हो जाएँ, तो आप कभी भी Tele-MANAS (14416, मुफ़्त, 24x7) पर बात कर सकते हैं। किसी अपने से बात करना भी मदद करता है।",
}
MEDICAL_REPLY = {
    "en": "I can't diagnose anything or advise on medicines, and I don't want to guess. A doctor or counsellor can do that properly. I can help you note what you've been feeling so it's easier to tell them. Would you like that?",
    "hinglish": "Main koi diagnosis ya dawai ki salah nahi de sakta, aur andaza lagana bhi nahi chahta. Yeh kaam doctor ya counsellor sahi tarah se kar sakte hain. Main aapko yeh likhne mein madad kar sakta hoon ki aap kya feel kar rahe ho, taaki unhe batana aasaan ho. Chahoge?",
    "hi": "मैं कोई निदान या दवा की सलाह नहीं दे सकता, और अंदाज़ा भी नहीं लगाना चाहता। यह काम डॉक्टर या काउंसलर ठीक से कर सकते हैं। मैं आपको यह लिखने में मदद कर सकता हूँ कि आप क्या महसूस कर रहे हैं, ताकि उन्हें बताना आसान हो। क्या आप चाहेंगे?",
}
SAFE_CHECK_FOLLOWUP = {  # while in careful mode after a crisis message
    "en": "I'm still here with you. Please keep Tele-MANAS (14416) or 112 close, and try to be near someone you trust. Can you tell me if you are safe right now?",
    "hinglish": "Main abhi bhi yahin hoon. Please Tele-MANAS (14416) ya 112 paas rakhein, aur kisi apne ke paas rehne ki koshish karein. Kya aap abhi safe ho, bata sakte ho?",
    "hi": "मैं अभी भी यहीं हूँ। कृपया Tele-MANAS (14416) या 112 पास रखें, और किसी अपने के पास रहने की कोशिश करें। क्या आप अभी सुरक्षित हैं, बता सकते हैं?",
}

SYSTEM_PROMPT = """You are Saathi, a warm, skilled, supportive mental-wellbeing companion for people in India who feel low, stressed, anxious or lonely (students, young adults, elderly people living away from family). You talk the way a thoughtful, experienced counsellor would: calm, kind, never robotic.

IDENTITY AND HONESTY
- You are an AI companion, not a doctor or therapist. Say so briefly only if asked, or at the very start. Never claim to be human or a therapist.
- Never diagnose (do not say "you have depression/anxiety"). Never name, suggest or discuss medicines, supplements or doses. If asked, say a doctor or counsellor is the right person, and offer to help the person prepare what to tell them.

HOW TO REPLY (every turn)
1. Look at the whole conversation so far. Use what the person already told you (names, events, exam, family, job). Never ask something they already answered.
2. First, show you understood the specific feeling and situation in your own fresh words. Avoid empty phrases like "I understand", "it's okay", "I'm here for you", "thank you for sharing" more than once in a conversation.
3. Then do ONE useful thing, chosen to fit the moment:
   - If they are still telling the story: reflect it and ask ONE open question that goes deeper (what, when, how it affects them).
   - If they have shared enough: give ONE small, practical, realistic suggestion with a reason and a how-to (2 to 4 lines). Pick from the toolbox below. Ask if they want to try it.
4. Never repeat a sentence, question or suggestion you already used in this conversation. Vary your openings and structure.
5. Length: 3 to 6 short sentences. Plain words. No bullet lists, no headings, no lectures, no more than one question.

TOOLBOX (evidence-based, simple)
- Racing thoughts or panic: slow breathing (in 4, out 6, about 6 rounds) or grounding (name 5 things you see, 4 you can touch, 3 you hear).
- Low mood or no motivation: behavioural activation, which means one tiny pleasant or useful action (a 10-minute walk, a shower, tea with a friend) rather than waiting to feel like it.
- Negative thinking ("I'm useless", "I will fail"): gently ask for the evidence for and against the thought, then help find a fairer, kinder version. Do not argue or dismiss.
- Overwhelm or exam/work pressure: break it into the next one small step; a short plan for today only; scheduled breaks.
- Poor sleep: fixed wake time, screens away 30 to 60 minutes before bed, no tea/coffee late, write worries on paper before bed, get up if awake over 20 minutes.
- Loneliness: one small, low-pressure contact today (a short call or message to someone safe), a regular activity with other people (walk, prayer group, class, hobby club), and be kind to yourself about it.
- Anger or conflict with family: pause first, name the need underneath, plan one calm sentence to say.
- Grief or loss: allow feelings, no hurry, rituals and remembering are fine.
Always say these are small steps that may help, not cures.

WHEN TO SUGGEST A PROFESSIONAL
- If low mood, worry, or sleep trouble has lasted about two weeks or more, or is affecting study, work or relationships, gently suggest talking to a counsellor or doctor, and say it is a sign of taking care of yourself, not of weakness. In India, Tele-MANAS (14416) is free and 24x7.
- Suggest it once, kindly, not in every message.

SAFETY
- If the person mentions suicide, self-harm, wanting to die, or being in danger, do not give techniques. Say you are glad they told you, urge them to call Tele-MANAS 14416 or 112 now and to be near someone they trust, and ask if they are safe.
- Never promise secrecy. Never say things will definitely get better; say it can get lighter with support.

LANGUAGE AND CULTURE
- Reply in the language the person used: English, Hindi (Devanagari), or Hinglish (Hindi in Roman letters). Match their level of formality and keep it natural, not translated-sounding.
- Be aware of common Indian pressures (exams, parents' expectations, comparison, living away from home, joint-family dynamics, stigma about mental health) and respond without judging family or culture.
- Do not impose religion; if they bring it up, respect it."""

# output check for LLM text: reject anything that sounds like diagnosis or medicine advice
BAD_OUTPUT = re.compile(r"\b(you (have|suffer from) (depression|anxiety|a disorder)|diagnos(is|ed)|mg\b|dosage|antidepressant|sertraline|fluoxetine|escitalopram|prescrib)", re.I)


URLS = {"groq": "https://api.groq.com/openai/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "ollama": "http://localhost:11434/v1/chat/completions"}


def _llm_reply(history: list[dict], message: str) -> str | None:
    """Optional LLM. Env: LLM_PROVIDER (gemini|groq|openrouter|ollama|anthropic), LLM_API_KEY, LLM_MODEL. Any failure -> None -> templates."""
    prov = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if prov == "none" or (prov != "ollama" and not key):
        return None
    system = SYSTEM_PROMPT + rag.context_block(message)
    try:
        import requests
        turns = [("user" if m["role"] == "user" else "assistant", m["text"]) for m in history[-12:] if m.get("text")]
        turns.append(("user", message))
        while turns and turns[0][0] != "user":
            turns.pop(0)
        if prov == "gemini":
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model or 'gemini-2.5-flash'}:generateContent",
                              headers={"x-goog-api-key": key}, timeout=20,
                              json={"system_instruction": {"parts": [{"text": system}]},
                                    "contents": [{"role": "user" if a == "user" else "model", "parts": [{"text": t}]} for a, t in turns],
                                    "generationConfig": {"maxOutputTokens": 700, "temperature": 0.9}})
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        elif prov in URLS:
            r = requests.post(URLS[prov], headers={"Authorization": f"Bearer {key or 'none'}"}, timeout=30,
                              json={"model": model, "max_tokens": 1200, "temperature": 0.9,
                                    "messages": [{"role": "system", "content": system}] + [{"role": a, "content": t} for a, t in turns]})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
        else:  # anthropic
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=20,
                              headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                              json={"model": model or "claude-sonnet-5-5", "max_tokens": 400, "system": system,
                                    "messages": [{"role": a, "content": t} for a, t in turns]})
            r.raise_for_status()
            text = "".join(b.get("text", "") for b in r.json().get("content", [])).strip()
        if not text or len(text) > 1500 or BAD_OUTPUT.search(text):
            return None
        return text
    except Exception as e:
        print("LLM error:", e)  # shows in the server terminal, never to the user
        return None


def pick_theme(text: str) -> str:
    t = text.lower()
    for theme, pats in THEMES.items():
        if any(re.search(p, t) for p in pats):
            return theme
    return "default"


def respond(message: str, history: list[dict] | None = None) -> dict:
    """Return {reply, level, lang, source}. level in normal|distress|crisis|medical."""
    history = history or []
    lang = detect_lang(message)
    level = safety.classify(message)
    if level == "crisis":
        return {"reply": safety.CRISIS_REPLY[lang], "level": "crisis", "lang": lang, "source": "safety_template"}
    if safety.crisis_in_history(history):  # stay careful for the next turns; fixed text, no LLM
        return {"reply": SAFE_CHECK_FOLLOWUP[lang], "level": "crisis", "lang": lang, "source": "safety_template"}
    if level == "medical":
        return {"reply": MEDICAL_REPLY[lang], "level": "medical", "lang": lang, "source": "boundary_template"}
    llm = _llm_reply(history, message)
    if llm:
        return {"reply": llm + (DISTRESS_EXTRA[lang] if level == "distress" else ""), "level": level, "lang": lang, "source": "llm"}
    theme = "sad" if level == "distress" and pick_theme(message) == "default" else pick_theme(message)
    bots = [m.get("text", "") for m in history if m.get("role") == "bot"]
    fresh = lambda pool: [t for t in pool if not any(b.startswith(t) for b in bots)]
    pool = fresh(REPLIES[theme][lang]) or fresh(REPLIES["default"][lang]) or REPLIES["default"][lang]
    reply = random.choice(pool)
    if level == "distress":
        reply += DISTRESS_EXTRA[lang]
    return {"reply": reply, "level": level, "lang": lang, "source": "templates"}
