"""Small retrieval layer for factual self-care content (NOT for emotional replies).

Baseline = TF-IDF over character n-grams (works across English / Hinglish spellings, needs no download).
Upgrade later to multilingual embeddings (e.g. a sentence-transformers model) + FAISS/Chroma; keep the same retrieve() interface.
Rule: if nothing is relevant enough, return [] so the companion says it has no verified material instead of guessing.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

KB_DIR = Path(__file__).resolve().parents[1] / "knowledge"
MIN_SCORE = 0.13   # tuned on eval/rag_cases.json; the margin is THIN (in-scope min 0.131, out-of-scope max 0.128). Swap to embeddings for a real margin.


@lru_cache(maxsize=1)
def _index():
    docs = [d for f in sorted(KB_DIR.glob("*.json")) for d in json.loads(f.read_text(encoding="utf-8"))]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
    mat = vec.fit_transform([f'{d["title"]} {d.get("keywords", "")} {d["text"]}' for d in docs]) if docs else None
    return docs, vec, mat


def retrieve(query: str, k: int = 3, min_score: float = MIN_SCORE) -> list[dict]:
    docs, vec, mat = _index()
    if not docs or not query.strip():
        return []
    sims = linear_kernel(vec.transform([query]), mat).ravel()
    order = sims.argsort()[::-1][:k]
    return [{**docs[i], "score": round(float(sims[i]), 3)} for i in order if sims[i] >= min_score]


def context_block(query: str, k: int = 2) -> str:
    """Text appended to the LLM system prompt. Empty if nothing relevant."""
    hits = retrieve(query, k)
    if not hits:
        return ""
    lines = [f'- [{h["title"]}] {h["text"]}' + ("" if str(h["source"]).startswith("UNVERIFIED") else f' (source: {h["source"]})') for h in hits]
    return ("\n\nREFERENCE PASSAGES (use only if they fit what the person said; paraphrase kindly in their language; do not read them out as a list; "
            "never invent a source; mention a source name only if shown in brackets after the passage):\n" + "\n".join(lines))
