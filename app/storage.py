"""Tiny local store for the daily check-in (SQLite). Chat text is NOT stored on the server.
Prototype note: the file is not encrypted; encrypt at rest before any real deployment."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB = Path(__file__).parent / "instance" / "checkins.db"


def _conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS checkin(day TEXT PRIMARY KEY, mood INT, loneliness INT)")
    c.execute("CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT)")
    return c


def save_checkin(mood: int, loneliness: int, day: str | None = None) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO checkin VALUES (?,?,?)", (day or date.today().isoformat(), mood, loneliness))


def entries(days: int = 21) -> list[dict]:
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    with _conn() as c:
        rows = c.execute("SELECT day, mood, loneliness FROM checkin WHERE day >= ? ORDER BY day", (since,)).fetchall()
    return [{"day": d, "mood": m, "loneliness": l} for d, m, l in rows]


def get_setting(k: str, default: str = "0") -> str:
    with _conn() as c:
        r = c.execute("SELECT v FROM settings WHERE k=?", (k,)).fetchone()
    return r[0] if r else default


def set_setting(k: str, v: str) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (k, v))


def delete_all() -> None:
    with _conn() as c:
        c.execute("DELETE FROM checkin")
        c.execute("DELETE FROM settings")


def trend(rows: list[dict], today: date | None = None) -> dict:
    """Last 7 days versus the person's own earlier 14 days (own baseline, not a population norm)."""
    today = today or date.today()
    cut = (today - timedelta(days=6)).isoformat()
    recent = [r for r in rows if r["day"] >= cut]
    earlier = [r for r in rows if r["day"] < cut]
    out = {"recent_n": len(recent), "earlier_n": len(earlier), "nudge": None}
    if len(recent) >= 3 and len(earlier) >= 3:
        avg = lambda rs, k: sum(r[k] for r in rs) / len(rs)
        dm = avg(recent, "mood") - avg(earlier, "mood")
        dl = avg(recent, "loneliness") - avg(earlier, "loneliness")
        out.update(mood_change=round(dm, 2), loneliness_change=round(dl, 2))
        if dm <= -0.8 or dl >= 0.8:
            out["nudge"] = "down"
        elif dm >= 0.5 and dl <= 0:
            out["nudge"] = "up"
    return out
