"""Dashboard server.  Run from the project root:  python -m app.server   ->  http://127.0.0.1:5000"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request

from . import companion, risk, safety, storage

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "models"
app = Flask(__name__, static_folder="static", static_url_path="")


def _json_file(name: str):
    p = MODEL_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.post("/api/chat")
def chat():
    d = request.get_json(silent=True) or {}
    msg = str(d.get("message", "")).strip()[:1000]
    if not msg:
        return jsonify(error="Type a message first."), 400
    out = companion.respond(msg, d.get("history") or [])
    if out["level"] == "crisis":
        out["helplines"] = safety.HELPLINES
    return jsonify(out)


@app.get("/api/checkins")
def get_checkins():
    rows = storage.entries()
    return jsonify(entries=rows, trend=storage.trend(rows), consent=storage.get_setting("consent_family_summary") == "1")


@app.post("/api/checkins")
def post_checkin():
    d = request.get_json(silent=True) or {}
    try:
        mood, lon = int(d["mood"]), int(d["loneliness"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Pick a mood and a loneliness level (1 to 5)."), 400
    if not (1 <= mood <= 5 and 1 <= lon <= 5):
        return jsonify(error="Values must be between 1 and 5."), 400
    storage.save_checkin(mood, lon)
    return get_checkins()


@app.post("/api/consent")
def consent():
    storage.set_setting("consent_family_summary", "1" if (request.get_json(silent=True) or {}).get("on") else "0")
    return jsonify(ok=True)


@app.delete("/api/data")
def delete_data():
    storage.delete_all()
    return jsonify(ok=True)


@app.get("/api/risk")
def get_risk():
    users, metrics = _json_file("demo_users.json"), _json_file("metrics.json")
    if not users:
        return jsonify(ready=False)
    return jsonify(ready=True, users=users["users"], method=users.get("method"), thresholds=users.get("thresholds"), metrics=metrics, support=risk.SUPPORT)


@app.get("/api/helplines")
def helplines():
    return jsonify(safety.HELPLINES)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
