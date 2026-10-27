"""Shared helpers: config loading, path resolution, small IO utilities."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load YAML config. Order: explicit path > $DEPRESSION_CONFIG > configs/config.yaml."""
    path = path or os.environ.get("DEPRESSION_CONFIG") or ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def _abs(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def raw_path(cfg: dict, key: str) -> Path:
    return _abs(cfg["paths"]["raw_dir"]) / cfg["paths"][key]


def reports_dir(cfg: dict) -> Path:
    d = _abs(cfg["paths"]["reports_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def outputs_dir(cfg: dict, sub: str = "") -> Path:
    d = _abs(cfg["paths"]["outputs_dir"]) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean(o):
    """Make numpy / pandas scalars and keys JSON-safe."""
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and o != o:
        return None
    return o


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, indent=2, default=str)
