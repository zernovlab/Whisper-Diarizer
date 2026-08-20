"""Persistent user settings stored in the home directory as JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".whisper_diarizer"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "hf_token": "",
    "model_size": "large-v3",
    "device": "auto",       # auto | cuda | cpu
    "language": "auto",     # auto | ru | en | ...
    "speakers_mode": "auto",  # auto | exact | range
    "num_speakers": 2,
    "min_speakers": 1,
    "max_speakers": 6,
    "last_output_dir": str(Path.home() / "Documents"),
}


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_CONFIG)
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    merged.update(data)
    return merged


def save_config(config: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
