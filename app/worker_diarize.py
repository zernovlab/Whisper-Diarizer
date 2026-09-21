"""Runs pyannote.audio diarization in an isolated subprocess.

See worker_transcribe.py for why this runs out-of-process.

Usage: python -m app.worker_diarize <config.json> <result.json>
Lines on stdout (parsed by pipeline.py):
  PROGRESS:<step_name>:<0..1 or ->   diarization progress
  STATUS:<text>                      a message to show the user (e.g. a retry)
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict


def main():
    # Hub timeouts are read at import time — must precede any huggingface_hub import.
    from app.hub_utils import configure_hub_env

    configure_hub_env()

    config_path, result_path = sys.argv[1], sys.argv[2]
    config = json.loads(open(config_path, encoding="utf-8").read())

    from app.diarize import Diarizer

    def on_status(text: str):
        print(f"STATUS:{text}", flush=True)

    diarizer = Diarizer(hf_token=config["hf_token"], device=config["device"], on_status=on_status)

    def on_progress(step_name: str, frac):
        frac_str = f"{frac}" if frac is not None else "-"
        print(f"PROGRESS:{step_name}:{frac_str}", flush=True)

    turns = diarizer.diarize(
        config["audio_path"],
        num_speakers=config.get("num_speakers"),
        min_speakers=config.get("min_speakers"),
        max_speakers=config.get("max_speakers"),
        progress_callback=on_progress,
    )

    result = {"turns": [asdict(t) for t in turns]}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
