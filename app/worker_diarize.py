"""Runs pyannote.audio diarization in an isolated subprocess.

See worker_transcribe.py for why this runs out-of-process.

Usage: python -m app.worker_diarize <config.json> <result.json>
Progress is reported on stdout as lines "PROGRESS:<step_name>:<0..1 or ->".
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict


def main():
    config_path, result_path = sys.argv[1], sys.argv[2]
    config = json.loads(open(config_path, encoding="utf-8").read())

    from app.diarize import Diarizer

    diarizer = Diarizer(hf_token=config["hf_token"], device=config["device"])

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
