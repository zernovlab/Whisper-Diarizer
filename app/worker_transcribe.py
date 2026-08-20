"""Runs faster-whisper transcription in an isolated subprocess.

ctranslate2 (used by faster-whisper) and torch (used by pyannote for
diarization) each bundle their own copy of cuDNN. When both are active on
the GPU in the same process, they crash with a native access violation in
cudnn64_9.dll (STATUS_STACK_BUFFER_OVERRUN) — not a catchable Python
exception, the whole process just dies. Running transcription and
diarization as separate subprocesses avoids the conflict entirely.

Usage: python -m app.worker_transcribe <config.json> <result.json>
Progress is reported on stdout as lines "PROGRESS:<0..1>".
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict


def main():
    config_path, result_path = sys.argv[1], sys.argv[2]
    config = json.loads(open(config_path, encoding="utf-8").read())

    from app.transcribe import Transcriber

    transcriber = Transcriber(model_size=config["model_size"], device=config["device"])

    def on_progress(frac: float):
        print(f"PROGRESS:{frac}", flush=True)

    segments, language = transcriber.transcribe(
        config["audio_path"],
        language=config.get("language"),
        duration_hint=config.get("duration"),
        progress_callback=on_progress,
    )

    result = {
        "language": language,
        "segments": [asdict(s) for s in segments],
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
