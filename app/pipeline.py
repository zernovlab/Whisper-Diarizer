"""Orchestrates the full transcribe + diarize + merge pipeline.

Transcription (ctranslate2) and diarization (torch/pyannote) run as separate
subprocesses — see worker_transcribe.py / worker_diarize.py for why sharing
one process crashes on Windows when both use the GPU (conflicting bundled
cuDNN copies).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from app.audio_utils import convert_to_wav, get_duration_seconds
from app.diarize import SpeakerTurn
from app.merge import Chunk, build_chunks, relabel_speakers
from app.transcribe import Segment, Word

ProgressFn = Callable[[int, str], None]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    input_path: str
    model_size: str = "large-v3"
    device: str = "auto"
    language: str = "auto"
    hf_token: str = ""
    enable_diarization: bool = True
    speakers_mode: str = "auto"  # auto | exact | range
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


def _run_worker(module: str, config: dict, tmp_dir: str, on_progress_line: Callable[[str], None]) -> dict:
    name = module.split(".")[-1]
    cfg_path = Path(tmp_dir) / f"{name}_config.json"
    result_path = Path(tmp_dir) / f"{name}_result.json"
    stderr_path = Path(tmp_dir) / f"{name}_stderr.log"

    cfg_path.write_text(json.dumps(config), encoding="utf-8")

    with open(stderr_path, "w", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", module, str(cfg_path), str(result_path)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
            encoding="utf-8",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("PROGRESS:"):
                on_progress_line(line[len("PROGRESS:"):])
        proc.wait()

    if proc.returncode != 0:
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        if not stderr_text.strip():
            stderr_text = (
                "(процесс завершился без сообщения об ошибке — вероятно, аварийно "
                "упал на уровне ОС, например из-за нехватки видеопамяти)"
            )
        raise RuntimeError(f"{module} завершился с ошибкой (код {proc.returncode}):\n{stderr_text[-3000:]}")

    return json.loads(result_path.read_text(encoding="utf-8"))


def run_pipeline(config: PipelineConfig, progress: Optional[ProgressFn] = None) -> list[Chunk]:
    def report(pct: int, msg: str):
        if progress:
            progress(min(max(pct, 0), 100), msg)

    with tempfile.TemporaryDirectory(prefix="whisper_diarizer_") as tmp:
        wav_path = str(Path(tmp) / "audio.wav")

        report(0, "Конвертация аудио...")
        convert_to_wav(config.input_path, wav_path)
        duration = get_duration_seconds(wav_path)
        report(5, f"Аудио готово ({duration:.0f} сек)")

        # --- transcription (separate process: ctranslate2) ---
        report(6, f"Загрузка модели Whisper ({config.model_size})...")

        def on_transcribe_line(payload: str):
            frac = float(payload)
            report(10 + int(frac * 45), "Распознавание речи...")

        transcribe_result = _run_worker(
            "app.worker_transcribe",
            {
                "audio_path": wav_path,
                "model_size": config.model_size,
                "device": config.device,
                "language": None if config.language in (None, "auto", "") else config.language,
                "duration": duration,
            },
            tmp,
            on_transcribe_line,
        )
        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"], words=[Word(**w) for w in s["words"]])
            for s in transcribe_result["segments"]
        ]
        report(55, f"Распознавание завершено (язык: {transcribe_result['language']})")

        # --- diarization (separate process: torch/pyannote) ---
        if config.enable_diarization:
            report(56, "Загрузка модели диаризации...")

            num_speakers = config.num_speakers if config.speakers_mode == "exact" else None
            min_speakers = config.min_speakers if config.speakers_mode == "range" else None
            max_speakers = config.max_speakers if config.speakers_mode == "range" else None

            def on_diarize_line(payload: str):
                step_name, _, frac_str = payload.partition(":")
                frac = float(frac_str) if frac_str not in ("", "-") else 0.0
                report(60 + int(frac * 30), f"Диаризация: {step_name}...")

            diarize_result = _run_worker(
                "app.worker_diarize",
                {
                    "audio_path": wav_path,
                    "hf_token": config.hf_token,
                    "device": config.device,
                    "num_speakers": num_speakers,
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                },
                tmp,
                on_diarize_line,
            )
            turns = [SpeakerTurn(**t) for t in diarize_result["turns"]]
            report(90, f"Диаризация завершена ({len(set(t.speaker for t in turns))} спикеров)")
        else:
            turns = []
            report(90, "Диаризация отключена, пропущена")

        report(92, "Объединение транскрипта со спикерами...")
        chunks = build_chunks(segments, turns)
        chunks = relabel_speakers(chunks)
        report(100, "Готово")

        return chunks
