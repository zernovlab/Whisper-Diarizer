"""Speech-to-text via faster-whisper (CTranslate2 backend)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        # Avoid importing torch here: this runs in the transcription
        # subprocess, which is deliberately kept torch-free (see
        # worker_transcribe.py) to prevent the cuDNN conflict between
        # ctranslate2 and torch when both touch the GPU in one process.
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except ImportError:
        return "cpu"


def resolve_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


# ctranslate2 can construct a CUDA model successfully but only discover a
# missing/incompatible CUDA runtime library (cuBLAS, cuDNN) once it actually
# runs inference — e.g. on a machine with an NVIDIA GPU but no CUDA Toolkit
# installed, only the display driver. Detect that class of error and retry
# on CPU instead of crashing the whole run.
_CUDA_RUNTIME_ERROR_HINTS = ("cublas", "cudnn", "is not found or cannot be loaded")


def _looks_like_missing_cuda_runtime(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _CUDA_RUNTIME_ERROR_HINTS)


class Transcriber:
    def __init__(self, model_size: str, device: str = "auto", compute_type: Optional[str] = None):
        self.model_size = model_size
        self.device = resolve_device(device)
        self.compute_type = compute_type or resolve_compute_type(self.device)
        self.model = self._load_model(self.device, self.compute_type)

    def _load_model(self, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        return WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        duration_hint: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> tuple[list[Segment], str]:
        """Returns (segments, detected_language)."""
        try:
            return self._transcribe_once(audio_path, language, duration_hint, progress_callback)
        except RuntimeError as exc:
            if self.device != "cuda" or not _looks_like_missing_cuda_runtime(exc):
                raise
            # GPU is visible but its CUDA runtime libraries aren't usable
            # (missing CUDA Toolkit install) — fall back to CPU rather than
            # failing the whole transcription.
            self.device = "cpu"
            self.compute_type = resolve_compute_type("cpu")
            self.model = self._load_model(self.device, self.compute_type)
            return self._transcribe_once(audio_path, language, duration_hint, progress_callback)

    def _transcribe_once(
        self,
        audio_path: str,
        language: Optional[str],
        duration_hint: Optional[float],
        progress_callback: Optional[Callable[[float], None]],
    ) -> tuple[list[Segment], str]:
        lang = None if language in (None, "auto", "") else language

        raw_segments, info = self.model.transcribe(
            audio_path,
            language=lang,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        total = duration_hint or getattr(info, "duration", 0) or 1.0
        segments: list[Segment] = []
        for raw in raw_segments:
            words = [
                Word(start=w.start, end=w.end, text=w.word.strip(), probability=w.probability)
                for w in (raw.words or [])
            ]
            segments.append(Segment(start=raw.start, end=raw.end, text=raw.text.strip(), words=words))
            if progress_callback:
                progress_callback(min(raw.end / total, 1.0))

        return segments, info.language
