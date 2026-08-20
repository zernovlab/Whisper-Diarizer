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
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


class Transcriber:
    def __init__(self, model_size: str, device: str = "auto", compute_type: Optional[str] = None):
        from faster_whisper import WhisperModel

        self.device = resolve_device(device)
        self.compute_type = compute_type or resolve_compute_type(self.device)
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        duration_hint: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> tuple[list[Segment], str]:
        """Returns (segments, detected_language)."""
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
