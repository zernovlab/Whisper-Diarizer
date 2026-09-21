"""Speech-to-text via faster-whisper (CTranslate2 backend)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
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


# A token that continues the previous word rather than starting a new one.
# faster-whisper marks a new word with a leading space (" какие") and a
# continuation by its absence ("-то"). Only these joiners are treated that way:
# in Chinese or Japanese *every* token lacks a leading space, and gluing those
# together would turn a whole sentence into one "word".
_CONTINUATION_STARTS = ("-", "'", "’")


def _merge_continuations(raw_words) -> list[Word]:
    """Word tokens with "-то", "-нибудь", "'s" glued to the word before them.
    Without this the words are joined with spaces later on ("какие -то")."""
    words: list[Word] = []
    for w in raw_words:
        text = w.word.strip()
        if not text:
            continue
        is_continuation = not w.word[:1].isspace() and text[0] in _CONTINUATION_STARTS
        if words and is_continuation:
            previous = words[-1]
            previous.text += text
            previous.end = w.end
            previous.probability = min(previous.probability, w.probability)
        else:
            words.append(Word(start=w.start, end=w.end, text=text, probability=w.probability))
    return words


_dll_dir_handles: list = []  # os.add_dll_directory() handles must stay referenced


def _expose_torch_cuda_libs() -> None:
    """Make torch's bundled cuBLAS visible to ctranslate2 on Windows.

    ctranslate2 ships cuDNN but not cuBLAS. Up to 4.8.1 it imported torch as a
    side effect, whose import registers torch/lib as a DLL directory — so GPU
    transcription worked without any system-wide CUDA Toolkit, by accident.
    4.8.2 stopped importing torch, and the GPU silently became unusable
    ("Library cublas64_12.dll is not found"). Do it deliberately instead:
    find the torch folder WITHOUT importing torch (no CUDA context, no second
    copy of the runtime in this process) and add its lib dir to the search path.

    ctranslate2 is imported first on purpose: its __init__ preloads its own
    cudnn64_9.dll by full path, and a DLL already loaded under that name is
    reused, so torch's same-named copy cannot shadow it."""
    if os.name != "nt":
        return
    import importlib.util

    import ctranslate2  # noqa: F401 — must load (and preload its cuDNN) first

    spec = importlib.util.find_spec("torch")  # locates the package without importing it
    if spec is None or not spec.submodule_search_locations:
        return
    lib_dir = Path(next(iter(spec.submodule_search_locations))) / "lib"
    if not (lib_dir / "cublas64_12.dll").exists():
        return
    # add_dll_directory alone is not enough: ctranslate2.dll asks for cuBLAS
    # with a plain LoadLibrary("cublas64_12.dll"), which ignores directories
    # added that way. What worked before was torch *pre-loading* these DLLs by
    # full path — a module already loaded under a name is reused. Do the same
    # (cublasLt first: cublas depends on it).
    import ctypes

    _dll_dir_handles.append(os.add_dll_directory(str(lib_dir)))
    for name in ("cublasLt64_12.dll", "cublas64_12.dll"):
        _dll_dir_handles.append(ctypes.WinDLL(str(lib_dir / name)))


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
    def __init__(
        self,
        model_size: str,
        device: str = "auto",
        compute_type: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_download: Optional[Callable[[int, int], None]] = None,
    ):
        from app.hub_utils import resolve_whisper_model

        self.model_size = model_size
        self.on_status = on_status
        self.device = resolve_device(device)
        if self.device == "cuda":
            _expose_torch_cuda_libs()
        self.compute_type = compute_type or resolve_compute_type(self.device)
        # Resolved to a local folder once, up front: WhisperModel(<size name>)
        # asks the Hub on every construction — including the CPU-fallback
        # reload below — and one dropped connection there kills the run.
        self.model_path = resolve_whisper_model(model_size, on_status=on_status, on_download=on_download)
        self.model = self._load_model(self.device, self.compute_type)

    def _load_model(self, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        return WhisperModel(self.model_path, device=device, compute_type=compute_type)

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
            if self.on_status:
                self.on_status(
                    "Видеокарта недоступна для распознавания (не найдены библиотеки CUDA) — "
                    "перезапускаю на процессоре, это будет заметно медленнее."
                )
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
            segments.append(
                Segment(
                    start=raw.start,
                    end=raw.end,
                    text=raw.text.strip(),
                    words=_merge_continuations(raw.words or []),
                )
            )
            if progress_callback:
                progress_callback(min(raw.end / total, 1.0))

        return segments, info.language
