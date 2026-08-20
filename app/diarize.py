"""Speaker diarization via pyannote.audio."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from app.transcribe import resolve_device

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# pyannote/speaker-diarization-3.1 internally pulls in extra gated repos
# (segmentation-3.0, and depending on the pyannote.audio version also
# speaker-diarization-community-1 for PLDA calibration). Each needs its
# license accepted separately on the SAME HF account as the token.
_GATED_URL_RE = re.compile(r"https://huggingface\.co/([\w.-]+/[\w.-]+)/resolve")


def _extract_gated_repo(exc: Exception) -> Optional[str]:
    match = _GATED_URL_RE.search(str(exc))
    return match.group(1) if match else None


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


class _ProgressBridge:
    """Adapts pyannote's hook(step_name, artifact, file, total, completed) callback
    into a simple (step_name, fraction_or_None) callback."""

    def __init__(self, callback: Optional[Callable[[str, Optional[float]], None]]):
        self.callback = callback

    def __call__(self, step_name, step_artifact=None, file=None, total=None, completed=None):
        if self.callback is None:
            return
        frac = None
        if total and completed is not None and total > 0:
            frac = min(completed / total, 1.0)
        self.callback(step_name, frac)


class Diarizer:
    def __init__(self, hf_token: str, device: str = "auto"):
        if not hf_token:
            raise ValueError(
                "Hugging Face token is required for pyannote.audio. "
                "Create one at https://huggingface.co/settings/tokens and accept the "
                "license at https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
        from huggingface_hub.errors import GatedRepoError
        from pyannote.audio import Pipeline
        import torch

        self.device = resolve_device(device)
        try:
            self.pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
        except GatedRepoError as exc:
            repo = _extract_gated_repo(exc)
            repo_line = f"https://huggingface.co/{repo}" if repo else "(см. ссылку в тексте ошибки ниже)"
            raise RuntimeError(
                "Hugging Face отклонил доступ к одной из моделей диаризации.\n"
                f"На аккаунте, которому принадлежит токен, откройте {repo_line} "
                "и нажмите «Agree and access repository».\n\n"
                "Модель speaker-diarization-3.1 использует внутри несколько "
                "gated-репозиториев — лицензию нужно принять на КАЖДОМ из них "
                "(обычно это pyannote/speaker-diarization-3.1, "
                "pyannote/segmentation-3.0 и pyannote/speaker-diarization-community-1). "
                "После этого запустите распознавание заново.\n\n"
                f"Исходная ошибка: {exc}"
            ) from exc
        if self.pipeline is None:
            raise RuntimeError(
                "Не удалось загрузить модель диаризации — проверьте токен и то, что "
                "приняты условия использования на страницах pyannote/speaker-diarization-3.1, "
                "pyannote/segmentation-3.0 и pyannote/speaker-diarization-community-1."
            )
        if self.device == "cuda":
            self.pipeline.to(torch.device("cuda"))

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        progress_callback: Optional[Callable[[str, Optional[float]], None]] = None,
    ) -> list[SpeakerTurn]:
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers:
                kwargs["min_speakers"] = min_speakers
            if max_speakers:
                kwargs["max_speakers"] = max_speakers

        # Load the audio ourselves and hand pyannote a waveform tensor instead
        # of a file path: its default file-loading path requires torchcodec,
        # which fails to install cleanly on Windows (DLL load errors). We
        # already normalize everything to a plain mono WAV via ffmpeg, so
        # soundfile can read it directly without any extra dependency.
        import soundfile as sf
        import torch

        data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # (channels, time)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        hook = _ProgressBridge(progress_callback)
        output = self.pipeline(audio_input, hook=hook, **kwargs)
        # Newer pyannote.audio versions return a DiarizeOutput wrapper
        # instead of a plain Annotation; older ones return the Annotation
        # directly (with .itertracks()).
        diarization = getattr(output, "speaker_diarization", output)

        turns: list[SpeakerTurn] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker))
        return turns
