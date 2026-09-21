"""Reliable model downloads from the Hugging Face Hub.

Deliberately free of torch and ctranslate2 imports: it is used by both worker
subprocesses (see worker_transcribe.py for why they must stay separate), and
huggingface_hub itself is only imported lazily inside the functions.

Why this exists: a 20-minute run died with
"httpx.RemoteProtocolError: Server disconnected without sending a response"
from a plain Hub metadata request. faster-whisper makes that request on every
start — even when the model is already on disk — so one flaky moment on the
network killed the whole transcription, and the multi-GB first download showed
no progress at all.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

StatusFn = Callable[[str], None]
DownloadFn = Callable[[int, int], None]  # (bytes_done, bytes_total)

# Aliases from faster-whisper's own registry win when it is importable; these
# only fill in names an older installed version does not know yet.
_FALLBACK_MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# The same file set faster-whisper's own download_model() fetches.
_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]

_TRANSIENT_ERROR_NAMES = {
    "RemoteProtocolError", "ConnectError", "ConnectTimeout", "ReadTimeout",
    "WriteTimeout", "PoolTimeout", "ReadError", "WriteError", "CloseError",
    "NetworkError", "TimeoutException", "ProxyError", "SSLError",
    "ConnectionError", "ChunkedEncodingError", "IncompleteRead",
    "ConnectionResetError", "ConnectionAbortedError", "TimeoutError",
    # huggingface_hub wraps a failed connection with no usable cache in this:
    "LocalEntryNotFoundError",
}
_TRANSIENT_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


def configure_hub_env() -> None:
    """Call first thing in a worker, before huggingface_hub is imported —
    its timeouts are read from the environment at import time."""
    # Defaults are 10 s, too short for a slow or filtered connection.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def is_transient_network_error(exc: BaseException) -> bool:
    """True for connection drops, timeouts and 5xx/429 — worth retrying.
    False for anything a retry cannot fix (gated repo, 404, bad token)."""
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        response = getattr(cur, "response", None)
        status = getattr(response, "status_code", None)
        if status is not None:
            return status in _TRANSIENT_HTTP_STATUS
        if type(cur).__name__ in _TRANSIENT_ERROR_NAMES:
            return True
        if isinstance(cur, (ConnectionError, TimeoutError)):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def friendly_network_error(what: str) -> str:
    return (
        f"Не удалось загрузить {what} с huggingface.co — соединение обрывается. "
        "Проверьте интернет, VPN/прокси и антивирус (он иногда рвёт большие "
        "загрузки) и запустите обработку ещё раз: уже скачанная часть не "
        "пропадёт, загрузка продолжится с места обрыва."
    )


def call_with_network_retries(
    fn: Callable[[], object],
    *,
    what: str,
    on_status: Optional[StatusFn] = None,
    attempts: int = 5,
    delays: tuple[float, ...] = (2, 5, 10, 20),
):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if not is_transient_network_error(exc):
                raise
            if attempt == attempts:
                raise RuntimeError(friendly_network_error(what)) from exc
            delay = delays[min(attempt - 1, len(delays) - 1)]
            if on_status:
                on_status(
                    f"Обрыв соединения с huggingface.co ({what}), повтор через "
                    f"{delay:g} с — попытка {attempt + 1} из {attempts}..."
                )
            time.sleep(delay)


def _model_dir_complete(path: str) -> bool:
    """A snapshot folder appears as soon as the small files land, long before
    model.bin does — so 'the folder exists' does not mean 'the model is here'."""
    p = Path(path)
    if not all((p / name).exists() for name in ("config.json", "model.bin", "tokenizer.json")):
        return False
    return any(p.glob("vocabulary.*"))


def _repo_id_for(model_size: str) -> Optional[str]:
    if "/" in model_size:
        return model_size
    models = dict(_FALLBACK_MODELS)
    try:
        from faster_whisper.utils import _MODELS

        models.update(_MODELS)
    except ImportError:
        pass
    return models.get(model_size)


class _DownloadProgress:
    """Turns huggingface_hub's tqdm byte bars into on_download(done, total).

    The tqdm subclass must stay *enabled* (tqdm stops counting when disabled)
    but write nowhere: the worker's stderr is captured to show error
    messages, and progress bars would bury the actual error line."""

    def __init__(self, on_download: DownloadFn):
        self._callback = on_download
        self._bars: dict[int, list[int]] = {}
        self._last_sent = 0.0

    def reset(self) -> None:
        # Every retry creates fresh bars; stale ones would inflate the sums.
        self._bars.clear()

    def _totals(self) -> tuple[int, int]:
        return (
            sum(b[0] for b in self._bars.values()),
            sum(b[1] for b in self._bars.values()),
        )

    def finish(self) -> None:
        # Updates are throttled, so the last visible value lags behind.
        _, total = self._totals()
        if total:
            self._callback(total, total)

    def tqdm_class(self):
        from tqdm.auto import tqdm

        outer = self

        class _Sink:
            def write(self, _text):
                pass

            def flush(self):
                pass

        class ProgressTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = False
                kwargs["file"] = _Sink()
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                result = super().update(n)
                # "Reconstructing..." is xet's second pass over bytes already
                # counted by "Downloading bytes"; ignore it.
                if self.unit == "B" and not str(self.desc or "").startswith("Reconstruct"):
                    outer._bars[id(self)] = [int(self.n or 0), int(self.total or 0)]
                    now = time.monotonic()
                    if now - outer._last_sent >= 0.5:
                        outer._last_sent = now
                        outer._callback(*outer._totals())
                return result

        return ProgressTqdm


def resolve_whisper_model(
    model_size: str,
    on_status: Optional[StatusFn] = None,
    on_download: Optional[DownloadFn] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Return a local folder with the model, downloading it if needed.

    Cache first, with no network at all: an already-downloaded model then works
    offline and can no longer be broken by a flaky Hub request. Only when the
    files are missing does it download, with retries and progress."""
    if os.path.isdir(model_size):
        return model_size
    repo_id = _repo_id_for(model_size)
    if repo_id is None:
        return model_size  # unknown alias: let faster-whisper report it

    from huggingface_hub import snapshot_download

    common: dict = {"allow_patterns": _ALLOW_PATTERNS}
    if cache_dir:
        common["cache_dir"] = cache_dir

    try:
        cached = snapshot_download(repo_id, local_files_only=True, **common)
        if _model_dir_complete(cached):
            return cached
    except Exception:  # noqa: BLE001 — not cached yet (or cache unreadable)
        pass

    if on_status:
        on_status(f"Скачиваю модель Whisper ({model_size}) — это разовая загрузка...")
    download_kwargs = dict(common, etag_timeout=30)
    progress = _DownloadProgress(on_download) if on_download else None
    if progress:
        download_kwargs["tqdm_class"] = progress.tqdm_class()

    def _download() -> str:
        if progress:
            progress.reset()
        return snapshot_download(repo_id, **download_kwargs)

    path = call_with_network_retries(
        _download,
        what=f"модель Whisper «{model_size}»",
        on_status=on_status,
    )
    if progress:
        progress.finish()
    return path
