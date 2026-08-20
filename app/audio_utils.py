"""ffmpeg-backed helpers for normalizing input media to 16kHz mono WAV."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> None:
    """Extract/convert any audio or video file into mono PCM16 WAV via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"ffmpeg failed converting {input_path}:\n{result.stderr[-2000:]}")


def get_duration_seconds(input_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {input_path}:\n{result.stderr[-2000:]}")
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def check_ffmpeg_available() -> bool:
    for exe in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([exe, "-version"], capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return False
    return True
