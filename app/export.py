"""Exporters for the merged transcript: TXT, SRT (utterances), SRT (subtitles), JSON, DOCX."""
from __future__ import annotations

import json
from pathlib import Path

from app.merge import Chunk


def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_srt_time(seconds: float) -> str:
    # Whole milliseconds first, then split: rounding only the fractional part
    # could yield ",1000" (e.g. 2.9996 s) — an invalid time code.
    total_ms = int(round(max(0.0, seconds) * 1000))
    h, rest = divmod(total_ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(path: str, cues: list[tuple[float, float, str]]) -> None:
    """UTF-8 *with BOM* and CRLF line breaks, on purpose: Premiere Pro reads a
    BOM-less UTF-8 SRT wrongly (Cyrillic turns into garbage) and expects
    Windows line endings even on a Mac. Other editors and players
    (DaVinci Resolve, VLC, ffmpeg) accept the BOM without complaint."""
    blocks = []
    for i, (start, end, text) in enumerate(cues, start=1):
        body = "\r\n".join(text.split("\n"))
        blocks.append(f"{i}\r\n{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}\r\n{body}\r\n")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("\r\n".join(blocks))


def to_txt(chunks: list[Chunk], path: str) -> None:
    lines = [f"[{_fmt_hms(c.start)} - {_fmt_hms(c.end)}] {c.speaker}: {c.text}" for c in chunks]
    Path(path).write_text("\n\n".join(lines), encoding="utf-8")


def to_srt(chunks: list[Chunk], path: str) -> None:
    """One cue per utterance, speaker name in front. Meant for reading the
    transcript on a timeline, not as on-screen subtitles — see
    to_subtitles_srt for that."""
    _write_srt(path, [(c.start, c.end, f"{c.speaker}: {c.text}") for c in chunks])


def to_subtitles_srt(chunks: list[Chunk], path: str, include_speakers: bool = False) -> None:
    """Real subtitles: short cues (two lines of ~42 characters at most),
    ready to import into Premiere Pro as a caption track."""
    from app.subtitles import build_subtitle_cues

    cues = build_subtitle_cues(chunks, include_speakers=include_speakers)
    _write_srt(path, [(c.start, c.end, c.text) for c in cues])


def to_json(chunks: list[Chunk], path: str) -> None:
    data = [
        {"start": round(c.start, 2), "end": round(c.end, 2), "speaker": c.speaker, "text": c.text}
        for c in chunks
    ]
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def to_premiere_json(chunks: list[Chunk], path: str) -> None:
    """Word-timed transcript for Premiere Pro's "Import Static Transcript"
    (see premiere.py). Pure ASCII (\\uXXXX escapes) so no reader can
    misdetect the encoding of the Cyrillic text."""
    from app.premiere import build_premiere_transcript

    data = build_premiere_transcript(chunks)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)


def to_docx(chunks: list[Chunk], path: str) -> None:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.styles["Normal"].font.size = Pt(11)
    for c in chunks:
        p = doc.add_paragraph()
        run = p.add_run(f"[{_fmt_hms(c.start)} - {_fmt_hms(c.end)}] {c.speaker}: ")
        run.bold = True
        p.add_run(c.text)
    doc.save(path)
