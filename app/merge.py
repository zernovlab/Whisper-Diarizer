"""Combine whisper segments (with word timestamps) and pyannote speaker turns
into speaker-labeled, semantically-grouped chunks with timestamps.

A new chunk starts whenever the speaker changes, a natural pause is detected,
or a monologue chunk has grown long enough that it should be cut at the next
sentence boundary. This keeps timestamps aligned to meaningful units of speech
(phrases/sentences) rather than fixed-size windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.diarize import SpeakerTurn
from app.transcribe import Segment

PAUSE_BREAK_SECONDS = 1.0
MAX_CHUNK_SECONDS = 30.0
MAX_CHUNK_CHARS = 400
SENTENCE_ENDINGS = (".", "!", "?", "…", ".\"", "!\"", "?\"", ".»", "!»", "?»")

UNKNOWN_SPEAKER = "SPEAKER_?"


@dataclass
class _FlatWord:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    start: float
    end: float
    speaker: str
    text: str


def _flatten_words(segments: list[Segment]) -> list[_FlatWord]:
    words: list[_FlatWord] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                if w.text:
                    words.append(_FlatWord(start=w.start, end=w.end, text=w.text))
        elif seg.text:
            words.append(_FlatWord(start=seg.start, end=seg.end, text=seg.text))
    return words


def _assign_speakers(words: list[_FlatWord], turns: list[SpeakerTurn]) -> list[str]:
    if not turns:
        return [UNKNOWN_SPEAKER] * len(words)

    turns = sorted(turns, key=lambda t: t.start)
    speakers: list[str] = []
    ptr = 0
    n = len(turns)

    for w in words:
        mid = (w.start + w.end) / 2
        while ptr < n - 1 and turns[ptr].end < w.start:
            ptr += 1

        best_speaker = None
        best_overlap = 0.0
        best_distance = float("inf")
        # scan a small neighborhood around ptr for the turn with maximum overlap
        for i in range(max(0, ptr - 1), min(n, ptr + 3)):
            t = turns[i]
            overlap = min(w.end, t.end) - max(w.start, t.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = t.speaker
            if best_overlap <= 0:
                distance = max(t.start - mid, mid - t.end, 0)
                if distance < best_distance:
                    best_distance = distance
                    if best_speaker is None:
                        best_speaker = t.speaker

        speakers.append(best_speaker or UNKNOWN_SPEAKER)

    return speakers


def build_chunks(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Chunk]:
    words = _flatten_words(segments)
    if not words:
        return []
    speakers = _assign_speakers(words, turns)

    chunks: list[Chunk] = []
    cur_words: list[str] = []
    cur_start = words[0].start
    cur_end = words[0].end
    cur_speaker = speakers[0]
    prev_word_end = words[0].start

    def flush():
        if cur_words:
            text = " ".join(cur_words).strip()
            if text:
                chunks.append(Chunk(start=cur_start, end=cur_end, speaker=cur_speaker, text=text))

    for i, w in enumerate(words):
        speaker = speakers[i]
        gap = w.start - prev_word_end
        duration_so_far = prev_word_end - cur_start
        prev_ends_sentence = bool(cur_words) and cur_words[-1].endswith(SENTENCE_ENDINGS)

        should_break = bool(cur_words) and (
            speaker != cur_speaker
            or gap > PAUSE_BREAK_SECONDS
            or (duration_so_far >= MAX_CHUNK_SECONDS and prev_ends_sentence)
            or (sum(len(t) for t in cur_words) >= MAX_CHUNK_CHARS and prev_ends_sentence)
        )

        if should_break:
            flush()
            cur_words = []
            cur_start = w.start
            cur_speaker = speaker

        cur_words.append(w.text)
        cur_end = w.end
        prev_word_end = w.end

    flush()
    return chunks


def relabel_speakers(chunks: list[Chunk]) -> list[Chunk]:
    """Rename raw pyannote labels (SPEAKER_00...) to human-friendly 'Speaker 1...'
    in order of first appearance."""
    mapping: dict[str, str] = {}
    for c in chunks:
        if c.speaker not in mapping:
            mapping[c.speaker] = f"Спикер {len(mapping) + 1}"
    return [Chunk(start=c.start, end=c.end, speaker=mapping[c.speaker], text=c.text) for c in chunks]
