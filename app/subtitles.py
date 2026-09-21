"""Turn transcript chunks into on-screen subtitle cues.

A transcript chunk is a whole utterance — up to 30 s and 400 characters, which
is right for reading but far too much for a subtitle track: the text would
overflow the frame. Subtitles want short cues: at most two lines of about 42
characters, on screen for a few seconds. This module re-cuts each chunk at real
word boundaries (the word timings are kept on Chunk for exactly this):

  1. at sentence ends;
  2. inside a sentence that is still too long, into balanced pieces, preferring
     to break after a comma;
  3. tiny neighbouring pieces ("Да.", "Ну, хорошо.") are merged back so a
     one-word cue does not flash by.

A cue never mixes two speakers, so no "- " dialogue dashes are needed.
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, replace
from typing import Optional

from app.merge import SENTENCE_ENDINGS, Chunk, TimedWord

MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CUE_SECONDS = 6.0
MIN_CUE_SECONDS = 1.0
MIN_GAP_SECONDS = 0.04     # keep consecutive cues from touching
SMALL_CUE_CHARS = 15       # shorter than this is a candidate for merging
MERGE_GAP_SECONDS = 0.6    # ...but only across a pause no longer than this

_CAPACITY = MAX_CHARS_PER_LINE * MAX_LINES
_COMMA_ENDINGS = (",", ";", ":", "—", "–")
_COMMA_BONUS_CHARS = 8     # how much closer to the ideal cut a comma is worth
# "т.е.", "и.о.", "г." and the like end in a dot without ending the sentence.
_ABBREVIATION = re.compile(r"^(?:\w\.){1,3}$|^(?:др|пр|см|рис|ул|им|гг)\.$", re.IGNORECASE)


@dataclass
class SubtitleCue:
    start: float
    end: float
    lines: list[str]
    speaker: str

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _text(words: list[TimedWord]) -> str:
    return " ".join(w.text for w in words)


def _duration(words: list[TimedWord]) -> float:
    return words[-1].end - words[0].start


def wrap_lines(text: str) -> Optional[list[str]]:
    """Split text into one or two lines of at most MAX_CHARS_PER_LINE, as
    balanced as possible; None if it cannot be done (a word is too long)."""
    text = " ".join(text.split())
    if len(text) <= MAX_CHARS_PER_LINE:
        return [text]
    best: Optional[tuple[int, list[str]]] = None
    for match in re.finditer(" ", text):
        i = match.start()
        top, bottom = text[:i], text[i + 1:]
        if len(top) > MAX_CHARS_PER_LINE or len(bottom) > MAX_CHARS_PER_LINE:
            continue
        score = abs(len(top) - len(bottom))
        if len(top) > len(bottom):
            score += 6  # convention: the shorter line goes on top
        if top.endswith(_COMMA_ENDINGS + (".", "!", "?", "…")):
            score -= 10  # break where the speaker would pause
        if best is None or score < best[0]:
            best = (score, [top, bottom])
    return best[1] if best else None


def _hard_wrap(text: str) -> list[str]:
    """Last resort for a single word longer than a line."""
    lines = textwrap.wrap(text, MAX_CHARS_PER_LINE, break_long_words=True) or [text]
    if len(lines) > MAX_LINES:
        lines = lines[: MAX_LINES - 1] + [" ".join(lines[MAX_LINES - 1:])]
    return lines


def _fits(words: list[TimedWord]) -> bool:
    if len(words) == 1:
        return True  # a lone word must always be emittable, however long
    text = _text(words)
    if len(text) > _CAPACITY or _duration(words) > MAX_CUE_SECONDS:
        return False
    return wrap_lines(text) is not None


def ends_sentence(word: str) -> bool:
    return word.endswith(SENTENCE_ENDINGS) and not _ABBREVIATION.match(word.rstrip('"»)'))


def _split_sentences(words: list[TimedWord]) -> list[list[TimedWord]]:
    sentences: list[list[TimedWord]] = []
    current: list[TimedWord] = []
    for w in words:
        current.append(w)
        if ends_sentence(w.text):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def _greedy_pieces(words: list[TimedWord]) -> list[list[TimedWord]]:
    pieces: list[list[TimedWord]] = []
    current: list[TimedWord] = []
    for w in words:
        if current and not _fits(current + [w]):
            pieces.append(current)
            current = [w]
        else:
            current.append(w)
    pieces.append(current)
    return pieces


def _split_sentence(words: list[TimedWord]) -> list[list[TimedWord]]:
    """One sentence -> pieces that each fit a cue, cut into even shares
    rather than "as much as fits, then a scrap" (a greedy split leaves a
    one-word tail)."""
    if _fits(words):
        return [words]
    greedy = _greedy_pieces(words)
    n = len(greedy)
    if n == 1:
        return greedy

    prefix: list[int] = []
    running = 0
    for w in words:
        running += len(w.text) + 1
        prefix.append(running)
    total = running

    cuts: list[int] = []  # a cut at j means a new piece starts at words[j]
    low = 1
    for k in range(1, n):
        target = total * k / n
        high = len(words) - (n - k)  # leave a word for every remaining piece
        if low > high:
            return greedy
        best = min(
            range(low, high + 1),
            key=lambda j: abs(prefix[j - 1] - target)
            - (_COMMA_BONUS_CHARS if words[j - 1].text.endswith(_COMMA_ENDINGS) else 0),
        )
        cuts.append(best)
        low = best + 1

    bounds = [0] + cuts + [len(words)]
    balanced = [words[a:b] for a, b in zip(bounds, bounds[1:])]
    return balanced if all(_fits(p) for p in balanced) else greedy


def _merge_small(pieces: list[list[TimedWord]]) -> list[list[TimedWord]]:
    merged: list[list[TimedWord]] = []
    for piece in pieces:
        if merged:
            prev = merged[-1]
            small = (
                len(_text(prev)) < SMALL_CUE_CHARS
                or len(_text(piece)) < SMALL_CUE_CHARS
                or _duration(prev) < MIN_CUE_SECONDS
                or _duration(piece) < MIN_CUE_SECONDS
            )
            if small and piece[0].start - prev[-1].end <= MERGE_GAP_SECONDS and _fits(prev + piece):
                merged[-1] = prev + piece
                continue
        merged.append(piece)
    return merged


def words_from_text(chunk: Chunk) -> list[TimedWord]:
    """Fallback for a chunk without word timings: spread the chunk's time
    over its words in proportion to their length."""
    tokens = chunk.text.split()
    if not tokens:
        return []
    weights = [len(t) + 1 for t in tokens]
    span = max(chunk.end - chunk.start, 0.001)
    words: list[TimedWord] = []
    at = chunk.start
    for token, weight in zip(tokens, weights):
        step = span * weight / sum(weights)
        words.append(TimedWord(start=at, end=at + step, text=token))
        at += step
    return words


def _fix_timing(cues: list[SubtitleCue]) -> list[SubtitleCue]:
    cues.sort(key=lambda c: c.start)
    for i, cue in enumerate(cues):
        following = cues[i + 1] if i + 1 < len(cues) else None
        limit = following.start - MIN_GAP_SECONDS if following else float("inf")
        if cue.end - cue.start < MIN_CUE_SECONDS:
            # Hold a too-short cue a little longer, but never into the next one.
            cue.end = max(cue.end, min(cue.start + MIN_CUE_SECONDS, limit))
        if following and cue.end > limit:
            cue.end = max(cue.start + 0.05, limit)
        if cue.end <= cue.start:
            cue.end = cue.start + 0.05
    return cues


def build_subtitle_cues(chunks: list[Chunk], include_speakers: bool = False) -> list[SubtitleCue]:
    """include_speakers puts "Имя: " in front of the first cue of each turn."""
    cues: list[SubtitleCue] = []
    previous_speaker: Optional[str] = None
    for chunk in chunks:
        words = list(chunk.words) or words_from_text(chunk)
        if not words:
            continue
        if include_speakers and chunk.speaker != previous_speaker:
            # Glue the label onto the first word so it is counted by the line
            # limits and can never end up alone on its own cue.
            words[0] = replace(words[0], text=f"{chunk.speaker}: {words[0].text}")
        previous_speaker = chunk.speaker

        pieces = [p for sentence in _split_sentences(words) for p in _split_sentence(sentence)]
        for piece in _merge_small(pieces):
            text = _text(piece)
            lines = wrap_lines(text) or _hard_wrap(text)
            cues.append(SubtitleCue(start=piece[0].start, end=piece[-1].end, lines=lines, speaker=chunk.speaker))
    return _fix_timing(cues)
