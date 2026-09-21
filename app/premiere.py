"""Premiere Pro transcript JSON — for Text panel > Transcript > Import >
"Import Static Transcript".

Unlike SRT this carries a timecode for every single word plus who said it, so
Premiere builds a real, word-synced transcript on the clip: click a word to
jump there, search, and edit the video by editing the text (Text-Based
Editing) — including for languages Premiere cannot transcribe by itself.

The layout follows the published examples of Adobe's import spec (its own
PremierePro_transcript_format_spec is an attachment on the Adobe community
forum and could not be fetched), and matches the entities reverse-engineered
from Premiere's binary .prtranscript export:

    {
      "language": "ru-ru",
      "segments": [ {                      # one per utterance
          "language": "ru-ru", "speaker": "<uuid>",
          "start": 1.2, "duration": 3.4,   # seconds from the start of the audio
          "words": [ {"text": "Привет,", "start": 1.2, "duration": 0.4,
                      "confidence": 1, "eos": false, "tags": [], "type": "word"} ]
      } ],
      "speakers": [ {"id": "<uuid>", "name": "Иван"} ]
    }

.prtranscript itself is intentionally not written: it is an undocumented
FlatBuffers binary, and a file that merely looks right but that Premiere
refuses to open is worse than no file.
"""
from __future__ import annotations

import re
import uuid

from app.merge import Chunk
from app.subtitles import ends_sentence, words_from_text

# Whisper (ISO 639-1) -> the locale-style codes Premiere uses. An unknown
# language falls back to en-us: the import still works, only the label is off.
_LANGUAGE_CODES = {
    "ru": "ru-ru", "en": "en-us", "uk": "uk-ua", "de": "de-de", "fr": "fr-fr",
    "es": "es-es", "it": "it-it", "pt": "pt-br", "pl": "pl-pl", "nl": "nl-nl",
    "tr": "tr-tr", "cs": "cs-cz", "sv": "sv-se", "da": "da-dk", "fi": "fi-fi",
    "no": "nb-no", "nb": "nb-no", "hi": "hi-in", "ja": "ja-jp", "ko": "ko-kr",
    "zh": "zh-cn", "ar": "ar-sa", "he": "he-il", "ro": "ro-ro", "hu": "hu-hu",
    "bg": "bg-bg", "el": "el-gr", "id": "id-id", "vi": "vi-vn", "th": "th-th",
}
_FALLBACK_LANGUAGE = "en-us"
_MIN_WORD_SECONDS = 0.01
_PUNCTUATION_ONLY = re.compile(r"^[\W_]+$")


def premiere_language(code: str) -> str:
    code = (code or "").strip().lower()
    if "-" in code:
        return code  # already locale-style
    return _LANGUAGE_CODES.get(code, _FALLBACK_LANGUAGE)


def _speaker_id(name: str) -> str:
    # Derived from the name, so the same person gets the same id in every
    # export (and different names never collide).
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pisar:speaker:{name}"))


def build_premiere_transcript(chunks: list[Chunk]) -> dict:
    language = premiere_language(next((c.language for c in chunks if c.language), ""))
    speakers: dict[str, str] = {}
    segments: list[dict] = []
    previous: dict | None = None  # the last word written, across segments

    for chunk in chunks:
        raw_words = [w for w in (list(chunk.words) or words_from_text(chunk)) if w.text.strip()]
        if not raw_words:
            continue
        speaker_id = speakers.setdefault(chunk.speaker, _speaker_id(chunk.speaker))
        words: list[dict] = []
        for w in raw_words:
            start = round(w.start, 3)
            if previous is not None:
                # Whisper word times can overlap, run backwards, or give two
                # words the very same start. Adobe's own files never overlap,
                # so: the later word may start no sooner than one minimum word
                # after the earlier one (nudged by at most 10 ms), and the
                # earlier word is then trimmed to end where the later begins.
                start = max(start, round(previous["start"] + _MIN_WORD_SECONDS, 3))
                if start < previous["start"] + previous["duration"]:
                    previous["duration"] = round(start - previous["start"], 3)
            word = {
                "confidence": 1,
                "duration": round(max(_MIN_WORD_SECONDS, w.end - start), 3),
                "eos": ends_sentence(w.text),
                "start": start,
                "tags": [],
                "text": w.text,
                "type": "punctuation" if _PUNCTUATION_ONLY.match(w.text) else "word",
            }
            words.append(word)
            previous = word
        segments.append({"language": language, "speaker": speaker_id, "words": words})

    # Segment bounds are derived last, once trimming has settled every word.
    for segment in segments:
        first, last = segment["words"][0], segment["words"][-1]
        segment["start"] = first["start"]
        segment["duration"] = round(last["start"] + last["duration"] - first["start"], 3)

    return {
        "language": language,
        "segments": segments,
        "speakers": [{"id": sid, "name": name} for name, sid in speakers.items()],
    }
