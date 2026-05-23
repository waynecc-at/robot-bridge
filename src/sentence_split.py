"""Chinese-aware sentence segmentation for streaming TTS.

Splits text at natural boundaries (。！？；：\n), and additionally
at pauses (，、) when the segment exceeds max_len.
"""

import re

# Sentence-ending punctuation (split here for sure)
_SENTENCE_END = re.compile(r'[。！？；：\n]')

# Pause punctuation (split here only as fallback for long segments)
_PAUSE = re.compile(r'[，、]')

# Maximum chars before forcing a split at the nearest pause
_MAX_SEGMENT = 80


def split_sentences(text: str, max_len: int = _MAX_SEGMENT) -> list[str]:
    """Split text into segments suitable for per-sentence TTS synthesis.

    Strategy:
    1. First split at sentence endings (。！？；：\n) — these are natural breaks.
    2. If any segment exceeds max_len, split further at pause marks (，、).
    3. Keep punctuation attached to the preceding segment.

    Returns list of non-empty text segments.
    """
    if not text:
        return []

    # Phase 1: split at sentence endings (keep delimiter with preceding text)
    raw = _split_keep_delim(text, _SENTENCE_END)
    if not raw:
        return []

    # Phase 2: further split any over-long segments at pause marks
    result = []
    for segment in raw:
        segment = segment.strip()
        if not segment:
            continue
        if len(segment) <= max_len:
            result.append(segment)
        else:
            # Split at pause marks
            sub_segments = _split_keep_delim(segment, _PAUSE)
            for sub in sub_segments:
                sub = sub.strip()
                if sub:
                    result.append(sub)

    return result or [text]


def _split_keep_delim(text: str, pattern: re.Pattern) -> list[str]:
    """Split text, keeping delimiter attached to the preceding part."""
    parts = []
    last = 0
    for m in pattern.finditer(text):
        start, end = m.start(), m.end()
        if start > last:
            parts.append(text[last:start] + text[start:end])
        else:
            parts.append(text[last:end])
        last = end
    if last < len(text):
        tail = text[last:]
        if parts:
            parts[-1] += tail
        else:
            parts.append(tail)
    return parts
