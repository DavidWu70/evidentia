from __future__ import annotations

"""
Format speaker utterances into compact transcript lines.

Design intent:
- Keep UI and prompt-facing transcript deterministic.
- Normalize punctuation and speaker aliases for readability.
"""

import re
from collections import defaultdict
from typing import Sequence

from backend.asr.models import SpeakerUtterance

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _ensure_terminal_punctuation(text: str) -> str:
    trimmed = text.strip()
    if not trimmed:
        return ""
    if trimmed[-1] in ".!?":
        return trimmed
    return f"{trimmed}."


def _speaker_alias_map(utterances: Sequence[SpeakerUtterance]) -> dict[str, str]:
    durations: dict[str, float] = defaultdict(float)
    for item in utterances:
        if item.speaker.strip().lower() == "other":
            continue
        durations[item.speaker] += max(0.0, item.end - item.start)

    ranked = sorted(durations.items(), key=lambda pair: (-pair[1], pair[0]))
    mapping: dict[str, str] = {}
    if ranked:
        mapping[ranked[0][0]] = "SpeakerA"
    if len(ranked) >= 2:
        mapping[ranked[1][0]] = "SpeakerB"
    return mapping


def _split_sentences(text: str, *, split_sentences: bool) -> list[str]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []
    if not split_sentences:
        return [normalized]
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()]


def format_for_display(
    utterances: Sequence[SpeakerUtterance],
    *,
    split_sentences: bool = True,
) -> str:
    if not utterances:
        return ""

    alias_map = _speaker_alias_map(utterances)
    lines: list[str] = []

    for utterance in utterances:
        speaker = alias_map.get(utterance.speaker, "Other")
        for sentence in _split_sentences(utterance.text, split_sentences=split_sentences):
            normalized = _ensure_terminal_punctuation(sentence)
            if normalized:
                lines.append(f"{speaker}: {normalized}")

    return "\n".join(lines)

