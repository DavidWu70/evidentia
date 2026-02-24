from __future__ import annotations

"""
Normalize heterogeneous transcript inputs into typed ASR segments and speaker turns.

Design intent:
- Keep `/transcribe_structured` orchestration thin by moving input-shape handling here.
- Support smooth migration from demo text/chunk inputs to real audio ASR outputs.
- Preserve speaker/timestamp traceability so downstream evidence remains auditable.
"""

import re
from typing import Sequence

from backend.asr.models import ASRSegment, SpeakerTurn

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SPEAKER_PREFIX_RE = re.compile(r"^\s*([A-Za-z][A-Za-z _-]{0,31})\s*:\s*(.+)$")


def speaker_from_prefix(text: str) -> tuple[str | None, str]:
    match = _SPEAKER_PREFIX_RE.match((text or "").strip())
    if not match:
        return None, (text or "").strip()

    speaker_raw = match.group(1).strip().lower()
    content = match.group(2).strip()
    return normalize_speaker(speaker_raw), content


def normalize_speaker(raw: str | None) -> str:
    label = (raw or "").strip().lower()
    if label in {"patient", "pt", "client"}:
        return "patient"
    if label in {"clinician", "doctor", "dr", "provider", "therapist"}:
        return "clinician"
    if label in {"other"}:
        return "other"
    return "other"


def build_segments_and_turns_from_text(
    transcript_text: str,
    *,
    start_at_sec: float = 0.0,
) -> tuple[list[ASRSegment], list[SpeakerTurn]]:
    source = (transcript_text or "").strip()
    if not source:
        return [], []

    parts = [item.strip() for item in _SENTENCE_SPLIT_RE.split(source) if item.strip()]
    segments: list[ASRSegment] = []
    turns: list[SpeakerTurn] = []
    cursor = float(max(0.0, start_at_sec))

    for sentence in parts:
        speaker, content = speaker_from_prefix(sentence)
        text = content or sentence
        word_count = len([token for token in text.split() if token])
        duration = max(1.0, min(word_count * 0.33, 5.2))
        t0 = round(cursor, 2)
        t1 = round(cursor + duration, 2)
        cursor = t1 + 0.2

        segments.append(
            ASRSegment(
                start=t0,
                end=t1,
                text=text,
                avg_logprob=None,
                no_speech_prob=None,
            )
        )
        turns.append(
            SpeakerTurn(
                start=t0,
                end=t1,
                speaker=speaker or "other",
                speaker_id=speaker or "other",
                speaker_role=speaker or "other",
                confidence=None,
                diar_confidence=None,
                role_confidence=None,
            )
        )

    return segments, turns


def infer_turns_from_segments(segments: Sequence[ASRSegment]) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        speaker, _ = speaker_from_prefix(segment.text)
        turns.append(
            SpeakerTurn(
                start=segment.start,
                end=segment.end,
                speaker=speaker or "other",
                speaker_id=speaker or "other",
                speaker_role=speaker or "other",
                confidence=None,
                diar_confidence=None,
                role_confidence=None,
            )
        )
    return turns
