from __future__ import annotations

"""
Session-stable speaker-role mapping for diarized turns.

Design intent:
- Keep `speaker_id -> patient/clinician` assignment stable across incremental windows.
- Use conservative evidence accumulation; avoid aggressive role flips on weak signals.
- Preserve safety: low-confidence mappings remain `other` instead of forced mislabels.
"""

import os
import re
from typing import Any, Sequence

from backend.asr.models import ASRSegment, SpeakerTurn

_ROLE_CLINICIAN_RE = re.compile(
    r"\b(how|what|when|where|why|can you|could you|tell me|let us|let's|thanks for sharing|thank you for sharing)\b|\?",
    flags=re.IGNORECASE,
)
_ROLE_PATIENT_RE = re.compile(
    r"\b(i|i'm|i've|my|me|feel|feeling|sleep|anxious|depressed|pain|tired|wish i would not|wish i wouldn't)\b",
    flags=re.IGNORECASE,
)
_ROLE_CHOICES = {"patient", "clinician", "other"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _score_text_role(text: str) -> tuple[float, float]:
    source = str(text or "")
    if not source:
        return 0.0, 0.0
    clinician_hits = float(len(_ROLE_CLINICIAN_RE.findall(source)))
    patient_hits = float(len(_ROLE_PATIENT_RE.findall(source)))
    return clinician_hits, patient_hits


def _valid_speaker_id(turn: SpeakerTurn) -> bool:
    sid = str(turn.speaker_id or "").strip().lower()
    if not sid:
        return False
    if sid in {"other", "patient", "clinician"}:
        return False
    return True


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))


def _collect_evidence(
    turns: Sequence[SpeakerTurn],
    segments: Sequence[ASRSegment],
) -> dict[str, dict[str, float]]:
    evidence: dict[str, dict[str, float]] = {}
    candidates = [turn for turn in turns if _valid_speaker_id(turn)]
    if not candidates:
        return evidence

    for seg in segments:
        text = str(seg.text or "").strip()
        if not text:
            continue
        c_hits, p_hits = _score_text_role(text)
        if c_hits <= 0.0 and p_hits <= 0.0:
            continue

        best_sid: str | None = None
        best_overlap = 0.0
        for turn in candidates:
            ov = _overlap(turn.start, turn.end, seg.start, seg.end)
            if ov > best_overlap:
                best_overlap = ov
                best_sid = str(turn.speaker_id)
        if best_sid is None or best_overlap <= 0.05:
            continue

        bucket = evidence.setdefault(best_sid, {"patient": 0.0, "clinician": 0.0})
        # Weight lexical cues by temporal overlap to reduce accidental cross-attribution.
        bucket["clinician"] += c_hits * best_overlap
        bucket["patient"] += p_hits * best_overlap

    return evidence


def _decide_role(
    patient_score: float,
    clinician_score: float,
    *,
    prev_role: str | None,
    prev_confidence: float | None,
) -> tuple[str, float]:
    min_score = _env_float("EVIDENTIA_ROLE_MAP_MIN_SCORE", 0.9)
    min_diff = _env_float("EVIDENTIA_ROLE_MAP_MIN_DIFF", 0.35)
    min_ratio = _env_float("EVIDENTIA_ROLE_MAP_MIN_RATIO", 1.25)
    hold_prev_threshold = _env_float("EVIDENTIA_ROLE_MAP_HOLD_PREV_THRESHOLD", 0.55)

    p = max(0.0, float(patient_score))
    c = max(0.0, float(clinician_score))
    total = p + c
    if total <= 1e-9:
        if prev_role in {"patient", "clinician"}:
            return prev_role, float(prev_confidence or 0.5)
        return "other", 0.0

    if p >= c:
        top_role = "patient"
        top = p
        second = c
    else:
        top_role = "clinician"
        top = c
        second = p

    diff = top - second
    ratio = top / max(1e-6, second)
    confidence = max(0.0, min(1.0, top / max(1e-6, total)))

    if top >= min_score and diff >= min_diff and ratio >= min_ratio:
        return top_role, confidence

    if prev_role in {"patient", "clinician"} and float(prev_confidence or 0.0) >= hold_prev_threshold:
        # Hysteresis: keep previous role under weak/conflicting new evidence.
        return prev_role, float(prev_confidence or confidence)

    return "other", confidence


def apply_stable_role_mapping(
    turns: Sequence[SpeakerTurn],
    segments: Sequence[ASRSegment],
    *,
    state: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[SpeakerTurn], dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Apply and update session-level role mapping for diarized turns.

    `state` is keyed by `speaker_id` and stores accumulated scores/role decisions.
    """
    current_state: dict[str, dict[str, Any]] = {
        str(k): dict(v)
        for k, v in (state or {}).items()
    }
    evidence = _collect_evidence(turns, segments)

    # Update accumulated scores and derive stable role per speaker_id.
    for sid, scores in evidence.items():
        slot = current_state.setdefault(
            sid,
            {
                "patient_score": 0.0,
                "clinician_score": 0.0,
                "role": "other",
                "role_confidence": 0.0,
            },
        )
        slot["patient_score"] = float(slot.get("patient_score", 0.0) or 0.0) + float(scores.get("patient", 0.0) or 0.0)
        slot["clinician_score"] = float(slot.get("clinician_score", 0.0) or 0.0) + float(scores.get("clinician", 0.0) or 0.0)
        role, conf = _decide_role(
            slot["patient_score"],
            slot["clinician_score"],
            prev_role=str(slot.get("role", "other") or "other"),
            prev_confidence=float(slot.get("role_confidence", 0.0) or 0.0),
        )
        slot["role"] = role
        slot["role_confidence"] = conf

    mapped: list[SpeakerTurn] = []
    for turn in turns:
        if not _valid_speaker_id(turn):
            role = str(turn.speaker_role or turn.speaker or "other").strip().lower()
            if role not in _ROLE_CHOICES:
                role = "other"
            mapped.append(
                SpeakerTurn(
                    start=turn.start,
                    end=turn.end,
                    speaker=role,
                    speaker_id=turn.speaker_id,
                    speaker_role=role,
                    confidence=turn.confidence,
                    diar_confidence=turn.diar_confidence,
                    role_confidence=turn.role_confidence,
                )
            )
            continue

        sid = str(turn.speaker_id)
        slot = current_state.get(sid, {})
        role = str(slot.get("role", turn.speaker_role or turn.speaker or "other")).strip().lower()
        if role not in _ROLE_CHOICES:
            role = "other"
        role_conf = float(slot.get("role_confidence", 0.0) or 0.0)
        mapped.append(
            SpeakerTurn(
                start=turn.start,
                end=turn.end,
                speaker=role,
                speaker_id=sid,
                speaker_role=role,
                confidence=turn.confidence,
                diar_confidence=turn.diar_confidence,
                role_confidence=role_conf,
            )
        )

    assigned = sum(1 for item in mapped if str(item.speaker_role or "") in {"patient", "clinician"})
    debug = {
        "status": "ok",
        "mapped_turns": len(mapped),
        "assigned_turns": assigned,
        "speakers_tracked": len(current_state),
        "evidence_speakers": len(evidence),
    }
    return mapped, current_state, debug
