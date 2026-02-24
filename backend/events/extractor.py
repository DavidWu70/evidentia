from __future__ import annotations

"""
Extract minimal evidence-backed clinical events.

Each extracted event must include:
- segment_id
- t0/t1
- quote (verbatim)

Design intent:
- Keep MVP extraction scope narrow (risk/symptom/duration_onset).
- Prioritize traceability over aggressive inference.
- Prevent hallucinated summaries from entering risk and note layers.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Sequence


EventType = Literal["risk_cue", "symptom", "duration_onset"]
Polarity = Literal["present", "absent", "uncertain"]


@dataclass(frozen=True)
class EventUtterance:
    segment_id: str
    t0: float
    t1: float
    speaker: str
    text: str
    asr_confidence: float | None = None


@dataclass(frozen=True)
class ExtractedEvent:
    event_id: str
    type: EventType
    label: str
    polarity: Polarity
    confidence: float
    speaker: str
    segment_id: str
    t0: float
    t1: float
    quote: str


_DURATION_RE = re.compile(
    r"\b(\d+\s+(?:day|days|week|weeks|month|months|year|years))\b",
    re.IGNORECASE,
)
_SINCE_RE = re.compile(r"\bsince\s+([^.?!,;]{1,40})", re.IGNORECASE)

_PASSIVE_SI_RE = re.compile(
    r"\b(wish i (?:wouldn't|would not) wake up|better off dead|don't want to live)\b",
    re.IGNORECASE,
)
_ACTIVE_SI_RE = re.compile(
    r"\b(suicidal|suicide|kill myself|end my life|want to die)\b",
    re.IGNORECASE,
)
_SI_NEGATION_RE = re.compile(
    r"\b(no (?:suicidal thoughts|thoughts of harming myself)|not suicidal|deny(?:ing)? suicidal)\b",
    re.IGNORECASE,
)
_PLAN_INTENT_RE = re.compile(
    r"\b(plan to (?:kill myself|end my life)|intent to (?:kill myself|die)|i will kill myself)\b",
    re.IGNORECASE,
)
_HI_RE = re.compile(
    r"\b(homicidal|want to hurt others|want to kill someone)\b",
    re.IGNORECASE,
)
_PSYCHOSIS_RE = re.compile(
    r"\b(hearing voices|hallucination|hallucinations|delusion|delusional|seeing things that aren't there)\b",
    re.IGNORECASE,
)
_CLINICIAN_RISK_ALLOW_RE = re.compile(
    r"\b("
    r"i\s+wish\s+i\s+(?:wouldn't|would\s+not)\s+wake\s+up|"
    r"i\s+don't\s+want\s+to\s+live|"
    r"i\s+want\s+to\s+die|"
    r"i\s+am\s+suicidal|"
    r"i\s+feel\s+suicidal|"
    r"i\s+will\s+kill\s+myself|"
    r"i\s+plan\s+to\s+(?:kill\s+myself|end\s+my\s+life)|"
    r"better\s+off\s+dead"
    r")\b",
    re.IGNORECASE,
)

_SYMPTOM_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("depressed_mood", ("depressed", "sad most of the day", "low mood", "down")),
    ("anxiety", ("anxious", "anxiety", "panic", "worrying")),
    ("sleep_disturbance", ("can't sleep", "insomnia", "wake up", "sleep is bad", "poor sleep")),
    ("appetite_change", ("appetite is lower", "lost appetite", "decreased appetite", "no appetite")),
    ("fatigue_low_energy", ("exhausted", "fatigue", "no energy", "low energy", "tired")),
]


def extract_minimal_events(utterances: Sequence[EventUtterance]) -> list[ExtractedEvent]:
    events: list[ExtractedEvent] = []
    seen: set[tuple[str, str, str, str]] = set()
    counter = 1

    stitched = _stitch_utterances_for_extraction(utterances)
    for utterance in stitched:
        text = (utterance.text or "").strip()
        if not text:
            continue
        normalized = text.lower()

        candidates = []
        candidates.extend(_extract_risk_candidates(normalized))
        candidates.extend(_extract_symptom_candidates(normalized))
        candidates.extend(_extract_duration_candidates(normalized))

        for event_type, label, polarity, confidence in candidates:
            key = (utterance.segment_id, event_type, label, polarity)
            if key in seen:
                continue
            seen.add(key)
            events.append(
                ExtractedEvent(
                    event_id=f"evt_{counter:05d}",
                    type=event_type,
                    label=label,
                    polarity=polarity,
                    confidence=confidence,
                    speaker=(utterance.speaker or "other"),
                    segment_id=utterance.segment_id,
                    t0=float(utterance.t0),
                    t1=float(utterance.t1),
                    quote=text,
                )
            )
            counter += 1

    return events


def apply_event_quality_guardrails(
    events: Sequence[ExtractedEvent],
) -> tuple[list[ExtractedEvent], dict[str, Any]]:
    """
    Apply conservative post-hoc constraints to suppress low-quality event false positives.

    Design intent:
    - Keep safety signals tied to the patient narrative.
    - Prevent obvious clinician-utterance risk false positives from leaking to UI/risk cards.
    """
    kept: list[ExtractedEvent] = []
    dropped_reasons: Counter[str] = Counter()
    dropped_samples: list[dict[str, str]] = []

    for item in events:
        speaker = str(item.speaker or "").strip().lower()
        quote = str(item.quote or "").strip()
        if item.type == "risk_cue" and speaker == "clinician":
            if not _CLINICIAN_RISK_ALLOW_RE.search(quote):
                reason = "drop_clinician_risk_without_first_person_signal"
                dropped_reasons[reason] += 1
                if len(dropped_samples) < 5:
                    dropped_samples.append(
                        {
                            "event_id": str(item.event_id),
                            "label": str(item.label),
                            "speaker": speaker or "unknown",
                            "quote": quote[:140],
                        }
                    )
                continue
        kept.append(item)

    debug = {
        "kept": len(kept),
        "dropped": max(0, len(events) - len(kept)),
        "drop_reasons": dict(dropped_reasons),
        "drop_samples": dropped_samples,
    }
    return kept, debug


def apply_rule_risk_backstop(
    events: Sequence[ExtractedEvent],
    utterances: Sequence[EventUtterance],
) -> tuple[list[ExtractedEvent], dict[str, Any]]:
    """
    Backfill high-priority patient risk cues using deterministic rules when model misses them.

    Design intent:
    - Protect safety sensitivity against model label drift (e.g., risk phrasing mislabeled as symptom).
    - Keep augmentation conservative: only add present risk cues not already present.
    """
    existing = list(events)
    if not utterances:
        return existing, {"status": "skipped", "reason": "empty_utterances"}

    risk_labels = {
        "passive_suicidal_ideation",
        "suicidal_ideation",
        "suicidal_plan_or_intent",
        "homicidal_ideation",
        "psychosis_cue",
    }
    seen_risk_keys = {
        (item.segment_id, item.label, item.polarity)
        for item in existing
        if item.type == "risk_cue"
    }
    rule_events = extract_minimal_events(utterances)
    candidates = [
        item
        for item in rule_events
        if item.type == "risk_cue"
        and item.polarity == "present"
        and item.label in risk_labels
        and str(item.speaker or "").strip().lower() != "clinician"
    ]

    added = 0
    for item in candidates:
        key = (item.segment_id, item.label, item.polarity)
        if key in seen_risk_keys:
            continue
        seen_risk_keys.add(key)
        added += 1
        existing.append(
            ExtractedEvent(
                event_id=f"evt_backstop_{added:05d}",
                type=item.type,
                label=item.label,
                polarity=item.polarity,
                confidence=max(0.75, min(0.95, float(item.confidence))),
                speaker=item.speaker,
                segment_id=item.segment_id,
                t0=item.t0,
                t1=item.t1,
                quote=item.quote,
            )
        )

    return existing, {
        "status": "applied",
        "rule_risk_candidates": len(candidates),
        "added": added,
    }


def apply_event_consistency_harmonization(
    events: Sequence[ExtractedEvent],
) -> tuple[list[ExtractedEvent], dict[str, Any]]:
    """
    Harmonization hook for post-extraction event consistency.

    Design intent:
    - Keep this stage explicit for future conflict rules.
    - Current policy is non-destructive: do not suppress model labels here.
    """
    if not events:
        return [], {"status": "skipped", "reason": "empty_events"}
    kept = list(events)
    return kept, {
        "status": "applied",
        "dropped": 0,
        "reason": "no_conflict_rules_active",
        "drop_samples": [],
    }


def _stitch_utterances_for_extraction(utterances: Sequence[EventUtterance]) -> list[EventUtterance]:
    """
    Repair short phrase splits across adjacent utterances before rule extraction.

    Design intent:
    - Recover risk/symptom phrases that are often cut by ASR chunk boundaries.
    - Keep evidence traceable by anchoring stitched text to the first segment_id.
    """
    ordered = sorted(utterances, key=lambda item: (float(item.t0), float(item.t1), str(item.segment_id)))
    if not ordered:
        return []

    stitched: list[EventUtterance] = [ordered[0]]
    for curr in ordered[1:]:
        prev = stitched[-1]
        if _should_stitch(prev, curr):
            stitched[-1] = EventUtterance(
                segment_id=prev.segment_id,
                t0=float(prev.t0),
                t1=max(float(prev.t1), float(curr.t1)),
                speaker=str(prev.speaker or curr.speaker or "other"),
                text=_join_text(prev.text, curr.text),
                asr_confidence=_merge_confidence(prev.asr_confidence, curr.asr_confidence),
            )
            continue
        stitched.append(curr)
    return stitched


def _should_stitch(prev: EventUtterance, curr: EventUtterance) -> bool:
    prev_text = str(prev.text or "").strip()
    curr_text = str(curr.text or "").strip()
    if not prev_text or not curr_text:
        return False
    if str(prev.speaker or "other").lower() != str(curr.speaker or "other").lower():
        return False
    gap = float(curr.t0) - float(prev.t1)
    if gap > 1.0:
        return False

    prev_last = prev_text.rstrip()[-1] if prev_text.rstrip() else ""
    combined = f"{prev_text} {curr_text}".strip().lower()

    # High-priority repair for split passive SI phrase.
    if "wish i would not wake up" in combined or "wish i wouldn't wake up" in combined:
        return True

    # Generic continuation heuristics for ASR boundary cuts.
    if prev_last not in ".!?":
        return True
    if curr_text[:1].islower():
        return True
    if curr_text.lower().startswith(("and ", "but ", "or ", "because ", "so ", "that ", "wake ", "up ")):
        return True
    return False


def _join_text(left: str, right: str) -> str:
    return f"{str(left or '').rstrip()} {str(right or '').lstrip()}".strip()


def _merge_confidence(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return float((float(a) + float(b)) * 0.5)


def _extract_risk_candidates(text: str) -> list[tuple[EventType, str, Polarity, float]]:
    out: list[tuple[EventType, str, Polarity, float]] = []

    if _SI_NEGATION_RE.search(text):
        out.append(("risk_cue", "suicidal_ideation", "absent", 0.86))
    else:
        if _PASSIVE_SI_RE.search(text):
            out.append(("risk_cue", "passive_suicidal_ideation", "present", 0.92))
        if _ACTIVE_SI_RE.search(text):
            out.append(("risk_cue", "suicidal_ideation", "present", 0.9))

    if _PLAN_INTENT_RE.search(text):
        out.append(("risk_cue", "suicidal_plan_or_intent", "present", 0.95))
    if _HI_RE.search(text):
        out.append(("risk_cue", "homicidal_ideation", "present", 0.9))
    if _PSYCHOSIS_RE.search(text):
        out.append(("risk_cue", "psychosis_cue", "present", 0.88))
    return out


def _extract_symptom_candidates(text: str) -> list[tuple[EventType, str, Polarity, float]]:
    out: list[tuple[EventType, str, Polarity, float]] = []
    for label, keywords in _SYMPTOM_RULES:
        if not any(keyword in text for keyword in keywords):
            continue
        polarity: Polarity = "absent" if _is_negated(text, keywords) else "present"
        confidence = 0.72 if polarity == "present" else 0.67
        out.append(("symptom", label, polarity, confidence))
    return out


def _extract_duration_candidates(text: str) -> list[tuple[EventType, str, Polarity, float]]:
    out: list[tuple[EventType, str, Polarity, float]] = []
    if _DURATION_RE.search(text):
        out.append(("duration_onset", "duration_mentioned", "present", 0.74))
    if _SINCE_RE.search(text):
        out.append(("duration_onset", "onset_mentioned", "present", 0.73))
    return out


def _is_negated(text: str, keywords: Sequence[str]) -> bool:
    for keyword in keywords:
        if f"no {keyword}" in text:
            return True
        if f"not {keyword}" in text:
            return True
    if "denies" in text or "deny" in text:
        return True
    return False
