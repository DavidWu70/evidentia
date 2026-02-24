from __future__ import annotations

"""
Align ASR segments to speaker turns with overlap-based heuristics.

Design intent:
- Provide stable speaker attribution without requiring perfect diarization.
- Preserve timestamp windows so evidence links remain verifiable.
"""

from collections import Counter
from typing import Any
from typing import Sequence

from backend.asr.models import ASRSegment, SpeakerTurn, SpeakerUtterance


def _overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def _assigned_speaker(
    segment: ASRSegment,
    turns: Sequence[SpeakerTurn],
    *,
    min_overlap_sec: float,
    ambiguity_margin_sec: float,
) -> SpeakerTurn:
    fallback = SpeakerTurn(
        start=segment.start,
        end=segment.end,
        speaker="other",
        speaker_id="other",
        speaker_role="other",
        confidence=None,
        diar_confidence=None,
        role_confidence=None,
    )
    if not turns:
        return fallback

    scored: list[tuple[float, SpeakerTurn]] = []
    for turn in turns:
        overlap = _overlap_duration(segment.start, segment.end, turn.start, turn.end)
        if overlap > 0.0:
            scored.append((overlap, turn))

    if not scored:
        return fallback

    scored.sort(key=lambda item: item[0], reverse=True)
    best_overlap, best_turn = scored[0]
    if best_overlap < min_overlap_sec:
        return fallback

    if len(scored) >= 2:
        second_overlap, second_turn = scored[1]
        if (
            second_turn.speaker != best_turn.speaker
            and abs(best_overlap - second_overlap) <= ambiguity_margin_sec
        ):
            return fallback

    return best_turn


def _assigned_speaker_with_reason(
    segment: ASRSegment,
    turns: Sequence[SpeakerTurn],
    *,
    min_overlap_sec: float,
    ambiguity_margin_sec: float,
) -> tuple[SpeakerTurn, str, float, float]:
    fallback = SpeakerTurn(
        start=segment.start,
        end=segment.end,
        speaker="other",
        speaker_id="other",
        speaker_role="other",
        confidence=None,
        diar_confidence=None,
        role_confidence=None,
    )
    if not turns:
        return fallback, "fallback_no_turns", 0.0, 0.0

    scored: list[tuple[float, SpeakerTurn]] = []
    for turn in turns:
        overlap = _overlap_duration(segment.start, segment.end, turn.start, turn.end)
        if overlap > 0.0:
            scored.append((overlap, turn))

    if not scored:
        return fallback, "fallback_no_overlap", 0.0, 0.0

    scored.sort(key=lambda item: item[0], reverse=True)
    best_overlap, best_turn = scored[0]
    second_overlap = scored[1][0] if len(scored) >= 2 else 0.0
    if best_overlap < min_overlap_sec:
        return fallback, "fallback_low_overlap", float(best_overlap), float(second_overlap)

    if len(scored) >= 2:
        _, second_turn = scored[1]
        if (
            second_turn.speaker != best_turn.speaker
            and abs(best_overlap - second_overlap) <= ambiguity_margin_sec
        ):
            return fallback, "fallback_ambiguous_overlap", float(best_overlap), float(second_overlap)

    return best_turn, "assigned", float(best_overlap), float(second_overlap)


def _merge_consecutive_utterances(
    utterances: Sequence[SpeakerUtterance],
    *,
    merge_gap_sec: float,
) -> list[SpeakerUtterance]:
    if not utterances:
        return []

    ordered = sorted(utterances, key=lambda item: (item.start, item.end))
    merged: list[SpeakerUtterance] = [ordered[0]]

    for item in ordered[1:]:
        prev = merged[-1]
        gap = item.start - prev.end
        if (
            item.speaker == prev.speaker
            and (item.speaker_id or "") == (prev.speaker_id or "")
            and (item.speaker_role or "") == (prev.speaker_role or "")
            and gap <= merge_gap_sec
        ):
            prev_dur = max(0.0, prev.end - prev.start)
            curr_dur = max(0.0, item.end - item.start)
            prev_conf = prev.diar_confidence
            curr_conf = item.diar_confidence
            prev_role_conf = prev.role_confidence
            curr_role_conf = item.role_confidence
            merged_conf: float | None
            merged_role_conf: float | None
            if prev_conf is not None and curr_conf is not None:
                total = max(1e-6, prev_dur + curr_dur)
                merged_conf = float((prev_conf * prev_dur + curr_conf * curr_dur) / total)
            elif prev_conf is not None:
                merged_conf = float(prev_conf)
            elif curr_conf is not None:
                merged_conf = float(curr_conf)
            else:
                merged_conf = None
            if prev_role_conf is not None and curr_role_conf is not None:
                total = max(1e-6, prev_dur + curr_dur)
                merged_role_conf = float((prev_role_conf * prev_dur + curr_role_conf * curr_dur) / total)
            elif prev_role_conf is not None:
                merged_role_conf = float(prev_role_conf)
            elif curr_role_conf is not None:
                merged_role_conf = float(curr_role_conf)
            else:
                merged_role_conf = None
            merged[-1] = SpeakerUtterance(
                start=prev.start,
                end=max(prev.end, item.end),
                speaker=prev.speaker,
                speaker_id=prev.speaker_id,
                speaker_role=prev.speaker_role,
                diar_confidence=merged_conf,
                role_confidence=merged_role_conf,
                text=f"{prev.text.rstrip()} {item.text.lstrip()}".strip(),
            )
            continue
        merged.append(item)

    return merged


def align_segments_to_turns(
    segments: Sequence[ASRSegment],
    turns: Sequence[SpeakerTurn],
    *,
    min_overlap_sec: float = 0.2,
    ambiguity_margin_sec: float = 0.05,
    merge_gap_sec: float = 0.6,
) -> list[SpeakerUtterance]:
    utterances: list[SpeakerUtterance] = []
    for segment in sorted(segments, key=lambda seg: (seg.start, seg.end)):
        text = segment.text.strip()
        if not text:
            continue
        speaker = _assigned_speaker(
            segment,
            turns,
            min_overlap_sec=min_overlap_sec,
            ambiguity_margin_sec=ambiguity_margin_sec,
        )
        utterances.append(
            SpeakerUtterance(
                start=segment.start,
                end=segment.end,
                speaker=speaker.speaker,
                speaker_id=speaker.speaker_id,
                speaker_role=speaker.speaker_role or speaker.speaker,
                diar_confidence=speaker.diar_confidence if speaker.diar_confidence is not None else speaker.confidence,
                role_confidence=speaker.role_confidence,
                text=text,
            )
        )
    return _merge_consecutive_utterances(utterances, merge_gap_sec=merge_gap_sec)


def align_segments_to_turns_with_debug(
    segments: Sequence[ASRSegment],
    turns: Sequence[SpeakerTurn],
    *,
    min_overlap_sec: float = 0.2,
    ambiguity_margin_sec: float = 0.05,
    merge_gap_sec: float = 0.6,
    fallback_sample_limit: int = 8,
) -> tuple[list[SpeakerUtterance], dict[str, Any]]:
    utterances: list[SpeakerUtterance] = []
    reason_counts: Counter[str] = Counter()
    fallback_samples: list[dict[str, Any]] = []
    empty_text_segments = 0

    for segment in sorted(segments, key=lambda seg: (seg.start, seg.end)):
        text = segment.text.strip()
        if not text:
            empty_text_segments += 1
            continue
        speaker, reason, best_overlap, second_overlap = _assigned_speaker_with_reason(
            segment,
            turns,
            min_overlap_sec=min_overlap_sec,
            ambiguity_margin_sec=ambiguity_margin_sec,
        )
        reason_counts[reason] += 1
        if reason != "assigned" and len(fallback_samples) < max(0, int(fallback_sample_limit)):
            fallback_samples.append(
                {
                    "reason": reason,
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text[:120],
                    "best_overlap_sec": round(float(best_overlap), 3),
                    "second_overlap_sec": round(float(second_overlap), 3),
                    "assigned_speaker": str(speaker.speaker or "other"),
                    "assigned_speaker_id": str(speaker.speaker_id or "other"),
                    "assigned_speaker_role": str(speaker.speaker_role or speaker.speaker or "other"),
                }
            )
        utterances.append(
            SpeakerUtterance(
                start=segment.start,
                end=segment.end,
                speaker=speaker.speaker,
                speaker_id=speaker.speaker_id,
                speaker_role=speaker.speaker_role or speaker.speaker,
                diar_confidence=speaker.diar_confidence if speaker.diar_confidence is not None else speaker.confidence,
                role_confidence=speaker.role_confidence,
                text=text,
            )
        )
    merged = _merge_consecutive_utterances(utterances, merge_gap_sec=merge_gap_sec)
    merged_speaker_counts: Counter[str] = Counter(str(item.speaker or "other") for item in merged)
    fallback_total = sum(count for key, count in reason_counts.items() if key != "assigned")
    debug = {
        "segments_in": len(segments),
        "segments_processed": len(utterances),
        "segments_empty_text": int(empty_text_segments),
        "turns_in": len(turns),
        "utterances_out": len(merged),
        "reason_counts": dict(reason_counts),
        "fallback_total": int(fallback_total),
        "fallback_rate": round(float(fallback_total) / max(1, len(utterances)), 4),
        "merged_speaker_counts": dict(merged_speaker_counts),
        "fallback_samples": fallback_samples,
        "min_overlap_sec": float(min_overlap_sec),
        "ambiguity_margin_sec": float(ambiguity_margin_sec),
        "merge_gap_sec": float(merge_gap_sec),
    }
    return merged, debug
