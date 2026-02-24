from __future__ import annotations

"""
Maintain a session-scoped incremental transcript buffer for quasi-streaming UX.

Design intent:
- Accept repeated/overlapping ASR segments across chunk updates.
- Keep only new segments in append responses to reduce UI redraw noise.
- Preserve a traceable full segment timeline for downstream event extraction.
"""

import re
from dataclasses import dataclass
from typing import Sequence

from backend.asr.models import ASRSegment

_WS_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip()).lower()


@dataclass(frozen=True)
class IncrementalAppendResult:
    new_segments: list[ASRSegment]
    all_segments: list[ASRSegment]
    duplicates_dropped: int


class IncrementalTranscriptBuffer:
    def __init__(self, *, time_tolerance_sec: float = 0.15, max_segments: int = 5000) -> None:
        self._time_tolerance_sec = max(0.01, float(time_tolerance_sec))
        self._max_segments = max(100, int(max_segments))
        self._segments: list[ASRSegment] = []
        self._keys: set[tuple[int, int, str]] = set()

    def clear(self) -> None:
        self._segments = []
        self._keys = set()

    def append(self, incoming: Sequence[ASRSegment]) -> IncrementalAppendResult:
        new_segments: list[ASRSegment] = []
        duplicates = 0

        for segment in sorted(incoming, key=lambda item: (item.start, item.end)):
            normalized_text = _normalize_text(segment.text)
            if not normalized_text:
                continue

            key = self._segment_key(segment=segment, normalized_text=normalized_text)
            if key in self._keys:
                duplicates += 1
                continue

            self._keys.add(key)
            self._segments.append(segment)
            new_segments.append(segment)

        self._segments.sort(key=lambda item: (item.start, item.end))
        self._truncate_if_needed()
        return IncrementalAppendResult(
            new_segments=new_segments,
            all_segments=list(self._segments),
            duplicates_dropped=duplicates,
        )

    def _segment_key(self, *, segment: ASRSegment, normalized_text: str) -> tuple[int, int, str]:
        tick = self._time_tolerance_sec
        start_bucket = int(round(segment.start / tick))
        end_bucket = int(round(segment.end / tick))
        return (start_bucket, end_bucket, normalized_text)

    def _truncate_if_needed(self) -> None:
        if len(self._segments) <= self._max_segments:
            return
        self._segments = self._segments[-self._max_segments :]
        self._keys = {
            self._segment_key(segment=item, normalized_text=_normalize_text(item.text))
            for item in self._segments
        }

