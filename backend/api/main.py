from __future__ import annotations

"""
MVP API surface for Evidentia backend skeleton.

Design intent:
- Keep API orchestration thin and typed.
- Delegate domain logic to asr/events/risk/note modules.
- Return evidence-traceable payloads for reviewer trust.
"""

import base64
import copy
import json
import logging
import mimetypes
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from backend.asr.alignment import align_segments_to_turns, align_segments_to_turns_with_debug
from backend.asr.diarization_adapter import diarize_audio_with_debug
from backend.asr.formatting import format_for_display
from backend.asr.incremental_buffer import IncrementalTranscriptBuffer
from backend.asr.models import ASRSegment, SpeakerTurn, SpeakerUtterance
from backend.asr.role_mapping import apply_stable_role_mapping
from backend.asr.repo_chunk_adapter import (
    RepoASRAdapterError,
    transcribe_audio_window_with_repo_chunk_pipeline,
    transcribe_audio_with_repo_chunk_pipeline,
)
from backend.asr.structured_input import build_segments_and_turns_from_text, infer_turns_from_segments
from backend.events.extractor import (
    EventUtterance,
    apply_event_consistency_harmonization,
    apply_event_quality_guardrails,
    apply_rule_risk_backstop,
    extract_minimal_events,
)
from backend.events.medgemma_adapter import MedGemmaAdapterError, extract_events_with_medgemma
from backend.note.draft import (
    DraftEventEvidence,
    DraftProblem,
    DraftRiskFlag,
    build_note_drafts,
    get_note_template_document,
    list_note_templates,
    save_note_template_document,
)
from backend.risk.snapshot import SnapshotEvent, build_state_snapshot


class IncrementalTranscriptRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    segments: list[ASRSegment] = Field(default_factory=list)
    turns: list[SpeakerTurn] = Field(default_factory=list)
    split_sentences: bool = True
    reset: bool = False


class IncrementalTranscriptResponse(BaseModel):
    session_id: str
    new_segments: list[ASRSegment]
    all_segments: list[ASRSegment]
    utterances: list[SpeakerUtterance]
    transcript_text: str
    debug: dict[str, Any] = Field(default_factory=dict)


class TranscribeStructuredRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=128)
    audio_path: str | None = None
    language: str = Field(default="en", min_length=2, max_length=16)
    segments: list[ASRSegment] = Field(default_factory=list)
    turns: list[SpeakerTurn] = Field(default_factory=list)
    transcript_text: str | None = None
    start_at_sec: float = Field(default=0.0, ge=0.0)
    audio_window_sec: float | None = Field(default=None, gt=0.0, le=120.0)
    split_sentences: bool = True
    incremental: bool = True
    reset: bool = False
    asr_provider: Literal["whisper_cpp", "medasr_hf", "mock"] | None = None
    asr_chunk_sec: int | None = Field(default=None, ge=1, le=120)
    asr_overlap_sec: float | None = Field(default=None, ge=0.0, le=20.0)


class TranscribeStructuredResponse(BaseModel):
    session_id: str
    new_segments: list[ASRSegment] = Field(default_factory=list)
    segments: list[ASRSegment] = Field(default_factory=list)
    turns: list[SpeakerTurn] = Field(default_factory=list)
    new_utterances: list[SpeakerUtterance] = Field(default_factory=list)
    utterances: list[SpeakerUtterance] = Field(default_factory=list)
    transcript_text: str
    debug: dict[str, Any] = Field(default_factory=dict)


class EventExtractionUtteranceInput(BaseModel):
    segment_id: str = Field(min_length=1)
    t0: float = Field(ge=0.0)
    t1: float = Field(ge=0.0)
    speaker: str
    text: str = Field(min_length=1)
    asr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class EventEvidence(BaseModel):
    segment_id: str = Field(min_length=1)
    t0: float = Field(ge=0.0)
    t1: float = Field(ge=0.0)
    quote: str = Field(min_length=1)


class EventItem(BaseModel):
    event_id: str
    type: Literal["risk_cue", "symptom", "duration_onset"]
    label: str
    polarity: Literal["present", "absent", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    speaker: str
    evidence: EventEvidence


class EventsExtractRequest(BaseModel):
    utterances: list[EventExtractionUtteranceInput] = Field(default_factory=list)
    engine: Literal["auto", "medgemma", "rule"] = "auto"
    fallback_to_rule: bool = True
    medgemma_model_path: str | None = None
    medgemma_max_tokens: int = Field(default=512, ge=64, le=2048)
    medgemma_n_ctx: int = Field(default=2048, ge=512, le=8192)
    medgemma_n_gpu_layers: int = Field(default=-1, ge=-1, le=128)
    medgemma_n_threads: int | None = Field(default=None, ge=1, le=64)
    medgemma_chat_format: str | None = None


class EventsExtractResponse(BaseModel):
    events: list[EventItem]
    debug: dict[str, Any] = Field(default_factory=dict)


class StateProblemItem(BaseModel):
    item: str
    evidence_refs: list[str] = Field(default_factory=list)


class StateRiskFlag(BaseModel):
    level: Literal["low", "moderate", "high"]
    flag: str
    why: str
    evidence_refs: list[str] = Field(default_factory=list)


class StateSnapshotRequest(BaseModel):
    events: list[EventItem] = Field(default_factory=list)
    ai_enhancement_enabled: bool | None = None


class StateSnapshotResponse(BaseModel):
    problem_list: list[StateProblemItem] = Field(default_factory=list)
    risk_flags: list[StateRiskFlag] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    mandatory_safety_questions: list[str] = Field(default_factory=list)
    contextual_followups: list[str] = Field(default_factory=list)
    rationale: str = ""
    updated_at: str
    debug: dict[str, Any] = Field(default_factory=dict)


class NoteDraftSnapshotInput(BaseModel):
    problem_list: list[StateProblemItem] = Field(default_factory=list)
    risk_flags: list[StateRiskFlag] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    mandatory_safety_questions: list[str] = Field(default_factory=list)
    contextual_followups: list[str] = Field(default_factory=list)
    rationale: str = ""


class NoteCitation(BaseModel):
    anchor: str
    segment_id: str
    t0: float = Field(ge=0.0)
    t1: float = Field(ge=0.0)


class NoteDraftRequest(BaseModel):
    department: str = Field(default="psych", min_length=1, max_length=64)
    template_ids: list[str] = Field(min_length=1, max_length=16)
    patient_identity: str = Field(default="", max_length=256)
    patient_basic_info: str = Field(default="", max_length=8000)
    snapshot: NoteDraftSnapshotInput
    events: list[EventItem] = Field(default_factory=list)


class NoteDraftTemplateItem(BaseModel):
    template_id: str
    template_name: str
    note_text: str
    citations: list[NoteCitation] = Field(default_factory=list)


class NoteDraftResponse(BaseModel):
    note_type: str
    note_text: str
    citations: list[NoteCitation] = Field(default_factory=list)
    drafts: list[NoteDraftTemplateItem] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class NoteDraftJobStartRequest(BaseModel):
    department: str = Field(default="psych", min_length=1, max_length=64)
    template_ids: list[str] = Field(min_length=1, max_length=16)
    patient_identity: str = Field(default="", max_length=256)
    patient_basic_info: str = Field(default="", max_length=8000)
    snapshot: NoteDraftSnapshotInput
    events: list[EventItem] = Field(default_factory=list)


class NoteDraftJobResponse(BaseModel):
    job_id: str
    status: str
    stop_requested: bool = False
    department: str
    requested_template_ids: list[str] = Field(default_factory=list)
    missing_template_ids: list[str] = Field(default_factory=list)
    template_statuses: dict[str, str] = Field(default_factory=dict)
    drafts: list[NoteDraftTemplateItem] = Field(default_factory=list)
    error: str = ""
    created_at: str = ""
    updated_at: str = ""
    debug: dict[str, Any] = Field(default_factory=dict)


class NoteTemplateInfo(BaseModel):
    template_id: str
    template_name: str


class NoteTemplatesResponse(BaseModel):
    templates_by_department: dict[str, list[NoteTemplateInfo]] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)


class NoteTemplateDocument(BaseModel):
    template_id: str = Field(min_length=1, max_length=128)
    template_name: str = Field(min_length=1, max_length=200)
    template_text: str = Field(min_length=1, max_length=20000)


class NoteTemplateDetailResponse(BaseModel):
    department: str
    template_id: str
    template: NoteTemplateDocument
    debug: dict[str, Any] = Field(default_factory=dict)


class NoteTemplateUpdateRequest(BaseModel):
    template: NoteTemplateDocument


class AudioUploadResponse(BaseModel):
    audio_path: str
    filename: str
    size_bytes: int = Field(ge=1)
    debug: dict[str, Any] = Field(default_factory=dict)


class SampleAudioFileInfo(BaseModel):
    name: str
    audio_path: str
    size_bytes: int = Field(ge=0)


class SampleAudioListResponse(BaseModel):
    files: list[SampleAudioFileInfo] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class LiveAudioStatusResponse(BaseModel):
    session_id: str
    status: str
    chunks_received: int = Field(ge=0)
    bytes_received: int = Field(ge=0)
    last_seq: int | None = None
    sample_rate_hz: int | None = None
    mime_type: str = ""
    updated_at: str = ""
    debug: dict[str, Any] = Field(default_factory=dict)


class LiveAudioTranscriptResponse(BaseModel):
    session_id: str
    status: str
    new_utterances: list[SpeakerUtterance] = Field(default_factory=list)
    utterances: list[SpeakerUtterance] = Field(default_factory=list)
    transcript_text: str = ""
    updated_at: str = ""
    debug: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="evidentia backend service")
logger = logging.getLogger(__name__)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ROLE_CLINICIAN_SENT_RE = re.compile(
    r"\b(how|what|when|where|why|can you|could you|tell me|let us|let's|thanks for sharing|thank you for sharing)\b|\?",
    flags=re.IGNORECASE,
)
_ROLE_PATIENT_SENT_RE = re.compile(
    r"\b(i|i'm|i've|my|me|feel|feeling|sleep|anxious|depressed|pain|tired|wish i would not|wish i wouldn't)\b",
    flags=re.IGNORECASE,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_incremental_buffer_store() -> dict[str, IncrementalTranscriptBuffer]:
    existing = getattr(app.state, "incremental_transcript_buffers", None)
    if isinstance(existing, dict):
        return existing
    created: dict[str, IncrementalTranscriptBuffer] = {}
    setattr(app.state, "incremental_transcript_buffers", created)
    return created


def _get_transcribe_turn_store() -> dict[str, list[SpeakerTurn]]:
    existing = getattr(app.state, "transcribe_structured_turns", None)
    if isinstance(existing, dict):
        return existing
    created: dict[str, list[SpeakerTurn]] = {}
    setattr(app.state, "transcribe_structured_turns", created)
    return created


def _get_role_mapping_state_store() -> dict[str, dict[str, dict[str, Any]]]:
    existing = getattr(app.state, "transcribe_structured_role_mapping_state", None)
    if isinstance(existing, dict):
        return existing
    created: dict[str, dict[str, dict[str, Any]]] = {}
    setattr(app.state, "transcribe_structured_role_mapping_state", created)
    return created


def _alignment_debug_log_enabled() -> bool:
    raw = os.getenv("EVIDENTIA_ALIGNMENT_DEBUG_LOG", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _log_alignment_debug(tag: str, session_id: str, debug: dict[str, Any]) -> None:
    if not _alignment_debug_log_enabled():
        return
    try:
        logger.warning(
            "alignment_debug tag=%s session_id=%s fallback=%s/%s reasons=%s samples=%s",
            str(tag),
            str(session_id),
            int(debug.get("fallback_total", 0) or 0),
            int(debug.get("segments_processed", 0) or 0),
            dict(debug.get("reason_counts", {}) or {}),
            list(debug.get("fallback_samples", []) or [])[:3],
        )
    except Exception:
        # Logging must never break transcription flow.
        return


def _should_apply_sentence_role_split(
    *,
    source_debug: dict[str, Any] | None,
    diarization_debug: dict[str, Any] | None,
    turns: list[SpeakerTurn],
    utterances: list[SpeakerUtterance],
) -> tuple[bool, str]:
    source = str((source_debug or {}).get("source", "")).strip().lower()
    is_audio_path_source = source.startswith("audio_path")
    is_live_ws_source = source == "live_audio_ws"
    if not (is_audio_path_source or is_live_ws_source):
        return False, "non_audio_path_source"

    speaker_ids = {
        str(item.speaker_id or "").strip().lower()
        for item in turns
        if str(item.speaker_id or "").strip()
    }
    valid_ids = {
        sid
        for sid in speaker_ids
        if sid not in {"other", "patient", "clinician"}
    }
    if not valid_ids:
        return False, "no_valid_speaker_id"
    if len(valid_ids) != 1:
        return False, "multi_speaker_id"

    if not _has_mixed_sentence_role_cues(utterances):
        return False, "no_mixed_sentence_role_cues"

    status = str((diarization_debug or {}).get("status", "")).strip().lower()
    if status.startswith("payload_turns"):
        return False, "payload_turns_source"
    if is_live_ws_source:
        return True, "single_speaker_mixed_sentence_roles_live"
    return True, "single_speaker_mixed_sentence_roles"


def _score_sentence_role(text: str) -> str | None:
    source = str(text or "").strip()
    if not source:
        return None
    clinician_hits = len(_ROLE_CLINICIAN_SENT_RE.findall(source))
    patient_hits = len(_ROLE_PATIENT_SENT_RE.findall(source))
    if clinician_hits > patient_hits:
        return "clinician"
    if patient_hits > clinician_hits:
        return "patient"
    return None


def _split_mixed_role_utterances_by_sentence(
    utterances: list[SpeakerUtterance],
) -> tuple[list[SpeakerUtterance], dict[str, Any]]:
    out: list[SpeakerUtterance] = []
    split_applied = 0
    split_candidates = 0

    for item in utterances:
        text = str(item.text or "").strip()
        if not text:
            out.append(item)
            continue
        parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
        if len(parts) < 2:
            out.append(item)
            continue

        scored = [(part, _score_sentence_role(part)) for part in parts]
        roles_present = {role for _part, role in scored if role in {"patient", "clinician"}}
        if len(roles_present) < 2:
            out.append(item)
            continue

        split_candidates += 1
        total_dur = max(0.0, float(item.end) - float(item.start))
        if total_dur <= 0.05:
            out.append(item)
            continue

        # Use sentence text length as lightweight duration proxy.
        weights = [max(1, len(part)) for part, _role in scored]
        total_weight = float(sum(weights))
        cursor = float(item.start)
        split_items: list[SpeakerUtterance] = []
        for idx, (part, role) in enumerate(scored):
            if idx == len(scored) - 1:
                end = float(item.end)
            else:
                share = float(weights[idx]) / max(1.0, total_weight)
                end = min(float(item.end), cursor + (total_dur * share))
            if end <= cursor:
                end = min(float(item.end), cursor + 0.05)
            sentence_role = str(role or item.speaker_role or item.speaker or "other").strip().lower()
            if sentence_role not in {"patient", "clinician", "other"}:
                sentence_role = "other"
            split_items.append(
                SpeakerUtterance(
                    start=round(cursor, 3),
                    end=round(end, 3),
                    speaker=sentence_role,
                    speaker_id=item.speaker_id,
                    speaker_role=sentence_role,
                    diar_confidence=item.diar_confidence,
                    role_confidence=item.role_confidence,
                    text=part,
                )
            )
            cursor = end

        if split_items:
            out.extend(split_items)
            split_applied += 1
        else:
            out.append(item)

    return out, {
        "status": "ok",
        "utterances_in": len(utterances),
        "utterances_out": len(out),
        "split_candidates": split_candidates,
        "split_applied": split_applied,
    }


def _has_mixed_sentence_role_cues(utterances: list[SpeakerUtterance]) -> bool:
    for item in utterances:
        text = str(item.text or "").strip()
        if not text:
            continue
        parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
        if len(parts) < 2:
            continue
        roles = {_score_sentence_role(part) for part in parts}
        if "patient" in roles and "clinician" in roles:
            return True
    return False


def _turn_key(turn: SpeakerTurn) -> tuple[int, int, str, str]:
    return (
        int(round(turn.start * 10)),
        int(round(turn.end * 10)),
        turn.speaker.strip().lower(),
        str(turn.speaker_id or "").strip().lower(),
    )


def _merge_turns(existing: list[SpeakerTurn], incoming: list[SpeakerTurn]) -> list[SpeakerTurn]:
    seen = {_turn_key(item) for item in existing}
    merged = list(existing)
    for item in incoming:
        key = _turn_key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    merged.sort(key=lambda item: (item.start, item.end))
    return merged


def _overlaps_window(start: float, end: float, window_start: float, window_end: float) -> bool:
    return float(end) > float(window_start) and float(start) < float(window_end)


def _resolve_diarization_turns(
    payload: TranscribeStructuredRequest,
    *,
    segments: list[ASRSegment],
    fallback_turns: list[SpeakerTurn],
) -> tuple[list[SpeakerTurn], dict[str, Any]]:
    if payload.turns:
        turns = list(payload.turns)
        return turns, {"status": "payload_turns", "turns_count": len(turns)}

    window_start = float(payload.start_at_sec) if payload.audio_window_sec is not None else None
    window_end = (
        float(payload.start_at_sec) + float(payload.audio_window_sec)
        if payload.audio_window_sec is not None
        else None
    )

    diarize_callable = getattr(app.state, "transcribe_structured_diarize_callable", None)
    if diarize_callable is None:
        raw_turns, diar_debug = diarize_audio_with_debug(
            str(payload.audio_path or ""),
            asr_segments=segments,
            window_start_sec=window_start,
            window_end_sec=window_end,
        )
        try:
            turns = [SpeakerTurn.model_validate(item) for item in raw_turns]
        except ValidationError:
            turns = []
            diar_debug = {**dict(diar_debug), "status": "error", "reason": "invalid_default_diarization_output"}
        if turns:
            diar_debug["turns_count"] = len(turns)
            return turns, diar_debug
        if fallback_turns:
            return (
                list(fallback_turns),
                {
                    "status": "fallback_asr_turns",
                    "turns_count": len(fallback_turns),
                    "diarization_adapter": diar_debug,
                },
            )
        inferred = infer_turns_from_segments(segments)
        return (
            inferred,
            {
                "status": "fallback_inferred_from_segment_text",
                "turns_count": len(inferred),
                "diarization_adapter": diar_debug,
            },
        )

    try:
        raw_diarization = diarize_callable(str(payload.audio_path or ""))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if fallback_turns:
            return (
                list(fallback_turns),
                {
                    "status": "fallback_asr_turns_custom_diarize_error",
                    "turns_count": len(fallback_turns),
                    "reason": str(exc),
                },
            )
        inferred = infer_turns_from_segments(segments)
        return (
            inferred,
            {
                "status": "fallback_inferred_custom_diarize_error",
                "turns_count": len(inferred),
                "reason": str(exc),
            },
        )

    diarization_debug: dict[str, Any]
    if isinstance(raw_diarization, tuple) and len(raw_diarization) == 2:
        raw_turns, raw_debug = raw_diarization
        diarization_debug = dict(raw_debug or {})
    else:
        raw_turns = raw_diarization
        diarization_debug = {"status": "custom_callable_no_debug"}

    try:
        turns = [SpeakerTurn.model_validate(item) for item in raw_turns]
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid diarization output: {exc}") from exc

    if payload.audio_window_sec is not None:
        turns = [
            item
            for item in turns
            if _overlaps_window(item.start, item.end, float(window_start or 0.0), float(window_end or 0.0))
        ]
        diarization_debug["window_applied"] = True

    if turns:
        diarization_debug["turns_count"] = len(turns)
        return turns, diarization_debug

    if fallback_turns:
        return (
            list(fallback_turns),
            {
                "status": "fallback_asr_turns_custom_empty",
                "turns_count": len(fallback_turns),
                "custom_diarization": diarization_debug,
            },
        )
    inferred = infer_turns_from_segments(segments)
    return (
        inferred,
        {
            "status": "fallback_inferred_custom_empty",
            "turns_count": len(inferred),
            "custom_diarization": diarization_debug,
        },
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_audio_filename_stem(filename: str) -> str:
    raw_stem = Path(str(filename or "audio")).stem.strip()
    if not raw_stem:
        raw_stem = "audio"
    safe = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in raw_stem)
    safe = safe.strip("_")
    return (safe or "audio")[:64]


def _get_uploaded_audio_dir() -> Path:
    configured = getattr(app.state, "uploaded_audio_dir", None)
    base_dir = Path(str(configured)).expanduser() if configured else Path("/tmp/evidentia_uploads")
    resolved = base_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _get_sample_audio_dir() -> Path:
    configured = getattr(app.state, "sample_audio_dir", None)
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "samples").resolve()


def _get_live_audio_store() -> dict[str, dict[str, Any]]:
    existing = getattr(app.state, "live_audio_sessions", None)
    if isinstance(existing, dict):
        return existing
    created: dict[str, dict[str, Any]] = {}
    setattr(app.state, "live_audio_sessions", created)
    return created


def _get_note_draft_job_store() -> dict[str, dict[str, Any]]:
    existing = getattr(app.state, "note_draft_jobs", None)
    if isinstance(existing, dict):
        return existing
    created: dict[str, dict[str, Any]] = {}
    setattr(app.state, "note_draft_jobs", created)
    return created


def _get_note_draft_job_lock() -> threading.Lock:
    existing = getattr(app.state, "note_draft_job_lock", None)
    if isinstance(existing, type(threading.Lock())):
        return existing
    created = threading.Lock()
    setattr(app.state, "note_draft_job_lock", created)
    return created


def _draft_to_note_template_item(draft: Any) -> NoteDraftTemplateItem:
    return NoteDraftTemplateItem(
        template_id=str(getattr(draft, "template_id", "") or ""),
        template_name=str(getattr(draft, "template_name", "") or ""),
        note_text=str(getattr(draft, "note_text", "") or ""),
        citations=[
            NoteCitation(
                anchor=str(getattr(item, "anchor", "") or ""),
                segment_id=str(getattr(item, "segment_id", "") or ""),
                t0=float(getattr(item, "t0", 0.0) or 0.0),
                t1=float(getattr(item, "t1", 0.0) or 0.0),
            )
            for item in list(getattr(draft, "citations", []) or [])
        ],
    )


def _build_note_draft_batch_from_payload(
    *,
    department: str,
    template_ids: list[str],
    patient_identity: str,
    patient_basic_info: str,
    snapshot: NoteDraftSnapshotInput,
    events: list[EventItem],
):
    return build_note_drafts(
        department=department,
        template_ids=template_ids,
        patient_identity=patient_identity,
        patient_basic_info=patient_basic_info,
        problem_list=[
            DraftProblem(item=item.item, evidence_refs=item.evidence_refs)
            for item in snapshot.problem_list
        ],
        risk_flags=[
            DraftRiskFlag(
                level=item.level,
                flag=item.flag,
                why=item.why,
                evidence_refs=item.evidence_refs,
            )
            for item in snapshot.risk_flags
        ],
        open_questions=snapshot.open_questions,
        event_evidence=[
            DraftEventEvidence(
                segment_id=item.evidence.segment_id,
                t0=item.evidence.t0,
                t1=item.evidence.t1,
                quote=item.evidence.quote,
            )
            for item in events
        ],
    )


def _serialize_note_job_response(job: dict[str, Any]) -> NoteDraftJobResponse:
    template_statuses = {
        str(template_id): str(status)
        for template_id, status in dict(job.get("template_statuses", {}) or {}).items()
    }
    drafts = [
        NoteDraftTemplateItem.model_validate(item)
        for item in list(job.get("drafts", []) or [])
    ]
    return NoteDraftJobResponse(
        job_id=str(job.get("job_id", "")),
        status=str(job.get("status", "unknown")),
        stop_requested=bool(job.get("stop_requested", False)),
        department=str(job.get("department", "")),
        requested_template_ids=[
            str(item) for item in list(job.get("requested_template_ids", []) or [])
        ],
        missing_template_ids=[
            str(item) for item in list(job.get("missing_template_ids", []) or [])
        ],
        template_statuses=template_statuses,
        drafts=drafts,
        error=str(job.get("error", "")),
        created_at=str(job.get("created_at", "")),
        updated_at=str(job.get("updated_at", "")),
        debug=dict(job.get("debug", {}) or {}),
    )


def _run_note_draft_job(job_id: str) -> None:
    store = _get_note_draft_job_store()
    lock = _get_note_draft_job_lock()

    with lock:
        job = store.get(job_id)
        if not isinstance(job, dict):
            return
        job["status"] = "generating"
        job["updated_at"] = _utc_now_iso()

    while True:
        with lock:
            job = store.get(job_id)
            if not isinstance(job, dict):
                return
            selected_template_ids = list(job.get("selected_template_ids", []) or [])
            template_statuses = dict(job.get("template_statuses", {}) or {})
            next_template_id = ""
            for template_id in selected_template_ids:
                status = str(template_statuses.get(template_id, ""))
                if status in {"queued", "pending"}:
                    next_template_id = str(template_id)
                    break
            if not next_template_id:
                # No queued templates remain.
                if str(job.get("status", "")) != "failed":
                    job["status"] = "completed"
                job["updated_at"] = _utc_now_iso()
                return
            if bool(job.get("stop_requested", False)):
                for template_id in selected_template_ids:
                    if str(template_statuses.get(template_id, "")) in {"queued", "pending"}:
                        template_statuses[str(template_id)] = "stopped"
                job["template_statuses"] = template_statuses
                if str(job.get("status", "")) != "failed":
                    job["status"] = "stopped_partial"
                job["updated_at"] = _utc_now_iso()
                return

            template_statuses[next_template_id] = "generating"
            job["template_statuses"] = template_statuses
            job["status"] = "stopping" if bool(job.get("stop_requested", False)) else "generating"
            job["updated_at"] = _utc_now_iso()
            payload = dict(job.get("payload", {}) or {})
            department = str(job.get("department", "") or "")

        try:
            snapshot = NoteDraftSnapshotInput.model_validate(payload.get("snapshot", {}))
            events = [
                EventItem.model_validate(item)
                for item in list(payload.get("events", []) or [])
            ]
            batch = _build_note_draft_batch_from_payload(
                department=department,
                template_ids=[next_template_id],
                patient_identity=str(payload.get("patient_identity", "") or ""),
                patient_basic_info=str(payload.get("patient_basic_info", "") or ""),
                snapshot=snapshot,
                events=events,
            )
            if not batch.drafts:
                raise ValueError(f"No note draft generated for template={next_template_id}.")
            draft_item = _draft_to_note_template_item(batch.drafts[0]).model_dump()
        except Exception as exc:
            with lock:
                job = store.get(job_id)
                if isinstance(job, dict):
                    template_statuses = dict(job.get("template_statuses", {}) or {})
                    template_statuses[next_template_id] = "failed"
                    job["template_statuses"] = template_statuses
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    job["updated_at"] = _utc_now_iso()
            return

        with lock:
            job = store.get(job_id)
            if not isinstance(job, dict):
                return
            template_statuses = dict(job.get("template_statuses", {}) or {})
            template_statuses[next_template_id] = "generated"
            job["template_statuses"] = template_statuses

            drafts = list(job.get("drafts", []) or [])
            drafts = [item for item in drafts if str(item.get("template_id", "")) != next_template_id]
            drafts.append(draft_item)
            requested_order = list(job.get("selected_template_ids", []) or [])
            draft_index = {str(item.get("template_id", "")): item for item in drafts}
            ordered_drafts = [
                draft_index[item] for item in requested_order if item in draft_index
            ]
            job["drafts"] = ordered_drafts
            job["status"] = "stopping" if bool(job.get("stop_requested", False)) else "generating"
            job["updated_at"] = _utc_now_iso()


def _ensure_live_audio_session(store: dict[str, dict[str, Any]], session_id: str) -> dict[str, Any]:
    session = store.get(session_id)
    if isinstance(session, dict):
        return session
    created = {
        "session_id": session_id,
        "status": "idle",
        "chunks_received": 0,
        "bytes_received": 0,
        "last_seq": None,
        "sample_rate_hz": None,
        "mime_type": "",
        "asr_status": "not_configured",
        "asr_error": "",
        "last_diarization_status": "not_configured",
        "last_diarization_mode": "not_configured",
        "last_asr_latency_ms": None,
        "transcribed_chunks": 0,
        "total_segments": 0,
        "chunk_offset_sec": 0.0,
        "chunk_duration_sec": 1.0,
        "utterances": [],
        "transcript_text": "",
        "last_sentence_role_split": {"status": "skipped", "reason": "not_started"},
        "delivered_utterance_count": 0,
        "processed_chunk_seqs": set(),
        "processed_archive_chunk_seqs": set(),
        "recording_audio_path": "",
        "recording_audio_mime_type": "",
        "recording_chunks_received": 0,
        "recording_bytes_received": 0,
        "updated_at": _utc_now_iso(),
    }
    store[session_id] = created
    return created


def _reset_live_audio_session_runtime(session: dict[str, Any]) -> None:
    session["chunks_received"] = 0
    session["bytes_received"] = 0
    session["last_seq"] = None
    session["asr_error"] = ""
    session["asr_status"] = "not_configured"
    session["last_diarization_status"] = "not_configured"
    session["last_diarization_mode"] = "not_configured"
    session["last_asr_latency_ms"] = None
    session["transcribed_chunks"] = 0
    session["total_segments"] = 0
    session["chunk_offset_sec"] = 0.0
    session["chunk_duration_sec"] = 1.0
    session["utterances"] = []
    session["transcript_text"] = ""
    session["last_sentence_role_split"] = {"status": "skipped", "reason": "reset"}
    session["delivered_utterance_count"] = 0
    session["processed_chunk_seqs"] = set()
    session["processed_archive_chunk_seqs"] = set()
    existing_recording_path = Path(str(session.get("recording_audio_path", "") or "")).expanduser()
    try:
        if existing_recording_path.exists() and existing_recording_path.is_file():
            existing_recording_path.unlink(missing_ok=True)
    except Exception:
        # Reset path must not fail runtime cleanup.
        pass
    session["recording_audio_path"] = ""
    session["recording_audio_mime_type"] = ""
    session["recording_chunks_received"] = 0
    session["recording_bytes_received"] = 0
    session["updated_at"] = _utc_now_iso()


def _validate_live_asr_segments(raw_segments: Any) -> list[ASRSegment]:
    if raw_segments is None:
        return []
    if not isinstance(raw_segments, list):
        raise ValueError("live chunk ASR must return a list of ASR segments.")
    return [ASRSegment.model_validate(item) for item in raw_segments]


def _guess_audio_suffix_from_mime(mime_type: str | None) -> str:
    mt = str(mime_type or "").strip().lower()
    if "wav" in mt:
        return ".wav"
    if "mpeg" in mt or "mp3" in mt:
        return ".mp3"
    if "ogg" in mt:
        return ".ogg"
    if "webm" in mt:
        return ".webm"
    return ".webm"


def _get_live_recordings_dir() -> Path:
    path = _get_uploaded_audio_dir() / "live_recordings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_live_recording_path(
    session: dict[str, Any],
    *,
    session_id: str,
    mime_type: str | None,
) -> Path:
    preferred_mime = str(mime_type or session.get("recording_audio_mime_type") or session.get("mime_type") or "").strip()
    suffix = _guess_audio_suffix_from_mime(preferred_mime)
    existing = Path(str(session.get("recording_audio_path", "") or "")).expanduser()
    if existing.exists() and existing.is_file() and existing.suffix.lower() == suffix.lower():
        return existing
    if existing.exists() and existing.is_file():
        existing.unlink(missing_ok=True)

    stem = _sanitize_audio_filename_stem(session_id)
    output_name = f"{stem}_{uuid4().hex[:12]}{suffix}"
    output_path = _get_live_recordings_dir() / output_name
    output_path.touch()
    session["recording_audio_path"] = str(output_path)
    session["recording_audio_mime_type"] = preferred_mime
    return output_path


def _append_live_recording_chunk(
    session: dict[str, Any],
    *,
    session_id: str,
    chunk_bytes: bytes,
    mime_type: str | None,
) -> str:
    output_path = _ensure_live_recording_path(
        session,
        session_id=session_id,
        mime_type=mime_type,
    )
    with output_path.open("ab") as handle:
        handle.write(chunk_bytes)
    session["recording_audio_path"] = str(output_path)
    session["recording_chunks_received"] = int(session.get("recording_chunks_received", 0) or 0) + 1
    session["recording_bytes_received"] = int(session.get("recording_bytes_received", 0) or 0) + len(chunk_bytes)
    return str(output_path)


def _default_live_audio_chunk_asr(
    chunk_bytes: bytes,
    *,
    session_id: str,
    seq: int | None,
    sample_rate_hz: int | None,
    mime_type: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _ = sample_rate_hz
    suffix = _guess_audio_suffix_from_mime(mime_type)
    upload_dir = _get_uploaded_audio_dir() / "live_chunks"
    upload_dir.mkdir(parents=True, exist_ok=True)
    seq_token = f"{int(seq):06d}" if isinstance(seq, int) and seq >= 0 else "na"
    tmp_path = upload_dir / f"{_sanitize_audio_filename_stem(session_id)}_{seq_token}{suffix}"
    tmp_path.write_bytes(chunk_bytes)

    language = str(getattr(app.state, "live_audio_language", "en") or "en")
    provider = getattr(app.state, "live_audio_asr_provider", None)
    chunk_sec = getattr(app.state, "live_audio_asr_chunk_sec", None)
    overlap_sec = getattr(app.state, "live_audio_asr_overlap_sec", None)
    try:
        repo_result = transcribe_audio_with_repo_chunk_pipeline(
            audio_path=str(tmp_path),
            language=language,
            provider=provider,
            chunk_sec=chunk_sec,
            overlap_sec=overlap_sec,
        )
    except RepoASRAdapterError:
        raise
    finally:
        tmp_path.unlink(missing_ok=True)

    duration_hint = max((item.end for item in repo_result.segments), default=1.0)
    return (
        [item.model_dump() for item in repo_result.segments],
        {
            **repo_result.debug,
            "asr_mode": "default_repo_adapter",
            "chunk_duration_sec": float(max(0.2, duration_hint)),
            "language": language,
        },
    )


def _default_live_audio_chunk_diarize(
    chunk_bytes: bytes,
    *,
    session_id: str,
    seq: int | None,
    sample_rate_hz: int | None,
    mime_type: str | None,
    asr_segments: list[ASRSegment] | None = None,
    window_duration_sec: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _ = sample_rate_hz
    suffix = _guess_audio_suffix_from_mime(mime_type)
    upload_dir = _get_uploaded_audio_dir() / "live_chunks_diar"
    upload_dir.mkdir(parents=True, exist_ok=True)
    seq_token = f"{int(seq):06d}" if isinstance(seq, int) and seq >= 0 else "na"
    tmp_path = upload_dir / f"{_sanitize_audio_filename_stem(session_id)}_{seq_token}{suffix}"
    tmp_path.write_bytes(chunk_bytes)
    try:
        turns, diar_debug = diarize_audio_with_debug(
            str(tmp_path),
            asr_segments=list(asr_segments or []),
            window_start_sec=0.0 if window_duration_sec is not None else None,
            window_end_sec=float(window_duration_sec) if window_duration_sec is not None else None,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return [item.model_dump() for item in turns], {
        **dict(diar_debug or {}),
        "diarization_mode": "default_local_adapter",
    }


def _run_live_chunk_diarization(
    session_id: str,
    session: dict[str, Any],
    *,
    chunk_bytes: bytes,
    seq: int | None,
    chunk_segments: list[ASRSegment],
    chunk_offset_sec: float,
    window_duration_sec: float | None = None,
) -> tuple[list[SpeakerTurn], dict[str, Any]]:
    diarize_callable = getattr(app.state, "live_audio_chunk_diarize_callable", None)
    default_enabled = bool(getattr(app.state, "live_audio_default_diarization_enabled", True))
    diar_mode = "injected_callable"
    if diarize_callable is None and default_enabled:
        diarize_callable = _default_live_audio_chunk_diarize
        diar_mode = "default_local_adapter"
    elif diarize_callable is None:
        return [], {"status": "not_configured", "mode": "not_configured", "turns_count": 0}

    try:
        try:
            raw_result = diarize_callable(
                chunk_bytes,
                session_id=session_id,
                seq=seq,
                sample_rate_hz=session.get("sample_rate_hz"),
                mime_type=session.get("mime_type"),
                asr_segments=chunk_segments,
                window_duration_sec=window_duration_sec,
            )
        except TypeError:
            try:
                raw_result = diarize_callable(chunk_bytes, asr_segments=chunk_segments)
            except TypeError:
                raw_result = diarize_callable(chunk_bytes)
    except Exception as exc:
        return (
            [],
            {
                "status": "error",
                "reason": str(exc),
                "mode": diar_mode,
                "turns_count": 0,
            },
        )

    diar_debug: dict[str, Any]
    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        raw_turns, raw_debug = raw_result
        diar_debug = dict(raw_debug or {})
    else:
        raw_turns = raw_result
        diar_debug = {"status": "custom_callable_no_debug"}

    try:
        local_turns = [SpeakerTurn.model_validate(item) for item in list(raw_turns or [])]
    except ValidationError as exc:
        return (
            [],
            {
                "status": "error",
                "reason": f"invalid_live_diarization_output: {exc}",
                "mode": diar_mode,
                "turns_count": 0,
            },
        )

    shifted_turns: list[SpeakerTurn] = []
    for item in local_turns:
        shifted_turns.append(
            SpeakerTurn(
                start=max(0.0, item.start + chunk_offset_sec),
                end=max(0.0, item.end + chunk_offset_sec),
                speaker=item.speaker or "other",
                speaker_id=item.speaker_id or item.speaker or "other",
                speaker_role=item.speaker_role or item.speaker or "other",
                confidence=item.confidence,
                diar_confidence=item.diar_confidence if item.diar_confidence is not None else item.confidence,
            )
        )

    return (
        shifted_turns,
        {
            **diar_debug,
            "mode": diar_mode,
            "turns_count": len(shifted_turns),
        },
    )


def _run_live_chunk_asr(
    session_id: str,
    session: dict[str, Any],
    *,
    chunk_bytes: bytes,
    seq: int | None,
    window_start_sec: float | None = None,
    window_duration_sec: float | None = None,
) -> dict[str, Any]:
    asr_callable = getattr(app.state, "live_audio_chunk_asr_callable", None)
    default_enabled = bool(getattr(app.state, "live_audio_default_asr_enabled", True))
    asr_mode = "injected_callable"
    if asr_callable is None and default_enabled:
        asr_callable = _default_live_audio_chunk_asr
        asr_mode = "default_repo_adapter"
    elif asr_callable is None:
        asr_mode = "not_configured"

    if asr_callable is None:
        session["asr_status"] = str(asr_mode)
        session["asr_error"] = ""
        session["last_diarization_status"] = "not_configured"
        session["last_diarization_mode"] = "not_configured"
        session["last_sentence_role_split"] = {"status": "skipped", "reason": "asr_not_configured"}
        session["last_asr_latency_ms"] = None
        return {
            "asr_status": str(asr_mode),
            "asr_error": "",
            "new_segments": 0,
            "total_segments": int(session.get("total_segments", 0) or 0),
            "new_utterances": 0,
            "total_utterances": len(session.get("utterances", [])),
            "asr_latency_ms": None,
            "asr_mode": str(asr_mode),
            "utterances_payload": [],
            "new_utterances_payload": [],
            "transcript_text": str(session.get("transcript_text", "") or ""),
            "sentence_role_split": dict(session.get("last_sentence_role_split", {}) or {}),
        }

    call_started = time.perf_counter()
    try:
        try:
            raw_result = asr_callable(
                chunk_bytes,
                session_id=session_id,
                seq=seq,
                sample_rate_hz=session.get("sample_rate_hz"),
                mime_type=session.get("mime_type"),
            )
        except TypeError:
            raw_result = asr_callable(chunk_bytes)
    except Exception as exc:
        elapsed = round((time.perf_counter() - call_started) * 1000, 2)
        session["asr_status"] = "error"
        session["asr_error"] = str(exc)
        session["last_diarization_status"] = "skipped_asr_error"
        session["last_diarization_mode"] = "skipped"
        session["last_sentence_role_split"] = {"status": "skipped", "reason": "asr_error"}
        session["last_asr_latency_ms"] = elapsed
        return {
            "asr_status": "error",
            "asr_error": str(exc),
            "new_segments": 0,
            "total_segments": int(session.get("total_segments", 0) or 0),
            "new_utterances": 0,
            "total_utterances": len(session.get("utterances", [])),
            "asr_latency_ms": elapsed,
            "asr_mode": str(asr_mode),
            "utterances_payload": [],
            "new_utterances_payload": [],
            "transcript_text": str(session.get("transcript_text", "") or ""),
            "sentence_role_split": dict(session.get("last_sentence_role_split", {}) or {}),
        }

    asr_debug: dict[str, Any]
    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        raw_segments, raw_debug = raw_result
        asr_debug = dict(raw_debug or {})
    else:
        raw_segments = raw_result
        asr_debug = {}

    elapsed = round((time.perf_counter() - call_started) * 1000, 2)
    try:
        chunk_segments = _validate_live_asr_segments(raw_segments)
    except Exception as exc:
        session["asr_status"] = "error"
        session["asr_error"] = f"invalid_live_asr_output: {exc}"
        session["last_diarization_status"] = "skipped_asr_error"
        session["last_diarization_mode"] = "skipped"
        session["last_sentence_role_split"] = {"status": "skipped", "reason": "invalid_asr_output"}
        session["last_asr_latency_ms"] = elapsed
        return {
            "asr_status": "error",
            "asr_error": str(session["asr_error"]),
            "new_segments": 0,
            "total_segments": int(session.get("total_segments", 0) or 0),
            "new_utterances": 0,
            "total_utterances": len(session.get("utterances", [])),
            "asr_latency_ms": elapsed,
            "asr_mode": str(asr_mode),
            "utterances_payload": [],
            "new_utterances_payload": [],
            "transcript_text": str(session.get("transcript_text", "") or ""),
            "sentence_role_split": dict(session.get("last_sentence_role_split", {}) or {}),
        }

    explicit_window_start = (
        float(window_start_sec)
        if isinstance(window_start_sec, (int, float)) and float(window_start_sec) >= 0.0
        else None
    )
    chunk_offset_sec = (
        explicit_window_start
        if explicit_window_start is not None
        else float(session.get("chunk_offset_sec", 0.0) or 0.0)
    )
    offset_segments = [
        ASRSegment(
            start=max(0.0, item.start + chunk_offset_sec),
            end=max(0.0, item.end + chunk_offset_sec),
            text=item.text,
            avg_logprob=item.avg_logprob,
            no_speech_prob=item.no_speech_prob,
        )
        for item in chunk_segments
    ]

    duration_hint = asr_debug.get("chunk_duration_sec")
    if isinstance(window_duration_sec, (int, float)) and float(window_duration_sec) > 0.0:
        duration_hint = float(window_duration_sec)
    if not isinstance(duration_hint, (int, float)) or float(duration_hint) <= 0.0:
        if chunk_segments:
            duration_hint = max(0.2, max(item.end for item in chunk_segments))
        else:
            duration_hint = float(session.get("chunk_duration_sec", 1.0) or 1.0)
    duration_sec = float(duration_hint)
    session["chunk_duration_sec"] = duration_sec
    if explicit_window_start is not None:
        session["chunk_offset_sec"] = max(
            float(session.get("chunk_offset_sec", 0.0) or 0.0),
            chunk_offset_sec + duration_sec,
        )
    else:
        session["chunk_offset_sec"] = chunk_offset_sec + duration_sec

    segment_store = _get_incremental_buffer_store()
    turn_store = _get_transcribe_turn_store()
    role_state_store = _get_role_mapping_state_store()
    buffer = segment_store.get(session_id)
    if buffer is None:
        buffer = IncrementalTranscriptBuffer()
        segment_store[session_id] = buffer
    append_result = buffer.append(offset_segments)
    incoming_turns, diarization_debug = _run_live_chunk_diarization(
        session_id,
        session,
        chunk_bytes=chunk_bytes,
        seq=seq,
        chunk_segments=chunk_segments,
        chunk_offset_sec=chunk_offset_sec,
        window_duration_sec=duration_sec,
    )
    if not incoming_turns:
        incoming_turns = infer_turns_from_segments(offset_segments)
        diarization_debug = {
            **dict(diarization_debug or {}),
            "status": str(diarization_debug.get("status") or "fallback_inferred"),
            "fallback": "infer_turns_from_segments",
            "turns_count": len(incoming_turns),
        }
    all_turns = _merge_turns(turn_store.get(session_id, []), incoming_turns)
    try:
        all_turns, next_role_state, role_mapping_debug = apply_stable_role_mapping(
            all_turns,
            append_result.new_segments,
            state=role_state_store.get(session_id, {}),
        )
        turn_store[session_id] = all_turns
        role_state_store[session_id] = next_role_state
        diarization_debug["role_mapping"] = role_mapping_debug
    except Exception as exc:
        turn_store[session_id] = all_turns
        diarization_debug["role_mapping"] = {"status": "error", "reason": str(exc)}

    utterances, align_debug_all = align_segments_to_turns_with_debug(
        append_result.all_segments,
        all_turns,
        merge_gap_sec=0.0,
    )
    new_utterances, align_debug_new = align_segments_to_turns_with_debug(
        append_result.new_segments,
        all_turns,
        merge_gap_sec=0.0,
    )
    sentence_role_split_debug: dict[str, Any] = {"status": "skipped", "reason": "unknown"}
    should_split, split_reason = _should_apply_sentence_role_split(
        source_debug={"source": "live_audio_ws"},
        diarization_debug=diarization_debug if isinstance(diarization_debug, dict) else {},
        turns=all_turns,
        utterances=utterances,
    )
    if should_split:
        utterances, split_all = _split_mixed_role_utterances_by_sentence(utterances)
        new_utterances, split_new = _split_mixed_role_utterances_by_sentence(new_utterances)
        sentence_role_split_debug = {
            "status": "applied",
            "condition": split_reason,
            "all": split_all,
            "new": split_new,
        }
    else:
        sentence_role_split_debug = {"status": "skipped", "reason": split_reason}
    diarization_debug["alignment"] = {"all": align_debug_all, "new": align_debug_new}
    diarization_debug["sentence_role_split"] = sentence_role_split_debug
    _log_alignment_debug("live_all", session_id, align_debug_all)
    _log_alignment_debug("live_new", session_id, align_debug_new)
    transcript_text = format_for_display(utterances, split_sentences=True)
    delivered = int(session.get("delivered_utterance_count", 0) or 0)
    delivered = max(0, min(delivered, len(utterances)))
    new_utterances_count = len(utterances) - delivered

    session["asr_status"] = "ok"
    session["asr_error"] = ""
    session["last_asr_latency_ms"] = elapsed
    session["last_diarization_status"] = str(diarization_debug.get("status", "unknown"))
    session["last_diarization_mode"] = str(diarization_debug.get("mode", "unknown"))
    session["transcribed_chunks"] = int(session.get("transcribed_chunks", 0) or 0) + 1
    session["total_segments"] = len(append_result.all_segments)
    session["utterances"] = [item.model_dump() for item in utterances]
    session["transcript_text"] = transcript_text
    session["last_sentence_role_split"] = sentence_role_split_debug

    return {
        "asr_status": "ok",
        "asr_error": "",
        "new_segments": len(append_result.new_segments),
        "total_segments": len(append_result.all_segments),
        "new_utterances": new_utterances_count,
        "total_utterances": len(utterances),
        "asr_latency_ms": elapsed,
        "asr_mode": str(asr_mode),
        "asr_debug": asr_debug,
        "diarization": diarization_debug,
        "sentence_role_split": sentence_role_split_debug,
        "utterances_payload": [item.model_dump() for item in utterances],
        "new_utterances_payload": [item.model_dump() for item in new_utterances],
        "transcript_text": transcript_text,
    }


def _resolve_structured_source(
    payload: TranscribeStructuredRequest,
) -> tuple[list[ASRSegment], list[SpeakerTurn], dict[str, Any]]:
    if payload.segments:
        return (
            list(payload.segments),
            list(payload.turns),
            {"source": "segments_payload", "asr": {"status": "bypassed"}},
        )

    if payload.transcript_text and payload.transcript_text.strip():
        segments, inferred_turns = build_segments_and_turns_from_text(
            payload.transcript_text,
            start_at_sec=payload.start_at_sec,
        )
        turns = list(payload.turns) if payload.turns else inferred_turns
        return (
            segments,
            turns,
            {
                "source": "transcript_text",
                "asr": {"status": "simulated_from_text"},
                "diarization": {
                    "status": "inferred_from_text_prefix",
                    "turns_count": len(turns),
                },
            },
        )

    if payload.audio_path:
        asr_callable = getattr(app.state, "transcribe_structured_asr_callable", None)
        if asr_callable is None:
            try:
                if payload.audio_window_sec is not None:
                    repo_result = transcribe_audio_window_with_repo_chunk_pipeline(
                        audio_path=payload.audio_path,
                        language=payload.language,
                        start_at_sec=payload.start_at_sec,
                        window_sec=payload.audio_window_sec,
                        provider=payload.asr_provider,
                        chunk_sec=payload.asr_chunk_sec,
                        overlap_sec=payload.asr_overlap_sec,
                    )
                    source = "audio_path_window"
                    asr_status = "repo_chunk_pipeline_window"
                else:
                    repo_result = transcribe_audio_with_repo_chunk_pipeline(
                        audio_path=payload.audio_path,
                        language=payload.language,
                        provider=payload.asr_provider,
                        chunk_sec=payload.asr_chunk_sec,
                        overlap_sec=payload.asr_overlap_sec,
                    )
                    source = "audio_path"
                    asr_status = "repo_chunk_pipeline"
            except RepoASRAdapterError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            turns, diarization_debug = _resolve_diarization_turns(
                payload,
                segments=list(repo_result.segments),
                fallback_turns=list(repo_result.turns),
            )
            return (
                repo_result.segments,
                turns,
                {
                    "source": source,
                    "asr": {
                        "status": asr_status,
                        **repo_result.debug,
                    },
                    "diarization": diarization_debug,
                },
            )

        try:
            try:
                raw_asr = asr_callable(
                    payload.audio_path,
                    language=payload.language,
                    start_at_sec=payload.start_at_sec,
                    window_sec=payload.audio_window_sec,
                )
            except TypeError:
                try:
                    raw_asr = asr_callable(payload.audio_path, language=payload.language)
                except TypeError:
                    raw_asr = asr_callable(payload.audio_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ASR adapter failed: {exc}") from exc

        asr_debug: dict[str, Any]
        if isinstance(raw_asr, tuple) and len(raw_asr) == 2:
            raw_segments, raw_debug = raw_asr
            asr_debug = dict(raw_debug or {})
        else:
            raw_segments = raw_asr
            asr_debug = {"status": "custom_callable_no_debug"}

        try:
            segments = [ASRSegment.model_validate(item) for item in raw_segments]
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid ASR output: {exc}") from exc

        if payload.audio_window_sec is not None:
            window_start = float(payload.start_at_sec)
            window_end = window_start + float(payload.audio_window_sec)
            segments = [item for item in segments if _overlaps_window(item.start, item.end, window_start, window_end)]
            asr_debug.setdefault(
                "stream",
                {
                    "start_at_sec": window_start,
                    "window_sec": float(payload.audio_window_sec),
                    "window_end_sec": window_end,
                    "next_start_sec": window_end,
                    "audio_duration_sec": None,
                    "has_more": True,
                },
            )

        turns, diarization_debug = _resolve_diarization_turns(
            payload,
            segments=segments,
            fallback_turns=[],
        )

        return (
            segments,
            turns,
            {
                "source": "audio_path_window_custom" if payload.audio_window_sec is not None else "audio_path",
                "asr": asr_debug,
                "diarization": diarization_debug,
            },
        )

    if payload.reset:
        return [], [], {"source": "reset_only"}

    raise HTTPException(
        status_code=400,
        detail="Provide one of: segments, transcript_text, or audio_path.",
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/files/upload-audio", response_model=AudioUploadResponse)
async def upload_audio(
    request: Request,
    filename: str = Query(min_length=1, max_length=255),
) -> AudioUploadResponse:
    filename = Path(str(filename or "")).name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(filename).suffix.lower()
    allowed_suffixes = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4"}
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail="Only .wav, .mp3, .webm, .ogg, .m4a, or .mp4 files are accepted.",
        )

    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_bytes = 200 * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds 200MB limit.")

    output_dir = _get_uploaded_audio_dir()
    stem = _sanitize_audio_filename_stem(filename)
    output_name = f"{stem}_{uuid4().hex[:10]}{suffix}"
    output_path = output_dir / output_name
    output_path.write_bytes(payload)

    return AudioUploadResponse(
        audio_path=str(output_path),
        filename=filename,
        size_bytes=len(payload),
        debug={
            "content_type": str(request.headers.get("content-type", "")),
            "saved_to": str(output_dir),
        },
    )


@app.get("/files/sample-audio", response_model=SampleAudioListResponse)
async def list_sample_audio_files() -> SampleAudioListResponse:
    sample_dir = _get_sample_audio_dir()
    allowed_suffixes = {".wav", ".mp3"}
    items: list[SampleAudioFileInfo] = []

    if sample_dir.exists() and sample_dir.is_dir():
        for path in sample_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            try:
                size_bytes = int(path.stat().st_size)
            except OSError:
                continue
            items.append(
                SampleAudioFileInfo(
                    name=path.name,
                    audio_path=str(path.resolve()),
                    size_bytes=size_bytes,
                )
            )

    items.sort(key=lambda item: (item.size_bytes, item.name.lower()))
    return SampleAudioListResponse(
        files=items,
        debug={
            "sample_dir": str(sample_dir),
            "count": len(items),
        },
    )


@app.get("/files/audio")
async def get_audio_file(
    audio_path: str = Query(default=""),
) -> FileResponse:
    normalized_path = str(audio_path or "").strip()
    if not normalized_path:
        raise HTTPException(status_code=400, detail="audio_path is required.")

    try:
        resolved_path = Path(normalized_path).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio_path: {exc}") from exc

    if not resolved_path.exists() or not resolved_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")

    media_type, _ = mimetypes.guess_type(str(resolved_path))
    return FileResponse(
        path=str(resolved_path),
        filename=resolved_path.name,
        media_type=media_type or "application/octet-stream",
    )


@app.get("/audio/live/status/{session_id}", response_model=LiveAudioStatusResponse)
async def live_audio_status(session_id: str) -> LiveAudioStatusResponse:
    normalized_session = str(session_id or "").strip()
    if not normalized_session:
        raise HTTPException(status_code=400, detail="session_id is required.")
    store = _get_live_audio_store()
    session = _ensure_live_audio_session(store, normalized_session)
    return LiveAudioStatusResponse(
        session_id=normalized_session,
        status=str(session.get("status", "idle")),
        chunks_received=int(session.get("chunks_received", 0) or 0),
        bytes_received=int(session.get("bytes_received", 0) or 0),
        last_seq=session.get("last_seq"),
        sample_rate_hz=session.get("sample_rate_hz"),
        mime_type=str(session.get("mime_type", "")),
        updated_at=str(session.get("updated_at", "")),
        debug={
            "store_size": len(store),
            "asr_status": str(session.get("asr_status", "not_configured")),
            "asr_error": str(session.get("asr_error", "")),
            "recording_audio_path": str(session.get("recording_audio_path", "") or ""),
            "recording_audio_mime_type": str(session.get("recording_audio_mime_type", "") or ""),
            "recording_chunks_received": int(session.get("recording_chunks_received", 0) or 0),
            "recording_bytes_received": int(session.get("recording_bytes_received", 0) or 0),
            "diarization_status": str(session.get("last_diarization_status", "")),
            "diarization_mode": str(session.get("last_diarization_mode", "")),
            "last_asr_latency_ms": session.get("last_asr_latency_ms"),
            "transcribed_chunks": int(session.get("transcribed_chunks", 0) or 0),
            "total_segments": int(session.get("total_segments", 0) or 0),
            "total_utterances": len(session.get("utterances", [])),
            "asr_mode": "injected_callable"
            if callable(getattr(app.state, "live_audio_chunk_asr_callable", None))
            else (
                "default_repo_adapter"
                if bool(getattr(app.state, "live_audio_default_asr_enabled", True))
                else "not_configured"
            ),
        },
    )


@app.get("/audio/live/transcript/{session_id}", response_model=LiveAudioTranscriptResponse)
async def live_audio_transcript(
    session_id: str,
    split_sentences: bool = Query(default=True),
) -> LiveAudioTranscriptResponse:
    normalized_session = str(session_id or "").strip()
    if not normalized_session:
        raise HTTPException(status_code=400, detail="session_id is required.")
    store = _get_live_audio_store()
    session = _ensure_live_audio_session(store, normalized_session)

    utterances_data = session.get("utterances", [])
    utterances = [SpeakerUtterance.model_validate(item) for item in utterances_data] if isinstance(utterances_data, list) else []
    delivered = int(session.get("delivered_utterance_count", 0) or 0)
    delivered = max(0, min(delivered, len(utterances)))
    new_utterances = utterances[delivered:]
    session["delivered_utterance_count"] = len(utterances)
    updated_at = _utc_now_iso()
    session["updated_at"] = updated_at

    transcript_text = str(session.get("transcript_text", "") or "")
    if not transcript_text and utterances:
        transcript_text = format_for_display(utterances, split_sentences=split_sentences)

    return LiveAudioTranscriptResponse(
        session_id=normalized_session,
        status=str(session.get("status", "idle")),
        new_utterances=new_utterances,
        utterances=utterances,
        transcript_text=transcript_text,
        updated_at=updated_at,
        debug={
            "asr_status": str(session.get("asr_status", "not_configured")),
            "asr_error": str(session.get("asr_error", "")),
            "recording_audio_path": str(session.get("recording_audio_path", "") or ""),
            "recording_audio_mime_type": str(session.get("recording_audio_mime_type", "") or ""),
            "recording_chunks_received": int(session.get("recording_chunks_received", 0) or 0),
            "recording_bytes_received": int(session.get("recording_bytes_received", 0) or 0),
            "diarization_status": str(session.get("last_diarization_status", "")),
            "diarization_mode": str(session.get("last_diarization_mode", "")),
            "sentence_role_split": session.get("last_sentence_role_split", {}),
            "last_asr_latency_ms": session.get("last_asr_latency_ms"),
            "transcribed_chunks": int(session.get("transcribed_chunks", 0) or 0),
            "total_segments": int(session.get("total_segments", 0) or 0),
            "total_utterances": len(utterances),
            "new_utterances_count": len(new_utterances),
        },
    )


@app.websocket("/ws/audio/live")
async def live_audio_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(websocket.query_params.get("session_id", "")).strip()
    if not session_id:
        await websocket.send_json({"type": "error", "detail": "session_id is required."})
        await websocket.close(code=1008)
        return

    store = _get_live_audio_store()
    session = _ensure_live_audio_session(store, session_id)
    session["status"] = "connected"
    session["updated_at"] = _utc_now_iso()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "detail": "invalid_json"})
                continue

            message_type = str(payload.get("type", "")).strip().lower()
            if message_type == "start":
                if payload.get("reset", True):
                    _get_incremental_buffer_store().pop(session_id, None)
                    _get_transcribe_turn_store().pop(session_id, None)
                    _get_role_mapping_state_store().pop(session_id, None)
                    _reset_live_audio_session_runtime(session)
                session["status"] = "streaming"
                sample_rate_hz = payload.get("sample_rate_hz")
                if isinstance(sample_rate_hz, int) and sample_rate_hz > 0:
                    session["sample_rate_hz"] = sample_rate_hz
                mime_type = str(payload.get("mime_type", "")).strip()
                if mime_type:
                    session["mime_type"] = mime_type
                session["updated_at"] = _utc_now_iso()
                await websocket.send_json(
                    {
                        "type": "ack_start",
                        "session_id": session_id,
                        "status": session["status"],
                        "recording_audio_path": str(session.get("recording_audio_path", "") or ""),
                    }
                )
                continue

            if message_type == "audio_chunk_archive":
                data_b64 = str(payload.get("data_b64", "")).strip()
                if not data_b64:
                    await websocket.send_json({"type": "error", "detail": "missing_data_b64"})
                    continue
                try:
                    chunk_bytes = base64.b64decode(data_b64, validate=True)
                except Exception:
                    await websocket.send_json({"type": "error", "detail": "invalid_base64"})
                    continue

                seq_raw = payload.get("seq")
                seq = int(seq_raw) if isinstance(seq_raw, int) or (isinstance(seq_raw, str) and seq_raw.isdigit()) else None
                if seq is not None:
                    session["last_seq"] = seq

                mime_type = str(payload.get("mime_type", "")).strip()
                if mime_type:
                    session["mime_type"] = mime_type

                processed_archive_seqs = session.get("processed_archive_chunk_seqs")
                if not isinstance(processed_archive_seqs, set):
                    processed_archive_seqs = set()
                    session["processed_archive_chunk_seqs"] = processed_archive_seqs
                if seq is not None and seq in processed_archive_seqs:
                    session["updated_at"] = _utc_now_iso()
                    await websocket.send_json(
                        {
                            "type": "ack_archive_chunk",
                            "session_id": session_id,
                            "seq": seq,
                            "duplicate": True,
                            "recording_audio_path": str(session.get("recording_audio_path", "") or ""),
                            "recording_chunks_received": int(session.get("recording_chunks_received", 0) or 0),
                            "recording_bytes_received": int(session.get("recording_bytes_received", 0) or 0),
                        }
                    )
                    continue

                recording_audio_path = _append_live_recording_chunk(
                    session,
                    session_id=session_id,
                    chunk_bytes=chunk_bytes,
                    mime_type=mime_type,
                )
                if seq is not None:
                    processed_archive_seqs.add(seq)
                session["updated_at"] = _utc_now_iso()
                await websocket.send_json(
                    {
                        "type": "ack_archive_chunk",
                        "session_id": session_id,
                        "seq": seq,
                        "duplicate": False,
                        "recording_audio_path": recording_audio_path,
                        "recording_chunks_received": int(session.get("recording_chunks_received", 0) or 0),
                        "recording_bytes_received": int(session.get("recording_bytes_received", 0) or 0),
                    }
                )
                continue

            if message_type == "audio_chunk":
                data_b64 = str(payload.get("data_b64", "")).strip()
                if not data_b64:
                    await websocket.send_json({"type": "error", "detail": "missing_data_b64"})
                    continue
                try:
                    chunk_bytes = base64.b64decode(data_b64, validate=True)
                except Exception:
                    await websocket.send_json({"type": "error", "detail": "invalid_base64"})
                    continue

                seq_raw = payload.get("seq")
                seq = int(seq_raw) if isinstance(seq_raw, int) or (isinstance(seq_raw, str) and seq_raw.isdigit()) else None
                if seq is not None:
                    session["last_seq"] = seq
                window_start_raw = payload.get("window_start_sec")
                window_duration_raw = payload.get("window_duration_sec")
                window_start_sec = None
                window_duration_sec = None
                if isinstance(window_start_raw, (int, float)):
                    window_start_sec = float(window_start_raw)
                elif isinstance(window_start_raw, str):
                    try:
                        window_start_sec = float(window_start_raw)
                    except ValueError:
                        window_start_sec = None
                if isinstance(window_duration_raw, (int, float)):
                    window_duration_sec = float(window_duration_raw)
                elif isinstance(window_duration_raw, str):
                    try:
                        window_duration_sec = float(window_duration_raw)
                    except ValueError:
                        window_duration_sec = None

                processed_seqs = session.get("processed_chunk_seqs")
                if not isinstance(processed_seqs, set):
                    processed_seqs = set()
                    session["processed_chunk_seqs"] = processed_seqs
                if seq is not None and seq in processed_seqs:
                    session["updated_at"] = _utc_now_iso()
                    await websocket.send_json(
                        {
                            "type": "ack_chunk",
                            "session_id": session_id,
                            "seq": seq,
                            "duplicate": True,
                            "window_start_sec": window_start_sec,
                            "window_duration_sec": window_duration_sec,
                            "chunks_received": session["chunks_received"],
                            "bytes_received": session["bytes_received"],
                            "asr_status": str(session.get("asr_status", "not_configured")),
                            "asr_error": str(session.get("asr_error", "")),
                            "new_segments": 0,
                            "total_segments": int(session.get("total_segments", 0) or 0),
                            "new_utterances": 0,
                            "total_utterances": len(session.get("utterances", [])),
                            "asr_latency_ms": 0.0,
                            "asr_mode": "duplicate_replay",
                            "sentence_role_split": session.get("last_sentence_role_split", {}),
                            "utterances": list(session.get("utterances", [])),
                            "new_utterances_payload": [],
                            "transcript_text": str(session.get("transcript_text", "") or ""),
                            "recording_audio_path": str(session.get("recording_audio_path", "") or ""),
                        }
                    )
                    continue

                sample_rate_hz = payload.get("sample_rate_hz")
                if isinstance(sample_rate_hz, int) and sample_rate_hz > 0:
                    session["sample_rate_hz"] = sample_rate_hz

                mime_type = str(payload.get("mime_type", "")).strip()
                if mime_type:
                    session["mime_type"] = mime_type

                session["status"] = "streaming"
                session["chunks_received"] = int(session.get("chunks_received", 0) or 0) + 1
                session["bytes_received"] = int(session.get("bytes_received", 0) or 0) + len(chunk_bytes)
                asr_stats = _run_live_chunk_asr(
                    session_id,
                    session,
                    chunk_bytes=chunk_bytes,
                    seq=seq,
                    window_start_sec=window_start_sec,
                    window_duration_sec=window_duration_sec,
                )
                if seq is not None:
                    if str(asr_stats.get("asr_status", "")) == "error":
                        processed_seqs.discard(seq)
                    else:
                        processed_seqs.add(seq)
                session["updated_at"] = _utc_now_iso()
                await websocket.send_json(
                    {
                        "type": "ack_chunk",
                        "session_id": session_id,
                        "seq": seq,
                        "duplicate": False,
                        "window_start_sec": window_start_sec,
                        "window_duration_sec": window_duration_sec,
                        "chunks_received": session["chunks_received"],
                        "bytes_received": session["bytes_received"],
                        "asr_status": asr_stats.get("asr_status", ""),
                        "asr_error": asr_stats.get("asr_error", ""),
                        "new_segments": asr_stats.get("new_segments", 0),
                        "total_segments": asr_stats.get("total_segments", 0),
                        "new_utterances": asr_stats.get("new_utterances", 0),
                        "total_utterances": asr_stats.get("total_utterances", 0),
                        "asr_latency_ms": asr_stats.get("asr_latency_ms"),
                        "asr_mode": asr_stats.get("asr_mode", "unknown"),
                        "diarization_status": str((asr_stats.get("diarization") or {}).get("status", "")),
                        "diarization_mode": str((asr_stats.get("diarization") or {}).get("mode", "")),
                        "sentence_role_split": asr_stats.get("sentence_role_split", {}),
                        "utterances": asr_stats.get("utterances_payload", []),
                        "new_utterances_payload": asr_stats.get("new_utterances_payload", []),
                        "transcript_text": asr_stats.get("transcript_text", ""),
                        "recording_audio_path": str(session.get("recording_audio_path", "") or ""),
                    }
                )
                continue

            if message_type == "stop":
                session["status"] = "stopped"
                session["updated_at"] = _utc_now_iso()
                await websocket.send_json(
                    {
                        "type": "ack_stop",
                        "session_id": session_id,
                        "status": session["status"],
                        "recording_audio_path": str(session.get("recording_audio_path", "") or ""),
                    }
                )
                continue

            await websocket.send_json({"type": "error", "detail": "unknown_message_type"})
    except WebSocketDisconnect:
        session["status"] = "disconnected"
        session["updated_at"] = _utc_now_iso()


@app.post("/transcript/incremental", response_model=IncrementalTranscriptResponse)
async def transcript_incremental(payload: IncrementalTranscriptRequest) -> IncrementalTranscriptResponse:
    store = _get_incremental_buffer_store()
    if payload.reset:
        store.pop(payload.session_id, None)

    buffer = store.get(payload.session_id)
    if buffer is None:
        buffer = IncrementalTranscriptBuffer()
        store[payload.session_id] = buffer

    append_result = buffer.append(payload.segments)
    utterances = align_segments_to_turns(append_result.all_segments, payload.turns)
    transcript_text = format_for_display(utterances, split_sentences=payload.split_sentences)
    return IncrementalTranscriptResponse(
        session_id=payload.session_id,
        new_segments=append_result.new_segments,
        all_segments=append_result.all_segments,
        utterances=utterances,
        transcript_text=transcript_text,
        debug={
            "dedupe": {
                "duplicates_dropped": append_result.duplicates_dropped,
                "new_segments": len(append_result.new_segments),
                "total_segments": len(append_result.all_segments),
            }
        },
    )


@app.post("/transcribe_structured", response_model=TranscribeStructuredResponse)
async def transcribe_structured(payload: TranscribeStructuredRequest) -> TranscribeStructuredResponse:
    session_id = payload.session_id
    segment_store = _get_incremental_buffer_store()
    turn_store = _get_transcribe_turn_store()
    role_state_store = _get_role_mapping_state_store()

    if payload.reset:
        segment_store.pop(session_id, None)
        turn_store.pop(session_id, None)
        role_state_store.pop(session_id, None)

    source_segments, source_turns, source_debug = _resolve_structured_source(payload)

    if payload.incremental:
        buffer = segment_store.get(session_id)
        if buffer is None:
            buffer = IncrementalTranscriptBuffer()
            segment_store[session_id] = buffer
        append_result = buffer.append(source_segments)
        all_segments = append_result.all_segments
        new_segments = append_result.new_segments
        duplicates_dropped = append_result.duplicates_dropped
    else:
        all_segments = list(source_segments)
        new_segments = list(source_segments)
        duplicates_dropped = 0

    if source_turns:
        if payload.incremental:
            all_turns = _merge_turns(turn_store.get(session_id, []), source_turns)
        else:
            all_turns = sorted(source_turns, key=lambda item: (item.start, item.end))
    else:
        all_turns = infer_turns_from_segments(all_segments)

    role_mapping_debug: dict[str, Any] = {"status": "skipped", "reason": "no_turns"}
    if all_turns:
        mapping_segments = new_segments if payload.incremental else all_segments
        try:
            all_turns, next_role_state, role_mapping_debug = apply_stable_role_mapping(
                all_turns,
                mapping_segments,
                state=role_state_store.get(session_id, {}),
            )
            role_state_store[session_id] = next_role_state
        except Exception as exc:
            role_mapping_debug = {"status": "error", "reason": str(exc)}

    if payload.incremental:
        turn_store[session_id] = all_turns

    # In audio-window mode we keep utterances granular to prevent cross-window
    # mega-merges that bloat downstream extraction prompts.
    merge_gap_sec = 0.0 if payload.audio_window_sec is not None else 0.6
    utterances, align_debug_all = align_segments_to_turns_with_debug(
        all_segments,
        all_turns,
        merge_gap_sec=merge_gap_sec,
    )
    new_utterances, align_debug_new = align_segments_to_turns_with_debug(
        new_segments,
        all_turns,
        merge_gap_sec=merge_gap_sec,
    )
    _log_alignment_debug("transcribe_all", session_id, align_debug_all)
    _log_alignment_debug("transcribe_new", session_id, align_debug_new)

    sentence_role_split_debug: dict[str, Any] = {"status": "skipped", "reason": "unknown"}
    diarization_debug = source_debug.get("diarization") if isinstance(source_debug, dict) else {}
    should_split, split_reason = _should_apply_sentence_role_split(
        source_debug=source_debug if isinstance(source_debug, dict) else {},
        diarization_debug=diarization_debug if isinstance(diarization_debug, dict) else {},
        turns=all_turns,
        utterances=utterances,
    )
    if should_split:
        utterances, split_all = _split_mixed_role_utterances_by_sentence(utterances)
        new_utterances, split_new = _split_mixed_role_utterances_by_sentence(new_utterances)
        sentence_role_split_debug = {
            "status": "applied",
            "condition": split_reason,
            "all": split_all,
            "new": split_new,
        }
    else:
        sentence_role_split_debug = {"status": "skipped", "reason": split_reason}

    transcript_text = format_for_display(utterances, split_sentences=payload.split_sentences)

    return TranscribeStructuredResponse(
        session_id=session_id,
        new_segments=new_segments,
        segments=all_segments,
        turns=all_turns,
        new_utterances=new_utterances,
        utterances=utterances,
        transcript_text=transcript_text,
        debug={
            **source_debug,
            "role_mapping": role_mapping_debug,
            "alignment": {"all": align_debug_all, "new": align_debug_new},
            "sentence_role_split": sentence_role_split_debug,
            "incremental": {
                "enabled": payload.incremental,
                "new_segments": len(new_segments),
                "total_segments": len(all_segments),
                "duplicates_dropped": duplicates_dropped,
                "turns_count": len(all_turns),
            },
        },
    )


@app.post("/events/extract", response_model=EventsExtractResponse)
async def events_extract(payload: EventsExtractRequest) -> EventsExtractResponse:
    utterances = [
        EventUtterance(
            segment_id=item.segment_id,
            t0=item.t0,
            t1=item.t1,
            speaker=item.speaker,
            text=item.text,
            asr_confidence=item.asr_confidence,
        )
        for item in payload.utterances
    ]
    debug: dict[str, Any] = {"engine_requested": payload.engine}

    extracted = []
    if payload.engine in {"auto", "medgemma"}:
        try:
            mg_result = extract_events_with_medgemma(
                utterances,
                model_path=payload.medgemma_model_path,
                max_tokens=payload.medgemma_max_tokens,
                n_ctx=payload.medgemma_n_ctx,
                n_gpu_layers=payload.medgemma_n_gpu_layers,
                n_threads=payload.medgemma_n_threads,
                chat_format=payload.medgemma_chat_format,
            )
            extracted = mg_result.events
            debug["engine_used"] = "medgemma"
            debug["medgemma"] = mg_result.debug
        except MedGemmaAdapterError as exc:
            debug["medgemma_error"] = str(exc)
            if payload.engine == "medgemma" and not payload.fallback_to_rule:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            extracted = extract_minimal_events(utterances)
            debug["engine_used"] = "rule_fallback"
    else:
        extracted = extract_minimal_events(utterances)
        debug["engine_used"] = "rule"

    extracted, guardrail_debug = apply_event_quality_guardrails(extracted)
    debug["event_guardrails"] = guardrail_debug
    if str(debug.get("engine_used", "")) == "medgemma":
        extracted, risk_backstop_debug = apply_rule_risk_backstop(extracted, utterances)
        debug["risk_backstop"] = risk_backstop_debug

    extracted, harmonization_debug = apply_event_consistency_harmonization(extracted)
    debug["event_harmonization"] = harmonization_debug

    if not extracted and payload.engine == "auto":
        # Preserve deterministic output for demos even when model emits no events.
        extracted = extract_minimal_events(utterances)
        extracted, fallback_guardrail_debug = apply_event_quality_guardrails(extracted)
        extracted, fallback_harmonization_debug = apply_event_consistency_harmonization(extracted)
        debug["engine_used"] = "rule_fallback_empty"
        debug["event_guardrails_fallback"] = fallback_guardrail_debug
        debug["event_harmonization_fallback"] = fallback_harmonization_debug

    type_counts = {"risk_cue": 0, "symptom": 0, "duration_onset": 0}
    events: list[EventItem] = []
    for item in extracted:
        type_counts[item.type] += 1
        events.append(
            EventItem(
                event_id=item.event_id,
                type=item.type,
                label=item.label,
                polarity=item.polarity,
                confidence=item.confidence,
                speaker=item.speaker,
                evidence=EventEvidence(
                    segment_id=item.segment_id,
                    t0=item.t0,
                    t1=item.t1,
                    quote=item.quote,
                ),
            )
        )

    return EventsExtractResponse(
        events=events,
        debug={
            "utterances": len(payload.utterances),
            "events": len(events),
            "type_counts": type_counts,
            **debug,
        },
    )


@app.post("/state/snapshot", response_model=StateSnapshotResponse)
async def state_snapshot(payload: StateSnapshotRequest) -> StateSnapshotResponse:
    snapshot_events = [
        SnapshotEvent(
            type=item.type,
            label=item.label,
            polarity=item.polarity,
            evidence_segment_id=item.evidence.segment_id,
        )
        for item in payload.events
    ]
    snapshot = build_state_snapshot(
        snapshot_events,
        ai_enhancement_enabled=payload.ai_enhancement_enabled,
    )
    return StateSnapshotResponse(
        problem_list=[
            StateProblemItem(item=item.item, evidence_refs=item.evidence_refs)
            for item in snapshot.problem_list
        ],
        risk_flags=[
            StateRiskFlag(
                level=item.level,
                flag=item.flag,
                why=item.why,
                evidence_refs=item.evidence_refs,
            )
            for item in snapshot.risk_flags
        ],
        open_questions=snapshot.open_questions,
        mandatory_safety_questions=snapshot.mandatory_safety_questions,
        contextual_followups=snapshot.contextual_followups,
        rationale=snapshot.rationale,
        updated_at=snapshot.updated_at,
        debug={
            "events": len(payload.events),
            "problem_count": len(snapshot.problem_list),
            "risk_flag_count": len(snapshot.risk_flags),
            "mandatory_question_count": len(snapshot.mandatory_safety_questions),
            "contextual_followup_count": len(snapshot.contextual_followups),
            "open_question_count": len(snapshot.open_questions),
            "open_questions_mode": "hybrid" if snapshot.contextual_followups else "rule",
            "ai_enhancement_enabled": snapshot.ai_enhancement_enabled,
            "ai_enhancement_applied": snapshot.ai_enhancement_applied,
            "ai_enhancement_error": snapshot.ai_enhancement_error,
        },
    )


@app.get("/note/templates", response_model=NoteTemplatesResponse)
async def note_templates() -> NoteTemplatesResponse:
    try:
        catalog = list_note_templates()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return NoteTemplatesResponse(
        templates_by_department={
            department: [
                NoteTemplateInfo(
                    template_id=item.template_id,
                    template_name=item.template_name,
                )
                for item in specs
            ]
            for department, specs in catalog.items()
        },
        debug={"department_count": len(catalog)},
    )


def _normalize_department_input(raw: str) -> str:
    return str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")


@app.get("/note/templates/{department}/{template_id}", response_model=NoteTemplateDetailResponse)
async def note_template_detail(department: str, template_id: str) -> NoteTemplateDetailResponse:
    try:
        template = get_note_template_document(department=department, template_id=template_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return NoteTemplateDetailResponse(
        department=_normalize_department_input(department),
        template_id=template_id,
        template=NoteTemplateDocument.model_validate(template),
        debug={"action": "read"},
    )


@app.put("/note/templates/{department}/{template_id}", response_model=NoteTemplateDetailResponse)
async def note_template_update(
    department: str,
    template_id: str,
    payload: NoteTemplateUpdateRequest,
) -> NoteTemplateDetailResponse:
    if payload.template.template_id != template_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "template_id in payload must match path template_id. "
                f"path={template_id!r}, payload={payload.template.template_id!r}"
            ),
        )
    try:
        template = save_note_template_document(
            department=department,
            template_id=template_id,
            template_document=payload.template.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return NoteTemplateDetailResponse(
        department=_normalize_department_input(department),
        template_id=template_id,
        template=NoteTemplateDocument.model_validate(template),
        debug={"action": "write"},
    )


@app.post("/note/draft", response_model=NoteDraftResponse)
async def note_draft(payload: NoteDraftRequest) -> NoteDraftResponse:
    try:
        batch = _build_note_draft_batch_from_payload(
            department=payload.department,
            template_ids=list(payload.template_ids),
            patient_identity=payload.patient_identity,
            patient_basic_info=payload.patient_basic_info,
            snapshot=payload.snapshot,
            events=list(payload.events),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not batch.drafts:
        raise HTTPException(status_code=500, detail="No note draft generated.")

    draft_items = [_draft_to_note_template_item(item) for item in batch.drafts]
    primary = draft_items[0]
    return NoteDraftResponse(
        note_type=primary.template_id,
        note_text=primary.note_text,
        citations=primary.citations,
        drafts=draft_items,
        debug={
            "department": batch.department,
            "patient_identity": payload.patient_identity,
            "patient_basic_info_included": bool(str(payload.patient_basic_info or "").strip()),
            "templates_requested": batch.requested_template_ids,
            "templates_generated": [item.template_id for item in batch.drafts],
            "templates_missing": batch.missing_template_ids,
            "templates_count": len(batch.drafts),
            "problem_count": len(payload.snapshot.problem_list),
            "risk_flag_count": len(payload.snapshot.risk_flags),
            "event_count": len(payload.events),
            "citation_count": len(primary.citations),
        },
    )


@app.post("/note/draft/jobs/start", response_model=NoteDraftJobResponse)
async def note_draft_job_start(payload: NoteDraftJobStartRequest) -> NoteDraftJobResponse:
    catalog = list_note_templates()
    resolved_department = _normalize_department_input(payload.department)
    specs = catalog.get(resolved_department)
    if not specs:
        raise HTTPException(status_code=400, detail=f"Unsupported department: {resolved_department}")

    requested_template_ids: list[str] = []
    seen_requested = set()
    for item in payload.template_ids:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen_requested:
            continue
        seen_requested.add(normalized)
        requested_template_ids.append(normalized)

    if not requested_template_ids:
        raise HTTPException(status_code=400, detail="template_ids is required and cannot be empty.")

    available_ids = {spec.template_id for spec in specs}
    selected_template_ids = [item for item in requested_template_ids if item in available_ids]
    missing_template_ids = [item for item in requested_template_ids if item not in available_ids]
    if not selected_template_ids:
        raise HTTPException(
            status_code=400,
            detail=f"No valid templates for department='{resolved_department}'. requested={requested_template_ids}",
        )

    job_id = f"notejob_{uuid4().hex[:12]}"
    now_iso = _utc_now_iso()
    template_statuses = {item: "queued" for item in selected_template_ids}
    for item in missing_template_ids:
        template_statuses[item] = "missing"

    payload_data = payload.model_dump()
    payload_data["department"] = resolved_department
    payload_data["template_ids"] = selected_template_ids

    job = {
        "job_id": job_id,
        "status": "pending",
        "stop_requested": False,
        "department": resolved_department,
        "requested_template_ids": requested_template_ids,
        "selected_template_ids": selected_template_ids,
        "missing_template_ids": missing_template_ids,
        "template_statuses": template_statuses,
        "drafts": [],
        "error": "",
        "payload": payload_data,
        "created_at": now_iso,
        "updated_at": now_iso,
        "debug": {
            "templates_requested_count": len(requested_template_ids),
            "templates_selected_count": len(selected_template_ids),
            "templates_missing_count": len(missing_template_ids),
            "event_count": len(payload.events),
            "problem_count": len(payload.snapshot.problem_list),
            "risk_flag_count": len(payload.snapshot.risk_flags),
        },
    }

    store = _get_note_draft_job_store()
    lock = _get_note_draft_job_lock()
    with lock:
        store[job_id] = job
        snapshot = copy.deepcopy(job)

    worker = threading.Thread(target=_run_note_draft_job, args=(job_id,), daemon=True)
    worker.start()
    return _serialize_note_job_response(snapshot)


@app.get("/note/draft/jobs/{job_id}", response_model=NoteDraftJobResponse)
async def note_draft_job_status(job_id: str) -> NoteDraftJobResponse:
    normalized_job_id = str(job_id or "").strip()
    if not normalized_job_id:
        raise HTTPException(status_code=400, detail="job_id is required.")

    store = _get_note_draft_job_store()
    lock = _get_note_draft_job_lock()
    with lock:
        job = store.get(normalized_job_id)
        if not isinstance(job, dict):
            raise HTTPException(status_code=404, detail=f"Note draft job not found: {normalized_job_id}")
        snapshot = copy.deepcopy(job)
    return _serialize_note_job_response(snapshot)


@app.post("/note/draft/jobs/{job_id}/stop", response_model=NoteDraftJobResponse)
async def note_draft_job_stop(job_id: str) -> NoteDraftJobResponse:
    normalized_job_id = str(job_id or "").strip()
    if not normalized_job_id:
        raise HTTPException(status_code=400, detail="job_id is required.")

    store = _get_note_draft_job_store()
    lock = _get_note_draft_job_lock()
    with lock:
        job = store.get(normalized_job_id)
        if not isinstance(job, dict):
            raise HTTPException(status_code=404, detail=f"Note draft job not found: {normalized_job_id}")

        if str(job.get("status", "")) in {"pending", "generating", "stopping"}:
            job["stop_requested"] = True
            if str(job.get("status", "")) != "pending":
                job["status"] = "stopping"
            job["updated_at"] = _utc_now_iso()

        snapshot = copy.deepcopy(job)
    return _serialize_note_job_response(snapshot)
