from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SessionState = Literal[
    "idle", "queued", "processing", "completed", "failed", "finalized"
]


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_sec: float
    end_sec: float
    text: str
    confidence: Optional[float] = None
    speaker: Optional[str] = None
    chunk_index: int


class TranscriptResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: List[TranscriptSegment] = Field(default_factory=list)
    incomplete: bool = False
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


ChunkASRStatus = Literal[
    "OK_PRIMARY",
    "WARN_PRIMARY_FAILED_FALLBACK_OK",
    "FAIL_BOTH_FAILED",
    "SKIP_SILENCE",
]

SessionASRStatus = Literal["GREEN", "YELLOW", "RED"]


class ChunkLagMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_index: int
    audio_start_sec: float
    audio_end_sec: float
    wall_clock_start_ts_iso: str
    wall_clock_end_ts_iso: str
    chunk_end_wall_ts_iso: Optional[str] = None
    chunk_ready_ts: Optional[float] = None
    asr_done_ts: Optional[float] = None
    display_wall_ts: Optional[float] = None
    processing_sec: Optional[float] = None
    post_chunk_latency_sec: Optional[float] = None
    display_ts_iso: str
    lag_sec: float
    provider_used: str
    decoding_method: str
    status: ChunkASRStatus
    is_silence: bool = False


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_sec: float
    end_sec: float
    quote: str
    segment_indices: List[int] = Field(default_factory=list)


SOAPEntryStatus = Literal["ok", "soft_fail", "hard_fail"]


class SOAPEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    status: SOAPEntryStatus = "soft_fail"
    notes: List[str] = Field(default_factory=list)


class SOAPDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subjective: List[SOAPEntry] = Field(default_factory=list)
    objective: List[SOAPEntry] = Field(default_factory=list)
    assessment: List[SOAPEntry] = Field(default_factory=list)
    plan: List[SOAPEntry] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


AuditEventType = Literal[
    "SESSION_CREATED",
    "AUDIO_SELECTED",
    "UPLOAD_USED",
    "ASR_STARTED",
    "ASR_CHUNK_DONE",
    "ASR_FAILED",
    "SOAP_STARTED",
    "SOAP_DONE",
    "FINALIZE",
    "RETAIN_REQUESTED",
    "SESSION_DESTROYED",
    "ERROR",
]


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts_iso: str
    session_id: str
    type: AuditEventType
    code: str
    detail: str
    duration_ms: Optional[int] = None


class FactEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_index: int
    start_sec: float
    end_sec: float
    quote: str


class FactCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = None
    concept: str
    section_hint: Optional[str] = None
    text: str
    polarity: Optional[str] = None
    temporality: Optional[str] = None
    confidence: Optional[float] = None
    evidence: List[FactEvidenceRef] = Field(default_factory=list)


class Fact(FactCandidate):
    id: str


class FactWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    chunk_index: Optional[int] = None


class FactLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.1"
    facts: List[Fact] = Field(default_factory=list)
    warnings: List[FactWarning] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class ChunkExtractResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.1"
    chunk_index: int
    facts: List[FactCandidate] = Field(default_factory=list)
    warnings: List[FactWarning] = Field(default_factory=list)


class TemplateField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    accept_concepts: List[str] = Field(default_factory=list)
    accept_sections: List[str] = Field(default_factory=list)
    max_items: Optional[int] = None
    order_by: Optional[str] = "time"
    render: Optional[str] = "bullets"
    empty_behavior: Optional[str] = "not_mentioned"
    max_chars: Optional[int] = None


class TemplateDedupe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalize: str = "lower_punct_ws"
    key_fields: List[str] = Field(default_factory=list)
    keep: str = "first"
    merge: List[str] = Field(default_factory=list)
    order_for_keep: Optional[str] = "time"


class TemplateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str
    version: str
    fields: List[TemplateField] = Field(default_factory=list)
    dedupe: TemplateDedupe


class RenderedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence: List[FactEvidenceRef] = Field(default_factory=list)
    fact_ids: List[str] = Field(default_factory=list)


class RenderedField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    items: List[RenderedItem] = Field(default_factory=list)


class RenderedTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str
    version: str
    fields: List[RenderedField] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
