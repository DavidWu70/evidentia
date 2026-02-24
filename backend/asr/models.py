from __future__ import annotations

"""
Typed ASR data contracts shared by transcript endpoints.

Design intent:
- Enforce timestamp-valid segment payloads at API boundaries.
- Keep speaker/utterance schemas explicit for downstream traceability.
"""

from pydantic import BaseModel, Field, model_validator


class ASRSegment(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    @model_validator(mode="after")
    def _validate_window(self) -> "ASRSegment":
        if self.end < self.start:
            raise ValueError("ASRSegment.end must be >= ASRSegment.start")
        return self


class SpeakerTurn(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    speaker: str
    speaker_id: str | None = None
    speaker_role: str | None = None
    confidence: float | None = None
    diar_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    role_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_window(self) -> "SpeakerTurn":
        if self.end < self.start:
            raise ValueError("SpeakerTurn.end must be >= SpeakerTurn.start")
        return self


class SpeakerUtterance(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    speaker: str
    speaker_id: str | None = None
    speaker_role: str | None = None
    diar_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    role_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    text: str

    @model_validator(mode="after")
    def _validate_window(self) -> "SpeakerUtterance":
        if self.end < self.start:
            raise ValueError("SpeakerUtterance.end must be >= SpeakerUtterance.start")
        return self
