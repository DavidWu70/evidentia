from __future__ import annotations

import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

from .contracts import (
    AuditEvent,
    ChunkLagMetric,
    SOAPDraft,
    SessionState,
    TranscriptResult,
    TranscriptSegment,
)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


class InMemorySessionStore:
    def __init__(self, ttl_seconds: int, tmp_dir: Path):
        self._ttl_seconds = ttl_seconds
        self._tmp_dir = tmp_dir
        self._lock = RLock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "created_at": now,
                "updated_at": now,
                "expires_at": now + self._ttl_seconds,
                "state": "idle",
                "audio_source": None,
                "audio_path": None,
                "transcript_segments": [],
                "transcript_result": None,
                "soap_draft": None,
                "lag_metrics": [],
                "audit_events": [],
                "temp_files": set(),
                "retention": None,
                "error": None,
            }
        return session_id

    def _touch(self, session_id: str) -> None:
        now = time.time()
        session = self._sessions[session_id]
        session["updated_at"] = now
        session["expires_at"] = now + self._ttl_seconds

    def set_audio_source(self, session_id: str, source_type: str, source_name: str) -> None:
        with self._lock:
            self._sessions[session_id]["audio_source"] = {
                "type": source_type,
                "name": source_name,
            }
            self._touch(session_id)

    def set_audio_path(self, session_id: str, path: str) -> None:
        with self._lock:
            self._sessions[session_id]["audio_path"] = path
            self._touch(session_id)

    def register_temp_file(self, session_id: str, path: str) -> None:
        with self._lock:
            self._sessions[session_id]["temp_files"].add(path)
            self._touch(session_id)

    def set_state(self, session_id: str, state: SessionState) -> None:
        with self._lock:
            self._sessions[session_id]["state"] = state
            self._touch(session_id)

    def set_error(self, session_id: str, message: Optional[str]) -> None:
        with self._lock:
            self._sessions[session_id]["error"] = message
            self._touch(session_id)

    def append_transcript_segment(self, session_id: str, segment: TranscriptSegment) -> None:
        with self._lock:
            self._sessions[session_id]["transcript_segments"].append(segment)
            self._touch(session_id)

    def set_transcript_result(self, session_id: str, result: TranscriptResult) -> None:
        with self._lock:
            self._sessions[session_id]["transcript_result"] = result
            self._touch(session_id)

    def append_lag_metric(self, session_id: str, metric: ChunkLagMetric) -> None:
        with self._lock:
            self._sessions[session_id]["lag_metrics"].append(metric)
            self._touch(session_id)

    def set_soap_draft(self, session_id: str, draft: SOAPDraft) -> None:
        with self._lock:
            self._sessions[session_id]["soap_draft"] = draft
            self._touch(session_id)

    def append_audit_event(self, session_id: str, event: AuditEvent) -> None:
        with self._lock:
            self._sessions[session_id]["audit_events"].append(event)
            self._touch(session_id)

    def set_retention_metadata(self, session_id: str, metadata: Dict[str, Any]) -> None:
        with self._lock:
            self._sessions[session_id]["retention"] = metadata
            self._touch(session_id)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown session_id: {session_id}")
            return {
                "session_id": session["session_id"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "expires_at": session["expires_at"],
                "state": session["state"],
                "audio_source": session["audio_source"],
                "audio_path": session["audio_path"],
                "transcript_segments": list(session["transcript_segments"]),
                "transcript_result": session["transcript_result"],
                "soap_draft": session["soap_draft"],
                "lag_metrics": list(session["lag_metrics"]),
                "audit_events": list(session["audit_events"]),
                "retention": session["retention"],
                "error": session["error"],
            }

    def destroy_session(self, session_id: str, reason: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return

        # Best-effort temp file cleanup.
        audio_path = session.get("audio_path")
        if audio_path:
            try:
                ap = Path(audio_path)
                if self._tmp_dir in ap.resolve().parents:
                    _safe_unlink(ap)
            except Exception:
                pass

        for p in list(session.get("temp_files") or []):
            try:
                _safe_unlink(Path(p))
            except Exception:
                pass

    def reset_transcript_and_soap(self, session_id: str, keep_audio_path: Optional[str]) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown session_id: {session_id}")

            keep = keep_audio_path or session.get("audio_path")
            keep = str(keep) if keep else None
            temp_files = set(session.get("temp_files") or set())
            session["temp_files"] = set()
            session["transcript_segments"] = []
            session["transcript_result"] = None
            session["soap_draft"] = None
            session["lag_metrics"] = []
            session["error"] = None
            self._touch(session_id)

        for p in temp_files:
            if keep and p == keep:
                continue
            try:
                _safe_unlink(Path(p))
            except Exception:
                pass

    def cleanup_expired_sessions(self) -> int:
        now = time.time()
        expired = []
        with self._lock:
            for session_id, session in self._sessions.items():
                if session["expires_at"] <= now:
                    expired.append(session_id)
        for session_id in expired:
            self.destroy_session(session_id, reason="ttl_expired")
        return len(expired)
