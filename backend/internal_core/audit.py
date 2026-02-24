from __future__ import annotations

import datetime as _dt
from typing import Optional

from .contracts import AuditEvent, AuditEventType
from .session_store import InMemorySessionStore


def _ts_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _sanitize_detail(detail: str) -> str:
    # IMPORTANT: Never include transcript / SOAP / audio bytes in detail.
    # This is a lightweight guard to keep demo metadata short and safe.
    detail = (detail or "").replace("\n", " ").strip()
    if len(detail) > 200:
        detail = detail[:200] + "…"
    return detail


def log_event(
    store: InMemorySessionStore,
    session_id: str,
    event_type: AuditEventType,
    code: str,
    detail: str,
    duration_ms: Optional[int] = None,
) -> None:
    event = AuditEvent(
        ts_iso=_ts_iso(),
        session_id=session_id,
        type=event_type,
        code=code,
        detail=_sanitize_detail(detail),
        duration_ms=duration_ms,
    )
    store.append_audit_event(session_id, event)
