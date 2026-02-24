from __future__ import annotations

import datetime as _dt
import hashlib
import time
from pathlib import Path
from typing import Callable, List, Optional

from .. import audit
from ..audio_utils import (
    chunk_wav,
    compute_rms,
    enforce_max_duration,
    load_wav_info,
    load_wav16k_mono_float32,
    normalize_min_rms,
    normalize_to_wav16k_mono,
    write_wav16k_mono_float32,
)
from ..config import DemoConfig
from ..contracts import ChunkASRStatus, ChunkLagMetric, SessionASRStatus, TranscriptResult, TranscriptSegment
from ..session_store import InMemorySessionStore
from .base import ASRError, ASRProvider


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()

def _iso_from_epoch(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()

def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def stitch_dedupe_text(prev_tail_words: List[str], new_text: str, window_words: int = 12) -> str:
    new_words = (new_text or "").strip().split()
    if not new_words:
        return ""
    prev = prev_tail_words[-window_words:] if prev_tail_words else []
    head = new_words[:window_words]
    max_k = min(len(prev), len(head))
    overlap = 0
    for k in range(max_k, 0, -1):
        if prev[-k:] == head[:k]:
            overlap = k
            break
    return " ".join(new_words[overlap:]).strip()


def _tail_words(segments: List[TranscriptSegment], window_words: int = 12) -> List[str]:
    words: List[str] = []
    for seg in segments[-3:]:
        words.extend((seg.text or "").split())
    return words[-window_words:]


def _session_status(any_red: bool, any_yellow: bool) -> SessionASRStatus:
    if any_red:
        return "RED"
    if any_yellow:
        return "YELLOW"
    return "GREEN"


def transcribe_in_chunks(
    store: InMemorySessionStore,
    cfg: DemoConfig,
    provider_primary: ASRProvider,
    provider_fallback: Optional[ASRProvider],
    session_id: str,
    audio_path: str,
    language: str = "en",
    on_chunk: Optional[Callable[[Optional[TranscriptSegment], int, int], Optional[float]]] = None,
    playback_start_wall_ts: Optional[float] = None,
    fallback_label: Optional[str] = None,
    fallback_available: Optional[bool] = None,
    fallback_reason: str = "",
) -> TranscriptResult:
    # backend/internal_core/asr/controller.py -> asr -> internal_core -> backend -> evidentia
    project_root = Path(__file__).resolve().parents[3]
    tmp_dir = cfg.tmp_dir_path(project_root)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    store.set_state(session_id, "processing")
    store.set_error(session_id, None)

    def _medasr_lm_state():
        if provider_primary.name() != "medasr_hf":
            return None
        return {
            "enabled": bool(getattr(provider_primary, "lm_enabled", False)),
            "loaded": bool(getattr(provider_primary, "lm_loaded", False)),
            "status": str(getattr(provider_primary, "lm_status", "")),
        }

    medasr_lm_requested = bool(getattr(cfg, "MEDASR_USE_LM", False))
    audit.log_event(
        store,
        session_id,
        "ASR_STARTED",
        "ASR_START",
        f"primary={provider_primary.name()} fallback={(provider_fallback.name() if provider_fallback else 'none')} chunk_seconds={cfg.ASR_CHUNK_SEC} medasr_lm={'on' if medasr_lm_requested else 'off'}",
    )

    input_path = Path(audio_path)
    wav_path = normalize_to_wav16k_mono(
        input_path,
        tmp_dir=tmp_dir,
        session_prefix=session_id,
    )
    if wav_path.resolve() != input_path.resolve():
        store.register_temp_file(session_id, str(wav_path))

    duration_sec, _, _ = load_wav_info(wav_path)
    enforce_max_duration(duration_sec, cfg.SCRIBE_MAX_AUDIO_SECONDS)
    audio_hash = _sha256_file(wav_path)

    chunks = chunk_wav(
        wav_path,
        chunk_seconds=cfg.ASR_CHUNK_SEC,
        overlap_seconds=cfg.ASR_OVERLAP_SEC,
        tmp_dir=tmp_dir,
        prefix=f"{session_id}",
    )
    for c in chunks:
        store.register_temp_file(session_id, c["chunk_path"])

    warnings: List[str] = []
    errors: List[str] = []
    incomplete = False
    segments: List[TranscriptSegment] = []
    if playback_start_wall_ts is None:
        playback_start_wall_ts = time.time()
    total_chunks = len(chunks)
    primary_available = True
    warned_primary_unavailable = False
    any_yellow = False
    any_red = False

    def _update_partial_result(processed_chunks: int) -> None:
        store.set_transcript_result(
            session_id,
            TranscriptResult(
                segments=list(segments),
                incomplete=incomplete,
                warnings=list(warnings),
                errors=list(errors),
                meta={
                    "primary": provider_primary.name(),
                    "fallback": fallback_label,
                    "fallback_available": bool(fallback_available),
                    "fallback_reason": fallback_reason,
                    "chunk_seconds": cfg.ASR_CHUNK_SEC,
                    "overlap_seconds": cfg.ASR_OVERLAP_SEC,
                    "min_text_chars": cfg.ASR_MIN_TEXT_CHARS,
                    "silence_rms": cfg.ASR_SILENCE_RMS,
                    "medasr_lm": _medasr_lm_state(),
                    "session_status": _session_status(any_red, any_yellow),
                    "created_at": _now_iso(),
                    "audio_sha256": audio_hash,
                    "timing": {
                        "duration_sec": float(duration_sec),
                        "chunks": total_chunks,
                        "processed_chunks": int(processed_chunks),
                    },
                },
            ),
        )

    for c in chunks:
        idx = int(c["index"])
        chunk_file = Path(c["chunk_path"])
        chunk_wall_start = _now_iso()
        start_monotonic = time.monotonic()
        provider_used = None
        decoding_method = ""
        status: ChunkASRStatus
        display_wall_ts = None
        asr_done_ts = None
        chunk_ready_ts = time.time()
        chunk_end_wall_ts = float(playback_start_wall_ts + float(c["end_sec"]))
        is_silence = False

        def _append_metric(
            status: ChunkASRStatus,
            provider_used: str,
            decoding_method: str,
            is_silence: bool,
        ) -> None:
            nonlocal display_wall_ts, asr_done_ts
            if not isinstance(asr_done_ts, (int, float)):
                asr_done_ts = time.time()
            if not isinstance(display_wall_ts, (int, float)):
                display_wall_ts = time.time()
            processing_sec = float(max(0.0, float(asr_done_ts) - float(chunk_ready_ts)))
            post_chunk_latency_sec = float(
                max(0.0, float(display_wall_ts) - float(chunk_ready_ts))
            )
            metric = ChunkLagMetric(
                chunk_index=idx,
                audio_start_sec=float(c["start_sec"]),
                audio_end_sec=float(c["end_sec"]),
                wall_clock_start_ts_iso=chunk_wall_start,
                wall_clock_end_ts_iso=_now_iso(),
                chunk_end_wall_ts_iso=_iso_from_epoch(chunk_end_wall_ts),
                chunk_ready_ts=float(chunk_ready_ts),
                asr_done_ts=float(asr_done_ts),
                display_wall_ts=float(display_wall_ts),
                processing_sec=processing_sec,
                post_chunk_latency_sec=post_chunk_latency_sec,
                display_ts_iso=_iso_from_epoch(float(display_wall_ts)),
                lag_sec=post_chunk_latency_sec,
                provider_used=provider_used,
                decoding_method=decoding_method,
                status=status,
                is_silence=is_silence,
            )
            store.append_lag_metric(session_id, metric)
        try:
            # Silence guard + optional loudness normalization (in-place on chunk wav).
            try:
                audio = load_wav16k_mono_float32(chunk_file)
                rms = compute_rms(audio)
                if 0 < rms < cfg.ASR_TARGET_MIN_RMS:
                    audio_norm = normalize_min_rms(audio, cfg.ASR_TARGET_MIN_RMS)
                    if audio_norm is not audio:
                        write_wav16k_mono_float32(chunk_file, audio_norm)
                        audio = audio_norm
                        rms = compute_rms(audio)

                peak = float(max(abs(audio.min(initial=0.0)), abs(audio.max(initial=0.0)))) if getattr(audio, "size", 0) else 0.0
                # Treat as silence only when both RMS and peak are very low (avoid skipping quiet speech).
                if rms < cfg.ASR_SILENCE_RMS and peak < 0.02:
                    is_silence = True
            except Exception:
                # Don't fail ASR for preprocessing; proceed best-effort.
                pass

            if is_silence:
                status = "SKIP_SILENCE"
                provider_used = "silence"
                decoding_method = ""
                audit.log_event(
                    store,
                    session_id,
                    "ASR_CHUNK_DONE",
                    "ASR_SILENCE",
                    f"chunk_index={idx} status={status}",
                    duration_ms=int((time.monotonic() - start_monotonic) * 1000),
                )
                asr_done_ts = time.time()
                if on_chunk:
                    display_wall_ts = on_chunk(None, idx, len(chunks))
                _append_metric(status, provider_used, decoding_method, True)
                _update_partial_result(processed_chunks=idx + 1)
                continue

            if not primary_available:
                raise ASRError("PRIMARY_DISABLED", "primary disabled", provider_primary.name())

            if (
                cfg.SCRIBE_TEST_INJECT_ASR_FAIL_AT_CHUNK is not None
                and idx == cfg.SCRIBE_TEST_INJECT_ASR_FAIL_AT_CHUNK
            ):
                raise ASRError(
                    "INJECTED_ASR_FAIL",
                    "Injected ASR chunk failure",
                    provider_primary.name(),
                )

            text = provider_primary.transcribe_chunk(
                c["chunk_path"], language=language, timeout_sec=30
            )
            provider_used = provider_primary.name()
            decoding_method = getattr(provider_primary, "last_decoding_method", "") or ""
            text = " ".join((text or "").split()).strip()
            if segments and cfg.ASR_OVERLAP_SEC > 0:
                text = stitch_dedupe_text(_tail_words(segments, window_words=12), text, window_words=12)
            if len(text) < cfg.ASR_MIN_TEXT_CHARS:
                raise ASRError("PRIMARY_EMPTY_OUTPUT", "empty output", provider_primary.name())
            seg = TranscriptSegment(
                start_sec=float(c["start_sec"]),
                end_sec=float(c["end_sec"]),
                text=text,
                confidence=None,
                speaker=None,
                chunk_index=idx,
            )
            segments.append(seg)
            store.append_transcript_segment(session_id, seg)
            status = "OK_PRIMARY"
            audit.log_event(
                store,
                session_id,
                "ASR_CHUNK_DONE",
                "ASR_CHUNK",
                f"chunk_index={idx} provider={provider_used} status={status} method={decoding_method}",
                duration_ms=int((time.monotonic() - start_monotonic) * 1000),
            )
            asr_done_ts = time.time()
            if on_chunk:
                display_wall_ts = on_chunk(seg, idx, len(chunks))
            _append_metric(status, provider_used, decoding_method, False)
            _update_partial_result(processed_chunks=idx + 1)
        except ASRError as e:
            # Primary failure: attempt fallback best-effort.
            primary_failed = e.code not in {"PRIMARY_DISABLED"}
            if primary_failed:
                audit.log_event(
                    store,
                    session_id,
                    "ASR_FAILED",
                    e.code,
                    f"chunk_index={idx} provider={e.provider_name}",
                    duration_ms=int((time.monotonic() - start_monotonic) * 1000),
                )

            # If MedASR isn't configured, treat this as "primary unavailable" (not partial processing).
            if e.code in {"MEDASR_NOT_CONFIGURED", "MEDASR_LOAD_FAILED"}:
                primary_available = False
                any_yellow = True
                if not warned_primary_unavailable:
                    warned_primary_unavailable = True
                    msg = (getattr(e, "message", "") or "").strip()
                    if len(msg) > 120:
                        msg = msg[:120] + "…"
                    detail = f" ({msg})" if msg else ""
                    warnings.append(f"Primary ASR unavailable: {e.code}{detail}; using fallback.")
            elif primary_failed:
                any_yellow = True
                warnings.append(f"ASR chunk {idx} primary failed: {e.code}")

            if provider_fallback is None:
                any_red = True
                incomplete = True
                errors.append(e.code)
                warnings.append(f"ASR chunk {idx} failed (no fallback): {e.code}")
                asr_done_ts = time.time()
                if on_chunk:
                    display_wall_ts = on_chunk(None, idx, len(chunks))
                status = "FAIL_BOTH_FAILED"
                _append_metric(status, "none", "", False)
                _update_partial_result(processed_chunks=idx + 1)
                continue

            if provider_fallback is not None:
                try:
                    text = provider_fallback.transcribe_chunk(
                        c["chunk_path"], language=language, timeout_sec=30
                    )
                    provider_used = provider_fallback.name()
                    decoding_method = getattr(provider_fallback, "last_decoding_method", "") or ""
                    text = " ".join((text or "").split()).strip()
                    if segments and cfg.ASR_OVERLAP_SEC > 0:
                        text = stitch_dedupe_text(_tail_words(segments, window_words=12), text, window_words=12)
                    if len(text) < cfg.ASR_MIN_TEXT_CHARS:
                        raise ASRError(
                            "WHISPER_EMPTY_OUTPUT",
                            "fallback empty output",
                            provider_fallback.name(),
                        )
                    seg = TranscriptSegment(
                        start_sec=float(c["start_sec"]),
                        end_sec=float(c["end_sec"]),
                        text=text,
                        confidence=None,
                        speaker=None,
                        chunk_index=idx,
                    )
                    segments.append(seg)
                    store.append_transcript_segment(session_id, seg)
                    status = "WARN_PRIMARY_FAILED_FALLBACK_OK"
                    audit.log_event(
                        store,
                        session_id,
                        "ASR_CHUNK_DONE",
                        "ASR_CHUNK_FALLBACK",
                        f"chunk_index={idx} provider={provider_used} status={status} method={decoding_method}",
                        duration_ms=int((time.monotonic() - start_monotonic) * 1000),
                    )
                    asr_done_ts = time.time()
                    if on_chunk:
                        display_wall_ts = on_chunk(seg, idx, len(chunks))
                    _append_metric(status, provider_used, decoding_method, False)
                    _update_partial_result(processed_chunks=idx + 1)
                    continue
                except ASRError as fe:
                    any_red = True
                    incomplete = True
                    errors.append(fe.code)
                    warnings.append(f"ASR chunk {idx} fallback failed: {fe.code}")
                    audit.log_event(
                        store,
                        session_id,
                        "ASR_FAILED",
                        fe.code,
                        f"chunk_index={idx} provider={fe.provider_name}",
                        duration_ms=int((time.monotonic() - start_monotonic) * 1000),
                    )
                except Exception:
                    any_red = True
                    incomplete = True
                    errors.append("ASR_FALLBACK_UNKNOWN")
                    warnings.append(f"ASR chunk {idx} fallback failed: ASR_FALLBACK_UNKNOWN")
                    audit.log_event(
                        store,
                        session_id,
                        "ASR_FAILED",
                        "ASR_FALLBACK_UNKNOWN",
                        f"chunk_index={idx} provider={provider_fallback.name()}",
                        duration_ms=int((time.monotonic() - start_monotonic) * 1000),
                    )

            asr_done_ts = time.time()
            if on_chunk:
                display_wall_ts = on_chunk(None, idx, len(chunks))
            status = "FAIL_BOTH_FAILED"
            _append_metric(status, (provider_used or "none"), decoding_method, False)
            _update_partial_result(processed_chunks=idx + 1)
            continue
        except Exception as e:
            any_red = True
            incomplete = True
            errors.append("ASR_UNKNOWN")
            warnings.append(f"ASR chunk {idx} failed: ASR_UNKNOWN")
            audit.log_event(
                store,
                session_id,
                "ASR_FAILED",
                "ASR_UNKNOWN",
                f"chunk_index={idx}",
                duration_ms=int((time.monotonic() - start_monotonic) * 1000),
            )
            asr_done_ts = time.time()
            if on_chunk:
                display_wall_ts = on_chunk(None, idx, len(chunks))
            status = "FAIL_BOTH_FAILED"
            _append_metric(status, "unknown", "", False)
            _update_partial_result(processed_chunks=idx + 1)
            continue
        finally:
            # Reduce on-disk footprint: chunk files are intermediates and can be deleted immediately.
            _safe_unlink(chunk_file)

    if len(segments) == 0:
        store.set_state(session_id, "failed")
        uniq = []
        for code in errors:
            if code not in uniq:
                uniq.append(code)
        codes = ",".join(uniq[:6]) if uniq else "none"
        store.set_error(session_id, f"ASR failed (no transcript produced). codes={codes}")
        incomplete = True
        any_red = True
        warnings.append("No transcript segments produced.")
    else:
        store.set_state(session_id, "completed")

    result = TranscriptResult(
        segments=segments,
        incomplete=incomplete,
        warnings=warnings,
        errors=errors,
        meta={
            "primary": provider_primary.name(),
            "fallback": fallback_label,
            "fallback_available": bool(fallback_available),
            "fallback_reason": fallback_reason,
            "chunk_seconds": cfg.ASR_CHUNK_SEC,
            "overlap_seconds": cfg.ASR_OVERLAP_SEC,
            "min_text_chars": cfg.ASR_MIN_TEXT_CHARS,
            "silence_rms": cfg.ASR_SILENCE_RMS,
            "medasr_lm": _medasr_lm_state(),
            "session_status": _session_status(any_red, any_yellow),
            "created_at": _now_iso(),
            "audio_sha256": audio_hash,
            "timing": {"duration_sec": float(duration_sec), "chunks": len(chunks)},
        },
    )
    store.set_transcript_result(session_id, result)
    # Clean up normalized WAV if it was an intermediate conversion.
    if wav_path.resolve() != input_path.resolve():
        _safe_unlink(wav_path)
    return result


def transcribe_audio_in_chunks(
    store: InMemorySessionStore,
    cfg: DemoConfig,
    provider: ASRProvider,
    session_id: str,
    audio_path: str,
    language: str = "en",
    on_chunk: Optional[callable] = None,
) -> TranscriptResult:
    # Back-compat wrapper (single provider, no fallback).
    return transcribe_in_chunks(
        store,
        cfg,
        provider_primary=provider,
        provider_fallback=None,
        session_id=session_id,
        audio_path=audio_path,
        language=language,
        on_chunk=on_chunk,
    )
    if fallback_label is None:
        fallback_label = provider_fallback.name() if provider_fallback else None
    if fallback_available is None:
        fallback_available = provider_fallback is not None
    if fallback_available is False:
        reason = (fallback_reason or "not configured").strip()
        warnings.append(f"Fallback ASR unavailable: {reason}.")
