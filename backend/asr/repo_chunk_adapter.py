from __future__ import annotations

"""
Bridge Evidentia backend to the chunk-based ASR pipeline bundled inside this repo.

Design intent:
- Reuse battle-tested chunking/transcription utilities instead of reimplementing ASR orchestration.
- Keep integration optional and failure-tolerant: callers receive actionable errors when providers are unavailable.
- Return typed segments/turns so downstream evidence contracts stay stable across ASR backend swaps.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import threading
import uuid

from backend.asr.models import ASRSegment, SpeakerTurn
from backend.asr.structured_input import normalize_speaker
from backend.internal_core import load_config
from backend.internal_core.asr import (
    MedASRHFProvider,
    MockASRProvider,
    WhisperCppProvider,
    transcribe_in_chunks,
    whisper_cpp_available,
)
from backend.internal_core.audio_utils import (
    load_wav16k_mono_float32,
    load_wav_info,
    normalize_to_wav16k_mono,
    write_wav16k_mono_float32,
)
from backend.internal_core.session_store import InMemorySessionStore


class RepoASRAdapterError(RuntimeError):
    """Raised when the repo-based ASR adapter cannot produce valid segments."""


@dataclass(frozen=True)
class RepoASRResult:
    segments: list[ASRSegment]
    turns: list[SpeakerTurn]
    debug: dict[str, Any]


_PROVIDER_CACHE: dict[tuple[Any, ...], Any] = {}
_PROVIDER_CACHE_LOCK = threading.Lock()
_SUPPORTED_PROVIDERS = {"whisper_cpp", "medasr_hf", "mock"}
_DISABLED_PROVIDER_NAMES = {"", "none", "disabled", "off", "false", "0"}


def _provider_cache_key(provider_name: str, cfg: Any) -> tuple[Any, ...]:
    if provider_name == "whisper_cpp":
        return (
            "whisper_cpp",
            str(cfg.SCRIBE_WHISPER_CPP_BIN),
            str(cfg.SCRIBE_WHISPER_CPP_MODEL),
            bool(cfg.SCRIBE_WHISPER_CPP_NO_GPU),
        )
    if provider_name == "medasr_hf":
        return (
            "medasr_hf",
            str(cfg.SCRIBE_MEDASR_MODEL),
            str(cfg.SCRIBE_MEDASR_DEVICE or ""),
            bool(cfg.SCRIBE_MEDASR_FP16),
            bool(cfg.MEDASR_USE_LM),
            str(cfg.MEDASR_LM_PATH),
            float(cfg.MEDASR_LM_ALPHA),
            float(cfg.MEDASR_LM_BETA),
            int(cfg.MEDASR_LM_BEAM_WIDTH),
            int(cfg.ASR_MIN_TEXT_CHARS),
        )
    return (provider_name,)


def _get_or_create_provider(provider_name: str, cfg: Any, *, builders: dict[str, Any]) -> Any:
    if provider_name == "mock":
        return builders["mock"]()
    key = _provider_cache_key(provider_name, cfg)
    with _PROVIDER_CACHE_LOCK:
        existing = _PROVIDER_CACHE.get(key)
        if existing is not None:
            return existing
        created = builders[provider_name]()
        _PROVIDER_CACHE[key] = created
        return created


def _normalize_provider_name(name: str | None) -> str:
    return (name or "").strip().lower()


def _provider_disabled(name: str | None) -> bool:
    return _normalize_provider_name(name) in _DISABLED_PROVIDER_NAMES


def _resolve_provider_plan(cfg: Any, provider_override: str | None) -> tuple[str, str, bool, str, bool]:
    """
    Returns:
    - primary_name
    - fallback_label
    - fallback_enabled
    - fallback_reason
    - explicit_override
    """

    override = _normalize_provider_name(provider_override)
    if override:
        if override not in _SUPPORTED_PROVIDERS:
            raise RepoASRAdapterError(f"Unsupported ASR provider: {override}")
        return override, "none", False, "disabled by explicit provider override", True

    primary_name = _normalize_provider_name(cfg.ASR_PRIMARY_PROVIDER or cfg.SCRIBE_ASR_PRIMARY or "whisper_cpp")
    if _provider_disabled(primary_name):
        primary_name = "mock"
    if primary_name not in _SUPPORTED_PROVIDERS:
        raise RepoASRAdapterError(f"Unsupported ASR provider: {primary_name}")

    fallback_name = _normalize_provider_name(cfg.ASR_FALLBACK_PROVIDER or cfg.SCRIBE_ASR_FALLBACK or "")
    if _provider_disabled(fallback_name):
        return primary_name, "none", False, "disabled by config", False
    if fallback_name not in _SUPPORTED_PROVIDERS:
        raise RepoASRAdapterError(f"Unsupported fallback ASR provider: {fallback_name}")
    if fallback_name == primary_name:
        return primary_name, fallback_name, False, "same as primary", False
    return primary_name, fallback_name, True, "", False


def transcribe_audio_with_repo_chunk_pipeline(
    *,
    audio_path: str,
    language: str,
    provider: str | None = None,
    chunk_sec: int | None = None,
    overlap_sec: float | None = None,
) -> RepoASRResult:
    """
    Run chunk-based ASR and convert to Evidentia types.

    `provider` supports `whisper_cpp`, `medasr_hf`, and `mock`.
    When `provider` is set, fallback is disabled and that provider is forced.
    """

    project_root = Path(__file__).resolve().parents[2]
    cfg = load_config()

    if chunk_sec is not None and chunk_sec > 0:
        cfg = replace(cfg, ASR_CHUNK_SEC=int(chunk_sec))
    if overlap_sec is not None and overlap_sec >= 0:
        cfg = replace(cfg, ASR_OVERLAP_SEC=float(overlap_sec))
    (
        primary_name,
        fallback_label,
        fallback_enabled,
        fallback_reason,
        explicit_override,
    ) = _resolve_provider_plan(cfg, provider)
    cfg = replace(
        cfg,
        ASR_PRIMARY_PROVIDER=primary_name,
        SCRIBE_ASR_PRIMARY=primary_name,
        ASR_FALLBACK_PROVIDER=fallback_label,
        SCRIBE_ASR_FALLBACK=fallback_label,
    )

    builders = {
        "whisper_cpp": lambda: WhisperCppProvider(
            cfg.SCRIBE_WHISPER_CPP_BIN,
            cfg.SCRIBE_WHISPER_CPP_MODEL,
            no_gpu=cfg.SCRIBE_WHISPER_CPP_NO_GPU,
        ),
        "medasr_hf": lambda: MedASRHFProvider(
            model_ref=cfg.SCRIBE_MEDASR_MODEL,
            device=cfg.SCRIBE_MEDASR_DEVICE or None,
            fp16=cfg.SCRIBE_MEDASR_FP16,
            use_lm=cfg.MEDASR_USE_LM,
            lm_path=cfg.MEDASR_LM_PATH,
            lm_alpha=cfg.MEDASR_LM_ALPHA,
            lm_beta=cfg.MEDASR_LM_BETA,
            lm_beam_width=cfg.MEDASR_LM_BEAM_WIDTH,
            min_text_chars=cfg.ASR_MIN_TEXT_CHARS,
        ),
        "mock": lambda: MockASRProvider(),
    }
    provider_primary = _get_or_create_provider(primary_name, cfg, builders=builders)

    fallback_available = False
    provider_fallback = None
    if fallback_enabled:
        if fallback_label == "whisper_cpp":
            fallback_available, fallback_reason = whisper_cpp_available(
                cfg.SCRIBE_WHISPER_CPP_BIN,
                cfg.SCRIBE_WHISPER_CPP_MODEL,
            )
            if fallback_available:
                provider_fallback = _get_or_create_provider(fallback_label, cfg, builders=builders)
        else:
            provider_fallback = _get_or_create_provider(fallback_label, cfg, builders=builders)
            fallback_available = True

    tmp_dir = (project_root / "tmp" / "evidentia_asr_adapter").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    store = InMemorySessionStore(ttl_seconds=cfg.SCRIBE_SESSION_TTL_SECONDS, tmp_dir=tmp_dir)
    session_id = store.create_session()
    try:
        result = transcribe_in_chunks(
            store=store,
            cfg=cfg,
            provider_primary=provider_primary,
            provider_fallback=provider_fallback,
            session_id=session_id,
            audio_path=audio_path,
            language=language,
            fallback_label=fallback_label,
            fallback_available=fallback_available,
            fallback_reason=fallback_reason,
        )
    except Exception as exc:
        raise RepoASRAdapterError(f"Repo chunk ASR failed: {exc}") from exc

    segments: list[ASRSegment] = []
    turns: list[SpeakerTurn] = []
    for item in list(result.segments or []):
        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        start = float(getattr(item, "start_sec", 0.0) or 0.0)
        end = float(getattr(item, "end_sec", start) or start)
        segments.append(
            ASRSegment(
                start=max(0.0, start),
                end=max(start, end),
                text=text,
                avg_logprob=None,
                no_speech_prob=None,
            )
        )
        speaker = normalize_speaker(getattr(item, "speaker", None))
        raw_conf = getattr(item, "confidence", 0.0)
        try:
            conf = float(raw_conf or 0.0)
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        turns.append(
            SpeakerTurn(
                start=max(0.0, start),
                end=max(start, end),
                speaker=speaker,
                speaker_id=speaker,
                speaker_role=speaker,
                confidence=conf,
                diar_confidence=conf,
                role_confidence=conf,
            )
        )

    if not segments:
        warnings = list(getattr(result, "warnings", []) or [])
        errors = list(getattr(result, "errors", []) or [])
        detail = "; ".join([*warnings, *errors]).strip() or "No segments produced."
        raise RepoASRAdapterError(detail)

    provider_name = primary_name
    debug = {
        "provider": provider_name,
        "provider_cached": provider_name in {"whisper_cpp", "medasr_hf"},
        "primary": primary_name,
        "fallback": fallback_label,
        "fallback_available": bool(fallback_available),
        "fallback_reason": fallback_reason,
        "explicit_provider_override": bool(explicit_override),
        "chunk_sec": cfg.ASR_CHUNK_SEC,
        "overlap_sec": cfg.ASR_OVERLAP_SEC,
        "warnings": list(getattr(result, "warnings", []) or []),
        "errors": list(getattr(result, "errors", []) or []),
        "meta": dict(getattr(result, "meta", {}) or {}),
    }
    return RepoASRResult(segments=segments, turns=turns, debug=debug)


def transcribe_audio_window_with_repo_chunk_pipeline(
    *,
    audio_path: str,
    language: str,
    start_at_sec: float,
    window_sec: float,
    provider: str | None = None,
    chunk_sec: int | None = None,
    overlap_sec: float | None = None,
) -> RepoASRResult:
    """
    Transcribe one fixed time window from audio and map timestamps back to absolute time.

    Design intent:
    - Enable chunk-by-chunk UI updates for `audio_path` mode without redesigning ASR internals.
    - Preserve provenance by keeping absolute timestamps in returned segments/turns.
    - Return explicit stream progress metadata so frontend can decide whether to continue polling.
    """

    if window_sec <= 0:
        raise RepoASRAdapterError("audio_window_sec must be > 0.")

    base_path = Path(audio_path).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    tmp_dir = (project_root / "tmp" / "evidentia_asr_adapter").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    session_prefix = f"evidentia_window_{uuid.uuid4().hex[:12]}"
    normalized_path: Path | None = None
    clip_path: Path | None = None
    try:
        normalized_path = normalize_to_wav16k_mono(base_path, tmp_dir=tmp_dir, session_prefix=session_prefix)
        audio_duration_sec, _, _ = load_wav_info(normalized_path)
        window_start_sec = max(0.0, float(start_at_sec))
        window_end_sec = min(float(audio_duration_sec), window_start_sec + float(window_sec))
        has_more = bool(window_end_sec < float(audio_duration_sec))

        if window_start_sec >= float(audio_duration_sec):
            return RepoASRResult(
                segments=[],
                turns=[],
                debug={
                    "provider": provider or "auto",
                    "stream": {
                        "start_at_sec": window_start_sec,
                        "window_sec": float(window_sec),
                        "window_end_sec": window_end_sec,
                        "next_start_sec": window_end_sec,
                        "audio_duration_sec": float(audio_duration_sec),
                        "has_more": False,
                    },
                    "warnings": ["window_start_out_of_range"],
                    "errors": [],
                },
            )

        audio = load_wav16k_mono_float32(normalized_path)
        frame_start = max(0, int(round(window_start_sec * 16000.0)))
        frame_end = min(int(round(window_end_sec * 16000.0)), int(audio.shape[0]))
        clip = audio[frame_start:frame_end]
        clip_path = tmp_dir / f"{session_prefix}_clip.wav"
        write_wav16k_mono_float32(clip_path, clip)

        try:
            clipped = transcribe_audio_with_repo_chunk_pipeline(
                audio_path=str(clip_path),
                language=language,
                provider=provider,
                chunk_sec=chunk_sec,
                overlap_sec=overlap_sec,
            )
            warnings = list(clipped.debug.get("warnings", []) or [])
            errors = list(clipped.debug.get("errors", []) or [])
            base_debug = dict(clipped.debug)
        except RepoASRAdapterError as exc:
            # Empty-window output should not break streaming loops.
            if "No segments produced" in str(exc):
                clipped = RepoASRResult(segments=[], turns=[], debug={})
                warnings = [str(exc)]
                errors = []
                base_debug = {}
            else:
                raise

        shifted_segments = [
            ASRSegment(
                start=max(0.0, item.start + window_start_sec),
                end=max(item.start + window_start_sec, item.end + window_start_sec),
                text=item.text,
                avg_logprob=item.avg_logprob,
                no_speech_prob=item.no_speech_prob,
            )
            for item in clipped.segments
        ]
        shifted_turns = [
            SpeakerTurn(
                start=max(0.0, item.start + window_start_sec),
                end=max(item.start + window_start_sec, item.end + window_start_sec),
                speaker=item.speaker,
                confidence=item.confidence,
            )
            for item in clipped.turns
        ]

        return RepoASRResult(
            segments=shifted_segments,
            turns=shifted_turns,
            debug={
                **base_debug,
                "stream": {
                    "start_at_sec": window_start_sec,
                    "window_sec": float(window_sec),
                    "window_end_sec": window_end_sec,
                    "next_start_sec": window_end_sec,
                    "audio_duration_sec": float(audio_duration_sec),
                    "has_more": has_more,
                },
                "warnings": warnings,
                "errors": errors,
            },
        )
    except RepoASRAdapterError:
        raise
    except Exception as exc:
        raise RepoASRAdapterError(f"Repo windowed ASR failed: {exc}") from exc
    finally:
        if clip_path is not None:
            clip_path.unlink(missing_ok=True)
        if normalized_path is not None and normalized_path != base_path:
            normalized_path.unlink(missing_ok=True)
