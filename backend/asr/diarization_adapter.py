from __future__ import annotations

"""
Lightweight diarization adapter for `audio_path` workflows.

Design intent:
- Provide a local, dependency-light fallback diarization path without blocking the MVP.
- Keep the pipeline explicit: VAD/SAD -> embedding -> clustering -> turn segmentation.
- Fail open: when diarization cannot run, return empty turns so caller can safely fallback to `other`.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
from time import perf_counter
from typing import Any, Sequence
import uuid

import numpy as np

from backend.asr.models import ASRSegment, SpeakerTurn
from backend.internal_core.audio_utils import load_wav16k_mono_float32, normalize_to_wav16k_mono


@dataclass(frozen=True)
class _EmbeddingChunk:
    start: float
    end: float
    vector: np.ndarray
    energy: float


@dataclass(frozen=True)
class _ClusteredChunk:
    start: float
    end: float
    cluster_id: int
    similarity: float
    energy: float


@dataclass(frozen=True)
class _DiarizationCacheValue:
    turns: list[SpeakerTurn]
    debug: dict[str, Any]


_CACHE_LOCK = threading.Lock()
_DIARIZATION_CACHE: dict[tuple[str, int, int], _DiarizationCacheValue] = {}
_MAX_CACHE_ITEMS = 12
_SR = 16000
_ROLE_CLINICIAN_RE = re.compile(
    r"\b(how|what|when|where|why|can you|could you|tell me|let us|let's|thanks for sharing|thank you for sharing)\b|\?",
    flags=re.IGNORECASE,
)
_ROLE_PATIENT_RE = re.compile(
    r"\b(i|i'm|i've|my|me|feel|feeling|sleep|anxious|depressed|pain|tired|wish i would not|wish i wouldn't)\b",
    flags=re.IGNORECASE,
)


def diarize_audio_with_debug(
    audio_path: str,
    *,
    asr_segments: Sequence[ASRSegment] | None = None,
    window_start_sec: float | None = None,
    window_end_sec: float | None = None,
) -> tuple[list[SpeakerTurn], dict[str, Any]]:
    """
    Run local diarization and return speaker turns plus debug metadata.

    Returns empty turns on runtime/setup errors so API handlers can fallback safely.
    """

    started = perf_counter()
    if not _env_enabled():
        return [], {"status": "disabled", "reason": "EVIDENTIA_DIARIZATION_ENABLED is false", "turns_count": 0}

    audio_file = Path(audio_path).expanduser().resolve()
    if not audio_file.exists():
        return [], {"status": "skipped", "reason": "audio_not_found", "turns_count": 0}

    cache_key = _cache_key(audio_file)
    cached = _cache_get(cache_key)
    if cached is not None:
        turns = _filter_turns_by_window(cached.turns, window_start_sec=window_start_sec, window_end_sec=window_end_sec)
        debug = {
            **dict(cached.debug),
            "cache_hit": True,
            "turns_count": len(turns),
            "window_applied": bool(window_start_sec is not None or window_end_sec is not None),
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 2),
        }
        return turns, debug

    normalized_path: Path | None = None
    try:
        audio, normalized_path = _load_audio_16k_mono(audio_file)
    except Exception as exc:
        return [], {"status": "error", "reason": f"audio_decode_failed: {exc}", "turns_count": 0}

    try:
        speech_segments, vad_debug = _detect_speech_segments(audio)
        chunks = _build_embedding_chunks(audio, speech_segments)
        clustered, cluster_debug = _cluster_chunks(chunks)
        turns = _build_turns(clustered)
        turns = _assign_roles(turns, asr_segments=asr_segments)
        base_debug = {
            "status": "ok",
            "provider": "energy_vad_embedding_cluster",
            "cache_hit": False,
            "audio_path": str(audio_file),
            "audio_duration_sec": round(float(audio.shape[0]) / float(_SR), 3),
            "speech_segments_count": len(speech_segments),
            "embedding_segments_count": len(chunks),
            "clusters_count": cluster_debug["clusters_count"],
            "cluster_threshold": cluster_debug["threshold"],
            "turns_count": len(turns),
            "vad": vad_debug,
        }
        _cache_put(cache_key, _DiarizationCacheValue(turns=turns, debug=base_debug))
        window_turns = _filter_turns_by_window(turns, window_start_sec=window_start_sec, window_end_sec=window_end_sec)
        return (
            window_turns,
            {
                **base_debug,
                "turns_count": len(window_turns),
                "window_applied": bool(window_start_sec is not None or window_end_sec is not None),
                "elapsed_ms": round((perf_counter() - started) * 1000.0, 2),
            },
        )
    except Exception as exc:
        return [], {"status": "error", "reason": f"runtime_failed: {exc}", "turns_count": 0}
    finally:
        if normalized_path is not None and normalized_path != audio_file:
            normalized_path.unlink(missing_ok=True)


def _env_enabled() -> bool:
    raw = os.getenv("EVIDENTIA_DIARIZATION_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _load_audio_16k_mono(audio_file: Path) -> tuple[np.ndarray, Path]:
    project_root = Path(__file__).resolve().parents[2]
    tmp_dir = (project_root / "tmp" / "evidentia_diarization").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_to_wav16k_mono(audio_file, tmp_dir=tmp_dir, session_prefix=f"evidentia_diar_{uuid.uuid4().hex[:10]}")
    audio = load_wav16k_mono_float32(normalized).astype(np.float32, copy=False)
    return audio, Path(normalized)


def _frame_signal(
    audio: np.ndarray,
    *,
    frame_len: int,
    hop_len: int,
) -> np.ndarray:
    if audio.size <= 0:
        return np.zeros((0, frame_len), dtype=np.float32)
    if audio.size < frame_len:
        padded = np.zeros((frame_len,), dtype=np.float32)
        padded[: audio.size] = audio
        return padded.reshape(1, frame_len)
    count = 1 + (audio.size - frame_len) // hop_len
    frames = np.zeros((count, frame_len), dtype=np.float32)
    for idx in range(count):
        start = idx * hop_len
        frames[idx] = audio[start : start + frame_len]
    return frames


def _smooth_vad(mask: np.ndarray, *, min_speech_frames: int, min_silence_frames: int) -> np.ndarray:
    if mask.size == 0:
        return mask
    out = mask.astype(bool, copy=True)
    idx = 0
    while idx < out.size:
        j = idx
        val = out[idx]
        while j < out.size and out[j] == val:
            j += 1
        run_len = j - idx
        if not val and run_len <= min_silence_frames and idx > 0 and j < out.size:
            out[idx:j] = True
        idx = j
    idx = 0
    while idx < out.size:
        j = idx
        val = out[idx]
        while j < out.size and out[j] == val:
            j += 1
        run_len = j - idx
        if val and run_len < min_speech_frames:
            out[idx:j] = False
        idx = j
    return out


def _detect_speech_segments(audio: np.ndarray) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    frame_sec = _env_float("EVIDENTIA_DIARIZATION_VAD_FRAME_SEC", 0.03)
    hop_sec = _env_float("EVIDENTIA_DIARIZATION_VAD_HOP_SEC", 0.01)
    min_speech_sec = _env_float("EVIDENTIA_DIARIZATION_MIN_SPEECH_SEC", 0.35)
    min_silence_sec = _env_float("EVIDENTIA_DIARIZATION_MIN_SILENCE_SEC", 0.20)

    frame_len = max(64, int(round(frame_sec * _SR)))
    hop_len = max(32, int(round(hop_sec * _SR)))
    frames = _frame_signal(audio, frame_len=frame_len, hop_len=hop_len)
    if frames.shape[0] == 0:
        return [], {"threshold": 0.0, "frame_sec": frame_sec, "hop_sec": hop_sec}

    window = np.hanning(frame_len).astype(np.float32)
    energies = np.sqrt(np.maximum(1e-12, np.mean((frames * window) ** 2, axis=1)))
    p20 = float(np.percentile(energies, 20))
    p95 = float(np.percentile(energies, 95))
    threshold = max(0.0018, p20 + max(0.0008, 0.20 * (p95 - p20)))
    raw_mask = energies >= threshold

    min_speech_frames = max(1, int(round(min_speech_sec / max(1e-6, hop_sec))))
    min_silence_frames = max(1, int(round(min_silence_sec / max(1e-6, hop_sec))))
    mask = _smooth_vad(raw_mask, min_speech_frames=min_speech_frames, min_silence_frames=min_silence_frames)

    segments: list[tuple[float, float]] = []
    idx = 0
    while idx < mask.size:
        if not mask[idx]:
            idx += 1
            continue
        j = idx
        while j < mask.size and mask[j]:
            j += 1
        start = idx * hop_sec
        end = min(float(audio.shape[0]) / float(_SR), (j * hop_sec) + frame_sec)
        if end - start >= min_speech_sec:
            segments.append((round(start, 3), round(end, 3)))
        idx = j

    return segments, {
        "threshold": round(threshold, 6),
        "frame_sec": frame_sec,
        "hop_sec": hop_sec,
        "min_speech_sec": min_speech_sec,
        "min_silence_sec": min_silence_sec,
    }


def _segment_embedding(audio_slice: np.ndarray) -> np.ndarray | None:
    if audio_slice.size < int(0.25 * _SR):
        return None

    frame_len = 400
    hop_len = 160
    n_fft = 512
    frames = _frame_signal(audio_slice, frame_len=frame_len, hop_len=hop_len)
    if frames.shape[0] == 0:
        return None

    window = np.hanning(frame_len).astype(np.float32)
    fft_mag = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=1)).astype(np.float32)
    power = np.maximum(1e-9, (fft_mag**2))
    log_power = np.log(power)

    bins = 24
    per_bin = max(1, log_power.shape[1] // bins)
    band_means = []
    for idx in range(bins):
        start = idx * per_bin
        end = min(log_power.shape[1], (idx + 1) * per_bin)
        if start >= end:
            break
        band_means.append(float(np.mean(log_power[:, start:end])))

    freqs = np.linspace(0.0, _SR * 0.5, num=log_power.shape[1], dtype=np.float32)
    power_sum = np.maximum(1e-9, np.sum(power, axis=1))
    centroid = np.sum(power * freqs[None, :], axis=1) / power_sum
    rolloff_target = 0.85 * power_sum
    cumulative = np.cumsum(power, axis=1)
    rolloff_idx = np.argmax(cumulative >= rolloff_target[:, None], axis=1)
    rolloff_hz = freqs[np.clip(rolloff_idx, 0, freqs.shape[0] - 1)]

    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1).astype(np.float32)
    rms = np.sqrt(np.maximum(1e-9, np.mean(frames**2, axis=1)))

    feature = np.array(
        [
            *band_means,
            float(np.mean(centroid)),
            float(np.std(centroid)),
            float(np.mean(rolloff_hz)),
            float(np.std(rolloff_hz)),
            float(np.mean(zcr)),
            float(np.std(zcr)),
            float(np.mean(rms)),
            float(np.std(rms)),
        ],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-8:
        return None
    return feature / norm


def _build_embedding_chunks(
    audio: np.ndarray,
    speech_segments: Sequence[tuple[float, float]],
) -> list[_EmbeddingChunk]:
    max_seg_sec = _env_float("EVIDENTIA_DIARIZATION_EMBED_SEC", 1.6)
    overlap_sec = _env_float("EVIDENTIA_DIARIZATION_EMBED_OVERLAP_SEC", 0.3)
    step_sec = max(0.2, max_seg_sec - overlap_sec)

    chunks: list[_EmbeddingChunk] = []
    for seg_start, seg_end in speech_segments:
        cursor = float(seg_start)
        while cursor < seg_end:
            chunk_end = min(seg_end, cursor + max_seg_sec)
            if chunk_end - cursor < 0.25:
                break
            frame_start = max(0, int(round(cursor * _SR)))
            frame_end = min(audio.shape[0], int(round(chunk_end * _SR)))
            if frame_end <= frame_start:
                break
            piece = audio[frame_start:frame_end]
            vector = _segment_embedding(piece)
            if vector is not None:
                chunks.append(
                    _EmbeddingChunk(
                        start=round(cursor, 3),
                        end=round(chunk_end, 3),
                        vector=vector,
                        energy=float(np.sqrt(np.maximum(1e-9, np.mean(piece**2)))),
                    )
                )
            cursor += step_sec
    return chunks


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _cluster_chunks(chunks: Sequence[_EmbeddingChunk]) -> tuple[list[_ClusteredChunk], dict[str, Any]]:
    if not chunks:
        return [], {"clusters_count": 0, "threshold": 0.0}

    max_speakers = max(1, _env_int("EVIDENTIA_DIARIZATION_MAX_SPEAKERS", 2))
    threshold = _env_float("EVIDENTIA_DIARIZATION_CLUSTER_SIMILARITY", 0.82)

    centroids: list[np.ndarray] = []
    counts: list[int] = []
    assigned: list[_ClusteredChunk] = []

    for chunk in chunks:
        best_idx = -1
        best_sim = -1.0
        for idx, centroid in enumerate(centroids):
            sim = _cosine_similarity(chunk.vector, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx == -1 or (best_sim < threshold and len(centroids) < max_speakers):
            centroids.append(chunk.vector.copy())
            counts.append(1)
            cluster_id = len(centroids) - 1
            similarity = 1.0
        else:
            cluster_id = best_idx
            similarity = best_sim if best_sim >= 0.0 else 0.0
            prev_count = counts[cluster_id]
            centroids[cluster_id] = (centroids[cluster_id] * prev_count + chunk.vector) / float(prev_count + 1)
            counts[cluster_id] = prev_count + 1

        assigned.append(
            _ClusteredChunk(
                start=chunk.start,
                end=chunk.end,
                cluster_id=cluster_id,
                similarity=float(np.clip(similarity, 0.0, 1.0)),
                energy=chunk.energy,
            )
        )

    return assigned, {"clusters_count": len(centroids), "threshold": threshold}


def _build_turns(chunks: Sequence[_ClusteredChunk]) -> list[SpeakerTurn]:
    if not chunks:
        return []

    sorted_chunks = sorted(chunks, key=lambda item: (item.start, item.end))
    gap_merge_sec = _env_float("EVIDENTIA_DIARIZATION_TURN_GAP_SEC", 0.35)

    merged: list[dict[str, Any]] = []
    for item in sorted_chunks:
        if not merged:
            merged.append(
                {
                    "start": item.start,
                    "end": item.end,
                    "cluster_id": item.cluster_id,
                    "weights": [max(0.1, item.end - item.start)],
                    "sims": [item.similarity],
                }
            )
            continue
        prev = merged[-1]
        if item.cluster_id == prev["cluster_id"] and (item.start - prev["end"]) <= gap_merge_sec:
            prev["end"] = max(prev["end"], item.end)
            prev["weights"].append(max(0.1, item.end - item.start))
            prev["sims"].append(item.similarity)
            continue
        merged.append(
            {
                "start": item.start,
                "end": item.end,
                "cluster_id": item.cluster_id,
                "weights": [max(0.1, item.end - item.start)],
                "sims": [item.similarity],
            }
        )

    first_seen: dict[int, float] = {}
    for item in merged:
        cid = int(item["cluster_id"])
        first_seen.setdefault(cid, float(item["start"]))
    cluster_order = sorted(first_seen.items(), key=lambda pair: pair[1])
    speaker_ids = {cid: f"spk_{idx + 1:02d}" for idx, (cid, _start) in enumerate(cluster_order)}

    turns: list[SpeakerTurn] = []
    for item in merged:
        weights = np.asarray(item["weights"], dtype=np.float32)
        sims = np.asarray(item["sims"], dtype=np.float32)
        conf = float(np.average(sims, weights=weights)) if weights.size else float(np.mean(sims))
        conf = float(np.clip(conf, 0.0, 1.0))
        cluster_id = int(item["cluster_id"])
        turns.append(
            SpeakerTurn(
                start=float(item["start"]),
                end=float(item["end"]),
                speaker="other",
                speaker_id=speaker_ids.get(cluster_id, f"spk_{cluster_id + 1:02d}"),
                speaker_role="other",
                confidence=conf,
                diar_confidence=conf,
            )
        )

    return turns


def _assign_roles(turns: list[SpeakerTurn], *, asr_segments: Sequence[ASRSegment] | None) -> list[SpeakerTurn]:
    if not turns:
        return []

    by_id: dict[str, list[SpeakerTurn]] = {}
    for turn in turns:
        sid = str(turn.speaker_id or "spk_00")
        by_id.setdefault(sid, []).append(turn)

    durations = {
        sid: sum(max(0.0, item.end - item.start) for item in items)
        for sid, items in by_id.items()
    }
    ranked = sorted(durations.items(), key=lambda pair: (-pair[1], pair[0]))
    speaker_ids = [sid for sid, _dur in ranked]
    role_map: dict[str, str] = {sid: "other" for sid in speaker_ids}

    if len(speaker_ids) == 1:
        sid = speaker_ids[0]
        if asr_segments:
            scores = _speaker_text_scores(by_id, asr_segments).get(sid, {"clinician": 0, "patient": 0})
            if scores["clinician"] > scores["patient"]:
                role_map[sid] = "clinician"
            else:
                role_map[sid] = "patient"
            rebuilt = _rebuild_single_speaker_turns_from_segments(
                speaker_id=sid,
                base_turns=by_id.get(sid, []),
                asr_segments=asr_segments,
            )
            if rebuilt:
                return rebuilt
        else:
            role_map[sid] = "patient"
    elif len(speaker_ids) >= 2:
        role_map[speaker_ids[0]] = "patient"
        role_map[speaker_ids[1]] = "clinician"

    if asr_segments:
        text_stats = _speaker_text_scores(by_id, asr_segments)
        clinician_id = None
        patient_id = None
        for sid, scores in text_stats.items():
            if scores["clinician"] > scores["patient"] and clinician_id is None:
                clinician_id = sid
            if scores["patient"] > scores["clinician"] and patient_id is None:
                patient_id = sid
        if clinician_id and patient_id and clinician_id != patient_id:
            role_map[clinician_id] = "clinician"
            role_map[patient_id] = "patient"

    out: list[SpeakerTurn] = []
    for turn in turns:
        sid = str(turn.speaker_id or "spk_00")
        role = role_map.get(sid, "other")
        if asr_segments:
            turn_scores = _turn_text_scores(turn, asr_segments)
            if turn_scores["clinician"] > turn_scores["patient"] and turn_scores["clinician"] >= 1:
                role = "clinician"
            elif turn_scores["patient"] > turn_scores["clinician"] and turn_scores["patient"] >= 1:
                role = "patient"
        out.append(
            SpeakerTurn(
                start=turn.start,
                end=turn.end,
                speaker=role,
                speaker_id=sid,
                speaker_role=role,
                confidence=turn.confidence,
                diar_confidence=turn.diar_confidence if turn.diar_confidence is not None else turn.confidence,
            )
        )
    return out


def _speaker_text_scores(
    by_id: dict[str, list[SpeakerTurn]],
    segments: Sequence[ASRSegment],
) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {
        sid: {"clinician": 0, "patient": 0}
        for sid in by_id
    }
    for seg in segments:
        mid = (float(seg.start) + float(seg.end)) * 0.5
        text = str(seg.text or "")
        if not text:
            continue
        sid = None
        for speaker_id, turns in by_id.items():
            for turn in turns:
                if mid >= turn.start and mid <= turn.end:
                    sid = speaker_id
                    break
            if sid is not None:
                break
        if sid is None:
            continue
        clinician_hits, patient_hits = _score_role_text(text)
        stats[sid]["clinician"] += clinician_hits
        stats[sid]["patient"] += patient_hits
    return stats


def _score_role_text(text: str) -> tuple[int, int]:
    source = str(text or "")
    if not source:
        return 0, 0
    clinician_hits = len(_ROLE_CLINICIAN_RE.findall(source))
    patient_hits = len(_ROLE_PATIENT_RE.findall(source))
    return int(clinician_hits), int(patient_hits)


def _turn_text_scores(
    turn: SpeakerTurn,
    segments: Sequence[ASRSegment],
) -> dict[str, int]:
    clinician = 0
    patient = 0
    for seg in segments:
        overlap = max(0.0, min(turn.end, seg.end) - max(turn.start, seg.start))
        if overlap <= 0.05:
            continue
        c_hits, p_hits = _score_role_text(seg.text)
        clinician += c_hits
        patient += p_hits
    return {"clinician": clinician, "patient": patient}


def _rebuild_single_speaker_turns_from_segments(
    *,
    speaker_id: str,
    base_turns: Sequence[SpeakerTurn],
    asr_segments: Sequence[ASRSegment],
) -> list[SpeakerTurn]:
    ordered_segments = sorted(asr_segments, key=lambda item: (item.start, item.end))
    if not ordered_segments:
        return []
    conf_values = [
        float(item.diar_confidence if item.diar_confidence is not None else item.confidence or 0.0)
        for item in base_turns
    ]
    base_conf = float(np.mean(conf_values)) if conf_values else 0.75
    raw_turns: list[SpeakerTurn] = []
    for seg in ordered_segments:
        text = str(seg.text or "").strip()
        if not text:
            continue
        c_hits, p_hits = _score_role_text(text)
        if c_hits > p_hits:
            role = "clinician"
        elif p_hits > c_hits:
            role = "patient"
        else:
            role = "patient"
        raw_turns.append(
            SpeakerTurn(
                start=float(seg.start),
                end=float(seg.end),
                speaker=role,
                speaker_id=speaker_id,
                speaker_role=role,
                confidence=base_conf,
                diar_confidence=base_conf,
            )
        )

    if not raw_turns:
        return []

    merged: list[SpeakerTurn] = [raw_turns[0]]
    for item in raw_turns[1:]:
        prev = merged[-1]
        if item.speaker == prev.speaker and (item.start - prev.end) <= 0.35:
            merged[-1] = SpeakerTurn(
                start=prev.start,
                end=max(prev.end, item.end),
                speaker=prev.speaker,
                speaker_id=prev.speaker_id,
                speaker_role=prev.speaker_role,
                confidence=prev.confidence,
                diar_confidence=prev.diar_confidence,
            )
            continue
        merged.append(item)
    return merged


def _filter_turns_by_window(
    turns: Sequence[SpeakerTurn],
    *,
    window_start_sec: float | None,
    window_end_sec: float | None,
) -> list[SpeakerTurn]:
    if window_start_sec is None and window_end_sec is None:
        return list(turns)
    start = float(window_start_sec) if window_start_sec is not None else float("-inf")
    end = float(window_end_sec) if window_end_sec is not None else float("inf")
    return [item for item in turns if item.end > start and item.start < end]


def _cache_key(audio_file: Path) -> tuple[str, int, int]:
    stat = audio_file.stat()
    return (str(audio_file), int(stat.st_mtime_ns), int(stat.st_size))


def _cache_get(key: tuple[str, int, int]) -> _DiarizationCacheValue | None:
    with _CACHE_LOCK:
        return _DIARIZATION_CACHE.get(key)


def _cache_put(key: tuple[str, int, int], value: _DiarizationCacheValue) -> None:
    with _CACHE_LOCK:
        _DIARIZATION_CACHE[key] = value
        if len(_DIARIZATION_CACHE) <= _MAX_CACHE_ITEMS:
            return
        for old_key in list(_DIARIZATION_CACHE.keys())[: len(_DIARIZATION_CACHE) - _MAX_CACHE_ITEMS]:
            _DIARIZATION_CACHE.pop(old_key, None)
