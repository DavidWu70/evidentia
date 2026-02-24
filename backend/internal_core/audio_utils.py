from __future__ import annotations

import math
import os
import shutil
import subprocess
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


ALLOWED_UPLOAD_EXTS = {".wav", ".mp3"}


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def generate_default_wav_samples(sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for name, freq in [("sample_a.wav", 440.0), ("sample_b.wav", 554.37)]:
        path = sample_dir / name
        if path.exists():
            continue
        _write_tone_wav(path, duration_sec=6.0, freq_hz=freq)


def _write_tone_wav(path: Path, duration_sec: float, freq_hz: float) -> None:
    sample_rate = 16000
    amplitude = 0.15
    frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(frames):
            t = i / sample_rate
            sample = amplitude * math.sin(2.0 * math.pi * freq_hz * t)
            s16 = int(max(-1.0, min(1.0, sample)) * 32767)
            wf.writeframesraw(int(s16).to_bytes(2, byteorder="little", signed=True))


def list_samples(sample_dir: Path) -> List[Dict[str, Any]]:
    if not sample_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(sample_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        duration = None
        try:
            if p.suffix.lower() == ".wav":
                duration, _, _ = load_wav_info(p)
        except Exception:
            duration = None
        out.append(
            {"name": p.name, "path": str(p), "duration_sec": duration}
        )
    return out


def load_wav_info(path: Path) -> Tuple[float, int, int]:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        channels = wf.getnchannels()
        duration = frames / float(rate) if rate else 0.0
        return duration, rate, channels


def enforce_max_duration(duration_sec: float, max_allowed: float) -> None:
    if duration_sec > max_allowed:
        raise ValueError(
            f"Audio too long ({duration_sec:.1f}s), max allowed is {max_allowed:.1f}s"
        )


def enforce_max_size_bytes(path: Path, max_bytes: int) -> None:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"Audio file too large ({size / (1024 * 1024):.1f}MB), "
            f"max allowed is {max_bytes / (1024 * 1024):.1f}MB"
        )


def normalize_to_wav16k_mono(
    input_path: Path,
    tmp_dir: Path,
    session_prefix: str,
    *,
    max_bytes: int = 200 * 1024 * 1024,
) -> Path:
    """
    Normalize any supported audio to 16kHz mono WAV.
    Prefers ffmpeg when present; falls back to `miniaudio` decode/convert.
    """
    if not input_path.exists():
        raise ValueError(f"Audio file not found: {input_path}")

    enforce_max_size_bytes(input_path, max_bytes=max_bytes)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".wav":
        try:
            _, sr, ch = load_wav_info(input_path)
            if int(sr) == 16000 and int(ch) == 1:
                return input_path
        except Exception:
            pass

    out_path = tmp_dir / f"{session_prefix}_norm_{uuid.uuid4().hex}.wav"

    ffmpeg = _which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return out_path
        except subprocess.CalledProcessError as e:
            stderr = (
                e.stderr.decode("utf-8", "ignore")
                if isinstance(e.stderr, (bytes, bytearray))
                else str(e.stderr)
            )
            raise ValueError(
                f"Audio conversion failed via ffmpeg: {stderr.strip() or 'unknown error'}"
            )

    try:
        import miniaudio  # type: ignore
    except Exception:
        raise ValueError(
            "Audio conversion requires `ffmpeg` or the Python dependency `miniaudio`."
        )

    try:
        decoded = miniaudio.decode_file(
            str(input_path),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=16000,
        )
        pcm_bytes = decoded.samples.tobytes()
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm_bytes)
        return out_path
    except Exception as e:
        raise ValueError(f"Audio conversion failed: {e}")


def load_wav16k_mono_float32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()
        frames = wf.getnframes()
        if channels != 1:
            raise ValueError(f"Expected mono WAV, got {channels} channels")
        if rate != 16000:
            raise ValueError(f"Expected 16kHz WAV, got {rate}Hz")
        if width != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got sampwidth={width}")
        raw = wf.readframes(frames)
    audio_i16 = np.frombuffer(raw, dtype="<i2")
    audio = (audio_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return audio


def compute_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    x = audio.astype(np.float32)
    return float(np.sqrt(np.mean(x * x)))


def normalize_min_rms(audio: np.ndarray, target_min_rms: float) -> np.ndarray:
    if target_min_rms <= 0:
        return audio
    rms = compute_rms(audio)
    if rms <= 0 or rms >= target_min_rms:
        return audio
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return audio
    gain = target_min_rms / max(rms, 1e-12)
    max_gain = 0.99 / peak
    gain = min(gain, max_gain)
    if gain <= 1.0:
        return audio
    out = (audio * gain).clip(-1.0, 1.0)
    return out.astype(np.float32)


def write_wav16k_mono_float32(path: Path, audio: np.ndarray) -> None:
    audio = np.asarray(audio, dtype=np.float32).clip(-1.0, 1.0)
    audio_i16 = (audio * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_i16.tobytes())


def ensure_wav(input_path: Path, tmp_dir: Path, session_prefix: str) -> Path:
    if input_path.suffix.lower() == ".wav":
        return input_path

    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{session_prefix}_converted_{uuid.uuid4().hex}.wav"

    # Prefer ffmpeg if present (fast, reliable).
    ffmpeg = _which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return out_path
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr.decode("utf-8", "ignore") if isinstance(e.stderr, (bytes, bytearray)) else str(e.stderr))
            raise ValueError(f"Audio conversion failed via ffmpeg: {stderr.strip() or 'unknown error'}")

    # Fallback: pure-Python-ish decode via miniaudio (no external binary).
    try:
        import miniaudio  # type: ignore
    except Exception:
        raise ValueError(
            f"Cannot transcribe {input_path.suffix} without a decoder. "
            "Install `ffmpeg` (recommended) or install Python dependency `miniaudio`."
        )

    try:
        decoded = miniaudio.decode_file(
            str(input_path),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=16000,
        )
        # decoded.samples is an array('h') for SIGNED16
        pcm_bytes = decoded.samples.tobytes()
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm_bytes)
        return out_path
    except Exception as e:
        raise ValueError(f"Audio conversion failed: {e}")


def chunk_wav(
    path: Path,
    chunk_seconds: int,
    tmp_dir: Path,
    prefix: str,
    overlap_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[Dict[str, Any]] = []
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()

        frames_per_chunk = max(1, int(chunk_seconds * framerate))
        overlap_frames = int(overlap_seconds * framerate)
        if overlap_frames >= frames_per_chunk:
            overlap_frames = max(0, frames_per_chunk - 1)
        step_frames = max(1, frames_per_chunk - overlap_frames)

        idx = 0
        start_frame = 0
        while start_frame < nframes or (nframes == 0 and idx == 0):
            end_frame = min(nframes, start_frame + frames_per_chunk)
            wf.setpos(min(start_frame, nframes))
            frames_to_read = max(0, end_frame - start_frame)
            raw = wf.readframes(frames_to_read)

            chunk_path = tmp_dir / f"{prefix}_chunk_{idx:04d}.wav"
            with wave.open(str(chunk_path), "wb") as out_wf:
                out_wf.setnchannels(nchannels)
                out_wf.setsampwidth(sampwidth)
                out_wf.setframerate(framerate)
                out_wf.writeframes(raw)

            start_sec = start_frame / float(framerate) if framerate else 0.0
            end_sec = end_frame / float(framerate) if framerate else 0.0
            chunks.append(
                {
                    "chunk_path": str(chunk_path),
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                    "index": idx,
                }
            )
            if end_frame >= nframes:
                break
            start_frame += step_frames
            idx += 1
    return chunks


def safe_save_upload(
    uploaded_file: Any, tmp_dir: Path, session_prefix: str, max_bytes: int = 50 * 1024 * 1024
) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)

    name = getattr(uploaded_file, "name", "") or ""
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise ValueError(f"Unsupported upload type: {ext}. Only WAV/MP3 are allowed.")

    data = uploaded_file.getvalue()
    if len(data) > max_bytes:
        raise ValueError("Upload too large for demo build.")

    out_path = tmp_dir / f"{session_prefix}_upload{ext}"
    with open(out_path, "wb") as f:
        f.write(data)

    # Validate decodeability when WAV; MP3 will be decoded later during conversion.
    if ext == ".wav":
        with wave.open(str(out_path), "rb"):
            pass
    return out_path
