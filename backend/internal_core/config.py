from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _project_root() -> Path:
    # backend/internal_core/config.py -> backend -> evidentia
    return Path(__file__).resolve().parents[2]


def _resolve_default_path(candidates: list[Path]) -> str:
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists():
            return str(resolved)
    # Keep deterministic fallback even when file is absent.
    if candidates:
        return str(candidates[0].expanduser().resolve())
    return ""


def _resolve_existing_path_or_empty(candidates: list[Path]) -> str:
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists():
            return str(resolved)
    return ""


def _model_root_from_env() -> Optional[Path]:
    raw = os.getenv("EVIDENTIA_MODEL_ROOT", "").strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except Exception:
        return None


def _getenv_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value


def _getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _getenv_opt_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value)


def _getenv_opt_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_set(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value != ""


def _getenv_int_preset(name: str, default: int, preset_value: Optional[int]) -> int:
    if _env_set(name):
        return _getenv_int(name, default)
    if preset_value is not None:
        return int(preset_value)
    return default


def _getenv_float_preset(name: str, default: float, preset_value: Optional[float]) -> float:
    if _env_set(name):
        return _getenv_float(name, default)
    if preset_value is not None:
        return float(preset_value)
    return default


def _getenv_bool_preset(name: str, default: bool, preset_value: Optional[bool]) -> bool:
    if _env_set(name):
        return _getenv_bool(name, default)
    if preset_value is not None:
        return bool(preset_value)
    return default


def _getenv_str_preset(name: str, default: str, preset_value: Optional[str]) -> str:
    if _env_set(name):
        return _getenv_str(name, default)
    if preset_value is not None:
        return str(preset_value)
    return default


def _preset_overrides(name: str) -> dict[str, object]:
    if name == "clinical_low_volume_noisy_v1":
        return {
            "ASR_SILENCE_RMS": 0.006,
            "ASR_TARGET_MIN_RMS": 0.015,
            "ASR_MIN_TEXT_CHARS": 1,
            "MEDASR_USE_LM": True,
            "MEDASR_LM_ALPHA": 0.5,
            "MEDASR_LM_BETA": 1.0,
        }
    return {}


@dataclass(frozen=True)
class DemoConfig:
    SCRIBE_DEMO_PASSWORD: str
    SCRIBE_UPLOAD_ENABLED: bool
    SCRIBE_SAMPLE_DIR: str
    SCRIBE_TMP_DIR: str
    SCRIBE_SESSION_TTL_SECONDS: int
    SCRIBE_ASR_PROVIDER: str
    SCRIBE_ASR_PRIMARY: str
    SCRIBE_ASR_FALLBACK: str
    SCRIBE_MEDASR_MODEL: str
    SCRIBE_MEDASR_DEVICE: str
    SCRIBE_MEDASR_FP16: bool
    SCRIBE_WHISPER_CPP_BIN: str
    SCRIBE_WHISPER_CPP_MODEL: str
    SCRIBE_WHISPER_CPP_NO_GPU: bool
    SCRIBE_LLM_BACKEND: str
    SCRIBE_OLLAMA_MODEL: str
    SCRIBE_LLAMA_CPP_BIN: str
    SCRIBE_LLAMA_CPP_MODEL: str
    SCRIBE_LLM_MAX_TOKENS: int
    SCRIBE_LLAMA_CPP_N_CTX: int
    SCRIBE_LLAMA_CPP_N_GPU_LAYERS: int
    SCRIBE_LLAMA_CPP_N_THREADS: Optional[int]
    SCRIBE_LLAMA_CPP_N_BATCH: Optional[int]
    SCRIBE_LLAMA_CPP_N_UBATCH: Optional[int]
    SCRIBE_LLAMA_CPP_CHAT_FORMAT: str
    SCRIBE_LLM_SAVE_RAW: bool
    SCRIBE_FACT_PIPELINE: bool
    SCRIBE_FACT_MODE: str
    SCRIBE_EP_ENABLE_SLICING: Optional[bool]
    SCRIBE_EP_SLICING_PROTOCOL: str
    SCRIBE_TEMPLATE_PATH: str
    SCRIBE_EXTRACT_PROMPT_VERSION: str
    SCRIBE_RENDER_PROMPT_VERSION: str
    SCRIBE_CHUNK_SECONDS: int
    SCRIBE_CHUNK_OVERLAP_SECONDS: float
    SCRIBE_MAX_AUDIO_SECONDS: int
    SCRIBE_LOG_LEVEL: str
    ASR_PRESET: str
    ASR_CHUNK_SEC: int
    ASR_OVERLAP_SEC: float
    ASR_MIN_TEXT_CHARS: int
    ASR_SILENCE_RMS: float
    ASR_TARGET_MIN_RMS: float
    MEDASR_USE_LM: bool
    MEDASR_LM_PATH: str
    MEDASR_LM_ALPHA: float
    MEDASR_LM_BETA: float
    MEDASR_LM_BEAM_WIDTH: int
    ASR_PRIMARY_PROVIDER: str
    ASR_FALLBACK_PROVIDER: str
    SCRIBE_EVIDENCE_REQUIRED: bool
    SCRIBE_EVIDENCE_WEAK_THRESHOLD: float
    SCRIBE_TEST_INJECT_ASR_FAIL_AT_CHUNK: Optional[int]
    SCRIBE_TEST_INJECT_LLM_FAIL: bool

    def sample_dir_path(self, repo_root: Path) -> Path:
        return (repo_root / self.SCRIBE_SAMPLE_DIR).resolve()

    def tmp_dir_path(self, repo_root: Path) -> Path:
        return (repo_root / self.SCRIBE_TMP_DIR).resolve()


def load_config() -> DemoConfig:
    project_root = _project_root()
    model_root = _model_root_from_env()

    asr_preset = _getenv_str("ASR_PRESET", "")
    preset = _preset_overrides(asr_preset)

    asr_chunk_sec = _getenv_int_preset(
        "ASR_CHUNK_SEC",
        _getenv_int("SCRIBE_CHUNK_SECONDS", 8),
        preset.get("ASR_CHUNK_SEC"),
    )
    asr_overlap_sec = _getenv_float_preset(
        "ASR_OVERLAP_SEC",
        _getenv_float("SCRIBE_CHUNK_OVERLAP_SECONDS", 0.8),
        preset.get("ASR_OVERLAP_SEC"),
    )
    asr_primary = _getenv_str("ASR_PRIMARY_PROVIDER", _getenv_str("SCRIBE_ASR_PRIMARY", "whisper_cpp"))
    asr_fallback = _getenv_str("ASR_FALLBACK_PROVIDER", _getenv_str("SCRIBE_ASR_FALLBACK", "medasr_hf"))

    model_prefixes: list[Path] = []
    if model_root is not None:
        model_prefixes.append(model_root)
    model_prefixes.extend([project_root, project_root / "models", project_root.parent])

    default_medasr_model = _resolve_default_path(
        [
            base / "MedASR"
            for base in model_prefixes
        ]
    )
    default_whisper_bin = _resolve_default_path(
        [
            base / "whisper.cpp" / "build" / "bin" / "whisper-cli"
            for base in model_prefixes
        ]
    )
    default_whisper_model = _resolve_default_path(
        [
            base / "whisper.cpp" / "models" / "ggml-small.en.bin"
            for base in model_prefixes
        ] + [
            base / "ggml-small.en.bin" for base in model_prefixes
        ]
    )
    default_medasr_lm = _resolve_default_path(
        [
            base / "MedASR" / "lm_6.kenlm"
            for base in model_prefixes
        ]
    )
    default_llama_cpp_model = _resolve_existing_path_or_empty(
        [
            base / "MedGemma" / "medgemma-1.5-4b-it-Q5_K_M.gguf"
            for base in model_prefixes
        ] + [
            base / "medgemma-1.5-4b-it-Q5_K_M.gguf" for base in model_prefixes
        ]
    )

    return DemoConfig(
        SCRIBE_DEMO_PASSWORD=_getenv_str("SCRIBE_DEMO_PASSWORD", ""),
        SCRIBE_UPLOAD_ENABLED=_getenv_bool("SCRIBE_UPLOAD_ENABLED", True),
        SCRIBE_SAMPLE_DIR=_getenv_str("SCRIBE_SAMPLE_DIR", "assets/samples"),
        SCRIBE_TMP_DIR=_getenv_str("SCRIBE_TMP_DIR", "./tmp"),
        SCRIBE_SESSION_TTL_SECONDS=_getenv_int("SCRIBE_SESSION_TTL_SECONDS", 14400),
        SCRIBE_ASR_PROVIDER=_getenv_str("SCRIBE_ASR_PROVIDER", "mock"),
        SCRIBE_ASR_PRIMARY=_getenv_str("SCRIBE_ASR_PRIMARY", "whisper_cpp"),
        SCRIBE_ASR_FALLBACK=_getenv_str("SCRIBE_ASR_FALLBACK", "medasr_hf"),
        SCRIBE_MEDASR_MODEL=_getenv_str(
            "SCRIBE_MEDASR_MODEL",
            _getenv_str("EVIDENTIA_MEDASR_MODEL", default_medasr_model),
        ),
        SCRIBE_MEDASR_DEVICE=_getenv_str("SCRIBE_MEDASR_DEVICE", ""),
        SCRIBE_MEDASR_FP16=_getenv_bool("SCRIBE_MEDASR_FP16", False),
        SCRIBE_WHISPER_CPP_BIN=_getenv_str(
            "SCRIBE_WHISPER_CPP_BIN",
            _getenv_str("EVIDENTIA_WHISPER_CPP_BIN", default_whisper_bin),
        ),
        SCRIBE_WHISPER_CPP_MODEL=_getenv_str(
            "SCRIBE_WHISPER_CPP_MODEL",
            _getenv_str("EVIDENTIA_WHISPER_CPP_MODEL", default_whisper_model),
        ),
        SCRIBE_WHISPER_CPP_NO_GPU=_getenv_bool("SCRIBE_WHISPER_CPP_NO_GPU", False),
        SCRIBE_LLM_BACKEND=_getenv_str("SCRIBE_LLM_BACKEND", "llama_cpp"),
        SCRIBE_OLLAMA_MODEL=_getenv_str("SCRIBE_OLLAMA_MODEL", "medgemma:4b-instruct"),
        SCRIBE_LLAMA_CPP_BIN=_getenv_str("SCRIBE_LLAMA_CPP_BIN", ""),
        SCRIBE_LLAMA_CPP_MODEL=_getenv_str(
            "SCRIBE_LLAMA_CPP_MODEL",
            _getenv_str("EVIDENTIA_MEDGEMMA_GGUF", default_llama_cpp_model),
        ),
        SCRIBE_LLM_MAX_TOKENS=_getenv_int("SCRIBE_LLM_MAX_TOKENS", 320),
        SCRIBE_LLAMA_CPP_N_CTX=_getenv_int("SCRIBE_LLAMA_CPP_N_CTX", 2048),
        SCRIBE_LLAMA_CPP_N_GPU_LAYERS=_getenv_int("SCRIBE_LLAMA_CPP_N_GPU_LAYERS", -1),
        SCRIBE_LLAMA_CPP_N_THREADS=_getenv_opt_int("SCRIBE_LLAMA_CPP_N_THREADS"),
        SCRIBE_LLAMA_CPP_N_BATCH=_getenv_opt_int("SCRIBE_LLAMA_CPP_N_BATCH"),
        SCRIBE_LLAMA_CPP_N_UBATCH=_getenv_opt_int("SCRIBE_LLAMA_CPP_N_UBATCH"),
        SCRIBE_LLAMA_CPP_CHAT_FORMAT=_getenv_str("SCRIBE_LLAMA_CPP_CHAT_FORMAT", "gemma"),
        SCRIBE_LLM_SAVE_RAW=_getenv_bool("SCRIBE_LLM_SAVE_RAW", False),
        SCRIBE_FACT_PIPELINE=_getenv_bool("SCRIBE_FACT_PIPELINE", True),
        SCRIBE_FACT_MODE=_getenv_str("SCRIBE_FACT_MODE", "chunk_json"),
        SCRIBE_EP_ENABLE_SLICING=_getenv_opt_bool("SCRIBE_EP_ENABLE_SLICING"),
        SCRIBE_EP_SLICING_PROTOCOL=_getenv_str("SCRIBE_EP_SLICING_PROTOCOL", "line"),
        SCRIBE_TEMPLATE_PATH=_getenv_str("SCRIBE_TEMPLATE_PATH", "assets/templates/soap_v1.json"),
        SCRIBE_EXTRACT_PROMPT_VERSION=_getenv_str(
            "SCRIBE_EXTRACT_PROMPT_VERSION", "extract_facts_v1_1"
        ),
        SCRIBE_RENDER_PROMPT_VERSION=_getenv_str(
            "SCRIBE_RENDER_PROMPT_VERSION", "render_template_v1_1"
        ),
        SCRIBE_CHUNK_SECONDS=_getenv_int("SCRIBE_CHUNK_SECONDS", 8),
        SCRIBE_CHUNK_OVERLAP_SECONDS=_getenv_float("SCRIBE_CHUNK_OVERLAP_SECONDS", 1.0),
        SCRIBE_MAX_AUDIO_SECONDS=_getenv_int("SCRIBE_MAX_AUDIO_SECONDS", 1800),
        SCRIBE_LOG_LEVEL=_getenv_str("SCRIBE_LOG_LEVEL", "INFO"),
        ASR_PRESET=asr_preset,
        ASR_CHUNK_SEC=asr_chunk_sec,
        ASR_OVERLAP_SEC=asr_overlap_sec,
        ASR_MIN_TEXT_CHARS=_getenv_int_preset(
            "ASR_MIN_TEXT_CHARS", 2, preset.get("ASR_MIN_TEXT_CHARS")
        ),
        ASR_SILENCE_RMS=_getenv_float_preset(
            "ASR_SILENCE_RMS", 0.008, preset.get("ASR_SILENCE_RMS")
        ),
        ASR_TARGET_MIN_RMS=_getenv_float_preset(
            "ASR_TARGET_MIN_RMS", 0.02, preset.get("ASR_TARGET_MIN_RMS")
        ),
        MEDASR_USE_LM=_getenv_bool_preset(
            "MEDASR_USE_LM", True, preset.get("MEDASR_USE_LM")
        ),
        MEDASR_LM_PATH=_getenv_str("MEDASR_LM_PATH", default_medasr_lm),
        MEDASR_LM_ALPHA=_getenv_float_preset(
            "MEDASR_LM_ALPHA", 0.5, preset.get("MEDASR_LM_ALPHA")
        ),
        MEDASR_LM_BETA=_getenv_float_preset(
            "MEDASR_LM_BETA", 1.0, preset.get("MEDASR_LM_BETA")
        ),
        MEDASR_LM_BEAM_WIDTH=_getenv_int("MEDASR_LM_BEAM_WIDTH", 50),
        ASR_PRIMARY_PROVIDER=asr_primary,
        ASR_FALLBACK_PROVIDER=asr_fallback,
        SCRIBE_EVIDENCE_REQUIRED=_getenv_bool("SCRIBE_EVIDENCE_REQUIRED", True),
        SCRIBE_EVIDENCE_WEAK_THRESHOLD=_getenv_float("SCRIBE_EVIDENCE_WEAK_THRESHOLD", 0.25),
        SCRIBE_TEST_INJECT_ASR_FAIL_AT_CHUNK=_getenv_opt_int(
            "SCRIBE_TEST_INJECT_ASR_FAIL_AT_CHUNK"
        ),
        SCRIBE_TEST_INJECT_LLM_FAIL=_getenv_bool("SCRIBE_TEST_INJECT_LLM_FAIL", False),
    )
