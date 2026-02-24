from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    # backend/utils/model_paths.py -> backend -> evidentia
    return Path(__file__).resolve().parents[2]


def model_search_roots() -> list[Path]:
    roots: list[Path] = []
    model_root = os.getenv("EVIDENTIA_MODEL_ROOT", "").strip()
    if model_root:
        roots.append(Path(model_root).expanduser())
    base = project_root()
    roots.extend([base, base / "models", base.parent, base.parent / "models"])
    # Keep order and uniqueness.
    out: list[Path] = []
    seen: set[str] = set()
    for item in roots:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _first_existing(candidates: list[Path]) -> str:
    for path in candidates:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists():
            return str(resolved)
    return ""


def discover_medgemma_gguf() -> str:
    candidates: list[Path] = []
    for root in model_search_roots():
        candidates.extend(
            [
                root / "MedGemma" / "medgemma-1.5-4b-it-Q5_K_M.gguf",
                root / "models" / "medgemma-1.5-4b-it-Q5_K_M.gguf",
                root / "medgemma-1.5-4b-it-Q5_K_M.gguf",
            ]
        )
    return _first_existing(candidates)


def resolve_medgemma_gguf_path(explicit_path: str | None = None) -> str:
    """
    Resolve MedGemma GGUF path with precedence:
    1) explicit arg
    2) EVIDENTIA_MEDGEMMA_GGUF
    3) SCRIBE_LLAMA_CPP_MODEL
    4) auto-discovery in common local paths
    """

    explicit = str(explicit_path or "").strip()
    if explicit:
        return explicit

    env_path = os.getenv("EVIDENTIA_MEDGEMMA_GGUF", "").strip()
    if env_path:
        return env_path

    legacy = os.getenv("SCRIBE_LLAMA_CPP_MODEL", "").strip()
    if legacy:
        return legacy

    return discover_medgemma_gguf()

