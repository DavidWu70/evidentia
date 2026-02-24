from __future__ import annotations

"""
Generate editable citation-backed draft notes from plain-text templates.

Design intent:
- Keep note templates as plain text files for clinician-friendly editing.
- Embed template text directly into LLM prompt during note generation.
- Preserve citation anchors for evidence traceability in MVP.
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from backend.utils.model_paths import resolve_medgemma_gguf_path


@dataclass(frozen=True)
class DraftProblem:
    item: str
    evidence_refs: list[str]


@dataclass(frozen=True)
class DraftRiskFlag:
    level: str
    flag: str
    why: str
    evidence_refs: list[str]


@dataclass(frozen=True)
class DraftEventEvidence:
    segment_id: str
    t0: float
    t1: float
    quote: str = ""


@dataclass(frozen=True)
class DraftCitation:
    anchor: str
    segment_id: str
    t0: float
    t1: float


@dataclass(frozen=True)
class DraftNote:
    note_type: str
    note_text: str
    citations: list[DraftCitation]


@dataclass(frozen=True)
class DraftTemplateNote:
    template_id: str
    template_name: str
    note_text: str
    citations: list[DraftCitation]


@dataclass(frozen=True)
class DraftBatch:
    department: str
    drafts: list[DraftTemplateNote]
    requested_template_ids: list[str]
    missing_template_ids: list[str]


@dataclass(frozen=True)
class DraftTemplateSpec:
    template_id: str
    template_name: str
    template_text: str
    path: Path


class DraftNoteGenerationError(RuntimeError):
    """Raised when note generation fails in strict LLM mode."""


def build_note_draft(
    *,
    note_type: str,
    patient_identity: str = "",
    patient_basic_info: str = "",
    problem_list: Sequence[DraftProblem],
    risk_flags: Sequence[DraftRiskFlag],
    open_questions: Sequence[str],
    event_evidence: Sequence[DraftEventEvidence],
) -> DraftNote:
    if note_type != "psych_soap":
        raise ValueError(f"Unsupported note_type: {note_type}")
    batch = build_note_drafts(
        department="psych",
        template_ids=[note_type],
        patient_identity=patient_identity,
        patient_basic_info=patient_basic_info,
        problem_list=problem_list,
        risk_flags=risk_flags,
        open_questions=open_questions,
        event_evidence=event_evidence,
    )
    if not batch.drafts:
        raise ValueError("No note draft generated.")
    primary = batch.drafts[0]
    return DraftNote(
        note_type=primary.template_id,
        note_text=primary.note_text,
        citations=primary.citations,
    )


def list_note_templates() -> dict[str, list[DraftTemplateSpec]]:
    registry = _load_template_registry()
    return {dept: list(specs.values()) for dept, specs in registry.items()}


def get_note_template_document(*, department: str, template_id: str) -> dict[str, object]:
    spec = _get_template_spec(department=department, template_id=template_id)
    return {
        "template_id": spec.template_id,
        "template_name": spec.template_name,
        "template_text": spec.template_text,
    }


def save_note_template_document(
    *,
    department: str,
    template_id: str,
    template_document: dict[str, object],
) -> dict[str, object]:
    spec = _get_template_spec(department=department, template_id=template_id)

    normalized_id = str(template_document.get("template_id", "")).strip()
    template_name = str(template_document.get("template_name", "")).strip()
    template_text = str(template_document.get("template_text", "")).strip()
    if not normalized_id:
        raise ValueError("template_id is required.")
    if normalized_id != spec.template_id:
        raise ValueError(
            "template_id in payload must match path template_id. "
            f"path={spec.template_id!r}, payload={normalized_id!r}"
        )
    if not template_name:
        raise ValueError("template_name is required.")
    if not template_text:
        raise ValueError("template_text is required.")

    payload = _encode_template_file_content(template_name, template_text)
    spec.path.write_text(payload, encoding="utf-8")
    return {
        "template_id": normalized_id,
        "template_name": template_name,
        "template_text": template_text,
    }


def build_note_drafts(
    *,
    department: str,
    template_ids: Sequence[str],
    patient_identity: str = "",
    patient_basic_info: str = "",
    problem_list: Sequence[DraftProblem],
    risk_flags: Sequence[DraftRiskFlag],
    open_questions: Sequence[str],
    event_evidence: Sequence[DraftEventEvidence],
) -> DraftBatch:
    registry = _load_template_registry()
    resolved_department = _normalize_department(department)
    specs_by_id = registry.get(resolved_department)
    if not specs_by_id:
        raise ValueError(f"Unsupported department: {resolved_department}")

    requested = _dedupe_in_order([str(item).strip() for item in template_ids if str(item).strip()])
    if not requested:
        raise ValueError("template_ids is required and cannot be empty.")

    selected_ids = [item for item in requested if item in specs_by_id]
    missing_ids = [item for item in requested if item not in specs_by_id]
    if not selected_ids:
        raise ValueError(
            f"No valid templates for department='{resolved_department}'. requested={requested}"
        )

    evidence_by_segment = {item.segment_id: item for item in event_evidence}
    context = _build_note_context(
        patient_identity=patient_identity,
        patient_basic_info=patient_basic_info,
        problem_list=problem_list,
        risk_flags=risk_flags,
        open_questions=open_questions,
        event_evidence=event_evidence,
    )
    citations = _collect_citations(
        problem_list=problem_list,
        risk_flags=risk_flags,
        evidence_by_segment=evidence_by_segment,
    )

    drafts: list[DraftTemplateNote] = []
    for template_id in selected_ids:
        spec = specs_by_id[template_id]
        note_text = _generate_note_text(
            department=resolved_department,
            template_spec=spec,
            context=context,
        )
        drafts.append(
            DraftTemplateNote(
                template_id=spec.template_id,
                template_name=spec.template_name,
                note_text=note_text,
                citations=list(citations),
            )
        )

    return DraftBatch(
        department=resolved_department,
        drafts=drafts,
        requested_template_ids=requested,
        missing_template_ids=missing_ids,
    )


def _generate_note_text(
    *,
    department: str,
    template_spec: DraftTemplateSpec,
    context: str,
) -> str:
    prompt = _build_note_prompt(
        department=department,
        template_name=template_spec.template_name,
        template_text=template_spec.template_text,
        context=context,
    )
    generated = _generate_note_via_llm(prompt)
    if generated:
        return generated.strip()
    return _fallback_template_fill(template_spec.template_text, context)


def _build_note_prompt(
    *,
    department: str,
    template_name: str,
    template_text: str,
    context: str,
) -> str:
    return (
        "Task: Draft one clinical note in plain text.\n"
        "Output requirements:\n"
        "- Follow TEMPLATE style/section ordering.\n"
        "- Use only CONTEXT facts.\n"
        "- If information is missing, write 'Data pending'.\n"
        "- Do not output JSON.\n"
        "- Do not output code fences.\n\n"
        f"Department: {department}\n"
        f"Template Name: {template_name}\n\n"
        "TEMPLATE:\n"
        "<note>\n"
        f"{template_text}\n"
        "</note>\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "Output:\n"
        "<note>\n"
    )


def _generate_note_via_llm(prompt: str) -> str:
    debug_log_path = _resolve_note_debug_log_path()
    strict_llm = _flag_enabled("EVIDENTIA_NOTE_STRICT_LLM", default=False)
    started_at = datetime.now(timezone.utc)
    _append_note_debug_log(
        debug_log_path,
        stage="note_generation_start",
        raw=prompt,
        metadata={"started_at": started_at.isoformat(), "strict_llm": strict_llm},
    )

    def finish(stage: str, raw: str, metadata: dict[str, Any] | None = None) -> None:
        ended_at = datetime.now(timezone.utc)
        elapsed_ms = round((ended_at - started_at).total_seconds() * 1000.0, 2)
        _append_note_debug_log(
            debug_log_path,
            stage=stage,
            raw=raw,
            metadata={
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "elapsed_ms": elapsed_ms,
                **(metadata or {}),
            },
        )

    model_path = resolve_medgemma_gguf_path(os.getenv("EVIDENTIA_NOTE_MODEL_PATH", "").strip())
    if not model_path:
        finish(
            "note_generation_end",
            raw="",
            metadata={"status": "model_path_missing", "fallback_to_template": not strict_llm},
        )
        if strict_llm:
            raise DraftNoteGenerationError(
                "Note model path is missing. Set EVIDENTIA_NOTE_MODEL_PATH (or EVIDENTIA_MEDGEMMA_GGUF), "
                "or place a MedGemma GGUF under local model defaults."
            )
        return ""
    if not os.path.exists(model_path):
        finish(
            "note_generation_end",
            raw="",
            metadata={
                "status": "model_path_not_found",
                "model_path": model_path,
                "fallback_to_template": not strict_llm,
            },
        )
        if strict_llm:
            raise DraftNoteGenerationError(f"Note model file not found: {model_path}")
        return ""

    chat_format = (
        os.getenv("EVIDENTIA_NOTE_CHAT_FORMAT", "").strip()
        or os.getenv("EVIDENTIA_MEDGEMMA_CHAT_FORMAT", "").strip()
        or os.getenv("SCRIBE_LLAMA_CPP_CHAT_FORMAT", "").strip()
        or "gemma"
    )
    max_tokens = _env_int("EVIDENTIA_NOTE_MAX_TOKENS", default=896, min_value=128, max_value=4096)
    temperature = _env_float("EVIDENTIA_NOTE_TEMPERATURE", default=0.1, min_value=0.0, max_value=1.0)
    n_ctx = _env_int("EVIDENTIA_NOTE_N_CTX", default=4096, min_value=512, max_value=16384)
    n_gpu_layers = _env_int("EVIDENTIA_NOTE_N_GPU_LAYERS", default=-1, min_value=-1, max_value=256)
    n_threads = _env_optional_int("EVIDENTIA_NOTE_N_THREADS", min_value=1, max_value=128)
    stop_sequences = ["<end_of_turn>", "</s>"]

    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as exc:
        finish(
            "note_generation_end",
            raw=str(exc),
            metadata={"status": "llama_cpp_import_failed", "fallback_to_template": not strict_llm},
        )
        if strict_llm:
            raise DraftNoteGenerationError(f"llama_cpp import failed: {exc}") from exc
        return ""

    _append_note_debug_log(
        debug_log_path,
        stage="note_prompt_input",
        raw="Prompt already logged at stage=note_generation_start.",
        metadata={
            "chat_format": chat_format,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model_path": model_path,
        },
    )

    chat_format_applied = True
    try:
        llm_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
            "verbose": False,
            "chat_format": chat_format,
        }
        if n_threads is not None:
            llm_kwargs["n_threads"] = n_threads
        try:
            llm = Llama(**llm_kwargs)
        except TypeError as exc:
            if "chat_format" not in str(exc):
                raise
            llm_kwargs.pop("chat_format", None)
            llm = Llama(**llm_kwargs)
            chat_format_applied = False

        started = datetime.now(timezone.utc)
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
            top_p=1.0,
            max_tokens=int(max_tokens),
            stop=stop_sequences,
        )
        ended = datetime.now(timezone.utc)
        raw = str(response["choices"][0]["message"]["content"] or "").strip()
        _append_note_debug_log(
            debug_log_path,
            stage="note_raw_output",
            raw=raw,
            metadata={
                "chat_format": chat_format,
                "chat_format_applied": chat_format_applied,
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
            },
        )
        cleaned = _strip_note_wrappers(raw)
        finish(
            "note_generation_end",
            raw="See stage=note_raw_output for generated note text.",
            metadata={
                "status": "ok",
                "model_path": model_path,
                "chat_format": chat_format,
                "chat_format_applied": chat_format_applied,
            },
        )
        return cleaned
    except Exception as exc:
        _append_note_debug_log(
            debug_log_path,
            stage="note_generation_error",
            raw=str(exc),
            metadata={"strict_llm": strict_llm},
        )
        finish(
            "note_generation_end",
            raw=str(exc),
            metadata={"status": "generation_error", "fallback_to_template": not strict_llm},
        )
        if strict_llm:
            raise DraftNoteGenerationError(f"Note generation failed: {exc}") from exc
        return ""


def _fallback_template_fill(template_text: str, context: str) -> str:
    # Keep fallback plain-text and template-first so UI still shows expected structure.
    rendered = template_text.strip()
    replaced = False
    for key in ("problem_list", "risk_flags", "open_questions", "evidence"):
        token = "{{" + key + "}}"
        if token in rendered:
            replaced = True
    if replaced:
        blocks = _extract_context_blocks(context)
        for key, value in blocks.items():
            rendered = rendered.replace("{{" + key + "}}", value)
        return rendered.strip()

    return f"{rendered}\n\nContext:\n{context}".strip()


def _strip_note_wrappers(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    wrapped = re.search(r"(?is)<note>\s*(.*?)\s*</note>", cleaned)
    if wrapped:
        return wrapped.group(1).strip()

    cleaned = re.sub(r"(?is)</?note>", "", cleaned)
    return cleaned.strip()


def _extract_context_blocks(context: str) -> dict[str, str]:
    blocks: dict[str, str] = {
        "problem_list": "",
        "risk_flags": "",
        "open_questions": "",
        "evidence": "",
    }
    active = ""
    lines = context.splitlines()
    for line in lines:
        if line.startswith("Problem List:"):
            active = "problem_list"
            continue
        if line.startswith("Risk Flags:"):
            active = "risk_flags"
            continue
        if line.startswith("Open Questions:"):
            active = "open_questions"
            continue
        if line.startswith("Evidence:"):
            active = "evidence"
            continue
        if active:
            blocks[active] += (line + "\n")
    return {key: value.strip() for key, value in blocks.items()}


def _build_note_context(
    *,
    patient_identity: str,
    patient_basic_info: str,
    problem_list: Sequence[DraftProblem],
    risk_flags: Sequence[DraftRiskFlag],
    open_questions: Sequence[str],
    event_evidence: Sequence[DraftEventEvidence],
) -> str:
    lines: list[str] = []
    lines.append("Patient:")
    normalized_identity = str(patient_identity or "").strip()
    lines.append(f"- {normalized_identity}" if normalized_identity else "- Data pending")
    lines.append("")

    normalized_basic_info = str(patient_basic_info or "").strip()
    if normalized_basic_info:
        lines.append("Patient Basic Info:")
        for row in normalized_basic_info.splitlines():
            row_text = str(row).strip()
            if row_text:
                lines.append(f"- {row_text}")
        lines.append("")

    lines.append("Problem List:")
    if problem_list:
        lines.extend([f"- {item.item}" for item in problem_list])
    else:
        lines.append("- Data pending")
    lines.append("")

    lines.append("Risk Flags:")
    if risk_flags:
        lines.extend([f"- ({item.level}) {item.flag}: {item.why}" for item in risk_flags])
    else:
        lines.append("- Data pending")
    lines.append("")

    lines.append("Evidence:")
    if event_evidence:
        for item in event_evidence[:24]:
            quote = str(item.quote or "").strip()
            if quote:
                lines.append(f"- \"{quote}\"")
            else:
                lines.append("- Data pending quote")
    else:
        lines.append("- Data pending")

    return "\n".join(lines).strip()


def _collect_citations(
    *,
    problem_list: Sequence[DraftProblem],
    risk_flags: Sequence[DraftRiskFlag],
    evidence_by_segment: dict[str, DraftEventEvidence],
) -> list[DraftCitation]:
    citations: list[DraftCitation] = []
    for idx, problem in enumerate(problem_list, start=1):
        _append_citation(
            citations=citations,
            anchor=f"P{idx}",
            refs=problem.evidence_refs,
            evidence_by_segment=evidence_by_segment,
        )
    for idx, flag in enumerate(risk_flags, start=1):
        _append_citation(
            citations=citations,
            anchor=f"R{idx}",
            refs=flag.evidence_refs,
            evidence_by_segment=evidence_by_segment,
        )
    return citations


def _template_root() -> Path:
    override = os.getenv("EVIDENTIA_NOTE_TEMPLATE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / "templates"


def _load_template_registry() -> dict[str, dict[str, DraftTemplateSpec]]:
    root = _template_root()
    if not root.exists():
        raise ValueError(f"Template directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Template path is not a directory: {root}")

    registry: dict[str, dict[str, DraftTemplateSpec]] = {}
    for department_dir in sorted(root.iterdir()):
        if not department_dir.is_dir():
            continue
        resolved_department = _normalize_department(department_dir.name)
        specs: dict[str, DraftTemplateSpec] = {}
        template_files = sorted(list(department_dir.glob("*.txt")) + list(department_dir.glob("*.md")))
        for template_file in template_files:
            spec = _load_template_spec_from_file(template_file)
            if spec.template_id in specs:
                raise ValueError(
                    f"Duplicate template_id '{spec.template_id}' in department '{resolved_department}'."
                )
            specs[spec.template_id] = spec
        if specs:
            registry[resolved_department] = specs
    return registry


def _load_template_spec_from_file(path: Path) -> DraftTemplateSpec:
    raw = path.read_text(encoding="utf-8")
    template_name, template_text = _decode_template_file_content(raw, default_name=path.stem)
    template_id = path.stem.strip()
    if not template_id:
        raise ValueError(f"Template filename must include template_id: {path}")
    if not template_name:
        raise ValueError(f"Template name is missing: {path}")
    if not template_text:
        raise ValueError(f"Template text is missing: {path}")
    return DraftTemplateSpec(
        template_id=template_id,
        template_name=template_name,
        template_text=template_text,
        path=path,
    )


def _decode_template_file_content(raw: str, *, default_name: str) -> tuple[str, str]:
    lines = raw.splitlines()
    template_name = ""
    body_start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("title:"):
            template_name = stripped.split(":", 1)[1].strip()
            body_start = idx + 1
        else:
            template_name = default_name.replace("_", " ").strip().title()
            body_start = idx
        break
    body = "\n".join(lines[body_start:]).strip()
    if not template_name:
        template_name = default_name.replace("_", " ").strip().title()
    return template_name, body


def _encode_template_file_content(template_name: str, template_text: str) -> str:
    return f"Title: {template_name.strip()}\n\n{template_text.strip()}\n"


def _get_template_spec(*, department: str, template_id: str) -> DraftTemplateSpec:
    registry = _load_template_registry()
    resolved_department = _normalize_department(department)
    specs_by_id = registry.get(resolved_department)
    if not specs_by_id:
        raise ValueError(f"Unsupported department: {resolved_department}")
    normalized_template_id = str(template_id or "").strip()
    spec = specs_by_id.get(normalized_template_id)
    if spec is None:
        raise ValueError(
            f"Unknown template_id '{normalized_template_id}' for department '{resolved_department}'."
        )
    return spec


def _normalize_department(raw: str) -> str:
    normalized = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or "psych"


def _dedupe_in_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _append_citation(
    *,
    citations: list[DraftCitation],
    anchor: str,
    refs: Sequence[str],
    evidence_by_segment: dict[str, DraftEventEvidence],
) -> None:
    for segment_id in refs:
        evidence = evidence_by_segment.get(segment_id)
        if evidence is None:
            continue
        citations.append(
            DraftCitation(
                anchor=anchor,
                segment_id=segment_id,
                t0=evidence.t0,
                t1=evidence.t1,
            )
        )
        return


def _resolve_note_debug_log_path() -> str | None:
    source_var = "EVIDENTIA_NOTE_DEBUG_LOG"
    raw = os.getenv(source_var, "").strip()
    if not raw:
        source_var = "EVIDENTIA_MEDGEMMA_DEBUG_LOG"
        raw = os.getenv(source_var, "").strip()
    if not raw:
        return None
    if raw.lower() in {"1", "true", "on", "yes"}:
        if source_var == "EVIDENTIA_MEDGEMMA_DEBUG_LOG":
            return "/tmp/evidentia_medgemma_raw.log"
        return "/tmp/evidentia_note_raw.log"
    return raw


def _append_note_debug_log(path: str | None, *, stage: str, raw: str, metadata: dict[str, Any] | None = None) -> None:
    if not path:
        return
    try:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        meta = json.dumps(metadata or {}, ensure_ascii=True)
        payload = (
            f"[{stamp}] stage={stage} meta={meta}\n"
            "-----BEGIN NOTE RAW-----\n"
            f"{raw}\n"
            "-----END NOTE RAW-----\n"
        )
        with target.open("a", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        return


def _flag_enabled(env_name: str, *, default: bool) -> bool:
    raw = os.getenv(env_name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "on", "yes"}


def _env_int(env_name: str, *, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _env_optional_int(env_name: str, *, min_value: int, max_value: int) -> int | None:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(min_value, min(max_value, value))


def _env_float(env_name: str, *, default: float, min_value: float, max_value: float) -> float:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))
