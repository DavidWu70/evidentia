from __future__ import annotations

"""
Build a rolling clinical snapshot from evidence-backed events.

Design intent:
- Aggregate minimal event stream into clinician-facing triage context.
- Keep every problem and risk item linked to segment evidence references.
- Prefer deterministic, explainable rules for MVP stability.
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Sequence


RiskLevel = Literal["low", "moderate", "high"]


@dataclass(frozen=True)
class SnapshotEvent:
    type: str
    label: str
    polarity: str
    evidence_segment_id: str


@dataclass(frozen=True)
class SnapshotProblem:
    item: str
    evidence_refs: list[str]


@dataclass(frozen=True)
class SnapshotRiskFlag:
    level: RiskLevel
    flag: str
    why: str
    evidence_refs: list[str]


@dataclass(frozen=True)
class SnapshotResult:
    problem_list: list[SnapshotProblem]
    risk_flags: list[SnapshotRiskFlag]
    open_questions: list[str]
    mandatory_safety_questions: list[str]
    contextual_followups: list[str]
    rationale: str
    ai_enhancement_enabled: bool
    ai_enhancement_applied: bool
    ai_enhancement_error: str
    updated_at: str


_PROBLEM_LABELS = {
    "depressed_mood": "Depressed mood reported",
    "anxiety": "Anxiety symptoms reported",
    "sleep_disturbance": "Sleep disturbance reported",
    "appetite_change": "Appetite change reported",
    "fatigue_low_energy": "Fatigue/low energy reported",
    "duration_mentioned": "Symptom duration mentioned",
    "onset_mentioned": "Symptom onset context mentioned",
}


def build_state_snapshot(
    events: Sequence[SnapshotEvent],
    *,
    ai_enhancement_enabled: bool | None = None,
) -> SnapshotResult:
    problem_list = _build_problem_list(events)
    risk_flags = _build_risk_flags(events)
    mandatory_questions = _build_mandatory_open_questions(risk_flags)
    resolved_ai_enhancement_enabled = _resolve_ai_enhancement_enabled(ai_enhancement_enabled)
    contextual_followups: list[str] = []
    rationale = "Safety-critical questions are rule-driven."
    ai_enhancement_applied = False
    ai_enhancement_error = ""

    if resolved_ai_enhancement_enabled:
        enhancement = _generate_contextual_followup_with_llm(
            mandatory_questions=mandatory_questions,
            problem_list=problem_list,
            risk_flags=risk_flags,
        )
        contextual_followups = enhancement["contextual_followups"]
        rationale = enhancement["rationale"] or rationale
        ai_enhancement_applied = enhancement["applied"]
        ai_enhancement_error = enhancement["error"]

    open_questions = _merge_questions(mandatory_questions, contextual_followups)
    return SnapshotResult(
        problem_list=problem_list,
        risk_flags=risk_flags,
        open_questions=open_questions,
        mandatory_safety_questions=mandatory_questions,
        contextual_followups=contextual_followups,
        rationale=rationale,
        ai_enhancement_enabled=resolved_ai_enhancement_enabled,
        ai_enhancement_applied=ai_enhancement_applied,
        ai_enhancement_error=ai_enhancement_error,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def _build_problem_list(events: Sequence[SnapshotEvent]) -> list[SnapshotProblem]:
    grouped: dict[str, set[str]] = {}
    for item in events:
        if item.polarity != "present":
            continue
        if item.type not in {"symptom", "duration_onset"}:
            continue
        text = _PROBLEM_LABELS.get(item.label)
        if not text:
            continue
        grouped.setdefault(text, set()).add(item.evidence_segment_id)

    return [
        SnapshotProblem(item=problem, evidence_refs=sorted(refs))
        for problem, refs in sorted(grouped.items())
    ]


def _build_risk_flags(events: Sequence[SnapshotEvent]) -> list[SnapshotRiskFlag]:
    grouped: dict[tuple[str, str, str], set[str]] = {}

    for item in events:
        if item.type != "risk_cue":
            continue

        if item.label == "suicidal_plan_or_intent" and item.polarity == "present":
            key = (
                "high",
                "urgent_suicide_risk",
                "Plan/intent language detected; consider urgent safety escalation.",
            )
        elif item.label in {"suicidal_ideation", "passive_suicidal_ideation"} and item.polarity == "present":
            key = (
                "moderate",
                "passive_or_active_si_detected",
                "Suicidal ideation language detected; consider structured safety assessment.",
            )
        elif item.label == "suicidal_ideation" and item.polarity == "absent":
            key = (
                "low",
                "si_explicitly_denied",
                "Suicidal ideation explicitly denied in current transcript window.",
            )
        elif item.label == "homicidal_ideation" and item.polarity == "present":
            key = (
                "high",
                "possible_homicidal_risk",
                "Potential homicidal ideation language detected; consider urgent assessment.",
            )
        elif item.label == "psychosis_cue" and item.polarity == "present":
            key = (
                "high",
                "possible_psychosis_risk",
                "Psychosis-related cue detected; consider urgent clinical evaluation.",
            )
        else:
            continue

        grouped.setdefault(key, set()).add(item.evidence_segment_id)

    ordered = sorted(
        grouped.items(),
        key=lambda item: (_risk_rank(item[0][0]), item[0][1]),
    )
    return [
        SnapshotRiskFlag(
            level=key[0],  # type: ignore[index]
            flag=key[1],  # type: ignore[index]
            why=key[2],  # type: ignore[index]
            evidence_refs=sorted(refs),
        )
        for key, refs in ordered
    ]


def _build_mandatory_open_questions(risk_flags: Sequence[SnapshotRiskFlag]) -> list[str]:
    questions: list[str] = []
    seen: set[str] = set()

    for flag in risk_flags:
        if flag.flag == "urgent_suicide_risk":
            _push(questions, seen, "Is there immediate intent or plan to self-harm right now?")
            _push(questions, seen, "What means are currently accessible to the patient?")
            _push(questions, seen, "Who can provide immediate supervision and safety support?")
        elif flag.flag == "passive_or_active_si_detected":
            _push(questions, seen, "Any active plan or intent since these thoughts started?")
            _push(questions, seen, "Any prior attempts or rehearsed behaviors?")
            _push(questions, seen, "What protective factors are present today?")
        elif flag.flag == "possible_homicidal_risk":
            _push(questions, seen, "Any specific target, plan, or recent escalation toward others?")
            _push(questions, seen, "Any access to weapons or other means to harm others?")
        elif flag.flag == "possible_psychosis_risk":
            _push(questions, seen, "Are hallucinations or delusions commanding unsafe actions?")
            _push(questions, seen, "Has reality testing changed compared with baseline?")

    if not questions:
        _push(questions, seen, "Any immediate safety concern that still needs clarification?")
        _push(questions, seen, "Any functional decline since the last stable baseline?")
    return questions


def _resolve_ai_enhancement_enabled(override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    raw = os.getenv("EVIDENTIA_OPEN_QUESTIONS_AI_ENHANCEMENT", "").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "on", "yes"}


def _merge_questions(mandatory_questions: Sequence[str], contextual_followups: Sequence[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for question in list(mandatory_questions) + list(contextual_followups):
        _push(merged, seen, str(question))
    return merged


def _generate_contextual_followup_with_llm(
    *,
    mandatory_questions: Sequence[str],
    problem_list: Sequence[SnapshotProblem],
    risk_flags: Sequence[SnapshotRiskFlag],
) -> dict[str, Any]:
    """
    Optional enhancement layer:
    - Keep mandatory safety questions rule-driven.
    - Let model optionally add up to one contextual follow-up and a short rationale.
    """

    model_path = (
        os.getenv("EVIDENTIA_OPEN_QUESTIONS_MODEL_PATH", "").strip()
        or os.getenv("EVIDENTIA_MEDGEMMA_GGUF", "").strip()
        or os.getenv("SCRIBE_LLAMA_CPP_MODEL", "").strip()
    )
    if not model_path:
        return {
            "contextual_followups": [],
            "rationale": "Safety-critical questions are rule-driven; AI enhancement unavailable.",
            "applied": False,
            "error": "model_path_missing",
        }

    if not os.path.exists(model_path):
        return {
            "contextual_followups": [],
            "rationale": "Safety-critical questions are rule-driven; AI enhancement unavailable.",
            "applied": False,
            "error": "model_path_not_found",
        }

    try:
        from llama_cpp import Llama  # type: ignore
    except Exception:
        return {
            "contextual_followups": [],
            "rationale": "Safety-critical questions are rule-driven; AI enhancement unavailable.",
            "applied": False,
            "error": "llama_cpp_import_failed",
        }

    chat_format = (
        os.getenv("EVIDENTIA_OPEN_QUESTIONS_CHAT_FORMAT", "").strip()
        or os.getenv("EVIDENTIA_MEDGEMMA_CHAT_FORMAT", "").strip()
        or os.getenv("SCRIBE_LLAMA_CPP_CHAT_FORMAT", "").strip()
        or "gemma"
    )
    max_tokens = _env_int("EVIDENTIA_OPEN_QUESTIONS_MAX_TOKENS", default=160, min_value=64, max_value=512)
    temperature = _env_float(
        "EVIDENTIA_OPEN_QUESTIONS_TEMPERATURE",
        default=0.2,
        min_value=0.0,
        max_value=1.0,
    )
    n_ctx = _env_int("EVIDENTIA_OPEN_QUESTIONS_N_CTX", default=1024, min_value=512, max_value=8192)
    n_gpu_layers = _env_int("EVIDENTIA_OPEN_QUESTIONS_N_GPU_LAYERS", default=-1, min_value=-1, max_value=128)
    n_threads = _env_optional_int("EVIDENTIA_OPEN_QUESTIONS_N_THREADS", min_value=1, max_value=64)

    llm_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "verbose": False,
        "chat_format": chat_format,
    }
    if n_threads is not None:
        llm_kwargs["n_threads"] = n_threads

    try:
        llm = Llama(**llm_kwargs)
    except TypeError as exc:
        if "chat_format" not in str(exc):
            return {
                "contextual_followups": [],
                "rationale": "Safety-critical questions are rule-driven; AI enhancement unavailable.",
                "applied": False,
                "error": "llama_init_failed",
            }
        llm_kwargs.pop("chat_format", None)
        try:
            llm = Llama(**llm_kwargs)
        except Exception:
            return {
                "contextual_followups": [],
                "rationale": "Safety-critical questions are rule-driven; AI enhancement unavailable.",
                "applied": False,
                "error": "llama_init_failed",
            }
    except Exception:
        return {
            "contextual_followups": [],
            "rationale": "Safety-critical questions are rule-driven; AI enhancement unavailable.",
            "applied": False,
            "error": "llama_init_failed",
        }

    prompt = _build_open_questions_enhancement_prompt(
        mandatory_questions=mandatory_questions,
        problem_list=problem_list,
        risk_flags=risk_flags,
    )

    try:
        completion_kwargs: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "top_p": 1.0,
            "max_tokens": int(max_tokens),
            "stop": ["```", "<end_of_turn>", "</s>"],
            "response_format": {"type": "json_object"},
        }
        try:
            resp = llm.create_chat_completion(**completion_kwargs)
        except TypeError:
            completion_kwargs.pop("response_format", None)
            resp = llm.create_chat_completion(**completion_kwargs)
        raw = str(resp["choices"][0]["message"]["content"] or "")
    except Exception:
        return {
            "contextual_followups": [],
            "rationale": "Safety-critical questions are rule-driven; AI enhancement unavailable.",
            "applied": False,
            "error": "llm_call_failed",
        }

    payload = _parse_json_object(raw)
    if payload is None:
        return {
            "contextual_followups": [],
            "rationale": "Safety-critical questions are rule-driven; AI enhancement unavailable.",
            "applied": False,
            "error": "invalid_json_output",
        }

    followups = _coerce_questions(payload.get("contextual_followups"), max_items=1)
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        rationale = "Safety-critical questions are rule-driven; contextual follow-up is AI-suggested."
    return {
        "contextual_followups": followups,
        "rationale": rationale,
        "applied": bool(followups),
        "error": "",
    }


def _build_open_questions_enhancement_prompt(
    *,
    mandatory_questions: Sequence[str],
    problem_list: Sequence[SnapshotProblem],
    risk_flags: Sequence[SnapshotRiskFlag],
) -> str:
    risk_lines = [
        f"- ({item.level}) {item.flag}: {item.why}"
        for item in risk_flags[:6]
    ] or ["- none"]
    problem_lines = [f"- {item.item}" for item in problem_list[:8]] or ["- none"]
    mandatory_lines = [f"- {item}" for item in mandatory_questions] or ["- none"]
    return (
        "Task: Provide optional contextual follow-up for clinician interview.\n"
        "Rules:\n"
        "- Keep mandatory safety questions unchanged.\n"
        "- Add at most ONE contextual follow-up question.\n"
        "- Follow-up must be neutral, safety-aware, and non-diagnostic.\n"
        "- Return JSON only.\n\n"
        "Return schema:\n"
        "{\n"
        '  "contextual_followups": ["..."],\n'
        '  "rationale": "..."\n'
        "}\n\n"
        "Mandatory safety questions:\n"
        + "\n".join(mandatory_lines)
        + "\n\nRisk flags:\n"
        + "\n".join(risk_lines)
        + "\n\nProblem list:\n"
        + "\n".join(problem_lines)
        + "\n\nJSON:"
    )


def _coerce_questions(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        if text in seen:
            continue
        seen.add(text)
        output.append(text)
        if len(output) >= max_items:
            break
    return output


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    extracted = _extract_first_json_object(raw)
    if not extracted:
        return None
    try:
        data = json.loads(extracted)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


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


def _risk_rank(level: str) -> int:
    if level == "high":
        return 0
    if level == "moderate":
        return 1
    return 2


def _push(target: list[str], seen: set[str], question: str) -> None:
    normalized = str(question or "").strip()
    if not normalized:
        return
    if normalized in seen:
        return
    seen.add(normalized)
    target.append(normalized)
