from __future__ import annotations

"""
MedGemma event extraction adapter with strict output validation.

Design intent:
- Reuse the local MedGemma/llama-cpp execution pattern proven in `eval_MedGemma`.
- Keep model output constrained to MVP event schema and known label set.
- Fail closed on malformed model output so rule-based fallback can protect reliability.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from backend.events.extractor import EventUtterance, ExtractedEvent
from backend.utils.model_paths import resolve_medgemma_gguf_path


EventType = Literal["risk_cue", "symptom", "duration_onset"]
Polarity = Literal["present", "absent", "uncertain"]

ALLOWED_TYPES: set[str] = {"risk_cue", "symptom", "duration_onset"}
ALLOWED_POLARITY: set[str] = {"present", "absent", "uncertain"}
ALLOWED_LABELS: set[str] = {
    "passive_suicidal_ideation",
    "suicidal_ideation",
    "suicidal_plan_or_intent",
    "homicidal_ideation",
    "psychosis_cue",
    "depressed_mood",
    "anxiety",
    "sleep_disturbance",
    "appetite_change",
    "fatigue_low_energy",
    "duration_mentioned",
    "onset_mentioned",
}
ALLOWED_LABELS_BY_TYPE: dict[str, set[str]] = {
    "risk_cue": {
        "passive_suicidal_ideation",
        "suicidal_ideation",
        "suicidal_plan_or_intent",
        "homicidal_ideation",
        "psychosis_cue",
    },
    "symptom": {
        "depressed_mood",
        "anxiety",
        "sleep_disturbance",
        "appetite_change",
        "fatigue_low_energy",
    },
    "duration_onset": {
        "duration_mentioned",
        "onset_mentioned",
    },
}


class MedGemmaAdapterError(RuntimeError):
    """Raised when MedGemma extraction fails or returns invalid payload."""


@dataclass(frozen=True)
class MedGemmaAdapterResult:
    events: list[ExtractedEvent]
    debug: dict[str, Any]


def extract_events_with_medgemma(
    utterances: Sequence[EventUtterance],
    *,
    model_path: str | None = None,
    max_tokens: int = 512,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    n_threads: int | None = None,
    chat_format: str | None = None,
) -> MedGemmaAdapterResult:
    """
    Extract events via local GGUF model.

    Environment fallbacks:
    - model_path: `EVIDENTIA_MEDGEMMA_GGUF` -> `SCRIBE_LLAMA_CPP_MODEL`
    - chat_format: `EVIDENTIA_MEDGEMMA_CHAT_FORMAT` -> `SCRIBE_LLAMA_CPP_CHAT_FORMAT` -> `gemma`
    - debug log: `EVIDENTIA_MEDGEMMA_DEBUG_LOG`
      - `1/true/on/yes` => `/tmp/evidentia_medgemma_raw.log`
      - any other non-empty value => write to that path
    """

    if not utterances:
        return MedGemmaAdapterResult(events=[], debug={"status": "empty_input"})

    resolved_model_path = resolve_medgemma_gguf_path(model_path).strip()
    if not resolved_model_path:
        raise MedGemmaAdapterError(
            "MedGemma model path is missing. Set EVIDENTIA_MEDGEMMA_GGUF (or SCRIBE_LLAMA_CPP_MODEL), "
            "or place a MedGemma GGUF under local model defaults."
        )
    if not os.path.exists(resolved_model_path):
        raise MedGemmaAdapterError(f"MedGemma model file not found: {resolved_model_path}")

    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as exc:
        raise MedGemmaAdapterError(f"llama_cpp import failed: {exc}") from exc

    debug_log_path = _resolve_debug_log_path()
    chosen_chat_format = (
        chat_format
        or os.getenv("EVIDENTIA_MEDGEMMA_CHAT_FORMAT", "")
        or os.getenv("SCRIBE_LLAMA_CPP_CHAT_FORMAT", "")
        or "gemma"
    )
    stop_sequences = _resolve_stop_sequences()
    use_filter_gate = _resolve_filter_gate_enabled()
    force_json_prefix = _resolve_force_json_prefix_enabled()
    per_utterance_mode = _resolve_per_utterance_enabled()

    lines = []
    index: dict[str, EventUtterance] = {}
    for i, item in enumerate(utterances, start=1):
        uid = f"u{i:04d}"
        index[uid] = item
        lines.append(
            f"[{uid}] speaker={item.speaker} t={item.t0:.2f}-{item.t1:.2f} text={json.dumps(item.text, ensure_ascii=False)}"
        )

    chat_format_applied = False
    chat_format_compat_mode = "constructor_arg"
    chat_format_supported: bool | None = None
    response_format_applied = False
    response_format_compat_mode = "not_requested"
    response_format_supported: bool | None = None
    filter_inference_ms = 0.0
    extract_inference_ms = 0.0
    try:
        llm_kwargs: dict[str, Any] = {
            "model_path": resolved_model_path,
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
            "verbose": False,
            "chat_format": chosen_chat_format,
        }
        if n_threads is not None:
            llm_kwargs["n_threads"] = int(n_threads)
        try:
            llm = Llama(**llm_kwargs)
            chat_format_applied = True
            chat_format_supported = True
            chat_format_compat_mode = "constructor_arg"
        except TypeError as exc:
            if "chat_format" not in str(exc):
                raise
            llm_kwargs.pop("chat_format", None)
            llm = Llama(**llm_kwargs)
            chat_format_applied = False
            chat_format_supported = False
            chat_format_compat_mode = "constructor_omitted_unsupported"

        if per_utterance_mode:
            seen: set[tuple[str, str, str, str]] = set()
            events: list[ExtractedEvent] = []
            counter = 1
            dropped = 0
            filter_inference_total_ms = 0.0
            extract_inference_total_ms = 0.0
            extracted_utterances = 0
            filter_no_count = 0

            for i, utterance in enumerate(utterances, start=1):
                single_line = json.dumps(str(utterance.text or ""), ensure_ascii=False)

                if use_filter_gate:
                    filter_prompt = _build_filter_prompt([single_line], per_utterance=True)
                    _append_debug_log(
                        debug_log_path,
                        stage="filter_prompt_input",
                        raw=filter_prompt,
                        metadata={
                            "utterances": 1,
                            "utterance_index": i,
                            "segment_id": utterance.segment_id,
                            "mode": "per_utterance",
                            "chat_format": chosen_chat_format,
                            "stop_sequences": stop_sequences,
                        },
                    )
                    filter_started = time.perf_counter()
                    _append_debug_log(
                        debug_log_path,
                        stage="filter_inference_start",
                        raw="Invoking llama_cpp.create_chat_completion for filter gate.",
                        metadata={
                            "max_tokens": 6,
                            "utterance_index": i,
                            "mode": "per_utterance",
                            "chat_format": chosen_chat_format,
                            "chat_format_applied": chat_format_applied,
                            "chat_format_compat_mode": chat_format_compat_mode,
                        },
                    )
                    filter_raw = _run_chat_completion(
                        llm,
                        prompt=filter_prompt,
                        max_tokens=6,
                        stop_sequences=stop_sequences,
                    )
                    filter_elapsed = round((time.perf_counter() - filter_started) * 1000.0, 2)
                    filter_inference_total_ms += filter_elapsed
                    _append_debug_log(
                        debug_log_path,
                        stage="filter_inference_end",
                        raw=filter_raw["content"],
                        metadata={
                            "elapsed_ms": filter_elapsed,
                            "utterance_index": i,
                            "mode": "per_utterance",
                            "chat_format_applied": chat_format_applied,
                            "chat_format_compat_mode": chat_format_compat_mode,
                        },
                    )
                    filter_output = filter_raw["content"]
                    filter_decision = _parse_filter_decision(filter_output)
                    _append_debug_log(
                        debug_log_path,
                        stage="filter_raw_output",
                        raw=filter_output,
                        metadata={
                            "decision": filter_decision,
                            "utterance_index": i,
                            "segment_id": utterance.segment_id,
                            "mode": "per_utterance",
                            "chat_format": chosen_chat_format,
                            "chat_format_applied": chat_format_applied,
                            "chat_format_compat_mode": chat_format_compat_mode,
                        },
                    )
                    if filter_decision == "no":
                        filter_no_count += 1
                        continue

                prompt = _build_prompt([single_line], per_utterance=True)
                _append_debug_log(
                    debug_log_path,
                    stage="prompt_input",
                    raw=prompt,
                    metadata={
                        "utterances": 1,
                        "utterance_index": i,
                        "segment_id": utterance.segment_id,
                        "mode": "per_utterance",
                        "chat_format": chosen_chat_format,
                        "stop_sequences": stop_sequences,
                        "force_json_prefix": force_json_prefix,
                    },
                )

                extract_started = time.perf_counter()
                _append_debug_log(
                    debug_log_path,
                    stage="extract_inference_start",
                    raw="Invoking llama_cpp.create_chat_completion for event extraction.",
                    metadata={
                        "max_tokens": int(max_tokens),
                        "utterance_index": i,
                        "mode": "per_utterance",
                        "chat_format": chosen_chat_format,
                        "chat_format_applied": chat_format_applied,
                        "chat_format_compat_mode": chat_format_compat_mode,
                        "response_format_requested": "json_object",
                    },
                )
                raw_result = _run_chat_completion(
                    llm,
                    prompt=prompt,
                    max_tokens=int(max_tokens),
                    stop_sequences=stop_sequences,
                    response_format_json=True,
                    response_format_supported=response_format_supported,
                )
                extract_elapsed = round((time.perf_counter() - extract_started) * 1000.0, 2)
                extract_inference_total_ms += extract_elapsed
                response_format_applied = raw_result["response_format_applied"]
                response_format_compat_mode = raw_result["response_format_compat_mode"]
                response_format_supported = raw_result["response_format_supported"]
                raw = raw_result["content"]
                _append_debug_log(
                    debug_log_path,
                    stage="extract_inference_end",
                    raw=raw,
                    metadata={
                        "elapsed_ms": extract_elapsed,
                        "utterance_index": i,
                        "mode": "per_utterance",
                        "chat_format_applied": chat_format_applied,
                        "chat_format_compat_mode": chat_format_compat_mode,
                        "response_format_applied": response_format_applied,
                        "response_format_compat_mode": response_format_compat_mode,
                    },
                )

                if force_json_prefix:
                    raw, prefixed = _coerce_json_prefix(raw)
                    if prefixed:
                        _append_debug_log(
                            debug_log_path,
                            stage="json_prefix_injected",
                            raw=raw,
                            metadata={"reason": "missing_leading_brace", "utterance_index": i, "mode": "per_utterance"},
                        )

                _append_debug_log(
                    debug_log_path,
                    stage="raw_output",
                    raw=raw,
                    metadata={
                        "utterance_index": i,
                        "segment_id": utterance.segment_id,
                        "mode": "per_utterance",
                        "chat_format": chosen_chat_format,
                        "chat_format_applied": chat_format_applied,
                        "chat_format_compat_mode": chat_format_compat_mode,
                        "chat_format_supported": chat_format_supported,
                        "response_format_applied": response_format_applied,
                        "response_format_compat_mode": response_format_compat_mode,
                        "response_format_supported": response_format_supported,
                        "stop_sequences": stop_sequences,
                        "filter_gate": use_filter_gate,
                        "extract_inference_ms": extract_elapsed,
                    },
                )

                payload = _parse_json_object(raw)
                if payload is None:
                    _append_debug_log(
                        debug_log_path,
                        stage="parse_error_invalid_json",
                        raw=raw,
                        metadata={"reason": "payload_none_after_parse", "utterance_index": i, "mode": "per_utterance"},
                    )
                    raise MedGemmaAdapterError("MedGemma output is not valid JSON.")
                items = payload.get("events")
                if not isinstance(items, list):
                    raise MedGemmaAdapterError("MedGemma JSON missing 'events' list.")

                extracted_utterances += 1
                for item in items:
                    if not isinstance(item, dict):
                        dropped += 1
                        continue
                    event_type = str(item.get("type", "")).strip()
                    label = str(item.get("label", "")).strip()
                    polarity = str(item.get("polarity", "")).strip()
                    confidence_raw = item.get("confidence", 0.7)
                    if event_type not in ALLOWED_TYPES:
                        dropped += 1
                        continue
                    if label not in ALLOWED_LABELS:
                        dropped += 1
                        continue
                    if not _is_label_allowed_for_type(event_type, label):
                        dropped += 1
                        continue
                    if polarity not in ALLOWED_POLARITY:
                        dropped += 1
                        continue
                    try:
                        confidence = float(confidence_raw)
                    except (TypeError, ValueError):
                        confidence = 0.7
                    confidence = min(1.0, max(0.0, confidence))
                    key = (utterance.segment_id, event_type, label, polarity)
                    if key in seen:
                        continue
                    seen.add(key)
                    events.append(
                        ExtractedEvent(
                            event_id=f"evt_mg_{counter:05d}",
                            type=event_type,  # type: ignore[arg-type]
                            label=label,
                            polarity=polarity,  # type: ignore[arg-type]
                            confidence=confidence,
                            speaker=utterance.speaker,
                            segment_id=utterance.segment_id,
                            t0=float(utterance.t0),
                            t1=float(utterance.t1),
                            quote=utterance.text,
                        )
                    )
                    counter += 1

            status = "ok"
            if use_filter_gate and filter_no_count >= len(utterances) and extracted_utterances == 0:
                status = "filtered_out_no_signal"
            filter_decision = "no" if status == "filtered_out_no_signal" else "mixed"
            return MedGemmaAdapterResult(
                events=events,
                debug={
                    "status": status,
                    "mode": "per_utterance",
                    "model_path": resolved_model_path,
                    "input_utterances": len(utterances),
                    "extracted_utterances": extracted_utterances,
                    "filter_no_count": filter_no_count,
                    "output_events": len(events),
                    "dropped_items": dropped,
                    "chat_format": chosen_chat_format,
                    "chat_format_applied": chat_format_applied,
                    "chat_format_compat_mode": chat_format_compat_mode,
                    "chat_format_supported": chat_format_supported,
                    "response_format_applied": response_format_applied,
                    "response_format_compat_mode": response_format_compat_mode,
                    "response_format_supported": response_format_supported,
                    "stop_sequences": stop_sequences,
                    "filter_gate": use_filter_gate,
                    "filter_decision": filter_decision,
                    "filter_inference_ms": round(filter_inference_total_ms, 2),
                    "extract_inference_ms": round(extract_inference_total_ms, 2),
                    "debug_log_path": debug_log_path or "",
                },
            )

        if use_filter_gate:
            filter_prompt = _build_filter_prompt(lines, per_utterance=False)
            _append_debug_log(
                debug_log_path,
                stage="filter_prompt_input",
                raw=filter_prompt,
                metadata={
                    "utterances": len(utterances),
                    "chat_format": chosen_chat_format,
                    "stop_sequences": stop_sequences,
                },
            )
            filter_started = time.perf_counter()
            _append_debug_log(
                debug_log_path,
                stage="filter_inference_start",
                raw="Invoking llama_cpp.create_chat_completion for filter gate.",
                metadata={
                    "max_tokens": 6,
                    "chat_format": chosen_chat_format,
                    "chat_format_applied": chat_format_applied,
                    "chat_format_compat_mode": chat_format_compat_mode,
                },
            )
            filter_raw = _run_chat_completion(
                llm,
                prompt=filter_prompt,
                max_tokens=6,
                stop_sequences=stop_sequences,
            )
            filter_inference_ms = round((time.perf_counter() - filter_started) * 1000.0, 2)
            _append_debug_log(
                debug_log_path,
                stage="filter_inference_end",
                raw=filter_raw["content"],
                metadata={
                    "elapsed_ms": filter_inference_ms,
                    "chat_format_applied": chat_format_applied,
                    "chat_format_compat_mode": chat_format_compat_mode,
                },
            )

            filter_output = filter_raw["content"]
            filter_decision = _parse_filter_decision(filter_output)
            _append_debug_log(
                debug_log_path,
                stage="filter_raw_output",
                raw=filter_output,
                metadata={
                    "decision": filter_decision,
                    "chat_format": chosen_chat_format,
                    "chat_format_applied": chat_format_applied,
                    "chat_format_compat_mode": chat_format_compat_mode,
                },
            )
            if filter_decision == "no":
                return MedGemmaAdapterResult(
                    events=[],
                    debug={
                        "status": "filtered_out_no_signal",
                        "model_path": resolved_model_path,
                        "input_utterances": len(utterances),
                        "output_events": 0,
                        "dropped_items": 0,
                        "chat_format": chosen_chat_format,
                        "chat_format_applied": chat_format_applied,
                        "chat_format_compat_mode": chat_format_compat_mode,
                        "chat_format_supported": chat_format_supported,
                        "response_format_applied": response_format_applied,
                        "response_format_compat_mode": response_format_compat_mode,
                        "response_format_supported": response_format_supported,
                        "stop_sequences": stop_sequences,
                        "filter_gate": True,
                        "filter_decision": "no",
                        "filter_inference_ms": filter_inference_ms,
                        "debug_log_path": debug_log_path or "",
                    },
                )

        prompt = _build_prompt(lines, per_utterance=False)
        _append_debug_log(
            debug_log_path,
            stage="prompt_input",
            raw=prompt,
            metadata={
                "utterances": len(utterances),
                "chat_format": chosen_chat_format,
                "stop_sequences": stop_sequences,
                "force_json_prefix": force_json_prefix,
            },
        )

        extract_started = time.perf_counter()
        _append_debug_log(
            debug_log_path,
            stage="extract_inference_start",
            raw="Invoking llama_cpp.create_chat_completion for event extraction.",
            metadata={
                "max_tokens": int(max_tokens),
                "chat_format": chosen_chat_format,
                "chat_format_applied": chat_format_applied,
                "chat_format_compat_mode": chat_format_compat_mode,
                "response_format_requested": "json_object",
            },
        )
        raw_result = _run_chat_completion(
            llm,
            prompt=prompt,
            max_tokens=int(max_tokens),
            stop_sequences=stop_sequences,
            response_format_json=True,
            response_format_supported=response_format_supported,
        )
        extract_inference_ms = round((time.perf_counter() - extract_started) * 1000.0, 2)
        response_format_applied = raw_result["response_format_applied"]
        response_format_compat_mode = raw_result["response_format_compat_mode"]
        response_format_supported = raw_result["response_format_supported"]
        raw = raw_result["content"]
        _append_debug_log(
            debug_log_path,
            stage="extract_inference_end",
            raw=raw,
            metadata={
                "elapsed_ms": extract_inference_ms,
                "chat_format_applied": chat_format_applied,
                "chat_format_compat_mode": chat_format_compat_mode,
                "response_format_applied": response_format_applied,
                "response_format_compat_mode": response_format_compat_mode,
            },
        )
    except MedGemmaAdapterError:
        raise
    except Exception as exc:
        raise MedGemmaAdapterError(f"MedGemma inference failed: {exc}") from exc

    if force_json_prefix:
        raw, prefixed = _coerce_json_prefix(raw)
        if prefixed:
            _append_debug_log(
                debug_log_path,
                stage="json_prefix_injected",
                raw=raw,
                metadata={"reason": "missing_leading_brace"},
            )

    _append_debug_log(
        debug_log_path,
        stage="raw_output",
        raw=raw,
        metadata={
            "chat_format": chosen_chat_format,
            "chat_format_applied": chat_format_applied,
            "chat_format_compat_mode": chat_format_compat_mode,
            "chat_format_supported": chat_format_supported,
            "response_format_applied": response_format_applied,
            "response_format_compat_mode": response_format_compat_mode,
            "response_format_supported": response_format_supported,
            "stop_sequences": stop_sequences,
            "filter_gate": use_filter_gate,
            "filter_inference_ms": filter_inference_ms,
            "extract_inference_ms": extract_inference_ms,
        },
    )

    payload = _parse_json_object(raw)
    if payload is None:
        _append_debug_log(
            debug_log_path,
            stage="parse_error_invalid_json",
            raw=raw,
            metadata={"reason": "payload_none_after_parse"},
        )
        raise MedGemmaAdapterError("MedGemma output is not valid JSON.")

    items = payload.get("events")
    if not isinstance(items, list):
        raise MedGemmaAdapterError("MedGemma JSON missing 'events' list.")

    events: list[ExtractedEvent] = []
    seen: set[tuple[str, str, str, str]] = set()
    counter = 1
    dropped = 0
    for item in items:
        if not isinstance(item, dict):
            dropped += 1
            continue
        uid = str(item.get("utterance_id", "")).strip()
        event_type = str(item.get("type", "")).strip()
        label = str(item.get("label", "")).strip()
        polarity = str(item.get("polarity", "")).strip()
        confidence_raw = item.get("confidence", 0.7)

        utterance = index.get(uid)
        if utterance is None:
            dropped += 1
            continue
        if event_type not in ALLOWED_TYPES:
            dropped += 1
            continue
        if label not in ALLOWED_LABELS:
            dropped += 1
            continue
        if not _is_label_allowed_for_type(event_type, label):
            dropped += 1
            continue
        if polarity not in ALLOWED_POLARITY:
            dropped += 1
            continue
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.7
        confidence = min(1.0, max(0.0, confidence))

        key = (utterance.segment_id, event_type, label, polarity)
        if key in seen:
            continue
        seen.add(key)

        events.append(
            ExtractedEvent(
                event_id=f"evt_mg_{counter:05d}",
                type=event_type,  # type: ignore[arg-type]
                label=label,
                polarity=polarity,  # type: ignore[arg-type]
                confidence=confidence,
                speaker=utterance.speaker,
                segment_id=utterance.segment_id,
                t0=float(utterance.t0),
                t1=float(utterance.t1),
                quote=utterance.text,
            )
        )
        counter += 1

    return MedGemmaAdapterResult(
        events=events,
        debug={
            "status": "ok",
            "model_path": resolved_model_path,
            "mode": "batch",
            "input_utterances": len(utterances),
            "output_events": len(events),
            "dropped_items": dropped,
            "chat_format": chosen_chat_format,
            "chat_format_applied": chat_format_applied,
            "chat_format_compat_mode": chat_format_compat_mode,
            "chat_format_supported": chat_format_supported,
            "response_format_applied": response_format_applied,
            "response_format_compat_mode": response_format_compat_mode,
            "response_format_supported": response_format_supported,
            "stop_sequences": stop_sequences,
            "filter_gate": use_filter_gate,
            "filter_inference_ms": filter_inference_ms,
            "extract_inference_ms": extract_inference_ms,
            "debug_log_path": debug_log_path or "",
        },
    )


def _build_prompt(lines: Sequence[str], *, per_utterance: bool = False) -> str:
    labels = ", ".join(sorted(ALLOWED_LABELS))
    if per_utterance:
        return (
            "Task: Extract clinical events (mental health focused) for one text into JSON.\n"
            f"Labels: [{labels}]\n"
            "Return strictly one JSON object with key \"events\".\n"
            "Each event must include: type, label, polarity, confidence.\n"
            "type in [risk_cue, symptom, duration_onset].\n"
            "polarity in [present, absent, uncertain].\n"
            "If no qualifying event exists, return {\"events\": []}.\n\n"
            "Example 1:\n"
            'In: "I feel very sad today."\n'
            'Out: {"events": [{"type": "symptom", "label": "depressed_mood", "polarity": "present", "confidence": 0.9}]}\n\n'
            "Example 2:\n"
            'In: "the ortho guy said that"\n'
            'Out: {"events": []}\n\n'
            "Target:\n"
            + "\n".join(lines)
            + "\nOut:\n"
        )
    return (
        "Task: Extract clinical events (mental health focused) into JSON.\n"
        f"Labels: [{labels}]\n"
        "Return strictly one JSON object with key \"events\".\n"
        "Each event must include: utterance_id, type, label, polarity, confidence.\n"
        "type in [risk_cue, symptom, duration_onset].\n"
        "polarity in [present, absent, uncertain].\n"
        "If no qualifying event exists, return {\"events\": []}.\n\n"
        "Example 1:\n"
        "In: [u0001] \"I feel very sad today.\"\n"
        "Out: {\"events\": [{\"utterance_id\": \"u0001\", \"type\": \"symptom\", \"label\": \"depressed_mood\", \"polarity\": \"present\", \"confidence\": 0.9}]}\n\n"
        "Example 2:\n"
        "In: [u0001] \"the ortho guy said that\"\n"
        "Out: {\"events\": []}\n\n"
        "Target:\n"
        + "\n".join(lines)
        + "\nOut:\n"
    )


def _is_label_allowed_for_type(event_type: str, label: str) -> bool:
    allowed = ALLOWED_LABELS_BY_TYPE.get(event_type)
    if not allowed:
        return False
    return label in allowed


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


def _build_filter_prompt(lines: Sequence[str], *, per_utterance: bool = False) -> str:
    if per_utterance:
        return (
            "Task: Does the target text contain any mental-health symptom or risk language?\n"
            "Answer with one word only: Yes or No.\n"
            "Mental-health signal includes depression, anxiety, psychosis, suicidal ideation, sleep disturbance, appetite change, fatigue.\n\n"
            "Target:\n"
            + "\n".join(lines)
            + "\nAnswer:"
        )
    return (
        "Task: Does the target text contain any mental-health symptom or risk language?\n"
        "Answer with one word only: Yes or No.\n"
        "Mental-health signal includes depression, anxiety, psychosis, suicidal ideation, sleep disturbance, appetite change, fatigue.\n\n"
        "Target:\n"
        + "\n".join(lines)
        + "\nAnswer:"
    )


def _parse_filter_decision(raw: str) -> str:
    text = raw.strip().lower()
    match = re.search(r"\b(yes|no)\b", text)
    if not match:
        return "unknown"
    return match.group(1)


def _resolve_stop_sequences() -> list[str]:
    raw = os.getenv("EVIDENTIA_MEDGEMMA_STOP_SEQUENCES", "").strip()
    if not raw:
        return ["\n\n", "json", "JSON", "```", "<end_of_turn>", "</s>"]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            values = [str(v) for v in parsed if str(v)]
            if values:
                return values
    except Exception:
        pass
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(item.encode("utf-8").decode("unicode_escape"))
    return values or ["\n\n", "json", "JSON", "```", "<end_of_turn>", "</s>"]


def _resolve_filter_gate_enabled() -> bool:
    raw = os.getenv("EVIDENTIA_MEDGEMMA_FILTER_GATE", "").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "on", "yes"}


def _resolve_per_utterance_enabled() -> bool:
    raw = os.getenv("EVIDENTIA_MEDGEMMA_PER_UTTERANCE", "").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "on", "yes"}


def _resolve_force_json_prefix_enabled() -> bool:
    raw = os.getenv("EVIDENTIA_MEDGEMMA_FORCE_JSON_PREFIX", "").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "on", "yes"}


def _run_chat_completion(
    llm: Any,
    *,
    prompt: str,
    max_tokens: int,
    stop_sequences: Sequence[str],
    response_format_json: bool = False,
    response_format_supported: bool | None = None,
) -> dict[str, Any]:
    completion_kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
        "stop": list(stop_sequences),
    }
    response_format_applied = False
    if not response_format_json:
        response_format_compat_mode = "not_requested"
    elif response_format_supported is False:
        response_format_compat_mode = "omitted_unsupported"
    else:
        response_format_compat_mode = "explicit_arg"

    if response_format_json and response_format_supported is not False:
        completion_kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = llm.create_chat_completion(**completion_kwargs)
        if "response_format" in completion_kwargs:
            response_format_applied = True
            response_format_supported = True
    except TypeError as exc:
        if "response_format" in str(exc) and "response_format" in completion_kwargs:
            completion_kwargs.pop("response_format", None)
            resp = llm.create_chat_completion(**completion_kwargs)
            response_format_applied = False
            response_format_supported = False
            response_format_compat_mode = "omitted_unsupported"
        else:
            raise

    raw = str(resp["choices"][0]["message"]["content"] or "").strip()
    return {
        "content": raw,
        "response_format_applied": response_format_applied,
        "response_format_compat_mode": response_format_compat_mode,
        "response_format_supported": response_format_supported,
    }


def _coerce_json_prefix(raw: str) -> tuple[str, bool]:
    text = raw.lstrip()
    if not text:
        return raw, False
    if text.startswith("{"):
        return text, False
    if text.startswith('"events"') or text.startswith("'events'") or text.startswith("events"):
        return "{" + text, True
    if text.startswith('"events'):
        return "{" + text, True
    return text, False


def _resolve_debug_log_path() -> str | None:
    raw = os.getenv("EVIDENTIA_MEDGEMMA_DEBUG_LOG", "").strip()
    if not raw:
        return None
    if raw.lower() in {"1", "true", "on", "yes"}:
        return "/tmp/evidentia_medgemma_raw.log"
    return raw


def _append_debug_log(path: str | None, *, stage: str, raw: str, metadata: dict[str, Any] | None = None) -> None:
    if not path:
        return
    try:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        meta = json.dumps(metadata or {}, ensure_ascii=True)
        payload = (
            f"[{stamp}] stage={stage} meta={meta}\n"
            "-----BEGIN MEDGEMMA RAW-----\n"
            f"{raw}\n"
            "-----END MEDGEMMA RAW-----\n"
        )
        with target.open("a", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        # Debug logging must never break the extraction pipeline.
        return
