import sys
from types import SimpleNamespace

from backend.events.extractor import EventUtterance
from backend.events.medgemma_adapter import MedGemmaAdapterError, extract_events_with_medgemma


def test_medgemma_adapter_uses_chat_format_when_supported(monkeypatch, tmp_path) -> None:
    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            assert kwargs["chat_format"] == "gemma"
            self.calls = 0

        def create_chat_completion(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert "response_format" not in kwargs
                return {"choices": [{"message": {"content": "Yes"}}]}
            assert kwargs["response_format"] == {"type": "json_object"}
            return {"choices": [{"message": {"content": '{"events": []}'}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    model_path = tmp_path / "mock.gguf"
    model_path.write_text("x", encoding="utf-8")

    result = extract_events_with_medgemma(
        [
            EventUtterance(
                segment_id="seg_001",
                t0=0.0,
                t1=1.0,
                speaker="patient",
                text="I feel down.",
            )
        ],
        model_path=str(model_path),
        chat_format="gemma",
    )

    assert result.debug["status"] == "ok"
    assert result.debug["chat_format"] == "gemma"
    assert result.debug["chat_format_applied"] is True
    assert result.debug["chat_format_compat_mode"] == "constructor_arg"
    assert result.debug["chat_format_supported"] is True
    assert result.debug["response_format_applied"] is True
    assert result.debug["response_format_compat_mode"] == "explicit_arg"
    assert result.debug["response_format_supported"] is True


def test_medgemma_adapter_falls_back_when_constructor_chat_format_is_unsupported(monkeypatch, tmp_path) -> None:
    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            if "chat_format" in kwargs:
                raise TypeError("Llama.__init__() got an unexpected keyword argument 'chat_format'")
            self.calls = 0

        def create_chat_completion(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"choices": [{"message": {"content": "Yes"}}]}
            assert kwargs["response_format"] == {"type": "json_object"}
            return {"choices": [{"message": {"content": '{"events": []}'}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    model_path = tmp_path / "mock.gguf"
    model_path.write_text("x", encoding="utf-8")

    result = extract_events_with_medgemma(
        [
            EventUtterance(
                segment_id="seg_001",
                t0=0.0,
                t1=1.0,
                speaker="patient",
                text="I feel down.",
            )
        ],
        model_path=str(model_path),
        chat_format="gemma",
    )

    assert result.debug["status"] == "ok"
    assert result.debug["chat_format"] == "gemma"
    assert result.debug["chat_format_applied"] is False
    assert result.debug["chat_format_compat_mode"] == "constructor_omitted_unsupported"
    assert result.debug["chat_format_supported"] is False
    assert result.debug["response_format_applied"] is True


def test_medgemma_adapter_debug_log_writes_raw_output_on_invalid_json(monkeypatch, tmp_path) -> None:
    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def create_chat_completion(self, **kwargs):
            _ = kwargs
            return {"choices": [{"message": {"content": "not-json-output"}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    model_path = tmp_path / "mock.gguf"
    model_path.write_text("x", encoding="utf-8")
    log_path = tmp_path / "medgemma_debug.log"
    monkeypatch.setenv("EVIDENTIA_MEDGEMMA_DEBUG_LOG", str(log_path))

    try:
        extract_events_with_medgemma(
            [
                EventUtterance(
                    segment_id="seg_001",
                    t0=0.0,
                    t1=1.0,
                    speaker="patient",
                    text="I feel down.",
                )
            ],
            model_path=str(model_path),
            chat_format="gemma",
        )
    except MedGemmaAdapterError as exc:
        assert str(exc) == "MedGemma output is not valid JSON."
    else:
        raise AssertionError("Expected MedGemmaAdapterError for invalid JSON output.")

    content = log_path.read_text(encoding="utf-8")
    assert "stage=filter_prompt_input" in content
    assert "stage=filter_inference_start" in content
    assert "stage=filter_inference_end" in content
    assert "stage=filter_raw_output" in content
    assert "stage=prompt_input" in content
    assert "stage=extract_inference_start" in content
    assert "stage=extract_inference_end" in content
    assert "Task: Extract clinical events (mental health focused)" in content
    assert "stage=raw_output" in content
    assert "stage=parse_error_invalid_json" in content
    assert "not-json-output" in content


def test_medgemma_adapter_filter_gate_no_returns_empty(monkeypatch, tmp_path) -> None:
    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def create_chat_completion(self, **kwargs):
            _ = kwargs
            return {"choices": [{"message": {"content": "No"}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    model_path = tmp_path / "mock.gguf"
    model_path.write_text("x", encoding="utf-8")

    result = extract_events_with_medgemma(
        [
            EventUtterance(
                segment_id="seg_001",
                t0=0.0,
                t1=1.0,
                speaker="patient",
                text="Ortho follow-up only.",
            )
        ],
        model_path=str(model_path),
        chat_format="gemma",
    )

    assert result.events == []
    assert result.debug["status"] == "filtered_out_no_signal"
    assert result.debug["filter_decision"] == "no"


def test_medgemma_adapter_per_utterance_prompt_is_compact_and_uid_optional(monkeypatch, tmp_path) -> None:
    captured_prompts: list[str] = []

    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            _ = kwargs
            self.calls = 0

        def create_chat_completion(self, **kwargs):
            self.calls += 1
            prompt = str(kwargs.get("messages", [{}])[0].get("content", ""))
            captured_prompts.append(prompt)
            if self.calls == 1:
                return {"choices": [{"message": {"content": "Yes"}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"events":[{"type":"risk_cue","label":"passive_suicidal_ideation","polarity":"present","confidence":0.92}]}'
                        }
                    }
                ]
            }

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    model_path = tmp_path / "mock.gguf"
    model_path.write_text("x", encoding="utf-8")

    result = extract_events_with_medgemma(
        [
            EventUtterance(
                segment_id="seg_700",
                t0=0.0,
                t1=1.5,
                speaker="patient",
                text="Sometimes I wish I would not wake up.",
            )
        ],
        model_path=str(model_path),
        chat_format="gemma",
    )

    assert result.debug["mode"] == "per_utterance"
    assert len(result.events) == 1
    assert result.events[0].segment_id == "seg_700"
    assert result.events[0].label == "passive_suicidal_ideation"
    assert len(captured_prompts) >= 2
    assert "[u0001]" not in captured_prompts[0]
    assert "speaker=" not in captured_prompts[0]
    assert "[u0001]" not in captured_prompts[1]
    assert "speaker=" not in captured_prompts[1]


def test_medgemma_adapter_drops_type_label_mismatch(monkeypatch, tmp_path) -> None:
    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            _ = kwargs
            self.calls = 0

        def create_chat_completion(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"choices": [{"message": {"content": "Yes"}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"events":['
                                '{"type":"symptom","label":"passive_suicidal_ideation","polarity":"present","confidence":0.9},'
                                '{"type":"risk_cue","label":"passive_suicidal_ideation","polarity":"present","confidence":0.95}'
                                "]}")
                        }
                    }
                ]
            }

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    model_path = tmp_path / "mock.gguf"
    model_path.write_text("x", encoding="utf-8")

    result = extract_events_with_medgemma(
        [
            EventUtterance(
                segment_id="seg_701",
                t0=0.0,
                t1=1.5,
                speaker="patient",
                text="Sometimes I wish I would not wake up.",
            )
        ],
        model_path=str(model_path),
        chat_format="gemma",
    )

    assert result.debug["mode"] == "per_utterance"
    assert result.debug["dropped_items"] == 1
    assert len(result.events) == 1
    assert result.events[0].type == "risk_cue"
    assert result.events[0].label == "passive_suicidal_ideation"
