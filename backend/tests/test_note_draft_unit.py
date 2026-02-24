import pytest

from backend.note.draft import (
    DraftEventEvidence,
    DraftProblem,
    DraftRiskFlag,
    build_note_draft,
    build_note_drafts,
)


def test_build_note_draft_contains_sections_and_citations() -> None:
    draft = build_note_draft(
        note_type="psych_soap",
        problem_list=[DraftProblem(item="Depressed mood reported", evidence_refs=["seg_1"])],
        risk_flags=[
            DraftRiskFlag(
                level="moderate",
                flag="passive_or_active_si_detected",
                why="Suicidal ideation language detected; consider structured safety assessment.",
                evidence_refs=["seg_1"],
            )
        ],
        open_questions=["Any active plan or intent since these thoughts started?"],
        event_evidence=[DraftEventEvidence(segment_id="seg_1", t0=0.0, t1=2.5)],
    )
    assert "Subjective:" in draft.note_text
    assert "Assessment:" in draft.note_text
    assert len(draft.citations) >= 1
    assert draft.citations[0].segment_id == "seg_1"


def test_build_note_draft_unsupported_type_raises() -> None:
    with pytest.raises(ValueError):
        build_note_draft(
            note_type="unknown_type",
            problem_list=[],
            risk_flags=[],
            open_questions=[],
            event_evidence=[],
        )


def test_build_note_drafts_supports_multiple_templates() -> None:
    batch = build_note_drafts(
        department="psych",
        template_ids=["psych_soap", "psych_follow_up"],
        problem_list=[DraftProblem(item="Depressed mood reported", evidence_refs=["seg_1"])],
        risk_flags=[],
        open_questions=["Any active plan or intent since these thoughts started?"],
        event_evidence=[DraftEventEvidence(segment_id="seg_1", t0=0.0, t1=2.5)],
    )
    assert batch.department == "psych"
    assert [item.template_id for item in batch.drafts] == ["psych_soap", "psych_follow_up"]
    assert "Subjective:" in batch.drafts[0].note_text
    assert "Interval Update:" in batch.drafts[1].note_text


def test_build_note_drafts_internal_med_template() -> None:
    batch = build_note_drafts(
        department="internal_med",
        template_ids=["internal_soap"],
        problem_list=[DraftProblem(item="Persistent fatigue reported", evidence_refs=["seg_1"])],
        risk_flags=[],
        open_questions=["Any fever, chest pain, or dyspnea since last visit?"],
        event_evidence=[DraftEventEvidence(segment_id="seg_1", t0=0.0, t1=2.5)],
    )
    assert batch.department == "internal_med"
    assert len(batch.drafts) == 1
    assert batch.drafts[0].template_id == "internal_soap"
    assert "HPI:" in batch.drafts[0].note_text


def test_build_note_drafts_unsupported_department_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported department"):
        build_note_drafts(
            department="unknown_dept",
            template_ids=["psych_soap"],
            problem_list=[],
            risk_flags=[],
            open_questions=[],
            event_evidence=[],
        )


def test_build_note_drafts_empty_template_ids_raises() -> None:
    with pytest.raises(ValueError, match="template_ids is required"):
        build_note_drafts(
            department="psych",
            template_ids=[],
            problem_list=[],
            risk_flags=[],
            open_questions=[],
            event_evidence=[],
        )


def test_build_note_drafts_includes_patient_identity_in_context_fallback(monkeypatch) -> None:
    monkeypatch.delenv("EVIDENTIA_NOTE_MODEL_PATH", raising=False)
    monkeypatch.delenv("EVIDENTIA_MEDGEMMA_GGUF", raising=False)
    monkeypatch.delenv("SCRIBE_LLAMA_CPP_MODEL", raising=False)

    batch = build_note_drafts(
        department="psych",
        template_ids=["psych_soap"],
        patient_identity="Alice Chen / MRN-12345",
        patient_basic_info="Age: 34\nAllergies: NKDA\nPMH: asthma",
        problem_list=[DraftProblem(item="Depressed mood reported", evidence_refs=["seg_1"])],
        risk_flags=[],
        open_questions=[],
        event_evidence=[DraftEventEvidence(segment_id="seg_1", t0=0.0, t1=2.5)],
    )
    assert len(batch.drafts) == 1
    assert "Patient:" in batch.drafts[0].note_text
    assert "Alice Chen / MRN-12345" in batch.drafts[0].note_text
    assert "Patient Basic Info:" in batch.drafts[0].note_text
    assert "Allergies: NKDA" in batch.drafts[0].note_text


def test_build_note_drafts_omits_patient_basic_info_when_empty(monkeypatch) -> None:
    monkeypatch.delenv("EVIDENTIA_NOTE_MODEL_PATH", raising=False)
    monkeypatch.delenv("EVIDENTIA_MEDGEMMA_GGUF", raising=False)
    monkeypatch.delenv("SCRIBE_LLAMA_CPP_MODEL", raising=False)

    batch = build_note_drafts(
        department="psych",
        template_ids=["psych_soap"],
        patient_identity="Alice Chen",
        patient_basic_info="   ",
        problem_list=[DraftProblem(item="Depressed mood reported", evidence_refs=["seg_1"])],
        risk_flags=[],
        open_questions=[],
        event_evidence=[DraftEventEvidence(segment_id="seg_1", t0=0.0, t1=2.5)],
    )
    assert len(batch.drafts) == 1
    assert "Patient Basic Info:" not in batch.drafts[0].note_text
