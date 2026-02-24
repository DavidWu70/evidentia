from backend.risk.snapshot import SnapshotEvent, build_state_snapshot


def test_state_snapshot_high_risk_generates_targeted_questions() -> None:
    snapshot = build_state_snapshot(
        [
            SnapshotEvent(
                type="risk_cue",
                label="suicidal_plan_or_intent",
                polarity="present",
                evidence_segment_id="seg_urgent",
            ),
            SnapshotEvent(
                type="symptom",
                label="depressed_mood",
                polarity="present",
                evidence_segment_id="seg_mood",
            ),
        ]
    )
    assert any(flag.flag == "urgent_suicide_risk" for flag in snapshot.risk_flags)
    assert any("immediate intent" in q.lower() for q in snapshot.open_questions)
    assert snapshot.mandatory_safety_questions
    assert snapshot.contextual_followups == []
    assert snapshot.open_questions == snapshot.mandatory_safety_questions
    assert snapshot.ai_enhancement_enabled is True
    assert any(item.item == "Depressed mood reported" for item in snapshot.problem_list)


def test_state_snapshot_without_risk_uses_generic_open_questions() -> None:
    snapshot = build_state_snapshot(
        [
            SnapshotEvent(
                type="symptom",
                label="fatigue_low_energy",
                polarity="present",
                evidence_segment_id="seg_energy",
            )
        ]
    )
    assert snapshot.risk_flags == []
    assert len(snapshot.open_questions) >= 1
    assert snapshot.open_questions == snapshot.mandatory_safety_questions
    assert snapshot.contextual_followups == []
    assert any("safety concern" in q.lower() for q in snapshot.open_questions)
