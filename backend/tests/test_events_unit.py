from backend.events.extractor import (
    ExtractedEvent,
    EventUtterance,
    apply_event_consistency_harmonization,
    apply_event_quality_guardrails,
    apply_rule_risk_backstop,
    extract_minimal_events,
)


def test_extract_minimal_events_always_contains_evidence_fields() -> None:
    events = extract_minimal_events(
        [
            EventUtterance(
                segment_id="seg_100",
                t0=0.0,
                t1=2.0,
                speaker="patient",
                text="I feel down and exhausted for 2 weeks. Sometimes I wish I would not wake up.",
            )
        ]
    )
    assert len(events) >= 3
    for item in events:
        assert item.segment_id == "seg_100"
        assert item.t0 >= 0.0
        assert item.t1 >= item.t0
        assert item.quote.strip() != ""


def test_extract_minimal_events_marks_suicidal_ideation_absent_when_denied() -> None:
    events = extract_minimal_events(
        [
            EventUtterance(
                segment_id="seg_200",
                t0=3.0,
                t1=5.0,
                speaker="patient",
                text="I am not suicidal and deny suicidal thoughts.",
            )
        ]
    )
    labels = {(item.type, item.label, item.polarity) for item in events}
    assert ("risk_cue", "suicidal_ideation", "absent") in labels


def test_extract_minimal_events_stitches_split_passive_si_phrase() -> None:
    events = extract_minimal_events(
        [
            EventUtterance(
                segment_id="seg_301",
                t0=0.0,
                t1=4.2,
                speaker="patient",
                text="I have felt very down for two weeks. Sometimes I wish I would not",
            ),
            EventUtterance(
                segment_id="seg_302",
                t0=4.2,
                t1=5.1,
                speaker="patient",
                text="wake up.",
            ),
        ]
    )
    labels = {(item.type, item.label, item.polarity) for item in events}
    assert ("risk_cue", "passive_suicidal_ideation", "present") in labels
    passive = next(item for item in events if item.label == "passive_suicidal_ideation")
    assert "wish i would not wake up" in passive.quote.lower()
    assert passive.segment_id == "seg_301"


def test_event_guardrail_drops_clinician_risk_without_first_person_signal() -> None:
    events = [
        ExtractedEvent(
            event_id="evt_1",
            type="risk_cue",
            label="passive_suicidal_ideation",
            polarity="present",
            confidence=0.9,
            speaker="clinician",
            segment_id="seg_999",
            t0=1.0,
            t1=2.0,
            quote="Thank you for sharing this today.",
        ),
        ExtractedEvent(
            event_id="evt_2",
            type="symptom",
            label="depressed_mood",
            polarity="present",
            confidence=0.8,
            speaker="patient",
            segment_id="seg_1000",
            t0=2.0,
            t1=3.0,
            quote="I have felt down.",
        ),
    ]

    kept, debug = apply_event_quality_guardrails(events)
    assert len(kept) == 1
    assert kept[0].type == "symptom"
    assert debug["dropped"] == 1


def test_event_guardrail_keeps_clinician_risk_with_strong_first_person_signal() -> None:
    events = [
        ExtractedEvent(
            event_id="evt_3",
            type="risk_cue",
            label="passive_suicidal_ideation",
            polarity="present",
            confidence=0.9,
            speaker="clinician",
            segment_id="seg_1001",
            t0=1.0,
            t1=2.0,
            quote='I wish I would not wake up.',
        ),
    ]

    kept, debug = apply_event_quality_guardrails(events)
    assert len(kept) == 1
    assert debug["dropped"] == 0


def test_rule_risk_backstop_adds_missing_patient_risk() -> None:
    existing = [
        ExtractedEvent(
            event_id="evt_existing_1",
            type="symptom",
            label="sleep_disturbance",
            polarity="present",
            confidence=0.9,
            speaker="patient",
            segment_id="seg_401",
            t0=4.0,
            t1=6.0,
            quote="Sometimes I wish I would not wake up.",
        )
    ]
    utterances = [
        EventUtterance(
            segment_id="seg_401",
            t0=4.0,
            t1=6.0,
            speaker="patient",
            text="Sometimes I wish I would not wake up.",
        ),
    ]
    merged, debug = apply_rule_risk_backstop(existing, utterances)
    labels = {(item.type, item.label, item.polarity) for item in merged}
    assert ("risk_cue", "passive_suicidal_ideation", "present") in labels
    assert debug["added"] >= 1


def test_event_harmonization_keeps_sleep_disturbance_with_passive_si() -> None:
    events = [
        ExtractedEvent(
            event_id="evt_501",
            type="risk_cue",
            label="passive_suicidal_ideation",
            polarity="present",
            confidence=0.9,
            speaker="patient",
            segment_id="seg_501",
            t0=4.0,
            t1=6.0,
            quote="Sometimes I wish I would not wake up.",
        ),
        ExtractedEvent(
            event_id="evt_502",
            type="symptom",
            label="sleep_disturbance",
            polarity="present",
            confidence=0.9,
            speaker="patient",
            segment_id="seg_501",
            t0=4.0,
            t1=6.0,
            quote="Sometimes I wish I would not wake up.",
        ),
    ]
    merged, debug = apply_event_consistency_harmonization(events)
    labels = {(item.type, item.label, item.polarity) for item in merged}
    assert ("risk_cue", "passive_suicidal_ideation", "present") in labels
    assert ("symptom", "sleep_disturbance", "present") in labels
    assert debug["dropped"] == 0
    assert debug["reason"] == "no_conflict_rules_active"
