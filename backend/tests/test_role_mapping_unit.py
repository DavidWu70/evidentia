from backend.asr.models import ASRSegment, SpeakerTurn
from backend.asr.role_mapping import apply_stable_role_mapping


def test_apply_stable_role_mapping_assigns_roles_from_text_cues() -> None:
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker="other", speaker_id="spk_01", speaker_role="other"),
        SpeakerTurn(start=1.0, end=2.0, speaker="other", speaker_id="spk_02", speaker_role="other"),
    ]
    segments = [
        ASRSegment(start=0.0, end=1.0, text="I feel very down and cannot sleep well."),
        ASRSegment(start=1.0, end=2.0, text="How long has this been happening?"),
    ]

    mapped, state, debug = apply_stable_role_mapping(turns, segments, state={})

    assert mapped[0].speaker_role == "patient"
    assert mapped[1].speaker_role == "clinician"
    assert mapped[0].role_confidence is not None
    assert mapped[1].role_confidence is not None
    assert "spk_01" in state and "spk_02" in state
    assert debug["assigned_turns"] == 2


def test_apply_stable_role_mapping_keeps_prior_role_without_new_cues() -> None:
    turns = [SpeakerTurn(start=0.0, end=1.0, speaker="other", speaker_id="spk_01", speaker_role="other")]
    first_segments = [ASRSegment(start=0.0, end=1.0, text="I feel exhausted and anxious.")]
    mapped1, state1, _ = apply_stable_role_mapping(turns, first_segments, state={})

    neutral_segments = [ASRSegment(start=1.0, end=2.0, text="uh-huh")]
    mapped2, state2, _ = apply_stable_role_mapping(turns, neutral_segments, state=state1)

    assert mapped1[0].speaker_role == "patient"
    assert mapped2[0].speaker_role == "patient"
    assert state2["spk_01"]["role"] == "patient"
