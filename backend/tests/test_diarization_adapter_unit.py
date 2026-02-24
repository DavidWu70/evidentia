from backend.asr.diarization_adapter import _assign_roles
from backend.asr.models import ASRSegment, SpeakerTurn


def test_assign_roles_single_speaker_rebuilds_turns_from_asr_text() -> None:
    turns = [
        SpeakerTurn(
            start=0.0,
            end=8.0,
            speaker="other",
            speaker_id="spk_01",
            speaker_role="other",
            confidence=0.91,
            diar_confidence=0.91,
        )
    ]
    segments = [
        ASRSegment(start=0.0, end=5.0, text="I feel very down and exhausted."),
        ASRSegment(start=5.0, end=8.0, text="Thank you for sharing this today. How are you sleeping now?"),
    ]

    out = _assign_roles(turns, asr_segments=segments)
    roles = [item.speaker_role for item in out]

    assert "patient" in roles
    assert "clinician" in roles
    assert all(item.speaker_id == "spk_01" for item in out)


def test_assign_roles_single_speaker_defaults_to_patient_when_unclear() -> None:
    turns = [
        SpeakerTurn(
            start=0.0,
            end=4.0,
            speaker="other",
            speaker_id="spk_01",
            speaker_role="other",
            confidence=0.8,
            diar_confidence=0.8,
        )
    ]
    segments = [ASRSegment(start=0.0, end=4.0, text="Audio transcription without clear role cues.")]

    out = _assign_roles(turns, asr_segments=segments)
    assert len(out) >= 1
    assert all(item.speaker_role in {"patient", "clinician"} for item in out)
