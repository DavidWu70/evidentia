import base64
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.api.main import app
from backend.note.draft import DraftBatch, DraftCitation, DraftTemplateNote


def test_healthz() -> None:
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_live_audio_ws_receives_chunk_and_updates_status() -> None:
    client = TestClient(app)
    session_id = "live_audio_case"
    chunk = b"\x00\x11\x22\x33\x44\x55"
    encoded = base64.b64encode(chunk).decode("ascii")
    app.state.live_audio_default_asr_enabled = False

    try:
        with client.websocket_connect(f"/ws/audio/live?session_id={session_id}") as ws:
            ws.send_text(json.dumps({"type": "start", "mime_type": "audio/webm", "sample_rate_hz": 48000}))
            ack_start = ws.receive_json()
            assert ack_start["type"] == "ack_start"

            ws.send_text(
                json.dumps(
                    {
                        "type": "audio_chunk",
                        "seq": 0,
                        "mime_type": "audio/webm",
                        "sample_rate_hz": 48000,
                        "data_b64": encoded,
                    }
                )
            )
            ack_chunk = ws.receive_json()
            assert ack_chunk["type"] == "ack_chunk"
            assert ack_chunk["chunks_received"] >= 1
            assert ack_chunk["bytes_received"] >= len(chunk)
            assert ack_chunk["asr_mode"] == "not_configured"

            ws.send_text(json.dumps({"type": "stop"}))
            ack_stop = ws.receive_json()
            assert ack_stop["type"] == "ack_stop"
    finally:
        if hasattr(app.state, "live_audio_default_asr_enabled"):
            delattr(app.state, "live_audio_default_asr_enabled")

    status_resp = client.get(f"/audio/live/status/{session_id}")
    assert status_resp.status_code == 200
    payload = status_resp.json()
    assert payload["session_id"] == session_id
    assert payload["chunks_received"] >= 1
    assert payload["bytes_received"] >= len(chunk)
    assert payload["sample_rate_hz"] == 48000


def test_live_audio_ws_archives_raw_chunks_and_exposes_recording_path(tmp_path) -> None:
    client = TestClient(app)
    session_id = "live_audio_archive_case"
    raw_chunk = b"\x99\x88\x77\x66"
    encoded = base64.b64encode(raw_chunk).decode("ascii")
    app.state.live_audio_default_asr_enabled = False
    app.state.uploaded_audio_dir = str(tmp_path)

    try:
        with client.websocket_connect(f"/ws/audio/live?session_id={session_id}") as ws:
            ws.send_text(json.dumps({"type": "start", "mime_type": "audio/webm", "sample_rate_hz": 48000}))
            ack_start = ws.receive_json()
            assert ack_start["type"] == "ack_start"

            ws.send_text(
                json.dumps(
                    {
                        "type": "audio_chunk_archive",
                        "seq": 0,
                        "mime_type": "audio/webm",
                        "data_b64": encoded,
                    }
                )
            )
            ack_archive = ws.receive_json()
            assert ack_archive["type"] == "ack_archive_chunk"
            assert ack_archive["duplicate"] is False
            archived_path = Path(str(ack_archive["recording_audio_path"]))
            assert archived_path.exists()
            assert archived_path.read_bytes() == raw_chunk

            ws.send_text(
                json.dumps(
                    {
                        "type": "audio_chunk_archive",
                        "seq": 0,
                        "mime_type": "audio/webm",
                        "data_b64": encoded,
                    }
                )
            )
            dup_ack = ws.receive_json()
            assert dup_ack["type"] == "ack_archive_chunk"
            assert dup_ack["duplicate"] is True
    finally:
        if hasattr(app.state, "live_audio_default_asr_enabled"):
            delattr(app.state, "live_audio_default_asr_enabled")
        if hasattr(app.state, "uploaded_audio_dir"):
            delattr(app.state, "uploaded_audio_dir")

    status_resp = client.get(f"/audio/live/status/{session_id}")
    assert status_resp.status_code == 200
    status_payload = status_resp.json()
    status_recording_path = Path(str(status_payload["debug"]["recording_audio_path"]))
    assert status_recording_path.exists()
    assert status_payload["debug"]["recording_chunks_received"] >= 1
    assert status_payload["debug"]["recording_bytes_received"] >= len(raw_chunk)

    transcript_resp = client.get(f"/audio/live/transcript/{session_id}")
    assert transcript_resp.status_code == 200
    transcript_payload = transcript_resp.json()
    assert transcript_payload["debug"]["recording_audio_path"] == str(status_recording_path)


def test_live_audio_transcript_endpoint_with_injected_chunk_asr() -> None:
    client = TestClient(app)
    session_id = "live_audio_transcript_case"

    def fake_chunk_asr(
        chunk_bytes: bytes,
        *,
        session_id: str,
        seq: int | None,
        sample_rate_hz: int | None,
        mime_type: str | None,
    ):
        assert chunk_bytes
        assert session_id == "live_audio_transcript_case"
        return (
            [{"start": 0.0, "end": 0.8, "text": "patient feels anxious"}],
            {"chunk_duration_sec": 1.0},
        )

    app.state.live_audio_chunk_asr_callable = fake_chunk_asr
    chunk = base64.b64encode(b"\x01\x02\x03\x04").decode("ascii")
    try:
        with client.websocket_connect(f"/ws/audio/live?session_id={session_id}") as ws:
            ws.send_text(json.dumps({"type": "start", "mime_type": "audio/webm", "sample_rate_hz": 16000}))
            ack_start = ws.receive_json()
            assert ack_start["type"] == "ack_start"

            ws.send_text(json.dumps({"type": "audio_chunk", "seq": 0, "mime_type": "audio/webm", "data_b64": chunk}))
            ack_chunk = ws.receive_json()
            assert ack_chunk["type"] == "ack_chunk"
            assert ack_chunk["asr_status"] == "ok"
            assert ack_chunk["new_segments"] == 1
            assert ack_chunk["total_utterances"] >= 1
            assert isinstance(ack_chunk["new_utterances_payload"], list)
            assert len(ack_chunk["new_utterances_payload"]) >= 1
            assert isinstance(ack_chunk["utterances"], list)
    finally:
        delattr(app.state, "live_audio_chunk_asr_callable")

    transcript_resp = client.get(f"/audio/live/transcript/{session_id}")
    assert transcript_resp.status_code == 200
    transcript_payload = transcript_resp.json()
    assert transcript_payload["status"] in {"streaming", "disconnected", "stopped"}
    assert len(transcript_payload["utterances"]) >= 1
    assert transcript_payload["debug"]["asr_status"] == "ok"
    assert "anxious" in transcript_payload["transcript_text"].lower()


def test_live_audio_ws_applies_incremental_diarization_when_available() -> None:
    client = TestClient(app)
    session_id = "live_audio_diar_case"

    def fake_chunk_asr(
        chunk_bytes: bytes,
        *,
        session_id: str,
        seq: int | None,
        sample_rate_hz: int | None,
        mime_type: str | None,
    ):
        _ = (chunk_bytes, session_id, seq, sample_rate_hz, mime_type)
        return (
            [{"start": 0.0, "end": 0.9, "text": "i feel anxious"}],
            {"chunk_duration_sec": 1.0},
        )

    def fake_chunk_diarize(
        chunk_bytes: bytes,
        *,
        session_id: str,
        seq: int | None,
        sample_rate_hz: int | None,
        mime_type: str | None,
        asr_segments=None,
        window_duration_sec: float | None = None,
    ):
        _ = (chunk_bytes, session_id, seq, sample_rate_hz, mime_type, asr_segments, window_duration_sec)
        return (
            [
                {
                    "start": 0.0,
                    "end": 0.9,
                    "speaker": "patient",
                    "speaker_id": "spk_01",
                    "speaker_role": "patient",
                    "diar_confidence": 0.93,
                }
            ],
            {"status": "ok"},
        )

    app.state.live_audio_chunk_asr_callable = fake_chunk_asr
    app.state.live_audio_chunk_diarize_callable = fake_chunk_diarize
    chunk = base64.b64encode(b"\x01\x02\x03\x04").decode("ascii")
    try:
        with client.websocket_connect(f"/ws/audio/live?session_id={session_id}") as ws:
            ws.send_text(json.dumps({"type": "start", "mime_type": "audio/webm", "sample_rate_hz": 16000}))
            ws.receive_json()

            ws.send_text(json.dumps({"type": "audio_chunk", "seq": 0, "mime_type": "audio/webm", "data_b64": chunk}))
            ack_chunk = ws.receive_json()
            assert ack_chunk["type"] == "ack_chunk"
            assert ack_chunk["asr_status"] == "ok"
            assert ack_chunk["diarization_status"] == "ok"
            assert ack_chunk["diarization_mode"] == "injected_callable"
            assert len(ack_chunk["utterances"]) >= 1
            assert ack_chunk["utterances"][0]["speaker"] == "patient"
            assert ack_chunk["utterances"][0]["speaker_id"] == "spk_01"
    finally:
        delattr(app.state, "live_audio_chunk_asr_callable")
        delattr(app.state, "live_audio_chunk_diarize_callable")


def test_live_audio_ws_sentence_role_split_for_single_speaker_mixed_sentences() -> None:
    client = TestClient(app)
    session_id = "live_audio_sentence_split_case"

    def fake_chunk_asr(
        chunk_bytes: bytes,
        *,
        session_id: str,
        seq: int | None,
        sample_rate_hz: int | None,
        mime_type: str | None,
    ):
        _ = (chunk_bytes, session_id, seq, sample_rate_hz, mime_type)
        return (
            [{"start": 0.0, "end": 2.0, "text": "I feel exhausted and cannot sleep well. Thank you for sharing this today."}],
            {"chunk_duration_sec": 2.0},
        )

    def fake_chunk_diarize(
        chunk_bytes: bytes,
        *,
        session_id: str,
        seq: int | None,
        sample_rate_hz: int | None,
        mime_type: str | None,
        asr_segments=None,
        window_duration_sec: float | None = None,
    ):
        _ = (chunk_bytes, session_id, seq, sample_rate_hz, mime_type, asr_segments, window_duration_sec)
        return (
            [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "speaker": "patient",
                    "speaker_id": "spk_01",
                    "speaker_role": "patient",
                    "diar_confidence": 0.95,
                }
            ],
            {"status": "ok"},
        )

    app.state.live_audio_chunk_asr_callable = fake_chunk_asr
    app.state.live_audio_chunk_diarize_callable = fake_chunk_diarize
    chunk = base64.b64encode(b"\x0a\x0b\x0c\x0d").decode("ascii")
    try:
        with client.websocket_connect(f"/ws/audio/live?session_id={session_id}") as ws:
            ws.send_text(json.dumps({"type": "start", "mime_type": "audio/webm", "sample_rate_hz": 16000}))
            ws.receive_json()

            ws.send_text(json.dumps({"type": "audio_chunk", "seq": 0, "mime_type": "audio/webm", "data_b64": chunk}))
            ack_chunk = ws.receive_json()
            assert ack_chunk["type"] == "ack_chunk"
            assert isinstance(ack_chunk.get("sentence_role_split"), dict)
            assert ack_chunk["sentence_role_split"]["status"] == "applied"
            roles = [str(item.get("speaker_role", "")) for item in ack_chunk.get("new_utterances_payload", [])]
            assert "patient" in roles
            assert "clinician" in roles
    finally:
        delattr(app.state, "live_audio_chunk_asr_callable")
        delattr(app.state, "live_audio_chunk_diarize_callable")


def test_live_audio_ws_dedup_by_seq_for_replay() -> None:
    client = TestClient(app)
    session_id = "live_audio_dedup_case"
    app.state.live_audio_default_asr_enabled = False
    encoded = base64.b64encode(b"\x10\x20\x30\x40").decode("ascii")
    try:
        with client.websocket_connect(f"/ws/audio/live?session_id={session_id}") as ws:
            ws.send_text(json.dumps({"type": "start", "mime_type": "audio/webm", "sample_rate_hz": 16000}))
            ws.receive_json()

            ws.send_text(json.dumps({"type": "audio_chunk", "seq": 12, "mime_type": "audio/webm", "data_b64": encoded}))
            first_ack = ws.receive_json()
            assert first_ack["type"] == "ack_chunk"
            assert first_ack["duplicate"] is False
            first_count = int(first_ack["chunks_received"])

            ws.send_text(json.dumps({"type": "audio_chunk", "seq": 12, "mime_type": "audio/webm", "data_b64": encoded}))
            dup_ack = ws.receive_json()
            assert dup_ack["type"] == "ack_chunk"
            assert dup_ack["duplicate"] is True
            assert int(dup_ack["chunks_received"]) == first_count
    finally:
        if hasattr(app.state, "live_audio_default_asr_enabled"):
            delattr(app.state, "live_audio_default_asr_enabled")


def test_live_audio_default_repo_adapter_path(monkeypatch) -> None:
    client = TestClient(app)
    session_id = "live_audio_default_repo_case"
    if hasattr(app.state, "live_audio_chunk_asr_callable"):
        delattr(app.state, "live_audio_chunk_asr_callable")
    app.state.live_audio_default_asr_enabled = True

    def fake_repo_adapter(*, audio_path: str, language: str, provider: str | None, chunk_sec: int | None, overlap_sec: float | None):
        assert Path(audio_path).exists()
        assert language == "en"
        _ = (provider, chunk_sec, overlap_sec)
        from backend.asr.models import ASRSegment, SpeakerTurn
        from backend.asr.repo_chunk_adapter import RepoASRResult

        return RepoASRResult(
            segments=[ASRSegment(start=0.0, end=0.9, text="live default adapter text")],
            turns=[SpeakerTurn(start=0.0, end=0.9, speaker="other")],
            debug={"provider": "mock"},
        )

    monkeypatch.setattr("backend.api.main.transcribe_audio_with_repo_chunk_pipeline", fake_repo_adapter)
    chunk = base64.b64encode(b"\xaa\xbb\xcc\xdd").decode("ascii")
    try:
        with client.websocket_connect(f"/ws/audio/live?session_id={session_id}") as ws:
            ws.send_text(json.dumps({"type": "start", "mime_type": "audio/webm", "sample_rate_hz": 16000}))
            ws.receive_json()
            ws.send_text(json.dumps({"type": "audio_chunk", "seq": 0, "mime_type": "audio/webm", "data_b64": chunk}))
            ack_chunk = ws.receive_json()
            assert ack_chunk["type"] == "ack_chunk"
            assert ack_chunk["asr_status"] == "ok"
            assert ack_chunk["asr_mode"] == "default_repo_adapter"
    finally:
        if hasattr(app.state, "live_audio_default_asr_enabled"):
            delattr(app.state, "live_audio_default_asr_enabled")

    transcript_resp = client.get(f"/audio/live/transcript/{session_id}")
    assert transcript_resp.status_code == 200
    transcript_payload = transcript_resp.json()
    assert transcript_payload["debug"]["asr_status"] == "ok"
    assert "default adapter text" in transcript_payload["transcript_text"].lower()


def test_incremental_transcript_dedupe() -> None:
    client = TestClient(app)
    session_id = "backend_dedupe_test"

    first = client.post(
        "/transcript/incremental",
        json={
            "session_id": session_id,
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello there"},
                {"start": 2.0, "end": 3.0, "text": "i am okay"},
            ],
        },
    )
    assert first.status_code == 200
    assert len(first.json()["new_segments"]) == 2

    second = client.post(
        "/transcript/incremental",
        json={
            "session_id": session_id,
            "segments": [
                {"start": 2.0, "end": 3.0, "text": "i am okay"},
                {"start": 4.0, "end": 5.0, "text": "new update"},
            ],
        },
    )
    assert second.status_code == 200
    payload = second.json()
    assert len(payload["new_segments"]) == 1
    assert payload["debug"]["dedupe"]["duplicates_dropped"] == 1


def test_transcribe_structured_from_text_infers_speakers() -> None:
    client = TestClient(app)
    response = client.post(
        "/transcribe_structured",
        json={
            "session_id": "ts_text_case",
            "reset": True,
            "transcript_text": "Patient: I feel down for 2 weeks. Clinician: Thank you for sharing.",
            "incremental": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["source"] == "transcript_text"
    assert [item["speaker"] for item in payload["utterances"]] == ["patient", "clinician"]
    assert len(payload["segments"]) == 2


def test_transcribe_structured_with_injected_adapters() -> None:
    def fake_asr(audio_path: str, *, language: str = "en") -> list[dict[str, object]]:
        assert audio_path == "samples/demo.wav"
        assert language == "en"
        return [
            {"start": 0.0, "end": 1.0, "text": "hello there"},
            {"start": 1.2, "end": 2.1, "text": "i am okay"},
        ]

    def fake_diarize(audio_path: str) -> list[dict[str, object]]:
        assert audio_path == "samples/demo.wav"
        return [
            {"start": 0.0, "end": 1.1, "speaker": "clinician"},
            {"start": 1.1, "end": 2.2, "speaker": "patient"},
        ]

    app.state.transcribe_structured_asr_callable = fake_asr
    app.state.transcribe_structured_diarize_callable = fake_diarize
    client = TestClient(app)
    try:
        response = client.post(
            "/transcribe_structured",
            json={"session_id": "ts_audio_case", "audio_path": "samples/demo.wav", "reset": True},
        )
    finally:
        delattr(app.state, "transcribe_structured_asr_callable")
        delattr(app.state, "transcribe_structured_diarize_callable")

    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["source"] == "audio_path"
    assert payload["debug"]["diarization"]["turns_count"] == 2
    assert [item["speaker"] for item in payload["utterances"]] == ["clinician", "patient"]


def test_transcribe_structured_audio_uses_repo_adapter_when_no_injected_callable(monkeypatch) -> None:
    def fake_repo_adapter(*, audio_path: str, language: str, provider: str | None, chunk_sec: int | None, overlap_sec: float | None):
        assert audio_path == "samples/demo.wav"
        assert language == "en"
        assert provider == "mock"
        assert chunk_sec == 4
        assert overlap_sec == 0.5
        from backend.asr.models import ASRSegment, SpeakerTurn
        from backend.asr.repo_chunk_adapter import RepoASRResult

        return RepoASRResult(
            segments=[
                ASRSegment(start=0.0, end=1.0, text="adapter first"),
                ASRSegment(start=1.2, end=2.0, text="adapter second"),
            ],
            turns=[
                SpeakerTurn(start=0.0, end=1.0, speaker="patient"),
                SpeakerTurn(start=1.2, end=2.0, speaker="clinician"),
            ],
            debug={"provider": "mock", "chunk_sec": 4, "overlap_sec": 0.5},
        )

    if hasattr(app.state, "transcribe_structured_asr_callable"):
        delattr(app.state, "transcribe_structured_asr_callable")
    if hasattr(app.state, "transcribe_structured_diarize_callable"):
        delattr(app.state, "transcribe_structured_diarize_callable")

    monkeypatch.setattr("backend.api.main.transcribe_audio_with_repo_chunk_pipeline", fake_repo_adapter)
    client = TestClient(app)
    response = client.post(
        "/transcribe_structured",
        json={
            "session_id": "ts_repo_adapter_case",
            "audio_path": "samples/demo.wav",
            "asr_provider": "mock",
            "asr_chunk_sec": 4,
            "asr_overlap_sec": 0.5,
            "reset": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["asr"]["status"] == "repo_chunk_pipeline"
    assert payload["debug"]["asr"]["provider"] == "mock"
    assert [item["speaker"] for item in payload["utterances"]] == ["patient", "clinician"]


def test_transcribe_structured_audio_populates_diarization_fields(monkeypatch) -> None:
    def fake_repo_adapter(*, audio_path: str, language: str, provider: str | None, chunk_sec: int | None, overlap_sec: float | None):
        _ = (audio_path, language, provider, chunk_sec, overlap_sec)
        from backend.asr.models import ASRSegment, SpeakerTurn
        from backend.asr.repo_chunk_adapter import RepoASRResult

        return RepoASRResult(
            segments=[
                ASRSegment(start=0.0, end=1.1, text="hello there"),
                ASRSegment(start=1.2, end=2.4, text="thanks for sharing"),
            ],
            turns=[
                SpeakerTurn(start=0.0, end=2.4, speaker="other", speaker_id="other", speaker_role="other"),
            ],
            debug={"provider": "mock"},
        )

    def fake_default_diarize(audio_path: str, *, asr_segments=None, window_start_sec=None, window_end_sec=None):
        _ = (audio_path, asr_segments, window_start_sec, window_end_sec)
        return (
            [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "speaker": "patient",
                    "speaker_id": "spk_01",
                    "speaker_role": "patient",
                    "diar_confidence": 0.88,
                },
                {
                    "start": 1.2,
                    "end": 2.5,
                    "speaker": "clinician",
                    "speaker_id": "spk_02",
                    "speaker_role": "clinician",
                    "diar_confidence": 0.76,
                },
            ],
            {"status": "ok", "turns_count": 2},
        )

    if hasattr(app.state, "transcribe_structured_asr_callable"):
        delattr(app.state, "transcribe_structured_asr_callable")
    if hasattr(app.state, "transcribe_structured_diarize_callable"):
        delattr(app.state, "transcribe_structured_diarize_callable")

    monkeypatch.setattr("backend.api.main.transcribe_audio_with_repo_chunk_pipeline", fake_repo_adapter)
    monkeypatch.setattr("backend.api.main.diarize_audio_with_debug", fake_default_diarize)
    client = TestClient(app)
    response = client.post(
        "/transcribe_structured",
        json={"session_id": "ts_repo_diar_fields", "audio_path": "samples/demo.wav", "reset": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["diarization"]["status"] == "ok"
    assert payload["utterances"][0]["speaker_id"] == "spk_01"
    assert payload["utterances"][0]["speaker_role"] == "patient"
    assert payload["utterances"][0]["diar_confidence"] == 0.88
    assert payload["utterances"][1]["speaker_id"] == "spk_02"
    assert payload["utterances"][1]["speaker_role"] == "clinician"


def test_transcribe_structured_audio_diarization_failure_falls_back_to_other(monkeypatch) -> None:
    def fake_repo_adapter(*, audio_path: str, language: str, provider: str | None, chunk_sec: int | None, overlap_sec: float | None):
        _ = (audio_path, language, provider, chunk_sec, overlap_sec)
        from backend.asr.models import ASRSegment, SpeakerTurn
        from backend.asr.repo_chunk_adapter import RepoASRResult

        return RepoASRResult(
            segments=[ASRSegment(start=0.0, end=1.0, text="neutral segment")],
            turns=[SpeakerTurn(start=0.0, end=1.0, speaker="other", speaker_id="other", speaker_role="other")],
            debug={"provider": "mock"},
        )

    def fake_default_diarize(audio_path: str, *, asr_segments=None, window_start_sec=None, window_end_sec=None):
        _ = (audio_path, asr_segments, window_start_sec, window_end_sec)
        return ([], {"status": "error", "reason": "runtime_failed", "turns_count": 0})

    if hasattr(app.state, "transcribe_structured_asr_callable"):
        delattr(app.state, "transcribe_structured_asr_callable")
    if hasattr(app.state, "transcribe_structured_diarize_callable"):
        delattr(app.state, "transcribe_structured_diarize_callable")

    monkeypatch.setattr("backend.api.main.transcribe_audio_with_repo_chunk_pipeline", fake_repo_adapter)
    monkeypatch.setattr("backend.api.main.diarize_audio_with_debug", fake_default_diarize)
    client = TestClient(app)
    response = client.post(
        "/transcribe_structured",
        json={"session_id": "ts_repo_diar_fallback", "audio_path": "samples/demo.wav", "reset": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["diarization"]["status"] == "fallback_asr_turns"
    assert payload["utterances"][0]["speaker"] == "other"
    assert payload["utterances"][0]["speaker_role"] == "other"


def test_transcribe_structured_audio_fallback_single_speaker_splits_mixed_roles(monkeypatch) -> None:
    def fake_repo_adapter(*, audio_path: str, language: str, provider: str | None, chunk_sec: int | None, overlap_sec: float | None):
        _ = (audio_path, language, provider, chunk_sec, overlap_sec)
        from backend.asr.models import ASRSegment, SpeakerTurn
        from backend.asr.repo_chunk_adapter import RepoASRResult

        return RepoASRResult(
            segments=[
                ASRSegment(
                    start=0.0,
                    end=8.0,
                    text="I feel exhausted and cannot sleep well. Thank you for sharing this today.",
                )
            ],
            turns=[
                SpeakerTurn(
                    start=0.0,
                    end=8.0,
                    speaker="other",
                    speaker_id="spk_01",
                    speaker_role="other",
                    confidence=0.9,
                    diar_confidence=0.9,
                )
            ],
            debug={"provider": "mock"},
        )

    def fake_default_diarize(audio_path: str, *, asr_segments=None, window_start_sec=None, window_end_sec=None):
        _ = (audio_path, asr_segments, window_start_sec, window_end_sec)
        return ([], {"status": "error", "reason": "runtime_failed", "turns_count": 0})

    if hasattr(app.state, "transcribe_structured_asr_callable"):
        delattr(app.state, "transcribe_structured_asr_callable")
    if hasattr(app.state, "transcribe_structured_diarize_callable"):
        delattr(app.state, "transcribe_structured_diarize_callable")

    monkeypatch.setattr("backend.api.main.transcribe_audio_with_repo_chunk_pipeline", fake_repo_adapter)
    monkeypatch.setattr("backend.api.main.diarize_audio_with_debug", fake_default_diarize)
    client = TestClient(app)
    response = client.post(
        "/transcribe_structured",
        json={"session_id": "ts_repo_fallback_sentence_role_split", "audio_path": "samples/demo.wav", "reset": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["diarization"]["status"] == "fallback_asr_turns"
    assert payload["debug"]["sentence_role_split"]["status"] == "applied"
    assert len(payload["utterances"]) >= 2
    roles = [str(item.get("speaker_role", "")) for item in payload["utterances"]]
    assert "patient" in roles
    assert "clinician" in roles


def test_transcribe_structured_audio_ok_single_speaker_splits_mixed_roles(monkeypatch) -> None:
    def fake_repo_adapter(*, audio_path: str, language: str, provider: str | None, chunk_sec: int | None, overlap_sec: float | None):
        _ = (audio_path, language, provider, chunk_sec, overlap_sec)
        from backend.asr.models import ASRSegment, SpeakerTurn
        from backend.asr.repo_chunk_adapter import RepoASRResult

        return RepoASRResult(
            segments=[
                ASRSegment(
                    start=0.0,
                    end=8.0,
                    text="I feel exhausted and cannot sleep well. Thank you for sharing this today.",
                )
            ],
            turns=[
                SpeakerTurn(
                    start=0.0,
                    end=8.0,
                    speaker="patient",
                    speaker_id="spk_01",
                    speaker_role="patient",
                    confidence=0.95,
                    diar_confidence=0.95,
                )
            ],
            debug={"provider": "mock"},
        )

    def fake_default_diarize(audio_path: str, *, asr_segments=None, window_start_sec=None, window_end_sec=None):
        _ = (audio_path, asr_segments, window_start_sec, window_end_sec)
        return (
            [
                {
                    "start": 0.0,
                    "end": 8.0,
                    "speaker": "patient",
                    "speaker_id": "spk_01",
                    "speaker_role": "patient",
                    "diar_confidence": 0.95,
                }
            ],
            {"status": "ok", "turns_count": 1},
        )

    if hasattr(app.state, "transcribe_structured_asr_callable"):
        delattr(app.state, "transcribe_structured_asr_callable")
    if hasattr(app.state, "transcribe_structured_diarize_callable"):
        delattr(app.state, "transcribe_structured_diarize_callable")

    monkeypatch.setattr("backend.api.main.transcribe_audio_with_repo_chunk_pipeline", fake_repo_adapter)
    monkeypatch.setattr("backend.api.main.diarize_audio_with_debug", fake_default_diarize)
    client = TestClient(app)
    response = client.post(
        "/transcribe_structured",
        json={"session_id": "ts_repo_ok_sentence_role_split", "audio_path": "samples/demo.wav", "reset": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["diarization"]["status"] == "ok"
    assert payload["debug"]["sentence_role_split"]["status"] == "applied"
    roles = [str(item.get("speaker_role", "")) for item in payload["utterances"]]
    assert "patient" in roles
    assert "clinician" in roles


def test_transcribe_structured_audio_window_uses_repo_window_adapter(monkeypatch) -> None:
    def fake_repo_window_adapter(
        *,
        audio_path: str,
        language: str,
        start_at_sec: float,
        window_sec: float,
        provider: str | None,
        chunk_sec: int | None,
        overlap_sec: float | None,
    ):
        assert audio_path == "samples/demo.wav"
        assert language == "en"
        assert start_at_sec == 6.0
        assert window_sec == 4.0
        assert provider == "mock"
        assert chunk_sec == 4
        assert overlap_sec == 0.5
        from backend.asr.models import ASRSegment, SpeakerTurn
        from backend.asr.repo_chunk_adapter import RepoASRResult

        return RepoASRResult(
            segments=[ASRSegment(start=6.0, end=7.0, text="window segment")],
            turns=[SpeakerTurn(start=6.0, end=7.0, speaker="patient")],
            debug={
                "provider": "mock",
                "stream": {
                    "start_at_sec": 6.0,
                    "window_sec": 4.0,
                    "window_end_sec": 10.0,
                    "next_start_sec": 10.0,
                    "audio_duration_sec": 14.0,
                    "has_more": True,
                },
            },
        )

    if hasattr(app.state, "transcribe_structured_asr_callable"):
        delattr(app.state, "transcribe_structured_asr_callable")
    if hasattr(app.state, "transcribe_structured_diarize_callable"):
        delattr(app.state, "transcribe_structured_diarize_callable")

    monkeypatch.setattr("backend.api.main.transcribe_audio_window_with_repo_chunk_pipeline", fake_repo_window_adapter)
    client = TestClient(app)
    response = client.post(
        "/transcribe_structured",
        json={
            "session_id": "ts_repo_window_case",
            "audio_path": "samples/demo.wav",
            "start_at_sec": 6.0,
            "audio_window_sec": 4.0,
            "asr_provider": "mock",
            "asr_chunk_sec": 4,
            "asr_overlap_sec": 0.5,
            "reset": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["source"] == "audio_path_window"
    assert payload["debug"]["asr"]["status"] == "repo_chunk_pipeline_window"
    assert payload["debug"]["asr"]["stream"]["has_more"] is True
    assert len(payload["segments"]) == 1
    assert len(payload["new_utterances"]) == 1
    assert payload["segments"][0]["start"] == 6.0


def test_transcribe_structured_incremental_dedupe() -> None:
    client = TestClient(app)
    session_id = "ts_dedupe_case"

    first = client.post(
        "/transcribe_structured",
        json={
            "session_id": session_id,
            "reset": True,
            "segments": [
                {"start": 0.0, "end": 1.1, "text": "first line"},
                {"start": 1.2, "end": 2.0, "text": "second line"},
            ],
            "turns": [
                {"start": 0.0, "end": 1.1, "speaker": "patient"},
                {"start": 1.2, "end": 2.0, "speaker": "patient"},
            ],
            "incremental": True,
        },
    )
    assert first.status_code == 200
    assert len(first.json()["new_segments"]) == 2
    assert len(first.json()["new_utterances"]) == 1

    second = client.post(
        "/transcribe_structured",
        json={
            "session_id": session_id,
            "segments": [
                {"start": 1.2, "end": 2.0, "text": "second line"},
                {"start": 2.1, "end": 3.1, "text": "third line"},
            ],
            "turns": [
                {"start": 1.2, "end": 2.0, "speaker": "patient"},
                {"start": 2.1, "end": 3.1, "speaker": "clinician"},
            ],
            "incremental": True,
        },
    )
    assert second.status_code == 200
    payload = second.json()
    assert len(payload["new_segments"]) == 1
    assert len(payload["new_utterances"]) == 1
    assert payload["debug"]["incremental"]["duplicates_dropped"] == 1
    assert len(payload["segments"]) == 3


def test_transcribe_structured_reset_clears_role_mapping_state() -> None:
    client = TestClient(app)
    session_id = "ts_role_reset_case"

    first = client.post(
        "/transcribe_structured",
        json={
            "session_id": session_id,
            "reset": True,
            "incremental": True,
            "segments": [{"start": 0.0, "end": 1.0, "text": "I feel exhausted and down."}],
            "turns": [{"start": 0.0, "end": 1.0, "speaker": "other", "speaker_id": "spk_01", "speaker_role": "other"}],
        },
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["utterances"][0]["speaker_role"] == "patient"
    assert first_payload["debug"]["role_mapping"]["status"] == "ok"

    second = client.post(
        "/transcribe_structured",
        json={
            "session_id": session_id,
            "reset": True,
            "incremental": True,
            "segments": [{"start": 0.0, "end": 1.0, "text": "How long has this been happening?"}],
            "turns": [{"start": 0.0, "end": 1.0, "speaker": "other", "speaker_id": "spk_01", "speaker_role": "other"}],
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["utterances"][0]["speaker_role"] == "clinician"
    assert second_payload["debug"]["role_mapping"]["status"] == "ok"


def test_transcribe_structured_audio_window_disables_cross_window_merge(monkeypatch) -> None:
    def fake_repo_window_adapter(
        *,
        audio_path: str,
        language: str,
        start_at_sec: float,
        window_sec: float,
        provider: str | None,
        chunk_sec: int | None,
        overlap_sec: float | None,
    ):
        _ = (audio_path, language, start_at_sec, window_sec, provider, chunk_sec, overlap_sec)
        from backend.asr.models import ASRSegment, SpeakerTurn
        from backend.asr.repo_chunk_adapter import RepoASRResult

        return RepoASRResult(
            segments=[
                ASRSegment(start=0.0, end=1.0, text="part one"),
                ASRSegment(start=1.1, end=2.0, text="part two"),
            ],
            turns=[
                SpeakerTurn(start=0.0, end=1.0, speaker="patient"),
                SpeakerTurn(start=1.1, end=2.0, speaker="patient"),
            ],
            debug={
                "provider": "mock",
                "stream": {
                    "start_at_sec": 0.0,
                    "window_sec": 6.0,
                    "window_end_sec": 6.0,
                    "next_start_sec": 6.0,
                    "audio_duration_sec": 12.0,
                    "has_more": True,
                },
            },
        )

    if hasattr(app.state, "transcribe_structured_asr_callable"):
        delattr(app.state, "transcribe_structured_asr_callable")
    if hasattr(app.state, "transcribe_structured_diarize_callable"):
        delattr(app.state, "transcribe_structured_diarize_callable")

    monkeypatch.setattr("backend.api.main.transcribe_audio_window_with_repo_chunk_pipeline", fake_repo_window_adapter)
    client = TestClient(app)
    response = client.post(
        "/transcribe_structured",
        json={
            "session_id": "ts_window_merge_case",
            "audio_path": "samples/demo.wav",
            "start_at_sec": 0.0,
            "audio_window_sec": 6.0,
            "incremental": True,
            "reset": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["new_segments"]) == 2
    assert len(payload["new_utterances"]) == 2
    assert len(payload["utterances"]) == 2


def test_events_extract_uses_medgemma_engine_when_available(monkeypatch) -> None:
    from backend.events.extractor import ExtractedEvent
    from backend.events.medgemma_adapter import MedGemmaAdapterResult

    def fake_medgemma(*args, **kwargs):
        _ = (args, kwargs)
        return MedGemmaAdapterResult(
            events=[
                ExtractedEvent(
                    event_id="evt_mg_00001",
                    type="risk_cue",
                    label="passive_suicidal_ideation",
                    polarity="present",
                    confidence=0.91,
                    speaker="patient",
                    segment_id="seg_mg_001",
                    t0=0.0,
                    t1=2.1,
                    quote="Sometimes I wish I would not wake up.",
                )
            ],
            debug={"status": "ok", "provider": "stub"},
        )

    monkeypatch.setattr("backend.api.main.extract_events_with_medgemma", fake_medgemma)
    client = TestClient(app)
    response = client.post(
        "/events/extract",
        json={
            "engine": "medgemma",
            "fallback_to_rule": False,
            "utterances": [
                {
                    "segment_id": "seg_mg_001",
                    "t0": 0.0,
                    "t1": 2.1,
                    "speaker": "patient",
                    "text": "Sometimes I wish I would not wake up.",
                }
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["engine_used"] == "medgemma"
    assert len(payload["events"]) == 1
    assert payload["events"][0]["label"] == "passive_suicidal_ideation"


def test_events_extract_guardrail_drops_clinician_risk_false_positive(monkeypatch) -> None:
    from backend.events.extractor import ExtractedEvent
    from backend.events.medgemma_adapter import MedGemmaAdapterResult

    def fake_medgemma(*args, **kwargs):
        _ = (args, kwargs)
        return MedGemmaAdapterResult(
            events=[
                ExtractedEvent(
                    event_id="evt_mg_bad_001",
                    type="risk_cue",
                    label="passive_suicidal_ideation",
                    polarity="present",
                    confidence=0.92,
                    speaker="clinician",
                    segment_id="seg_mg_bad_001",
                    t0=6.2,
                    t1=8.0,
                    quote="Thank you for sharing this today.",
                )
            ],
            debug={"status": "ok", "provider": "stub"},
        )

    monkeypatch.setattr("backend.api.main.extract_events_with_medgemma", fake_medgemma)
    client = TestClient(app)
    response = client.post(
        "/events/extract",
        json={
            "engine": "medgemma",
            "fallback_to_rule": False,
            "utterances": [
                {
                    "segment_id": "seg_mg_bad_001",
                    "t0": 6.2,
                    "t1": 8.0,
                    "speaker": "clinician",
                    "text": "Thank you for sharing this today.",
                }
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["engine_used"] == "medgemma"
    assert payload["debug"]["event_guardrails"]["dropped"] == 1
    assert payload["events"] == []


def test_events_extract_medgemma_backstop_recovers_patient_risk(monkeypatch) -> None:
    from backend.events.extractor import ExtractedEvent
    from backend.events.medgemma_adapter import MedGemmaAdapterResult

    def fake_medgemma(*args, **kwargs):
        _ = (args, kwargs)
        return MedGemmaAdapterResult(
            events=[
                ExtractedEvent(
                    event_id="evt_mg_001",
                    type="symptom",
                    label="sleep_disturbance",
                    polarity="present",
                    confidence=0.9,
                    speaker="patient",
                    segment_id="seg_mix_001",
                    t0=4.1,
                    t1=6.2,
                    quote="Sometimes I wish I would not wake up.",
                ),
                ExtractedEvent(
                    event_id="evt_mg_002",
                    type="risk_cue",
                    label="passive_suicidal_ideation",
                    polarity="present",
                    confidence=0.9,
                    speaker="clinician",
                    segment_id="seg_mix_002",
                    t0=6.2,
                    t1=8.0,
                    quote="Thank you for sharing this today.",
                ),
            ],
            debug={"status": "ok", "provider": "stub"},
        )

    monkeypatch.setattr("backend.api.main.extract_events_with_medgemma", fake_medgemma)
    client = TestClient(app)
    response = client.post(
        "/events/extract",
        json={
            "engine": "medgemma",
            "fallback_to_rule": False,
            "utterances": [
                {
                    "segment_id": "seg_mix_001",
                    "t0": 4.1,
                    "t1": 6.2,
                    "speaker": "patient",
                    "text": "Sometimes I wish I would not wake up.",
                },
                {
                    "segment_id": "seg_mix_002",
                    "t0": 6.2,
                    "t1": 8.0,
                    "speaker": "clinician",
                    "text": "Thank you for sharing this today.",
                },
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    labels = {(item["type"], item["label"], item["polarity"], item["speaker"]) for item in payload["events"]}
    assert ("risk_cue", "passive_suicidal_ideation", "present", "patient") in labels
    assert ("risk_cue", "passive_suicidal_ideation", "present", "clinician") not in labels
    assert ("symptom", "sleep_disturbance", "present", "patient") in labels
    assert payload["debug"]["event_guardrails"]["dropped"] == 1
    assert payload["debug"]["risk_backstop"]["added"] >= 1
    assert payload["debug"]["event_harmonization"]["dropped"] == 0


def test_events_extract_auto_falls_back_to_rule_when_medgemma_errors(monkeypatch) -> None:
    from backend.events.medgemma_adapter import MedGemmaAdapterError

    def fake_medgemma(*args, **kwargs):
        _ = (args, kwargs)
        raise MedGemmaAdapterError("model unavailable")

    monkeypatch.setattr("backend.api.main.extract_events_with_medgemma", fake_medgemma)
    client = TestClient(app)
    response = client.post(
        "/events/extract",
        json={
            "engine": "auto",
            "utterances": [
                {
                    "segment_id": "seg_rule_001",
                    "t0": 0.0,
                    "t1": 2.0,
                    "speaker": "patient",
                    "text": "I have been very down for 2 weeks.",
                }
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["engine_used"] in {"rule_fallback", "rule_fallback_empty"}
    assert len(payload["events"]) >= 1


def test_events_extract_detects_split_passive_si_phrase() -> None:
    client = TestClient(app)
    response = client.post(
        "/events/extract",
        json={
            "engine": "rule",
            "utterances": [
                {
                    "segment_id": "seg_split_001",
                    "t0": 0.0,
                    "t1": 3.9,
                    "speaker": "patient",
                    "text": "I have felt very down for two weeks. Sometimes I wish I would not",
                },
                {
                    "segment_id": "seg_split_002",
                    "t0": 3.9,
                    "t1": 4.8,
                    "speaker": "patient",
                    "text": "wake up.",
                },
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    labels = {(item["type"], item["label"], item["polarity"]) for item in payload["events"]}
    assert ("risk_cue", "passive_suicidal_ideation", "present") in labels


def test_events_snapshot_note_pipeline() -> None:
    client = TestClient(app)

    events_response = client.post(
        "/events/extract",
        json={
            "utterances": [
                {
                    "segment_id": "seg_001",
                    "t0": 0.0,
                    "t1": 2.0,
                    "speaker": "patient",
                    "text": "I have been very down for 2 weeks.",
                },
                {
                    "segment_id": "seg_002",
                    "t0": 2.1,
                    "t1": 3.8,
                    "speaker": "patient",
                    "text": "Sometimes I wish I would not wake up.",
                },
            ]
        },
    )
    assert events_response.status_code == 200
    events = events_response.json()["events"]
    assert len(events) >= 3

    snapshot_response = client.post("/state/snapshot", json={"events": events})
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()
    assert len(snapshot["problem_list"]) >= 1
    assert len(snapshot["risk_flags"]) >= 1
    assert "mandatory_safety_questions" in snapshot
    assert "contextual_followups" in snapshot
    assert "rationale" in snapshot
    assert snapshot["open_questions"] == snapshot["mandatory_safety_questions"]

    note_response = client.post(
        "/note/draft",
        json={
            "department": "psych",
            "template_ids": ["psych_soap"],
            "patient_identity": "Alice Chen",
            "patient_basic_info": "Age: 34; allergies: NKDA",
            "snapshot": {
                "problem_list": snapshot["problem_list"],
                "risk_flags": snapshot["risk_flags"],
                "open_questions": snapshot["open_questions"],
            },
            "events": events,
        },
    )
    assert note_response.status_code == 200
    note = note_response.json()
    assert "Subjective:" in note["note_text"]
    assert "Assessment:" in note["note_text"]
    assert len(note["drafts"]) >= 1
    assert note["debug"]["department"] == "psych"
    assert note["debug"]["patient_identity"] == "Alice Chen"
    assert note["debug"]["patient_basic_info_included"] is True


def test_note_draft_supports_multiple_templates_and_department() -> None:
    client = TestClient(app)
    response = client.post(
        "/note/draft",
        json={
            "department": "internal_med",
            "template_ids": ["internal_soap", "internal_progress"],
            "snapshot": {
                "problem_list": [{"item": "Persistent fatigue reported", "evidence_refs": ["seg_001"]}],
                "risk_flags": [],
                "open_questions": ["Any fever, chest pain, or dyspnea since last visit?"],
            },
            "events": [
                {
                    "event_id": "evt_001",
                    "type": "symptom",
                    "label": "fatigue_low_energy",
                    "polarity": "present",
                    "confidence": 0.8,
                    "speaker": "patient",
                    "evidence": {
                        "segment_id": "seg_001",
                        "t0": 0.0,
                        "t1": 1.8,
                        "quote": "I feel tired most days.",
                    },
                }
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["department"] == "internal_med"
    assert payload["debug"]["patient_basic_info_included"] is False
    assert payload["debug"]["templates_generated"] == ["internal_soap", "internal_progress"]
    assert len(payload["drafts"]) == 2
    assert payload["drafts"][0]["template_id"] == "internal_soap"
    assert "HPI:" in payload["drafts"][0]["note_text"]
    assert payload["drafts"][1]["template_id"] == "internal_progress"
    assert "Interval Update:" in payload["drafts"][1]["note_text"]


def test_note_draft_job_start_and_complete(monkeypatch) -> None:
    client = TestClient(app)
    app.state.note_draft_jobs = {}

    def fake_build_note_drafts(**kwargs):
        template_id = str(kwargs["template_ids"][0])
        return DraftBatch(
            department=str(kwargs["department"]),
            drafts=[
                DraftTemplateNote(
                    template_id=template_id,
                    template_name=f"Template {template_id}",
                    note_text=f"Generated for {template_id}",
                    citations=[
                        DraftCitation(anchor="A1", segment_id="seg_001", t0=0.0, t1=1.0),
                    ],
                )
            ],
            requested_template_ids=list(kwargs["template_ids"]),
            missing_template_ids=[],
        )

    monkeypatch.setattr(api_main, "build_note_drafts", fake_build_note_drafts)

    start_response = client.post(
        "/note/draft/jobs/start",
        json={
            "department": "psych",
            "template_ids": ["psych_soap", "psych_consultation"],
            "patient_identity": "Case A",
            "snapshot": {
                "problem_list": [{"item": "Low mood", "evidence_refs": ["seg_001"]}],
                "risk_flags": [],
                "open_questions": ["Any worsening over the last week?"],
            },
            "events": [
                {
                    "event_id": "evt_001",
                    "type": "symptom",
                    "label": "depressed_mood",
                    "polarity": "present",
                    "confidence": 0.9,
                    "speaker": "patient",
                    "evidence": {
                        "segment_id": "seg_001",
                        "t0": 0.0,
                        "t1": 2.0,
                        "quote": "I feel down.",
                    },
                }
            ],
        },
    )
    assert start_response.status_code == 200
    start_payload = start_response.json()
    assert start_payload["job_id"].startswith("notejob_")

    job_id = start_payload["job_id"]
    final_payload = None
    for _ in range(80):
        status_response = client.get(f"/note/draft/jobs/{job_id}")
        assert status_response.status_code == 200
        final_payload = status_response.json()
        if final_payload["status"] in {"completed", "failed", "stopped_partial"}:
            break
        time.sleep(0.02)

    assert final_payload is not None
    assert final_payload["status"] == "completed"
    assert final_payload["template_statuses"]["psych_soap"] == "generated"
    assert final_payload["template_statuses"]["psych_consultation"] == "generated"
    assert [item["template_id"] for item in final_payload["drafts"]] == [
        "psych_soap",
        "psych_consultation",
    ]


def test_note_draft_job_stop_graceful(monkeypatch) -> None:
    client = TestClient(app)
    app.state.note_draft_jobs = {}

    def fake_build_note_drafts(**kwargs):
        template_id = str(kwargs["template_ids"][0])
        time.sleep(0.12)
        return DraftBatch(
            department=str(kwargs["department"]),
            drafts=[
                DraftTemplateNote(
                    template_id=template_id,
                    template_name=f"Template {template_id}",
                    note_text=f"Generated for {template_id}",
                    citations=[],
                )
            ],
            requested_template_ids=list(kwargs["template_ids"]),
            missing_template_ids=[],
        )

    monkeypatch.setattr(api_main, "build_note_drafts", fake_build_note_drafts)

    start_response = client.post(
        "/note/draft/jobs/start",
        json={
            "department": "psych",
            "template_ids": ["psych_soap", "psych_consultation"],
            "snapshot": {
                "problem_list": [{"item": "Low mood", "evidence_refs": ["seg_001"]}],
                "risk_flags": [],
                "open_questions": ["Any worsening over the last week?"],
            },
            "events": [],
        },
    )
    assert start_response.status_code == 200
    job_id = start_response.json()["job_id"]

    time.sleep(0.03)
    stop_response = client.post(f"/note/draft/jobs/{job_id}/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["stop_requested"] is True

    final_payload = None
    for _ in range(120):
        status_response = client.get(f"/note/draft/jobs/{job_id}")
        assert status_response.status_code == 200
        final_payload = status_response.json()
        if final_payload["status"] in {"completed", "failed", "stopped_partial"}:
            break
        time.sleep(0.02)

    assert final_payload is not None
    assert final_payload["status"] in {"completed", "stopped_partial"}
    if final_payload["status"] == "stopped_partial":
        assert final_payload["template_statuses"]["psych_soap"] == "generated"
        assert final_payload["template_statuses"]["psych_consultation"] in {"stopped", "queued"}
        assert len(final_payload["drafts"]) == 1


def test_note_templates_endpoint_returns_department_catalog() -> None:
    client = TestClient(app)
    response = client.get("/note/templates")
    assert response.status_code == 200
    payload = response.json()

    templates_by_department = payload["templates_by_department"]
    assert "psych" in templates_by_department
    assert "internal_med" in templates_by_department

    psych_ids = {item["template_id"] for item in templates_by_department["psych"]}
    assert "psych_soap" in psych_ids
    assert "psych_consultation" in psych_ids

    internal_ids = {item["template_id"] for item in templates_by_department["internal_med"]}
    assert "internal_soap" in internal_ids
    assert "internal_progress" in internal_ids


def test_note_template_detail_and_update_roundtrip(tmp_path, monkeypatch) -> None:
    template_root = tmp_path / "templates"
    psych_dir = template_root / "psych"
    psych_dir.mkdir(parents=True, exist_ok=True)
    template_file = psych_dir / "psych_demo.txt"
    template_file.write_text(
        "Title: Psych Demo\n\nSubjective:\n- ...\n\nPlan:\n- ...\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EVIDENTIA_NOTE_TEMPLATE_DIR", str(template_root))

    client = TestClient(app)
    get_response = client.get("/note/templates/psych/psych_demo")
    assert get_response.status_code == 200
    assert get_response.json()["template"]["template_name"] == "Psych Demo"

    put_response = client.put(
        "/note/templates/psych/psych_demo",
        json={
            "template": {
                "template_id": "psych_demo",
                "template_name": "Psych Demo Updated",
                "template_text": "Subjective:\n- Updated\n\nAssessment:\n- Updated\n\nPlan:\n- Updated",
            }
        },
    )
    assert put_response.status_code == 200
    assert put_response.json()["template"]["template_name"] == "Psych Demo Updated"

    verify_response = client.get("/note/templates/psych/psych_demo")
    assert verify_response.status_code == 200
    payload = verify_response.json()
    assert payload["template"]["template_name"] == "Psych Demo Updated"
    assert "Assessment:" in payload["template"]["template_text"]
