from fastapi.testclient import TestClient
from pathlib import Path

from backend.api.main import app
from backend.asr.repo_chunk_adapter import RepoASRAdapterError
from backend.events.medgemma_adapter import MedGemmaAdapterError


def _clear_injected_transcribe_callables() -> None:
    if hasattr(app.state, "transcribe_structured_asr_callable"):
        delattr(app.state, "transcribe_structured_asr_callable")
    if hasattr(app.state, "transcribe_structured_diarize_callable"):
        delattr(app.state, "transcribe_structured_diarize_callable")


def test_transcribe_structured_rejects_empty_payload() -> None:
    client = TestClient(app)
    response = client.post("/transcribe_structured", json={"session_id": "err_empty_payload"})
    assert response.status_code == 400
    assert "Provide one of" in response.json()["detail"]


def test_transcribe_structured_repo_adapter_failure_returns_500(monkeypatch) -> None:
    _clear_injected_transcribe_callables()

    def fake_repo_adapter(**kwargs):
        _ = kwargs
        raise RepoASRAdapterError("repo adapter failed for test")

    monkeypatch.setattr("backend.api.main.transcribe_audio_with_repo_chunk_pipeline", fake_repo_adapter)
    client = TestClient(app)
    response = client.post(
        "/transcribe_structured",
        json={"session_id": "err_repo_adapter", "audio_path": "samples/demo.wav"},
    )
    assert response.status_code == 500
    assert "repo adapter failed for test" in response.json()["detail"]


def test_transcribe_structured_injected_asr_file_missing_returns_404() -> None:
    def fake_asr(audio_path: str, *, language: str = "en"):
        _ = (audio_path, language)
        raise FileNotFoundError("missing file for test")

    app.state.transcribe_structured_asr_callable = fake_asr
    client = TestClient(app)
    try:
        response = client.post(
            "/transcribe_structured",
            json={"session_id": "err_missing_file", "audio_path": "samples/missing.wav"},
        )
    finally:
        _clear_injected_transcribe_callables()

    assert response.status_code == 404
    assert "missing file for test" in response.json()["detail"]


def test_transcribe_structured_injected_asr_invalid_schema_returns_400() -> None:
    def fake_asr(audio_path: str, *, language: str = "en"):
        _ = (audio_path, language)
        return [{"start": 1.0, "end": 0.5, "text": "bad window"}]

    app.state.transcribe_structured_asr_callable = fake_asr
    client = TestClient(app)
    try:
        response = client.post(
            "/transcribe_structured",
            json={"session_id": "err_invalid_schema", "audio_path": "samples/demo.wav"},
        )
    finally:
        _clear_injected_transcribe_callables()

    assert response.status_code == 400
    assert "Invalid ASR output" in response.json()["detail"]


def test_events_extract_forced_medgemma_without_fallback_returns_500(monkeypatch) -> None:
    def fake_medgemma(*args, **kwargs):
        _ = (args, kwargs)
        raise MedGemmaAdapterError("medgemma unavailable for test")

    monkeypatch.setattr("backend.api.main.extract_events_with_medgemma", fake_medgemma)
    client = TestClient(app)
    response = client.post(
        "/events/extract",
        json={
            "engine": "medgemma",
            "fallback_to_rule": False,
            "utterances": [
                {
                    "segment_id": "seg_med_001",
                    "t0": 0.0,
                    "t1": 1.2,
                    "speaker": "patient",
                    "text": "I feel down.",
                }
            ],
        },
    )
    assert response.status_code == 500
    assert "medgemma unavailable for test" in response.json()["detail"]


def test_events_extract_payload_validation_error_returns_422() -> None:
    client = TestClient(app)
    response = client.post(
        "/events/extract",
        json={"utterances": [{"segment_id": "seg_missing_fields"}]},
    )
    assert response.status_code == 422


def test_note_draft_missing_template_ids_returns_422() -> None:
    client = TestClient(app)
    response = client.post(
        "/note/draft",
        json={
            "department": "psych",
            "snapshot": {"problem_list": [], "risk_flags": [], "open_questions": []},
            "events": [],
        },
    )
    assert response.status_code == 422


def test_note_draft_invalid_template_ids_returns_400() -> None:
    client = TestClient(app)
    response = client.post(
        "/note/draft",
        json={
            "department": "internal_med",
            "template_ids": ["unknown_template_a", "unknown_template_b"],
            "snapshot": {"problem_list": [], "risk_flags": [], "open_questions": []},
            "events": [],
        },
    )
    assert response.status_code == 400
    assert "No valid templates" in response.json()["detail"]


def test_note_draft_unsupported_department_returns_400() -> None:
    client = TestClient(app)
    response = client.post(
        "/note/draft",
        json={
            "department": "unknown_dept",
            "template_ids": ["psych_soap"],
            "snapshot": {"problem_list": [], "risk_flags": [], "open_questions": []},
            "events": [],
        },
    )
    assert response.status_code == 400
    assert "Unsupported department" in response.json()["detail"]


def test_note_template_detail_unknown_template_returns_400() -> None:
    client = TestClient(app)
    response = client.get("/note/templates/psych/unknown_template")
    assert response.status_code == 400
    assert "Unknown template_id" in response.json()["detail"]


def test_note_template_update_payload_template_id_mismatch_returns_400() -> None:
    client = TestClient(app)
    response = client.put(
        "/note/templates/psych/psych_soap",
        json={
            "template": {
                "template_id": "psych_follow_up",
                "template_name": "Psychiatric SOAP Note",
                "template_text": "Subjective:\n- ...\n\nObjective:\n- ...\n\nAssessment:\n- ...\n\nPlan:\n- ...",
            }
        },
    )
    assert response.status_code == 400
    assert "must match path template_id" in response.json()["detail"]


def test_note_draft_job_status_unknown_job_returns_404() -> None:
    client = TestClient(app)
    response = client.get("/note/draft/jobs/notejob_missing")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_note_draft_job_stop_unknown_job_returns_404() -> None:
    client = TestClient(app)
    response = client.post("/note/draft/jobs/notejob_missing/stop")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_upload_audio_rejects_unsupported_extension() -> None:
    client = TestClient(app)
    response = client.post(
        "/files/upload-audio?filename=notes.txt",
        data=b"hello",
        headers={"content-type": "text/plain"},
    )
    assert response.status_code == 400
    assert "Only .wav, .mp3, .webm, .ogg, .m4a, or .mp4 files are accepted." in response.json()["detail"]


def test_upload_audio_saves_file_and_returns_absolute_path(tmp_path) -> None:
    app.state.uploaded_audio_dir = str(tmp_path)
    client = TestClient(app)
    try:
        response = client.post(
            "/files/upload-audio?filename=sample.wav",
            data=b"RIFF....WAVEfmt ",
            headers={"content-type": "audio/wav"},
        )
    finally:
        if hasattr(app.state, "uploaded_audio_dir"):
            delattr(app.state, "uploaded_audio_dir")

    assert response.status_code == 200
    payload = response.json()
    saved_path = Path(payload["audio_path"])
    assert saved_path.is_absolute()
    assert saved_path.exists()
    assert saved_path.suffix == ".wav"
    assert payload["size_bytes"] > 0


def test_get_audio_file_requires_audio_path() -> None:
    client = TestClient(app)
    response = client.get("/files/audio")
    assert response.status_code == 400
    assert "audio_path is required" in response.json()["detail"]


def test_get_audio_file_not_found_returns_404(tmp_path) -> None:
    client = TestClient(app)
    missing = tmp_path / "missing.wav"
    response = client.get("/files/audio", params={"audio_path": str(missing)})
    assert response.status_code == 404
    assert "Audio file not found" in response.json()["detail"]


def test_get_audio_file_returns_file_content(tmp_path) -> None:
    client = TestClient(app)
    sample = tmp_path / "clip.wav"
    payload = b"RIFFtestWAVEfmt "
    sample.write_bytes(payload)

    response = client.get("/files/audio", params={"audio_path": str(sample)})
    assert response.status_code == 200
    assert response.content == payload


def test_list_sample_audio_files_returns_sorted_wav_mp3_only(tmp_path) -> None:
    app.state.sample_audio_dir = str(tmp_path)
    client = TestClient(app)
    try:
        (tmp_path / "b_larger.mp3").write_bytes(b"\x01" * 30)
        (tmp_path / "a_smaller.wav").write_bytes(b"\x01" * 10)
        (tmp_path / "ignore.txt").write_text("not audio", encoding="utf-8")

        response = client.get("/files/sample-audio")
    finally:
        if hasattr(app.state, "sample_audio_dir"):
            delattr(app.state, "sample_audio_dir")

    assert response.status_code == 200
    payload = response.json()
    files = payload["files"]
    assert [item["name"] for item in files] == ["a_smaller.wav", "b_larger.mp3"]
    assert [item["size_bytes"] for item in files] == [10, 30]
    assert all(Path(str(item["audio_path"])).is_absolute() for item in files)


def test_list_sample_audio_files_returns_empty_when_directory_missing(tmp_path) -> None:
    missing_dir = tmp_path / "missing_samples"
    app.state.sample_audio_dir = str(missing_dir)
    client = TestClient(app)
    try:
        response = client.get("/files/sample-audio")
    finally:
        if hasattr(app.state, "sample_audio_dir"):
            delattr(app.state, "sample_audio_dir")

    assert response.status_code == 200
    payload = response.json()
    assert payload["files"] == []
