# Evidentia – An Agentic Clinical Reasoning Assistant

Minimal local run:

```bash
cd /path/to/evidentia
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn backend.api.main:app --reload --port 8000
```

Project layout:

- `backend/`: FastAPI service (ASR, events, risk, notes)
- `frontend/`: static web UI
- `samples/`: sample audio files for demos

Optional local-model features (MedGemma / MedASR):

```bash
pip install -r requirements.optional.txt
```

Run backend tests:

```bash
pip install -r requirements.dev.txt
pytest -q backend/tests
```

MedGemma timing/json-valid summary from debug log:

```bash
python backend/scripts/medgemma_log_stats.py --log-path /tmp/evidentia_medgemma_raw.log
```

Model path overrides (recommended in production):

- `EVIDENTIA_MODEL_ROOT`
- `EVIDENTIA_MEDGEMMA_GGUF`
- `EVIDENTIA_MEDASR_MODEL`
- `EVIDENTIA_WHISPER_CPP_BIN`
- `EVIDENTIA_WHISPER_CPP_MODEL`

Legacy `SCRIBE_*` model env vars are still supported for compatibility.
