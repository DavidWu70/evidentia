# Evidentia Backend (New Skeleton)

This directory is the active backend for MVP iteration.

## Run

```bash
cd /Users/davidwu/Scribe-AI/evidentia
/Users/davidwu/Scribe-AI/.venv311/bin/python -m uvicorn backend.api.main:app --reload --port 8000
```

## Tests

```bash
cd /Users/davidwu/Scribe-AI/evidentia
/Users/davidwu/Scribe-AI/.venv311/bin/python -m pytest -q backend/tests
```

## E2E smoke

```bash
cd /Users/davidwu/Scribe-AI/evidentia
/Users/davidwu/Scribe-AI/.venv311/bin/python scripts/smoke_e2e_mvp.py --in-process --mode transcript --event-engine rule
```

## MedGemma debug metrics

If `EVIDENTIA_MEDGEMMA_DEBUG_LOG` is enabled, summarize timing and JSON-valid stats from raw logs:

```bash
cd /Users/davidwu/Scribe-AI/evidentia
/Users/davidwu/Scribe-AI/.venv311/bin/python backend/scripts/medgemma_log_stats.py --log-path /private/tmp/evidentia_medgemma_raw.log
```

MedGemma extraction mode (for smaller local models):
- `EVIDENTIA_MEDGEMMA_PER_UTTERANCE=1` (default): run one inference per utterance with compact prompt text
- `EVIDENTIA_MEDGEMMA_PER_UTTERANCE=0`: legacy batch prompt across all utterances
- extraction output applies strict `type-label` compatibility validation (invalid combinations are dropped)

## Local model paths (repo-independent defaults)

The backend no longer depends on `../repo` for ASR/audio utilities.

Model path precedence:
- explicit API/model argument
- env var
- local auto-discovery

Useful env overrides:
- `EVIDENTIA_MODEL_ROOT`: optional base directory for all local models
- `EVIDENTIA_WHISPER_CPP_BIN` / `SCRIBE_WHISPER_CPP_BIN`
- `EVIDENTIA_WHISPER_CPP_MODEL` / `SCRIBE_WHISPER_CPP_MODEL`
- `EVIDENTIA_MEDASR_MODEL` / `SCRIBE_MEDASR_MODEL`
- `EVIDENTIA_MEDGEMMA_GGUF` / `SCRIBE_LLAMA_CPP_MODEL`

Auto-discovery checks common local locations under:
- `<evidentia>/`
- `<evidentia>/models/`
- `<evidentia>/../`
- `<evidentia>/../models/`

## Note generation debug log

Enable note-generation raw logs with:

- `EVIDENTIA_NOTE_DEBUG_LOG=1` (writes to `/tmp/evidentia_note_raw.log`)
- or set `EVIDENTIA_NOTE_DEBUG_LOG=/custom/path.log`
- if `EVIDENTIA_NOTE_DEBUG_LOG` is unset, note debug logging can reuse `EVIDENTIA_MEDGEMMA_DEBUG_LOG`

Typical stages include:

- `note_generation_start` (prompt + start metadata)
- `note_prompt_input` (runtime settings only; prompt body is not duplicated)
- `note_raw_output` (raw model output text)
- `note_generation_end` (status + elapsed metadata)

## Module map

- `backend/asr`: transcript contracts, alignment, formatting, incremental buffer
- `backend/events`: minimal evidence-backed event extraction
- `backend/risk`: rolling state snapshot (problem/risk/open questions)
- `backend/note`: citation-backed draft note generation from plain-text templates
- `backend/api`: FastAPI endpoints and orchestration

## MVP endpoints

- `GET /healthz`
- `GET /audio/live/status/{session_id}`
- `GET /audio/live/transcript/{session_id}`
- `GET /note/templates`
- `GET /note/templates/{department}/{template_id}`
- `POST /transcribe_structured`
- `POST /transcript/incremental`
- `POST /events/extract`
- `POST /state/snapshot`
- `POST /note/draft`
- `POST /files/upload-audio`
- `PUT /note/templates/{department}/{template_id}`
- `WS /ws/audio/live?session_id=...`

`/transcribe_structured` supports three input modes:
- `segments` payload
- `transcript_text` payload
- `audio_path` (default path uses repo chunk ASR adapter; override with `app.state.transcribe_structured_asr_callable`)

`/transcribe_structured` diarization behavior (`audio_path`):
- `payload.turns` present: use caller-provided turns directly
- `app.state.transcribe_structured_diarize_callable` present: use injected diarization callable
- otherwise use built-in adapter (`VAD/SAD + embedding + clustering + turn segmentation`) for `audio_path`
- diarization failure does not fail transcription: fallback to ASR turns, then fallback to inferred `other`

`utterances[]` now includes diarization metadata (when available):
- `speaker` (backward-compatible role-like label: `patient/clinician/other`)
- `speaker_id` (e.g. `spk_01`)
- `speaker_role` (`patient/clinician/other`)
- `diar_confidence` (`0..1`, optional)
- `role_confidence` (`0..1`, optional; confidence of `speaker_id -> speaker_role` mapping)

Role mapping behavior:
- backend keeps a session-stable `speaker_id -> patient/clinician/other` map for `audio_path` and `ws/audio/live`
- low-confidence or conflicting evidence keeps role as `other` (safety-first fallback)
- `reset=true` clears role-mapping state for that `session_id`
- debug visibility:
  - `/transcribe_structured`: `debug.role_mapping`
  - `ws/audio/live`: response exposes `diarization_status/mode`; detailed `role_mapping` debug is kept in backend diarization debug payload

Alignment diagnostics (for `other` fallback investigation):
- `/transcribe_structured` now includes `debug.alignment` with:
  - `reason_counts` (e.g., `fallback_no_overlap`, `fallback_low_overlap`, `fallback_ambiguous_overlap`)
  - `fallback_total/fallback_rate`
  - `fallback_samples` (timestamp + text snippet + overlap metrics)
- `/transcribe_structured` also includes `debug.sentence_role_split` for fallback single-speaker sentence-level role split diagnostics.
- `ws/audio/live` now also includes sentence-level split diagnostics:
  - ws `ack_chunk`: `sentence_role_split`
  - `GET /audio/live/transcript/{session_id}`: `debug.sentence_role_split`
- optional backend log switch:
  - `EVIDENTIA_ALIGNMENT_DEBUG_LOG=1`
  - when enabled, backend emits `alignment_debug ...` lines (summary + sample fallbacks)

`/transcribe_structured` audio streaming knobs:
- `start_at_sec`: absolute audio offset for current window start
- `audio_window_sec`: when set, run ASR on one window and return stream progress in `debug.asr.stream`
- `incremental=true`: merge each window result into session buffer for rolling transcript/evidence updates

`/events/extract` supports engines:
- `engine=auto` (default): try MedGemma adapter, then fallback to rule extraction
- `engine=medgemma`: force MedGemma (set `fallback_to_rule=true` to allow fallback)
- `engine=rule`: deterministic regex/keyword extraction only
- post-processing notes:
  - `event_guardrails`: conservative false-positive suppression (e.g., clinician risk false positives)
  - `risk_backstop` (when `engine_used=medgemma`): deterministic risk recovery for missed high-priority cues
  - `event_harmonization`: currently non-destructive (`reason=no_conflict_rules_active`)

`/state/snapshot` supports hybrid open-questions output:
- request field:
  - `ai_enhancement_enabled` (optional bool)
    - default behavior: enabled
    - set `false` to force rule-only open questions
- response fields:
  - `open_questions` (merged list, backward-compatible)
  - `mandatory_safety_questions` (rule-based mandatory questions)
  - `contextual_followups` (optional AI-generated follow-up, at most one)
  - `rationale` (explainability text)
- debug fields:
  - `open_questions_mode` (`rule` or `hybrid`)
  - `ai_enhancement_enabled`
  - `ai_enhancement_applied`
  - `ai_enhancement_error` (empty when successful)

`/note/draft` now supports department-scoped multi-template generation:
- request fields:
  - `department` (e.g., `psych`, `internal_med`)
  - `template_ids` (array; one request can generate multiple drafts)
- strict behavior:
  - unknown `department` returns `400`
  - missing/empty `template_ids` returns `422` (request validation)
  - invalid `template_ids` (not found in department) returns `400`
  - no implicit template fallback is applied
- response fields:
  - legacy-compatible `note_text` / `citations` (first generated draft)
  - `drafts[]` with `{template_id, template_name, note_text, citations}`
- generation behavior:
  - note templates are embedded into LLM prompt as plain text
  - model output is expected to be plain note text (not JSON)
  - if model emits `<note>...</note>`, backend strips wrapper tags before returning `note_text`
  - if model is unavailable and `EVIDENTIA_NOTE_STRICT_LLM` is not set, backend falls back to template-first plain-text fill

`/note/templates` returns file-backed editable template catalog:
- source directory: `backend/note/templates/<department>/*.txt` (or `*.md`)
- env override: `EVIDENTIA_NOTE_TEMPLATE_DIR=/custom/path`
- each template file uses:
  - first line `Title: <template name>`
  - remaining content as plain-text template body

Template editor endpoints:
- `GET /note/templates/{department}/{template_id}` returns `{template_id, template_name, template_text}`
- `PUT /note/templates/{department}/{template_id}` writes updated plain-text template to disk
- strict validation:
  - `template_id` in payload must match path
  - unknown `department` or `template_id` returns `400`

## Incremental Extraction Mode

For long-session stability in `audio_path` window mode:

- `/transcribe_structured` returns both `utterances` (session-accumulated) and `new_utterances` (this tick only)
- `new_utterances/utterances` preserve `speaker_id/speaker_role/diar_confidence` fields for frontend transcript rendering
- Audio-window mode uses a strict merge gap to avoid cross-window mega-utterance merges
- Frontend can run extraction with local reconcile windows:
  - `reconcile_lookback_windows = 0`: pure incremental extraction
  - `reconcile_lookback_windows = N > 0`: extract on `current_window + previous N windows`

Design intent:
- avoid full-history MedGemma prompt growth on long audio
- preserve near-context correction capability without full-session recompute

All core modules should keep explanatory design-intent comments.

## Next Iteration Roadmap (Added 2026-02-23)

Priority order:
1. Real-time transcription
2. Postgres persistence + replay/audit

### 1) Real-time transcription

Current status:
- `transcript` mode is quasi-stream tick-based.
- `audio_path` mode is window streaming based on `start_at_sec + audio_window_sec`.
- `ws/audio/live` + `audio/live/transcript/{session_id}` provide a basic mic-stream ingest + incremental transcript pull path.
- Frontend `Start Mic` now consumes WS `ack_chunk` payloads (`new_utterances`) directly for event-driven transcript/event updates.
- `GET /audio/live/transcript/{session_id}` remains as fallback/debug pull endpoint.
- Default live chunk ASR path is enabled:
  - if `app.state.live_audio_chunk_asr_callable` is provided, use injected callable first
  - otherwise use repo adapter on per-chunk temp audio files (provider from repo config or `app.state.live_audio_asr_provider`)
  - set `app.state.live_audio_default_asr_enabled = False` to force `not_configured` mode
- Live diarization path is enabled for ws chunks:
  - if `app.state.live_audio_chunk_diarize_callable` is provided, use injected callable first
  - otherwise use built-in local diarization adapter per ws window chunk
  - ws `ack_chunk` now includes `diarization_status` / `diarization_mode`
  - diarization failure falls back to `infer_turns_from_segments` and keeps transcript available
- Resilience behavior:
  - frontend persists unsent mic chunks in IndexedDB (survives refresh/crash) and auto-reconnects websocket
  - backend deduplicates replayed chunks by `seq` to avoid duplicate transcript/events
  - if reconnect budget is exhausted, frontend auto-builds a fallback audio file from buffered chunks, uploads it, and switches to `audio_path` mode
  - `Stop Mic` is graceful: stop capture first, drain queued/in-flight chunks, then close websocket (timeout falls back to `stopped_with_pending_queue`)
- Live mic ASR windowing:
  - frontend captures mic in configurable slices (`Live Mic Slice (ms)`, default `1000`)
  - ASR requests are emitted with configurable sliding window (`Live ASR Window (s)`, default `20`)
  - window push cadence is configurable (`Live Push Step (s)`, default `4`)
  - backend applies `window_start_sec/window_duration_sec` to align overlapping windows on absolute timeline

Next target:
- replace polling with event-driven push for transcript/event updates
- surface connection/recovery state in pipeline debug
- optimize live chunk ASR latency/stability under long sessions

### 2) Postgres persistence + replay/audit

Next target:
- persist patient/session/config/template/transcript/events/snapshot/note artifacts
- support replay by `session_id` after restart
- provide audit query fields (timestamps, model/template/version, runtime knobs)
