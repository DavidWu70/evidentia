# Frontend MVP Runbook

This frontend is a lightweight browser UI under `evidentia/frontend/`.
Overall project status/backlog: `evidentia/docs/mvp_status_backlog.md`.

Frontend stack note:
- The current UI is a minimal React app loaded via ESM CDN (`react`, `react-dom`, `htm`).
- Keep network access enabled for first-load dependency fetch.

## Purpose

Provide a review demo for:

1. incremental transcript updates
2. evidence timeline
3. risk/open-questions view
4. editable draft note export

## Prerequisites

1. Start backend service from `evidentia/` (new backend skeleton):

```bash
cd /Users/davidwu/Scribe-AI/evidentia
/Users/davidwu/Scribe-AI/.venv311/bin/python -m uvicorn backend.api.main:app --reload --port 8000
```

2. Serve frontend static files from `evidentia/frontend`:

```bash
cd /Users/davidwu/Scribe-AI/evidentia/frontend
python3 -m http.server 5173 --bind 127.0.0.1
```

Open: `http://127.0.0.1:5173`

Quick check in another terminal:

```bash
curl -I http://127.0.0.1:5173
```

Expected: HTTP 200.

## Demo flow

1. Use left navigation:
   - `Transcript`: review/default page
   - `Template`: template list + editor
   - `Config`: runtime knobs
2. Keep default backend URL `http://127.0.0.1:8000`
3. Choose `Input Mode`:
   - `transcript` for deterministic quasi-stream demo
   - `audio_path` for real ASR path
4. If `audio_path`, fill `Audio Path` with an absolute local file path
5. Set `Audio Window (s)` (recommended `4-8` for demo smoothness)
6. Set `Reconcile Lookback (windows)`:
   - `0` for pure incremental extraction
   - `1-3` for short local reconcile
7. Set `LLM Update Every (windows)` (only used when `Event Engine=auto`, recommended `2-4`)
8. Choose `Event Engine` (`auto` recommended)
9. In `Config`, set `Open Questions AI Enhancement`:
   - `enabled` (default): rule-based mandatory questions + optional contextual AI follow-up
   - `disabled`: rule-only questions
10. Choose `Department` (global field under subtitle)
11. Paste or keep demo transcript text when in `transcript` mode
12. Click `Start`
13. Watch pipeline update in order:
   - transcript cards
   - event timeline
   - risk flags/open questions
   - draft note
14. Click any timeline card to jump/highlight matching transcript segment
15. Confirm `Pipeline Debug` card in right panel:
   - `source=audio_path_window` and `asr=repo_chunk_pipeline_window` means chunked audio ASR is active
   - `status` field progress like `running (12.0/38.5s)` indicates window-by-window streaming
   - In `auto`, timeline is refreshed by `rule` first, then asynchronously upgraded by `medgemma` when available
   - `event engine used=medgemma` means async MedGemma refresh succeeded on latest update
   - `open_questions_mode=hybrid` means contextual AI follow-up was included; `rule` means rule-only output
16. (Optional live mic path) In `Transcript` tab click `Start Mic`:
   - browser starts mic capture and streams chunk-by-chunk to `WS /ws/audio/live`
   - frontend consumes WS `ack_chunk.new_utterances_payload` directly (event-driven)
   - `GET /audio/live/transcript/{session_id}` is kept as fallback/debug pull endpoint
   - mic pipeline uses configurable parameters from Config:
     - `Live Mic Slice (ms)` (default `1000`)
     - `Live ASR Window (s)` (default `20`)
     - `Live Push Step (s)` (default `4`)
   - frontend keeps unsent mic chunks in IndexedDB persistent queue; on disconnect it auto-reconnects and replays buffered chunks
   - backend deduplicates replayed chunks by `seq` (prevents duplicate transcript append)
   - if reconnect attempts exceed threshold, frontend auto-generates a fallback audio file from buffered chunks, uploads it, and switches to `audio_path` mode
   - `Stop Mic` is graceful: capture stops immediately, then queued/in-flight chunks are drained before websocket close
   - `Pipeline Debug` should show `source=live_audio_ws` and live counters (`live_audio_chunks/live_audio_bytes`)
   - default backend path runs repo chunk ASR per mic chunk; if provider/env is unavailable, `asr_status` becomes `error`
   - set `app.state.live_audio_default_asr_enabled = False` (or provide custom `live_audio_chunk_asr_callable`) for deterministic debugging
17. In `Template` nav:
   - select a template from left list
   - edit template via form fields (template name + section rows) on right
   - click `Save Template` to write back

## Notes

- `transcript` mode keeps simulated chunk streaming for deterministic demos.
- `audio_path` mode uses chunked window streaming via `start_at_sec + audio_window_sec`.
- `Start Mic` path defaults to repo chunk ASR adapter; custom adapter override is `app.state.live_audio_chunk_asr_callable`.
- Live mic unsent queue is persisted in IndexedDB for recovery after refresh/crash.
- Segment timing and quote evidence are preserved end-to-end.
- Disclaimer is visible in UI header by design.
- Note templates are file-backed under `backend/note/templates/<department>/*.txt` (compat: `*.md`); edit template files then refresh frontend.

## Extraction Behavior (Implemented)

- Event extraction in `audio_path` mode uses incremental `new_utterances` per window tick.
- Reconcile strategy is bounded by `Reconcile Lookback (windows)` in UI:
  - `0`: no reconcile (pure incremental mode)
  - `N > 0`: reconcile only within `current + previous N windows`
- In `Event Engine=auto`, the UI runs:
  - realtime `rule` extraction each window
  - asynchronous `medgemma` refresh every `LLM Update Every (windows)` ticks
- Full-session reconcile is intentionally avoided for long audio to reduce model latency and JSON-format instability.
- Open Questions are hybrid by default:
  - mandatory safety questions are rule-driven
  - optional contextual follow-up is LLM-generated
  - when LLM enhancement is disabled or fails, output degrades to rule-only without breaking UI

## Next Iteration Priorities (Added 2026-02-23)

1. Real-time transcription first
2. Postgres persistence + replay/audit second

Notes:
- Current frontend streaming is quasi-realtime (`transcript` tick or `audio_path` window).
- Next realtime milestone is true live mic streaming with event-driven UI updates.
- Persistence/replay milestone comes after realtime path stabilizes, to avoid premature schema lock-in.

## Smoke script

In-process smoke (no running server required):

```bash
cd /Users/davidwu/Scribe-AI/evidentia
/Users/davidwu/Scribe-AI/.venv311/bin/python scripts/smoke_e2e_mvp.py --in-process --mode transcript --event-engine rule
```

Server smoke (requires backend running):

```bash
cd /Users/davidwu/Scribe-AI/evidentia
/Users/davidwu/Scribe-AI/.venv311/bin/python scripts/smoke_e2e_mvp.py --base-url http://127.0.0.1:8000 --mode transcript --event-engine auto
```

Audio-path smoke (requires backend running + local audio):

```bash
cd /Users/davidwu/Scribe-AI/evidentia
/Users/davidwu/Scribe-AI/.venv311/bin/python scripts/smoke_e2e_mvp.py --base-url http://127.0.0.1:8000 --mode audio --audio-path /absolute/path/to/sample.wav --event-engine auto
```

Fallback checks (in-process, deterministic):

```bash
cd /Users/davidwu/Scribe-AI/evidentia
/Users/davidwu/Scribe-AI/.venv311/bin/python scripts/demo_fallback_checks.py --in-process
```
