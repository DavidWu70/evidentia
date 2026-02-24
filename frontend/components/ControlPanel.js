import React from "https://esm.sh/react@18";
import htm from "https://esm.sh/htm@3";

const html = htm.bind(React.createElement);

/**
 * React control panel for demo pipeline configuration.
 *
 * Design intent:
 * - Keep runtime controls explicit so reviewers can reproduce edge cases quickly.
 * - Expose engine selection to validate MedGemma vs rule fallback behavior.
 */
export function ControlPanel({
  mode = "context",
  showOpenQuestionsAiSetting = false,
  baseUrl,
  sessionId,
  intervalMs,
  inputMode,
  audioPath,
  audioPathDisplay = "",
  audioSampleOptions = [],
  selectedAudioSource = "",
  localAudioSourceValue = "__local_file__",
  audioWindowSec,
  reconcileLookbackWindows,
  llmUpdateIntervalWindows,
  openQuestionsAiEnhancementEnabled = true,
  micCaptureSliceMs = 1000,
  micAsrWindowSec = 20,
  micAsrStepSec = 4,
  audioPathLocked = false,
  eventEngine,
  streamText,
  statusText,
  runStatus = "",
  micStatus = "",
  micSupported = false,
  onBaseUrlChange,
  onSessionIdChange,
  onIntervalChange,
  onInputModeChange,
  onAudioPathChange,
  onAudioSourceChange,
  onUploadAudioFile,
  onAudioWindowSecChange,
  onReconcileLookbackChange,
  onLlmUpdateIntervalChange,
  onOpenQuestionsAiEnhancementChange,
  onMicCaptureSliceMsChange,
  onMicAsrWindowSecChange,
  onMicAsrStepSecChange,
  onEngineChange,
  onStreamTextChange,
  onStart,
  onStop,
  onReset,
  onStartMic,
  onStopMic,
}) {
  const [showUploadPanel, setShowUploadPanel] = React.useState(false);
  const [uploadingAudio, setUploadingAudio] = React.useState(false);
  const [uploadStatus, setUploadStatus] = React.useState("");
  const [uploadStatusTone, setUploadStatusTone] = React.useState("info");
  const [dropzoneDragState, setDropzoneDragState] = React.useState("idle");
  const filePickerRef = React.useRef(null);
  const dragDepthRef = React.useRef(0);
  const isMicRecording = (
    micStatus === "requesting_permission"
    || micStatus === "connecting"
    || micStatus === "reconnecting"
    || micStatus === "streaming"
  );
  const isBatchRunning = inputMode !== "live" && runStatus === "running";
  const showStopButton = isMicRecording || micStatus === "stopping";
  const showLocalAudioPicker = !selectedAudioSource || selectedAudioSource === localAudioSourceValue;

  const uploadAvailable = typeof onUploadAudioFile === "function";

  const validateAudioFile = React.useCallback((file) => {
    if (!file) {
      return "No file selected.";
    }
    const name = String(file.name || "").toLowerCase();
    if (!name.endsWith(".wav") && !name.endsWith(".mp3")) {
      return "Only .wav or .mp3 files are accepted.";
    }
    return "";
  }, []);

  const isSupportedAudioFile = React.useCallback((file) => {
    const name = String(file?.name || "").toLowerCase();
    const mime = String(file?.type || "").toLowerCase();
    if (name.endsWith(".wav") || name.endsWith(".mp3")) {
      return true;
    }
    return mime === "audio/wav"
      || mime === "audio/x-wav"
      || mime === "audio/mpeg"
      || mime === "audio/mp3";
  }, []);

  const resolveDraggedFileState = React.useCallback((dataTransfer) => {
    if (!dataTransfer) {
      return "idle";
    }
    const directFiles = Array.from(dataTransfer.files || []);
    if (directFiles.length) {
      return isSupportedAudioFile(directFiles[0]) ? "valid" : "invalid";
    }
    const items = Array.from(dataTransfer.items || []).filter((item) => item?.kind === "file");
    if (!items.length) {
      return "idle";
    }
    const previewFile = items[0]?.getAsFile ? items[0].getAsFile() : null;
    if (!previewFile) {
      return "valid";
    }
    return isSupportedAudioFile(previewFile) ? "valid" : "invalid";
  }, [isSupportedAudioFile]);

  const resolveLocalAudioPath = React.useCallback((file, pathHint = "") => {
    const preferredPath = String(pathHint || "").trim();
    if (preferredPath) {
      return preferredPath;
    }
    const nativePath = String(file?.path || "").trim();
    if (nativePath) {
      return nativePath;
    }
    const relativePath = String(file?.webkitRelativePath || "").trim();
    if (relativePath) {
      return relativePath;
    }
    return String(file?.name || "").trim();
  }, []);

  const handleUpload = React.useCallback(async (file, options = {}) => {
    if (!uploadAvailable) {
      return;
    }
    const validationError = validateAudioFile(file);
    if (validationError) {
      setUploadStatus(validationError);
      setUploadStatusTone("error");
      return;
    }

    setUploadingAudio(true);
    setUploadStatus("Uploading...");
    setUploadStatusTone("info");
    try {
      const localPathHint = resolveLocalAudioPath(file, options.localPathHint);
      await onUploadAudioFile(file, { localPathHint });
      setUploadStatus(`Uploaded: ${localPathHint || String(file.name || "audio")}`);
      setUploadStatusTone("success");
      setDropzoneDragState("idle");
      setShowUploadPanel(false);
    } catch (error) {
      setUploadStatus(String(error?.message || error));
      setUploadStatusTone("error");
    } finally {
      setUploadingAudio(false);
    }
  }, [onUploadAudioFile, resolveLocalAudioPath, uploadAvailable, validateAudioFile]);

  const onPickedFile = React.useCallback((event) => {
    const picked = event.target.files && event.target.files[0] ? event.target.files[0] : null;
    if (!picked) {
      return;
    }
    const localPathHint = String(event?.target?.value || "").trim();
    void handleUpload(picked, { localPathHint });
    event.target.value = "";
  }, [handleUpload]);

  const onDropUpload = React.useCallback((event) => {
    event.preventDefault();
    dragDepthRef.current = 0;
    const dragState = resolveDraggedFileState(event.dataTransfer);
    setDropzoneDragState("idle");
    if (dragState === "invalid") {
      setUploadStatus("Only .wav or .mp3 files are accepted.");
      setUploadStatusTone("error");
      return;
    }
    const dropped = event.dataTransfer?.files && event.dataTransfer.files[0] ? event.dataTransfer.files[0] : null;
    if (!dropped) {
      return;
    }
    void handleUpload(dropped);
  }, [handleUpload, resolveDraggedFileState]);

  const onUploadDragEnter = React.useCallback((event) => {
    if (!uploadAvailable) {
      return;
    }
    event.preventDefault();
    dragDepthRef.current += 1;
    const dragState = resolveDraggedFileState(event.dataTransfer);
    if (dragState !== "idle") {
      setDropzoneDragState(dragState);
    }
  }, [resolveDraggedFileState, uploadAvailable]);

  const onUploadDragOver = React.useCallback((event) => {
    if (!uploadAvailable) {
      return;
    }
    event.preventDefault();
    const dragState = resolveDraggedFileState(event.dataTransfer);
    if (dragState !== "idle") {
      setDropzoneDragState(dragState);
    }
  }, [resolveDraggedFileState, uploadAvailable]);

  const onUploadDragLeave = React.useCallback((event) => {
    if (!uploadAvailable) {
      return;
    }
    event.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setDropzoneDragState("idle");
    }
  }, [uploadAvailable]);

  React.useEffect(() => {
    if (inputMode !== "audio" || showLocalAudioPicker) {
      return;
    }
    setShowUploadPanel(false);
    setUploadStatus("");
    setUploadStatusTone("info");
    setDropzoneDragState("idle");
  }, [inputMode, showLocalAudioPicker]);

  const renderConfigField = React.useCallback((label, description, controlNode) => html`
    <div className="field config-field-row">
      <div className="config-field-left">
        <div className="config-field-label">${label}</div>
        <div className="config-field-desc">${description}</div>
      </div>
      <div className="config-field-right">
        ${controlNode}
      </div>
    </div>
  `, []);

  if (mode === "transcript") {
    return html`
      <section className="controls controls-transcript">
        <div className="field">
          <label>Input Mode</label>
          <select value=${inputMode} onChange=${(e) => onInputModeChange(e.target.value)}>
            <option value="live">Live Transcription</option>
            <option value="audio">Upload Audio File</option>
            <option value="transcript">Text Input</option>
          </select>
        </div>

        <div className="field">
          <label>Status</label>
          <input className="status-readonly" value=${statusText} readOnly />
        </div>

        ${inputMode === "transcript"
          ? html`
              <div className="field span-5">
                <label>TEXT INPUT</label>
                <textarea
                  className="text-input-area"
                  value=${streamText}
                  onChange=${(e) => onStreamTextChange(e.target.value)}
                />
              </div>
            `
          : null}

        ${inputMode === "audio"
          ? html`
              <div className="field span-5 audio-source-field">
                <label>AUDIO SOURCE</label>
                <select
                  className="audio-source-select"
                  value=${selectedAudioSource || localAudioSourceValue}
                  onChange=${(e) => onAudioSourceChange && onAudioSourceChange(e.target.value)}
                >
                  ${audioSampleOptions.map(
                    (item) => html`<option key=${item.value} value=${item.value}>${item.label}</option>`,
                  )}
                  <option value=${localAudioSourceValue}>Local file ...</option>
                </select>
              </div>

              ${showLocalAudioPicker
                ? html`
                    <div className="field span-5">
                      <label>LOCAL AUDIO FILE</label>
                      <div className="audio-path-row local-audio-row upload-popover-anchor">
                        <input
                          value=${audioPathDisplay || ""}
                          placeholder="/absolute/path/to/local.wav"
                          readOnly=${audioPathLocked}
                          title=${audioPathLocked ? "Fixed for local debugging." : ""}
                          onChange=${(e) => onAudioPathChange(e.target.value)}
                        />
                        <button
                          type="button"
                          onClick=${() => {
                            setShowUploadPanel((prev) => !prev);
                            setUploadStatus("");
                            setUploadStatusTone("info");
                          }}
                          disabled=${!uploadAvailable}
                          title=${uploadAvailable ? "Select local wav/mp3 file." : "File selection is unavailable."}
                        >
                          Select File...
                        </button>

                        ${showUploadPanel
                          ? html`
                              <div className="upload-popover" role="dialog" aria-label="Select local audio file">
                                <div
                                  className=${`upload-dropzone ${dropzoneDragState === "valid" ? "drag-valid" : ""} ${dropzoneDragState === "invalid" ? "drag-invalid" : ""}`.trim()}
                                  onDragEnter=${onUploadDragEnter}
                                  onDragOver=${onUploadDragOver}
                                  onDragLeave=${onUploadDragLeave}
                                  onDrop=${onDropUpload}
                                >
                                  <div>Drop .wav / .mp3 file here</div>
                                  ${dropzoneDragState === "valid"
                                    ? html`<div className="meta dropzone-hint valid">Release to upload.</div>`
                                    : null}
                                  ${dropzoneDragState === "invalid"
                                    ? html`<div className="meta dropzone-hint invalid">Only .wav / .mp3 files are supported.</div>`
                                    : null}
                                  <div className="meta">or</div>
                                  <button
                                    type="button"
                                    disabled=${uploadingAudio}
                                    onClick=${() => filePickerRef.current && filePickerRef.current.click()}
                                  >
                                    Choose file
                                  </button>
                                  <input
                                    ref=${filePickerRef}
                                    type="file"
                                    accept=".wav,.mp3,audio/wav,audio/x-wav,audio/mpeg,audio/mp3"
                                    style=${{ display: "none" }}
                                    onChange=${onPickedFile}
                                  />
                                </div>
                                ${uploadStatus
                                  ? html`<div className=${`meta upload-status ${uploadStatusTone}`.trim()}>${uploadStatus}</div>`
                                  : null}
                              </div>
                            `
                          : null}
                      </div>
                    </div>
                  `
                : null}
            `
          : null}

        ${inputMode !== "live"
          ? html`
              <div className="field actions span-5">
                <button
                  type="button"
                  className=${`primary ${isBatchRunning ? "processing-active" : ""}`.trim()}
                  disabled=${isBatchRunning}
                  onClick=${onStart}
                >
                  ${isBatchRunning ? "Processing…" : "Start Session"}
                </button>
                ${isBatchRunning
                  ? html`<button type="button" className="primary" onClick=${onStop}>Stop</button>`
                  : null}
                <button type="button" className="reset-session-btn" onClick=${onReset}>Reset Session</button>
              </div>
            `
          : null}

        ${inputMode === "live"
          ? html`
              <div className="field actions span-5">
                <button
                  type="button"
                  className=${`primary ${isMicRecording ? "recording-active" : ""}`.trim()}
                  disabled=${!micSupported || isMicRecording || micStatus === "stopping"}
                  onClick=${onStartMic}
                >
                  ${isMicRecording ? "Recording…" : "Start Session"}
                </button>
                ${showStopButton
                  ? html`
                      <button
                        type="button"
                        className="primary"
                        disabled=${!micSupported || micStatus === "stopping"}
                        onClick=${onStopMic}
                      >
                        Stop
                      </button>
                    `
                  : null}
                <button type="button" className="reset-session-btn" onClick=${onReset}>Reset Session</button>
              </div>
            `
          : null}
      </section>
    `;
  }

  return html`
    <section className="controls controls-config">
      <div className="config-page-shell">
        <section className="config-stack-section">
          <h3 className="config-section-title">Core Session</h3>
          ${renderConfigField(
            "Backend URL",
            "API endpoint used by the frontend for all requests.",
            html`<input value=${baseUrl} onChange=${(e) => onBaseUrlChange(e.target.value)} />`,
          )}
          ${renderConfigField(
            "Session ID",
            "Unique identifier for transcript, timeline, and note state.",
            html`<input value=${sessionId} onChange=${(e) => onSessionIdChange(e.target.value)} />`,
          )}
          ${renderConfigField(
            "Refresh Tick (ms)",
            "Polling interval for non-live processing updates.",
            html`
              <input
                type="number"
                min="300"
                value=${String(intervalMs)}
                onChange=${(e) => onIntervalChange(e.target.value)}
              />
            `,
          )}
        </section>

        <section className="config-stack-section">
          <h3 className="config-section-title">Processing Engine</h3>
          ${renderConfigField(
            "Event Engine",
            "Extraction strategy used to generate timeline events.",
            html`
              <select value=${eventEngine} onChange=${(e) => onEngineChange(e.target.value)}>
                <option value="auto">MedGemma and rule fallback</option>
                <option value="medgemma">MedGemma</option>
              </select>
            `,
          )}
          ${renderConfigField(
            "Audio Window (s)",
            "Chunk duration for audio_path processing. Larger windows increase context but add latency.",
            html`
              <div className="config-slider-wrap">
                <input
                  className="config-slider"
                  type="range"
                  min="1"
                  max="120"
                  step="1"
                  value=${String(audioWindowSec)}
                  onChange=${(e) => onAudioWindowSecChange(e.target.value)}
                />
                <span className="config-slider-value">${Number(audioWindowSec).toFixed(0)}s</span>
              </div>
            `,
          )}
        </section>

        <section className="config-stack-section">
          <h3 className="config-section-title">Live Streaming</h3>
          ${renderConfigField(
            "Live ASR Window (s)",
            "Rolling ASR context size for live microphone sessions.",
            html`
              <div className="config-slider-wrap">
                <input
                  className="config-slider"
                  type="range"
                  min="1"
                  max="120"
                  step="0.5"
                  value=${String(micAsrWindowSec)}
                  onChange=${(e) => onMicAsrWindowSecChange(e.target.value)}
                />
                <span className="config-slider-value">${Number(micAsrWindowSec).toFixed(1)}s</span>
              </div>
            `,
          )}
          ${renderConfigField(
            "Live Push Step (s)",
            "How often incremental live audio chunks are pushed for processing.",
            html`
              <input
                type="number"
                min="0.5"
                max="120"
                step="0.5"
                value=${String(micAsrStepSec)}
                onChange=${(e) => onMicAsrStepSecChange(e.target.value)}
              />
            `,
          )}
          ${renderConfigField(
            "Live Mic Slice (ms)",
            "Recorder chunk size produced by the browser before websocket upload.",
            html`
              <input
                type="number"
                min="200"
                max="10000"
                value=${String(micCaptureSliceMs)}
                onChange=${(e) => onMicCaptureSliceMsChange(e.target.value)}
              />
            `,
          )}
        </section>

        <section className="config-stack-section">
          <h3 className="config-section-title">Clinical Reasoning</h3>
          ${showOpenQuestionsAiSetting
            ? renderConfigField(
                "Enable AI Follow-up Questions",
                "When enabled, model-generated contextual follow-up questions are added after rule-based safety questions.",
                html`
                  <label className="config-toggle-row">
                    <input
                      type="checkbox"
                      checked=${openQuestionsAiEnhancementEnabled}
                      onChange=${(e) => onOpenQuestionsAiEnhancementChange(Boolean(e.target.checked))}
                    />
                    <span className="config-toggle-switch" aria-hidden="true"></span>
                    <span className="config-toggle-text">
                      ${openQuestionsAiEnhancementEnabled ? "Enabled" : "Disabled"}
                    </span>
                  </label>
                `,
              )
            : renderConfigField(
                "Enable AI Follow-up Questions",
                "Managed by backend defaults in this environment.",
                html`<div className="config-static-muted">Managed by backend defaults in this environment.</div>`,
              )}
        </section>

        <details className="config-advanced-accordion">
          <summary className="config-section-title">Advanced Tuning</summary>
          <div className="config-advanced-body">
            ${renderConfigField(
              "Reconcile Lookback (windows)",
              "Reprocess previous N windows to stabilize cross-window event continuity.",
              html`
                <input
                  type="number"
                  min="0"
                  max="20"
                  value=${String(reconcileLookbackWindows)}
                  onChange=${(e) => onReconcileLookbackChange(e.target.value)}
                />
              `,
            )}
            ${renderConfigField(
              "LLM Update Every (windows)",
              "MedGemma refresh cadence in auto mode. Lower is more responsive, higher is cheaper.",
              html`
                <input
                  type="number"
                  min="1"
                  max="20"
                  value=${String(llmUpdateIntervalWindows)}
                  onChange=${(e) => onLlmUpdateIntervalChange(e.target.value)}
                />
              `,
            )}
          </div>
        </details>
      </div>
    </section>
  `;
}
