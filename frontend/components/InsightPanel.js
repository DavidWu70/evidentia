import React from "https://esm.sh/react@18";
import htm from "https://esm.sh/htm@3";

const html = htm.bind(React.createElement);
const INSIGHT_SPLITTER_HEIGHT_PX = 12;
const INSIGHT_PANEL_MIN_HEIGHTS = [36, 36, 36];

/**
 * Render risk flags, open questions, and editable draft note.
 *
 * Design intent:
 * - Keep safety-oriented interpretation and note editing in one reviewer surface.
 * - Preserve citations count so draft trust can be inspected at a glance.
 */
export function InsightPanel({
  mode = "transcript",
  headerStatus = "",
  riskFlags,
  riskEmptyStateMessage = "No risk flag yet.",
  riskFlagTimelineEvidence = {},
  onSelectEvidenceEvent = () => {},
  openQuestions,
  mandatoryQuestions = [],
  contextualFollowups = [],
  openQuestionsRationale = "",
  noteText,
  citations,
  draftOptions = [],
  activeDraftTemplateId,
  onDraftTemplateChange,
  noteDepartment = "",
  onNoteDepartmentChange = () => {},
  noteTemplateCatalog = {},
  noteTemplateIds,
  noteTemplateOptions = [],
  noteTemplateStatuses = {},
  noteGenerationStatus = "idle",
  onNoteTemplatesChange,
  onGenerateDrafts,
  onStopGenerateDrafts = () => {},
  pipelineDebug,
  onNoteChange,
  onUndoNote = () => {},
  onRedoNote = () => {},
  canUndoNote = false,
  canRedoNote = false,
  onNoteKeyDown = () => {},
  onExport,
  onEmail = () => {},
  lastError,
}) {
  const [insightBlockHeights, setInsightBlockHeights] = React.useState([35, 45, 20]);
  const [activeInsightSplitter, setActiveInsightSplitter] = React.useState(-1);
  const [expandedDepartments, setExpandedDepartments] = React.useState({});
  const insightLayoutRef = React.useRef(null);
  const insightDragRef = React.useRef(null);

  React.useEffect(() => {
    const currentDepartment = String(noteDepartment || "").trim();
    if (!currentDepartment) {
      return;
    }
    setExpandedDepartments((prev) => {
      if (prev[currentDepartment]) {
        return prev;
      }
      return { ...prev, [currentDepartment]: true };
    });
  }, [noteDepartment]);

  const stopInsightResize = React.useCallback(() => {
    insightDragRef.current = null;
    setActiveInsightSplitter(-1);
    document.body.classList.remove("is-row-resizing");
    window.removeEventListener("pointermove", onInsightResizeMove);
    window.removeEventListener("pointerup", stopInsightResize);
    window.removeEventListener("pointercancel", stopInsightResize);
  }, []);

  const onInsightResizeMove = React.useCallback((event) => {
    const drag = insightDragRef.current;
    if (!drag) {
      return;
    }

    const deltaY = event.clientY - drag.startY;
    const next = [...drag.startHeights];
    const topIdx = drag.splitterIndex;
    const bottomIdx = drag.splitterIndex + 1;
    const totalPairPx = drag.topStartPx + drag.bottomStartPx;

    const topMinPx = INSIGHT_PANEL_MIN_HEIGHTS[topIdx] || 24;
    const bottomMinPx = INSIGHT_PANEL_MIN_HEIGHTS[bottomIdx] || 24;
    const topLowerBound = topMinPx;
    const topUpperBound = Math.max(topLowerBound, totalPairPx - bottomMinPx);
    const topPx = clamp(drag.topStartPx + deltaY, topLowerBound, topUpperBound);
    const bottomPx = totalPairPx - topPx;

    next[topIdx] = (topPx / drag.totalBlockHeightPx) * 100;
    next[bottomIdx] = (bottomPx / drag.totalBlockHeightPx) * 100;
    setInsightBlockHeights(next);
  }, []);

  const startInsightResize = React.useCallback((splitterIndex, clientY) => {
    if (window.matchMedia("(max-width: 1100px)").matches) {
      return;
    }
    const host = insightLayoutRef.current;
    if (!host) {
      return;
    }
    const rect = host.getBoundingClientRect();
    const totalSplitterHeight = INSIGHT_SPLITTER_HEIGHT_PX * 2;
    const totalBlockHeightPx = Math.max(1, rect.height - totalSplitterHeight);
    const startHeights = insightBlockHeights.slice(0, 3);
    const topStartPx = (startHeights[splitterIndex] / 100) * totalBlockHeightPx;
    const bottomStartPx = (startHeights[splitterIndex + 1] / 100) * totalBlockHeightPx;

    insightDragRef.current = {
      splitterIndex,
      startY: clientY,
      startHeights,
      totalBlockHeightPx,
      topStartPx,
      bottomStartPx,
    };
    setActiveInsightSplitter(splitterIndex);
    document.body.classList.add("is-row-resizing");
    window.addEventListener("pointermove", onInsightResizeMove);
    window.addEventListener("pointerup", stopInsightResize);
    window.addEventListener("pointercancel", stopInsightResize);
  }, [insightBlockHeights, onInsightResizeMove, stopInsightResize]);

  React.useEffect(() => () => {
    document.body.classList.remove("is-row-resizing");
    window.removeEventListener("pointermove", onInsightResizeMove);
    window.removeEventListener("pointerup", stopInsightResize);
    window.removeEventListener("pointercancel", stopInsightResize);
  }, [onInsightResizeMove, stopInsightResize]);

  if (mode === "notes") {
    const selectedTemplateSet = new Set((noteTemplateIds || []).map((item) => String(item || "").trim()));
    const availableDepartments = Object.entries(noteTemplateCatalog || {}).sort(([leftId], [rightId]) =>
      compareNoteDepartments(leftId, rightId),
    );
    const activeGenerationStatus = String(noteGenerationStatus || "idle").toLowerCase();
    const isGenerating = ["pending", "generating", "stopping"].includes(activeGenerationStatus);
    const isStopping = activeGenerationStatus === "stopping";
    const generationButtonLabel = isStopping
      ? "Stopping..."
      : isGenerating
        ? "Generating..."
        : "Generate Notes";
    const stopButtonLabel = isStopping ? "Stopping..." : "Stop";
    const selectedDraftValue =
      activeDraftTemplateId ||
      String(draftOptions?.[0]?.template_id || "");

    return html`
      <section className="panel notes-root" id="insight-panel">
        <div className="notes-layout">
          <div className="card notes-template-panel">
            <strong className="notes-section-title">Select Note Templates</strong>
            <div className="notes-template-tree">
              ${availableDepartments.length
                ? availableDepartments.map(([departmentId, templates]) => {
                    const normalizedDepartmentId = String(departmentId || "").trim();
                    const isExpanded = Boolean(expandedDepartments[normalizedDepartmentId]);
                    const isActiveDepartment = normalizedDepartmentId === String(noteDepartment || "");
                    return html`
                      <div key=${normalizedDepartmentId} className="notes-department-group">
                        <button
                          type="button"
                          className=${`notes-department-toggle ${isActiveDepartment ? "active" : ""}`}
                          onClick=${() => {
                            setExpandedDepartments((prev) => ({
                              ...prev,
                              [normalizedDepartmentId]: !isExpanded,
                            }));
                            if (!isActiveDepartment) {
                              onNoteDepartmentChange(normalizedDepartmentId);
                            }
                          }}
                        >
                          <span>${isExpanded ? "▾" : "▸"} ${formatDepartmentLabel(normalizedDepartmentId)}</span>
                          ${isActiveDepartment ? html`<span className="notes-department-current">Current</span>` : null}
                        </button>

                        ${isExpanded
                          ? html`
                              <div className="notes-template-list">
                                ${(Array.isArray(templates)
                                  ? [...templates].sort((left, right) =>
                                      compareTemplatesWithinDepartment(
                                        normalizedDepartmentId,
                                        left,
                                        right,
                                      ),
                                    )
                                  : []).map((item) => {
                                  const templateId = String(item?.id || "").trim();
                                  const isSelected = selectedTemplateSet.has(templateId);
                                  const generatedDraft = draftOptions.find(
                                    (draft) => String(draft?.template_id || "") === templateId,
                                  );
                                  const statusCode = generatedDraft
                                    ? "generated"
                                    : String(noteTemplateStatuses?.[templateId] || "not_generated");
                                  return html`
                                    <div key=${templateId} className="notes-template-row">
                                      <label className="notes-template-select">
                                        <input
                                          type="checkbox"
                                          checked=${isSelected}
                                          disabled=${!isActiveDepartment || isGenerating}
                                          onChange=${(e) => {
                                            const nextSet = new Set(
                                              isActiveDepartment
                                                ? (noteTemplateIds || []).map((value) => String(value || ""))
                                                : [],
                                            );
                                            if (e.target.checked) {
                                              nextSet.add(templateId);
                                            } else {
                                              nextSet.delete(templateId);
                                            }
                                            onNoteTemplatesChange(Array.from(nextSet));
                                          }}
                                        />
                                        ${generatedDraft
                                          ? html`
                                              <button
                                                type="button"
                                                className="notes-template-link"
                                                onClick=${(event) => {
                                                  event.preventDefault();
                                                  event.stopPropagation();
                                                  onDraftTemplateChange(templateId);
                                                }}
                                              >
                                                ${String(item?.label || templateId)}
                                              </button>
                                            `
                                          : html`<span>${String(item?.label || templateId)}</span>`}
                                      </label>
                                      ${shouldShowTemplateStatus(statusCode)
                                        ? html`
                                            <span className=${`notes-template-status status-${statusCode}`}>
                                              ${formatTemplateStatus(statusCode)}
                                            </span>
                                          `
                                        : null}
                                    </div>
                                  `;
                                })}
                              </div>
                            `
                          : null}
                      </div>
                    `;
                  })
                : html`<div className="meta">No templates available.</div>`}
            </div>

            <div className="row notes-action-row">
              <button
                type="button"
                className="primary"
                disabled=${isGenerating}
                onClick=${onGenerateDrafts}
              >
                ${generationButtonLabel}
              </button>
              ${isGenerating
                ? html`
                    <button
                      type="button"
                      className="primary"
                      disabled=${isStopping}
                      onClick=${onStopGenerateDrafts}
                    >
                      ${stopButtonLabel}
                    </button>
                  `
                : null}
              <span className="status">${formatNoteGenerationStatus(activeGenerationStatus)}</span>
            </div>
          </div>

          <div className="card notes-draft-panel">
            <div className="row notes-draft-toolbar">
              <div className="notes-draft-heading">
                <strong className="notes-section-title notes-section-title-inline">Draft Note</strong>
                <span className="status">${citations.length} citations</span>
              </div>
              ${draftOptions.length
                ? html`
                    <select
                      value=${selectedDraftValue}
                      onChange=${(e) => onDraftTemplateChange(e.target.value)}
                    >
                      ${draftOptions.map(
                        (item) => html`
                          <option key=${item.template_id} value=${item.template_id}>
                            ${item.template_name || item.template_id}
                          </option>
                        `,
                      )}
                    </select>
                  `
                : null}
              <div className="notes-toolbar-actions">
                <button
                  type="button"
                  className="notes-tool-btn"
                  title="Undo (Ctrl/Cmd+Z)"
                  aria-label="Undo"
                  disabled=${!canUndoNote}
                  onClick=${onUndoNote}
                >
                  <svg
                    className="notes-tool-icon"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.9"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="m9 14-5-5 5-5" />
                    <path d="M4 9h8a6 6 0 0 1 0 12h-1" />
                  </svg>
                </button>
                <button
                  type="button"
                  className="notes-tool-btn"
                  title="Redo (Ctrl/Cmd+Shift+Z)"
                  aria-label="Redo"
                  disabled=${!canRedoNote}
                  onClick=${onRedoNote}
                >
                  <svg
                    className="notes-tool-icon"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.9"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="m15 14 5-5-5-5" />
                    <path d="M20 9h-8a6 6 0 0 0 0 12h1" />
                  </svg>
                </button>
                <button
                  type="button"
                  className="notes-export-btn"
                  title="Download"
                  aria-label="Download"
                  onClick=${() => onExport(noteText)}
                >
                  <svg
                    className="notes-export-icon"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.9"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M12 3v11" />
                    <path d="m8 10 4 4 4-4" />
                    <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
                  </svg>
                </button>
                <button
                  type="button"
                  className="notes-export-btn"
                  title="Email"
                  aria-label="Email"
                  onClick=${() => onEmail(noteText)}
                >
                  <svg
                    className="notes-export-icon"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.9"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <rect x="3" y="5" width="18" height="14" rx="2" ry="2" />
                    <path d="m4 7 8 6 8-6" />
                  </svg>
                </button>
              </div>
            </div>

            <textarea
              className="note"
              value=${noteText}
              placeholder="Draft note will appear here..."
              onKeyDown=${onNoteKeyDown}
              onInput=${(e) => onNoteChange(e.target.value)}
            />
          </div>
        </div>

        ${lastError
          ? html`
              <div className="card error-card">
                <strong>Notice</strong>
                <div className="meta">${lastError}</div>
              </div>
            `
          : null}
      </section>
    `;
  }

  const debug = pipelineDebug || {};

  return html`
    <section className="panel insight-panel-transcript" id="insight-panel">
      <div className="panel-title-row">
        <h2>Clinical Assistant</h2>
        ${headerStatus ? html`<span className="panel-title-status">${headerStatus}</span>` : null}
      </div>

      <div
        className=${`insight-resizable-stack ${activeInsightSplitter >= 0 ? "is-resizing" : ""}`}
        ref=${insightLayoutRef}
        style=${{
          gridTemplateRows:
            `minmax(${INSIGHT_PANEL_MIN_HEIGHTS[0]}px, ${insightBlockHeights[0]}%) ` +
            `${INSIGHT_SPLITTER_HEIGHT_PX}px ` +
            `minmax(${INSIGHT_PANEL_MIN_HEIGHTS[1]}px, ${insightBlockHeights[1]}%) ` +
            `${INSIGHT_SPLITTER_HEIGHT_PX}px ` +
            `minmax(${INSIGHT_PANEL_MIN_HEIGHTS[2]}px, ${insightBlockHeights[2]}%)`,
        }}
      >
        <div className="card insight-block insight-risk">
          <strong>Risk Assessment</strong>
          ${riskFlags.length
            ? riskFlags.map(
                (flag, index) => {
                  const riskFlagKey = buildRiskFlagStoreKey(flag, index);
                  const linkedEvidence = Array.isArray(riskFlagTimelineEvidence?.[riskFlagKey])
                    ? riskFlagTimelineEvidence[riskFlagKey]
                    : [];
                  return html`
                  <div key=${`${flag.flag}_${flag.level}_${index}`} style=${{ marginTop: "0.45rem" }}>
                    <span className=${`badge ${flag.level}`}>${flag.level}</span>
                    <strong>${formatClinicalDisplayLabel(flag.flag, "risk_flag")}</strong>
                    <div className="meta">${flag.why}</div>
                    ${linkedEvidence.length
                      ? html`
                          <div className="assistant-evidence-list">
                            ${linkedEvidence.map(
                              (evidenceItem) => html`
                                <button
                                  key=${evidenceItem.eventKey}
                                  type="button"
                                  className="assistant-evidence-link"
                                  onClick=${() =>
                                    onSelectEvidenceEvent({
                                      eventKey: evidenceItem.eventKey,
                                      segmentId: evidenceItem.segmentId,
                                    })}
                                  title="Jump to Evidence Timeline"
                                >
                                  <span className="assistant-evidence-title">
                                    ${formatClinicalDisplayLabel(evidenceItem.label, "event_label")}
                                  </span>
                                  <span className="meta">
                                    ${formatWindow(evidenceItem.t0, evidenceItem.t1)} | "${truncateQuote(evidenceItem.quote)}"
                                  </span>
                                </button>
                              `,
                            )}
                          </div>
                        `
                      : null}
                  </div>
                `;
                },
              )
            : html`<div className="meta">${riskEmptyStateMessage}</div>`}
        </div>

        <button
          type="button"
          className=${`pane-splitter pane-splitter-row ${activeInsightSplitter === 0 ? "active" : ""}`}
          aria-label="Resize risk assessment and follow-up questions"
          onPointerDown=${(e) => {
            e.preventDefault();
            startInsightResize(0, e.clientY);
          }}
        >
          <span className="pane-splitter-grip">::</span>
        </button>

        <div className="card insight-block insight-followups">
          <strong>Follow-up Questions</strong>
          ${mandatoryQuestions.length
            ? html`
                <div className="meta" style=${{ marginTop: "0.35rem" }}>
                  mandatory_safety_questions:
                </div>
                ${mandatoryQuestions.map((q, idx) => html`<div key=${`m_${idx}`} className="meta">- ${q}</div>`)}
              `
            : null}
          ${contextualFollowups.length
            ? html`
                <div className="meta" style=${{ marginTop: "0.45rem" }}>
                  contextual_followups:
                </div>
                ${contextualFollowups.map((q, idx) => html`<div key=${`c_${idx}`} className="meta">- ${q}</div>`)}
              `
            : null}
          ${!mandatoryQuestions.length && !contextualFollowups.length
            ? openQuestions.length
              ? openQuestions.map((q, idx) => html`<div key=${idx} className="meta">- ${q}</div>`)
              : html`<div className="meta">No follow-up question yet.</div>`
            : null}
          ${openQuestionsRationale
            ? html`<div className="meta" style=${{ marginTop: "0.5rem" }}>rationale: ${openQuestionsRationale}</div>`
            : null}
        </div>

        <button
          type="button"
          className=${`pane-splitter pane-splitter-row ${activeInsightSplitter === 1 ? "active" : ""}`}
          aria-label="Resize follow-up questions and technical details"
          onPointerDown=${(e) => {
            e.preventDefault();
            startInsightResize(1, e.clientY);
          }}
        >
          <span className="pane-splitter-grip">::</span>
        </button>

        <div className="card insight-block insight-technical">
          <strong>Technical Details</strong>
          <div className="meta">Source: ${debug.source || "n/a"}</div>
          <div className="meta">ASR: ${debug.asr_status || "n/a"}</div>
          <div className="meta">Diarization: ${debug.diarization_status || "n/a"}</div>
          <div className="meta">Diarization Reason: ${debug.diarization_reason || "n/a"}</div>
          <div className="meta">Alignment Reason Counts: ${debug.alignment_reason_counts || "n/a"}</div>
          <div className="meta">Alignment Fallback Rate: ${debug.alignment_fallback_rate || "n/a"}</div>
          <div className="meta">Sentence Role Split: ${debug.sentence_role_split || "n/a"}</div>
          <div className="meta">Event Guardrails: ${debug.event_guardrails || "n/a"}</div>
          <div className="meta">Risk Backstop: ${debug.risk_backstop || "n/a"}</div>
          <div className="meta">Event Harmonization: ${debug.event_harmonization || "n/a"}</div>
          <div className="meta">Event Engine Used: ${debug.event_engine_used || "n/a"}</div>
          ${debug.medgemma_status ? html`<div className="meta">MedGemma: ${debug.medgemma_status}</div>` : null}
          ${debug.medgemma_error ? html`<div className="meta">MedGemma Error: ${debug.medgemma_error}</div>` : null}
          <div className="meta">MedGemma Calls: ${debug.medgemma_calls || 0}</div>
          <div className="meta">MedGemma JSON Valid Rate: ${debug.medgemma_json_valid_rate || "n/a"}</div>
          <div className="meta">MedGemma Fallback Rate: ${debug.medgemma_fallback_rate || "n/a"}</div>
          <div className="meta">MedGemma Filter Latency: ${debug.medgemma_filter_p50_p95_ms || "n/a"}</div>
          <div className="meta">MedGemma Extract Latency: ${debug.medgemma_extract_p50_p95_ms || "n/a"}</div>
        <div className="meta">Open Questions Mode: ${debug.open_questions_mode || "n/a"}</div>
        <div className="meta">Open Questions AI Error: ${debug.open_questions_ai_error || "n/a"}</div>
        <div className="meta">Note Department: ${debug.note_department || "n/a"}</div>
        <div className="meta">Note Templates: ${debug.note_templates || "n/a"}</div>
        <div className="meta">Note Templates Generated: ${debug.note_templates_generated || "n/a"}</div>
        <div className="meta">Live Audio Status: ${debug.live_audio_status || "n/a"}</div>
        <div className="meta">Live Audio Chunks: ${debug.live_audio_chunks || "n/a"}</div>
        <div className="meta">Live Audio Bytes: ${debug.live_audio_bytes || "n/a"}</div>
        <div className="meta">Live Audio Recording Path: ${debug.live_audio_recording_path || "n/a"}</div>
        <div className="meta">Live Audio Archive End Sec: ${debug.live_audio_archive_end_sec || "n/a"}</div>
        ${lastError ? html`<div className="meta">Error: ${lastError}</div>` : null}
      </div>
      </div>
    </section>
  `;
}

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function buildRiskFlagStoreKey(riskFlag, index) {
  return [
    String(riskFlag?.level || ""),
    String(riskFlag?.flag || ""),
    String(index),
  ].join("|");
}

function formatWindow(t0, t1) {
  const start = Number(t0) || 0;
  const end = Number(t1) || 0;
  return `${start.toFixed(1)}-${end.toFixed(1)}s`;
}

function truncateQuote(raw, maxLen = 88) {
  const text = String(raw || "").replace(/\s+/g, " ").trim();
  if (text.length <= maxLen) {
    return text;
  }
  return `${text.slice(0, Math.max(1, maxLen - 1)).trimEnd()}…`;
}

function formatClinicalDisplayLabel(raw, kind = "") {
  const code = String(raw || "").trim().toLowerCase();
  const mapped = CLINICAL_LABEL_MAP[code];
  if (mapped) {
    return mapped;
  }
  if (kind === "risk_flag" || kind === "event_label") {
    return code
      .split("_")
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }
  return String(raw || "");
}

const CLINICAL_LABEL_MAP = {
  passive_or_active_si_detected: "Suicidal Thoughts Detected",
  passive_suicidal_ideation: "Passive Suicidal Thoughts",
  suicidal_ideation: "Suicidal Thoughts",
  urgent_suicide_risk: "Urgent Suicide Risk",
  si_explicitly_denied: "Suicidal Thoughts Denied",
  suicidal_plan_or_intent: "Suicide Plan or Intent",
};

function formatTemplateStatus(status) {
  const normalized = String(status || "").trim().toLowerCase();
  const mapping = {
    not_generated: "Not Generated",
    queued: "Queued",
    pending: "Queued",
    generating: "Generating...",
    stopping: "Stopping...",
    stopped: "Stopped",
    stopped_partial: "Stopped",
    generated: "Generated ✓",
    missing: "Unavailable",
    failed: "Failed",
  };
  return mapping[normalized] || "Not Generated";
}

function shouldShowTemplateStatus(status) {
  const normalized = String(status || "").trim().toLowerCase();
  return normalized !== "not_generated";
}

function formatNoteGenerationStatus(status) {
  const normalized = String(status || "").trim().toLowerCase();
  const mapping = {
    idle: "Idle",
    pending: "Preparing...",
    generating: "Generating...",
    stopping: "Stopping...",
    completed: "Completed",
    stopped_partial: "Stopped",
    failed: "Failed",
  };
  return `Status: ${mapping[normalized] || "Idle"}`;
}

function formatDepartmentLabel(value) {
  const normalized = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/-/g, "_")
    .replace(/\s+/g, "_");
  const known = {
    psych: "Psychiatry",
    internal_med: "Internal Medicine",
    family_med: "Family Medicine",
    general_med: "General Medicine",
    primary_care: "Primary Care",
    emergency_med: "Emergency Medicine",
    pediatrics: "Pediatrics",
    ob_gyn: "Ob/Gyn",
    cardiology: "Cardiology",
    neurology: "Neurology",
  };
  if (known[normalized]) {
    return known[normalized];
  }
  return normalized
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function compareNoteDepartments(left, right) {
  const priority = {
    psych: 0,
    internal_med: 1,
  };
  const leftKey = String(left || "").trim().toLowerCase();
  const rightKey = String(right || "").trim().toLowerCase();
  const leftRank = Object.prototype.hasOwnProperty.call(priority, leftKey) ? priority[leftKey] : 99;
  const rightRank = Object.prototype.hasOwnProperty.call(priority, rightKey) ? priority[rightKey] : 99;
  if (leftRank !== rightRank) {
    return leftRank - rightRank;
  }
  return formatDepartmentLabel(leftKey).localeCompare(formatDepartmentLabel(rightKey));
}

function compareTemplatesWithinDepartment(departmentId, left, right) {
  const department = String(departmentId || "").trim().toLowerCase();
  if (department === "internal_med") {
    const leftId = String(left?.id || "").trim().toLowerCase();
    const rightId = String(right?.id || "").trim().toLowerCase();
    if (leftId === "internal_soap" && rightId !== "internal_soap") {
      return -1;
    }
    if (rightId === "internal_soap" && leftId !== "internal_soap") {
      return 1;
    }
  }
  return 0;
}
