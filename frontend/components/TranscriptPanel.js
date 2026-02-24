import React from "https://esm.sh/react@18";
import htm from "https://esm.sh/htm@3";

const html = htm.bind(React.createElement);

/**
 * Render transcript cards with stable segment anchors.
 *
 * Design intent:
 * - Keep evidence review auditable via visible segment IDs and timing windows.
 * - Support timeline-to-transcript focus by attaching deterministic DOM anchors.
 */
export function TranscriptPanel({
  rows,
  selectedSegmentId,
  headerStatus = "",
  canPlayAudio = false,
  playingClipKey = "",
  editLocked = false,
  onToggleClip = () => {},
  onSaveEdit = async () => {},
  onDeleteRow = () => {},
}) {
  const [editingSegmentId, setEditingSegmentId] = React.useState("");
  const [draftSpeakerRole, setDraftSpeakerRole] = React.useState("other");
  const [draftText, setDraftText] = React.useState("");
  const [savingEdit, setSavingEdit] = React.useState(false);

  React.useEffect(() => {
    if (!editingSegmentId) {
      return;
    }
    const exists = rows.some((item) => String(item?.segment_id || "") === editingSegmentId);
    if (!exists) {
      setEditingSegmentId("");
      setDraftSpeakerRole("other");
      setDraftText("");
      setSavingEdit(false);
    }
  }, [editingSegmentId, rows]);

  const startEditing = React.useCallback((item) => {
    setEditingSegmentId(String(item?.segment_id || ""));
    setDraftSpeakerRole(normalizeEditableSpeakerRole(item?.speaker_role || item?.speaker));
    setDraftText(String(item?.text || ""));
  }, []);

  const cancelEditing = React.useCallback(() => {
    if (savingEdit) {
      return;
    }
    setEditingSegmentId("");
    setDraftSpeakerRole("other");
    setDraftText("");
  }, [savingEdit]);

  const saveEditing = React.useCallback(async () => {
    const segmentId = String(editingSegmentId || "").trim();
    const text = String(draftText || "").trim();
    if (!segmentId || !text || savingEdit) {
      return;
    }
    setSavingEdit(true);
    try {
      await onSaveEdit({
        segmentId,
        speakerRole: normalizeEditableSpeakerRole(draftSpeakerRole),
        text,
      });
      setEditingSegmentId("");
      setDraftSpeakerRole("other");
      setDraftText("");
    } finally {
      setSavingEdit(false);
    }
  }, [draftSpeakerRole, draftText, editingSegmentId, onSaveEdit, savingEdit]);

  return html`
    <section className="panel" id="transcript-panel">
      <div className="panel-title-row">
        <h2>Live Transcript</h2>
        ${headerStatus ? html`<span className="panel-title-status">${headerStatus}</span>` : null}
      </div>
      <div className="list">
        ${rows.length
          ? rows.map((item) => {
              const clipKey = `transcript|${item.segment_id}`;
              const isPlaying = clipKey === playingClipKey;
              const isEditing = String(item?.segment_id || "") === editingSegmentId;
              const classes = ["card"];
              if (item.segment_id === selectedSegmentId) {
                classes.push("selected");
              }
              if (isPlaying) {
                classes.push("playing-clip");
              }
              if (isEditing) {
                classes.push("editing");
              }
              return html`
                <div key=${item.segment_id} className=${classes.join(" ")} data-segment-id=${item.segment_id}>
                  <div className="card-head">
                    <div>
                      <span className="badge">${item.speaker || "other"}</span>
                      <span className="meta">
                        ${formatSpeakerDetails(item)}
                      </span>
                    </div>
                    <div className="transcript-card-actions">
                      <button
                        type="button"
                        className=${`transcript-edit-toggle ${isEditing ? "editing" : ""}`}
                        aria-label=${isEditing ? "Editing transcript row" : "Edit transcript row"}
                        title=${isEditing ? "Editing..." : "Edit"}
                        disabled=${editLocked || savingEdit || isEditing}
                        onClick=${() => startEditing(item)}
                      >
                        <svg
                          className="transcript-edit-icon"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          aria-hidden="true"
                        >
                          <path d="M12 20h9" />
                          <path d="m16.5 3.5 4 4L8 20l-5 1 1-5 12.5-12.5z" />
                        </svg>
                      </button>
                      ${canPlayAudio
                        ? html`
                            <button
                              type="button"
                              className=${`clip-toggle ${isPlaying ? "playing" : ""}`}
                              aria-label=${isPlaying ? "Stop clip" : "Play clip"}
                              title=${isPlaying ? "Stop clip" : "Play clip"}
                              onClick=${() =>
                                onToggleClip({
                                  clipKey,
                                  t0: Number(item.start || 0),
                                  t1: Number(item.end || 0),
                                })}
                            >
                              ${isPlaying
                                ? html`
                                    <svg
                                      className="clip-toggle-icon"
                                      viewBox="0 0 24 24"
                                      fill="currentColor"
                                      aria-hidden="true"
                                    >
                                      <rect x="8" y="8" width="8" height="8" rx="1.2" ry="1.2" />
                                    </svg>
                                  `
                                : html`
                                    <svg
                                      className="clip-toggle-icon"
                                      viewBox="0 0 24 24"
                                      fill="currentColor"
                                      aria-hidden="true"
                                    >
                                      <path d="M9 7.6 17 12l-8 4.4z" />
                                    </svg>
                                  `}
                            </button>
                          `
                        : null}
                    </div>
                  </div>
                  ${isEditing
                    ? html`
                        <div className="transcript-edit-form">
                          <div className="transcript-edit-row">
                            <label>Role</label>
                            <select
                              value=${draftSpeakerRole}
                              onChange=${(event) => setDraftSpeakerRole(event.target.value)}
                              disabled=${savingEdit || editLocked}
                            >
                              <option value="patient">Patient</option>
                              <option value="clinician">Clinician</option>
                              <option value="other">Other</option>
                            </select>
                          </div>
                          <textarea
                            className="transcript-edit-textarea"
                            value=${draftText}
                            onInput=${(event) => setDraftText(event.target.value)}
                            disabled=${savingEdit || editLocked}
                          />
                          <div className="row transcript-edit-actions">
                            <button
                              type="button"
                              className="primary"
                              disabled=${savingEdit || editLocked || !String(draftText || "").trim()}
                              onClick=${saveEditing}
                            >
                              ${savingEdit ? "Saving..." : "Save"}
                            </button>
                            <button
                              type="button"
                              disabled=${savingEdit}
                              onClick=${cancelEditing}
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      `
                    : html`<div>${item.text || ""}</div>`}
                  <div className="transcript-card-footer">
                    <div className="meta">
                      ${`${formatTime(item.start)} - ${formatTime(item.end)} | `}
                      <span title=${String(item.segment_id || "")}>
                        ${formatSegmentRef(item.segment_id)}
                      </span>
                    </div>
                    <button
                      type="button"
                      className="transcript-delete-toggle"
                      aria-label="Delete transcript row"
                      title="Delete row"
                      disabled=${editLocked || savingEdit}
                      onClick=${() => onDeleteRow({ segmentId: item.segment_id })}
                    >
                      <svg
                        className="transcript-delete-icon"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.1"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <path d="M8 8l8 8" />
                        <path d="M16 8l-8 8" />
                      </svg>
                    </button>
                  </div>
                </div>
              `;
            })
          : html`
              <div className="card">
                <div className="meta panel-empty-message">Waiting for transcript updates...</div>
              </div>
            `}
      </div>
    </section>
  `;
}

function normalizeEditableSpeakerRole(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "patient" || normalized === "clinician") {
    return normalized;
  }
  return "other";
}

function formatTime(value) {
  const num = Number(value) || 0;
  return `${num.toFixed(1)}s`;
}

function formatSpeakerDetails(item) {
  const speakerId = String(item?.speaker_id || "").trim();
  const speakerRole = String(item?.speaker_role || item?.speaker || "other").trim();
  const conf = Number(item?.diar_confidence);
  const confText = Number.isFinite(conf) ? ` ${Math.max(0, Math.min(1, conf)).toFixed(2)}` : "";
  if (speakerId) {
    return `${speakerId} / ${speakerRole}${confText}`;
  }
  return `${speakerRole}${confText}`.trim();
}

function formatSegmentRef(segmentId) {
  const raw = String(segmentId || "").trim();
  if (!raw) {
    return "segment";
  }
  const stableMatch = raw.match(/_seg_\d+_\d+_[a-z0-9]+_([a-f0-9]{4,})$/i);
  if (stableMatch?.[1]) {
    return `seg#${stableMatch[1].slice(0, 6)}`;
  }
  const hashTail = raw.match(/([a-f0-9]{6,})$/i);
  if (hashTail?.[1]) {
    return `seg#${hashTail[1].slice(0, 6)}`;
  }
  return raw.length > 18 ? `${raw.slice(0, 18)}…` : raw;
}
