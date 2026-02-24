import React from "https://esm.sh/react@18";
import htm from "https://esm.sh/htm@3";

const html = htm.bind(React.createElement);

/**
 * Render evidence timeline cards with clickable provenance anchors.
 *
 * Design intent:
 * - Keep interpretation and quote provenance in the same visual block.
 * - Let reviewers jump directly from event evidence to source transcript segment.
 */
export function TimelinePanel({
  events,
  selectedSegmentId,
  selectedEventKey = "",
  headerStatus = "",
  emptyStateMessage = "No events extracted yet.",
  onSelectEvidence = () => {},
  onSelectEvent = () => {},
  canPlayAudio = false,
  playingClipKey = "",
  onToggleClip = () => {},
}) {
  const groupedEvents = React.useMemo(() => groupTimelineEvents(events), [events]);

  return html`
    <section className="panel" id="timeline-panel">
      <div className="panel-title-row">
        <h2>Evidence Timeline</h2>
        ${headerStatus ? html`<span className="panel-title-status">${headerStatus}</span>` : null}
      </div>
      <div className="list">
        ${groupedEvents.length
          ? groupedEvents.map((groupedEvent) => {
              const segmentId = groupedEvent.segmentId;
              const eventKey = groupedEvent.eventKey;
              const clipKey = `timeline|${eventKey}`;
              const selectedByEvent = Boolean(selectedEventKey)
                && groupedEvent.eventKeys.includes(selectedEventKey);
              const selectedBySegment = !selectedEventKey && segmentId && segmentId === selectedSegmentId;
              const isPlaying = clipKey === playingClipKey;
              const classes = ["card"];
              if (selectedBySegment) {
                classes.push("selected-evidence");
              }
              if (selectedByEvent) {
                classes.push("selected-event");
              }
              if (isPlaying) {
                classes.push("playing-clip");
              }
              return html`
                <div
                  key=${groupedEvent.groupKey}
                  className=${classes.join(" ")}
                  data-event-key=${eventKey}
                  data-event-keys=${groupedEvent.eventKeys.join("\n")}
                >
                  <div className="card-head">
                    <div>
                      ${groupedEvent.types.map((typeItem) =>
                        html`<span className="badge">${typeItem}</span>`
                      )}
                      ${groupedEvent.labels.map((labelItem) =>
                        html`<span className="badge">${labelItem}</span>`
                      )}
                      ${groupedEvent.polarities.map((polarityItem) =>
                        html`<span className="badge">${polarityItem}</span>`
                      )}
                    </div>
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
                                t0: Number(groupedEvent.t0 || 0),
                                t1: Number(groupedEvent.t1 || 0),
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
                  <button
                    type="button"
                    className="timeline-evidence-main"
                    title=${segmentId ? `Jump to ${segmentId}` : "Evidence card"}
                    onClick=${() => {
                      if (segmentId) {
                        onSelectEvidence(segmentId);
                      }
                      onSelectEvent({ eventKey, segmentId });
                    }}
                  >
                    <div>"${groupedEvent.quote}"</div>
                    <div className="meta">
                      ${`${Number(groupedEvent.t0).toFixed(1)}-${Number(groupedEvent.t1).toFixed(1)}s | `}
                      <span title=${String(groupedEvent.segmentId || "")}>
                        ${formatSegmentRef(groupedEvent.segmentId)}
                      </span>
                      ${` | conf=${Number(groupedEvent.confidence).toFixed(2)}`}
                    </div>
                  </button>
                </div>
              `;
            })
          : html`
              <div className="card">
                <div className="meta panel-empty-message">${emptyStateMessage}</div>
              </div>
            `}
      </div>
    </section>
  `;
}

function groupTimelineEvents(events) {
  if (!Array.isArray(events) || !events.length) {
    return [];
  }
  const grouped = new Map();
  for (const event of events) {
    const evidence = event?.evidence || {};
    const segmentId = String(evidence.segment_id || "");
    const t0 = Number(evidence.t0 || 0);
    const t1 = Number(evidence.t1 || 0);
    const quote = String(evidence.quote || "");
    const normalizedQuote = normalizeQuoteForGroup(quote);
    const groupKey = [
      segmentId,
      Number.isFinite(t0) ? t0.toFixed(2) : "0.00",
      Number.isFinite(t1) ? t1.toFixed(2) : "0.00",
      normalizedQuote,
    ].join("|");
    const eventKey = resolveEventKey(event);
    if (!grouped.has(groupKey)) {
      grouped.set(groupKey, {
        groupKey,
        eventKey,
        eventKeys: [eventKey],
        segmentId,
        t0,
        t1,
        quote,
        confidence: Number(event?.confidence || 0),
        types: dedupNonEmpty([String(event?.type || "")]),
        labels: dedupNonEmpty([String(event?.label || "")]),
        polarities: dedupNonEmpty([String(event?.polarity || "")]),
      });
      continue;
    }
    const current = grouped.get(groupKey);
    current.eventKeys = dedupNonEmpty([...current.eventKeys, eventKey]);
    current.confidence = Math.max(current.confidence, Number(event?.confidence || 0));
    current.types = dedupNonEmpty([...current.types, String(event?.type || "")]);
    current.labels = dedupNonEmpty([...current.labels, String(event?.label || "")]);
    current.polarities = dedupNonEmpty([...current.polarities, String(event?.polarity || "")]);
  }
  return Array.from(grouped.values());
}

function normalizeQuoteForGroup(value) {
  return String(value || "")
    .trim()
    .replace(/\s+/g, " ")
    .toLowerCase();
}

function dedupNonEmpty(values) {
  const out = [];
  const seen = new Set();
  for (const value of values) {
    const normalized = String(value || "").trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function resolveEventKey(event) {
  const evidence = event?.evidence || {};
  const eventId = String(event?.event_id || "").trim();
  if (eventId) {
    return eventId;
  }
  return [
    String(evidence.segment_id || ""),
    String(event?.type || ""),
    String(event?.label || ""),
    String(event?.polarity || ""),
  ].join("|");
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
