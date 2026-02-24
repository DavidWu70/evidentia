/**
 * Generate deterministic pseudo-stream chunks from input text.
 *
 * Design intent:
 * - Simulate real-time updates without browser audio capture dependency.
 * - Keep segment_id stable across reruns for event dedupe behavior.
 * - Provide consistent timing windows for evidence trace tests.
 */

const SENTENCE_SPLIT_RE = /(?<=[.!?])\s+/;

export function buildSimulationSegments(text, startAt = 0) {
  const source = String(text || "").trim();
  if (!source) {
    return [];
  }

  const parts = source
    .split(SENTENCE_SPLIT_RE)
    .map((item) => item.trim())
    .filter(Boolean);

  const segments = [];
  let cursor = Number(startAt) || 0;
  for (let i = 0; i < parts.length; i += 1) {
    const sentence = parts[i];
    const duration = estimateDurationSeconds(sentence);
    const t0 = Number(cursor.toFixed(2));
    const t1 = Number((cursor + duration).toFixed(2));
    cursor = t1 + 0.2;
    segments.push({
      segment_id: `seg_${String(i + 1).padStart(4, "0")}`,
      start: t0,
      end: t1,
      text: sentence,
      speaker: inferSpeaker(sentence),
      asr_confidence: 0.88,
    });
  }

  return segments;
}

function estimateDurationSeconds(sentence) {
  const words = sentence.split(/\s+/).filter(Boolean).length;
  const minDuration = 1.0;
  const estimated = words * 0.33;
  return Math.max(minDuration, Math.min(estimated, 5.2));
}

function inferSpeaker(sentence) {
  const s = sentence.toLowerCase();
  if (s.startsWith("doctor:") || s.startsWith("clinician:")) {
    return "clinician";
  }
  if (s.startsWith("patient:")) {
    return "patient";
  }
  return "patient";
}
