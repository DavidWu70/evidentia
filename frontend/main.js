import React, { useCallback, useEffect, useMemo, useRef, useState } from "https://esm.sh/react@18";
import { createRoot } from "https://esm.sh/react-dom@18/client";
import htm from "https://esm.sh/htm@3";

import { ApiClient } from "./hooks/api.js";
import { buildSimulationSegments } from "./hooks/streamSimulator.js";
import {
  cleanupPendingMicChunks,
  clearPendingMicChunksForSession,
  listPendingMicChunks,
  persistPendingMicChunk,
  removePendingMicChunk,
} from "./hooks/micQueueStore.js";
import { ControlPanel } from "./components/ControlPanel.js";
import { TranscriptPanel } from "./components/TranscriptPanel.js";
import { TimelinePanel } from "./components/TimelinePanel.js";
import { InsightPanel } from "./components/InsightPanel.js";

const html = htm.bind(React.createElement);

/**
 * React orchestration entry for review MVP.
 *
 * Design intent:
 * - Keep all UI state in one component for deterministic demo playback.
 * - Route transcript mode and audio mode through one backend contract.
 * - Surface pipeline debug so reviewers can verify ASR/model usage quickly.
 */

const DEFAULT_TEXT =
  "Patient: I have felt very down for 2 weeks. Patient: I feel exhausted and cannot sleep well. Patient: Sometimes I wish I would not wake up. Clinician: Thank you for sharing this today.";
const DEFAULT_AUDIO_PATH = "";
const LOCAL_AUDIO_SOURCE_VALUE = "__local_file__";
const MAX_LLM_UTTERANCES_PER_CALL = 8;
const MAX_MIC_BUFFERED_CHUNKS = 600;
const MAX_MIC_BUFFERED_ARCHIVE_CHUNKS = 1200;
const MAX_MIC_ARCHIVE_CHUNKS = 1800;
const MAX_MIC_RECONNECT_ATTEMPTS = 8;
const MIC_STOP_DRAIN_TIMEOUT_MS = 12000;
const MIC_STOP_DRAIN_POLL_MS = 200;
const DEFAULT_MIC_CAPTURE_SLICE_MS = 1000;
const DEFAULT_MIC_ASR_WINDOW_SEC = 20;
const DEFAULT_MIC_ASR_STEP_SEC = 4;
const MIC_PERSIST_MAX_CHUNKS = 2400;
const MIC_PERSIST_TTL_MS = 1000 * 60 * 60 * 12;
const TRANSCRIPT_SPLITTER_WIDTH_PX = 14;
const TRANSCRIPT_PANEL_MIN_WIDTHS = [140, 140, 140];
const NOTE_HISTORY_MAX = 300;
const NAV_ITEMS = [
  { id: "transcript", label: "Assistant", icon: "A" },
  { id: "template", label: "Template", icon: "M" },
  { id: "config", label: "Config", icon: "C" },
];
const REVIEW_TAB_ITEMS = [
  { id: "context", label: "Context" },
  { id: "transcript", label: "Transcript" },
  { id: "notes", label: "Notes" },
];
const NOTE_JOB_POLL_MS = 700;
const NOTE_JOB_ACTIVE_STATES = new Set(["pending", "generating", "stopping"]);
const EMPTY_DELETE_DIALOG_STATE = {
  open: false,
  segmentId: "",
  deletedRow: null,
  deletedIndex: -1,
  rowsAfterDelete: [],
  previousOverride: undefined,
  wasSelected: false,
};

const EMPTY_SNAPSHOT = {
  problem_list: [],
  risk_flags: [],
  open_questions: [],
  mandatory_safety_questions: [],
  contextual_followups: [],
  rationale: "",
};

const EMPTY_PIPELINE_DEBUG = {
  source: "",
  asr_status: "",
  diarization_status: "",
  diarization_reason: "",
  alignment_reason_counts: "",
  alignment_fallback_rate: "",
  sentence_role_split: "",
  event_guardrails: "",
  risk_backstop: "",
  event_harmonization: "",
  event_engine_used: "",
  medgemma_status: "",
  medgemma_error: "",
  medgemma_calls: 0,
  medgemma_json_valid_rate: "",
  medgemma_fallback_rate: "",
  medgemma_filter_p50_p95_ms: "",
  medgemma_extract_p50_p95_ms: "",
  open_questions_mode: "",
  open_questions_ai_error: "",
  note_department: "",
  note_templates: "",
  note_templates_generated: "",
  live_audio_status: "",
  live_audio_chunks: "",
  live_audio_bytes: "",
  live_audio_recording_path: "",
  live_audio_archive_end_sec: "",
};

function App() {
  const [activeNav, setActiveNav] = useState("transcript");
  const [activeReviewTab, setActiveReviewTab] = useState("transcript");
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8000");
  const [sessionId, setSessionId] = useState(`s_${Math.floor(Date.now() / 1000)}`);
  const [intervalMs, setIntervalMs] = useState(1200);

  const [inputMode, setInputMode] = useState("live");
  const [audioPath, setAudioPath] = useState(DEFAULT_AUDIO_PATH);
  const [audioPathDisplay, setAudioPathDisplay] = useState("");
  const [audioSampleOptions, setAudioSampleOptions] = useState([]);
  const [selectedAudioSource, setSelectedAudioSource] = useState("");
  const [audioWindowSec, setAudioWindowSec] = useState(10);
  const [reconcileLookbackWindows, setReconcileLookbackWindows] = useState(1);
  const [llmUpdateIntervalWindows, setLlmUpdateIntervalWindows] = useState(3);
  const [openQuestionsAiEnhancementEnabled, setOpenQuestionsAiEnhancementEnabled] = useState(true);
  const [micCaptureSliceMs, setMicCaptureSliceMs] = useState(DEFAULT_MIC_CAPTURE_SLICE_MS);
  const [micAsrWindowSec, setMicAsrWindowSec] = useState(DEFAULT_MIC_ASR_WINDOW_SEC);
  const [micAsrStepSec, setMicAsrStepSec] = useState(DEFAULT_MIC_ASR_STEP_SEC);
  const [audioCursorSec, setAudioCursorSec] = useState(0);
  const [audioDurationSec, setAudioDurationSec] = useState(null);

  const [eventEngine, setEventEngine] = useState("medgemma");
  const [noteTemplateCatalog, setNoteTemplateCatalog] = useState({});
  const [noteDepartment, setNoteDepartment] = useState("psych");
  const [patientIdentity, setPatientIdentity] = useState("");
  const [noteTemplateIds, setNoteTemplateIds] = useState([]);
  const [noteTemplateStatuses, setNoteTemplateStatuses] = useState({});
  const [noteGenerationStatus, setNoteGenerationStatus] = useState("idle");
  const [noteJobId, setNoteJobId] = useState("");
  const [generatedDrafts, setGeneratedDrafts] = useState([]);
  const [activeDraftTemplateId, setActiveDraftTemplateId] = useState("");
  const [selectedTemplateEditorId, setSelectedTemplateEditorId] = useState("");
  const [templateExpandedDepartments, setTemplateExpandedDepartments] = useState({});
  const [templateEditorDoc, setTemplateEditorDoc] = useState(null);
  const [templateEditorDirty, setTemplateEditorDirty] = useState(false);
  const [templateEditorLoading, setTemplateEditorLoading] = useState(false);
  const [templateEditorMessage, setTemplateEditorMessage] = useState("");
  const [canUndoTemplate, setCanUndoTemplate] = useState(false);
  const [canRedoTemplate, setCanRedoTemplate] = useState(false);
  const [patientContextText, setPatientContextText] = useState("");
  const [streamText, setStreamText] = useState(DEFAULT_TEXT);

  const [streamSegments, setStreamSegments] = useState([]);
  const [cursor, setCursor] = useState(0);
  const [transcriptRows, setTranscriptRows] = useState([]);
  const [manualTranscriptOverrides, setManualTranscriptOverrides] = useState({});
  const [events, setEvents] = useState([]);
  const [snapshot, setSnapshot] = useState(EMPTY_SNAPSHOT);
  const [noteText, setNoteText] = useState("");
  const [canUndoNote, setCanUndoNote] = useState(false);
  const [canRedoNote, setCanRedoNote] = useState(false);
  const [citations, setCitations] = useState([]);
  const [pipelineDebug, setPipelineDebug] = useState(EMPTY_PIPELINE_DEBUG);
  const [status, setStatus] = useState("idle");
  const [lastError, setLastError] = useState("");
  const [micStatus, setMicStatus] = useState("idle");
  const [liveRecordingAudioPath, setLiveRecordingAudioPath] = useState("");
  const [liveArchiveEndSec, setLiveArchiveEndSec] = useState(0);
  const [selectedSegmentId, setSelectedSegmentId] = useState("");
  const [selectedEventKey, setSelectedEventKey] = useState("");
  const [playingClipKey, setPlayingClipKey] = useState("");
  const [isTranscriptReprocessing, setIsTranscriptReprocessing] = useState(false);
  const [transcriptPaneWidths, setTranscriptPaneWidths] = useState([32, 33, 35]);
  const [activeSplitter, setActiveSplitter] = useState(-1);
  const [deleteDecisionDialog, setDeleteDecisionDialog] = useState(EMPTY_DELETE_DIALOG_STATE);

  const apiRef = useRef(new ApiClient(baseUrl));
  const inFlightRef = useRef(false);
  const noteDirtyRef = useRef(false);
  const eventStoreRef = useRef(new Map());
  const recentWindowRowsRef = useRef([]);
  const medgemmaAsyncInFlightRef = useRef(false);
  const autoWindowCounterRef = useRef(0);
  const medgemmaStatsRef = useRef(createEmptyMedgemmaStats());
  const stateRef = useRef(null);
  const transcriptLayoutRef = useRef(null);
  const paneDragRef = useRef(null);
  const micRecorderRef = useRef(null);
  const micStreamRef = useRef(null);
  const micSocketRef = useRef(null);
  const micChunkSeqRef = useRef(0);
  const micHydrateChainRef = useRef(Promise.resolve());
  const micQueueRef = useRef([]);
  const micInFlightChunkRef = useRef(null);
  const micArchiveQueueRef = useRef([]);
  const micArchiveInFlightChunkRef = useRef(null);
  const micReconnectTimerRef = useRef(null);
  const micStopDrainTimerRef = useRef(null);
  const micReconnectAttemptRef = useRef(0);
  const micShouldRunRef = useRef(false);
  const micStoppingRef = useRef(false);
  const micActiveMimeTypeRef = useRef("");
  const micArchiveRef = useRef([]);
  const micFallbackTriggeredRef = useRef(false);
  const micRawChunkBufferRef = useRef([]);
  const micWindowSeqRef = useRef(0);
  const notePollTimerRef = useRef(null);
  const noteUndoStackRef = useRef([]);
  const noteRedoStackRef = useRef([]);
  const templateUndoStackRef = useRef([]);
  const templateRedoStackRef = useRef([]);
  const clipAudioRef = useRef(null);
  const manualTranscriptOverridesRef = useRef({});
  const clipPlaybackRef = useRef({
    clipKey: "",
    t0: 0,
    t1: 0,
    audioUrl: "",
  });

  const micSupported = useMemo(() => isMicCaptureSupported(), []);

  useEffect(() => {
    apiRef.current.setBaseUrl(baseUrl);
  }, [baseUrl]);

  useEffect(() => {
    let cancelled = false;

    const loadTemplates = async () => {
      try {
        const response = await apiRef.current.noteTemplates();
        const catalog = sanitizeTemplateCatalog(response);
        const departments = Object.keys(catalog);
        if (!departments.length) {
          throw new Error("No note templates found from backend /note/templates.");
        }
        if (cancelled) {
          return;
        }
        setNoteTemplateCatalog(catalog);

        const stateSnapshot = stateRef.current || {};
        const resolvedDepartment = normalizeDepartment(
          stateSnapshot.noteDepartment || noteDepartment,
          catalog,
        );
        const resolvedTemplateIds = normalizeTemplateIds(
          resolvedDepartment,
          stateSnapshot.noteTemplateIds || noteTemplateIds,
          catalog,
        );
        noteDirtyRef.current = false;
        setNoteDepartment(resolvedDepartment);
        setNoteTemplateIds(resolvedTemplateIds);
        setNoteTemplateStatuses(buildDefaultTemplateStatuses(resolvedDepartment, catalog));
        setNoteGenerationStatus("idle");
        setNoteJobId("");
        setActiveDraftTemplateId(resolvedTemplateIds[0] || "");
        setSelectedTemplateEditorId(resolvedTemplateIds[0] || "");
        setTemplateEditorDoc(null);
        templateUndoStackRef.current = [];
        templateRedoStackRef.current = [];
        setCanUndoTemplate(false);
        setCanRedoTemplate(false);
        setTemplateEditorDirty(false);
        setTemplateEditorMessage("");
      } catch (error) {
        if (!cancelled) {
          setLastError(String(error?.message || error));
          setNoteTemplateCatalog({});
          setNoteTemplateIds([]);
          setNoteTemplateStatuses({});
          setNoteGenerationStatus("idle");
          setNoteJobId("");
          setActiveDraftTemplateId("");
          setSelectedTemplateEditorId("");
          setTemplateEditorDoc(null);
          templateUndoStackRef.current = [];
          templateRedoStackRef.current = [];
          setCanUndoTemplate(false);
          setCanRedoTemplate(false);
          setTemplateEditorDirty(false);
          setTemplateEditorMessage("");
        }
      }
    };

    void loadTemplates();
    return () => {
      cancelled = true;
    };
  }, [baseUrl]);

  useEffect(() => {
    let cancelled = false;

    const loadSampleAudioFiles = async () => {
      try {
        const response = await apiRef.current.listSampleAudioFiles();
        if (cancelled) {
          return;
        }
        const files = Array.isArray(response?.files) ? response.files : [];
        const normalized = files
          .map((item) => ({
            value: String(item?.audio_path || "").trim(),
            label: String(item?.name || "").trim(),
            sizeBytes: Number(item?.size_bytes || 0),
          }))
          .filter((item) => item.value && item.label);
        setAudioSampleOptions(normalized);
        setSelectedAudioSource((prev) => {
          const current = String(prev || "").trim();
          if (current && (current === LOCAL_AUDIO_SOURCE_VALUE || normalized.some((item) => item.value === current))) {
            return current;
          }
          if (normalized.length) {
            return normalized[0].value;
          }
          return LOCAL_AUDIO_SOURCE_VALUE;
        });
      } catch (_) {
        if (cancelled) {
          return;
        }
        setAudioSampleOptions([]);
        setSelectedAudioSource((prev) => (String(prev || "").trim() || LOCAL_AUDIO_SOURCE_VALUE));
      }
    };

    void loadSampleAudioFiles();
    return () => {
      cancelled = true;
    };
  }, [baseUrl]);

  useEffect(() => {
    const selected = String(selectedAudioSource || "").trim();
    if (!selected || selected === LOCAL_AUDIO_SOURCE_VALUE) {
      return;
    }
    setAudioPath(selected);
    setAudioPathDisplay("");
  }, [selectedAudioSource]);

  useEffect(() => {
    stateRef.current = {
      sessionId,
      intervalMs,
      inputMode,
      audioPath,
      audioPathDisplay,
      selectedAudioSource,
      audioWindowSec,
      reconcileLookbackWindows,
      llmUpdateIntervalWindows,
      openQuestionsAiEnhancementEnabled,
      micCaptureSliceMs,
      micAsrWindowSec,
      micAsrStepSec,
      audioCursorSec,
      liveRecordingAudioPath,
      liveArchiveEndSec,
      eventEngine,
      noteTemplateCatalog,
      noteDepartment,
      noteTemplateIds,
      streamText,
      streamSegments,
      cursor,
      status,
    };
  }, [
    sessionId,
    intervalMs,
    inputMode,
    audioPath,
    audioPathDisplay,
    selectedAudioSource,
    audioWindowSec,
    reconcileLookbackWindows,
    llmUpdateIntervalWindows,
    openQuestionsAiEnhancementEnabled,
    micCaptureSliceMs,
    micAsrWindowSec,
    micAsrStepSec,
    audioCursorSec,
    liveRecordingAudioPath,
    liveArchiveEndSec,
    eventEngine,
    noteTemplateCatalog,
    noteDepartment,
    noteTemplateIds,
    streamText,
    streamSegments,
    cursor,
    status,
  ]);

  useEffect(() => {
    manualTranscriptOverridesRef.current = manualTranscriptOverrides || {};
  }, [manualTranscriptOverrides]);

  const statusText = useMemo(() => {
    if (lastError) {
      return `error: ${lastError}`;
    }

    if (inputMode === "live") {
      return `${status} (Mic: ${micStatus || "idle"})`;
    }

    if (inputMode === "audio") {
      if (Number.isFinite(audioDurationSec) && Number(audioDurationSec) > 0) {
        const shown = Math.min(Number(audioCursorSec) || 0, Number(audioDurationSec));
        return `${status} (${shown.toFixed(1)}/${Number(audioDurationSec).toFixed(1)}s)`;
      }
      return `${status} (${(Number(audioCursorSec) || 0).toFixed(1)}s)`;
    }

    const progressed = streamSegments.length ? `${cursor}/${streamSegments.length}` : "0/0";
    return `${status} (${progressed})`;
  }, [audioCursorSec, audioDurationSec, cursor, inputMode, lastError, micStatus, status, streamSegments.length]);

  const clipSourceAudioPath = useMemo(() => {
    if (isAudioPathSource(pipelineDebug.source)) {
      return String(audioPath || "").trim();
    }
    if (isLiveAudioSource(pipelineDebug.source)) {
      return String(liveRecordingAudioPath || "").trim();
    }
    return "";
  }, [audioPath, liveRecordingAudioPath, pipelineDebug.source]);
  const canPlayAudioClips = useMemo(
    () => Boolean(clipSourceAudioPath),
    [clipSourceAudioPath],
  );
  const audioFileUrl = useMemo(
    () => (
      canPlayAudioClips
        ? buildAudioFileUrl(baseUrl, clipSourceAudioPath)
        : ""
    ),
    [baseUrl, canPlayAudioClips, clipSourceAudioPath],
  );
  const templateDepartmentGroups = useMemo(
    () => Object.entries(noteTemplateCatalog || {}).sort(([left], [right]) => compareNoteDepartments(left, right)),
    [noteTemplateCatalog],
  );
  const selectedTemplateEditorLabel = useMemo(() => {
    const targetId = String(selectedTemplateEditorId || "").trim();
    if (!targetId) {
      return "Template";
    }
    for (const templates of Object.values(noteTemplateCatalog || {})) {
      for (const item of Array.isArray(templates) ? templates : []) {
        if (String(item?.id || "").trim() === targetId) {
          return String(item?.label || targetId);
        }
      }
    }
    return targetId;
  }, [noteTemplateCatalog, selectedTemplateEditorId]);

  useEffect(() => {
    if (!templateDepartmentGroups.length) {
      setTemplateExpandedDepartments({});
      return;
    }
    setTemplateExpandedDepartments((prev) => {
      const next = {};
      for (const [departmentId] of templateDepartmentGroups) {
        const key = String(departmentId || "").trim();
        if (!key) {
          continue;
        }
        next[key] = Object.prototype.hasOwnProperty.call(prev, key)
          ? Boolean(prev[key])
          : true;
      }
      return next;
    });
  }, [templateDepartmentGroups]);

  useEffect(() => {
    const options = getTemplateOptions(noteDepartment, noteTemplateCatalog);
    if (!options.length) {
      if (selectedTemplateEditorId) {
        setSelectedTemplateEditorId("");
      }
      setTemplateEditorDoc(null);
      templateUndoStackRef.current = [];
      templateRedoStackRef.current = [];
      setCanUndoTemplate(false);
      setCanRedoTemplate(false);
      setTemplateEditorDirty(false);
      setTemplateEditorMessage("");
      return;
    }
    if (!options.some((item) => item.id === selectedTemplateEditorId)) {
      setSelectedTemplateEditorId(options[0].id);
    }
  }, [noteDepartment, noteTemplateCatalog, selectedTemplateEditorId]);

  const loadTemplateEditor = useCallback(async (templateId, options = {}) => {
    const targetId = String(templateId || "").trim();
    if (!targetId) {
      return;
    }
    const preserveMessage = Boolean(options.preserveMessage);
    setTemplateEditorLoading(true);
    try {
      const response = await apiRef.current.noteTemplateDocument(noteDepartment, targetId);
      const normalizedDoc = normalizeTemplateEditorDocument(response?.template || {});
      setTemplateEditorDoc(normalizedDoc);
      templateUndoStackRef.current = [];
      templateRedoStackRef.current = [];
      setCanUndoTemplate(false);
      setCanRedoTemplate(false);
      setTemplateEditorDirty(false);
      if (!preserveMessage) {
        setTemplateEditorMessage(`Loaded: ${targetId}`);
      }
      setLastError("");
    } catch (error) {
      setLastError(String(error?.message || error));
      if (!preserveMessage) {
        setTemplateEditorMessage("");
      }
    } finally {
      setTemplateEditorLoading(false);
    }
  }, [noteDepartment]);

  useEffect(() => {
    if (!selectedTemplateEditorId) {
      return;
    }
    void loadTemplateEditor(selectedTemplateEditorId, { preserveMessage: true });
  }, [loadTemplateEditor, noteDepartment, selectedTemplateEditorId]);

  const clearNotePollTimer = useCallback(() => {
    const timer = notePollTimerRef.current;
    if (timer) {
      window.clearTimeout(timer);
    }
    notePollTimerRef.current = null;
  }, []);

  const syncNoteHistoryAvailability = useCallback(() => {
    setCanUndoNote(noteUndoStackRef.current.length > 0);
    setCanRedoNote(noteRedoStackRef.current.length > 0);
  }, []);

  const syncTemplateHistoryAvailability = useCallback(() => {
    setCanUndoTemplate(templateUndoStackRef.current.length > 0);
    setCanRedoTemplate(templateRedoStackRef.current.length > 0);
  }, []);

  const setNoteTextAndResetHistory = useCallback((value) => {
    const normalized = String(value || "");
    setNoteText(normalized);
    noteUndoStackRef.current = [];
    noteRedoStackRef.current = [];
    syncNoteHistoryAvailability();
  }, [syncNoteHistoryAvailability]);

  const applyNoteDraftJobStatus = useCallback((jobResp, options = {}) => {
    const respectDirty = options.respectDirty !== false;
    const requestedIdsRaw = Array.isArray(jobResp?.requested_template_ids)
      ? jobResp.requested_template_ids
      : [];
    const requestedIds = requestedIdsRaw.map((item) => String(item || "").trim()).filter(Boolean);
    const templateStatusesRaw = jobResp?.template_statuses && typeof jobResp.template_statuses === "object"
      ? jobResp.template_statuses
      : {};
    const nextStatuses = {};
    for (const templateId of requestedIds) {
      nextStatuses[templateId] = String(templateStatusesRaw[templateId] || "queued");
    }
    setNoteTemplateStatuses(nextStatuses);

    const drafts = Array.isArray(jobResp?.drafts) ? jobResp.drafts : [];
    setGeneratedDrafts(drafts);
    if (drafts.length) {
      const preferredTemplateId = drafts.some((item) => item.template_id === activeDraftTemplateId)
        ? activeDraftTemplateId
        : String(drafts[0]?.template_id || "");
      const activeDraft = drafts.find((item) => item.template_id === preferredTemplateId) || drafts[0];
      setActiveDraftTemplateId(preferredTemplateId);
      if ((!respectDirty || !noteDirtyRef.current) && activeDraft) {
        setNoteTextAndResetHistory(String(activeDraft.note_text || ""));
        setCitations(Array.isArray(activeDraft.citations) ? activeDraft.citations : []);
      }
    } else if (!respectDirty || !noteDirtyRef.current) {
      setNoteTextAndResetHistory("");
      setCitations([]);
    }

    const status = String(jobResp?.status || "");
    setNoteGenerationStatus(status || "idle");
    if (status && !NOTE_JOB_ACTIVE_STATES.has(status)) {
      clearNotePollTimer();
      setNoteJobId("");
    }

    const generatedTemplateIds = drafts.map((item) => String(item?.template_id || ""));
    if (generatedTemplateIds.length || requestedIds.length) {
      setPipelineDebug((prev) => ({
        ...prev,
        note_department: String(jobResp?.department || noteDepartment || ""),
        note_templates: requestedIds.join(", "),
        note_templates_generated: generatedTemplateIds.join(", "),
      }));
    }
  }, [activeDraftTemplateId, clearNotePollTimer, noteDepartment, setNoteTextAndResetHistory]);

  const runPipelineFromRows = useCallback(async (rows, engine, options = {}) => {
    if (!rows.length) {
      return;
    }

    const fallbackToRule =
      typeof options.fallbackToRule === "boolean" ? options.fallbackToRule : engine !== "medgemma";
    let eventsResp;
    try {
      eventsResp = await apiRef.current.eventsExtract({
        engine,
        fallback_to_rule: fallbackToRule,
        utterances: rows.map((item) => ({
          segment_id: item.segment_id,
          t0: item.start,
          t1: item.end,
          speaker: item.speaker,
          text: item.text,
          asr_confidence: 0.88,
        })),
      });
    } catch (error) {
      if (engine === "medgemma" || engine === "auto") {
        setPipelineDebug((prev) => ({
          ...prev,
          medgemma_error: String(error?.message || error),
          ...recordMedgemmaCallError(medgemmaStatsRef.current),
        }));
      }
      throw error;
    }

    const incomingEvents = Array.isArray(eventsResp?.events) ? eventsResp.events : [];
    const impactedSegmentIds = new Set(rows.map((item) => item.segment_id));
    const nextStore = new Map(eventStoreRef.current);

    for (const [key, eventItem] of nextStore.entries()) {
      const segId = String(eventItem?.evidence?.segment_id || "");
      if (segId && impactedSegmentIds.has(segId)) {
        nextStore.delete(key);
      }
    }

    for (const eventItem of incomingEvents) {
      const eventKey = buildEventStoreKey(eventItem);
      nextStore.set(eventKey, { ...eventItem, event_id: eventKey });
    }

    eventStoreRef.current = nextStore;
    const mergedEvents = sortEventsForTimeline(Array.from(nextStore.values()));
    setEvents(mergedEvents);

    const aiEnhancementEnabled = Boolean(stateRef.current?.openQuestionsAiEnhancementEnabled ?? true);
    const snapshotResp = await apiRef.current.stateSnapshot({
      events: mergedEvents,
      ai_enhancement_enabled: aiEnhancementEnabled,
    });
    const nextSnapshot = {
      problem_list: snapshotResp.problem_list || [],
      risk_flags: snapshotResp.risk_flags || [],
      open_questions: snapshotResp.open_questions || [],
      mandatory_safety_questions: snapshotResp.mandatory_safety_questions || [],
      contextual_followups: snapshotResp.contextual_followups || [],
      rationale: String(snapshotResp.rationale || ""),
    };
    setSnapshot(nextSnapshot);

    setPipelineDebug((prev) => ({
      ...prev,
      event_engine_used: String(eventsResp?.debug?.engine_used || ""),
      medgemma_status: String(eventsResp?.debug?.medgemma?.status || ""),
      medgemma_error: String(eventsResp?.debug?.medgemma_error || ""),
      open_questions_mode: String(snapshotResp?.debug?.open_questions_mode || ""),
      open_questions_ai_error: String(snapshotResp?.debug?.ai_enhancement_error || ""),
      ...updateMedgemmaStats(medgemmaStatsRef.current, eventsResp?.debug || {}, engine),
    }));
  }, []);

  const runAutoMedgemmaRefresh = useCallback(async (rows) => {
    if (!rows.length || medgemmaAsyncInFlightRef.current) {
      return;
    }

    medgemmaAsyncInFlightRef.current = true;
    try {
      const capped = rows.slice(-MAX_LLM_UTTERANCES_PER_CALL);
      await runPipelineFromRows(capped, "medgemma", { fallbackToRule: false });
    } catch (error) {
      setPipelineDebug((prev) => ({
        ...prev,
        medgemma_error: String(error?.message || error),
      }));
    } finally {
      medgemmaAsyncInFlightRef.current = false;
    }
  }, [runPipelineFromRows]);

  const hydrateFromTranscribeResponse = useCallback(async (transcribeResp, currentSessionId, engine, options = {}) => {
    const allowEmpty = Boolean(options.allowEmpty);
    const isAudioWindowMode = Boolean(options.isAudioWindowMode);
    const reconcileLookback = Math.max(0, Number(options.reconcileLookbackWindows) || 0);
    const overrides = manualTranscriptOverridesRef.current || {};
    const rows = applyManualTranscriptOverrides(
      buildRowsFromUtterances(transcribeResp?.utterances || [], currentSessionId),
      overrides,
    );
    let newRows = applyManualTranscriptOverrides(
      buildRowsFromUtterances(transcribeResp?.new_utterances || [], currentSessionId),
      overrides,
    );

    if (!newRows.length) {
      const incrementalCount = Number(transcribeResp?.debug?.incremental?.new_segments || 0);
      if (incrementalCount > 0 && rows.length >= incrementalCount) {
        newRows = rows.slice(rows.length - incrementalCount);
      }
    }

    const stream = transcribeResp?.debug?.asr?.stream || null;
    if (stream) {
      const duration = Number(stream.audio_duration_sec);
      if (Number.isFinite(duration) && duration >= 0) {
        setAudioDurationSec(duration);
      }
    }

    setPipelineDebug((prev) => ({
      ...prev,
      source: String(transcribeResp?.debug?.source || ""),
      asr_status: String(transcribeResp?.debug?.asr?.status || ""),
      diarization_status: String(transcribeResp?.debug?.diarization?.status || ""),
      diarization_reason: extractDiarizationReason(transcribeResp?.debug?.diarization || {}),
      alignment_reason_counts: extractAlignmentReasonCounts(transcribeResp?.debug?.alignment || {}),
      alignment_fallback_rate: extractAlignmentFallbackRate(transcribeResp?.debug?.alignment || {}),
      sentence_role_split: extractSentenceRoleSplit(transcribeResp?.debug?.sentence_role_split || {}),
      event_guardrails: extractSimpleDropSummary(transcribeResp?.debug?.event_guardrails || {}),
      risk_backstop: extractRiskBackstopSummary(transcribeResp?.debug?.risk_backstop || {}),
      event_harmonization: extractSimpleDropSummary(transcribeResp?.debug?.event_harmonization || {}),
    }));

    if (rows.length) {
      setTranscriptRows(rows);
    }

    if (!newRows.length) {
      if (allowEmpty) {
        return;
      }
      throw new Error("No utterances produced by /transcribe_structured.");
    }

    let rowsForExtraction = newRows;
    if (isAudioWindowMode) {
      recentWindowRowsRef.current.push(newRows);
      const maxWindows = Math.max(1, reconcileLookback + 1);
      if (recentWindowRowsRef.current.length > maxWindows) {
        recentWindowRowsRef.current = recentWindowRowsRef.current.slice(-maxWindows);
      }
      rowsForExtraction = recentWindowRowsRef.current.flat();
    }

    if (isAudioWindowMode && engine === "auto") {
      await runPipelineFromRows(rowsForExtraction, "rule", { fallbackToRule: true });
      autoWindowCounterRef.current += 1;
      const llmInterval = Math.max(1, Number(options.llmUpdateIntervalWindows) || 3);
      if (autoWindowCounterRef.current % llmInterval === 0) {
        void runAutoMedgemmaRefresh(rowsForExtraction);
      }
      return;
    }

    await runPipelineFromRows(rowsForExtraction, engine);
  }, [runAutoMedgemmaRefresh, runPipelineFromRows]);

  const hydrateFromLiveTranscriptResponse = useCallback(async (liveResp, currentSessionId, engine) => {
    const overrides = manualTranscriptOverridesRef.current || {};
    const rows = applyManualTranscriptOverrides(
      buildRowsFromUtterances(liveResp?.utterances || [], currentSessionId),
      overrides,
    );
    const newRows = applyManualTranscriptOverrides(
      buildRowsFromUtterances(liveResp?.new_utterances || [], currentSessionId),
      overrides,
    );
    const recordingAudioPath = String(liveResp?.debug?.recording_audio_path || "").trim();

    if (recordingAudioPath) {
      setLiveRecordingAudioPath(recordingAudioPath);
    }

    if (rows.length) {
      setTranscriptRows(rows);
    }

    setPipelineDebug((prev) => ({
      ...prev,
      source: "live_audio_ws",
      asr_status: String(liveResp?.debug?.asr_status || prev.asr_status || ""),
      diarization_status: String(liveResp?.debug?.diarization_status || prev.diarization_status || ""),
      diarization_reason: String(liveResp?.debug?.diarization_reason || prev.diarization_reason || ""),
      alignment_reason_counts: String(liveResp?.debug?.alignment_reason_counts || prev.alignment_reason_counts || ""),
      alignment_fallback_rate: String(liveResp?.debug?.alignment_fallback_rate || prev.alignment_fallback_rate || ""),
      sentence_role_split:
        typeof liveResp?.debug?.sentence_role_split === "object"
          ? extractSentenceRoleSplit(liveResp?.debug?.sentence_role_split || {})
          : String(liveResp?.debug?.sentence_role_split || prev.sentence_role_split || ""),
      live_audio_status: String(liveResp?.status || prev.live_audio_status || ""),
      live_audio_recording_path: recordingAudioPath || String(prev.live_audio_recording_path || ""),
    }));

    if (!newRows.length) {
      return;
    }

    if (engine === "auto") {
      await runPipelineFromRows(newRows, "rule", { fallbackToRule: true });
      autoWindowCounterRef.current += 1;
      const llmInterval = Math.max(1, Number(stateRef.current?.llmUpdateIntervalWindows) || 3);
      if (autoWindowCounterRef.current % llmInterval === 0) {
        void runAutoMedgemmaRefresh(newRows);
      }
      return;
    }

    await runPipelineFromRows(newRows, engine);
  }, [runAutoMedgemmaRefresh, runPipelineFromRows]);

  const reprocessPipelineFromTranscriptRows = useCallback(async (rows) => {
    const normalizedRows = Array.isArray(rows) ? rows : [];
    setSelectedEventKey("");
    eventStoreRef.current = new Map();
    setEvents([]);
    if (!normalizedRows.length) {
      setSnapshot(EMPTY_SNAPSHOT);
      return;
    }
    const nextEngine = String(stateRef.current?.eventEngine || eventEngine || "auto");
    setIsTranscriptReprocessing(true);
    try {
      await runPipelineFromRows(normalizedRows, nextEngine);
      setLastError("");
    } catch (error) {
      setLastError(String(error?.message || error));
    } finally {
      setIsTranscriptReprocessing(false);
    }
  }, [eventEngine, runPipelineFromRows]);

  const onSaveTranscriptEdit = useCallback(async ({ segmentId, speakerRole, text }) => {
    const normalizedSegmentId = String(segmentId || "").trim();
    const normalizedText = String(text || "").trim();
    if (!normalizedSegmentId || !normalizedText) {
      return;
    }
    const normalizedRole = normalizeEditableSpeakerRole(speakerRole);
    let changed = false;
    const nextRows = transcriptRows.map((item) => {
      if (String(item?.segment_id || "") !== normalizedSegmentId) {
        return item;
      }
      changed = true;
      return {
        ...item,
        text: normalizedText,
        speaker: normalizedRole,
        speaker_role: normalizedRole,
      };
    });
    if (!changed || !nextRows.length) {
      return;
    }
    setTranscriptRows(nextRows);
    setSelectedSegmentId(normalizedSegmentId);
    setManualTranscriptOverrides((prev) => ({
      ...(prev || {}),
      [normalizedSegmentId]: {
        text: normalizedText,
        speaker: normalizedRole,
        speaker_role: normalizedRole,
      },
    }));
    void reprocessPipelineFromTranscriptRows(nextRows);
  }, [reprocessPipelineFromTranscriptRows, transcriptRows]);

  const onDeleteTranscriptSegment = useCallback(({ segmentId }) => {
    const normalizedSegmentId = String(segmentId || "").trim();
    if (!normalizedSegmentId) {
      return;
    }
    const deletedIndex = transcriptRows.findIndex(
      (item) => String(item?.segment_id || "") === normalizedSegmentId,
    );
    if (deletedIndex < 0) {
      return;
    }
    const deletedRow = transcriptRows[deletedIndex] || null;
    const nextRows = transcriptRows.filter(
      (item) => String(item?.segment_id || "") !== normalizedSegmentId,
    );
    if (nextRows.length === transcriptRows.length) {
      return;
    }
    const hasPreviousOverride = Object.prototype.hasOwnProperty.call(
      manualTranscriptOverridesRef.current || {},
      normalizedSegmentId,
    );
    const previousOverride = hasPreviousOverride
      ? manualTranscriptOverridesRef.current[normalizedSegmentId]
      : undefined;
    setTranscriptRows(nextRows);
    setManualTranscriptOverrides((prev) => ({
      ...(prev || {}),
      [normalizedSegmentId]: {
        deleted: true,
      },
    }));
    if (selectedSegmentId === normalizedSegmentId) {
      setSelectedSegmentId("");
    }
    setDeleteDecisionDialog({
      open: true,
      segmentId: normalizedSegmentId,
      deletedRow,
      deletedIndex,
      rowsAfterDelete: nextRows,
      previousOverride,
      wasSelected: selectedSegmentId === normalizedSegmentId,
    });
  }, [selectedSegmentId, transcriptRows]);

  const onCancelDeleteSegment = useCallback(() => {
    if (!deleteDecisionDialog.open || !deleteDecisionDialog.deletedRow || !deleteDecisionDialog.segmentId) {
      setDeleteDecisionDialog(EMPTY_DELETE_DIALOG_STATE);
      return;
    }
    const segmentId = deleteDecisionDialog.segmentId;
    const restoredRow = deleteDecisionDialog.deletedRow;
    const restoreIndex = Number(deleteDecisionDialog.deletedIndex);
    const previousOverride = deleteDecisionDialog.previousOverride;

    setTranscriptRows((prevRows) => restoreDeletedTranscriptRow(prevRows, restoredRow, restoreIndex, segmentId));
    setManualTranscriptOverrides((prev) =>
      restoreManualOverrideAfterDeleteCancel(prev, segmentId, previousOverride)
    );
    if (deleteDecisionDialog.wasSelected) {
      setSelectedSegmentId(segmentId);
    }
    setDeleteDecisionDialog(EMPTY_DELETE_DIALOG_STATE);
  }, [deleteDecisionDialog]);

  const onConfirmDeleteOnly = useCallback(() => {
    setDeleteDecisionDialog(EMPTY_DELETE_DIALOG_STATE);
  }, []);

  const onConfirmDeleteAndReprocess = useCallback(() => {
    const rowsAfterDelete = Array.isArray(deleteDecisionDialog.rowsAfterDelete)
      ? deleteDecisionDialog.rowsAfterDelete
      : [];
    setDeleteDecisionDialog(EMPTY_DELETE_DIALOG_STATE);
    if (!rowsAfterDelete.length) {
      eventStoreRef.current = new Map();
      setEvents([]);
      setSnapshot(EMPTY_SNAPSHOT);
      return;
    }
    void reprocessPipelineFromTranscriptRows(rowsAfterDelete);
  }, [deleteDecisionDialog.rowsAfterDelete, reprocessPipelineFromTranscriptRows]);

  useEffect(() => {
    if (!deleteDecisionDialog.open) {
      return undefined;
    }
    const onKeyDown = (event) => {
      if (event.key !== "Escape") {
        return;
      }
      event.preventDefault();
      onCancelDeleteSegment();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [deleteDecisionDialog.open, onCancelDeleteSegment]);

  const tick = useCallback(async () => {
    const current = stateRef.current;
    if (!current || inFlightRef.current || current.status !== "running") {
      return;
    }

    if (current.inputMode === "audio") {
      const path = String(current.audioPath || "").trim();
      if (!path) {
        setLastError("Audio mode requires a non-empty audio path.");
        setStatus("error");
        return;
      }

      inFlightRef.current = true;
      try {
        const transcribeResp = await apiRef.current.transcribeStructured({
          session_id: current.sessionId,
          audio_path: path,
          start_at_sec: Number(current.audioCursorSec) || 0,
          audio_window_sec: Number(current.audioWindowSec) || 6,
          split_sentences: true,
          incremental: true,
          reset: false,
        });

        await hydrateFromTranscribeResponse(transcribeResp, current.sessionId, current.eventEngine, {
          allowEmpty: true,
          isAudioWindowMode: true,
          reconcileLookbackWindows: current.reconcileLookbackWindows,
          llmUpdateIntervalWindows: current.llmUpdateIntervalWindows,
        });

        const stream = transcribeResp?.debug?.asr?.stream || {};
        const nextStart = Number(stream.next_start_sec);
        const fallbackNext = (Number(current.audioCursorSec) || 0) + (Number(current.audioWindowSec) || 6);
        const resolvedNext = Number.isFinite(nextStart) ? nextStart : fallbackNext;
        const hasMore = stream.has_more === true;

        setAudioCursorSec(resolvedNext);
        setCursor((prev) => prev + 1);
        setLastError("");

        if (!hasMore || resolvedNext <= (Number(current.audioCursorSec) || 0)) {
          setStatus("completed");
        }
      } catch (error) {
        setLastError(String(error?.message || error));
        setStatus("error");
      } finally {
        inFlightRef.current = false;
      }
      return;
    }

    if (current.cursor >= current.streamSegments.length) {
      setStatus("completed");
      return;
    }

    const segment = current.streamSegments[current.cursor];
    const nextCursor = current.cursor + 1;
    inFlightRef.current = true;
    try {
      const transcriptResp = await apiRef.current.transcribeStructured({
        session_id: current.sessionId,
        segments: [
          {
            start: segment.start,
            end: segment.end,
            text: segment.text,
            avg_logprob: null,
            no_speech_prob: null,
          },
        ],
        turns: [
          {
            start: segment.start,
            end: segment.end,
            speaker: segment.speaker || "other",
          },
        ],
        split_sentences: true,
        incremental: true,
        reset: false,
      });

      await hydrateFromTranscribeResponse(transcriptResp, current.sessionId, current.eventEngine);

      setLastError("");
      setCursor(nextCursor);
      if (nextCursor >= current.streamSegments.length) {
        setStatus("completed");
      }
    } catch (error) {
      setLastError(String(error?.message || error));
      setStatus("error");
    } finally {
      inFlightRef.current = false;
    }
  }, [hydrateFromTranscribeResponse]);

  useEffect(() => {
    if (status !== "running") {
      return undefined;
    }

    void tick();
    const timerId = window.setInterval(() => {
      void tick();
    }, intervalMs);
    return () => window.clearInterval(timerId);
  }, [status, intervalMs, tick]);

  useEffect(() => {
    if (!selectedSegmentId) {
      return;
    }
    const node = document.querySelector(`[data-segment-id="${selectedSegmentId}"]`);
    if (!node) {
      return;
    }
    node.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [selectedSegmentId, transcriptRows]);

  useEffect(() => {
    if (!selectedEventKey) {
      return;
    }
    const exists = events.some((item) => buildEventStoreKey(item) === selectedEventKey);
    if (!exists) {
      setSelectedEventKey("");
    }
  }, [events, selectedEventKey]);

  useEffect(() => {
    if (!selectedEventKey) {
      return;
    }
    const host = document.getElementById("timeline-panel");
    if (!host) {
      return;
    }
    const nodes = host.querySelectorAll("[data-event-key]");
    for (const node of nodes) {
      const primaryKey = String(node.getAttribute("data-event-key") || "");
      const mergedKeys = String(node.getAttribute("data-event-keys") || "")
        .split("\n")
        .map((item) => item.trim())
        .filter(Boolean);
      const matches = primaryKey === selectedEventKey || mergedKeys.includes(selectedEventKey);
      if (!matches) {
        continue;
      }
      node.scrollIntoView({ behavior: "smooth", block: "center" });
      break;
    }
  }, [events, selectedEventKey]);

  const stopPaneResize = useCallback(() => {
    paneDragRef.current = null;
    setActiveSplitter(-1);
    document.body.classList.remove("is-pane-resizing");
    window.removeEventListener("pointermove", onPaneResizeMove);
    window.removeEventListener("pointerup", stopPaneResize);
    window.removeEventListener("pointercancel", stopPaneResize);
  }, []);

  const onPaneResizeMove = useCallback((event) => {
    const drag = paneDragRef.current;
    if (!drag) {
      return;
    }

    const deltaX = event.clientX - drag.startX;
    const next = [...drag.startWidths];
    const leftIdx = drag.splitterIndex;
    const rightIdx = drag.splitterIndex + 1;
    const totalPairPx = drag.leftStartPx + drag.rightStartPx;

    const leftMinPx = TRANSCRIPT_PANEL_MIN_WIDTHS[leftIdx] || 120;
    const rightMinPx = TRANSCRIPT_PANEL_MIN_WIDTHS[rightIdx] || 120;
    const leftLowerBound = leftMinPx;
    const leftUpperBound = Math.max(leftLowerBound, totalPairPx - rightMinPx);
    const leftPx = clamp(drag.leftStartPx + deltaX, leftLowerBound, leftUpperBound);
    const rightPx = totalPairPx - leftPx;

    next[leftIdx] = (leftPx / drag.totalPanelWidthPx) * 100;
    next[rightIdx] = (rightPx / drag.totalPanelWidthPx) * 100;
    setTranscriptPaneWidths(next);
  }, []);

  const startPaneResize = useCallback((splitterIndex, clientX) => {
    if (window.matchMedia("(max-width: 1100px)").matches) {
      return;
    }
    const host = transcriptLayoutRef.current;
    if (!host) {
      return;
    }
    const rect = host.getBoundingClientRect();
    const totalSplitterWidth = TRANSCRIPT_SPLITTER_WIDTH_PX * 2;
    const totalPanelWidthPx = Math.max(1, rect.width - totalSplitterWidth);
    const startWidths = transcriptPaneWidths.slice(0, 3);
    const leftStartPx = (startWidths[splitterIndex] / 100) * totalPanelWidthPx;
    const rightStartPx = (startWidths[splitterIndex + 1] / 100) * totalPanelWidthPx;

    paneDragRef.current = {
      splitterIndex,
      startX: clientX,
      startWidths,
      totalPanelWidthPx,
      leftStartPx,
      rightStartPx,
    };
    setActiveSplitter(splitterIndex);
    document.body.classList.add("is-pane-resizing");
    window.addEventListener("pointermove", onPaneResizeMove);
    window.addEventListener("pointerup", stopPaneResize);
    window.addEventListener("pointercancel", stopPaneResize);
  }, [onPaneResizeMove, stopPaneResize, transcriptPaneWidths]);

  useEffect(() => () => {
    document.body.classList.remove("is-pane-resizing");
    window.removeEventListener("pointermove", onPaneResizeMove);
    window.removeEventListener("pointerup", stopPaneResize);
    window.removeEventListener("pointercancel", stopPaneResize);
  }, [onPaneResizeMove, stopPaneResize]);

  const stopClip = useCallback(() => {
    const audio = clipAudioRef.current;
    if (audio) {
      audio.pause();
    }
    clipPlaybackRef.current = {
      clipKey: "",
      t0: 0,
      t1: 0,
      audioUrl: clipPlaybackRef.current.audioUrl || "",
    };
    setPlayingClipKey("");
  }, []);

  const playClip = useCallback(async ({ clipKey, t0, t1 }) => {
    if (!canPlayAudioClips || !audioFileUrl) {
      return;
    }
    const resolvedClipKey = String(clipKey || "").trim();
    if (!resolvedClipKey) {
      return;
    }
    const startSec = Math.max(0, Number(t0) || 0);
    const rawEnd = Number(t1);
    const endSec = Number.isFinite(rawEnd) && rawEnd > startSec ? rawEnd : startSec + 0.2;

    if (!clipAudioRef.current) {
      const sharedAudio = new Audio();
      sharedAudio.preload = "auto";
      sharedAudio.addEventListener("timeupdate", () => {
        const activeClip = clipPlaybackRef.current;
        if (!activeClip.clipKey) {
          return;
        }
        if (sharedAudio.currentTime >= Number(activeClip.t1 || 0)) {
          sharedAudio.pause();
          setPlayingClipKey("");
          clipPlaybackRef.current = {
            ...activeClip,
            clipKey: "",
          };
        }
      });
      sharedAudio.addEventListener("ended", () => {
        setPlayingClipKey("");
        clipPlaybackRef.current = {
          ...clipPlaybackRef.current,
          clipKey: "",
        };
      });
      clipAudioRef.current = sharedAudio;
    }

    const audio = clipAudioRef.current;
    if (clipPlaybackRef.current.clipKey && clipPlaybackRef.current.clipKey !== resolvedClipKey) {
      audio.pause();
    }

    if (audio.src !== audioFileUrl) {
      audio.src = audioFileUrl;
      audio.load();
    }

    clipPlaybackRef.current = {
      clipKey: resolvedClipKey,
      t0: startSec,
      t1: endSec,
      audioUrl: audioFileUrl,
    };

    const startPlayback = () => {
      const duration = Number(audio.duration);
      const resolvedStart = Number.isFinite(duration) && duration > 0
        ? Math.min(startSec, Math.max(0, duration - 0.05))
        : startSec;
      const resolvedEnd = Number.isFinite(duration) && duration > 0
        ? Math.min(endSec, duration)
        : endSec;
      clipPlaybackRef.current = {
        ...clipPlaybackRef.current,
        t0: resolvedStart,
        t1: Math.max(resolvedStart + 0.05, resolvedEnd),
      };
      try {
        audio.currentTime = resolvedStart;
      } catch (_) {
        // Browser may reject seek before metadata is stable.
      }
      const playPromise = audio.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(() => {
          setPlayingClipKey("");
          clipPlaybackRef.current = {
            ...clipPlaybackRef.current,
            clipKey: "",
          };
        });
      }
    };

    if (audio.readyState >= 1) {
      startPlayback();
    } else {
      audio.addEventListener("loadedmetadata", startPlayback, { once: true });
    }
    setPlayingClipKey(resolvedClipKey);
  }, [audioFileUrl, canPlayAudioClips]);

  useEffect(() => {
    if (canPlayAudioClips) {
      return;
    }
    stopClip();
  }, [canPlayAudioClips, stopClip]);

  useEffect(() => () => {
    const audio = clipAudioRef.current;
    if (!audio) {
      return;
    }
    audio.pause();
    audio.src = "";
    clipAudioRef.current = null;
  }, []);

  const startStreaming = useCallback(async () => {
    if (status === "running") {
      return;
    }

    noteDirtyRef.current = false;
    eventStoreRef.current = new Map();
    recentWindowRowsRef.current = [];
    medgemmaAsyncInFlightRef.current = false;
    autoWindowCounterRef.current = 0;
    medgemmaStatsRef.current = createEmptyMedgemmaStats();
    manualTranscriptOverridesRef.current = {};
    setManualTranscriptOverrides({});
    setIsTranscriptReprocessing(false);
    const selectedTemplateIds = normalizeTemplateIds(
      noteDepartment,
      noteTemplateIds,
      noteTemplateCatalog,
    );
    if (!selectedTemplateIds.length) {
      setStatus("idle");
      setLastError("No note templates configured for selected department.");
      return;
    }
    setLastError("");
    setSelectedSegmentId("");
    setSelectedEventKey("");
    stopClip();
    setLiveRecordingAudioPath("");
    setLiveArchiveEndSec(0);
    setEvents([]);
    setGeneratedDrafts([]);
    clearNotePollTimer();
    setNoteJobId("");
    setNoteGenerationStatus("idle");
    setNoteTemplateStatuses(buildDefaultTemplateStatuses(noteDepartment, noteTemplateCatalog));
    setActiveDraftTemplateId(selectedTemplateIds[0] || "");

    if (inputMode === "live") {
      setStatus("idle");
      setLastError("Use Start Mic to run Live Transcription.");
      return;
    }

    if (inputMode === "audio") {
      const path = String(audioPath || "").trim();
      const windowSec = Number(audioWindowSec);
      if (!path) {
        setStatus("idle");
        setLastError("Audio mode requires a non-empty audio path.");
        return;
      }
      if (!Number.isFinite(windowSec) || windowSec <= 0) {
        setStatus("idle");
        setLastError("Audio window must be > 0 seconds.");
        return;
      }

      try {
        await apiRef.current.transcribeStructured({
          session_id: sessionId,
          segments: [],
          turns: [],
          split_sentences: true,
          incremental: true,
          reset: true,
        });
      } catch (error) {
        setStatus("error");
        setLastError(String(error?.message || error));
        return;
      }

      setPipelineDebug(EMPTY_PIPELINE_DEBUG);
      setLiveRecordingAudioPath("");
      setLiveArchiveEndSec(0);
      setStreamSegments([]);
      setCursor(0);
      setAudioCursorSec(0);
      setAudioDurationSec(null);
      setStatus("running");
      return;
    }

    const built = buildSimulationSegments(streamText, 0);
    if (!built.length) {
      setStatus("idle");
      setLastError("Text input is empty.");
      return;
    }

    try {
      await apiRef.current.transcribeStructured({
        session_id: sessionId,
        segments: [],
        turns: [],
        split_sentences: true,
        incremental: true,
        reset: true,
      });
    } catch (error) {
      setStatus("error");
      setLastError(String(error?.message || error));
      return;
    }

    setPipelineDebug(EMPTY_PIPELINE_DEBUG);
    setLiveRecordingAudioPath("");
    setLiveArchiveEndSec(0);
    setAudioCursorSec(0);
    setAudioDurationSec(null);
    setStreamSegments(built);
    setCursor(0);
    setStatus("running");
  }, [
    status,
    inputMode,
    audioPath,
    audioWindowSec,
    noteDepartment,
    noteTemplateCatalog,
    noteTemplateIds,
    sessionId,
    stopClip,
    streamText,
    clearNotePollTimer,
  ]);

  const stopStreaming = useCallback(() => {
    setStatus((prev) => {
      if (prev === "idle" || prev === "completed" || prev === "error") {
        return prev;
      }
      return "stopped";
    });
  }, []);

  const resetSession = useCallback(async () => {
    setStatus("idle");
    setCursor(0);
    setStreamSegments([]);
    setAudioCursorSec(0);
    setAudioDurationSec(null);
    setTranscriptRows([]);
    manualTranscriptOverridesRef.current = {};
    setManualTranscriptOverrides({});
    setIsTranscriptReprocessing(false);
    setEvents([]);
    setSelectedSegmentId("");
    setSelectedEventKey("");
    stopClip();
    setLiveRecordingAudioPath("");
    setLiveArchiveEndSec(0);
    setSnapshot(EMPTY_SNAPSHOT);
    setNoteTextAndResetHistory("");
    setCitations([]);
    setGeneratedDrafts([]);
    clearNotePollTimer();
    setNoteJobId("");
    setNoteGenerationStatus("idle");
    setNoteTemplateStatuses(buildDefaultTemplateStatuses(noteDepartment, noteTemplateCatalog));
    setActiveDraftTemplateId(
      normalizeTemplateIds(noteDepartment, noteTemplateIds, noteTemplateCatalog)[0] || "",
    );
    setPipelineDebug(EMPTY_PIPELINE_DEBUG);
    setLastError("");
    noteDirtyRef.current = false;
    eventStoreRef.current = new Map();
    recentWindowRowsRef.current = [];
    medgemmaAsyncInFlightRef.current = false;
    autoWindowCounterRef.current = 0;
    medgemmaStatsRef.current = createEmptyMedgemmaStats();
    micQueueRef.current = [];
    micInFlightChunkRef.current = null;
    await clearPendingMicChunksForSession(sessionId);
    try {
      await apiRef.current.transcribeStructured({
        session_id: sessionId,
        segments: [],
        turns: [],
        split_sentences: true,
        incremental: true,
        reset: true,
      });
    } catch (error) {
      setLastError(String(error?.message || error));
    }
  }, [clearNotePollTimer, noteDepartment, noteTemplateCatalog, noteTemplateIds, sessionId, stopClip, setNoteTextAndResetHistory]);

  const uploadAudioFile = useCallback(async (file, options = {}) => {
    const response = await apiRef.current.uploadAudioFile(file);
    const nextPath = String(response?.audio_path || "").trim();
    if (!nextPath) {
      throw new Error("Upload succeeded but backend returned empty audio_path.");
    }
    setAudioPath(nextPath);
    const localPathHint = String(options?.localPathHint || "").trim();
    const nativePath = String(file?.path || "").trim();
    const relativePath = String(file?.webkitRelativePath || "").trim();
    const fallbackName = String(file?.name || "").trim();
    const displayPath = localPathHint || nativePath || relativePath || fallbackName;
    if (displayPath) {
      setAudioPathDisplay(displayPath);
    }
    setSelectedAudioSource(LOCAL_AUDIO_SOURCE_VALUE);
    setLastError("");
    return response;
  }, []);

  const onAudioSourceChange = useCallback((nextValue) => {
    const normalized = String(nextValue || "").trim();
    if (!normalized || normalized === LOCAL_AUDIO_SOURCE_VALUE) {
      setSelectedAudioSource(LOCAL_AUDIO_SOURCE_VALUE);
      setAudioPath("");
      setAudioPathDisplay("");
      return;
    }
    setSelectedAudioSource(normalized);
    setAudioPath(normalized);
    setAudioPathDisplay("");
  }, []);

  const clearMicReconnectTimer = useCallback(() => {
    const timer = micReconnectTimerRef.current;
    if (typeof timer === "number") {
      window.clearTimeout(timer);
    }
    micReconnectTimerRef.current = null;
  }, []);

  const clearMicStopDrainTimer = useCallback(() => {
    const timer = micStopDrainTimerRef.current;
    if (typeof timer === "number") {
      window.clearInterval(timer);
    }
    micStopDrainTimerRef.current = null;
  }, []);

  const triggerMicFileFallback = useCallback(async (reason) => {
    if (micFallbackTriggeredRef.current) {
      return;
    }
    micFallbackTriggeredRef.current = true;
    micShouldRunRef.current = false;
    clearMicReconnectTimer();
    clearMicStopDrainTimer();

    const recorder = micRecorderRef.current;
    micRecorderRef.current = null;
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch (_) {
        // noop
      }
    }

    const stream = micStreamRef.current;
    micStreamRef.current = null;
    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
    }

    const archived = [...micArchiveRef.current];
    if (!archived.length) {
      setMicStatus("error");
      setPipelineDebug((prev) => ({
        ...prev,
        live_audio_status: "degraded_no_buffer",
      }));
      setLastError(`Realtime link failed (${reason}); no buffered audio available for fallback.`);
      return;
    }

    const sorted = archived.sort((a, b) => Number(a.seq || 0) - Number(b.seq || 0));
    const mimeType = micActiveMimeTypeRef.current || "audio/webm";
    const ext = mimeTypeToAudioExt(mimeType);
    const parts = sorted.map((item) => base64ToUint8Array(String(item.data_b64 || "")));
    const fallbackBlob = new Blob(parts, { type: mimeType });
    let fallbackFile;
    try {
      fallbackFile = new File(
        [fallbackBlob],
        `live_fallback_${sessionId}_${Date.now()}${ext}`,
        { type: mimeType },
      );
    } catch (_) {
      setMicStatus("error");
      setPipelineDebug((prev) => ({
        ...prev,
        live_audio_status: "degraded_fallback_failed",
      }));
      setLastError("Browser does not support File construction for fallback upload.");
      return;
    }

    setMicStatus("degraded");
    setPipelineDebug((prev) => ({
      ...prev,
      live_audio_status: "degraded_uploading_fallback",
    }));
    try {
      const uploadResp = await apiRef.current.uploadAudioFile(fallbackFile);
      const fallbackPath = String(uploadResp?.audio_path || "").trim();
      if (!fallbackPath) {
        throw new Error("Fallback upload succeeded but audio_path is empty.");
      }
      await clearPendingMicChunksForSession(sessionId);
      micQueueRef.current = [];
      micInFlightChunkRef.current = null;
      micArchiveQueueRef.current = [];
      micArchiveInFlightChunkRef.current = null;
      setInputMode("audio");
      setAudioPath(fallbackPath);
      setAudioPathDisplay(String(fallbackFile?.name || ""));
      setSelectedAudioSource(LOCAL_AUDIO_SOURCE_VALUE);
      setStatus("idle");
      setMicStatus("stopped");
      setLiveRecordingAudioPath("");
      setPipelineDebug((prev) => ({
        ...prev,
        live_audio_status: "degraded_file_fallback_ready",
      }));
      setLastError(
        `Realtime connection failed (${reason}). Switched to audio_path fallback file. Click Start to continue.`,
      );
    } catch (error) {
      setMicStatus("error");
      setPipelineDebug((prev) => ({
        ...prev,
        live_audio_status: "degraded_fallback_failed",
      }));
      setLastError(`Realtime fallback upload failed: ${String(error?.message || error)}`);
    }
  }, [clearMicReconnectTimer, clearMicStopDrainTimer, sessionId]);

  const flushMicQueue = useCallback(() => {
    const socket = micSocketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    if (!micInFlightChunkRef.current) {
      const nextChunk = micQueueRef.current[0];
      if (nextChunk) {
        micInFlightChunkRef.current = nextChunk;
        socket.send(
          JSON.stringify({
            type: "audio_chunk",
            seq: nextChunk.seq,
            mime_type: nextChunk.mime_type,
            data_b64: nextChunk.data_b64,
            window_start_sec: nextChunk.window_start_sec,
            window_duration_sec: nextChunk.window_duration_sec,
          }),
        );
      }
    }
    if (!micArchiveInFlightChunkRef.current) {
      const nextArchive = micArchiveQueueRef.current[0];
      if (nextArchive) {
        micArchiveInFlightChunkRef.current = nextArchive;
        socket.send(
          JSON.stringify({
            type: "audio_chunk_archive",
            seq: nextArchive.seq,
            mime_type: nextArchive.mime_type,
            data_b64: nextArchive.data_b64,
          }),
        );
      }
    }
  }, []);

  const connectMicSocket = useCallback((resetSession) => {
    const wsUrl = toWebSocketUrl(baseUrl, `/ws/audio/live?session_id=${encodeURIComponent(sessionId)}`);
    const socket = new WebSocket(wsUrl);
    micSocketRef.current = socket;

    socket.onopen = () => {
      if (!micShouldRunRef.current) {
        socket.close();
        return;
      }
      clearMicReconnectTimer();
      micReconnectAttemptRef.current = 0;
      const mimeType = micActiveMimeTypeRef.current || "audio/webm";
      socket.send(
        JSON.stringify({
          type: "start",
          mime_type: mimeType,
          reset: Boolean(resetSession),
        }),
      );
      setMicStatus("streaming");
      setPipelineDebug((prev) => ({
        ...prev,
        live_audio_status: "streaming",
      }));
      flushMicQueue();
    };

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(String(event.data || "{}"));
        const captureSec = getCaptureSliceSec(stateRef.current?.micCaptureSliceMs);
        if (payload.type === "ack_start") {
          const recordingAudioPath = String(payload.recording_audio_path || "").trim();
          if (recordingAudioPath) {
            setLiveRecordingAudioPath(recordingAudioPath);
          }
          setPipelineDebug((prev) => ({
            ...prev,
            source: "live_audio_ws",
            live_audio_status: String(payload.status || "streaming"),
            live_audio_recording_path: recordingAudioPath || String(prev.live_audio_recording_path || ""),
          }));
          flushMicQueue();
          return;
        }
        if (payload.type === "ack_archive_chunk") {
          const ackSeq =
            typeof payload.seq === "number"
              ? payload.seq
              : (typeof payload.seq === "string" && /^\d+$/.test(payload.seq) ? Number(payload.seq) : null);
          let archiveEndSec = Number(payload.recording_chunks_received || 0) * captureSec;
          if (ackSeq !== null) {
            if (micArchiveInFlightChunkRef.current && micArchiveInFlightChunkRef.current.seq === ackSeq) {
              micArchiveInFlightChunkRef.current = null;
            }
            const index = micArchiveQueueRef.current.findIndex((item) => item.seq === ackSeq);
            if (index >= 0) {
              micArchiveQueueRef.current.splice(index, 1);
            }
            void removePendingMicChunk(sessionId, ackSeq, { kind: "archive_raw" });
            archiveEndSec = Math.max(archiveEndSec, (ackSeq + 1) * captureSec);
          }
          setLiveArchiveEndSec((prev) => Math.max(Number(prev) || 0, archiveEndSec));
          const recordingAudioPath = String(payload.recording_audio_path || "").trim();
          if (recordingAudioPath) {
            setLiveRecordingAudioPath(recordingAudioPath);
          }
          setPipelineDebug((prev) => ({
            ...prev,
            source: "live_audio_ws",
            live_audio_recording_path: recordingAudioPath || String(prev.live_audio_recording_path || ""),
            live_audio_archive_end_sec: formatArchiveEndSec(
              Math.max(parseArchiveEndSec(prev.live_audio_archive_end_sec), archiveEndSec),
            ),
          }));
          flushMicQueue();
          return;
        }
        if (payload.type === "ack_chunk") {
          const ackSeq =
            typeof payload.seq === "number"
              ? payload.seq
              : (typeof payload.seq === "string" && /^\d+$/.test(payload.seq) ? Number(payload.seq) : null);
          if (ackSeq !== null) {
            if (micInFlightChunkRef.current && micInFlightChunkRef.current.seq === ackSeq) {
              micInFlightChunkRef.current = null;
            }
            const index = micQueueRef.current.findIndex((item) => item.seq === ackSeq);
            if (index >= 0) {
              micQueueRef.current.splice(index, 1);
            }
            void removePendingMicChunk(sessionId, ackSeq, { kind: "asr_window" });
          }

          const recordingAudioPath = String(payload.recording_audio_path || "").trim();
          if (recordingAudioPath) {
            setLiveRecordingAudioPath(recordingAudioPath);
          }

          setPipelineDebug((prev) => ({
            ...prev,
            source: "live_audio_ws",
            asr_status: payload.asr_mode
              ? `${String(payload.asr_status || prev.asr_status || "")} (${String(payload.asr_mode)})`
              : String(payload.asr_status || prev.asr_status || ""),
            diarization_status: payload.diarization_mode
              ? `${String(payload.diarization_status || prev.diarization_status || "")} (${String(payload.diarization_mode)})`
              : String(payload.diarization_status || prev.diarization_status || ""),
            sentence_role_split: typeof payload.sentence_role_split === "object"
              ? extractSentenceRoleSplit(payload.sentence_role_split || {})
              : String(payload.sentence_role_split || prev.sentence_role_split || ""),
            live_audio_status: "streaming",
            live_audio_chunks: String(payload.chunks_received || 0),
            live_audio_bytes: String(payload.bytes_received || 0),
            live_audio_recording_path: recordingAudioPath || String(prev.live_audio_recording_path || ""),
          }));

          if (payload.asr_status === "error" && payload.asr_error) {
            setLastError(`Live audio ASR error: ${String(payload.asr_error)}`);
          }

          const current = stateRef.current;
          if (current) {
            const liveResponse = {
              status: "streaming",
              utterances: Array.isArray(payload.utterances) ? payload.utterances : [],
              new_utterances: Array.isArray(payload.new_utterances_payload) ? payload.new_utterances_payload : [],
              transcript_text: String(payload.transcript_text || ""),
              debug: {
                asr_status: String(payload.asr_status || ""),
                asr_mode: String(payload.asr_mode || ""),
                diarization_status: String(payload.diarization_status || ""),
                diarization_mode: String(payload.diarization_mode || ""),
                sentence_role_split: payload.sentence_role_split || {},
                recording_audio_path: recordingAudioPath,
              },
            };
            if (liveResponse.new_utterances.length || liveResponse.utterances.length) {
              micHydrateChainRef.current = micHydrateChainRef.current
                .then(() =>
                  hydrateFromLiveTranscriptResponse(
                    liveResponse,
                    current.sessionId,
                    current.eventEngine,
                  ))
                .catch((error) => {
                  setLastError(String(error?.message || error));
                });
            }
          }

          flushMicQueue();
          return;
        }
        if (payload.type === "ack_stop") {
          const recordingAudioPath = String(payload.recording_audio_path || "").trim();
          if (recordingAudioPath) {
            setLiveRecordingAudioPath(recordingAudioPath);
          }
          setPipelineDebug((prev) => ({
            ...prev,
            live_audio_status: "stopped",
            live_audio_recording_path: recordingAudioPath || String(prev.live_audio_recording_path || ""),
          }));
          return;
        }
        if (payload.type === "error") {
          setLastError(`Live audio ws error: ${String(payload.detail || "unknown")}`);
        }
      } catch (_) {
        // ignore non-json messages
      }
    };

    socket.onerror = () => {
      setPipelineDebug((prev) => ({
        ...prev,
        live_audio_status: "socket_error",
      }));
    };

    socket.onclose = () => {
      if (micSocketRef.current === socket) {
        micSocketRef.current = null;
      }
      micInFlightChunkRef.current = null;
      micArchiveInFlightChunkRef.current = null;
      if (micStoppingRef.current) {
        setMicStatus("stopping");
        setPipelineDebug((prev) => ({
          ...prev,
          live_audio_status: "stopping_waiting_drain",
        }));
        return;
      }
      if (!micShouldRunRef.current) {
        clearMicStopDrainTimer();
        micStoppingRef.current = false;
        setMicStatus((prev) => (prev === "idle" ? "idle" : "stopped"));
        setPipelineDebug((prev) => ({
          ...prev,
          live_audio_status: "stopped",
        }));
        return;
      }

      setMicStatus("reconnecting");
      setPipelineDebug((prev) => ({
        ...prev,
        live_audio_status: "reconnecting",
      }));
      const nextAttempt = micReconnectAttemptRef.current + 1;
      micReconnectAttemptRef.current = nextAttempt;
      if (nextAttempt > MAX_MIC_RECONNECT_ATTEMPTS) {
        void triggerMicFileFallback(`max reconnect attempts exceeded (${MAX_MIC_RECONNECT_ATTEMPTS})`);
        return;
      }
      const delayMs = Math.min(10000, 500 * (2 ** Math.min(nextAttempt, 5)));
      clearMicReconnectTimer();
      micReconnectTimerRef.current = window.setTimeout(() => {
        connectMicSocket(false);
      }, delayMs);
    };
  }, [
    baseUrl,
    clearMicReconnectTimer,
    clearMicStopDrainTimer,
    flushMicQueue,
    hydrateFromLiveTranscriptResponse,
    sessionId,
    triggerMicFileFallback,
  ]);

  const stopMicStreaming = useCallback(() => {
    const hasPendingChunks = () => (
      micQueueRef.current.length > 0
      || Boolean(micInFlightChunkRef.current)
      || micArchiveQueueRef.current.length > 0
      || Boolean(micArchiveInFlightChunkRef.current)
    );
    const finalizeStop = (timedOut = false) => {
      clearMicStopDrainTimer();
      const socket = micSocketRef.current;
      micSocketRef.current = null;
      if (socket) {
        try {
          if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "stop" }));
          }
          socket.close();
        } catch (_) {
          // noop
        }
      }

      micChunkSeqRef.current = 0;
      micQueueRef.current = [];
      micInFlightChunkRef.current = null;
      micArchiveQueueRef.current = [];
      micArchiveInFlightChunkRef.current = null;
      micReconnectAttemptRef.current = 0;
      micActiveMimeTypeRef.current = "";
      micHydrateChainRef.current = Promise.resolve();
      micArchiveRef.current = [];
      micFallbackTriggeredRef.current = false;
      micRawChunkBufferRef.current = [];
      micWindowSeqRef.current = 0;
      micStoppingRef.current = false;
      setMicStatus((prev) => (prev === "idle" ? "idle" : "stopped"));
      setPipelineDebug((prev) => ({
        ...prev,
        live_audio_status: timedOut ? "stopped_with_pending_queue" : "stopped",
      }));
      if (timedOut) {
        setLastError((prev) => (
          prev || "Stop Mic timed out before queue drain; pending chunks stay persisted for next resume."
        ));
      }
    };

    micShouldRunRef.current = false;
    micStoppingRef.current = true;
    clearMicReconnectTimer();
    clearMicStopDrainTimer();

    const recorder = micRecorderRef.current;
    micRecorderRef.current = null;
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch (_) {
        // noop
      }
    }

    const stream = micStreamRef.current;
    micStreamRef.current = null;
    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
    }

    setMicStatus("stopping");
    setPipelineDebug((prev) => ({
      ...prev,
      live_audio_status: "stopping_drain_queue",
    }));

    if (!hasPendingChunks()) {
      finalizeStop(false);
      return;
    }

    flushMicQueue();
    const startedAt = Date.now();
    micStopDrainTimerRef.current = window.setInterval(() => {
      if (!micStoppingRef.current) {
        clearMicStopDrainTimer();
        return;
      }
      if (!hasPendingChunks()) {
        finalizeStop(false);
        return;
      }
      const socket = micSocketRef.current;
      if (
        socket
        && socket.readyState === WebSocket.OPEN
        && !micInFlightChunkRef.current
        && !micArchiveInFlightChunkRef.current
      ) {
        flushMicQueue();
      }
      if ((Date.now() - startedAt) >= MIC_STOP_DRAIN_TIMEOUT_MS) {
        finalizeStop(true);
      }
    }, MIC_STOP_DRAIN_POLL_MS);
  }, [clearMicReconnectTimer, clearMicStopDrainTimer, flushMicQueue]);

  const startMicStreaming = useCallback(async () => {
    if (!micSupported) {
      setMicStatus("unsupported");
      setLastError("Microphone capture is not supported in this browser.");
      return;
    }
    if (micRecorderRef.current && micRecorderRef.current.state !== "inactive") {
      setMicStatus("streaming");
      return;
    }

    try {
      eventStoreRef.current = new Map();
      recentWindowRowsRef.current = [];
      medgemmaAsyncInFlightRef.current = false;
      autoWindowCounterRef.current = 0;
      medgemmaStatsRef.current = createEmptyMedgemmaStats();
      manualTranscriptOverridesRef.current = {};
      setManualTranscriptOverrides({});
      setIsTranscriptReprocessing(false);
      setTranscriptRows([]);
      setEvents([]);
      setSnapshot(EMPTY_SNAPSHOT);
      setSelectedSegmentId("");
      setSelectedEventKey("");
      stopClip();
      setLiveRecordingAudioPath("");
      setLiveArchiveEndSec(0);
      setPipelineDebug(EMPTY_PIPELINE_DEBUG);
      micHydrateChainRef.current = Promise.resolve();
      micQueueRef.current = [];
      micInFlightChunkRef.current = null;
      micArchiveQueueRef.current = [];
      micArchiveInFlightChunkRef.current = null;
      micChunkSeqRef.current = 0;
      micReconnectAttemptRef.current = 0;
      micShouldRunRef.current = true;
      micStoppingRef.current = false;
      micArchiveRef.current = [];
      micFallbackTriggeredRef.current = false;
      micRawChunkBufferRef.current = [];
      micWindowSeqRef.current = 0;
      clearMicReconnectTimer();
      clearMicStopDrainTimer();

      await cleanupPendingMicChunks(sessionId, {
        ttlMs: MIC_PERSIST_TTL_MS,
        maxRecords: MIC_PERSIST_MAX_CHUNKS,
      });
      const restoredPendingChunks = await listPendingMicChunks(sessionId, { kind: "asr_window" });
      const restoredArchivePendingChunks = await listPendingMicChunks(sessionId, { kind: "archive_raw" });
      if (restoredPendingChunks.length) {
        const restoredQueue = restoredPendingChunks.map((item) => ({
          seq: Number(item.seq || 0),
          mime_type: String(item.mime_type || "audio/webm"),
          data_b64: String(item.data_b64 || ""),
          window_start_sec: Number(item.window_start_sec || 0),
          window_duration_sec: Number(item.window_duration_sec || 0),
        }));
        micQueueRef.current = restoredQueue;
        const maxRestoredSeq = restoredQueue.reduce(
          (maxSeq, item) => Math.max(maxSeq, Number(item.seq) || 0),
          -1,
        );
        micWindowSeqRef.current = maxRestoredSeq + 1;
        setPipelineDebug((prev) => ({
          ...prev,
          live_audio_status: `restored_${restoredQueue.length}_pending_chunks`,
        }));
      }
      if (restoredArchivePendingChunks.length) {
        const restoredArchiveQueue = restoredArchivePendingChunks.map((item) => ({
          seq: Number(item.seq || 0),
          mime_type: String(item.mime_type || "audio/webm"),
          data_b64: String(item.data_b64 || ""),
        }));
        micArchiveQueueRef.current = restoredArchiveQueue;
        micArchiveRef.current = restoredArchiveQueue
          .map((item) => ({
            seq: item.seq,
            mime_type: item.mime_type,
            data_b64: item.data_b64,
          }))
          .slice(-MAX_MIC_ARCHIVE_CHUNKS);
        const maxArchiveSeq = restoredArchiveQueue.reduce(
          (maxSeq, item) => Math.max(maxSeq, Number(item.seq) || 0),
          -1,
        );
        if (maxArchiveSeq >= 0) {
          setLiveArchiveEndSec((maxArchiveSeq + 1) * getCaptureSliceSec(micCaptureSliceMs));
        }
        micChunkSeqRef.current = Math.max(micChunkSeqRef.current, maxArchiveSeq + 1);
      }

      setMicStatus("requesting_permission");
      setLastError("");

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;

      let recorder;
      const mimeType = pickSupportedRecorderMimeType();
      try {
        recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      } catch (error) {
        setMicStatus("error");
        setLastError(`MediaRecorder init failed: ${String(error?.message || error)}`);
        return;
      }
      micRecorderRef.current = recorder;
      const activeMimeType = recorder.mimeType || mimeType || "audio/webm";
      micActiveMimeTypeRef.current = activeMimeType;
      const captureSliceMs = Math.max(200, Math.floor(Number(micCaptureSliceMs) || DEFAULT_MIC_CAPTURE_SLICE_MS));
      const captureSec = captureSliceMs / 1000;
      const windowSec = Math.max(captureSec, Number(micAsrWindowSec) || DEFAULT_MIC_ASR_WINDOW_SEC);
      const stepSec = Math.max(captureSec, Number(micAsrStepSec) || DEFAULT_MIC_ASR_STEP_SEC);
      recorder.ondataavailable = async (event) => {
        if (!event.data || event.data.size <= 0 || (!micShouldRunRef.current && !micStoppingRef.current)) {
          return;
        }
        const rawSeq = micChunkSeqRef.current;
        micChunkSeqRef.current += 1;
        const arrayBuffer = await event.data.arrayBuffer();
        const dataB64 = arrayBufferToBase64(arrayBuffer);
        const rawChunk = {
          seq: rawSeq,
          mime_type: activeMimeType,
          data_b64: dataB64,
        };
        micArchiveQueueRef.current.push(rawChunk);
        void persistPendingMicChunk(sessionId, rawChunk, { kind: "archive_raw" });
        if (micArchiveQueueRef.current.length > MAX_MIC_BUFFERED_ARCHIVE_CHUNKS) {
          let droppedArchiveChunk = null;
          if (micArchiveInFlightChunkRef.current && micArchiveQueueRef.current.length > 1) {
            droppedArchiveChunk = micArchiveQueueRef.current.splice(1, 1)[0] || null;
          } else {
            droppedArchiveChunk = micArchiveQueueRef.current.shift() || null;
          }
          if (droppedArchiveChunk && Number.isFinite(Number(droppedArchiveChunk.seq))) {
            void removePendingMicChunk(sessionId, Number(droppedArchiveChunk.seq), { kind: "archive_raw" });
          }
          setLastError("Network unstable: raw archive queue is full, dropped oldest buffered chunk.");
        }
        micArchiveRef.current.push(rawChunk);
        if (micArchiveRef.current.length > MAX_MIC_ARCHIVE_CHUNKS) {
          micArchiveRef.current.shift();
        }
        micRawChunkBufferRef.current.push(rawChunk);
        const stepChunks = Math.max(1, Math.round(stepSec / captureSec));
        const windowChunks = Math.max(1, Math.round(windowSec / captureSec));
        const maxRawBufferChunks = Math.max(windowChunks * 2, windowChunks + stepChunks);
        if (micRawChunkBufferRef.current.length > maxRawBufferChunks) {
          micRawChunkBufferRef.current = micRawChunkBufferRef.current.slice(-maxRawBufferChunks);
        }

        const shouldEmitWindow = ((rawSeq + 1) % stepChunks) === 0;
        if (!shouldEmitWindow) {
          flushMicQueue();
          return;
        }

        const selectedRawChunks = micRawChunkBufferRef.current.slice(-windowChunks);
        if (!selectedRawChunks.length) {
          return;
        }
        const combinedBytes = mergeBase64Chunks(selectedRawChunks.map((item) => String(item.data_b64 || "")));
        const combinedB64 = arrayBufferToBase64(combinedBytes.buffer);
        const sentSeq = micWindowSeqRef.current;
        micWindowSeqRef.current += 1;
        const windowEndSec = (rawSeq + 1) * captureSec;
        const windowDurationSec = selectedRawChunks.length * captureSec;
        const windowStartSec = Math.max(0, windowEndSec - windowDurationSec);
        const queuedWindowChunk = {
          seq: sentSeq,
          mime_type: activeMimeType,
          data_b64: combinedB64,
          window_start_sec: windowStartSec,
          window_duration_sec: windowDurationSec,
        };
        micQueueRef.current.push(queuedWindowChunk);
        void persistPendingMicChunk(sessionId, queuedWindowChunk, { kind: "asr_window" });
        if (micQueueRef.current.length > MAX_MIC_BUFFERED_CHUNKS) {
          let droppedChunk = null;
          if (micInFlightChunkRef.current && micQueueRef.current.length > 1) {
            droppedChunk = micQueueRef.current.splice(1, 1)[0] || null;
          } else {
            droppedChunk = micQueueRef.current.shift() || null;
          }
          if (droppedChunk && Number.isFinite(Number(droppedChunk.seq))) {
            void removePendingMicChunk(sessionId, Number(droppedChunk.seq), { kind: "asr_window" });
          }
          setLastError("Network unstable: mic buffer is full, dropped oldest buffered chunk.");
        }
        flushMicQueue();
      };
      recorder.onerror = (event) => {
        const reason = String(event?.error?.message || "unknown_recorder_error");
        setMicStatus("error");
        setLastError(`Microphone recorder error: ${reason}`);
      };
      recorder.start(captureSliceMs);

      setMicStatus("connecting");
      connectMicSocket(true);
    } catch (error) {
      micShouldRunRef.current = false;
      setMicStatus("error");
      setLastError(`Microphone start failed: ${String(error?.message || error)}`);
    }
  }, [
    clearMicReconnectTimer,
    clearMicStopDrainTimer,
    connectMicSocket,
    flushMicQueue,
    micAsrStepSec,
    micAsrWindowSec,
    micCaptureSliceMs,
    micSupported,
    sessionId,
    stopClip,
  ]);

  useEffect(() => () => {
    stopMicStreaming();
  }, [stopMicStreaming]);

  const exportNote = useCallback((text) => {
    const blob = new Blob([String(text || "")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `evidentia_draft_${sessionId}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }, [sessionId]);

  const emailNote = useCallback((text) => {
    const body = normalizeEmailBody(String(text || "").trim());
    const subject = "Evidentia Draft Note";
    const href =
      `mailto:?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    window.location.href = href;
  }, []);

  const onNoteChange = useCallback((value) => {
    const nextValue = String(value || "");
    noteDirtyRef.current = true;
    setNoteText((prev) => {
      if (nextValue === prev) {
        return prev;
      }
      noteUndoStackRef.current.push(String(prev || ""));
      if (noteUndoStackRef.current.length > NOTE_HISTORY_MAX) {
        noteUndoStackRef.current.shift();
      }
      noteRedoStackRef.current = [];
      return nextValue;
    });
    syncNoteHistoryAvailability();
  }, [syncNoteHistoryAvailability]);

  const onUndoNote = useCallback(() => {
    if (!noteUndoStackRef.current.length) {
      return;
    }
    noteDirtyRef.current = true;
    setNoteText((prev) => {
      const previous = noteUndoStackRef.current.pop();
      if (typeof previous !== "string") {
        return prev;
      }
      noteRedoStackRef.current.push(String(prev || ""));
      if (noteRedoStackRef.current.length > NOTE_HISTORY_MAX) {
        noteRedoStackRef.current.shift();
      }
      return previous;
    });
    syncNoteHistoryAvailability();
  }, [syncNoteHistoryAvailability]);

  const onRedoNote = useCallback(() => {
    if (!noteRedoStackRef.current.length) {
      return;
    }
    noteDirtyRef.current = true;
    setNoteText((prev) => {
      const restored = noteRedoStackRef.current.pop();
      if (typeof restored !== "string") {
        return prev;
      }
      noteUndoStackRef.current.push(String(prev || ""));
      if (noteUndoStackRef.current.length > NOTE_HISTORY_MAX) {
        noteUndoStackRef.current.shift();
      }
      return restored;
    });
    syncNoteHistoryAvailability();
  }, [syncNoteHistoryAvailability]);

  const onNoteKeyDown = useCallback((event) => {
    const withModifier = Boolean(event?.metaKey || event?.ctrlKey);
    if (!withModifier) {
      return;
    }
    const key = String(event?.key || "").toLowerCase();
    if (key === "z" && !event.shiftKey) {
      event.preventDefault();
      onUndoNote();
      return;
    }
    if ((key === "z" && event.shiftKey) || key === "y") {
      event.preventDefault();
      onRedoNote();
    }
  }, [onRedoNote, onUndoNote]);

  const onGenerateDrafts = useCallback(async () => {
    if (NOTE_JOB_ACTIVE_STATES.has(noteGenerationStatus)) {
      return;
    }
    const selectedTemplateIds = normalizeTemplateIds(
      noteDepartment,
      noteTemplateIds,
      noteTemplateCatalog,
    );
    if (!selectedTemplateIds.length) {
      setLastError("No note templates configured for selected department.");
      return;
    }
    const hasSnapshotContext =
      snapshot.problem_list.length || snapshot.risk_flags.length || snapshot.open_questions.length;
    if (!events.length && !hasSnapshotContext) {
      setLastError("No transcript yet. Run Start Session first, then Generate Notes.");
      return;
    }

    setNoteTemplateStatuses(
      selectedTemplateIds.reduce((acc, templateId) => {
        acc[templateId] = "queued";
        return acc;
      }, {}),
    );
    setNoteGenerationStatus("pending");
    setGeneratedDrafts([]);
    setLastError("");
    noteDirtyRef.current = false;

    try {
      const jobResp = await apiRef.current.startNoteDraftJob({
        department: noteDepartment,
        template_ids: selectedTemplateIds,
        patient_identity: patientIdentity,
        patient_basic_info: patientContextText,
        snapshot,
        events,
      });
      setNoteJobId(String(jobResp?.job_id || ""));
      applyNoteDraftJobStatus(jobResp, { respectDirty: false });
    } catch (error) {
      setNoteGenerationStatus("idle");
      setNoteJobId("");
      setNoteTemplateStatuses(buildDefaultTemplateStatuses(noteDepartment, noteTemplateCatalog));
      setLastError(String(error?.message || error));
    }
  }, [
    events,
    noteDepartment,
    noteGenerationStatus,
    noteTemplateCatalog,
    noteTemplateIds,
    patientContextText,
    patientIdentity,
    snapshot,
    applyNoteDraftJobStatus,
  ]);

  const onStopGenerateDrafts = useCallback(async () => {
    const activeJobId = String(noteJobId || "").trim();
    if (!activeJobId || !NOTE_JOB_ACTIVE_STATES.has(noteGenerationStatus)) {
      return;
    }
    setNoteGenerationStatus("stopping");
    try {
      const jobResp = await apiRef.current.stopNoteDraftJob(activeJobId);
      applyNoteDraftJobStatus(jobResp, { respectDirty: true });
      setLastError("");
    } catch (error) {
      setLastError(String(error?.message || error));
    }
  }, [noteGenerationStatus, noteJobId, applyNoteDraftJobStatus]);

  useEffect(() => {
    const activeJobId = String(noteJobId || "").trim();
    if (!activeJobId || !NOTE_JOB_ACTIVE_STATES.has(noteGenerationStatus)) {
      clearNotePollTimer();
      return undefined;
    }

    let cancelled = false;
    const poll = async () => {
      if (cancelled) {
        return;
      }
      try {
        const jobResp = await apiRef.current.getNoteDraftJob(activeJobId);
        if (cancelled) {
          return;
        }
        applyNoteDraftJobStatus(jobResp, { respectDirty: true });
        setLastError("");
        const nextStatus = String(jobResp?.status || "");
        if (!NOTE_JOB_ACTIVE_STATES.has(nextStatus)) {
          clearNotePollTimer();
          return;
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        setLastError(String(error?.message || error));
      }
      notePollTimerRef.current = window.setTimeout(() => {
        void poll();
      }, NOTE_JOB_POLL_MS);
    };

    void poll();
    return () => {
      cancelled = true;
      clearNotePollTimer();
    };
  }, [applyNoteDraftJobStatus, clearNotePollTimer, noteGenerationStatus, noteJobId]);

  useEffect(() => () => {
    clearNotePollTimer();
  }, [clearNotePollTimer]);

  const onDraftTemplateChange = useCallback((templateId) => {
    const nextId = String(templateId || "").trim();
    if (!nextId) {
      return;
    }
    const draft = generatedDrafts.find((item) => String(item?.template_id || "") === nextId);
    if (!draft) {
      return;
    }
    noteDirtyRef.current = false;
    setActiveDraftTemplateId(nextId);
    setNoteTextAndResetHistory(String(draft.note_text || ""));
    setCitations(Array.isArray(draft.citations) ? draft.citations : []);
  }, [generatedDrafts, setNoteTextAndResetHistory]);

  const onDepartmentChange = useCallback((value) => {
    const nextDepartment = normalizeDepartment(value, noteTemplateCatalog);
    const nextTemplates = normalizeTemplateIds(nextDepartment, [], noteTemplateCatalog);
    noteDirtyRef.current = false;
    clearNotePollTimer();
    setNoteJobId("");
    setNoteGenerationStatus("idle");
    setNoteDepartment(nextDepartment);
    setNoteTemplateIds(nextTemplates);
    setNoteTemplateStatuses(buildDefaultTemplateStatuses(nextDepartment, noteTemplateCatalog));
    setActiveDraftTemplateId(nextTemplates[0] || "");
    setSelectedTemplateEditorId(nextTemplates[0] || "");
    setTemplateEditorDoc(null);
    templateUndoStackRef.current = [];
    templateRedoStackRef.current = [];
    setCanUndoTemplate(false);
    setCanRedoTemplate(false);
    setTemplateEditorDirty(false);
    setTemplateEditorMessage("");
    setGeneratedDrafts([]);
    setNoteTextAndResetHistory("");
    setCitations([]);
  }, [clearNotePollTimer, noteTemplateCatalog, setNoteTextAndResetHistory]);

  const onNoteTemplatesChange = useCallback((values) => {
    const nextTemplates = normalizeTemplateIds(noteDepartment, values, noteTemplateCatalog);
    noteDirtyRef.current = false;
    clearNotePollTimer();
    setNoteJobId("");
    setNoteGenerationStatus("idle");
    setNoteTemplateIds(nextTemplates);
    setNoteTemplateStatuses((prev) => ({
      ...buildDefaultTemplateStatuses(noteDepartment, noteTemplateCatalog),
      ...(prev || {}),
    }));
  }, [clearNotePollTimer, noteDepartment, noteTemplateCatalog]);

  const onToggleTemplateDepartment = useCallback((departmentId) => {
    const nextDepartment = String(departmentId || "").trim();
    if (!nextDepartment) {
      return;
    }
    setTemplateExpandedDepartments((prev) => ({
      ...prev,
      [nextDepartment]: !prev[nextDepartment],
    }));
    if (nextDepartment !== noteDepartment) {
      setNoteDepartment(nextDepartment);
    }
  }, [noteDepartment]);

  const onSelectTemplateFromDepartment = useCallback((departmentId, templateId) => {
    const nextDepartment = String(departmentId || "").trim();
    const nextTemplateId = String(templateId || "").trim();
    if (!nextDepartment || !nextTemplateId) {
      return;
    }
    if (nextDepartment !== noteDepartment) {
      setNoteDepartment(nextDepartment);
    }
    if (nextTemplateId !== selectedTemplateEditorId) {
      setSelectedTemplateEditorId(nextTemplateId);
    }
    setTemplateEditorMessage("");
    setTemplateEditorDirty(false);
  }, [noteDepartment, selectedTemplateEditorId]);

  const onTemplateNameChange = useCallback((value) => {
    setTemplateEditorDoc((prev) => {
      if (!prev) {
        return prev;
      }
      templateUndoStackRef.current.push(cloneTemplateEditorDocument(prev));
      if (templateUndoStackRef.current.length > NOTE_HISTORY_MAX) {
        templateUndoStackRef.current.shift();
      }
      templateRedoStackRef.current = [];
      syncTemplateHistoryAvailability();
      return { ...prev, template_name: String(value || "") };
    });
    setTemplateEditorDirty(true);
    setTemplateEditorMessage("");
  }, [syncTemplateHistoryAvailability]);

  const onTemplateTextChange = useCallback((value) => {
    setTemplateEditorDoc((prev) => {
      if (!prev) {
        return prev;
      }
      templateUndoStackRef.current.push(cloneTemplateEditorDocument(prev));
      if (templateUndoStackRef.current.length > NOTE_HISTORY_MAX) {
        templateUndoStackRef.current.shift();
      }
      templateRedoStackRef.current = [];
      syncTemplateHistoryAvailability();
      return { ...prev, template_text: String(value || "") };
    });
    setTemplateEditorDirty(true);
    setTemplateEditorMessage("");
  }, [syncTemplateHistoryAvailability]);

  const onUndoTemplate = useCallback(() => {
    if (!templateUndoStackRef.current.length) {
      return;
    }
    setTemplateEditorDoc((prev) => {
      if (!prev) {
        return prev;
      }
      const previous = templateUndoStackRef.current.pop();
      if (!previous) {
        return prev;
      }
      templateRedoStackRef.current.push(cloneTemplateEditorDocument(prev));
      if (templateRedoStackRef.current.length > NOTE_HISTORY_MAX) {
        templateRedoStackRef.current.shift();
      }
      syncTemplateHistoryAvailability();
      return cloneTemplateEditorDocument(previous);
    });
    setTemplateEditorDirty(true);
    setTemplateEditorMessage("");
  }, [syncTemplateHistoryAvailability]);

  const onRedoTemplate = useCallback(() => {
    if (!templateRedoStackRef.current.length) {
      return;
    }
    setTemplateEditorDoc((prev) => {
      if (!prev) {
        return prev;
      }
      const restored = templateRedoStackRef.current.pop();
      if (!restored) {
        return prev;
      }
      templateUndoStackRef.current.push(cloneTemplateEditorDocument(prev));
      if (templateUndoStackRef.current.length > NOTE_HISTORY_MAX) {
        templateUndoStackRef.current.shift();
      }
      syncTemplateHistoryAvailability();
      return cloneTemplateEditorDocument(restored);
    });
    setTemplateEditorDirty(true);
    setTemplateEditorMessage("");
  }, [syncTemplateHistoryAvailability]);

  const onTemplateEditorKeyDown = useCallback((event) => {
    if (!event || !(event.metaKey || event.ctrlKey) || event.altKey) {
      return;
    }
    const key = String(event.key || "").toLowerCase();
    if (key === "z" && !event.shiftKey) {
      event.preventDefault();
      onUndoTemplate();
      return;
    }
    if ((key === "z" && event.shiftKey) || key === "y") {
      event.preventDefault();
      onRedoTemplate();
    }
  }, [onRedoTemplate, onUndoTemplate]);

  const onSaveTemplateEditor = useCallback(async () => {
    if (!selectedTemplateEditorId) {
      setLastError("No template selected.");
      return;
    }
    if (!templateEditorDoc) {
      setLastError("Template is not loaded yet.");
      return;
    }
    const validationError = validateTemplateEditorDocument(templateEditorDoc, selectedTemplateEditorId);
    if (validationError) {
      setLastError(validationError);
      return;
    }

    setTemplateEditorLoading(true);
    try {
      const response = await apiRef.current.saveNoteTemplateDocument(
        noteDepartment,
        selectedTemplateEditorId,
        templateEditorDoc,
      );
      const normalizedDoc = normalizeTemplateEditorDocument(response?.template || {});
      setTemplateEditorDoc(normalizedDoc);
      templateUndoStackRef.current = [];
      templateRedoStackRef.current = [];
      setCanUndoTemplate(false);
      setCanRedoTemplate(false);
      setTemplateEditorDirty(false);
      setTemplateEditorMessage(`Saved: ${selectedTemplateEditorId}`);
      setLastError("");

      const catalogResp = await apiRef.current.noteTemplates();
      const catalog = sanitizeTemplateCatalog(catalogResp);
      setNoteTemplateCatalog(catalog);
      const resolvedDepartment = normalizeDepartment(noteDepartment, catalog);
      const resolvedTemplateIds = normalizeTemplateIds(resolvedDepartment, noteTemplateIds, catalog);
      setNoteDepartment(resolvedDepartment);
      setNoteTemplateIds(resolvedTemplateIds);
      setNoteTemplateStatuses(buildDefaultTemplateStatuses(resolvedDepartment, catalog));
      if (!resolvedTemplateIds.includes(activeDraftTemplateId)) {
        setActiveDraftTemplateId(resolvedTemplateIds[0] || "");
      }
    } catch (error) {
      setLastError(String(error?.message || error));
    } finally {
      setTemplateEditorLoading(false);
    }
  }, [
    activeDraftTemplateId,
    noteDepartment,
    noteTemplateIds,
    selectedTemplateEditorId,
    templateEditorDoc,
  ]);

  const onToggleClip = useCallback((clip) => {
    const clipKey = String(clip?.clipKey || "").trim();
    if (!clipKey) {
      return;
    }
    if (isLiveAudioSource(pipelineDebug.source)) {
      const clipEnd = Number(clip?.t1);
      if (Number.isFinite(clipEnd) && clipEnd > (Number(liveArchiveEndSec) + 0.05)) {
        setLastError(
          `Audio archive syncing: clip needs ${clipEnd.toFixed(1)}s, currently archived to ${Number(liveArchiveEndSec).toFixed(1)}s.`,
        );
        return;
      }
    }
    if (clipKey === playingClipKey) {
      stopClip();
      return;
    }
    void playClip(clip);
  }, [liveArchiveEndSec, pipelineDebug.source, playClip, playingClipKey, stopClip]);

  const onSelectTimelineEvent = useCallback(({ eventKey, segmentId }) => {
    const normalizedEventKey = String(eventKey || "").trim();
    const normalizedSegmentId = String(segmentId || "").trim();
    if (normalizedSegmentId) {
      setSelectedSegmentId(normalizedSegmentId);
    }
    setSelectedEventKey(normalizedEventKey);
  }, []);

  const onSelectAssistantEvidenceEvent = useCallback(({ eventKey, segmentId }) => {
    const normalizedEventKey = String(eventKey || "").trim();
    const normalizedSegmentId = String(segmentId || "").trim();
    if (!normalizedEventKey) {
      return;
    }
    if (normalizedSegmentId) {
      setSelectedSegmentId(normalizedSegmentId);
    }
    setSelectedEventKey(normalizedEventKey);
  }, []);

  const riskFlagTimelineEvidence = useMemo(() => {
    const eventsBySegment = new Map();
    for (const event of events) {
      const segmentId = String(event?.evidence?.segment_id || "").trim();
      if (!segmentId) {
        continue;
      }
      const eventKey = buildEventStoreKey(event);
      const existing = eventsBySegment.get(segmentId) || [];
      existing.push({
        eventKey,
        segmentId,
        type: String(event?.type || ""),
        label: String(event?.label || event?.type || "event"),
        polarity: String(event?.polarity || ""),
        t0: Number(event?.evidence?.t0 || 0),
        t1: Number(event?.evidence?.t1 || 0),
        quote: String(event?.evidence?.quote || ""),
      });
      eventsBySegment.set(segmentId, existing);
    }

    for (const values of eventsBySegment.values()) {
      values.sort((a, b) => {
        if (a.t0 !== b.t0) {
          return a.t0 - b.t0;
        }
        return a.eventKey.localeCompare(b.eventKey);
      });
    }

    const map = {};
    const riskFlags = Array.isArray(snapshot?.risk_flags) ? snapshot.risk_flags : [];
    for (let index = 0; index < riskFlags.length; index += 1) {
      const riskFlag = riskFlags[index];
      const riskFlagKey = buildRiskFlagStoreKey(riskFlag, index);
      const refs = Array.isArray(riskFlag?.evidence_refs) ? riskFlag.evidence_refs : [];
      const dedup = new Set();
      const linked = [];
      for (const ref of refs) {
        const segmentId = String(ref || "").trim();
        if (!segmentId) {
          continue;
        }
        const matches = eventsBySegment.get(segmentId) || [];
        for (const item of matches) {
          if (!shouldLinkRiskEvidenceItem(riskFlag, item)) {
            continue;
          }
          if (dedup.has(item.eventKey)) {
            continue;
          }
          dedup.add(item.eventKey);
          linked.push(item);
        }
      }
      map[riskFlagKey] = linked;
    }
    return map;
  }, [events, snapshot?.risk_flags]);

  const maxEventEndSec = useMemo(
    () => events.reduce((maxValue, item) => Math.max(maxValue, Number(item?.evidence?.t1 || 0)), 0),
    [events],
  );

  const liveTranscriptHeaderStatus = useMemo(() => {
    const parts = [
      `Source: ${formatSourceLabel(pipelineDebug.source)}`,
      `Session: ${formatRunState(status, lastError)}`,
      `Transcription: ${formatAsrHealth(pipelineDebug.asr_status)}`,
    ];
    if (isTranscriptReprocessing) {
      parts.push("Edits: Reprocessing");
    }
    if (isLiveAudioSource(pipelineDebug.source)) {
      parts.push(`Archive Sync: ${formatArchiveSyncLabel(liveArchiveEndSec)}`);
    }
    return parts.join(" · ");
  }, [
    isTranscriptReprocessing,
    liveArchiveEndSec,
    lastError,
    pipelineDebug.asr_status,
    pipelineDebug.source,
    status,
  ]);

  const timelinePlaybackStatus = useMemo(() => {
    if (!canPlayAudioClips) {
      return "Unavailable";
    }
    if (!isLiveAudioSource(pipelineDebug.source)) {
      return "Ready";
    }
    if (maxEventEndSec <= 0) {
      return "Ready";
    }
    if ((Number(liveArchiveEndSec) + 0.05) >= maxEventEndSec) {
      return "Ready";
    }
    return `Syncing (${Number(liveArchiveEndSec).toFixed(1)}s/${maxEventEndSec.toFixed(1)}s)`;
  }, [canPlayAudioClips, liveArchiveEndSec, maxEventEndSec, pipelineDebug.source]);

  const timelineHeaderStatus = useMemo(() => {
    const parts = [
      `Evidence Items: ${events.length}`,
      `Extraction Engine: ${formatEngineLabel(pipelineDebug.event_engine_used, eventEngine)}`,
      `Playback: ${timelinePlaybackStatus}`,
    ];
    return parts.join(" · ");
  }, [eventEngine, events.length, pipelineDebug.event_engine_used, timelinePlaybackStatus]);

  const isMicActivelyProcessing = useMemo(() => {
    const micActiveStates = new Set(["requesting_permission", "connecting", "reconnecting", "streaming"]);
    return inputMode === "live" && micActiveStates.has(String(micStatus || "").trim().toLowerCase());
  }, [inputMode, micStatus]);

  const timelineEmptyStateMessage = useMemo(() => {
    if (isTranscriptReprocessing) {
      return "Reprocessing timeline from transcript edits...";
    }
    if (status === "running") {
      return "Processing timeline events...";
    }
    if (isMicActivelyProcessing) {
      return "Listening and extracting events...";
    }
    return "No events extracted yet.";
  }, [isMicActivelyProcessing, isTranscriptReprocessing, status]);

  const assistantRiskEmptyStateMessage = useMemo(() => {
    if (isTranscriptReprocessing) {
      return "Reprocessing risk assessment from transcript edits...";
    }
    if (status === "running") {
      return "Analyzing risk signals...";
    }
    if (isMicActivelyProcessing) {
      return "Listening and analyzing risk signals...";
    }
    return "No risk flag yet.";
  }, [isMicActivelyProcessing, isTranscriptReprocessing, status]);

  const assistantHeaderStatus = useMemo(() => {
    const parts = [
      `Risk Flags: ${Array.isArray(snapshot?.risk_flags) ? snapshot.risk_flags.length : 0}`,
      `Question Mode: ${formatQuestionModeLabel(pipelineDebug.open_questions_mode)}`,
      `Update State: ${formatRunState(status, lastError)}`,
    ];
    return parts.join(" · ");
  }, [lastError, pipelineDebug.open_questions_mode, snapshot?.risk_flags, status]);

  const templateEditorValidationError = validateTemplateEditorDocument(
    templateEditorDoc,
    selectedTemplateEditorId,
  );
  const templateEditorJustSaved = String(templateEditorMessage || "")
    .toLowerCase()
    .startsWith("saved:");
  const templateEditorStatusText = templateEditorLoading
    ? "Loading..."
    : templateEditorDirty
      ? "Unsaved changes"
      : templateEditorJustSaved
        ? "Saved"
        : "Ready";

  return html`
    <div className="app-root">
      <div className="safety-banner">Not a medical device. Clinician-in-the-loop</div>
      <header className="topbar app-header">
        <h1 className="brand-title">
          <span className="brand-main">Evidentia</span>
          <span className="brand-sub">An Agentic Clinical Reasoning Assistant</span>
        </h1>
      </header>

      <div className="app-shell">
        <aside className="side-nav" aria-label="Primary Navigation">
          ${NAV_ITEMS.map(
            (item) => html`
              <button
                key=${item.id}
                type="button"
                className=${`side-nav-item ${activeNav === item.id ? "active" : ""}`}
                onClick=${() => setActiveNav(item.id)}
                title=${item.label}
              >
                <span className="side-nav-icon">${item.icon}</span>
                <span className="side-nav-label">${item.label}</span>
              </button>
            `,
          )}
        </aside>

        <section className="content-pane">
          ${activeNav === "transcript"
            ? html`
                <section className="transcript-head">
                  <div className="global-fields-row">
                    <div className="global-field">
                      <label>Patient</label>
                      <input
                        value=${patientIdentity}
                        placeholder="Name / MRN / Identifier"
                        onChange=${(e) => setPatientIdentity(e.target.value)}
                      />
                    </div>
                    <div className="global-field">
                      <label>Department</label>
                      <select value=${noteDepartment} onChange=${(e) => onDepartmentChange(e.target.value)}>
                        ${getDepartmentOptions(noteTemplateCatalog).map(
                          (item) => html`<option key=${item.value} value=${item.value}>${item.label}</option>`,
                        )}
                      </select>
                    </div>
                  </div>
                </section>

                <nav className="review-tabbar" aria-label="Transcript sub-tabs" role="tablist">
                  ${REVIEW_TAB_ITEMS.map(
                    (item) => html`
                      <button
                        key=${item.id}
                        type="button"
                        role="tab"
                        aria-selected=${activeReviewTab === item.id}
                        className=${`review-tab ${activeReviewTab === item.id ? "active" : ""}`}
                        onClick=${() => setActiveReviewTab(item.id)}
                      >
                        ${item.label}
                      </button>
                    `,
                  )}
                </nav>

                ${activeReviewTab === "context"
                  ? html`
                      <main className="layout layout-notes context-layout">
                        <section className="panel context-panel">
                          <div className="field context-field">
                            <label>Patient Basic Info</label>
                            <textarea
                              className="context-patient-textarea"
                              value=${patientContextText}
                              placeholder="Age, sex, chief complaint, PMH, medications, allergies, social history..."
                              onInput=${(e) => setPatientContextText(e.target.value)}
                            />
                          </div>
                        </section>
                      </main>
                    `
                  : null}

                ${activeReviewTab === "transcript"
                  ? html`
                      <${ControlPanel}
                        mode="transcript"
                        inputMode=${inputMode}
                        audioPath=${audioPath}
                        audioPathDisplay=${audioPathDisplay}
                        audioSampleOptions=${audioSampleOptions}
                        selectedAudioSource=${selectedAudioSource || LOCAL_AUDIO_SOURCE_VALUE}
                        localAudioSourceValue=${LOCAL_AUDIO_SOURCE_VALUE}
                        audioPathLocked=${false}
                        streamText=${streamText}
                        statusText=${statusText}
                        runStatus=${status}
                        onInputModeChange=${(value) => setInputMode(value)}
                        onAudioPathChange=${(value) => {
                          setAudioPath(value);
                          setAudioPathDisplay(value);
                        }}
                        onAudioSourceChange=${onAudioSourceChange}
                        onUploadAudioFile=${uploadAudioFile}
                        onStreamTextChange=${setStreamText}
                        onStart=${startStreaming}
                        onStop=${stopStreaming}
                        onReset=${resetSession}
                        micSupported=${micSupported}
                        micStatus=${micStatus}
                        onStartMic=${startMicStreaming}
                        onStopMic=${stopMicStreaming}
                      />

                      <main
                        className=${`layout transcript-resizable-layout ${activeSplitter >= 0 ? "is-resizing" : ""}`}
                        ref=${transcriptLayoutRef}
                        style=${{
                          gridTemplateColumns:
                            `minmax(${TRANSCRIPT_PANEL_MIN_WIDTHS[0]}px, ${transcriptPaneWidths[0]}%) ` +
                            `${TRANSCRIPT_SPLITTER_WIDTH_PX}px ` +
                            `minmax(${TRANSCRIPT_PANEL_MIN_WIDTHS[1]}px, ${transcriptPaneWidths[1]}%) ` +
                            `${TRANSCRIPT_SPLITTER_WIDTH_PX}px ` +
                            `minmax(${TRANSCRIPT_PANEL_MIN_WIDTHS[2]}px, ${transcriptPaneWidths[2]}%)`,
                        }}
                      >
                        <${TranscriptPanel}
                          rows=${transcriptRows}
                          selectedSegmentId=${selectedSegmentId}
                          headerStatus=${liveTranscriptHeaderStatus}
                          canPlayAudio=${canPlayAudioClips}
                          playingClipKey=${playingClipKey}
                          onToggleClip=${onToggleClip}
                          editLocked=${isTranscriptReprocessing}
                          onSaveEdit=${onSaveTranscriptEdit}
                          onDeleteRow=${onDeleteTranscriptSegment}
                        />
                        <button
                          type="button"
                          className=${`pane-splitter ${activeSplitter === 0 ? "active" : ""}`}
                          aria-label="Resize transcript and timeline panels"
                          onPointerDown=${(e) => {
                            e.preventDefault();
                            startPaneResize(0, e.clientX);
                          }}
                        >
                          <span className="pane-splitter-grip">::</span>
                        </button>
                        <${TimelinePanel}
                          events=${events}
                          selectedSegmentId=${selectedSegmentId}
                          selectedEventKey=${selectedEventKey}
                          headerStatus=${timelineHeaderStatus}
                          emptyStateMessage=${timelineEmptyStateMessage}
                          onSelectEvidence=${setSelectedSegmentId}
                          onSelectEvent=${onSelectTimelineEvent}
                          canPlayAudio=${canPlayAudioClips}
                          playingClipKey=${playingClipKey}
                          onToggleClip=${onToggleClip}
                        />
                        <button
                          type="button"
                          className=${`pane-splitter ${activeSplitter === 1 ? "active" : ""}`}
                          aria-label="Resize timeline and assistant panels"
                          onPointerDown=${(e) => {
                            e.preventDefault();
                            startPaneResize(1, e.clientX);
                          }}
                        >
                          <span className="pane-splitter-grip">::</span>
                        </button>
                        <${InsightPanel}
                          mode="transcript"
                          headerStatus=${assistantHeaderStatus}
                          riskFlags=${snapshot.risk_flags}
                          riskEmptyStateMessage=${assistantRiskEmptyStateMessage}
                          riskFlagTimelineEvidence=${riskFlagTimelineEvidence}
                          onSelectEvidenceEvent=${onSelectAssistantEvidenceEvent}
                          openQuestions=${snapshot.open_questions}
                          mandatoryQuestions=${snapshot.mandatory_safety_questions}
                          contextualFollowups=${snapshot.contextual_followups}
                          openQuestionsRationale=${snapshot.rationale}
                          pipelineDebug=${pipelineDebug}
                          lastError=${lastError}
                        />
                      </main>
                    `
                  : null}

                ${activeReviewTab === "notes"
                  ? html`
                      <main className="layout layout-notes">
                        <${InsightPanel}
                          mode="notes"
                          noteText=${noteText}
                          citations=${citations}
                          draftOptions=${generatedDrafts}
                          activeDraftTemplateId=${activeDraftTemplateId}
                          onDraftTemplateChange=${onDraftTemplateChange}
                          noteDepartment=${noteDepartment}
                          onNoteDepartmentChange=${onDepartmentChange}
                          noteTemplateCatalog=${noteTemplateCatalog}
                          noteTemplateIds=${noteTemplateIds}
                          noteTemplateOptions=${getTemplateOptions(noteDepartment, noteTemplateCatalog)}
                          noteTemplateStatuses=${noteTemplateStatuses}
                          noteGenerationStatus=${noteGenerationStatus}
                          onNoteTemplatesChange=${onNoteTemplatesChange}
                          onGenerateDrafts=${onGenerateDrafts}
                          onStopGenerateDrafts=${onStopGenerateDrafts}
                          onNoteChange=${onNoteChange}
                          onUndoNote=${onUndoNote}
                          onRedoNote=${onRedoNote}
                          canUndoNote=${canUndoNote}
                          canRedoNote=${canRedoNote}
                          onNoteKeyDown=${onNoteKeyDown}
                          onExport=${exportNote}
                          onEmail=${emailNote}
                          lastError=${lastError}
                        />
                      </main>
                    `
                  : null}
              `
            : null}

          ${activeNav === "template"
            ? html`
                <main className="template-layout">
                  <section className="panel template-list-panel">
                    <h2>Template List</h2>
                    <div className="notes-template-tree template-template-tree">
                      ${templateDepartmentGroups.length
                        ? templateDepartmentGroups.map(([departmentId, templates]) => {
                            const normalizedDepartmentId = String(departmentId || "").trim();
                            const isExpanded = Boolean(templateExpandedDepartments[normalizedDepartmentId]);
                            const isActiveDepartment = normalizedDepartmentId === String(noteDepartment || "");
                            const sortedTemplates = (Array.isArray(templates) ? [...templates] : []).sort((left, right) =>
                              compareTemplatesWithinDepartment(normalizedDepartmentId, left, right),
                            );
                            return html`
                              <div key=${normalizedDepartmentId} className="notes-department-group">
                                <button
                                  type="button"
                                  className=${`notes-department-toggle ${isActiveDepartment ? "active" : ""}`}
                                  onClick=${() => onToggleTemplateDepartment(normalizedDepartmentId)}
                                >
                                  <span>${isExpanded ? "▾" : "▸"} ${formatDepartmentLabel(normalizedDepartmentId)}</span>
                                  ${isActiveDepartment ? html`<span className="notes-department-current">Current</span>` : null}
                                </button>

                                ${isExpanded
                                  ? html`
                                      <div className="notes-template-list template-template-list">
                                        ${sortedTemplates.map((item) => {
                                          const templateId = String(item?.id || "").trim();
                                          const isSelected =
                                            isActiveDepartment && selectedTemplateEditorId === templateId;
                                          return html`
                                            <button
                                              key=${templateId}
                                              type="button"
                                              className=${`notes-template-row template-template-row ${isSelected ? "selected" : ""}`}
                                              onClick=${() =>
                                                onSelectTemplateFromDepartment(normalizedDepartmentId, templateId)}
                                            >
                                              <span className="template-template-name">${String(item?.label || templateId)}</span>
                                            </button>
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
                  </section>

                  <section className="panel template-editor-panel">
                    <div className="template-editor-toolbar">
                      <h2>${selectedTemplateEditorLabel}</h2>
                      ${selectedTemplateEditorId
                        ? html`<span className="status template-editor-title-status">${templateEditorStatusText}</span>`
                        : null}
                    </div>

                    ${templateEditorMessage ? html`<div className="status template-editor-message">${templateEditorMessage}</div>` : null}
                    ${templateEditorDoc
                      ? html`
                          <div className="field">
                            <label>Template Name</label>
                            <input
                              value=${templateEditorDoc.template_name || ""}
                              onKeyDown=${onTemplateEditorKeyDown}
                              onInput=${(e) => onTemplateNameChange(e.target.value)}
                              disabled=${templateEditorLoading}
                            />
                          </div>

                          <div className="field template-content-field">
                            <label>Template Content</label>
                            <textarea
                              className="template-textarea"
                              value=${templateEditorDoc.template_text || ""}
                              onKeyDown=${onTemplateEditorKeyDown}
                              onInput=${(e) => onTemplateTextChange(e.target.value)}
                              disabled=${templateEditorLoading}
                            />
                          </div>

                          <div className="template-editor-actions template-editor-actions-bottom">
                            <div className="template-editor-actions-primary">
                              <button
                                type="button"
                                className="primary"
                                onClick=${onSaveTemplateEditor}
                                disabled=${templateEditorLoading || !selectedTemplateEditorId || !templateEditorDoc || Boolean(templateEditorValidationError)}
                              >
                                Save Template
                              </button>
                            </div>
                            <div className="template-editor-actions-tools">
                              <button
                                type="button"
                                className="notes-tool-btn"
                                title="Redo (Ctrl/Cmd+Shift+Z)"
                                aria-label="Redo"
                                disabled=${!canRedoTemplate || !templateEditorDoc || templateEditorLoading}
                                onClick=${onRedoTemplate}
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
                                className="notes-tool-btn"
                                title="Undo (Ctrl/Cmd+Z)"
                                aria-label="Undo"
                                disabled=${!canUndoTemplate || !templateEditorDoc || templateEditorLoading}
                                onClick=${onUndoTemplate}
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
                            </div>
                          </div>
                        `
                      : html`<div className="status">No template selected.</div>`}
                    ${templateEditorValidationError && !templateEditorLoading
                      ? html`<div className="status template-validation-error">${templateEditorValidationError}</div>`
                      : null}
                  </section>
                </main>
              `
            : null}

          ${activeNav === "config"
            ? html`
                <${ControlPanel}
                  mode="context"
                  showOpenQuestionsAiSetting=${true}
                  baseUrl=${baseUrl}
                  sessionId=${sessionId}
                  intervalMs=${intervalMs}
                  inputMode=${inputMode}
                  audioPath=${audioPath}
                  audioPathDisplay=${audioPathDisplay}
                  audioSampleOptions=${audioSampleOptions}
                  selectedAudioSource=${selectedAudioSource || LOCAL_AUDIO_SOURCE_VALUE}
                  localAudioSourceValue=${LOCAL_AUDIO_SOURCE_VALUE}
                  audioWindowSec=${audioWindowSec}
                  reconcileLookbackWindows=${reconcileLookbackWindows}
                  llmUpdateIntervalWindows=${llmUpdateIntervalWindows}
                  openQuestionsAiEnhancementEnabled=${openQuestionsAiEnhancementEnabled}
                  micCaptureSliceMs=${micCaptureSliceMs}
                  micAsrWindowSec=${micAsrWindowSec}
                  micAsrStepSec=${micAsrStepSec}
                  audioPathLocked=${false}
                  eventEngine=${eventEngine}
                  statusText=${statusText}
                  onBaseUrlChange=${(value) => setBaseUrl(value || baseUrl)}
                  onSessionIdChange=${(value) => setSessionId(value || sessionId)}
                  onIntervalChange=${(value) => {
                    const parsed = Number(value);
                    if (Number.isFinite(parsed) && parsed >= 300) {
                      setIntervalMs(parsed);
                    }
                  }}
                  onInputModeChange=${(value) => setInputMode(value)}
                  onAudioPathChange=${(value) => {
                    setAudioPath(value);
                    setAudioPathDisplay(value);
                  }}
                  onAudioSourceChange=${onAudioSourceChange}
                  onAudioWindowSecChange=${(value) => {
                    const parsed = Number(value);
                    if (Number.isFinite(parsed) && parsed > 0 && parsed <= 120) {
                      setAudioWindowSec(parsed);
                    }
                  }}
                  onReconcileLookbackChange=${(value) => {
                    const parsed = Number(value);
                    if (Number.isFinite(parsed) && parsed >= 0 && parsed <= 20) {
                      setReconcileLookbackWindows(Math.floor(parsed));
                    }
                  }}
                  onLlmUpdateIntervalChange=${(value) => {
                    const parsed = Number(value);
                    if (Number.isFinite(parsed) && parsed >= 1 && parsed <= 20) {
                      setLlmUpdateIntervalWindows(Math.floor(parsed));
                    }
                  }}
                  onOpenQuestionsAiEnhancementChange=${setOpenQuestionsAiEnhancementEnabled}
                  onMicCaptureSliceMsChange=${(value) => {
                    const parsed = Number(value);
                    if (Number.isFinite(parsed) && parsed >= 200 && parsed <= 10000) {
                      setMicCaptureSliceMs(Math.floor(parsed));
                    }
                  }}
                  onMicAsrWindowSecChange=${(value) => {
                    const parsed = Number(value);
                    if (Number.isFinite(parsed) && parsed > 0 && parsed <= 120) {
                      setMicAsrWindowSec(parsed);
                    }
                  }}
                  onMicAsrStepSecChange=${(value) => {
                    const parsed = Number(value);
                    if (Number.isFinite(parsed) && parsed > 0 && parsed <= 120) {
                      setMicAsrStepSec(parsed);
                    }
                  }}
                  onEngineChange=${(value) => setEventEngine(value)}
                />
              `
            : null}
        </section>
      </div>

      <footer className="app-footer">
        <div className="app-footer-main">Open Research Demonstrator © 2026 Evidentia Labs</div>
        <div className="app-footer-sub">Parallel Production Architecture: Rust Core + Electron Secure Client</div>
      </footer>

      ${deleteDecisionDialog.open
        ? html`
            <div className="transcript-delete-dialog-backdrop" aria-hidden="true"></div>
            <section
              className="transcript-delete-dialog"
              role="dialog"
              aria-modal="true"
              aria-labelledby="transcript-delete-dialog-title"
            >
              <h3 id="transcript-delete-dialog-title">Transcript Segment Deleted</h3>
              <p className="meta">Choose what to do next for pipeline updates.</p>
              ${deleteDecisionDialog.deletedRow?.text
                ? html`
                    <p className="transcript-delete-dialog-quote">
                      "${String(deleteDecisionDialog.deletedRow.text || "").trim()}"
                    </p>
                  `
                : null}
              <div className="transcript-delete-dialog-actions">
                <button type="button" onClick=${onCancelDeleteSegment}>Cancel</button>
                <button type="button" onClick=${onConfirmDeleteOnly}>Delete Only</button>
                <button type="button" className="primary" onClick=${onConfirmDeleteAndReprocess}>
                  Delete + Reprocess
                </button>
              </div>
            </section>
          `
        : null}
    </div>
  `;
}

function buildRowsFromUtterances(utterances, sessionId) {
  return utterances.map((item) => ({
    segment_id: buildStableSegmentId(item, sessionId),
    start: Number(item.start || 0),
    end: Number(item.end || 0),
    text: String(item.text || ""),
    speaker: String(item.speaker || item.speaker_role || "other").toLowerCase(),
    speaker_id: String(item.speaker_id || ""),
    speaker_role: String(item.speaker_role || item.speaker || "other").toLowerCase(),
    diar_confidence: Number.isFinite(Number(item.diar_confidence))
      ? Number(item.diar_confidence)
      : null,
  }));
}

function applyManualTranscriptOverrides(rows, overrides) {
  if (!Array.isArray(rows) || !rows.length) {
    return [];
  }
  const map = overrides && typeof overrides === "object" ? overrides : {};
  return rows.map((item) => {
    const segmentId = String(item?.segment_id || "");
    if (!segmentId || !Object.prototype.hasOwnProperty.call(map, segmentId)) {
      return item;
    }
    const override = map[segmentId] && typeof map[segmentId] === "object" ? map[segmentId] : {};
    if (Boolean(override.deleted)) {
      return null;
    }
    const nextRole = normalizeEditableSpeakerRole(
      String(override.speaker_role || override.speaker || item.speaker_role || item.speaker || ""),
    );
    const nextText = String(override.text ?? item.text ?? "");
    return {
      ...item,
      text: nextText,
      speaker: nextRole,
      speaker_role: nextRole,
    };
  }).filter(Boolean);
}

function normalizeEditableSpeakerRole(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "patient" || normalized === "clinician") {
    return normalized;
  }
  return "other";
}

function restoreDeletedTranscriptRow(rows, deletedRow, deletedIndex, segmentId) {
  const safeRows = Array.isArray(rows) ? rows : [];
  const withoutSegment = safeRows.filter((item) => String(item?.segment_id || "") !== String(segmentId || ""));
  const index = Number.isFinite(deletedIndex)
    ? Math.max(0, Math.min(Math.floor(deletedIndex), withoutSegment.length))
    : withoutSegment.length;
  return [
    ...withoutSegment.slice(0, index),
    deletedRow,
    ...withoutSegment.slice(index),
  ];
}

function restoreManualOverrideAfterDeleteCancel(overrides, segmentId, previousOverride) {
  const next = { ...(overrides || {}) };
  if (previousOverride === undefined) {
    delete next[segmentId];
    return next;
  }
  next[segmentId] = previousOverride;
  return next;
}

function buildStableSegmentId(utterance, sessionId) {
  const startTick = Math.round((Number(utterance?.start) || 0) * 10);
  const endTick = Math.round((Number(utterance?.end) || 0) * 10);
  const speaker = String(utterance?.speaker || "other")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  const textHash = fnv1a32(String(utterance?.text || ""));
  return `${sessionId}_seg_${startTick}_${endTick}_${speaker || "other"}_${textHash}`;
}

function fnv1a32(input) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function buildEventStoreKey(eventItem) {
  const evidence = eventItem?.evidence || {};
  return [
    String(evidence.segment_id || ""),
    String(eventItem?.type || ""),
    String(eventItem?.label || ""),
    String(eventItem?.polarity || ""),
  ].join("|");
}

function buildRiskFlagStoreKey(riskFlag, index) {
  return [
    String(riskFlag?.level || ""),
    String(riskFlag?.flag || ""),
    String(index),
  ].join("|");
}

function shouldLinkRiskEvidenceItem(riskFlag, evidenceItem) {
  const eventType = String(evidenceItem?.type || "").trim().toLowerCase();
  if (eventType !== "risk_cue") {
    return false;
  }

  const flag = String(riskFlag?.flag || "").trim().toLowerCase();
  const label = String(evidenceItem?.label || "").trim().toLowerCase();
  const polarity = String(evidenceItem?.polarity || "").trim().toLowerCase();

  if (flag === "urgent_suicide_risk") {
    return label === "suicidal_plan_or_intent" && polarity === "present";
  }
  if (flag === "passive_or_active_si_detected") {
    return (
      (label === "suicidal_ideation" || label === "passive_suicidal_ideation") &&
      polarity === "present"
    );
  }
  if (flag === "si_explicitly_denied") {
    return label === "suicidal_ideation" && polarity === "absent";
  }
  if (flag === "possible_homicidal_risk") {
    return label === "homicidal_ideation" && polarity === "present";
  }
  if (flag === "possible_psychosis_risk") {
    return label === "psychosis_cue" && polarity === "present";
  }

  // Unknown flags: keep strict at least to risk_cue.
  return true;
}

function isAudioPathSource(source) {
  const normalizedSource = String(source || "").trim().toLowerCase();
  return normalizedSource.startsWith("audio_path");
}

function isLiveAudioSource(source) {
  const normalizedSource = String(source || "").trim().toLowerCase();
  return normalizedSource === "live_audio_ws";
}

function getCaptureSliceSec(micCaptureSliceMs) {
  const raw = Number(micCaptureSliceMs);
  if (!Number.isFinite(raw) || raw <= 0) {
    return DEFAULT_MIC_CAPTURE_SLICE_MS / 1000;
  }
  return Math.max(0.2, raw / 1000);
}

function parseArchiveEndSec(value) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw || raw === "n/a") {
    return 0;
  }
  const normalized = raw.endsWith("s") ? raw.slice(0, -1) : raw;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

function formatArchiveEndSec(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) {
    return "";
  }
  return `${num.toFixed(1)}s`;
}

function formatArchiveSyncLabel(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) {
    return "Pending";
  }
  return `${num.toFixed(1)}s`;
}

function formatSourceLabel(source) {
  const normalized = String(source || "").trim().toLowerCase();
  if (!normalized) {
    return "Not Available";
  }
  if (isLiveAudioSource(normalized)) {
    return "Live Mic";
  }
  if (isAudioPathSource(normalized)) {
    return "Audio File";
  }
  if (normalized === "transcript_text") {
    return "Transcript Text";
  }
  if (normalized === "segments_payload") {
    return "Segment Payload";
  }
  return toTitleCaseWords(normalized.replace(/_/g, " "));
}

function formatRunState(status, lastError) {
  const normalized = String(status || "").trim().toLowerCase();
  if (lastError || normalized === "error") {
    return "Error";
  }
  if (normalized === "running") {
    return "Running";
  }
  if (normalized === "paused") {
    return "Paused";
  }
  if (normalized === "completed") {
    return "Completed";
  }
  return "Idle";
}

function formatAsrHealth(asrStatus) {
  const normalized = String(asrStatus || "").trim().toLowerCase();
  if (!normalized) {
    return "Not Available";
  }
  if (normalized.includes("error")) {
    return "Error";
  }
  if (normalized.includes("ok")) {
    return "Healthy";
  }
  if (normalized.includes("not_configured")) {
    return "Not Configured";
  }
  if (normalized.includes("simulated")) {
    return "Simulated";
  }
  return toTitleCaseWords(String(asrStatus || "").replace(/_/g, " "));
}

function formatEngineLabel(engineUsed, requestedEngine) {
  const normalized = String(engineUsed || requestedEngine || "").trim().toLowerCase();
  if (!normalized) {
    return "Not Available";
  }
  if (normalized.startsWith("rule_fallback")) {
    return "Rule Fallback";
  }
  if (normalized === "medgemma") {
    return "MedGemma";
  }
  if (normalized === "rule") {
    return "Rule";
  }
  if (normalized === "auto") {
    return "Auto";
  }
  return toTitleCaseWords(normalized.replace(/_/g, " "));
}

function formatQuestionModeLabel(mode) {
  const normalized = String(mode || "").trim().toLowerCase();
  if (!normalized) {
    return "Not Available";
  }
  if (normalized === "hybrid") {
    return "Hybrid";
  }
  if (normalized === "rule") {
    return "Rule";
  }
  if (normalized === "llm") {
    return "LLM";
  }
  return toTitleCaseWords(normalized.replace(/_/g, " "));
}

function toTitleCaseWords(input) {
  return String(input || "")
    .split(/\s+/)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

function buildAudioFileUrl(baseUrl, audioPath) {
  const normalizedPath = String(audioPath || "").trim();
  if (!normalizedPath) {
    return "";
  }
  try {
    const origin = String(baseUrl || "").trim() || window.location.origin;
    const url = new URL("/files/audio", origin);
    url.searchParams.set("audio_path", normalizedPath);
    return url.toString();
  } catch (_) {
    return "";
  }
}

function pickAlignmentDebugSlice(alignmentDebug) {
  if (!alignmentDebug || typeof alignmentDebug !== "object") {
    return null;
  }
  if (alignmentDebug.new && typeof alignmentDebug.new === "object") {
    return alignmentDebug.new;
  }
  if (alignmentDebug.all && typeof alignmentDebug.all === "object") {
    return alignmentDebug.all;
  }
  return alignmentDebug;
}

function extractDiarizationReason(diarizationDebug) {
  if (!diarizationDebug || typeof diarizationDebug !== "object") {
    return "";
  }
  const direct = String(diarizationDebug.reason || "").trim();
  if (direct) {
    return direct;
  }
  const nested = [diarizationDebug.diarization_adapter, diarizationDebug.custom_diarization];
  for (const item of nested) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const reason = String(item.reason || "").trim();
    if (reason) {
      return reason;
    }
  }
  return "";
}

function extractAlignmentReasonCounts(alignmentDebug) {
  const picked = pickAlignmentDebugSlice(alignmentDebug);
  if (!picked || typeof picked !== "object") {
    return "";
  }
  const counts = picked.reason_counts;
  if (!counts || typeof counts !== "object") {
    return "";
  }
  const parts = Object.entries(counts)
    .filter(([, value]) => Number(value) > 0)
    .map(([key, value]) => `${key}=${Number(value)}`);
  return parts.join(", ");
}

function extractAlignmentFallbackRate(alignmentDebug) {
  const picked = pickAlignmentDebugSlice(alignmentDebug);
  if (!picked || typeof picked !== "object") {
    return "";
  }
  const fallbackTotal = Number(picked.fallback_total);
  const processed = Number(picked.segments_processed);
  const rate = Number(picked.fallback_rate);
  if (Number.isFinite(rate) && Number.isFinite(fallbackTotal) && Number.isFinite(processed) && processed > 0) {
    return `${(rate * 100).toFixed(1)}% (${fallbackTotal}/${processed})`;
  }
  if (Number.isFinite(fallbackTotal) && Number.isFinite(processed) && processed > 0) {
    return `${fallbackTotal}/${processed}`;
  }
  return "";
}

function extractSentenceRoleSplit(splitDebug) {
  if (!splitDebug || typeof splitDebug !== "object") {
    return "";
  }
  const status = String(splitDebug.status || "").trim();
  if (!status) {
    return "";
  }
  if (status === "skipped") {
    const reason = String(splitDebug.reason || "").trim();
    return reason ? `skipped (${reason})` : "skipped";
  }
  const all = splitDebug.all && typeof splitDebug.all === "object" ? splitDebug.all : null;
  if (!all) {
    return status;
  }
  const applied = Number(all.split_applied);
  const candidates = Number(all.split_candidates);
  if (Number.isFinite(applied) && Number.isFinite(candidates)) {
    return `${status} (${applied}/${candidates})`;
  }
  return status;
}

function extractSimpleDropSummary(debugObj) {
  if (!debugObj || typeof debugObj !== "object") {
    return "";
  }
  const dropped = Number(debugObj.dropped);
  if (!Number.isFinite(dropped)) {
    return "";
  }
  const reasons = debugObj.drop_reasons && typeof debugObj.drop_reasons === "object"
    ? Object.entries(debugObj.drop_reasons).map(([k, v]) => `${k}=${Number(v)}`).join(", ")
    : "";
  return reasons ? `${dropped} (${reasons})` : String(dropped);
}

function extractRiskBackstopSummary(debugObj) {
  if (!debugObj || typeof debugObj !== "object") {
    return "";
  }
  const status = String(debugObj.status || "").trim();
  const added = Number(debugObj.added);
  const candidates = Number(debugObj.rule_risk_candidates);
  if (Number.isFinite(added) && Number.isFinite(candidates)) {
    return `${status || "applied"} added=${added}/${candidates}`;
  }
  return status;
}

function sortEventsForTimeline(events) {
  return [...events].sort((a, b) => {
    const at = Number(a?.evidence?.t0 || 0);
    const bt = Number(b?.evidence?.t0 || 0);
    if (at !== bt) {
      return at - bt;
    }
    const ak = buildEventStoreKey(a);
    const bk = buildEventStoreKey(b);
    return ak.localeCompare(bk);
  });
}

function createEmptyMedgemmaStats() {
  return {
    calls: 0,
    jsonValid: 0,
    fallback: 0,
    filterLatenciesMs: [],
    extractLatenciesMs: [],
  };
}

function normalizeTemplateEditorDocument(raw) {
  const templateId = String(raw?.template_id || "").trim();
  const templateName = String(raw?.template_name || "").trim();
  const templateText = String(raw?.template_text || "");
  return {
    template_id: templateId,
    template_name: templateName,
    template_text: templateText,
  };
}

function cloneTemplateEditorDocument(raw) {
  return normalizeTemplateEditorDocument(raw || {});
}

function validateTemplateEditorDocument(template, selectedTemplateId) {
  if (!template || typeof template !== "object") {
    return "";
  }
  const templateId = String(template.template_id || "").trim();
  if (!templateId) {
    return "template_id is required.";
  }
  if (selectedTemplateId && templateId !== String(selectedTemplateId)) {
    return `template_id must equal selected template (${selectedTemplateId}).`;
  }
  const templateName = String(template.template_name || "").trim();
  if (!templateName) {
    return "Template Name is required.";
  }
  const templateText = String(template.template_text || "").trim();
  if (!templateText) {
    return "Template Content is required.";
  }
  return "";
}

function sanitizeTemplateCatalog(response) {
  const raw = response?.templates_by_department;
  if (!raw || typeof raw !== "object") {
    return {};
  }
  const output = {};
  for (const [department, templates] of Object.entries(raw)) {
    const normalizedDepartment = String(department || "")
      .trim()
      .toLowerCase()
      .replace(/-/g, "_")
      .replace(/\s+/g, "_");
    if (!normalizedDepartment || !Array.isArray(templates)) {
      continue;
    }
    const seenIds = new Set();
    const normalizedTemplates = [];
    for (const item of templates) {
      const templateId = String(item?.template_id || "").trim();
      const templateName = String(item?.template_name || "").trim();
      if (!templateId || seenIds.has(templateId)) {
        continue;
      }
      seenIds.add(templateId);
      normalizedTemplates.push({
        id: templateId,
        label: templateName || templateId,
      });
    }
    if (normalizedTemplates.length) {
      output[normalizedDepartment] = normalizedTemplates;
    }
  }
  return output;
}

function normalizeDepartment(raw, catalog) {
  const departments = Object.keys(catalog || {});
  if (!departments.length) {
    return "";
  }
  const normalized = String(raw || "")
    .trim()
    .toLowerCase()
    .replace(/-/g, "_")
    .replace(/\s+/g, "_");
  if (Object.prototype.hasOwnProperty.call(catalog, normalized)) {
    return normalized;
  }
  return departments.includes("psych") ? "psych" : departments[0];
}

function getDepartmentOptions(catalog) {
  return Object.keys(catalog || {}).map((item) => ({
    value: item,
    label: formatDepartmentLabel(item),
  }));
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

function getTemplateOptions(department, catalog) {
  const key = normalizeDepartment(department, catalog);
  if (!key) {
    return [];
  }
  return catalog[key] || [];
}

function normalizeTemplateIds(department, values, catalog) {
  const options = getTemplateOptions(department, catalog);
  const optionIds = options.map((item) => String(item.id));
  const input = Array.isArray(values) ? values : [];
  const seen = new Set();
  const picked = [];
  for (const item of input) {
    const value = String(item || "").trim();
    if (!value || seen.has(value) || !optionIds.includes(value)) {
      continue;
    }
    seen.add(value);
    picked.push(value);
  }
  if (!picked.length && optionIds.length) {
    return [optionIds[0]];
  }
  return picked;
}

function buildDefaultTemplateStatuses(department, catalog) {
  const options = getTemplateOptions(department, catalog);
  const statuses = {};
  for (const item of options) {
    const templateId = String(item?.id || "").trim();
    if (!templateId) {
      continue;
    }
    statuses[templateId] = "not_generated";
  }
  return statuses;
}

function normalizeEmailBody(raw) {
  const value = String(raw || "");
  if (!value) {
    return "";
  }
  const looksFormEncoded =
    (value.includes("+") && !value.includes(" ")) || /%[0-9A-Fa-f]{2}/.test(value);
  if (!looksFormEncoded) {
    return value;
  }
  try {
    return decodeURIComponent(value.replace(/\+/g, "%20"));
  } catch (_) {
    if (value.includes("+") && !value.includes(" ")) {
      return value.replace(/\+/g, " ");
    }
    return value;
  }
}

function updateMedgemmaStats(stats, debug, requestedEngine) {
  const engineRequested = String(debug?.engine_requested || requestedEngine || "");
  const medgemmaAttempted = engineRequested === "auto" || engineRequested === "medgemma";
  if (!medgemmaAttempted) {
    return formatMedgemmaStats(stats);
  }

  stats.calls += 1;
  const medgemmaError = String(debug?.medgemma_error || "");
  if (medgemmaError.toLowerCase().includes("not valid json")) {
    // Explicit parse failure from adapter.
  } else if (debug?.medgemma) {
    stats.jsonValid += 1;
  }

  const engineUsed = String(debug?.engine_used || "");
  if (engineUsed.startsWith("rule_fallback")) {
    stats.fallback += 1;
  }

  const filterMs = Number(debug?.medgemma?.filter_inference_ms);
  if (Number.isFinite(filterMs) && filterMs > 0) {
    stats.filterLatenciesMs.push(filterMs);
  }
  const extractMs = Number(debug?.medgemma?.extract_inference_ms);
  if (Number.isFinite(extractMs) && extractMs > 0) {
    stats.extractLatenciesMs.push(extractMs);
  }

  return formatMedgemmaStats(stats);
}

function recordMedgemmaCallError(stats) {
  stats.calls += 1;
  return formatMedgemmaStats(stats);
}

function formatMedgemmaStats(stats) {
  return {
    medgemma_calls: stats.calls,
    medgemma_json_valid_rate: formatRate(stats.jsonValid, stats.calls),
    medgemma_fallback_rate: formatRate(stats.fallback, stats.calls),
    medgemma_filter_p50_p95_ms: formatP50P95(stats.filterLatenciesMs),
    medgemma_extract_p50_p95_ms: formatP50P95(stats.extractLatenciesMs),
  };
}

function formatRate(numerator, denominator) {
  if (!denominator) {
    return "n/a";
  }
  const percent = (Number(numerator) / Number(denominator)) * 100;
  return `${percent.toFixed(1)}% (${numerator}/${denominator})`;
}

function formatP50P95(values) {
  if (!values.length) {
    return "n/a";
  }
  const p50 = percentile(values, 50);
  const p95 = percentile(values, 95);
  return `p50=${p50.toFixed(0)}ms, p95=${p95.toFixed(0)}ms (n=${values.length})`;
}

function percentile(values, p) {
  if (!values.length) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const rank = (p / 100) * (sorted.length - 1);
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  if (low === high) {
    return sorted[low];
  }
  const w = rank - low;
  return sorted[low] * (1 - w) + sorted[high] * w;
}

function isMicCaptureSupported() {
  if (typeof window === "undefined") {
    return false;
  }
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    return false;
  }
  return typeof MediaRecorder !== "undefined";
}

function pickSupportedRecorderMimeType() {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
    return "";
  }
  const preferred = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  for (const item of preferred) {
    if (MediaRecorder.isTypeSupported(item)) {
      return item;
    }
  }
  return "";
}

function mimeTypeToAudioExt(mimeType) {
  const mt = String(mimeType || "").toLowerCase();
  if (mt.includes("wav")) {
    return ".wav";
  }
  if (mt.includes("mpeg") || mt.includes("mp3")) {
    return ".mp3";
  }
  if (mt.includes("ogg")) {
    return ".ogg";
  }
  if (mt.includes("mp4") || mt.includes("m4a")) {
    return ".m4a";
  }
  return ".webm";
}

function toWebSocketUrl(baseUrl, path) {
  const source = String(baseUrl || "").trim() || window.location.origin;
  const parsed = new URL(source);
  const protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${parsed.host}${path}`;
}

function arrayBufferToBase64(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function base64ToUint8Array(dataB64) {
  const binary = atob(String(dataB64 || ""));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function mergeBase64Chunks(parts) {
  const chunks = (Array.isArray(parts) ? parts : []).map((item) => base64ToUint8Array(item));
  const totalLength = chunks.reduce((acc, item) => acc + item.length, 0);
  const merged = new Uint8Array(totalLength);
  let offset = 0;
  for (const item of chunks) {
    merged.set(item, offset);
    offset += item.length;
  }
  return merged;
}

const rootNode = document.getElementById("app");
if (!rootNode) {
  throw new Error("Missing #app root container");
}
createRoot(rootNode).render(html`<${App} />`);
