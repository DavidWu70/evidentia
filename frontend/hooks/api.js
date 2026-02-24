/**
 * Thin API client for MVP orchestration.
 *
 * Design intent:
 * - Keep transport logic centralized and predictable.
 * - Preserve payload shape parity with backend contracts.
 * - Make endpoint swapping low-risk for iteration.
 */

export class ApiClient {
  constructor(baseUrl) {
    this.baseUrl = (baseUrl || "http://127.0.0.1:8000").replace(/\/$/, "");
  }

  setBaseUrl(baseUrl) {
    this.baseUrl = (baseUrl || this.baseUrl).replace(/\/$/, "");
  }

  async transcriptIncremental(payload) {
    return this.#post("/transcript/incremental", payload);
  }

  async transcribeStructured(payload) {
    return this.#post("/transcribe_structured", payload);
  }

  async eventsExtract(payload) {
    return this.#post("/events/extract", payload);
  }

  async stateSnapshot(payload) {
    return this.#post("/state/snapshot", payload);
  }

  async noteDraft(payload) {
    return this.#post("/note/draft", payload);
  }

  async startNoteDraftJob(payload) {
    return this.#post("/note/draft/jobs/start", payload);
  }

  async getNoteDraftJob(jobId) {
    const jid = encodeURIComponent(String(jobId || ""));
    return this.#get(`/note/draft/jobs/${jid}`);
  }

  async stopNoteDraftJob(jobId) {
    const jid = encodeURIComponent(String(jobId || ""));
    return this.#post(`/note/draft/jobs/${jid}/stop`, {});
  }

  async noteTemplates() {
    return this.#get("/note/templates");
  }

  async noteTemplateDocument(department, templateId) {
    const dep = encodeURIComponent(String(department || ""));
    const tid = encodeURIComponent(String(templateId || ""));
    return this.#get(`/note/templates/${dep}/${tid}`);
  }

  async liveAudioTranscript(sessionId) {
    const sid = encodeURIComponent(String(sessionId || ""));
    return this.#get(`/audio/live/transcript/${sid}`);
  }

  async uploadAudioFile(file) {
    const filename = encodeURIComponent(String(file?.name || "audio.wav"));
    const response = await fetch(`${this.baseUrl}/files/upload-audio?filename=${filename}`, {
      method: "POST",
      headers: {
        "Content-Type": String(file?.type || "application/octet-stream"),
      },
      body: file,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`/files/upload-audio failed (${response.status}): ${text}`);
    }
    return response.json();
  }

  async listSampleAudioFiles() {
    return this.#get("/files/sample-audio");
  }

  async saveNoteTemplateDocument(department, templateId, template) {
    const dep = encodeURIComponent(String(department || ""));
    const tid = encodeURIComponent(String(templateId || ""));
    return this.#put(`/note/templates/${dep}/${tid}`, { template });
  }

  async #post(path, payload) {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`${path} failed (${response.status}): ${text}`);
    }

    return response.json();
  }

  async #get(path) {
    const response = await fetch(`${this.baseUrl}${path}`);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`${path} failed (${response.status}): ${text}`);
    }
    return response.json();
  }

  async #put(path, payload) {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`${path} failed (${response.status}): ${text}`);
    }

    return response.json();
  }
}
