/**
 * IndexedDB-backed pending queue for live microphone chunks.
 *
 * Design intent:
 * - Survive page refresh/tab crash while preserving unsent audio windows.
 * - Keep API small and failure-tolerant (returns safe defaults on errors).
 */

const DB_NAME = "evidentia_live_audio";
const DB_VERSION = 1;
const STORE_NAME = "mic_pending_chunks";

let dbPromise = null;

function supportsIndexedDb() {
  return typeof indexedDB !== "undefined";
}

function normalizeKind(kind) {
  const raw = String(kind || "").trim().toLowerCase();
  if (!raw) {
    return "asr_window";
  }
  return raw;
}

function buildRecordId(sessionId, seq, kind = "asr_window") {
  return `${String(sessionId || "").trim()}:${normalizeKind(kind)}:${String(seq)}`;
}

function openDb() {
  if (!supportsIndexedDb()) {
    return Promise.resolve(null);
  }
  if (dbPromise) {
    return dbPromise;
  }
  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("session_id", "session_id", { unique: false });
        store.createIndex("created_at", "created_at", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("indexeddb_open_failed"));
  }).catch((error) => {
    dbPromise = null;
    throw error;
  });
  return dbPromise;
}

function readAllBySession(store, sessionId) {
  return new Promise((resolve, reject) => {
    if (!store.indexNames.contains("session_id")) {
      const request = store.getAll();
      request.onsuccess = () => {
        const rows = Array.isArray(request.result) ? request.result : [];
        resolve(rows.filter((item) => String(item?.session_id || "") === sessionId));
      };
      request.onerror = () => reject(request.error || new Error("indexeddb_getall_failed"));
      return;
    }
    const index = store.index("session_id");
    const request = index.getAll(IDBKeyRange.only(sessionId));
    request.onsuccess = () => resolve(Array.isArray(request.result) ? request.result : []);
    request.onerror = () => reject(request.error || new Error("indexeddb_index_getall_failed"));
  });
}

export async function persistPendingMicChunk(sessionId, chunk, options = {}) {
  try {
    const db = await openDb();
    if (!db) {
      return false;
    }
    const normalizedSession = String(sessionId || "").trim();
    const seq = Number(chunk?.seq);
    const kind = normalizeKind(options?.kind || chunk?.kind || "asr_window");
    if (!normalizedSession || !Number.isFinite(seq)) {
      return false;
    }
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const nowMs = Date.now();
      store.put({
        id: buildRecordId(normalizedSession, seq, kind),
        session_id: normalizedSession,
        kind,
        seq,
        mime_type: String(chunk?.mime_type || ""),
        data_b64: String(chunk?.data_b64 || ""),
        window_start_sec: Number(chunk?.window_start_sec || 0),
        window_duration_sec: Number(chunk?.window_duration_sec || 0),
        created_at: nowMs,
        updated_at: nowMs,
      });
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error || new Error("indexeddb_put_failed"));
      tx.onabort = () => reject(tx.error || new Error("indexeddb_put_aborted"));
    });
    return true;
  } catch (_) {
    return false;
  }
}

export async function removePendingMicChunk(sessionId, seq, options = {}) {
  try {
    const db = await openDb();
    if (!db) {
      return false;
    }
    const normalizedSession = String(sessionId || "").trim();
    const normalizedSeq = Number(seq);
    const kind = normalizeKind(options?.kind || "asr_window");
    if (!normalizedSession || !Number.isFinite(normalizedSeq)) {
      return false;
    }
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      store.delete(buildRecordId(normalizedSession, normalizedSeq, kind));
      // Legacy id without kind separator (from DB v1 prior schema extension).
      store.delete(`${normalizedSession}:${String(normalizedSeq)}`);
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error || new Error("indexeddb_delete_failed"));
      tx.onabort = () => reject(tx.error || new Error("indexeddb_delete_aborted"));
    });
    return true;
  } catch (_) {
    return false;
  }
}

export async function listPendingMicChunks(sessionId, options = {}) {
  try {
    const db = await openDb();
    if (!db) {
      return [];
    }
    const normalizedSession = String(sessionId || "").trim();
    const kind = normalizeKind(options?.kind || "asr_window");
    if (!normalizedSession) {
      return [];
    }

    const records = await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      readAllBySession(store, normalizedSession).then(resolve).catch(reject);
      tx.onerror = () => reject(tx.error || new Error("indexeddb_list_failed"));
      tx.onabort = () => reject(tx.error || new Error("indexeddb_list_aborted"));
    });

    return (Array.isArray(records) ? records : [])
      .filter((item) => {
        const itemKind = normalizeKind(item?.kind || "asr_window");
        return itemKind === kind;
      })
      .sort((a, b) => {
        const aSeq = Number(a?.seq || 0);
        const bSeq = Number(b?.seq || 0);
        if (aSeq !== bSeq) {
          return aSeq - bSeq;
        }
        return Number(a?.created_at || 0) - Number(b?.created_at || 0);
      })
      .map((item) => ({
        kind: normalizeKind(item?.kind || "asr_window"),
        seq: Number(item?.seq || 0),
        mime_type: String(item?.mime_type || ""),
        data_b64: String(item?.data_b64 || ""),
        window_start_sec: Number(item?.window_start_sec || 0),
        window_duration_sec: Number(item?.window_duration_sec || 0),
        created_at: Number(item?.created_at || 0),
      }));
  } catch (_) {
    return [];
  }
}

export async function clearPendingMicChunksForSession(sessionId) {
  try {
    const db = await openDb();
    if (!db) {
      return 0;
    }
    const normalizedSession = String(sessionId || "").trim();
    if (!normalizedSession) {
      return 0;
    }
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      readAllBySession(store, normalizedSession)
        .then((rows) => {
          const items = Array.isArray(rows) ? rows : [];
          for (const row of items) {
            if (row?.id) {
              store.delete(row.id);
            }
          }
          tx.oncomplete = () => resolve(items.length);
          tx.onerror = () => reject(tx.error || new Error("indexeddb_clear_failed"));
          tx.onabort = () => reject(tx.error || new Error("indexeddb_clear_aborted"));
        })
        .catch(reject);
    });
  } catch (_) {
    return 0;
  }
}

export async function cleanupPendingMicChunks(sessionId, { ttlMs = 0, maxRecords = 0 } = {}) {
  try {
    const db = await openDb();
    if (!db) {
      return { deleted: 0, remaining: 0 };
    }
    const normalizedSession = String(sessionId || "").trim();
    if (!normalizedSession) {
      return { deleted: 0, remaining: 0 };
    }
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      readAllBySession(store, normalizedSession)
        .then((rows) => {
          const nowMs = Date.now();
          const allRows = (Array.isArray(rows) ? rows : []).sort(
            (a, b) => Number(a?.created_at || 0) - Number(b?.created_at || 0),
          );

          let deleted = 0;
          const survivors = [];
          for (const row of allRows) {
            const createdAt = Number(row?.created_at || 0);
            const expired = Number(ttlMs) > 0 && createdAt > 0 && (nowMs - createdAt) > Number(ttlMs);
            if (expired && row?.id) {
              store.delete(row.id);
              deleted += 1;
            } else {
              survivors.push(row);
            }
          }

          const keepLimit = Math.max(0, Math.floor(Number(maxRecords) || 0));
          if (keepLimit > 0 && survivors.length > keepLimit) {
            const overflow = survivors.length - keepLimit;
            for (let i = 0; i < overflow; i += 1) {
              const row = survivors[i];
              if (row?.id) {
                store.delete(row.id);
                deleted += 1;
              }
            }
            resolve({ deleted, remaining: keepLimit });
            return;
          }

          resolve({ deleted, remaining: survivors.length });
        })
        .catch(reject);
      tx.onerror = () => reject(tx.error || new Error("indexeddb_cleanup_failed"));
      tx.onabort = () => reject(tx.error || new Error("indexeddb_cleanup_aborted"));
    });
  } catch (_) {
    return { deleted: 0, remaining: 0 };
  }
}
