/**
 * Local API adapter — replaces HTTP-based API with Pyodide Web Worker calls.
 * Exposes the same interface as the original `API` object in main.jsx.
 */

import { unzipSync } from "fflate";
import catalogData from "./catalog.json";
import { fileDownload } from "./analytics.js";

const EXPECTED_FILES = new Set([
  "conversations.json",
  "projects.json",
  "users.json",
]);

/**
 * Send a message to the worker and wait for a "result" or "error" response.
 * Progress messages are forwarded to onProgress if provided.
 */
function workerCall(worker, msg, onProgress) {
  return new Promise((resolve, reject) => {
    const handler = (e) => {
      const { type, ...rest } = e.data;
      if (type === "result") {
        worker.removeEventListener("message", handler);
        resolve(rest.payload);
      } else if (type === "error") {
        worker.removeEventListener("message", handler);
        reject(new Error(rest.message));
      } else if (type === "progress" && onProgress) {
        onProgress(rest.stage, rest.detail);
      }
    };
    worker.addEventListener("message", handler);
    worker.postMessage(msg);
  });
}

/**
 * Extract the 3 expected JSON files from a ZIP File object.
 * Returns { "conversations.json": ArrayBuffer, ... }
 */
function extractZip(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed to read ZIP file"));
    reader.onload = () => {
      try {
        const zipData = new Uint8Array(reader.result);
        const entries = unzipSync(zipData);
        const result = {};

        for (const [path, data] of Object.entries(entries)) {
          // Match expected files at any nesting level
          const basename = path.split("/").pop();
          if (EXPECTED_FILES.has(basename) && !result[basename]) {
            result[basename] = data.buffer;
          }
        }

        const missing = [...EXPECTED_FILES].filter((f) => !result[f]);
        if (missing.length > 0) {
          reject(new Error(`Missing files in ZIP: ${missing.join(", ")}. Expected a Claude.ai export ZIP.`));
          return;
        }
        resolve(result);
      } catch (e) {
        reject(new Error(`ZIP extraction failed: ${e.message || e}`));
      }
    };
    reader.readAsArrayBuffer(file);
  });
}

/**
 * Create a local API adapter backed by a Pyodide Web Worker.
 *
 * @param {Worker} worker - initialized Pyodide worker
 * @returns API object with the same interface as the HTTP-based API
 */
export function createLocalAPI(worker) {
  // In-memory session state (single session)
  let session = null;

  // Detect IANA timezone from browser (pytz rejects non-IANA like "GMT+3")
  const browserTz = (() => {
    try {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
      // Only use if it looks like an IANA name (contains "/")
      if (tz && tz.includes("/")) return tz;
    } catch { /* ignore */ }
    return "UTC";
  })();

  return {
    /** List available analysis modules — returns instantly from bundled data. */
    async listModules() {
      return catalogData;
    },

    /**
     * Upload and process a ZIP file.
     * Performs: extract → load to FS → ETL → prepare → overview.
     */
    async uploadZip(file, onProgress) {
      if (!file.name.toLowerCase().endsWith(".zip")) {
        throw new Error("Only .zip files are accepted");
      }

      const P = (pct, stage) => onProgress && onProgress({ pct, stage });

      P(0.03, "extract");
      const files = await extractZip(file);

      P(0.08, "upload");
      await workerCall(worker, { type: "load_zip", files }, null);

      P(0.12, "etl");
      await workerCall(worker, { type: "run_etl" }, (stage) => {
        if (stage === "etl") P(0.30, "etl");
      });

      P(0.45, "prepare");
      await workerCall(worker, { type: "run_prepare", tz: browserTz }, (stage) => {
        if (stage === "prepare") P(0.60, "prepare");
      });

      P(0.85, "overview");
      const overview = await workerCall(worker, { type: "get_overview" });

      P(1.0, "done");

      const sid = "local-" + Date.now().toString(36);
      session = {
        session_id: sid,
        created_at: Date.now() / 1000,
        status: "ready",
        progress: { stage: "done" },
        error: null,
        overview,
        modules_ran: [],
      };
      return session;
    },

    async getSession(_sid) {
      if (!session) throw new Error("No active session");
      return session;
    },

    async startPrepare(_sid) {
      // In local mode, prepare runs as part of uploadZip.
      // This is a no-op called by the existing UI flow.
      if (!session) throw new Error("No active session");
      return session;
    },

    async runModule(_sid, key) {
      if (!session) throw new Error("No active session");
      const mod = catalogData.modules.find((m) => m.key === key);
      if (!mod) throw new Error(`Unknown module: ${key}`);

      const t0 = Date.now();
      const partial = await workerCall(
        worker,
        { type: "run_module", key, slug: mod.slug },
      );

      // Build full payload matching server format
      const payload = {
        key,
        slug: mod.slug,
        group: mod.group,
        title: mod.title,
        summary_json: partial.summary_json,
        sections: partial.sections,
        assets: partial.assets || [],
        images: [],
        csvs: partial.csvs || [],
        elapsed_seconds: (Date.now() - t0) / 1000,
        ran_at: Date.now() / 1000,
      };

      if (!session.modules_ran.includes(key)) {
        session.modules_ran.push(key);
      }
      return payload;
    },

    async getModuleResult(_sid, _key) {
      // In local mode, results are not cached server-side.
      return null;
    },

    async deleteSession(_sid) {
      session = null;
      return { ok: true };
    },

    resultUrl(_sid, _slug, _filename) {
      // In client-only mode, we return null — downloads use downloadFile() instead
      return null;
    },

    async downloadFile(slug, filename) {
      const path = `/results/${slug}/${filename}`;
      const buffer = await workerCall(worker, { type: "read_file", path });
      const ext = filename.split(".").pop()?.toLowerCase();
      const mimeMap = { csv: "text/csv", json: "application/json", md: "text/markdown", png: "image/png" };
      const blob = new Blob([buffer], { type: mimeMap[ext] || "application/octet-stream" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      fileDownload(ext || "unknown", slug);
    },
  };
}
