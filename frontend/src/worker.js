/**
 * Pyodide Web Worker — runs all Python analysis in the browser.
 *
 * Message protocol:
 *   Main → Worker:
 *     { type: "init" }
 *     { type: "load_zip", files: { "conversations.json": ArrayBuffer, ... } }
 *     { type: "run_etl" }
 *     { type: "run_prepare", tz: "UTC" }
 *     { type: "get_overview" }
 *     { type: "run_module", key: "m04", slug: "04-activity-rhythm" }
 *
 *   Worker → Main:
 *     { type: "progress", stage: "...", detail: "..." }
 *     { type: "init_ready" }
 *     { type: "result", payload: {...} }
 *     { type: "error", message: "..." }
 */

/* global importScripts, loadPyodide */

// -- Bundled Python scripts (Vite raw imports) --------------------------------
// import.meta.glob returns { "./relative/path": "raw content" }
const scriptModules = import.meta.glob(
  "../../scripts/**/*.py",
  { eager: true, query: "?raw", import: "default" },
);

// Flatten to { "etl.py": "...", "analysis/_common.py": "...", ... }
const SCRIPTS = {};
for (const [viteKey, raw] of Object.entries(scriptModules)) {
  // viteKey looks like "../../scripts/etl.py" or "../../scripts/analysis/m01_*.py"
  const relative = viteKey.replace(/^.*?\/scripts\//, "");
  SCRIPTS[relative] = raw;
}

let pyodide = null;

// -- Helpers ------------------------------------------------------------------

function progress(stage, detail = "") {
  postMessage({ type: "progress", stage, detail });
}

function error(message) {
  postMessage({ type: "error", message: String(message) });
}

// -- Init ---------------------------------------------------------------------

async function init() {
  try {
    progress("pyodide_boot", "Loading Python runtime…");
    const { loadPyodide: _loadPyodide } = await import(
      /* @vite-ignore */ "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs"
    );
    pyodide = await _loadPyodide();

    progress("packages", "Installing packages…");
    await pyodide.loadPackage(["micropip", "numpy"]);
    const micropip = pyodide.pyimport("micropip");
    await micropip.install(["duckdb", "tabulate", "pytz"]);

    const scriptKeys = Object.keys(SCRIPTS);
    progress("scripts", `Mounting ${scriptKeys.length} scripts…`);
    if (scriptKeys.length === 0) {
      throw new Error("No Python scripts were bundled. Check Vite glob config.");
    }
    // Create directory structure in virtual FS
    try { pyodide.FS.mkdir("/scripts"); } catch { /* exists */ }
    const dirs = new Set();
    for (const path of scriptKeys) {
      const parts = path.split("/");
      for (let i = 1; i < parts.length; i++) {
        dirs.add("/scripts/" + parts.slice(0, i).join("/"));
      }
    }
    for (const dir of [...dirs].sort()) {
      try { pyodide.FS.mkdir(dir); } catch { /* exists */ }
    }
    // Write files
    for (const [path, content] of Object.entries(SCRIPTS)) {
      pyodide.FS.writeFile(`/scripts/${path}`, content);
    }

    // Create data and results directories
    for (const dir of ["/data", "/db", "/results"]) {
      try { pyodide.FS.mkdir(dir); } catch { /* exists */ }
    }

    // Add scripts root to sys.path
    pyodide.runPython(`
import sys
sys.path.insert(0, "/scripts")
sys.path.insert(0, "/scripts/analysis")
`);

    progress("ready");
    postMessage({ type: "init_ready" });
  } catch (e) {
    const msg = e instanceof Error ? e.message : (typeof e === "string" ? e : JSON.stringify(e, null, 2));
    error(`init failed: ${msg}`);
  }
}

// -- ZIP loading --------------------------------------------------------------

function loadZip(files) {
  try {
    for (const [name, buffer] of Object.entries(files)) {
      pyodide.FS.writeFile(`/data/${name}`, new Uint8Array(buffer));
    }
    postMessage({ type: "result", payload: { ok: true } });
  } catch (e) {
    error(`load_zip failed: ${e.message || e}`);
  }
}

// -- ETL ----------------------------------------------------------------------

async function runEtl() {
  try {
    progress("etl", "Running ETL…");
    await pyodide.runPythonAsync(`
from pathlib import Path
from etl import run_etl

counts = run_etl(
    Path("/data/conversations.json"),
    Path("/data/projects.json"),
    Path("/data/users.json"),
    Path("/db/claude.duckdb"),
)
`);
    const counts = pyodide.globals.get("counts").toJs();
    postMessage({ type: "result", payload: { ok: true, counts: Object.fromEntries(counts) } });
  } catch (e) {
    error(`ETL failed: ${e.message || e}`);
  }
}

// -- Prepare ------------------------------------------------------------------

async function runPrepare(tz = "UTC") {
  try {
    progress("prepare", "Building statistics…");
    // Force DuckDB to use UTC internally to avoid pytz issues with non-IANA zones
    await pyodide.runPythonAsync(`
import duckdb, os
from analysis._prepare import ensure

os.environ["TZ"] = "UTC"
con = duckdb.connect("/db/claude.duckdb")
con.execute("SET TimeZone = 'UTC'")
ensure(con, tz="${tz}")
con.close()
`);
    postMessage({ type: "result", payload: { ok: true } });
  } catch (e) {
    error(`prepare failed: ${e.message || e}`);
  }
}

// -- Overview -----------------------------------------------------------------

async function getOverview() {
  try {
    await pyodide.runPythonAsync(`
import duckdb, json

def _load_overview(con):
    con.execute("SET TimeZone = 'UTC'")
    q_scope = con.execute("""
        SELECT
          COUNT(*)                             AS conversations,
          SUM(message_count)                   AS messages,
          CAST(MIN(first_msg_at) AS VARCHAR)   AS first_msg_at,
          CAST(MAX(last_msg_at) AS VARCHAR)    AS last_msg_at,
          AVG(message_count)                   AS msg_mean,
          MAX(message_count)                   AS msg_max
        FROM _stats_conversation
        WHERE message_count > 0
    """).fetchone()
    n_conv, n_msg, first_at, last_at, mean, maxc = q_scope
    q_proj = con.execute("""
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN is_private THEN 1 ELSE 0 END) AS n_private,
               SUM(CASE WHEN prompt_template IS NOT NULL AND LENGTH(prompt_template) > 0 THEN 1 ELSE 0 END) AS n_filled
        FROM projects
    """).fetchone()
    n_proj, n_private, n_filled = q_proj
    q_docs = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    q_senders = con.execute("SELECT sender, COUNT(*) FROM messages GROUP BY 1").fetchall()
    q_blocks = con.execute("SELECT type, COUNT(*) FROM content_blocks GROUP BY 1 ORDER BY 2 DESC").fetchall()
    q_monthly = con.execute("""
        SELECT strftime(created_local, '%Y-%m') AS ym, COUNT(*)
        FROM _stats_conversation WHERE message_count > 0
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    return {
        "scope": {
            "conversations": int(n_conv or 0), "messages": int(n_msg or 0),
            "first_msg_at": str(first_at) if first_at else None,
            "last_msg_at": str(last_at) if last_at else None,
            "msg_mean": float(mean or 0), "msg_max": int(maxc or 0),
        },
        "projects": {
            "total": int(n_proj or 0), "private": int(n_private or 0),
            "with_instructions": int(n_filled or 0), "docs": int(q_docs or 0),
        },
        "senders": [{"sender": s, "count": int(n)} for s, n in q_senders],
        "content_blocks": [{"type": t, "count": int(n)} for t, n in q_blocks],
        "monthly": [{"month": m, "conversations": int(n)} for m, n in q_monthly],
    }

con = duckdb.connect("/db/claude.duckdb", read_only=True)
_overview_result = json.dumps(_load_overview(con))
con.close()
`);
    const raw = pyodide.globals.get("_overview_result");
    postMessage({ type: "result", payload: JSON.parse(raw) });
  } catch (e) {
    error(`overview failed: ${e.message || e}`);
  }
}

// -- Module run ---------------------------------------------------------------

async function runModule(key, slug) {
  try {
    progress("module", `Running ${key}…`);
    pyodide.globals.set("_mod_key", key);
    pyodide.globals.set("_mod_slug", slug);
    await pyodide.runPythonAsync(`
import importlib, duckdb, json, math, os
from pathlib import Path
from analysis import _common

os.environ["TZ"] = "UTC"
_common.setup_matplotlib()

_mod_slug_val = str(_mod_slug)
_mod_key_val = str(_mod_key)

# Resolve module name from key
_module_map = {}
import os
for f in os.listdir("/scripts/analysis"):
    if f.startswith("m") and f.endswith(".py") and "_" in f:
        k = f.split("_")[0]  # "m01"
        _module_map[k] = f[:-3]  # "m01_conversation_length"

_mod_name = _module_map.get(_mod_key_val)
if not _mod_name:
    raise ValueError(f"Unknown module: {_mod_key_val}")

mod = importlib.import_module(f"analysis.{_mod_name}")

out_dir = Path(f"/results/{_mod_slug_val}")
out_dir.mkdir(parents=True, exist_ok=True)

con = duckdb.connect("/db/claude.duckdb", read_only=False)
con.execute("SET TimeZone = 'UTC'")
summary = mod.run(con, out_dir, {"tz": "UTC"})
con.close()

# Read sections.json if written
sections = None
sp = out_dir / "sections.json"
if sp.exists():
    sections = json.loads(sp.read_text())["sections"]

# Sanitize NaN/Inf
def _safe(obj):
    if obj is None:
        return None
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return obj

summary_safe = _safe(summary)
sections_safe = _safe(sections) if sections else None

# Localize
summary_payload = _common.localize_ui_payload(summary_safe)
sections_payload = _common.localize_ui_payload(sections_safe) if sections_safe else None

# Scan output directory for assets and CSVs
_assets = []
_csvs = []
for f in sorted(out_dir.iterdir()):
    if f.suffix == ".png":
        _assets.append({"name": f.name, "kind": "legacy_png", "size_bytes": f.stat().st_size})
    elif f.suffix == ".csv":
        _csvs.append({"name": f.name, "size_bytes": f.stat().st_size})

_module_result = json.dumps({
    "summary_json": summary_payload,
    "sections": sections_payload,
    "assets": _assets,
    "csvs": _csvs,
})
`);
    const raw = pyodide.globals.get("_module_result");
    postMessage({ type: "result", payload: JSON.parse(raw) });
  } catch (e) {
    error(`module ${key} failed: ${e.message || e}`);
  }
}

// -- File read (for downloads) ------------------------------------------------

function readFile(path) {
  try {
    // List what's actually in the directory for debugging
    const dir = path.substring(0, path.lastIndexOf("/"));
    let listing = [];
    try { listing = pyodide.FS.readdir(dir).filter((f) => f !== "." && f !== ".."); } catch { /* dir missing */ }

    let data;
    try {
      data = pyodide.FS.readFile(path, { encoding: "binary" });
    } catch {
      error(`read_file: not found — ${path} (dir contains: ${listing.join(", ") || "empty/missing"})`);
      return;
    }
    const copy = new Uint8Array(data).buffer;
    postMessage({ type: "result", payload: copy }, [copy]);
  } catch (e) {
    const msg = e instanceof Error ? e.message : JSON.stringify(e);
    error(`read_file failed: ${path} — ${msg}`);
  }
}

// -- Message handler ----------------------------------------------------------

self.onmessage = async (e) => {
  const { type, ...data } = e.data;
  switch (type) {
    case "init":
      await init();
      break;
    case "load_zip":
      loadZip(data.files);
      break;
    case "run_etl":
      await runEtl();
      break;
    case "run_prepare":
      await runPrepare(data.tz || "UTC");
      break;
    case "get_overview":
      await getOverview();
      break;
    case "run_module":
      await runModule(data.key, data.slug);
      break;
    case "read_file":
      readFile(data.path);
      break;
    default:
      error(`unknown message type: ${type}`);
  }
};
