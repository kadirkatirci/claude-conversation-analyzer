import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import {
  I18N,
  InlineText,
  KpiTile,
  cls,
  fmtNum,
  fmtPct,
  fmtVal,
  pick,
  t,
  useLang,
} from "./runtime.jsx";
import "./ui.css";
import { createLocalAPI } from "./api-local.js";
import * as analytics from "./analytics.js";
import PyodideLoader from "./pyodide-loader.jsx";
import AnalysisWorker from "./worker.js?worker";

const ModuleResult = React.lazy(() => import("./result-view.jsx"));

const LS_THEME = "cca:theme";
const THEMES = ["system", "light", "dark"];

function resolveTheme(pref) {
  if (pref === "system") {
    return (typeof matchMedia !== "undefined" && matchMedia("(prefers-color-scheme: dark)").matches)
      ? "dark" : "light";
  }
  return pref;
}
const START_PATHS = [
  { key: "rhythm", module: "m04" },
  { key: "depth", module: "m01" },
  { key: "tools", module: "m05" },
  { key: "making", module: "m09" },
];
const RECOMMENDED_MODULES = START_PATHS.map((p) => p.module);

// Singleton worker + API — initialized once
let _worker = null;
let _api = null;
let _initPromise = null;

function getAPI() {
  if (_api) return { api: _api, worker: _worker, promise: _initPromise };
  _worker = new AnalysisWorker();
  _api = createLocalAPI(_worker);
  const t0 = performance.now();
  _initPromise = new Promise((resolve, reject) => {
    const handler = (e) => {
      if (e.data.type === "init_ready") {
        _worker.removeEventListener("message", handler);
        analytics.engineLoaded(performance.now() - t0);
        resolve();
      } else if (e.data.type === "error") {
        _worker.removeEventListener("message", handler);
        reject(new Error(e.data.message));
      }
    };
    _worker.addEventListener("message", handler);
    _worker.postMessage({ type: "init" });
  });
  return { api: _api, worker: _worker, promise: _initPromise };
}

function safeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function formatDateOnly(value) {
  if (!value) return "—";
  return String(value).slice(0, 10);
}

function monthSpan(firstValue, lastValue) {
  if (!firstValue || !lastValue) return 0;
  const first = new Date(firstValue);
  const last = new Date(lastValue);
  if (Number.isNaN(first.getTime()) || Number.isNaN(last.getTime())) return 0;
  return Math.max(1, (last.getFullYear() - first.getFullYear()) * 12 + last.getMonth() - first.getMonth() + 1);
}

function parseMonth(value) {
  const m = /^(\d{4})-(\d{1,2})/.exec(String(value || ""));
  if (!m) return null;
  const year = Number(m[1]);
  const month = Number(m[2]);
  if (!Number.isFinite(year) || !Number.isFinite(month)) return null;
  return new Date(year, month - 1, 1);
}

function formatMonthShort(value) {
  const date = parseMonth(value);
  if (!date) return String(value || "");
  return new Intl.DateTimeFormat(I18N.getLang() === "tr" ? "tr-TR" : "en-US", {
    month: "short",
    year: "2-digit",
  }).format(date);
}

function formatMonthLong(value) {
  const date = parseMonth(value);
  if (!date) return String(value || "—");
  return new Intl.DateTimeFormat(I18N.getLang() === "tr" ? "tr-TR" : "en-US", {
    month: "long",
    year: "numeric",
  }).format(date);
}

function metricFromRows(rows, names) {
  const targets = new Set(names.map((name) => String(name).toLowerCase()));
  const row = (rows || []).find((item) => {
    const label = String(item.type ?? item.sender ?? item.label ?? item.name ?? "").toLowerCase();
    return targets.has(label);
  });
  return safeNumber(row?.count ?? row?.value, 0);
}

function normalizeMonthly(monthly) {
  return (monthly || []).map((row) => ({
    ...row,
    month: row.month ?? row.label,
    conversations: safeNumber(row.conversations ?? row.count ?? row.value, 0),
  })).filter((row) => row.month);
}

function buildOverviewPortrait(overview) {
  const scope = overview?.scope || {};
  const projects = overview?.projects || {};
  const senders = overview?.senders || [];
  const blocks = overview?.content_blocks || [];
  const monthly = normalizeMonthly(overview?.monthly);
  const peak = monthly.reduce((best, row) => (
    safeNumber(row.conversations) > safeNumber(best?.conversations) ? row : best
  ), null);
  const human = metricFromRows(senders, ["human", "user"]);
  const assistant = metricFromRows(senders, ["assistant"]);
  const senderTotal = (senders || []).reduce((sum, row) => sum + safeNumber(row.count ?? row.value, 0), 0);
  const toolUse = metricFromRows(blocks, ["tool_use"]);
  const thinking = metricFromRows(blocks, ["thinking"]);

  return {
    scope,
    projects,
    senders,
    blocks,
    monthly,
    first: formatDateOnly(scope.first_msg_at),
    last: formatDateOnly(scope.last_msg_at),
    months: monthSpan(scope.first_msg_at, scope.last_msg_at),
    peak,
    human,
    assistant,
    senderTotal,
    assistantShare: senderTotal ? (assistant / senderTotal) * 100 : null,
    toolUse,
    thinking,
  };
}

const UPLOAD_STAGES = [
  { key: "extract", icon: "📦" },
  { key: "upload", icon: "💾" },
  { key: "etl", icon: "🔄" },
  { key: "prepare", icon: "📊" },
  { key: "overview", icon: "🔍" },
  { key: "done", icon: "✓" },
];

function stageName(key, lang) {
  const names = {
    extract: { en: "Extracting ZIP", tr: "ZIP açılıyor" },
    upload: { en: "Loading files", tr: "Dosyalar yükleniyor" },
    etl: { en: "Building database", tr: "Veritabanı oluşturuluyor" },
    prepare: { en: "Computing statistics", tr: "İstatistikler hesaplanıyor" },
    overview: { en: "Generating overview", tr: "Genel bakış hazırlanıyor" },
    done: { en: "Ready", tr: "Hazır" },
  };
  return names[key]?.[lang] || names[key]?.en || key;
}

function Uploader({ onSession }) {
  const lang = useLang();
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState({ pct: 0, stage: null });
  const [err, setErr] = useState(null);

  const handleFile = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".zip")) {
      setErr(t("upload.error.zip_only"));
      return;
    }
    setErr(null);
    setBusy(true);
    setProgress({ pct: 0, stage: null });
    const t0 = performance.now();
    try {
      const result = await _api.uploadZip(file, setProgress);
      const ov = result.overview?.scope || {};
      analytics.zipUploaded({
        fileSizeMb: +(file.size / 1048576).toFixed(1),
        conversations: ov.conversations || 0,
        messages: ov.messages || 0,
        durationMs: performance.now() - t0,
      });
      onSession(result);
    } catch (e) {
      analytics.zipError(e.message || e, progress.stage);
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const activeIdx = UPLOAD_STAGES.findIndex((s) => s.key === progress.stage);
  const pctDisplay = Math.round((progress.pct || 0) * 100);

  return (
    <section className="landing">
      <div className="eyebrow"><span />{t("upload.eyebrow")}</div>
      <h1>
        {t("upload.heading.pre")} <em>{t("upload.heading.em")}</em> {t("upload.heading.post")}
      </h1>
      <p>{t("upload.lead")}</p>
      {!busy && (
        <label
          className={cls("dropzone", dragging && "dragging")}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFile(e.dataTransfer.files?.[0]);
          }}
        >
          <input
            className="sr-only"
            type="file"
            accept=".zip,application/zip,application/x-zip-compressed"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          <span className="dropzone-title">{t("upload.dropzone.idle")}</span>
          <span className="dropzone-sub">{t("upload.dropzone.hint")}</span>
        </label>
      )}
      {busy && (
        <div className="upload-progress">
          <div className="upload-progress-bar">
            <div className="upload-progress-fill" style={{ width: `${pctDisplay}%` }} />
          </div>
          <div className="upload-progress-pct">{pctDisplay}%</div>
          <div className="upload-stages">
            {UPLOAD_STAGES.filter((s) => s.key !== "done").map((s, i) => (
              <div
                key={s.key}
                className={cls(
                  "upload-stage",
                  i < activeIdx && "done",
                  i === activeIdx && "active",
                )}
              >
                <span className="upload-stage-icon">
                  {i < activeIdx ? "✓" : s.icon}
                </span>
                <span>{stageName(s.key, lang)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {err && <div className="err-box">{err}</div>}
    </section>
  );
}

const PREPARE_STEPS = ["etl:users", "etl:projects", "etl:conversations", "prepare", "done"];

function PrepareView({ session, onReady }) {
  useLang();
  const [state, setState] = useState(session);

  useEffect(() => {
    let alive = true;
    let timer;
    const tick = async () => {
      try {
        const next = await _api.getSession(session.session_id);
        if (!alive) return;
        setState(next);
        if (next.status === "ready") { onReady(next); return; }
        if (next.status === "error") return;
      } catch {}
      timer = setTimeout(tick, 700);
    };
    tick();
    return () => { alive = false; if (timer) clearTimeout(timer); };
  }, [session.session_id, onReady]);

  const stage = state.progress?.stage || "starting";
  const idx = PREPARE_STEPS.findIndex((s) => stage.startsWith(s));

  return (
    <section className="prepare">
      <div className="eyebrow"><span />{t("prepare.session", { sid: state.session_id })}</div>
      <h1>{t("prepare.heading")}</h1>
      <p>{t("prepare.lead")}</p>
      <div className="prepare-steps">
        {PREPARE_STEPS.map((step, i) => (
          <div key={step} className={cls("prepare-step", i < idx && "done", i === idx && "active")}>
            <span>{t(`prepare.step.${step}`)}</span>
            <span>{i < idx ? "✓" : i === idx ? "..." : ""}</span>
          </div>
        ))}
      </div>
      {state.error && <div className="err-box">{t("prepare.error", { msg: state.error })}</div>}
    </section>
  );
}

function SessionBar({ session, onDrop }) {
  useLang();
  return (
    <div className="session-bar">
      <div>
        <span>{t("session.label")}</span>
        <code>{session.session_id}</code>
        {session.overview && (
          <small>
            {t("session.conversations", { n: fmtNum(session.overview.scope.conversations) })}
            {" · "}
            {t("session.messages", { n: fmtNum(session.overview.scope.messages) })}
          </small>
        )}
      </div>
      <button type="button" onClick={onDrop}>{t("session.drop")}</button>
    </div>
  );
}

function Overview({ overview }) {
  const lang = useLang();
  const portrait = useMemo(() => buildOverviewPortrait(overview), [overview, lang]);
  if (!overview) return null;
  const { scope, projects, senders, blocks, monthly, first, last, months, peak } = portrait;
  const insights = [
    peak && {
      key: "peak",
      title: t("overview.signal.peak.title"),
      value: formatMonthLong(peak.month),
      text: t("overview.signal.peak.text", { n: fmtNum(peak.conversations) }),
    },
    safeNumber(scope.msg_max) > 0 && {
      key: "depth",
      title: t("overview.signal.depth.title"),
      value: fmtNum(scope.msg_max),
      text: t("overview.signal.depth.text", { avg: fmtVal(scope.msg_mean, 1) }),
    },
    (portrait.toolUse > 0 || portrait.thinking > 0) && {
      key: "work",
      title: t("overview.signal.work.title"),
      value: fmtNum(Math.max(portrait.toolUse, portrait.thinking)),
      text: t("overview.signal.work.text", {
        tools: fmtNum(portrait.toolUse),
        thinking: fmtNum(portrait.thinking),
      }),
    },
    (safeNumber(projects.total) > 0 || safeNumber(projects.docs) > 0) && {
      key: "knowledge",
      title: t("overview.signal.knowledge.title"),
      value: t("overview.signal.knowledge.value", {
        projects: fmtNum(projects.total || 0),
        docs: fmtNum(projects.docs || 0),
      }),
      text: t("overview.signal.knowledge.text"),
    },
    portrait.senderTotal > 0 && {
      key: "collab",
      title: t("overview.signal.collab.title"),
      value: portrait.assistantShare == null ? fmtNum(portrait.senderTotal) : fmtPct(portrait.assistantShare),
      text: t("overview.signal.collab.text", {
        human: fmtNum(portrait.human),
        assistant: fmtNum(portrait.assistant),
      }),
    },
  ].filter(Boolean).slice(0, 4);

  return (
    <section className="overview">
      <div className="portrait-hero">
        <div className="portrait-copy">
          <div className="eyebrow portrait-eyebrow"><span />{t("overview.portrait.eyebrow")}</div>
          <h1>
            <InlineText>
              {t("overview.portrait.heading", {
                conv: fmtNum(scope.conversations),
                msg: fmtNum(scope.messages),
                months: fmtNum(months),
              })}
            </InlineText>
          </h1>
          <p className="portrait-lead">
            {t("overview.portrait.lead", { first, last })}
          </p>
          <div className="portrait-stats" aria-label={t("overview.portrait.stats")}>
            <span>
              <strong>{first}</strong>
              <small>{t("overview.first")}</small>
            </span>
            <span>
              <strong>{last}</strong>
              <small>{t("overview.last")}</small>
            </span>
            <span>
              <strong>{fmtVal(scope.msg_mean, 1)}</strong>
              <small>{t("overview.avg")}</small>
            </span>
          </div>
        </div>
        <PortraitTimeline monthly={monthly} peak={peak} />
      </div>
      {insights.length > 0 && (
        <div className="signal-grid">
          {insights.map((item) => <SignalCard key={item.key} item={item} />)}
        </div>
      )}
      <details className="overview-details">
        <summary>{t("overview.details")}</summary>
        <div className="kpi-grid">
          <KpiTile label={t("overview.first")} value={first} compact />
          <KpiTile label={t("overview.last")} value={last} compact />
          <KpiTile label={t("overview.avg")} value={fmtVal(scope.msg_mean, 1)} compact />
          <KpiTile label={t("overview.max")} value={scope.msg_max} compact />
          <KpiTile label={t("overview.projects")} value={projects.total} compact />
          <KpiTile label={t("overview.docs")} value={projects.docs} compact />
        </div>
        <div className="overview-panels">
          <MiniBars title={t("overview.senders")} rows={(senders || []).map((s) => ({ label: s.sender, value: s.count }))} />
          <MiniBars title={t("overview.blocks")} rows={(blocks || []).map((b) => ({ label: b.type, value: b.count }))} />
        </div>
      </details>
    </section>
  );
}

function SignalCard({ item }) {
  return (
    <article className="signal-card">
      <div className="signal-title">{item.title}</div>
      <div className="signal-value">{item.value}</div>
      <p>{item.text}</p>
    </article>
  );
}

function buildTimelinePoints(monthly, width, height, paddingX, paddingTop, paddingBottom) {
  const usableHeight = height - paddingTop - paddingBottom;
  const max = Math.max(...monthly.map((row) => safeNumber(row.conversations)), 1);
  const step = monthly.length > 1 ? (width - paddingX * 2) / (monthly.length - 1) : 0;
  return {
    max,
    baseline: height - paddingBottom,
    points: monthly.map((row, index) => {
      const value = safeNumber(row.conversations);
      return {
        ...row,
        value,
        x: Number((paddingX + step * index).toFixed(2)),
        y: Number((paddingTop + usableHeight - (value / max) * usableHeight).toFixed(2)),
      };
    }),
  };
}

function PortraitTimeline({ monthly, peak }) {
  const data = monthly || [];
  const width = 560;
  const height = 240;
  const paddingX = 16;
  const paddingTop = 18;
  const paddingBottom = 24;
  const gradientId = "portrait-area-gradient";
  const { baseline, points } = data.length > 0
    ? buildTimelinePoints(data, width, height, paddingX, paddingTop, paddingBottom)
    : { baseline: height - paddingBottom, points: [] };
  const linePath = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  const areaPath = points.length
    ? [`M ${points[0].x} ${baseline}`, ...points.map((point) => `L ${point.x} ${point.y}`), `L ${points.at(-1).x} ${baseline}`, "Z"].join(" ")
    : "";
  const peakPoint = points.reduce((best, point) => (
    point.value > (best?.value ?? -1) ? point : best
  ), null);
  const gridYs = [0.25, 0.5, 0.75].map((ratio) => paddingTop + (height - paddingTop - paddingBottom) * ratio);
  return (
    <div className="portrait-timeline">
      <div className="portrait-timeline-head">
        <div>
          <div className="panel-title">{t("overview.monthly")}</div>
          <strong>{peak ? formatMonthLong(peak.month) : t("overview.timeline.empty")}</strong>
        </div>
        {peak && (
          <span>
            {t("overview.timeline.peak", { n: fmtNum(peak.conversations) })}
          </span>
        )}
      </div>
      {data.length > 1 ? (
        <div className="portrait-spark" role="img" aria-label={t("overview.monthly")}>
          <svg className="portrait-spark-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
            <defs>
              <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="var(--chart-1)" stopOpacity="0.22" />
                <stop offset="100%" stopColor="var(--chart-1)" stopOpacity="0.02" />
              </linearGradient>
            </defs>
            {gridYs.map((y) => (
              <line key={y} x1={paddingX} x2={width - paddingX} y1={y} y2={y} className="portrait-spark-grid" />
            ))}
            <line x1={paddingX} x2={width - paddingX} y1={baseline} y2={baseline} className="portrait-spark-grid baseline" />
            <path d={areaPath} className="portrait-spark-area" fill={`url(#${gradientId})`} />
            <path d={linePath} className="portrait-spark-line" />
            {peakPoint && (
              <>
                <line
                  x1={peakPoint.x}
                  x2={peakPoint.x}
                  y1={paddingTop}
                  y2={baseline}
                  className="portrait-spark-peak-line"
                />
                <circle cx={peakPoint.x} cy={peakPoint.y} r="5.5" className="portrait-spark-peak" />
              </>
            )}
          </svg>
          <div className="portrait-timeline-foot">
            <span>{formatMonthShort(data[0].month)}</span>
            <span>{peak ? formatMonthShort(peak.month) : t("overview.timeline.empty")}</span>
            <span>{formatMonthShort(data.at(-1).month)}</span>
          </div>
        </div>
      ) : (
        <div className="empty-state">{t("overview.timeline.empty")}</div>
      )}
    </div>
  );
}

function MiniBars({ title, rows }) {
  if (!rows?.length) return null;
  const max = Math.max(...rows.map((r) => r.value || 0), 1);
  return (
    <div className="panel">
      <div className="panel-title">{title}</div>
      <div className="mini-bars">
        {rows.map((r) => (
          <div className="mini-row" key={r.label}>
            <span>{pick(r.label)}</span>
            <div><i style={{ transform: `scaleX(${Math.max(0.006, (r.value || 0) / max)})` }} /></div>
            <strong>{fmtNum(r.value)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function ModuleGrid({ modules, session, groupFilter, onSessionRefresh }) {
  useLang();
  const [runningKey, setRunningKey] = useState(null);
  const [errors, setErrors] = useState({});
  const [activeKey, setActiveKey] = useState(null);
  const [resultCache, setResultCache] = useState({});
  const runningRef = useRef(null);

  const ranSet = useMemo(() => new Set(session.modules_ran || []), [session.modules_ran]);
  const filtered = useMemo(
    () => modules.filter((m) => groupFilter === "all" || m.group === groupFilter),
    [modules, groupFilter]
  );
  const recommended = groupFilter === "all"
    ? modules.filter((m) => RECOMMENDED_MODULES.includes(m.key))
    : [];
  const startPaths = groupFilter === "all"
    ? START_PATHS
        .map((path) => ({ path, mod: modules.find((m) => m.key === path.module) }))
        .filter((item) => item.mod)
    : [];
  const rest = filtered.filter((m) => !recommended.some((r) => r.key === m.key));
  const actionsLocked = !!runningKey || !!activeKey;

  useEffect(() => {
    if (!activeKey) return undefined;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previous; };
  }, [activeKey]);

  const run = async (mod) => {
    if (runningRef.current) return;
    runningRef.current = mod.key;
    setRunningKey(mod.key);
    setErrors((e) => ({ ...e, [mod.key]: null }));
    const t0 = performance.now();
    try {
      const result = await _api.runModule(session.session_id, mod.key);
      analytics.moduleRun(mod.key, mod.group, performance.now() - t0);
      setResultCache((c) => ({ ...c, [mod.key]: result }));
      setActiveKey(mod.key);
      onSessionRefresh?.();
    } catch (e) {
      analytics.moduleError(mod.key, e.message || e);
      setErrors((x) => ({ ...x, [mod.key]: String(e.message || e) }));
    } finally {
      runningRef.current = null;
      setRunningKey(null);
    }
  };

  return (
    <>
      {startPaths.length > 0 && (
        <section className="module-section path-section" id="analysis-start">
          <div className="section-title">
            <span>{t("modules.recommended")}</span>
            <small>{t("modules.recommended_hint")}</small>
          </div>
          <div className="path-grid">
            {startPaths.map(({ path, mod }) => (
              <ExplorePathCard
                key={path.key}
                path={path}
                mod={mod}
                running={runningKey === mod.key}
                disabled={actionsLocked && runningKey !== mod.key}
                ran={ranSet.has(mod.key) || !!resultCache[mod.key]}
                err={errors[mod.key]}
                onRun={() => run(mod)}
                onView={() => setActiveKey(mod.key)}
              />
            ))}
          </div>
        </section>
      )}
      <section className="module-section">
        <div className="section-title">
          <span>{groupFilter === "all" ? t("modules.other") : t(`group.${groupFilter}`)}</span>
        </div>
        <div className="module-grid">
          {rest.map((mod) => (
            <ModuleCard
              key={mod.key}
              mod={mod}
              running={runningKey === mod.key}
              disabled={actionsLocked && runningKey !== mod.key}
              ran={ranSet.has(mod.key) || !!resultCache[mod.key]}
              err={errors[mod.key]}
              onRun={() => run(mod)}
              onView={() => setActiveKey(mod.key)}
            />
          ))}
        </div>
      </section>
      {activeKey && (
        <div
          className="result-overlay"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setActiveKey(null);
          }}
        >
          <React.Suspense fallback={<ResultDrawerFallback activeKey={activeKey} onClose={() => setActiveKey(null)} />}>
            <ModuleResult
              sessionId={session.session_id}
              activeKey={activeKey}
              cachedResult={resultCache[activeKey]}
              onClose={() => setActiveKey(null)}
              overlay
            />
          </React.Suspense>
        </div>
      )}
    </>
  );
}

function ExplorePathCard({ path, mod, running, ran, err, disabled, onRun, onView }) {
  useLang();
  return (
    <article className={cls("path-card", ran && "ready", err && "error")}>
      <div className="path-head">
        <span>{t(`group.${mod.group}`)}</span>
        <code>{mod.key}</code>
      </div>
      <h3>{pick(mod.title)}</h3>
      <p>{pick(mod.summary)}</p>
      {err && <div className="err-box">{err}</div>}
      <div className="module-actions">
        {ran ? (
          <>
            <button type="button" disabled={running || disabled} onClick={onView}>{t("card.view")}</button>
            <button type="button" className="ghost" disabled={running || disabled} onClick={onRun}>
              {running ? t("card.running") : t("card.rerun")}
            </button>
          </>
        ) : (
          <button type="button" disabled={running || disabled} onClick={onRun}>
            {running ? t("card.running") : t("card.run")}
          </button>
        )}
        <span>{t("card.est", { n: mod.est_seconds })}</span>
      </div>
    </article>
  );
}

function ModuleCard({ mod, running, ran, err, disabled, recommended, onRun, onView }) {
  useLang();
  return (
    <article className={cls("module-card", recommended && "is-recommended", ran && "ready", err && "error")}>
      <div className="module-head">
        <div>
          <span className="module-group">{t(`group.${mod.group}`)}</span>
          <h3>{pick(mod.title)}</h3>
        </div>
        <code>{mod.key}</code>
      </div>
      <p>{pick(mod.summary)}</p>
      {err && <div className="err-box">{err}</div>}
      <div className="module-actions">
        {ran ? (
          <>
            <button type="button" disabled={running || disabled} onClick={onView}>{t("card.view")}</button>
            <button type="button" className="ghost" disabled={running || disabled} onClick={onRun}>
              {running ? t("card.running") : t("card.rerun")}
            </button>
          </>
        ) : (
          <button type="button" disabled={running || disabled} onClick={onRun}>
            {running ? t("card.running") : t("card.run")}
          </button>
        )}
        <span>{t("card.est", { n: mod.est_seconds })}</span>
      </div>
    </article>
  );
}

function ResultDrawerFallback({ activeKey, onClose }) {
  return (
    <section
      className={cls("result-wrap", "drawer")}
      role="dialog"
      aria-modal="true"
      aria-label={activeKey}
      tabIndex={-1}
    >
      <div className="result-head">
        <div>
          <span className="module-group">{activeKey}</span>
          <h2>{t("result.loading")}</h2>
        </div>
        <div className="result-head-actions">
          <button type="button" className="ghost" onClick={onClose}>{t("result.close")}</button>
        </div>
      </div>
      <div className="skeleton-grid">{[0, 1, 2].map((index) => <SkeletonCard key={index} />)}</div>
    </section>
  );
}

function SkeletonCard() {
  return (
    <div className="skeleton-card">
      <span />
      <span />
      <span />
    </div>
  );
}

function App() {
  const lang = useLang();
  const [theme, setTheme] = useState(() => {
    try {
      const stored = localStorage.getItem(LS_THEME);
      return THEMES.includes(stored) ? stored : "system";
    } catch { return "system"; }
  });
  const [modules, setModules] = useState([]);
  const [groups, setGroups] = useState([]);
  const [session, setSession] = useState(null);
  const [bootError, setBootError] = useState(null);
  const [groupFilter, setGroupFilter] = useState("all");

  // Pyodide engine state
  const [engineReady, setEngineReady] = useState(false);
  const [engineStage, setEngineStage] = useState(null);
  const [engineError, setEngineError] = useState(null);
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", resolveTheme(theme));
    try { localStorage.setItem(LS_THEME, theme); } catch {}
    // Listen for OS theme changes when in "system" mode
    if (theme === "system" && typeof matchMedia !== "undefined") {
      const mq = matchMedia("(prefers-color-scheme: dark)");
      const handler = () => document.documentElement.setAttribute("data-theme", resolveTheme("system"));
      mq.addEventListener("change", handler);
      return () => mq.removeEventListener("change", handler);
    }
  }, [theme]);

  // Boot: start Pyodide worker + load module catalog
  useEffect(() => {
    const { api, worker, promise } = getAPI();
    // Expose for download helper in result-view.jsx
    window.__cca_api = api;

    // Listen for progress during init
    const progressHandler = (e) => {
      if (e.data.type === "progress") setEngineStage(e.data.stage);
    };
    worker.addEventListener("message", progressHandler);

    // Load catalog immediately (static, no Pyodide needed)
    api.listModules().then((mods) => {
      setModules(mods.modules || []);
      setGroups(mods.groups || []);
    }).catch((e) => setBootError(String(e.message || e)));

    // Wait for Pyodide init
    promise
      .then(() => { setEngineReady(true); setEngineStage("ready"); })
      .catch((e) => setEngineError(String(e.message || e)));

    return () => worker.removeEventListener("message", progressHandler);
  }, []);

  const handleDrop = useCallback(async () => {
    if (!session || !_api) return;
    analytics.sessionDrop();
    try { await _api.deleteSession(session.session_id); } catch {}
    setSession(null);
    setGroupFilter("all");
  }, [session]);

  const refreshSession = useCallback(async () => {
    if (!session?.session_id || !_api) return;
    try { setSession(await _api.getSession(session.session_id)); } catch {}
  }, [session?.session_id]);

  const navItems = session?.status === "ready"
    ? [{ key: "all", label: t("filter.all") }, ...groups.map((g) => ({ key: g, label: t(`group.${g}`) }))]
    : [];

  const body = (() => {
    // Show engine loading screen until Pyodide is ready
    if (!engineReady) {
      if (engineError) {
        return (
          <section className="landing">
            <div className="err-box">{engineError}</div>
          </section>
        );
      }
      return <PyodideLoader stage={engineStage} error={engineError} />;
    }
    if (bootError) {
      return (
        <section className="landing">
          <div className="err-box">{t("error.boot", { msg: bootError })}</div>
          <p>{t("error.boot.hint")}</p>
        </section>
      );
    }
    if (!session) return <Uploader onSession={setSession} />;
    if (session.status === "error") {
      return (
        <section className="landing">
          <div className="err-box">{t("error.prepare", { msg: session.error || "unknown" })}</div>
          <button type="button" onClick={handleDrop}>{t("error.prepare.retry")}</button>
        </section>
      );
    }
    if (session.status !== "ready") {
      // In client-only mode, prepare runs inside uploadZip, so this shouldn't happen.
      // But if it does, show the standard prepare view.
      return <PrepareView session={session} onReady={setSession} />;
    }
    return (
      <>
        <SessionBar session={session} onDrop={handleDrop} />
        <Overview overview={session.overview} />
        <ModuleGrid
          modules={modules}
          session={session}
          groupFilter={groupFilter}
          onSessionRefresh={refreshSession}
        />
      </>
    );
  })();

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <span>{t("app.brand_short")}</span>
            <strong>{t("app.title")}</strong>
          </div>
          <nav className="nav" aria-label={t("filter.label")}>
            {navItems.map((item) => (
              <button
                key={item.key}
                type="button"
                className={groupFilter === item.key ? "active" : ""}
                onClick={() => setGroupFilter(item.key)}
              >
                {item.label}
              </button>
            ))}
          </nav>
          <div className="switches">
            <div className="seg" aria-label={t("theme.label")}>
              {THEMES.map((name) => (
                <button key={name} type="button" className={theme === name ? "active" : ""} onClick={() => { setTheme(name); analytics.themeChange(name); }}>
                  {t(`theme.${name}`)}
                </button>
              ))}
            </div>
            <div className="seg" aria-label={t("lang.label")}>
              {I18N.supported().map((code) => (
                <button key={code} type="button" className={lang === code ? "active" : ""} onClick={() => { I18N.setLang(code); analytics.langChange(code); }}>
                  {code.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>
      <main className="shell">{body}</main>
    </>
  );
}

createRoot(document.getElementById("app")).render(<App />);
