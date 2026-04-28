import React, { useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

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
  useArrowTabs,
  useLang,
} from "./runtime.jsx";

const CHART_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
  "var(--chart-6)",
];

const RESULT_TABS = ["summary", "explore", "data"];

function triggerDownload(slug, filename) {
  // Use the global _api singleton's downloadFile method
  if (typeof window.__cca_api?.downloadFile === "function") {
    window.__cca_api.downloadFile(slug, filename).catch((e) => {
      console.error("Download failed:", e);
    });
  }
}

export default function ModuleResult({ sessionId, activeKey, cachedResult, onClose, overlay = false }) {
  useLang();
  const [data, setData] = useState(cachedResult || null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState("summary");
  const wrapRef = useRef(null);
  const onTabKey = useArrowTabs(RESULT_TABS, tab, setTab);

  useEffect(() => {
    setErr(null);
    setTab("summary");
    setData(cachedResult || null);
  }, [sessionId, activeKey, cachedResult]);

  useEffect(() => {
    if (!wrapRef.current || !data) return;
    if (overlay) {
      wrapRef.current.focus({ preventScroll: true });
      return;
    }
    wrapRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [data, overlay]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (err) {
    return (
      <section
        className={cls("result-wrap", overlay && "drawer")}
        ref={wrapRef}
        role={overlay ? "dialog" : undefined}
        aria-modal={overlay ? "true" : undefined}
        aria-label={activeKey}
        tabIndex={overlay ? -1 : undefined}
      >
        <div className="err-box result-error">{err}</div>
      </section>
    );
  }

  if (!data) {
    return (
      <section
        className={cls("result-wrap", overlay && "drawer")}
        ref={wrapRef}
        role={overlay ? "dialog" : undefined}
        aria-modal={overlay ? "true" : undefined}
        aria-label={activeKey}
        tabIndex={overlay ? -1 : undefined}
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
        <div className="skeleton-grid">{[0, 1, 2].map((i) => <SkeletonCard key={i} />)}</div>
      </section>
    );
  }

  const charts = collectBlocks(data.sections, "chart")
    .filter((block) => !["legacy_asset", "image"].includes(block.payload?.kind));
  const tables = collectBlocks(data.sections, "table");
  const assets = collectAssets(data);

  return (
    <section
      className={cls("result-wrap", overlay && "drawer")}
      ref={wrapRef}
      role={overlay ? "dialog" : undefined}
      aria-modal={overlay ? "true" : undefined}
      aria-label={pick(data.title)}
      tabIndex={overlay ? -1 : undefined}
    >
      <div className="result-head sticky">
        <div>
          <span className="module-group">{data.key} · {t(`group.${data.group || ""}`)}</span>
          <h2>{pick(data.title)}</h2>
          <div className="result-meta">
            <span>{t("result.elapsed", { n: Number(data.elapsed_seconds || 0).toFixed(1) })}</span>
            <span>{t("result.charts", { n: charts.length })}</span>
            <span>{t("result.csvs", { n: (data.csvs || []).length })}</span>
          </div>
        </div>
        <div className="result-head-actions">
          <button type="button" className="ghost-link" onClick={() => triggerDownload(data.slug, "summary.json")}>
            {t("result.export")}
          </button>
          <button type="button" className="ghost" onClick={onClose}>{t("result.close")}</button>
        </div>
      </div>
      <div className="tabs" role="tablist" aria-label={t("result.tabs")}>
        {RESULT_TABS.map((id) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            tabIndex={tab === id ? 0 : -1}
            className={tab === id ? "active" : ""}
            onClick={() => setTab(id)}
            onKeyDown={onTabKey}
          >
            {t(`result.tab.${id}`)}
          </button>
        ))}
      </div>
      {tab === "summary" && <ResultSummary data={data} charts={charts} />}
      {tab === "explore" && <ReportSections sections={data.sections || []} />}
      {tab === "data" && (
        <DataTab
          sessionId={sessionId}
          data={data}
          assets={assets}
          tables={tables}
          charts={charts}
        />
      )}
    </section>
  );
}

function ResultSummary({ data, charts }) {
  const summary = data.summary_json || {};
  const bullets = firstBullets(data.sections);
  const kpis = topKpis(summary, 4);
  const primaryChart = charts[0];

  return (
    <div className="result-tab-panel">
      {summary.headline && <p className="headline"><InlineText>{pick(summary.headline)}</InlineText></p>}
      {kpis.length > 0 && (
        <div className="kpi-grid calm">
          {kpis.map(([k, v]) => <KpiTile key={k} label={I18N.summaryLabel(k)} value={v} />)}
        </div>
      )}
      {bullets.length > 0 && (
        <div className="summary-layout">
          <div className="panel">
            <div className="panel-title">{t("result.key_findings")}</div>
            <ul className="clean-list">
              {bullets.slice(0, 5).map((item, i) => (
                <li key={i}><InlineText>{pick(item)}</InlineText></li>
              ))}
            </ul>
          </div>
          {primaryChart && (
            <div className="panel primary-chart">
              <ChartBlock payload={primaryChart.payload} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ReportSections({ sections }) {
  const visible = (sections || [])
    .map((section) => ({
      ...section,
      blocks: (section.blocks || []).filter((block) => !["image", "legacy_asset"].includes(block.type)),
    }))
    .filter((section) => section.blocks.length > 0);

  if (!visible.length) {
    return <div className="empty-state">{t("result.no_structured")}</div>;
  }

  return (
    <div className="result-tab-panel report-sections">
      {visible.map((section, index) => (
        <ReportSection key={`${pick(section.heading) || index}`} section={section} />
      ))}
    </div>
  );
}

function ReportSection({ section }) {
  return (
    <section className="report-section">
      <h3>{pick(section.heading)}</h3>
      <div className="report-blocks">
        {section.blocks.map((block, index) => <Block key={index} block={block} />)}
      </div>
    </section>
  );
}

function Block({ block }) {
  const { type, payload = {} } = block;
  if (type === "bullets") return <BulletsBlock items={payload.items || []} />;
  if (type === "paragraph") return <p className="section-paragraph"><InlineText>{pick(payload.text)}</InlineText></p>;
  if (type === "table") return <TableBlock columns={payload.columns || []} rows={payload.rows || []} caption={payload.caption} />;
  if (type === "chart") return <ChartBlock payload={payload} />;
  return (
    <div className="raw-block">
      <code>{JSON.stringify(block, null, 2)}</code>
    </div>
  );
}

function BulletsBlock({ items }) {
  if (!items.length) return null;
  return (
    <ul className="clean-list section-list">
      {items.map((item, index) => <li key={index}><InlineText>{pick(item)}</InlineText></li>)}
    </ul>
  );
}

function TableBlock({ columns, rows, caption }) {
  if (!columns.length || !rows.length) return null;
  return (
    <div className="table-wrap">
      {caption && <div className="table-caption"><InlineText>{pick(caption)}</InlineText></div>}
      <table>
        <thead>
          <tr>
            {columns.map((column, index) => (
              <th key={column.key || index} className={column.align === "right" ? "num" : ""}>
                {pick(column.label) || I18N.summaryLabel(column.key || "")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className={columns[cellIndex]?.align === "right" ? "num" : ""}>
                  <InlineText>{pick(cell)}</InlineText>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChartBlock({ payload }) {
  const kind = payload?.kind;
  if (kind === "bucket_bar" || kind === "histogram" || kind === "delta") return <BucketChart payload={payload} />;
  if (kind === "bar") return <BarChartBlock payload={payload} />;
  if (kind === "grouped_bar") return <GroupedBarChartBlock payload={payload} />;
  if (kind === "line" || kind === "multi_line") return <LineChartBlock payload={payload} />;
  if (kind === "heatmap") return <HeatmapChart payload={payload} />;
  if (kind === "scatter") return <ScatterChartBlock payload={payload} />;
  return (
    <div className="chart-card unsupported">
      <div className="chart-title">{pick(payload?.label) || t("chart.unsupported")}</div>
      <pre>{JSON.stringify(payload, null, 2)}</pre>
    </div>
  );
}

function ChartShell({ title, children }) {
  return (
    <div className="chart-card">
      {title && <div className="chart-title"><InlineText>{pick(title)}</InlineText></div>}
      {children}
    </div>
  );
}

function BucketChart({ payload }) {
  const data = (payload.buckets || []).map((bucket) => ({
    label: pick(bucket.label),
    count: Number(bucket.count || 0),
    pct: typeof bucket.pct === "number" ? bucket.pct : null,
  }));
  if (!data.length) return null;
  const height = Math.max(220, Math.min(520, data.length * 34 + 64));
  return (
    <ChartShell title={payload.label}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" tickFormatter={fmtNum} />
          <YAxis dataKey="label" type="category" width={110} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(value, name, row) => [fmtNum(value), row?.payload?.pct != null ? `${name} (${fmtPct(row.payload.pct)})` : name]} />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {data.map((_, index) => <Cell key={index} fill={CHART_COLORS[index % CHART_COLORS.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {payload.xlabel && <div className="chart-xlabel">{pick(payload.xlabel)}</div>}
    </ChartShell>
  );
}

function BarChartBlock({ payload }) {
  const data = (payload.data || payload.buckets || []).map((item) => ({
    ...item,
    [payload.x_key || "label"]: pick(item[payload.x_key || "label"]),
  }));
  const xKey = payload.x_key || "label";
  const yKey = payload.y_key || "count";
  if (!data.length) return null;
  return (
    <ChartShell title={payload.label}>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 10, right: 24, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey={xKey} tick={{ fontSize: 11 }} />
          <YAxis tickFormatter={fmtNum} />
          <Tooltip formatter={(value) => fmtVal(value)} />
          <Bar dataKey={yKey} fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      {payload.xlabel && <div className="chart-xlabel">{pick(payload.xlabel)}</div>}
      {payload.ylabel && <div className="chart-ylabel">{pick(payload.ylabel)}</div>}
    </ChartShell>
  );
}

function GroupedBarChartBlock({ payload }) {
  const data = (payload.data || []).map((item) => ({
    ...item,
    [payload.x_key || "label"]: pick(item[payload.x_key || "label"]),
  }));
  const xKey = payload.x_key || "label";
  const series = payload.series || [];
  if (!data.length || !series.length) return null;
  return (
    <ChartShell title={payload.label}>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} margin={{ top: 10, right: 24, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey={xKey} tick={{ fontSize: 11 }} />
          <YAxis tickFormatter={fmtNum} />
          <Tooltip formatter={(value, name) => [fmtVal(value), pick(series.find((seriesItem) => seriesItem.key === name)?.label) || name]} />
          <Legend formatter={(value) => pick(series.find((seriesItem) => seriesItem.key === value)?.label) || value} />
          {series.map((seriesItem, index) => (
            <Bar
              key={seriesItem.key}
              dataKey={seriesItem.key}
              name={pick(seriesItem.label) || seriesItem.key}
              fill={CHART_COLORS[index % CHART_COLORS.length]}
              radius={[3, 3, 0, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

function LineChartBlock({ payload }) {
  const data = (payload.data || []).map((item) => ({
    ...item,
    [payload.x_key || "label"]: pick(item[payload.x_key || "label"]),
  }));
  const xKey = payload.x_key || "label";
  const series = payload.series || [{ key: payload.y_key || "count", label: payload.label || payload.y_key || "count" }];
  if (!data.length || !series.length) return null;
  return (
    <ChartShell title={payload.label}>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 10, right: 24, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey={xKey} tick={{ fontSize: 11 }} />
          <YAxis tickFormatter={fmtNum} />
          <Tooltip formatter={(value, name) => [fmtVal(value), pick(series.find((seriesItem) => seriesItem.key === name)?.label) || name]} />
          {series.length > 1 && <Legend formatter={(value) => pick(series.find((seriesItem) => seriesItem.key === value)?.label) || value} />}
          {series.map((seriesItem, index) => (
            <Line
              key={seriesItem.key}
              type="monotone"
              dataKey={seriesItem.key}
              name={pick(seriesItem.label) || seriesItem.key}
              stroke={CHART_COLORS[index % CHART_COLORS.length]}
              strokeWidth={2}
              dot={{ r: 2 }}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {payload.xlabel && <div className="chart-xlabel">{pick(payload.xlabel)}</div>}
      {payload.ylabel && <div className="chart-ylabel">{pick(payload.ylabel)}</div>}
    </ChartShell>
  );
}

function HeatmapChart({ payload }) {
  const cells = payload.cells || [];
  const xLabels = (payload.x_labels || Array.from(new Set(cells.map((cell) => cell.x)))).map((value) => pick(value));
  const yLabels = (payload.y_labels || Array.from(new Set(cells.map((cell) => cell.y)))).map((value) => pick(value));
  if (!cells.length || !xLabels.length || !yLabels.length) return null;
  const max = Math.max(...cells.map((cell) => Number(cell.value || 0)), 1);
  const byKey = new Map(cells.map((cell) => [`${pick(cell.x)}|${pick(cell.y)}`, cell]));
  return (
    <ChartShell title={payload.label}>
      <div className="heatmap" style={{ gridTemplateColumns: `72px repeat(${xLabels.length}, minmax(16px, 1fr))` }}>
        <div />
        {xLabels.map((x) => <div key={x} className="heat-axis">{x}</div>)}
        {yLabels.map((y) => (
          <React.Fragment key={y}>
            <div className="heat-axis y">{y}</div>
            {xLabels.map((x) => {
              const cell = byKey.get(`${x}|${y}`);
              const value = Number(cell?.value || 0);
              return (
                <div
                  key={`${x}-${y}`}
                  className="heat-cell"
                  title={`${y} / ${x}: ${fmtVal(value)}`}
                  style={{ "--heat": value / max }}
                >
                  {value > 0 ? fmtNum(value) : ""}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </ChartShell>
  );
}

function ScatterChartBlock({ payload }) {
  const data = payload.data || [];
  const xKey = payload.x_key || "x";
  const yKey = payload.y_key || "y";
  const sizeKey = payload.size_key || "size";
  if (!data.length) return null;
  return (
    <ChartShell title={payload.label}>
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ top: 10, right: 24, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey={xKey} name={pick(payload.xlabel) || xKey} tickFormatter={fmtNum} />
          <YAxis type="number" dataKey={yKey} name={pick(payload.ylabel) || yKey} tickFormatter={fmtNum} />
          <ZAxis type="number" dataKey={sizeKey} range={[40, 320]} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(value) => fmtVal(value)} />
          <Scatter data={data} fill="var(--chart-1)" />
        </ScatterChart>
      </ResponsiveContainer>
      {payload.xlabel && <div className="chart-xlabel">{pick(payload.xlabel)}</div>}
      {payload.ylabel && <div className="chart-ylabel">{pick(payload.ylabel)}</div>}
    </ChartShell>
  );
}

function DataTab({ sessionId, data, assets, tables, charts }) {
  return (
    <div className="result-tab-panel data-tab">
      <div className="data-grid">
        <div className="panel">
          <div className="panel-title">{t("data.downloads")}</div>
          <div className="download-list">
            <button type="button" onClick={() => triggerDownload(data.slug, "report.md")}>{t("result.download.report")}</button>
            <button type="button" onClick={() => triggerDownload(data.slug, "summary.json")}>{t("result.download.summary")}</button>
            {(data.csvs || []).map((csv) => (
              <button type="button" key={csv.name} onClick={() => triggerDownload(data.slug, csv.name)}>
                {csv.name} <small>{csv.size_bytes ? `${fmtNum(csv.size_bytes)} B` : ""}</small>
              </button>
            ))}
            {assets.map((asset) => (
              <button type="button" key={asset.name} onClick={() => triggerDownload(data.slug, asset.name)}>
                {asset.name}
              </button>
            ))}
          </div>
        </div>
        <div className="panel">
          <div className="panel-title">{t("data.inventory")}</div>
          <dl className="inventory">
            <dt>{t("result.charts", { n: charts.length })}</dt><dd>{charts.length}</dd>
            <dt>{t("data.tables")}</dt><dd>{tables.length}</dd>
            <dt>{t("data.assets")}</dt><dd>{assets.length}</dd>
            <dt>{t("result.csvs", { n: (data.csvs || []).length })}</dt><dd>{(data.csvs || []).length}</dd>
          </dl>
        </div>
      </div>
      {tables.length > 0 && (
        <details className="data-details">
          <summary>{t("data.all_tables")}</summary>
          <div className="data-table-stack">
            {tables.map((block, index) => (
              <TableBlock key={index} columns={block.payload.columns || []} rows={block.payload.rows || []} caption={block.payload.caption} />
            ))}
          </div>
        </details>
      )}
      <details className="data-details">
        <summary>{t("data.raw_summary")}</summary>
        <pre>{JSON.stringify(data.summary_json || {}, null, 2)}</pre>
      </details>
      <details className="data-details">
        <summary>{t("data.raw_sections")}</summary>
        <pre>{JSON.stringify(data.sections || [], null, 2)}</pre>
      </details>
    </div>
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

function collectBlocks(sections, type) {
  const out = [];
  for (const section of sections || []) {
    for (const block of section.blocks || []) {
      if (block.type === type) out.push(block);
    }
  }
  return out;
}

function collectAssets(data) {
  // Only show assets that actually exist on disk (from the worker scan).
  // Section blocks may reference PNG filenames that weren't generated (e.g.
  // when matplotlib is unavailable in the browser), so we don't collect those.
  const byName = new Map();
  for (const asset of data.assets || []) {
    if (asset?.name) byName.set(asset.name, asset);
  }
  for (const image of data.images || []) {
    if (image?.name) byName.set(image.name, { name: image.name, size_bytes: image.size_bytes });
  }
  return Array.from(byName.values()).sort((a, b) => a.name.localeCompare(b.name));
}

function firstBullets(sections) {
  for (const section of sections || []) {
    const heading = pick(section.heading).toLowerCase();
    const block = (section.blocks || []).find((item) => item.type === "bullets");
    if (block && (heading.includes("highlight") || heading.includes("summary"))) {
      return block.payload?.items || [];
    }
  }
  const any = collectBlocks(sections, "bullets")[0];
  return any?.payload?.items || [];
}

function topKpis(summary, maxCount) {
  return Object.entries(summary || {})
    .filter(([key, value]) => key !== "headline" && typeof value === "number" && Number.isFinite(value))
    .slice(0, maxCount);
}
