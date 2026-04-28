import React, { useCallback, useSyncExternalStore } from "react";

import "../i18n/dict-ui.js";
import "../i18n/dict-summary.js";
import "../i18n/core.js";

export const I18N = window.I18N;

export function useLang() {
  return useSyncExternalStore(
    (cb) => I18N.subscribe(cb),
    () => I18N.getLang()
  );
}

export function t(key, params) {
  return I18N.t(key, params);
}

export function pick(value) {
  return I18N.pick(value);
}

export function fmtNum(n) {
  if (n == null || n === "") return "—";
  const num = Number(n);
  if (!Number.isFinite(num)) return String(n);
  return new Intl.NumberFormat(I18N.getLang() === "tr" ? "tr-TR" : "en-US").format(num);
}

export function fmtVal(v, digits = 2) {
  if (v == null) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    if (Number.isInteger(v)) return fmtNum(v);
    return new Intl.NumberFormat(I18N.getLang() === "tr" ? "tr-TR" : "en-US", {
      maximumFractionDigits: digits,
    }).format(v);
  }
  return String(v);
}

export function fmtPct(v) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  return `${Number(v).toFixed(1)}%`;
}

export function cls(...items) {
  return items.filter(Boolean).join(" ");
}

function renderInline(text) {
  const input = String(text == null ? "" : text);
  const out = [];
  let i = 0;
  let key = 0;

  const pushText = (s) => {
    if (s) out.push(<React.Fragment key={`t${key++}`}>{s}</React.Fragment>);
  };

  while (i < input.length) {
    if (input.startsWith("`", i)) {
      const end = input.indexOf("`", i + 1);
      if (end > i) {
        out.push(<code key={`c${key++}`}>{input.slice(i + 1, end)}</code>);
        i = end + 1;
        continue;
      }
    }
    if (input.startsWith("**", i)) {
      const end = input.indexOf("**", i + 2);
      if (end > i) {
        out.push(<strong key={`b${key++}`}>{input.slice(i + 2, end)}</strong>);
        i = end + 2;
        continue;
      }
    }
    const nextCode = input.indexOf("`", i + 1);
    const nextBold = input.indexOf("**", i + 1);
    const candidates = [nextCode, nextBold].filter((x) => x !== -1);
    const next = candidates.length ? Math.min(...candidates) : input.length;
    pushText(input.slice(i, next));
    i = next;
  }
  return out;
}

export function InlineText({ children }) {
  return <>{renderInline(children)}</>;
}

export function useArrowTabs(ids, activeId, setActiveId) {
  return useCallback((event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const current = Math.max(0, ids.indexOf(activeId));
    let next = current;
    if (event.key === "ArrowLeft") next = (current - 1 + ids.length) % ids.length;
    if (event.key === "ArrowRight") next = (current + 1) % ids.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = ids.length - 1;
    setActiveId(ids[next]);
  }, [ids, activeId, setActiveId]);
}

export function KpiTile({ label, value, compact }) {
  useLang();
  return (
    <div className={cls("kpi-tile", compact && "compact")}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{fmtVal(value)}</div>
    </div>
  );
}
