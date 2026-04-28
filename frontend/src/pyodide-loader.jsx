import React, { useEffect, useState } from "react";
import { t, useLang } from "./runtime.jsx";

const STAGES = [
  { key: "pyodide_boot", weight: 40 },
  { key: "packages", weight: 40 },
  { key: "scripts", weight: 15 },
  { key: "ready", weight: 5 },
];

const STAGE_LABELS = {
  pyodide_boot: { tr: "Python runtime yükleniyor…", en: "Loading Python runtime…" },
  packages: { tr: "Paketler kuruluyor…", en: "Installing packages…" },
  scripts: { tr: "Analiz modülleri hazırlanıyor…", en: "Mounting analysis scripts…" },
  ready: { tr: "Hazır", en: "Ready" },
  error: { tr: "Hata oluştu", en: "An error occurred" },
};

function stageLabel(stage, lang) {
  const labels = STAGE_LABELS[stage];
  return labels ? (labels[lang] || labels.en) : stage;
}

function stageProgress(currentStage) {
  let total = 0;
  for (const s of STAGES) {
    if (s.key === currentStage) return total + s.weight / 2;
    total += s.weight;
  }
  return 100;
}

export default function PyodideLoader({ stage, error }) {
  const lang = useLang();
  const pct = error ? 0 : stageProgress(stage || "pyodide_boot");
  const label = error
    ? stageLabel("error", lang)
    : stageLabel(stage || "pyodide_boot", lang);

  return (
    <section className="landing">
      <div className="eyebrow"><span />{lang === "tr" ? "Motor" : "Engine"}</div>
      <h1>{lang === "tr" ? "Analiz motoru başlatılıyor" : "Starting analysis engine"}</h1>
      <p style={{ opacity: 0.7, marginBottom: "2rem" }}>
        {lang === "tr"
          ? "İlk açılışta ~15 saniye sürebilir. Sonraki ziyaretlerde önbellekten yüklenir."
          : "May take ~15 seconds on the first visit. Cached on subsequent visits."}
      </p>
      <div className="prepare-steps">
        {STAGES.map((s, i) => {
          const idx = STAGES.findIndex((x) => x.key === stage);
          return (
            <div
              key={s.key}
              className={`prepare-step${i < idx ? " done" : ""}${i === idx ? " active" : ""}`}
            >
              <span>{stageLabel(s.key, lang)}</span>
              <span>{i < idx ? "✓" : i === idx ? "…" : ""}</span>
            </div>
          );
        })}
      </div>
      {error && <div className="err-box">{error}</div>}
    </section>
  );
}
