/**
 * GA4 event tracking helper.
 * All custom events are defined here for easy auditing.
 */

function track(name, params) {
  if (typeof gtag === "function") {
    gtag("event", name, params);
  }
}

export function engineLoaded(durationMs) {
  track("engine_loaded", { duration_ms: Math.round(durationMs) });
}

export function zipUploaded({ fileSizeMb, conversations, messages, durationMs }) {
  track("zip_uploaded", {
    file_size_mb: fileSizeMb,
    conversations,
    messages,
    duration_ms: Math.round(durationMs),
  });
}

export function zipError(error, stage) {
  track("zip_error", { error: String(error).slice(0, 200), stage });
}

export function moduleRun(moduleKey, moduleGroup, durationMs) {
  track("module_run", {
    module_key: moduleKey,
    module_group: moduleGroup,
    duration_ms: Math.round(durationMs),
  });
}

export function moduleError(moduleKey, error) {
  track("module_error", {
    module_key: moduleKey,
    error: String(error).slice(0, 200),
  });
}

export function fileDownload(fileType, moduleSlug) {
  track("file_download", { file_type: fileType, module_slug: moduleSlug });
}

export function themeChange(theme) {
  track("theme_change", { theme });
}

export function langChange(lang) {
  track("lang_change", { lang });
}

export function sessionDrop() {
  track("session_drop", {});
}
