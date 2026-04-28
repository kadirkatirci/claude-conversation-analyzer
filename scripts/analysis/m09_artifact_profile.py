"""
m09 — Artifact üretim profili

- tool_calls.name='artifacts' kayıtlarında input JSON parse:
  type (markdown/svg/code/react/html/mermaid), command (create/update/rewrite),
  language (code artifact'lerinde)
- Konuşma başına artifact sayısı kova
- Artifact'li vs yok konuşma uzunluk kıyası
- content uzunluğu dağılımı (create/rewrite komutlarında)
- İterasyon oranı = update / create
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np

from analysis import _common
from analysis._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "09-artifact-profile"
TITLE = "Artifact üretim profili"

PER_CONVO_BUCKETS = [
    ("1",     1,   2),
    ("2–5",   2,   6),
    ("6–15",  6,   16),
    ("16–50", 16,  51),
    ("51+",   51,  10 ** 9),
]

CONTENT_LEN_BUCKETS = [
    ("<500",      0,       500),
    ("500–2 K",   500,     2_000),
    ("2–10 K",    2_000,   10_000),
    ("10–50 K",   10_000,  50_000),
    ("50 K+",     50_000,  10 ** 9),
]

# Ham type string'leri daha okunur etiketlere maplenir
TYPE_LABELS = {
    "text/markdown":              "markdown",
    "image/svg+xml":              "svg",
    "application/vnd.ant.svg":    "svg",
    "application/vnd.ant.code":   "code",
    "application/vnd.ant.react":  "react",
    "text/html":                  "html",
    "application/vnd.ant.mermaid":"mermaid",
    "text/plain":                 "plain",
}


def _bucket_counts(values: np.ndarray, buckets: list[tuple[str, float, float]]) -> list[int]:
    return [int(((values >= lo) & (values < hi)).sum()) for _lbl, lo, hi in buckets]


def _bucket_bar(ax, labels: list[str], counts: list[int], color: str, ylabel: str, title: str) -> None:
    ax.bar(labels, counts, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(i, c, f"{c}", ha="center", va="bottom", fontsize=9)


def _fmt_chars(n: float) -> str:
    if n is None or n != n:
        return "—"
    n = float(n)
    if n < 1_000:
        return f"{int(round(n))}"
    if n < 1_000_000:
        return f"{n / 1_000:.1f}K"
    return f"{n / 1_000_000:.2f}M"


def run(con: duckdb.DuckDBPyConnection, out_dir: Path, cfg: dict) -> dict:
    # ---------- artifact tool_calls.input parse ----------
    rows = con.execute("""
        SELECT conversation_uuid, input
        FROM tool_calls
        WHERE name = 'artifacts' AND input IS NOT NULL
    """).fetchall()

    type_ctr: Counter[str] = Counter()
    type_raw_missing = 0
    cmd_ctr: Counter[str] = Counter()
    lang_ctr: Counter[str] = Counter()
    content_lengths: list[int] = []
    per_convo_counts: Counter[str] = Counter()
    per_convo_cmds: dict[str, Counter] = {}  # convo -> Counter(cmd)
    # type × cmd çapraz
    type_cmd_ctr: Counter[tuple[str, str]] = Counter()
    # code artifact'ler için language dağılımı
    lang_by_type: Counter[str] = Counter()

    for convo_uuid, raw in rows:
        if not raw:
            continue
        try:
            d = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            continue

        raw_type = d.get("type")
        if raw_type:
            label = TYPE_LABELS.get(raw_type, raw_type)
            type_ctr[label] += 1
        else:
            label = "(tipsiz)"
            type_raw_missing += 1
            type_ctr[label] += 1

        cmd = d.get("command") or "(yok)"
        cmd_ctr[cmd] += 1
        type_cmd_ctr[(label, cmd)] += 1

        lang = d.get("language")
        if lang:
            lang_ctr[lang] += 1
            if label == "code":
                lang_by_type[lang] += 1

        content = d.get("content")
        if content:
            content_lengths.append(len(content))

        per_convo_counts[convo_uuid] += 1
        per_convo_cmds.setdefault(convo_uuid, Counter())[cmd] += 1

    total_calls = int(sum(type_ctr.values()))
    n_convos_with_artifact = int(len(per_convo_counts))
    content_arr = np.array(content_lengths, dtype=float) if content_lengths else np.array([])
    stats_content = _common.percentiles(content_arr)

    per_convo_arr = np.array(list(per_convo_counts.values()), dtype=float)
    stats_per_convo = _common.percentiles(per_convo_arr)
    per_convo_bucket_counts = _bucket_counts(per_convo_arr, PER_CONVO_BUCKETS)
    content_bucket_counts = _bucket_counts(content_arr, CONTENT_LEN_BUCKETS)

    # İterasyon oranı = (update + rewrite) / create
    n_create = int(cmd_ctr.get("create", 0))
    n_update = int(cmd_ctr.get("update", 0))
    n_rewrite = int(cmd_ctr.get("rewrite", 0))
    iteration_ratio = ((n_update + n_rewrite) / n_create) if n_create else 0.0

    # Konuşma başına iterasyon sayısı (update + rewrite)
    iter_per_convo = np.array([
        (c.get("update", 0) + c.get("rewrite", 0))
        for c in per_convo_cmds.values()
    ], dtype=float)
    stats_iter_per_convo = _common.percentiles(iter_per_convo[iter_per_convo > 0]) if (iter_per_convo > 0).any() else _common.percentiles([])

    # ---------- artifact var/yok konuşma uzunluk kıyası ----------
    compare = con.execute("""
        WITH art_convos AS (
            SELECT DISTINCT conversation_uuid FROM tool_calls
            WHERE name='artifacts' AND input IS NOT NULL
        )
        SELECT
            CASE WHEN ac.conversation_uuid IS NOT NULL THEN TRUE ELSE FALSE END AS has_artifact,
            COUNT(*) AS n,
            quantile_cont(sc.message_count, 0.5) msg_p50,
            quantile_cont(sc.message_count, 0.95) msg_p95,
            quantile_cont(COALESCE(sc.tokens_human,0)+COALESCE(sc.tokens_assistant,0), 0.5) tok_p50,
            quantile_cont(COALESCE(sc.tokens_human,0)+COALESCE(sc.tokens_assistant,0), 0.95) tok_p95
        FROM _stats_conversation sc
        LEFT JOIN art_convos ac ON ac.conversation_uuid = sc.conversation_uuid
        WHERE sc.message_count > 0
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    compare_map = {bool(r[0]): r for r in compare}

    # ---------- grafikler ----------
    tsorted = type_ctr.most_common()
    csorted = cmd_ctr.most_common()
    if HAS_MPL:
        # 1) type bar
        fig, ax = plt.subplots()
        if tsorted:
            names = [n for n, _ in tsorted]
            counts = [c for _, c in tsorted]
            y = np.arange(len(names))
            ax.barh(y, counts, color="#4C72B0")
            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel("çağrı sayısı")
            ax.set_title("Artifact type dağılımı")
            for i, c in enumerate(counts):
                ax.text(c, i, f" {c}", va="center", fontsize=9)
        _common.save_fig(fig, out_dir / "artifact_types.png")

        # 2) command bar
        fig, ax = plt.subplots()
        if csorted:
            names = [n for n, _ in csorted]
            counts = [c for _, c in csorted]
            ax.bar(names, counts, color="#55A868")
            ax.set_ylabel("çağrı sayısı")
            ax.set_title("Artifact command dağılımı")
            for i, c in enumerate(counts):
                ax.text(i, c, f"{c}", ha="center", va="bottom", fontsize=9)
        _common.save_fig(fig, out_dir / "artifact_commands.png")

        # 3) konuşma başına kova
        fig, ax = plt.subplots()
        _bucket_bar(
            ax, [b[0] for b in PER_CONVO_BUCKETS], per_convo_bucket_counts,
            "#8172B2", "konuşma sayısı", "Konuşma başına artifact çağrısı",
        )
        _common.save_fig(fig, out_dir / "bucket_per_conversation.png")

        # 4) content length kova
        fig, ax = plt.subplots()
        _bucket_bar(
            ax, [b[0] for b in CONTENT_LEN_BUCKETS], content_bucket_counts,
            "#C44E52", "artifact sayısı",
            f"Artifact content uzunluğu kovaları (n={content_arr.size})",
        )
        _common.save_fig(fig, out_dir / "bucket_content_length.png")

        # 5) code artifact'lerinde language (eğer varsa)
        fig, ax = plt.subplots()
        lsorted = lang_by_type.most_common(10)
        if lsorted:
            names = [n for n, _ in lsorted]
            counts = [c for _, c in lsorted]
            y = np.arange(len(names))
            ax.barh(y, counts, color="#DD8452")
            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel("artifact sayısı")
            ax.set_title("Code artifact'lerinde language (top 10)")
            for i, c in enumerate(counts):
                ax.text(c, i, f" {c}", va="center", fontsize=9)
        else:
            ax.text(0.5, 0.5, "code artifact yok", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        _common.save_fig(fig, out_dir / "code_languages.png")

    # ---------- tablolar ----------
    # Type tablosu
    type_rows = []
    for name, c in tsorted:
        pct = 100.0 * c / total_calls if total_calls else 0.0
        type_rows.append([name, _common.fmt_int(c), f"%{pct:.1f}"])

    # Command tablosu
    cmd_rows = []
    for name, c in csorted:
        pct = 100.0 * c / total_calls if total_calls else 0.0
        cmd_rows.append([name, _common.fmt_int(c), f"%{pct:.1f}"])

    # Type × command çapraz tablosu (satır=type, sütun=command)
    main_cmds = ["create", "update", "rewrite"]
    type_cmd_rows = []
    for tname, _ in tsorted:
        row = [tname]
        row_total = 0
        for cmd in main_cmds:
            v = int(type_cmd_ctr.get((tname, cmd), 0))
            row_total += v
            row.append(_common.fmt_int(v))
        row.append(_common.fmt_int(row_total))
        type_cmd_rows.append(row)

    # Per-conversation kovalar
    per_convo_bucket_rows = []
    for (lbl, _lo, _hi), c in zip(PER_CONVO_BUCKETS, per_convo_bucket_counts):
        p = (100.0 * c / n_convos_with_artifact) if n_convos_with_artifact else 0.0
        per_convo_bucket_rows.append([lbl, _common.fmt_int(c), f"%{p:.1f}"])

    # Content kovalar
    content_bucket_rows = []
    n_content = int(content_arr.size)
    for (lbl, _lo, _hi), c in zip(CONTENT_LEN_BUCKETS, content_bucket_counts):
        p = (100.0 * c / n_content) if n_content else 0.0
        content_bucket_rows.append([lbl, _common.fmt_int(c), f"%{p:.1f}"])

    # artifact var/yok compare
    def _fmt_num_short(n: float) -> str:
        if n is None or n != n:
            return "—"
        n = float(n)
        if n < 1000:
            return f"{int(round(n))}"
        if n < 1_000_000:
            return f"{n / 1000:.1f}K"
        return f"{n / 1_000_000:.1f}M"

    compare_rows = []
    for has_art in (False, True):
        r = compare_map.get(has_art)
        if r is None:
            continue
        label = "artifact var" if has_art else "artifact yok"
        compare_rows.append([
            label,
            _common.fmt_int(int(r[1])),
            _common.fmt_int(int(r[2] or 0)),
            _common.fmt_int(int(r[3] or 0)),
            _fmt_num_short(float(r[4] or 0)),
            _fmt_num_short(float(r[5] or 0)),
        ])

    # percentile tabloları
    def _safe_int(v):
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0
        return int(v)

    def _int_row(label: str, st: dict) -> list[str]:
        return [
            label,
            _common.fmt_int(st["n"]),
            _common.fmt_int(_safe_int(st["min"])),
            _common.fmt_int(_safe_int(st["p50"])),
            _common.fmt_int(_safe_int(st["p90"])),
            _common.fmt_int(_safe_int(st["p95"])),
            _common.fmt_int(_safe_int(st["p99"])),
            _common.fmt_int(_safe_int(st["max"])),
        ]

    def _chars_row(label: str, st: dict) -> list[str]:
        return [
            label,
            _common.fmt_int(st["n"]),
            _fmt_chars(st["min"]),
            _fmt_chars(st["p50"]),
            _fmt_chars(st["p90"]),
            _fmt_chars(st["p95"]),
            _fmt_chars(st["p99"]),
            _fmt_chars(st["max"]),
        ]

    pct_headers = ["metric", "n", "min", "p50", "p90", "p95", "p99", "max"]
    per_convo_tbl = _common.markdown_table([_int_row("artifact/konuşma", stats_per_convo)], headers=pct_headers)
    content_tbl = _common.markdown_table([_chars_row("content (karakter)", stats_content)], headers=pct_headers)
    iter_tbl = _common.markdown_table([_int_row("iter/konuşma (update+rewrite)", stats_iter_per_convo)], headers=pct_headers)

    # ---------- CSV ----------
    _common.write_csv(
        out_dir, "artifact_types.csv",
        [(n, int(c)) for n, c in tsorted],
        headers=["type", "count"],
    )
    _common.write_csv(
        out_dir, "artifact_commands.csv",
        [(n, int(c)) for n, c in csorted],
        headers=["command", "count"],
    )
    _common.write_csv(
        out_dir, "code_languages.csv",
        [(n, int(c)) for n, c in lang_by_type.most_common()],
        headers=["language", "count"],
    )
    _common.write_csv(
        out_dir, "artifact_per_conversation.csv",
        [(k, int(v)) for k, v in per_convo_counts.items()],
        headers=["conversation_uuid", "artifact_calls"],
    )

    # ---------- summary ----------
    summary = {
        "total_artifact_calls": total_calls,
        "n_conversations_with_artifact": n_convos_with_artifact,
        "type_counts": dict(type_ctr),
        "command_counts": dict(cmd_ctr),
        "iteration_ratio": round(iteration_ratio, 3),
        "n_create": n_create,
        "n_update": n_update,
        "n_rewrite": n_rewrite,
        "content_length_chars": stats_content,
        "artifact_per_conversation": stats_per_convo,
        "iterations_per_conversation": stats_iter_per_convo,
        "headline": (
            f"{total_calls} artifact çağrısı, {n_convos_with_artifact} konuşmada; "
            f"en sık tip: **{(tsorted[0][0] if tsorted else '—')}** "
            f"(**{tsorted[0][1] if tsorted else 0}**); "
            f"iterasyon oranı (update+rewrite)/create = **{iteration_ratio:.2f}**."
        ),
    }
    _common.write_json(out_dir, "summary.json", summary)

    # ---------- structured data ----------
    pct_cols = [
        {"key": "metric", "label": "metric", "align": "left"},
        {"key": "n",      "label": "n",      "align": "right"},
        {"key": "min",    "label": "min",    "align": "right"},
        {"key": "p50",    "label": "p50",    "align": "right"},
        {"key": "p90",    "label": "p90",    "align": "right"},
        {"key": "p95",    "label": "p95",    "align": "right"},
        {"key": "p99",    "label": "p99",    "align": "right"},
        {"key": "max",    "label": "max",    "align": "right"},
    ]

    type_buckets = [
        {"label": n, "count": int(c), "pct": (100.0 * c / total_calls) if total_calls else 0.0}
        for n, c in tsorted
    ]
    cmd_buckets = [
        {"label": n, "count": int(c), "pct": (100.0 * c / total_calls) if total_calls else 0.0}
        for n, c in csorted
    ]
    per_convo_buckets_data = [
        {"label": lbl, "count": int(c),
         "pct": (100.0 * c / n_convos_with_artifact) if n_convos_with_artifact else 0.0}
        for (lbl, _lo, _hi), c in zip(PER_CONVO_BUCKETS, per_convo_bucket_counts)
    ]
    content_buckets_data = [
        {"label": lbl, "count": int(c),
         "pct": (100.0 * c / n_content) if n_content else 0.0}
        for (lbl, _lo, _hi), c in zip(CONTENT_LEN_BUCKETS, content_bucket_counts)
    ]
    lang_total = sum(lang_by_type.values()) if lang_by_type else 0
    lang_buckets = [
        {"label": n, "count": int(c), "pct": (100.0 * c / lang_total) if lang_total else 0.0}
        for n, c in lang_by_type.most_common(10)
    ]

    # ---------- report ----------
    sections = [
        _common.Section(
            "Öne çıkanlar",
            f"- Toplam artifact çağrısı: **{_common.fmt_int(total_calls)}** "
            f"(parse edilebilen input'lu)\n"
            f"- Artifact üretilen konuşma: **{_common.fmt_int(n_convos_with_artifact)}**\n"
            f"- Komut dağılımı: create **{n_create}** · update **{n_update}** · "
            f"rewrite **{n_rewrite}** · diğer **{total_calls - n_create - n_update - n_rewrite}**\n"
            f"- İterasyon oranı (update+rewrite) / create = **{iteration_ratio:.2f}**\n"
            f"- Tipsiz kayıt: **{type_raw_missing}** "
            f"(input'ta `type` alanı yok — büyük ihtimalle `update` komutu)\n"
            f"- content uzunluğu (create/rewrite için, n=**{_common.fmt_int(n_content)}**): "
            f"medyan **{_fmt_chars(stats_content['p50'])}**, "
            f"p95 **{_fmt_chars(stats_content['p95'])}**, "
            f"max **{_fmt_chars(stats_content['max'])}**\n"
            f"- Konuşma başına artifact çağrısı: medyan **{int(stats_per_convo['p50'] or 0)}**, "
            f"max **{int(stats_per_convo['max'] or 0)}**",
            blocks=[
                _common.block_bullets([
                    f"Toplam artifact çağrısı: **{_common.fmt_int(total_calls)}** (parse edilebilen input'lu)",
                    f"Artifact üretilen konuşma: **{_common.fmt_int(n_convos_with_artifact)}**",
                    f"Komut dağılımı: create **{n_create}** · update **{n_update}** · "
                    f"rewrite **{n_rewrite}** · diğer **{total_calls - n_create - n_update - n_rewrite}**",
                    f"İterasyon oranı (update+rewrite) / create = **{iteration_ratio:.2f}**",
                    f"Tipsiz kayıt: **{type_raw_missing}** "
                    f"(input'ta `type` alanı yok — büyük ihtimalle `update` komutu)",
                    f"content uzunluğu (create/rewrite için, n=**{_common.fmt_int(n_content)}**): "
                    f"medyan **{_fmt_chars(stats_content['p50'])}**, "
                    f"p95 **{_fmt_chars(stats_content['p95'])}**, "
                    f"max **{_fmt_chars(stats_content['max'])}**",
                    f"Konuşma başına artifact çağrısı: "
                    f"medyan **{int(stats_per_convo['p50'] or 0)}**, "
                    f"max **{int(stats_per_convo['max'] or 0)}**",
                ]),
            ],
        ),
        _common.Section(
            "Type dağılımı",
            _common.markdown_table(type_rows, headers=["type", "çağrı", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    "Artifact type dağılımı",
                    type_buckets,
                    image="artifact_types.png",
                    xlabel="type",
                ),
                _common.block_table(
                    [
                        {"key": "type",  "label": "type",  "align": "left"},
                        {"key": "count", "label": "çağrı", "align": "right"},
                        {"key": "pct",   "label": "oran",  "align": "right"},
                    ],
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"] for b in type_buckets],
                ),
            ] if type_buckets else [],
        ),
        _common.Section(
            "Command dağılımı",
            _common.markdown_table(cmd_rows, headers=["command", "çağrı", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    "Artifact command dağılımı",
                    cmd_buckets,
                    image="artifact_commands.png",
                    xlabel="command",
                ),
                _common.block_table(
                    [
                        {"key": "command", "label": "command", "align": "left"},
                        {"key": "count",   "label": "çağrı",   "align": "right"},
                        {"key": "pct",     "label": "oran",    "align": "right"},
                    ],
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"] for b in cmd_buckets],
                ),
            ] if cmd_buckets else [],
        ),
        _common.Section(
            "Type × command çapraz",
            _common.markdown_table(
                type_cmd_rows,
                headers=["type \\ cmd", "create", "update", "rewrite", "toplam"],
            ),
            blocks=[
                _common.block_table(
                    [
                        {"key": "type",    "label": "type",    "align": "left"},
                        {"key": "create",  "label": "create",  "align": "right"},
                        {"key": "update",  "label": "update",  "align": "right"},
                        {"key": "rewrite", "label": "rewrite", "align": "right"},
                        {"key": "total",   "label": "toplam",  "align": "right"},
                    ],
                    type_cmd_rows,
                ),
            ] if type_cmd_rows else [],
        ),
        _common.Section(
            "Code artifact'lerinde language (top 10)",
            _common.markdown_table(
                [[n, _common.fmt_int(c),
                  f"%{100.0 * c / sum(lang_by_type.values()):.1f}" if lang_by_type else "—"]
                 for n, c in lang_by_type.most_common(10)] or [["—", "—", "—"]],
                headers=["language", "artifact", "oran"],
            ),
            blocks=[
                _common.block_bucket_chart(
                    "Code artifact'lerinde language (top 10)",
                    lang_buckets,
                    image="code_languages.png",
                    xlabel="language",
                ),
                _common.block_table(
                    [
                        {"key": "language", "label": "language", "align": "left"},
                        {"key": "count",    "label": "artifact", "align": "right"},
                        {"key": "pct",      "label": "oran",     "align": "right"},
                    ],
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"] for b in lang_buckets],
                ),
            ] if lang_buckets else [
                _common.block_paragraph("code artifact yok"),
            ],
        ),
        _common.Section(
            "Konuşma başına artifact çağrısı",
            per_convo_tbl,
            blocks=[
                _common.block_table(pct_cols, [_int_row("artifact/konuşma", stats_per_convo)]),
            ],
        ),
        _common.Section(
            "Konuşma kovaları",
            _common.markdown_table(
                per_convo_bucket_rows, headers=["çağrı aralığı", "konuşma", "oran"],
            ),
            blocks=[
                _common.block_bucket_chart(
                    "Konuşma başına artifact çağrısı",
                    per_convo_buckets_data,
                    image="bucket_per_conversation.png",
                    xlabel="çağrı aralığı",
                ),
                _common.block_table(
                    [
                        {"key": "bucket", "label": "çağrı aralığı", "align": "left"},
                        {"key": "count",  "label": "konuşma",       "align": "right"},
                        {"key": "pct",    "label": "oran",          "align": "right"},
                    ],
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"]
                     for b in per_convo_buckets_data],
                ),
            ],
        ),
        _common.Section(
            "Konuşma başına iterasyon (update + rewrite)",
            iter_tbl,
            blocks=[
                _common.block_table(pct_cols, [_int_row("iter/konuşma (update+rewrite)", stats_iter_per_convo)]),
            ],
        ),
        _common.Section(
            "Artifact content uzunluğu",
            content_tbl,
            blocks=[
                _common.block_table(pct_cols, [_chars_row("content (karakter)", stats_content)]),
            ],
        ),
        _common.Section(
            "Content uzunluğu kovaları",
            _common.markdown_table(
                content_bucket_rows, headers=["uzunluk aralığı", "artifact", "oran"],
            ),
            blocks=[
                _common.block_bucket_chart(
                    f"Artifact content uzunluğu kovaları (n={n_content})",
                    content_buckets_data,
                    image="bucket_content_length.png",
                    xlabel="uzunluk aralığı",
                ),
                _common.block_table(
                    [
                        {"key": "bucket", "label": "uzunluk aralığı", "align": "left"},
                        {"key": "count",  "label": "artifact",        "align": "right"},
                        {"key": "pct",    "label": "oran",            "align": "right"},
                    ],
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"]
                     for b in content_buckets_data],
                ),
            ],
        ),
        _common.Section(
            "Artifact'li vs yok — konuşma uzunluğu",
            _common.markdown_table(
                compare_rows,
                headers=["kitle", "n konuşma", "mesaj p50", "mesaj p95", "token p50", "token p95"],
            ),
            blocks=[
                _common.block_table(
                    [
                        {"key": "cohort",    "label": "kitle",     "align": "left"},
                        {"key": "n",         "label": "n konuşma", "align": "right"},
                        {"key": "msg_p50",   "label": "mesaj p50", "align": "right"},
                        {"key": "msg_p95",   "label": "mesaj p95", "align": "right"},
                        {"key": "tok_p50",   "label": "token p50", "align": "right"},
                        {"key": "tok_p95",   "label": "token p95", "align": "right"},
                    ],
                    compare_rows,
                ),
            ] if compare_rows else [],
        ),
        _common.Section(
            "Grafikler",
            "- `artifact_types.png` — type dağılımı\n"
            "- `artifact_commands.png` — command dağılımı\n"
            "- `bucket_per_conversation.png` — konuşma başına artifact sayısı kovaları\n"
            "- `bucket_content_length.png` — content uzunluğu kovaları\n"
            "- `code_languages.png` — code artifact'lerinde language (top 10)",
            blocks=[],
        ),
        _common.Section(
            "Notlar",
            "- Kaynak: `tool_calls.name='artifacts'` kayıtlarının `input` JSON'u.\n"
            "- `type` etiketleri normalize edildi: `text/markdown→markdown`, "
            "`image/svg+xml→svg`, `application/vnd.ant.code→code`, "
            "`application/vnd.ant.react→react`, `application/vnd.ant.mermaid→mermaid`.\n"
            "- `update` komutunda input'ta genelde `type` ve `content` bulunmaz "
            "(yalnız `id` + patch); bu yüzden type dağılımında `(tipsiz)` satırı görülür ve "
            "content uzunluğu yalnız create/rewrite için raporlanır.\n"
            "- İterasyon oranı = (update + rewrite) / create; 1.0 üstü, aynı artifact'in "
            "ortalama birden çok kez düzenlendiğini gösterir.\n"
            "- Artifact'li konuşma = içinde en az bir `artifacts` tool çağrısı olan konuşma.\n"
            "- **Önemli:** Bu analiz yalnızca `artifacts` tool'unu kapsar. Claude.ai'de "
            "bash_tool, create_file, str_replace gibi araçlarla da dosya üretilir; bunlar "
            "arayüzde artifact gibi görünse de export'ta farklı tool adıyla kayıt düşer "
            "ve bu modülün kapsamı dışındadır. Claude.ai'nin araç yönlendirme davranışı "
            "zaman içinde değişebilir — eski konuşmalarda aynı içerik `artifacts` ile "
            "üretilmişken yeni konuşmalarda `create_file` veya `bash_tool` tercih "
            "edilmiş olabilir.",
            blocks=[
                _common.block_bullets([
                    "Kaynak: `tool_calls.name='artifacts'` kayıtlarının `input` JSON'u.",
                    "`type` etiketleri normalize edildi: `text/markdown→markdown`, "
                    "`image/svg+xml→svg`, `application/vnd.ant.code→code`, "
                    "`application/vnd.ant.react→react`, `application/vnd.ant.mermaid→mermaid`.",
                    "`update` komutunda input'ta genelde `type` ve `content` bulunmaz "
                    "(yalnız `id` + patch); bu yüzden type dağılımında `(tipsiz)` satırı görülür ve "
                    "content uzunluğu yalnız create/rewrite için raporlanır.",
                    "İterasyon oranı = (update + rewrite) / create; 1.0 üstü, aynı artifact'in "
                    "ortalama birden çok kez düzenlendiğini gösterir.",
                    "Artifact'li konuşma = içinde en az bir `artifacts` tool çağrısı olan konuşma.",
                    "**Önemli:** Bu analiz yalnızca `artifacts` tool'unu kapsar. Claude.ai'de "
                    "bash_tool, create_file, str_replace gibi araçlarla da dosya üretilir; bunlar "
                    "arayüzde artifact gibi görünse de export'ta farklı tool adıyla kayıt düşer "
                    "ve bu modülün kapsamı dışındadır. Claude.ai'nin araç yönlendirme davranışı "
                    "zaman içinde değişebilir — eski konuşmalarda aynı içerik `artifacts` ile "
                    "üretilmişken yeni konuşmalarda `create_file` veya `bash_tool` tercih "
                    "edilmiş olabilir.",
                ]),
            ],
        ),
    ]
    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)
    return summary
