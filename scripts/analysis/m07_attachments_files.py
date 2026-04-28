"""
m07 — Attachment ve file zaman serisi + yoğunluk

- Ay bazında attachment + file sayısı
- Ay bazında attachment'lı konuşma oranı
- file_size ve content_length dağılımı
- file_type breakdown + top-4 aylık trend
- Konuşma başına attachment/file sayısı
- Attachment'lı vs attachment'sız konuşma uzunluk kıyası
- İlk human mesajında attachment oranı
- file_type × boyut kıyası
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from analysis import _common
from analysis._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "07-attachments-files"
TITLE = "Attachment ve file zaman serisi"

SIZE_BUCKETS = [
    ("<1 KB",       0,              1 * 1024),
    ("1–10 KB",     1 * 1024,       10 * 1024),
    ("10–100 KB",   10 * 1024,      100 * 1024),
    ("100 KB–1 MB", 100 * 1024,     1024 * 1024),
    (">1 MB",       1024 * 1024,    10 ** 12),
]

PER_CONVO_BUCKETS = [
    ("1",     1,  2),
    ("2–5",   2,  6),
    ("6–15",  6,  16),
    ("16+",   16, 10 ** 9),
]


def _fmt_token_short(n: float) -> str:
    if n is None:
        return "—"
    n = float(n)
    if n != n:
        return "—"
    if n < 1_000:
        return f"{int(round(n))}"
    if n < 10_000:
        return f"{n/1000:.1f}K"
    if n < 1_000_000:
        return f"{int(round(n/1000))}K"
    return f"{n/1_000_000:.1f}M"


def _fmt_bytes(b: float) -> str:
    if b is None or b != b:
        return "—"
    b = float(b)
    if b < 1024:
        return f"{int(b)} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    if b < 1024 * 1024 * 1024:
        return f"{b / 1024 / 1024:.2f} MB"
    return f"{b / 1024 / 1024 / 1024:.2f} GB"


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


def run(con: duckdb.DuckDBPyConnection, out_dir: Path, cfg: dict) -> dict:
    tz = cfg.get("tz", _common.DEFAULT_TZ)

    # ---------- aylık sayılar ----------
    monthly = con.execute(f"""
        WITH msg_month AS (
          SELECT uuid, strftime(date_trunc('month', created_at AT TIME ZONE '{tz}'), '%Y-%m') AS month
          FROM messages
        )
        SELECT mm.month,
               COALESCE(SUM(CASE WHEN a.message_uuid IS NOT NULL THEN 1 ELSE 0 END), 0) AS attachments,
               COALESCE(SUM(CASE WHEN f.message_uuid IS NOT NULL THEN 1 ELSE 0 END), 0) AS files
        FROM msg_month mm
        LEFT JOIN attachments a ON a.message_uuid = mm.uuid
        LEFT JOIN files       f ON f.message_uuid = mm.uuid
        GROUP BY 1 ORDER BY 1
    """).fetchall()

    attach_rate = con.execute(f"""
        WITH convo_month AS (
          SELECT conversation_uuid,
                 strftime(date_trunc('month', created_at AT TIME ZONE '{tz}'), '%Y-%m') AS month,
                 (attachment_total + file_total) AS has_items
          FROM _stats_conversation
          WHERE message_count > 0
        )
        SELECT month, COUNT(*) AS total,
               SUM(CASE WHEN has_items > 0 THEN 1 ELSE 0 END) AS with_items
        FROM convo_month
        GROUP BY 1 ORDER BY 1
    """).fetchall()

    # ---------- file_type breakdown ----------
    top_types = con.execute("""
        SELECT COALESCE(NULLIF(file_type, ''), '(unknown)') AS ft, COUNT(*) AS n
        FROM attachments
        GROUP BY 1 ORDER BY n DESC LIMIT 8
    """).fetchall()
    top4 = [r[0] for r in top_types[:4]]

    type_monthly: dict[str, dict[str, int]] = {}
    if top4:
        placeholders = ",".join(["?"] * len(top4))
        rows = con.execute(f"""
            WITH a2 AS (
              SELECT COALESCE(NULLIF(a.file_type, ''), '(unknown)') AS ft,
                     strftime(date_trunc('month', m.created_at AT TIME ZONE '{tz}'), '%Y-%m') AS month
              FROM attachments a
              JOIN messages m ON m.uuid = a.message_uuid
            )
            SELECT month, ft, COUNT(*) n
            FROM a2 WHERE ft IN ({placeholders})
            GROUP BY 1, 2 ORDER BY 1, 2
        """, top4).fetchall()
        for month, ft, n in rows:
            type_monthly.setdefault(month, {})[ft] = n

    # ---------- file_size ----------
    size_rows = con.execute("""
        SELECT file_size FROM attachments
        WHERE file_size IS NOT NULL AND file_size > 0
    """).fetchall()
    sizes = np.array([r[0] for r in size_rows], dtype=float)
    stats_size = _common.percentiles(sizes)

    # ---------- konuşma başına attachment/file ----------
    per_convo_rows = con.execute("""
        SELECT attachment_total, file_total
        FROM _stats_conversation
        WHERE message_count > 0
    """).fetchall()
    att_per = np.array([r[0] for r in per_convo_rows], dtype=float)
    file_per = np.array([r[1] for r in per_convo_rows], dtype=float)
    att_per_pos = att_per[att_per > 0]
    file_per_pos = file_per[file_per > 0]
    stats_att_per = _common.percentiles(att_per_pos)
    stats_file_per = _common.percentiles(file_per_pos)

    att_per_bucket_counts = _bucket_counts(att_per_pos, PER_CONVO_BUCKETS)
    file_per_bucket_counts = _bucket_counts(file_per_pos, PER_CONVO_BUCKETS)

    # ---------- attachment-li vs -siz konuşma uzunluğu ----------
    compare = con.execute("""
        SELECT (attachment_total + file_total) > 0 AS has_item,
               COUNT(*) n,
               quantile_cont(message_count, 0.5) msg_p50,
               quantile_cont(message_count, 0.95) msg_p95,
               quantile_cont(COALESCE(tokens_human,0)+COALESCE(tokens_assistant,0), 0.5) tok_p50,
               quantile_cont(COALESCE(tokens_human,0)+COALESCE(tokens_assistant,0), 0.95) tok_p95
        FROM _stats_conversation
        WHERE message_count > 0
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    compare_map = {bool(r[0]): r for r in compare}

    # ---------- ilk human mesajında attachment ----------
    first_human = con.execute("""
        WITH fh AS (
            SELECT m.conversation_uuid, m.attachment_count, m.file_count,
                   ROW_NUMBER() OVER (PARTITION BY m.conversation_uuid ORDER BY m.created_at) rn
            FROM messages m WHERE m.sender='human'
        )
        SELECT COUNT(*) n,
               SUM(CASE WHEN attachment_count > 0 OR file_count > 0 THEN 1 ELSE 0 END) with_item
        FROM fh WHERE rn = 1
    """).fetchall()
    fh_n = int(first_human[0][0]) if first_human else 0
    fh_with = int(first_human[0][1]) if first_human else 0
    fh_pct = (100.0 * fh_with / fh_n) if fh_n else 0.0

    # ---------- file_type × boyut ----------
    type_size = con.execute("""
        SELECT COALESCE(NULLIF(file_type,''),'(unknown)') ft, COUNT(*) n,
               quantile_cont(file_size, 0.5) p50,
               quantile_cont(file_size, 0.95) p95,
               MAX(file_size) mx
        FROM attachments WHERE file_size > 0
        GROUP BY 1 ORDER BY n DESC LIMIT 6
    """).fetchall()

    # ---------- grafikler ----------
    if HAS_MPL:
        # 1) aylık grouped bar
        fig, ax = plt.subplots(figsize=(12, 5))
        if monthly:
            months = [r[0] for r in monthly]
            atts = np.array([r[1] for r in monthly], dtype=float)
            fls = np.array([r[2] for r in monthly], dtype=float)
            x = np.arange(len(months))
            w = 0.42
            ax.bar(x - w/2, atts, w, label="attachment", color="#4C72B0")
            ax.bar(x + w/2, fls, w, label="file", color="#55A868")
            ax.set_xticks(x)
            ax.set_xticklabels(months, rotation=45, ha="right")
            ax.set_ylabel("adet")
            ax.set_title("Ay bazında attachment ve file sayısı")
            ax.legend()
        _common.save_fig(fig, out_dir / "monthly_counts.png")

        # 2) aylık attach oranı
        fig, ax = plt.subplots(figsize=(12, 5))
        if attach_rate:
            months_r = [r[0] for r in attach_rate]
            totals = np.array([r[1] for r in attach_rate], dtype=float)
            withs = np.array([r[2] for r in attach_rate], dtype=float)
            rates = np.where(totals > 0, withs / totals * 100.0, 0.0)
            ax.plot(range(len(months_r)), rates, marker="o", color="#C44E52", linewidth=1.5)
            ax.set_xticks(range(len(months_r)))
            ax.set_xticklabels(months_r, rotation=45, ha="right")
            ax.set_ylabel("attachment/file içeren konuşma (%)")
            ax.set_title("Ay bazında attachment'lı konuşma oranı")
            ax.set_ylim(0, max(100.0, float(rates.max()) + 5 if rates.size else 100.0))
        _common.save_fig(fig, out_dir / "monthly_attach_rate.png")

        # 3) top file types
        fig, ax = plt.subplots()
        if top_types:
            names = [r[0] for r in top_types]
            counts = [r[1] for r in top_types]
            y = np.arange(len(names))
            ax.barh(y, counts, color="#8172B2")
            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel("adet")
            ax.set_title("Attachment file_type dağılımı (top 8)")
        _common.save_fig(fig, out_dir / "top_file_types.png")

        # 4) top 4 aylık trend
        fig, ax = plt.subplots(figsize=(12, 5))
        if top4 and type_monthly:
            months_sorted = sorted(type_monthly.keys())
            colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
            for i, ft in enumerate(top4):
                series = [type_monthly.get(m, {}).get(ft, 0) for m in months_sorted]
                ax.plot(range(len(months_sorted)), series, marker="o",
                        label=ft, color=colors[i % len(colors)], linewidth=1.3)
            ax.set_xticks(range(len(months_sorted)))
            ax.set_xticklabels(months_sorted, rotation=45, ha="right")
            ax.set_ylabel("attachment sayısı")
            ax.set_title("Top-4 file_type — aylık trend")
            ax.legend(loc="upper left", fontsize=9)
        _common.save_fig(fig, out_dir / "top4_types_monthly.png")

    # Compute bucket counts (used by both chart and table)
    size_bucket_counts = _bucket_counts(sizes, SIZE_BUCKETS)

    if HAS_MPL:
        # 5) file_size kova bar (log-log hist yerine)
        fig, ax = plt.subplots()
        _bucket_bar(
            ax,
            [b[0] for b in SIZE_BUCKETS],
            size_bucket_counts,
            "#4C72B0",
            "attachment sayısı",
            "Attachment boyut kovaları",
        )
        _common.save_fig(fig, out_dir / "bucket_file_size.png")

        # 6) konuşma başına attachment / file kova pair
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        _bucket_bar(
            axes[0],
            [b[0] for b in PER_CONVO_BUCKETS],
            att_per_bucket_counts,
            "#55A868",
            "konuşma sayısı",
            "Konuşma başına attachment sayısı",
        )
        _bucket_bar(
            axes[1],
            [b[0] for b in PER_CONVO_BUCKETS],
            file_per_bucket_counts,
            "#8172B2",
            "konuşma sayısı",
            "Konuşma başına file sayısı",
        )
        _common.save_fig(fig, out_dir / "bucket_per_conversation.png")

    # ---------- CSV ----------
    _common.write_csv(
        out_dir, "monthly.csv",
        [(m, int(a), int(f)) for m, a, f in monthly],
        headers=["month", "attachments", "files"],
    )
    _common.write_csv(
        out_dir, "monthly_attach_rate.csv",
        [(m, int(t), int(w), round(100 * w / t if t else 0, 2)) for m, t, w in attach_rate],
        headers=["month", "conversations", "with_attach_or_file", "rate_pct"],
    )
    _common.write_csv(
        out_dir, "file_types.csv",
        [(n, int(c)) for n, c in top_types],
        headers=["file_type", "count"],
    )
    _common.write_csv(
        out_dir, "file_type_size.csv",
        [(ft, int(n), int(p50 or 0), int(p95 or 0), int(mx or 0))
         for ft, n, p50, p95, mx in type_size],
        headers=["file_type", "count", "size_p50", "size_p95", "size_max"],
    )

    # ---------- totals / oranlar ----------
    total_attachments = int(con.execute("SELECT COUNT(*) FROM attachments").fetchone()[0])
    total_files = int(con.execute("SELECT COUNT(*) FROM files").fetchone()[0])
    total_with_size = int(sizes.size)
    n_att_convos = int(att_per_pos.size)
    n_file_convos = int(file_per_pos.size)
    n_total_convos = int(att_per.size)

    type_rows = []
    for name, c in top_types:
        pct = 100.0 * c / total_attachments if total_attachments else 0.0
        type_rows.append([name, _common.fmt_int(c), f"%{pct:.1f}"])

    size_bucket_rows = []
    for (label, _lo, _hi), cnt in zip(SIZE_BUCKETS, size_bucket_counts):
        pct = (100.0 * cnt / total_with_size) if total_with_size else 0.0
        size_bucket_rows.append([label, _common.fmt_int(cnt), f"%{pct:.1f}"])

    per_convo_rows_att = []
    for (label, _lo, _hi), cnt in zip(PER_CONVO_BUCKETS, att_per_bucket_counts):
        pct = (100.0 * cnt / n_att_convos) if n_att_convos else 0.0
        per_convo_rows_att.append([label, _common.fmt_int(cnt), f"%{pct:.1f}"])

    per_convo_rows_file = []
    for (label, _lo, _hi), cnt in zip(PER_CONVO_BUCKETS, file_per_bucket_counts):
        pct = (100.0 * cnt / n_file_convos) if n_file_convos else 0.0
        per_convo_rows_file.append([label, _common.fmt_int(cnt), f"%{pct:.1f}"])

    # compare tablosu
    compare_rows = []
    for has_item in (False, True):
        r = compare_map.get(has_item)
        if r is None:
            continue
        label = "attachment/file var" if has_item else "attachment/file yok"
        compare_rows.append([
            label,
            _common.fmt_int(int(r[1])),
            _common.fmt_int(int(r[2] or 0)),
            _common.fmt_int(int(r[3] or 0)),
            _fmt_token_short(float(r[4] or 0)),
            _fmt_token_short(float(r[5] or 0)),
        ])

    # type × size
    type_size_rows = []
    for ft, n, p50, p95, mx in type_size:
        type_size_rows.append([
            ft, _common.fmt_int(int(n)),
            _fmt_bytes(p50 or 0),
            _fmt_bytes(p95 or 0),
            _fmt_bytes(mx or 0),
        ])

    # ---------- summary ----------
    summary = {
        "total_attachments": total_attachments,
        "total_files": total_files,
        "attachments_with_size": total_with_size,
        "file_size_bytes": stats_size,
        "file_size_human": {
            k: _fmt_bytes(v) for k, v in stats_size.items()
            if k in ("p50", "p90", "p95", "p99", "max")
        },
        "n_conversations_with_attachment": n_att_convos,
        "n_conversations_with_file": n_file_convos,
        "first_human_msg_with_item": {"n": fh_n, "with_item": fh_with, "pct": round(fh_pct, 2)},
        "top_file_types": [{"type": n, "count": int(c)} for n, c in top_types],
        "headline": (
            f"{_common.fmt_int(total_attachments)} attachment + "
            f"{_common.fmt_int(total_files)} file; "
            f"medyan boy {_fmt_bytes(stats_size['p50'] or 0)}, "
            f"p95 {_fmt_bytes(stats_size['p95'] or 0)}; "
            f"ilk human mesajında attachment/file oranı %{fh_pct:.1f}."
        ),
    }
    _common.write_json(out_dir, "summary.json", summary)

    # ---------- tablolar ----------
    size_tbl = _common.markdown_table(
        [[
            "file_size",
            _common.fmt_int(stats_size["n"]),
            _fmt_bytes(stats_size["min"]),
            _fmt_bytes(stats_size["p50"]),
            _fmt_bytes(stats_size["p90"]),
            _fmt_bytes(stats_size["p95"]),
            _fmt_bytes(stats_size["p99"]),
            _fmt_bytes(stats_size["max"]),
        ]],
        headers=["metric", "n", "min", "p50", "p90", "p95", "p99", "max"],
    )

    per_convo_tbl = _common.markdown_table(
        [
            [
                "attachment/konuşma",
                _common.fmt_int(stats_att_per["n"]),
                _common.fmt_int(int(stats_att_per["min"] or 0)),
                _common.fmt_int(int(stats_att_per["p50"] or 0)),
                _common.fmt_int(int(stats_att_per["p90"] or 0)),
                _common.fmt_int(int(stats_att_per["p95"] or 0)),
                _common.fmt_int(int(stats_att_per["p99"] or 0)),
                _common.fmt_int(int(stats_att_per["max"] or 0)),
            ],
            [
                "file/konuşma",
                _common.fmt_int(stats_file_per["n"]),
                _common.fmt_int(int(stats_file_per["min"] or 0)),
                _common.fmt_int(int(stats_file_per["p50"] or 0)),
                _common.fmt_int(int(stats_file_per["p90"] or 0)),
                _common.fmt_int(int(stats_file_per["p95"] or 0)),
                _common.fmt_int(int(stats_file_per["p99"] or 0)),
                _common.fmt_int(int(stats_file_per["max"] or 0)),
            ],
        ],
        headers=["metric", "n", "min", "p50", "p90", "p95", "p99", "max"],
    )

    # ── Structured blocks ──
    highlight_items = [
        f"Toplam attachment: **{_common.fmt_int(total_attachments)}** "
        f"(boyut bilgisi olan: **{_common.fmt_int(total_with_size)}**)",
        f"Toplam file (referans): **{_common.fmt_int(total_files)}**",
        f"Attachment içeren konuşma: **{_common.fmt_int(n_att_convos)}** / "
        f"**{_common.fmt_int(n_total_convos)}** (**%{100.0 * n_att_convos / max(n_total_convos,1):.1f}**)",
        f"File referansı içeren konuşma: **{_common.fmt_int(n_file_convos)}** "
        f"(**%{100.0 * n_file_convos / max(n_total_convos,1):.1f}**)",
        f"Attachment boy: medyan **{_fmt_bytes(stats_size['p50'])}**, "
        f"p95 **{_fmt_bytes(stats_size['p95'])}**, max **{_fmt_bytes(stats_size['max'])}**",
        f"İlk human mesajında attachment/file: **{_common.fmt_int(fh_with)}** / "
        f"**{_common.fmt_int(fh_n)}** (**%{fh_pct:.1f}**)",
        f"En sık file_type: **{top_types[0][0] if top_types else '—'}** "
        f"(**{_common.fmt_int(top_types[0][1]) if top_types else 0}** adet)",
    ]

    size_quantile_columns = [
        {"key": "metric", "label": "metric", "align": "left"},
        {"key": "n",      "label": "n",      "align": "right"},
        {"key": "min",    "label": "min",    "align": "right"},
        {"key": "p50",    "label": "p50",    "align": "right"},
        {"key": "p90",    "label": "p90",    "align": "right"},
        {"key": "p95",    "label": "p95",    "align": "right"},
        {"key": "p99",    "label": "p99",    "align": "right"},
        {"key": "max",    "label": "max",    "align": "right"},
    ]
    size_quantile_rows = [[
        "file_size",
        _common.fmt_int(stats_size["n"]),
        _fmt_bytes(stats_size["min"]),
        _fmt_bytes(stats_size["p50"]),
        _fmt_bytes(stats_size["p90"]),
        _fmt_bytes(stats_size["p95"]),
        _fmt_bytes(stats_size["p99"]),
        _fmt_bytes(stats_size["max"]),
    ]]

    size_bucket_cols = [
        {"key": "range", "label": "boyut aralığı", "align": "left"},
        {"key": "n",     "label": "attachment",    "align": "right"},
        {"key": "pct",   "label": "oran",          "align": "right"},
    ]
    size_buckets_payload = [
        {"label": lbl, "count": int(c),
         "pct": (100.0 * c / total_with_size) if total_with_size else 0.0}
        for (lbl, _, _), c in zip(SIZE_BUCKETS, size_bucket_counts)
    ]

    per_convo_quantile_rows = [
        [
            "attachment/konuşma",
            _common.fmt_int(stats_att_per["n"]),
            _common.fmt_int(int(stats_att_per["min"] or 0)),
            _common.fmt_int(int(stats_att_per["p50"] or 0)),
            _common.fmt_int(int(stats_att_per["p90"] or 0)),
            _common.fmt_int(int(stats_att_per["p95"] or 0)),
            _common.fmt_int(int(stats_att_per["p99"] or 0)),
            _common.fmt_int(int(stats_att_per["max"] or 0)),
        ],
        [
            "file/konuşma",
            _common.fmt_int(stats_file_per["n"]),
            _common.fmt_int(int(stats_file_per["min"] or 0)),
            _common.fmt_int(int(stats_file_per["p50"] or 0)),
            _common.fmt_int(int(stats_file_per["p90"] or 0)),
            _common.fmt_int(int(stats_file_per["p95"] or 0)),
            _common.fmt_int(int(stats_file_per["p99"] or 0)),
            _common.fmt_int(int(stats_file_per["max"] or 0)),
        ],
    ]

    per_convo_bucket_cols = [
        {"key": "range", "label": "adet aralığı", "align": "left"},
        {"key": "n",     "label": "konuşma",      "align": "right"},
        {"key": "pct",   "label": "oran",         "align": "right"},
    ]
    att_buckets_payload = [
        {"label": lbl, "count": int(c),
         "pct": (100.0 * c / n_att_convos) if n_att_convos else 0.0}
        for (lbl, _, _), c in zip(PER_CONVO_BUCKETS, att_per_bucket_counts)
    ]
    file_buckets_payload = [
        {"label": lbl, "count": int(c),
         "pct": (100.0 * c / n_file_convos) if n_file_convos else 0.0}
        for (lbl, _, _), c in zip(PER_CONVO_BUCKETS, file_per_bucket_counts)
    ]

    compare_columns = [
        {"key": "kitle", "label": "kitle",     "align": "left"},
        {"key": "n",     "label": "n konuşma", "align": "right"},
        {"key": "m50",   "label": "mesaj p50", "align": "right"},
        {"key": "m95",   "label": "mesaj p95", "align": "right"},
        {"key": "t50",   "label": "token p50", "align": "right"},
        {"key": "t95",   "label": "token p95", "align": "right"},
    ]

    type_columns = [
        {"key": "type", "label": "file_type", "align": "left"},
        {"key": "n",    "label": "adet",      "align": "right"},
        {"key": "pct",  "label": "oran",      "align": "right"},
    ]
    top_type_buckets = [
        {"label": name,
         "count": int(c),
         "pct": (100.0 * c / total_attachments) if total_attachments else 0.0}
        for name, c in top_types
    ]
    type_months_sorted = sorted(type_monthly.keys())
    top4_monthly_data = [
        {"month": m, **{ft: int(type_monthly.get(m, {}).get(ft, 0)) for ft in top4}}
        for m in type_months_sorted
    ]
    top4_monthly_series = [{"key": ft, "label": ft} for ft in top4]

    type_size_columns = [
        {"key": "type", "label": "file_type", "align": "left"},
        {"key": "n",    "label": "n",         "align": "right"},
        {"key": "p50",  "label": "boy p50",   "align": "right"},
        {"key": "p95",  "label": "boy p95",   "align": "right"},
        {"key": "max",  "label": "boy max",   "align": "right"},
    ]

    # Aylık
    monthly_count_buckets_att = [
        {"label": m, "count": int(a), "pct": None}
        for m, a, _ in monthly
    ]
    monthly_count_buckets_file = [
        {"label": m, "count": int(f), "pct": None}
        for m, _, f in monthly
    ]
    monthly_attach_rate_buckets = [
        {"label": m, "count": round(100.0 * w / t if t else 0.0, 1), "pct": None}
        for m, t, w in attach_rate
    ]

    notes_items = [
        "`attachments` = ekran yapıştırması / yüklenen metin dosyası (içerik DB'de).",
        "`files` = sadece referans (uuid + ad); içerik veya boyut yoktur.",
        "`attachments.file_type` bazı kayıtlarda boş — `(unknown)` olarak etiketlendi.",
        "`file_size` ile `content_length` paste metni için neredeyse aynı değerleri verir; "
        "ayrı tablo olarak sunulmaz.",
        f"Boyut bilgisi eksik kayıt sayısı: "
        f"**{total_attachments - total_with_size}** (toplam içinde).",
    ]

    sections = [
        _common.Section(
            "Öne çıkanlar",
            f"- Toplam attachment: **{_common.fmt_int(total_attachments)}** "
            f"(boyut bilgisi olan: **{_common.fmt_int(total_with_size)}**)\n"
            f"- Toplam file (referans): **{_common.fmt_int(total_files)}**\n"
            f"- Attachment içeren konuşma: **{_common.fmt_int(n_att_convos)}** / "
            f"**{_common.fmt_int(n_total_convos)}** (**%{100.0 * n_att_convos / max(n_total_convos,1):.1f}**)\n"
            f"- File referansı içeren konuşma: **{_common.fmt_int(n_file_convos)}** "
            f"(**%{100.0 * n_file_convos / max(n_total_convos,1):.1f}**)\n"
            f"- Attachment boy: medyan **{_fmt_bytes(stats_size['p50'])}**, "
            f"p95 **{_fmt_bytes(stats_size['p95'])}**, max **{_fmt_bytes(stats_size['max'])}**\n"
            f"- İlk human mesajında attachment/file: **{_common.fmt_int(fh_with)}** / "
            f"**{_common.fmt_int(fh_n)}** (**%{fh_pct:.1f}**)\n"
            f"- En sık file_type: **{top_types[0][0] if top_types else '—'}** "
            f"(**{_common.fmt_int(top_types[0][1]) if top_types else 0}** adet)",
            blocks=[_common.block_bullets(highlight_items)],
        ),
        _common.Section(
            "Ay bazında attachment + file",
            "",
            blocks=[
                _common.block_bucket_chart(
                    label="Attachment — aylık sayı",
                    buckets=monthly_count_buckets_att,
                    image="monthly_counts.png",
                    xlabel="ay",
                ),
                _common.block_bucket_chart(
                    label="File — aylık sayı",
                    buckets=monthly_count_buckets_file,
                    xlabel="ay",
                ),
            ],
        ),
        _common.Section(
            "Ay bazında attachment'lı konuşma oranı",
            "",
            blocks=[
                _common.block_bucket_chart(
                    label="Attachment/file içeren konuşma oranı (%)",
                    buckets=monthly_attach_rate_buckets,
                    image="monthly_attach_rate.png",
                    xlabel="ay",
                ),
            ],
        ),
        _common.Section(
            "Attachment boyut dağılımı",
            size_tbl,
            blocks=[_common.block_table(size_quantile_columns, size_quantile_rows)],
        ),
        _common.Section(
            "Boyut kovaları",
            _common.markdown_table(size_bucket_rows, headers=["boyut aralığı", "attachment", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    label="Attachment boyut kovaları",
                    buckets=size_buckets_payload,
                    image="bucket_file_size.png",
                    xlabel="boyut aralığı",
                ),
                _common.block_table(size_bucket_cols, size_bucket_rows),
            ],
        ),
        _common.Section(
            "Konuşma başına attachment/file sayısı",
            per_convo_tbl,
            blocks=[_common.block_table(size_quantile_columns, per_convo_quantile_rows)],
        ),
        _common.Section(
            "Konuşma başına kovalar — attachment",
            _common.markdown_table(
                per_convo_rows_att,
                headers=["adet aralığı", "konuşma", "oran"],
            ),
            blocks=[
                _common.block_bucket_chart(
                    label="Konuşma başına attachment sayısı",
                    buckets=att_buckets_payload,
                    xlabel="adet aralığı",
                ),
                _common.block_table(per_convo_bucket_cols, per_convo_rows_att),
            ],
        ),
        _common.Section(
            "Konuşma başına kovalar — file",
            _common.markdown_table(
                per_convo_rows_file,
                headers=["adet aralığı", "konuşma", "oran"],
            ),
            blocks=[
                _common.block_bucket_chart(
                    label="Konuşma başına file sayısı",
                    buckets=file_buckets_payload,
                    xlabel="adet aralığı",
                ),
                _common.block_table(per_convo_bucket_cols, per_convo_rows_file),
            ],
        ),
        _common.Section(
            "Attachment/file'lı vs yok — konuşma uzunluğu",
            _common.markdown_table(
                compare_rows,
                headers=["kitle", "n konuşma", "mesaj p50", "mesaj p95", "token p50", "token p95"],
            ),
            blocks=[_common.block_table(compare_columns, compare_rows)],
        ),
        _common.Section(
            "En yaygın file_type (top 8)",
            _common.markdown_table(type_rows, headers=["file_type", "adet", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    label="Attachment file_type dağılımı (top 8)",
                    buckets=top_type_buckets,
                    image="top_file_types.png",
                    xlabel="adet",
                ),
                _common.block_table(type_columns, type_rows),
            ],
        ),
        _common.Section(
            "Top-4 file_type — aylık trend",
            "",
            blocks=[_common.block_line_chart(
                "Top-4 file_type — aylık trend",
                top4_monthly_data,
                top4_monthly_series,
                x_key="month",
            )],
        ),
        _common.Section(
            "file_type × boyut (top 6)",
            _common.markdown_table(
                type_size_rows,
                headers=["file_type", "n", "boy p50", "boy p95", "boy max"],
            ),
            blocks=[_common.block_table(type_size_columns, type_size_rows)],
        ),
        _common.Section(
            "Grafikler",
            "- `monthly_counts.png` — ay bazında attachment vs file\n"
            "- `monthly_attach_rate.png` — attachment'lı konuşma oranı\n"
            "- `top_file_types.png` — top-8 file_type\n"
            "- `top4_types_monthly.png` — top-4 tipin aylık trendi\n"
            "- `bucket_file_size.png` — attachment boyut kovaları\n"
            "- `bucket_per_conversation.png` — konuşma başına attachment ve file sayısı",
            blocks=[],
        ),
        _common.Section(
            "Notlar",
            "- `attachments` = ekran yapıştırması / yüklenen metin dosyası (içerik DB'de).\n"
            "- `files` = sadece referans (uuid + ad); içerik veya boyut yoktur.\n"
            "- `attachments.file_type` bazı kayıtlarda boş — `(unknown)` olarak etiketlendi.\n"
            "- `file_size` ile `content_length` paste metni için neredeyse aynı değerleri verir; "
            "ayrı tablo olarak sunulmaz.\n"
            "- Boyut bilgisi eksik kayıt sayısı: "
            f"**{total_attachments - total_with_size}** (toplam içinde).",
            blocks=[_common.block_bullets(notes_items)],
        ),
    ]
    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)
    return summary
