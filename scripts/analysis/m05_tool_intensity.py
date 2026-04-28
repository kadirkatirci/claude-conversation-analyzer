"""
m05 — Tool çağrısı yoğunluğu

- Konuşma bazında `tool_calls_total` dağılımı (quantile + kova bar)
- Kategorik kovalar: 0 / 1 / 2-9 / 10-49 / 50+
- Tool kullanan konuşmalar alt-kitle quantile
- Tool kullanan vs kullanmayan konuşma uzunluğu kıyası
- Konuşma başına tool çeşitliliği (distinct tool count) kova dağılımı
- Tool kategori gruplaması (dosya yazma / okuma / execute / web / ...)
- En popüler 15 tool — toplam çağrı + farklı konuşma + çağrı/konuşma + oran
- Ay bazında üst-5 tool trendi (line)
- Tool sonuç hata oranı (tool_results.is_error)
- Tool çağrısı süresi dağılımı (stop - start)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from analysis import _common
from analysis._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "05-tool-intensity"
TITLE = "Tool çağrısı yoğunluğu"


TOOL_BUCKETS: list[tuple[str, float, float]] = [
    ("0",      0,   1),
    ("1",      1,   2),
    ("2–9",    2,   10),
    ("10–49",  10,  50),
    ("50+",    50,  1e9),
]

DIVERSITY_BUCKETS: list[tuple[str, float, float]] = [
    ("0",    0,  1),
    ("1",    1,  2),
    ("2–3",  2,  4),
    ("4–6",  4,  7),
    ("7+",   7,  1e9),
]

DURATION_BUCKETS: list[tuple[str, float, float]] = [
    ("<1sn",     0,    1),
    ("1–5sn",    1,    5),
    ("5–30sn",   5,    30),
    ("30sn–2dk", 30,   120),
    ("2dk+",     120,  1e9),
]


# Kategori kuralları — önce prefix/regex değil, explicit mapping (anlaşılır)
TOOL_CATEGORIES: list[tuple[str, list[str]]] = [
    ("dosya yazma",     ["write_file", "create_file", "str_replace", "edit_file"]),
    ("dosya okuma",     ["read_file", "read_text_file", "view"]),
    ("execute",         ["bash_tool", "repl"]),
    ("web",             ["web_search", "web_fetch"]),
    ("artifact",        ["artifacts", "present_files"]),
    ("dizin",           ["create_directory", "list_directory", "directory_tree"]),
    ("arama",           ["grep", "glob", "search_files"]),
    ("diğer",           []),  # fallback — bu listeye giren olmadığı için boş
]


def _categorize(name: str) -> str:
    """Tool adını kategoriye eşler. MCP prefix'i (`filesystem:`) çıkarılır."""
    base = name.split(":", 1)[-1] if ":" in name else name
    for cat, members in TOOL_CATEGORIES:
        if base in members:
            return cat
    return "diğer"


def _bucket_counts(values: np.ndarray, buckets: list[tuple[str, float, float]]) -> list[tuple[str, int, float]]:
    total = values.size
    out = []
    for label, lo, hi in buckets:
        mask = (values >= lo) & (values < hi)
        n = int(mask.sum())
        pct = (100.0 * n / total) if total else 0.0
        out.append((label, n, pct))
    return out


def _bucket_table(counts: list[tuple[str, int, float]]) -> list[list]:
    return [[label, _common.fmt_int(n), f"%{pct:.1f}"] for label, n, pct in counts]


def _bucket_bar_single(
    counts: list[tuple[str, int, float]],
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    path: Path,
) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots()
    labels = [c[0] for c in counts]
    ns = [c[1] for c in counts]
    pcts = [c[2] for c in counts]
    bars = ax.bar(labels, ns, color=color)
    ymax = max(ns) if ns else 1
    ax.set_ylim(0, ymax * 1.15)
    for bar, n, pct in zip(bars, ns, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{n:,}\n%{pct:.1f}".replace(",", " "),
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    _common.save_fig(fig, path)


def _fmt_duration(sec: float) -> str:
    if sec is None or sec != sec:
        return "—"
    if sec < 1:
        return f"{sec*1000:.0f}ms"
    if sec < 60:
        return f"{sec:.1f}sn"
    if sec < 3600:
        return f"{sec/60:.1f}dk"
    return f"{sec/3600:.1f}sa"


def run(con: duckdb.DuckDBPyConnection, out_dir: Path, cfg: dict) -> dict:
    # ---------- Konuşma bazında tool_calls_total ----------
    rows = con.execute("""
        SELECT tool_calls_total
        FROM _stats_conversation
        WHERE message_count > 0
    """).fetchall()
    tool_totals = np.array([r[0] for r in rows], dtype=float)

    stats_all = _common.percentiles(tool_totals)
    tool_positive = tool_totals[tool_totals > 0]
    stats_users = _common.percentiles(tool_positive)

    n_convo = int(tool_totals.size)
    n_with_tool = int(tool_positive.size)

    counts_tool = _bucket_counts(tool_totals, TOOL_BUCKETS)

    # ---------- Tool-kullanan vs kullanmayan: konuşma uzunluğu kıyası ----------
    len_compare_rows = con.execute("""
        SELECT
          (tool_calls_total > 0) AS has_tool,
          COUNT(*) AS n,
          MEDIAN(message_count) AS p50_msg,
          QUANTILE_CONT(message_count, 0.95) AS p95_msg,
          MEDIAN(tokens_human + tokens_assistant) AS p50_tok,
          QUANTILE_CONT(tokens_human + tokens_assistant, 0.95) AS p95_tok
        FROM _stats_conversation
        WHERE message_count > 0
        GROUP BY 1
        ORDER BY 1
    """).fetchall()
    len_by_tool = {bool(r[0]): r for r in len_compare_rows}
    # Guard: if only one side exists, use sentinel
    no_tool = len_by_tool.get(False, (False, 0, 0, 0, 0, 0))
    yes_tool = len_by_tool.get(True,  (True,  0, 0, 0, 0, 0))

    # ---------- Konuşma başına tool çeşitliliği ----------
    diversity_rows = con.execute("""
        SELECT COUNT(DISTINCT name) AS n_distinct
        FROM tool_calls
        WHERE name IS NOT NULL
        GROUP BY conversation_uuid
    """).fetchall()
    # Tool kullanmayan konuşmalar buraya düşmez; 0'ları manuel ekle
    diversity = np.array([r[0] for r in diversity_rows], dtype=float)
    n_zero_diversity = n_convo - int(diversity.size)
    diversity_full = np.concatenate([diversity, np.zeros(n_zero_diversity)])
    counts_diversity = _bucket_counts(diversity_full, DIVERSITY_BUCKETS)
    stats_diversity = _common.percentiles(diversity)  # sadece tool kullananlar

    # ---------- Top 15 tool + farklı konuşma + oran ----------
    top_tools = con.execute("""
        SELECT name,
               COUNT(*) AS total_calls,
               COUNT(DISTINCT conversation_uuid) AS n_convo
        FROM tool_calls
        WHERE name IS NOT NULL
        GROUP BY 1
        ORDER BY total_calls DESC
        LIMIT 15
    """).fetchall()

    total_tool_calls = con.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]

    # ---------- Tool kategori gruplaması ----------
    all_tool_counts = con.execute("""
        SELECT name, COUNT(*) AS n
        FROM tool_calls
        WHERE name IS NOT NULL
        GROUP BY 1
    """).fetchall()
    category_totals: dict[str, int] = {cat: 0 for cat, _ in TOOL_CATEGORIES}
    category_examples: dict[str, set] = {cat: set() for cat, _ in TOOL_CATEGORIES}
    for name, n in all_tool_counts:
        cat = _categorize(name)
        category_totals[cat] += int(n)
        category_examples[cat].add(name)
    total_categorized = sum(category_totals.values())
    category_rows = [
        [
            cat,
            _common.fmt_int(category_totals[cat]),
            f"%{100*category_totals[cat]/max(total_categorized,1):.1f}",
            ", ".join(sorted(category_examples[cat])[:6])
            + ("…" if len(category_examples[cat]) > 6 else ""),
        ]
        for cat, _ in TOOL_CATEGORIES
    ]

    # ---------- Ay bazında üst-5 tool trendi ----------
    top5 = [r[0] for r in top_tools[:5]]
    monthly_top: dict[str, dict[str, int]] = {}
    if top5:
        placeholders = ",".join(["?"] * len(top5))
        tz = cfg.get("tz", _common.DEFAULT_TZ)
        monthly_rows = con.execute(f"""
            SELECT
              strftime(date_trunc('month', start_timestamp AT TIME ZONE '{tz}'), '%Y-%m') AS month,
              name,
              COUNT(*) AS n
            FROM tool_calls
            WHERE name IN ({placeholders})
              AND start_timestamp IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
        """, top5).fetchall()
        for m, name, n in monthly_rows:
            monthly_top.setdefault(m, {})[name] = n
    months_sorted = sorted(monthly_top.keys())

    # ---------- Tool hata oranı ----------
    err_row = con.execute("""
        SELECT
          COUNT(*) AS total_results,
          SUM(CASE WHEN is_error THEN 1 ELSE 0 END) AS errors
        FROM tool_results
    """).fetchone()
    total_results, total_errors = int(err_row[0]), int(err_row[1] or 0)
    error_rate = (100.0 * total_errors / total_results) if total_results else 0.0

    # Hata oranı — tool bazında top 15 (mutlak error sayısına göre)
    err_per_tool = con.execute("""
        SELECT
          name,
          COUNT(*) AS total,
          SUM(CASE WHEN is_error THEN 1 ELSE 0 END) AS errors
        FROM tool_results
        WHERE name IS NOT NULL
        GROUP BY 1
        HAVING errors > 0
        ORDER BY errors DESC
        LIMIT 15
    """).fetchall()

    # ---------- Tool çağrısı süresi ----------
    duration_rows = con.execute("""
        SELECT epoch(stop_timestamp - start_timestamp) AS dur_s
        FROM tool_calls
        WHERE start_timestamp IS NOT NULL AND stop_timestamp IS NOT NULL
    """).fetchall()
    durations = np.array([r[0] for r in duration_rows], dtype=float)
    durations = durations[durations >= 0]  # negatif/bozuk kayıtları at
    stats_duration = _common.percentiles(durations)
    counts_duration = _bucket_counts(durations, DURATION_BUCKETS)

    # ---------- Grafikler ----------
    # G1: kova bar (log-log hist yerine)
    _bucket_bar_single(
        counts_tool,
        title=f"Konuşma başına tool sayısı — kova dağılımı "
              f"(n={_common.fmt_int(n_convo)})",
        xlabel="tool çağrısı (aralık)",
        ylabel="konuşma sayısı",
        color="#4C72B0",
        path=out_dir / "bucket_tool_calls.png",
    )

    # G2: tool çeşitliliği
    _bucket_bar_single(
        counts_diversity,
        title=f"Konuşma başına farklı tool sayısı — kova dağılımı "
              f"(n={_common.fmt_int(n_convo)})",
        xlabel="farklı tool sayısı (distinct)",
        ylabel="konuşma sayısı",
        color="#55A868",
        path=out_dir / "bucket_tool_diversity.png",
    )

    # G3: tool süre kovası
    _bucket_bar_single(
        counts_duration,
        title=f"Tool çağrı süresi — kova dağılımı (n={_common.fmt_int(int(durations.size))})",
        xlabel="süre aralığı",
        ylabel="çağrı sayısı",
        color="#8172B3",
        path=out_dir / "bucket_tool_duration.png",
    )

    if HAS_MPL:
        # G4: top tools bar (mevcut)
        fig, ax = plt.subplots(figsize=(11, 6))
        if top_tools:
            names = [r[0] for r in top_tools]
            counts_b = [r[1] for r in top_tools]
            y = np.arange(len(names))
            ax.barh(y, counts_b, color="#4C72B0")
            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel("toplam çağrı")
            ax.set_title(
                f"En çok kullanılan 15 tool (toplam {total_tool_calls:,} çağrı)"
                .replace(",", " ")
            )
        _common.save_fig(fig, out_dir / "top_tools.png")

        # G5: kategori bar
        fig, ax = plt.subplots(figsize=(10, 5))
        cat_names = [cat for cat, _ in TOOL_CATEGORIES]
        cat_counts = [category_totals[c] for c in cat_names]
        ax.bar(cat_names, cat_counts, color="#55A868")
        ax.set_ylabel("toplam çağrı")
        ax.set_title("Tool kategori dağılımı")
        for i, v in enumerate(cat_counts):
            if v > 0:
                pct = 100 * v / max(total_categorized, 1)
                ax.text(i, v, f"{v:,}\n%{pct:.1f}".replace(",", " "),
                        ha="center", va="bottom", fontsize=9)
        ymax = max(cat_counts) if cat_counts else 1
        ax.set_ylim(0, ymax * 1.18)
        ax.grid(axis="x", visible=False)
        _common.save_fig(fig, out_dir / "tool_categories.png")

        # G6: üst-5 aylık (mevcut)
        fig, ax = plt.subplots(figsize=(12, 5))
        if months_sorted and top5:
            colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
            x = np.arange(len(months_sorted))
            for i, name in enumerate(top5):
                series = [monthly_top.get(m, {}).get(name, 0) for m in months_sorted]
                ax.plot(x, series, marker="o", label=name, color=colors[i % len(colors)], linewidth=1.3)
            ax.set_xticks(x)
            ax.set_xticklabels(months_sorted, rotation=45, ha="right")
            ax.set_ylabel("çağrı sayısı")
            ax.set_title("Üst-5 tool — aylık trend")
            ax.legend(loc="upper left", fontsize=9)
        _common.save_fig(fig, out_dir / "top5_monthly.png")

    # ---------- CSV'ler ----------
    _common.write_csv(
        out_dir, "top_tools.csv",
        [(name, int(tc), int(nc),
          round(tc/nc, 2) if nc else 0,
          round(100*tc/max(total_tool_calls,1), 2))
         for name, tc, nc in top_tools],
        headers=["tool_name", "calls", "distinct_conversations",
                 "calls_per_conversation", "share_pct"],
    )
    mt_rows = []
    for m in months_sorted:
        for name in top5:
            mt_rows.append((m, name, int(monthly_top.get(m, {}).get(name, 0))))
    _common.write_csv(
        out_dir, "monthly_top5.csv", mt_rows,
        headers=["month", "tool_name", "calls"],
    )
    _common.write_csv(
        out_dir, "conversation_tool_totals.csv",
        [(int(v),) for v in tool_totals],
        headers=["tool_calls_total"],
    )
    _common.write_csv(
        out_dir, "tool_categories.csv",
        [(cat, category_totals[cat],
          round(100*category_totals[cat]/max(total_categorized,1), 2))
         for cat, _ in TOOL_CATEGORIES],
        headers=["category", "calls", "share_pct"],
    )
    _common.write_csv(
        out_dir, "tool_errors.csv",
        [(name, int(total), int(errors),
          round(100*errors/max(total,1), 2))
         for name, total, errors in err_per_tool],
        headers=["tool_name", "calls", "errors", "error_rate_pct"],
    )

    # ---------- Summary ----------
    summary = {
        "conversations_total": n_convo,
        "conversations_with_tool": n_with_tool,
        "conversations_without_tool": n_convo - n_with_tool,
        "tool_calls_total_in_dataset": int(total_tool_calls),
        "per_conversation_all":       stats_all,
        "per_conversation_users_only": stats_users,
        "buckets_tool_calls":  {lbl: n for lbl, n, _ in counts_tool},
        "buckets_diversity":   {lbl: n for lbl, n, _ in counts_diversity},
        "diversity_stats_users_only": stats_diversity,
        "length_compare": {
            "no_tool":  {"n": int(no_tool[1]),  "p50_msg": float(no_tool[2] or 0),
                         "p95_msg": float(no_tool[3] or 0),
                         "p50_tok": float(no_tool[4] or 0),
                         "p95_tok": float(no_tool[5] or 0)},
            "with_tool":{"n": int(yes_tool[1]), "p50_msg": float(yes_tool[2] or 0),
                         "p95_msg": float(yes_tool[3] or 0),
                         "p50_tok": float(yes_tool[4] or 0),
                         "p95_tok": float(yes_tool[5] or 0)},
        },
        "categories": {cat: {
            "calls": category_totals[cat],
            "share_pct": round(100*category_totals[cat]/max(total_categorized,1), 2),
            "members": sorted(category_examples[cat]),
        } for cat, _ in TOOL_CATEGORIES},
        "top_tools": [
            {"name": n, "calls": int(c), "distinct_conversations": int(cv),
             "calls_per_conversation": round(c/cv, 2) if cv else 0,
             "share_pct": round(100*c/max(total_tool_calls,1), 2)}
            for n, c, cv in top_tools
        ],
        "errors": {
            "total_results":    total_results,
            "total_errors":     total_errors,
            "error_rate_pct":   round(error_rate, 2),
            "top_error_tools":  [
                {"name": n, "calls": int(t), "errors": int(e),
                 "error_rate_pct": round(100*e/max(t,1), 2)}
                for n, t, e in err_per_tool
            ],
        },
        "duration_seconds": stats_duration,
        "buckets_duration": {lbl: n for lbl, n, _ in counts_duration},
        "headline": (
            f"{n_with_tool}/{n_convo} konuşmada tool kullanılmış "
            f"(%{100*n_with_tool/max(n_convo,1):.1f}); "
            f"tool kullanan konuşmalarda medyan {int(stats_users['p50'])}, "
            f"p95 {int(stats_users['p95'])}, max {int(stats_users['max'])} tool; "
            f"toplam {total_tool_calls:,} çağrı, %{error_rate:.1f} hata, "
            f"medyan süre {_fmt_duration(stats_duration['p50'])}."
        ).replace(",", " "),
    }
    _common.write_json(out_dir, "summary.json", summary)

    # ---------- Rapor ----------
    highlights = (
        f"- Toplam konuşma (non-empty): **{_common.fmt_int(n_convo)}**\n"
        f"- En az 1 tool çağrısı olan: **{_common.fmt_int(n_with_tool)}** "
        f"(**%{100*n_with_tool/max(n_convo,1):.1f}**)\n"
        f"- Toplam tool çağrısı: **{_common.fmt_int(total_tool_calls)}**\n"
        f"- Tool kullanan konuşmalarda medyan **{int(stats_users['p50'])}**, "
        f"p95 **{int(stats_users['p95'])}**, max **{int(stats_users['max'])}** tool\n"
        f"- Tool kullanan konuşmalar ortalama **{int(yes_tool[2])}** mesaj / "
        f"tool kullanmayanlar **{int(no_tool[2])}** (medyan)\n"
        f"- Konuşma başına farklı tool (tool kullananlarda): "
        f"medyan **{int(stats_diversity['p50'])}**, max **{int(stats_diversity['max'])}**\n"
        f"- Tool sonuç hata oranı: "
        f"**{_common.fmt_int(total_errors)}** / **{_common.fmt_int(total_results)}** "
        f"→ **%{error_rate:.1f}**\n"
        f"- Tool çağrısı süresi: medyan **{_fmt_duration(stats_duration['p50'])}**, "
        f"p95 **{_fmt_duration(stats_duration['p95'])}**, "
        f"max **{_fmt_duration(stats_duration['max'])}**"
    )

    # Tool-kullanan alt-kitle quantile
    users_quantile = _common.percentile_table(stats_users, "tool_calls/konuşma (tool-kullanan)")

    # Uzunluk kıyası tablosu
    length_compare_table = _common.markdown_table(
        [
            ["tool yok",  _common.fmt_int(int(no_tool[1])),
             _common.fmt_int(int(no_tool[2])),  _common.fmt_int(int(no_tool[3])),
             _fmt_token_or_int(no_tool[4]),     _fmt_token_or_int(no_tool[5])],
            ["tool var",  _common.fmt_int(int(yes_tool[1])),
             _common.fmt_int(int(yes_tool[2])), _common.fmt_int(int(yes_tool[3])),
             _fmt_token_or_int(yes_tool[4]),    _fmt_token_or_int(yes_tool[5])],
        ],
        headers=["kitle", "n konuşma",
                 "mesaj p50", "mesaj p95",
                 "token p50", "token p95"],
    )

    # Top 15 tool genişletilmiş tablo
    top_table = _common.markdown_table(
        [
            [i + 1, name, _common.fmt_int(tc),
             _common.fmt_int(nc),
             f"{tc/nc:.1f}" if nc else "—",
             f"%{100*tc/max(total_tool_calls,1):.1f}"]
            for i, (name, tc, nc) in enumerate(top_tools)
        ],
        headers=["#", "tool", "çağrı", "farklı konuşma",
                 "çağrı/konuşma", "toplamdaki oran"],
    )

    # Kategori tablosu
    category_table = _common.markdown_table(
        category_rows,
        headers=["kategori", "çağrı", "oran", "üyeler"],
    )

    # Hata tablosu (top 15 mutlak error)
    if err_per_tool:
        error_table = _common.markdown_table(
            [
                [name, _common.fmt_int(total), _common.fmt_int(errors),
                 f"%{100*errors/max(total,1):.1f}"]
                for name, total, errors in err_per_tool
            ],
            headers=["tool", "çağrı", "hata", "hata oranı"],
        )
    else:
        error_table = "_Hata kaydı yok._"

    # ── Structured blocks ──
    def _buckets_payload(counts: list[tuple[str, int, float]]) -> list[dict]:
        return [{"label": lbl, "count": n, "pct": pct} for lbl, n, pct in counts]

    highlight_items = [
        f"Toplam konuşma (non-empty): **{_common.fmt_int(n_convo)}**",
        f"En az 1 tool çağrısı olan: **{_common.fmt_int(n_with_tool)}** "
        f"(**%{100*n_with_tool/max(n_convo,1):.1f}**)",
        f"Toplam tool çağrısı: **{_common.fmt_int(total_tool_calls)}**",
        f"Tool kullanan konuşmalarda medyan **{int(stats_users['p50'])}**, "
        f"p95 **{int(stats_users['p95'])}**, max **{int(stats_users['max'])}** tool",
        f"Tool kullanan konuşmalar ortalama **{int(yes_tool[2])}** mesaj / "
        f"tool kullanmayanlar **{int(no_tool[2])}** (medyan)",
        f"Konuşma başına farklı tool (tool kullananlarda): "
        f"medyan **{int(stats_diversity['p50'])}**, max **{int(stats_diversity['max'])}**",
        f"Tool sonuç hata oranı: "
        f"**{_common.fmt_int(total_errors)}** / **{_common.fmt_int(total_results)}** "
        f"→ **%{error_rate:.1f}**",
        f"Tool çağrısı süresi: medyan **{_fmt_duration(stats_duration['p50'])}**, "
        f"p95 **{_fmt_duration(stats_duration['p95'])}**, "
        f"max **{_fmt_duration(stats_duration['max'])}**",
    ]

    # Quantile — tümü + tool-kullananlar tek tabloda
    quantile_columns = [
        {"key": "metric", "label": "metric", "align": "left"},
        {"key": "n",      "label": "n",      "align": "right"},
        {"key": "min",    "label": "min",    "align": "right"},
        {"key": "mean",   "label": "mean",   "align": "right"},
        {"key": "p50",    "label": "p50",    "align": "right"},
        {"key": "p90",    "label": "p90",    "align": "right"},
        {"key": "p95",    "label": "p95",    "align": "right"},
        {"key": "p99",    "label": "p99",    "align": "right"},
        {"key": "max",    "label": "max",    "align": "right"},
    ]
    def _qrow(label: str, s: dict) -> list:
        return [
            label,
            _common.fmt_int(s.get("n")),
            _common.fmt_int(s.get("min")),
            _common.fmt_float(s.get("mean"), 1),
            _common.fmt_int(s.get("p50")),
            _common.fmt_int(s.get("p90")),
            _common.fmt_int(s.get("p95")),
            _common.fmt_int(s.get("p99")),
            _common.fmt_int(s.get("max")),
        ]
    tool_quantile_rows = [
        _qrow("tool_calls/konuşma (tümü)", stats_all),
        _qrow("tool_calls/konuşma (tool-kullanan)", stats_users),
    ]

    bucket_table_cols = [
        {"key": "range", "label": "aralık",  "align": "left"},
        {"key": "n",     "label": "konuşma", "align": "right"},
        {"key": "pct",   "label": "oran",    "align": "right"},
    ]

    # Length compare
    length_compare_columns = [
        {"key": "kitle",     "label": "kitle",      "align": "left"},
        {"key": "n",         "label": "n konuşma",  "align": "right"},
        {"key": "msg_p50",   "label": "mesaj p50",  "align": "right"},
        {"key": "msg_p95",   "label": "mesaj p95",  "align": "right"},
        {"key": "tok_p50",   "label": "token p50",  "align": "right"},
        {"key": "tok_p95",   "label": "token p95",  "align": "right"},
    ]
    length_compare_rows = [
        ["tool yok", _common.fmt_int(int(no_tool[1])),
         _common.fmt_int(int(no_tool[2])), _common.fmt_int(int(no_tool[3])),
         _fmt_token_or_int(no_tool[4]),    _fmt_token_or_int(no_tool[5])],
        ["tool var", _common.fmt_int(int(yes_tool[1])),
         _common.fmt_int(int(yes_tool[2])), _common.fmt_int(int(yes_tool[3])),
         _fmt_token_or_int(yes_tool[4]),    _fmt_token_or_int(yes_tool[5])],
    ]

    # Diversity bucket-table
    diversity_bucket_cols = [
        {"key": "range", "label": "farklı tool", "align": "left"},
        {"key": "n",     "label": "konuşma",     "align": "right"},
        {"key": "pct",   "label": "oran",        "align": "right"},
    ]

    # Category table
    category_columns = [
        {"key": "category", "label": "kategori", "align": "left"},
        {"key": "calls",    "label": "çağrı",    "align": "right"},
        {"key": "share",    "label": "oran",     "align": "right"},
        {"key": "members",  "label": "üyeler",   "align": "left"},
    ]
    # category_rows zaten var — olduğu gibi kullan

    # Top 15 tool
    top_columns = [
        {"key": "rank",   "label": "#",               "align": "right"},
        {"key": "tool",   "label": "tool",            "align": "left"},
        {"key": "calls",  "label": "çağrı",           "align": "right"},
        {"key": "nconvo", "label": "farklı konuşma",  "align": "right"},
        {"key": "cpc",    "label": "çağrı/konuşma",   "align": "right"},
        {"key": "share",  "label": "toplamdaki oran", "align": "right"},
    ]
    top_rows = [
        [i + 1, name, _common.fmt_int(tc),
         _common.fmt_int(nc),
         f"{tc/nc:.1f}" if nc else "—",
         f"%{100*tc/max(total_tool_calls,1):.1f}"]
        for i, (name, tc, nc) in enumerate(top_tools)
    ]
    top_tool_buckets = [
        {"label": name, "count": int(tc),
         "pct": (100.0 * tc / max(total_tool_calls, 1))}
        for name, tc, _ in top_tools
    ]
    top5_monthly_data = [
        {"month": m, **{name: int(monthly_top.get(m, {}).get(name, 0)) for name in top5}}
        for m in months_sorted
    ]
    top5_monthly_series = [{"key": name, "label": name} for name in top5]

    # Kategori bucket chart
    category_buckets = [
        {"label": cat, "count": int(category_totals[cat]),
         "pct": 100.0 * category_totals[cat] / max(total_categorized, 1)}
        for cat, _ in TOOL_CATEGORIES
    ]

    # Error table
    error_columns = [
        {"key": "tool",   "label": "tool",       "align": "left"},
        {"key": "calls",  "label": "çağrı",      "align": "right"},
        {"key": "errors", "label": "hata",       "align": "right"},
        {"key": "rate",   "label": "hata oranı", "align": "right"},
    ]
    error_rows = [
        [name, _common.fmt_int(total), _common.fmt_int(errors),
         f"%{100*errors/max(total,1):.1f}"]
        for name, total, errors in err_per_tool
    ]

    duration_bucket_cols = [
        {"key": "range", "label": "süre aralığı", "align": "left"},
        {"key": "n",     "label": "çağrı",        "align": "right"},
        {"key": "pct",   "label": "oran",         "align": "right"},
    ]

    notes_items = [
        "Kategori eşleme, MCP prefix (`filesystem:` gibi) atılıp temel ada bakar; "
        "kategorize edilemeyen tool'lar `diğer`'e düşer.",
        "Tool çeşitliliği = konuşmadaki distinct `tool_calls.name` sayısı. Tool "
        "kullanmayan konuşmalar `0` kovasında.",
        "Tool süresi = `stop_timestamp − start_timestamp`. Timestamp'i eksik "
        "çağrılar hesap dışı; negatif süreli kayıtlar da atılır.",
        "Hata oranı `tool_results.is_error=true` üzerinden hesaplanır.",
    ]

    sections = [
        _common.Section(
            "Öne çıkanlar",
            highlights,
            blocks=[_common.block_bullets(highlight_items)],
        ),
        _common.Section(
            "Konuşma başına tool sayısı — quantile",
            _common.percentile_table(stats_all, "tool_calls/konuşma (tümü)")
            + "\n\n"
            + users_quantile,
            blocks=[_common.block_table(quantile_columns, tool_quantile_rows)],
        ),
        _common.Section(
            "Konuşma kovaları",
            _common.markdown_table(
                _bucket_table(counts_tool),
                headers=["tool aralığı", "konuşma", "oran"],
            ),
            blocks=[
                _common.block_bucket_chart(
                    label=f"Konuşma başına tool sayısı — kova dağılımı "
                          f"(n={_common.fmt_int(n_convo)})",
                    buckets=_buckets_payload(counts_tool),
                    image="bucket_tool_calls.png",
                    xlabel="tool çağrısı (aralık)",
                ),
                _common.block_table(
                    bucket_table_cols,
                    [[lbl, _common.fmt_int(n), f"%{pct:.1f}"] for lbl, n, pct in counts_tool],
                ),
            ],
        ),
        _common.Section(
            "Tool kullanan vs kullanmayan — konuşma uzunluğu",
            length_compare_table,
            blocks=[_common.block_table(length_compare_columns, length_compare_rows)],
        ),
        _common.Section(
            "Konuşma başına tool çeşitliliği",
            _common.percentile_table(stats_diversity,
                                     "farklı tool/konuşma (tool-kullananlar)")
            + "\n\n"
            + _common.markdown_table(
                _bucket_table(counts_diversity),
                headers=["farklı tool", "konuşma", "oran"],
            ),
            blocks=[
                _common.block_percentile_table(
                    stats_diversity, "farklı tool/konuşma (tool-kullananlar)"),
                _common.block_bucket_chart(
                    label=f"Konuşma başına farklı tool sayısı — kova dağılımı "
                          f"(n={_common.fmt_int(n_convo)})",
                    buckets=_buckets_payload(counts_diversity),
                    image="bucket_tool_diversity.png",
                    xlabel="farklı tool sayısı (distinct)",
                ),
                _common.block_table(
                    diversity_bucket_cols,
                    [[lbl, _common.fmt_int(n), f"%{pct:.1f}"] for lbl, n, pct in counts_diversity],
                ),
            ],
        ),
        _common.Section(
            "Tool kategori dağılımı",
            category_table,
            blocks=[
                _common.block_bucket_chart(
                    label="Tool kategori dağılımı",
                    buckets=category_buckets,
                    image="tool_categories.png",
                    xlabel="kategori",
                ),
                _common.block_table(category_columns, category_rows),
            ],
        ),
        _common.Section(
            "En çok kullanılan 15 tool",
            top_table,
            blocks=[
                _common.block_bucket_chart(
                    label=f"En çok kullanılan 15 tool (toplam {_common.fmt_int(total_tool_calls)} çağrı)",
                    buckets=top_tool_buckets,
                    image="top_tools.png",
                    xlabel="toplam çağrı",
                ),
                _common.block_table(top_columns, top_rows),
            ],
        ),
        _common.Section(
            "Üst-5 tool — aylık trend",
            "",
            blocks=[_common.block_line_chart(
                "Üst-5 tool — aylık trend",
                top5_monthly_data,
                top5_monthly_series,
                x_key="month",
            )],
        ),
        _common.Section(
            "Tool sonuç hata oranı",
            f"- Toplam tool_result: **{_common.fmt_int(total_results)}**\n"
            f"- Hata (`is_error=true`): **{_common.fmt_int(total_errors)}** "
            f"(**%{error_rate:.1f}**)\n\n"
            "**Hata üreten üst-15 tool:**\n\n"
            + error_table,
            blocks=[
                _common.block_bullets([
                    f"Toplam tool_result: **{_common.fmt_int(total_results)}**",
                    f"Hata (`is_error=true`): **{_common.fmt_int(total_errors)}** "
                    f"(**%{error_rate:.1f}**)",
                ]),
                _common.block_paragraph("**Hata üreten üst-15 tool:**") if err_per_tool else _common.block_paragraph("_Hata kaydı yok._"),
            ] + ([_common.block_table(error_columns, error_rows)] if err_per_tool else []),
        ),
        _common.Section(
            "Tool çağrı süresi",
            _common.percentile_table(stats_duration, "süre (saniye)")
            + "\n\n"
            + _common.markdown_table(
                _bucket_table(counts_duration),
                headers=["süre aralığı", "çağrı", "oran"],
            ),
            blocks=[
                _common.block_percentile_table(stats_duration, "süre (saniye)"),
                _common.block_bucket_chart(
                    label=f"Tool çağrı süresi — kova dağılımı (n={_common.fmt_int(int(durations.size))})",
                    buckets=_buckets_payload(counts_duration),
                    image="bucket_tool_duration.png",
                    xlabel="süre aralığı",
                ),
                _common.block_table(
                    duration_bucket_cols,
                    [[lbl, _common.fmt_int(n), f"%{pct:.1f}"] for lbl, n, pct in counts_duration],
                ),
            ],
        ),
        _common.Section(
            "Grafikler",
            "- `bucket_tool_calls.png` — konuşma başına tool sayısı, kova dağılımı\n"
            "- `bucket_tool_diversity.png` — konuşma başına farklı tool sayısı\n"
            "- `bucket_tool_duration.png` — tool çağrısı süresi kovası\n"
            "- `top_tools.png` — en popüler 15 tool\n"
            "- `tool_categories.png` — tool kategori dağılımı\n"
            "- `top5_monthly.png` — üst-5 tool'un aylık trendi",
            blocks=[],
        ),
        _common.Section(
            "Notlar",
            "- Kategori eşleme, MCP prefix (`filesystem:` gibi) atılıp temel ada bakar; "
            "kategorize edilemeyen tool'lar `diğer`'e düşer.\n"
            "- Tool çeşitliliği = konuşmadaki distinct `tool_calls.name` sayısı. Tool "
            "kullanmayan konuşmalar `0` kovasında.\n"
            "- Tool süresi = `stop_timestamp − start_timestamp`. Timestamp'i eksik "
            "çağrılar hesap dışı; negatif süreli kayıtlar da atılır.\n"
            "- Hata oranı `tool_results.is_error=true` üzerinden hesaplanır.",
            blocks=[_common.block_bullets(notes_items)],
        ),
    ]
    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)
    return summary


def _fmt_token_or_int(n) -> str:
    """Tabela hücresi için kısa sayı formatı (None/NaN güvenli)."""
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
