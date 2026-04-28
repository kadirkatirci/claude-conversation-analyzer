"""
m08 — Project ve knowledge doc profili

- Toplam project (starter / private / custom-instruction oranı)
- Project başına doc sayısı kovası
- Doc content_length dağılımı (byte-char kovaları)
- prompt_template (custom instruction) uzunluğu dağılımı
- Project oluşturma aylık seri
- NOT: Conversation ↔ project bağlantısı export'ta yer almıyor.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from analysis import _common
from analysis._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "08-projects-docs"
TITLE = "Project ve knowledge doc profili"

DOCS_PER_PROJECT_BUCKETS = [
    ("0",    0,   1),
    ("1",    1,   2),
    ("2–5",  2,   6),
    ("6–15", 6,   16),
    ("16+",  16,  10 ** 9),
]

DOC_LEN_BUCKETS = [
    ("<1 K",       0,              1_000),
    ("1–10 K",     1_000,          10_000),
    ("10–100 K",   10_000,         100_000),
    ("100 K–1 M",  100_000,        1_000_000),
    (">1 M",       1_000_000,      10 ** 12),
]

PROMPT_LEN_BUCKETS = [
    ("yok",        -1,              0),   # 0 uzunluk = boş
    ("<200",       0,               200),
    ("200–1 K",    200,             1_000),
    ("1–5 K",      1_000,           5_000),
    ("5 K+",       5_000,           10 ** 9),
]


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
    tz = cfg.get("tz", _common.DEFAULT_TZ)

    # ---------- proje toplamları ----------
    totals = con.execute("""
        SELECT COUNT(*) total,
               SUM(CASE WHEN is_starter_project THEN 1 ELSE 0 END) starter,
               SUM(CASE WHEN is_private THEN 1 ELSE 0 END) private_n,
               SUM(CASE WHEN description IS NOT NULL AND description != '' THEN 1 ELSE 0 END) has_desc,
               SUM(CASE WHEN prompt_template IS NOT NULL AND prompt_template != '' THEN 1 ELSE 0 END) has_prompt
        FROM projects
    """).fetchone()
    n_projects = int(totals[0])
    n_starter = int(totals[1] or 0)
    n_private = int(totals[2] or 0)
    n_desc = int(totals[3] or 0)
    n_prompt = int(totals[4] or 0)

    # ---------- doc toplamları ----------
    doc_totals = con.execute("""
        SELECT COUNT(*) total,
               COUNT(DISTINCT project_uuid) n_projects_with_doc
        FROM docs
    """).fetchone()
    n_docs = int(doc_totals[0])
    n_projects_with_doc = int(doc_totals[1] or 0)

    # ---------- project başına doc ----------
    per_project = con.execute("""
        SELECT COALESCE(COUNT(d.uuid), 0) AS docs_n
        FROM projects p LEFT JOIN docs d ON d.project_uuid = p.uuid
        GROUP BY p.uuid
    """).fetchall()
    docs_per_project = np.array([r[0] for r in per_project], dtype=float)
    stats_dpp = _common.percentiles(docs_per_project)
    dpp_counts = _bucket_counts(docs_per_project, DOCS_PER_PROJECT_BUCKETS)

    # ---------- doc content_length ----------
    doc_lens = np.array([
        r[0] for r in con.execute("""
            SELECT content_length FROM docs
            WHERE content_length IS NOT NULL AND content_length > 0
        """).fetchall()
    ], dtype=float)
    stats_doc_len = _common.percentiles(doc_lens)
    doc_len_counts = _bucket_counts(doc_lens, DOC_LEN_BUCKETS)

    # ---------- prompt_template uzunluğu ----------
    # 91 proje: boş olanlar 0 → "yok" kovasına; dolu olanlar içerik uzunluğu
    prompt_lens = np.array([
        r[0] for r in con.execute("""
            SELECT COALESCE(LENGTH(prompt_template), 0) FROM projects
        """).fetchall()
    ], dtype=float)
    prompt_positive = prompt_lens[prompt_lens > 0]
    stats_prompt = _common.percentiles(prompt_positive)

    # "yok" kovası = 0 uzunluk; gerçek uzunluk kovaları yalnız pozitif değerler üzerinden
    prompt_counts = [int((prompt_lens == 0).sum())]
    for _lbl, lo, hi in PROMPT_LEN_BUCKETS[1:]:
        prompt_counts.append(int(((prompt_positive >= lo) & (prompt_positive < hi)).sum()))

    # ---------- aylık proje oluşturma ----------
    monthly = con.execute(f"""
        SELECT strftime(date_trunc('month', created_at AT TIME ZONE '{tz}'), '%Y-%m') AS month,
               COUNT(*) AS n
        FROM projects WHERE created_at IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetchall()

    # ---------- grafikler ----------
    if HAS_MPL:
        # 1) docs/project kova
        fig, ax = plt.subplots()
        _bucket_bar(
            ax, [b[0] for b in DOCS_PER_PROJECT_BUCKETS], dpp_counts,
            "#4C72B0", "project sayısı", "Project başına doc sayısı",
        )
        _common.save_fig(fig, out_dir / "bucket_docs_per_project.png")

        # 2) doc content_length kova
        fig, ax = plt.subplots()
        _bucket_bar(
            ax, [b[0] for b in DOC_LEN_BUCKETS], doc_len_counts,
            "#55A868", "doc sayısı", "Doc content_length kovaları (karakter)",
        )
        _common.save_fig(fig, out_dir / "bucket_doc_length.png")

        # 3) prompt_template uzunluğu kova
        fig, ax = plt.subplots()
        _bucket_bar(
            ax, [b[0] for b in PROMPT_LEN_BUCKETS], prompt_counts,
            "#8172B2", "project sayısı", "prompt_template uzunluğu (karakter)",
        )
        _common.save_fig(fig, out_dir / "bucket_prompt_template.png")

        # 4) aylık project creation
        fig, ax = plt.subplots(figsize=(12, 5))
        if monthly:
            months = [r[0] for r in monthly]
            counts = [r[1] for r in monthly]
            x = np.arange(len(months))
            ax.bar(x, counts, color="#C44E52")
            ax.set_xticks(x)
            ax.set_xticklabels(months, rotation=45, ha="right")
            ax.set_ylabel("yeni project sayısı")
            ax.set_title("Ay bazında yeni project oluşturma")
            for i, c in enumerate(counts):
                if c > 0:
                    ax.text(i, c, f"{c}", ha="center", va="bottom", fontsize=9)
        _common.save_fig(fig, out_dir / "monthly_projects.png")

    # ---------- CSV ----------
    _common.write_csv(
        out_dir, "monthly_projects.csv",
        [(m, int(c)) for m, c in monthly],
        headers=["month", "new_projects"],
    )
    _common.write_csv(
        out_dir, "docs_per_project.csv",
        [(int(v),) for v in docs_per_project],
        headers=["docs_per_project"],
    )
    _common.write_csv(
        out_dir, "doc_lengths.csv",
        [(int(v),) for v in doc_lens],
        headers=["content_length"],
    )

    # ---------- tablolar ----------
    def _pct_row(label: str, st: dict, formatter) -> list[str]:
        return [
            label,
            _common.fmt_int(st["n"]),
            formatter(st["min"]),
            formatter(st["p50"]),
            formatter(st["p90"]),
            formatter(st["p95"]),
            formatter(st["p99"]),
            formatter(st["max"]),
        ]

    headers = ["metric", "n", "min", "p50", "p90", "p95", "p99", "max"]

    int_fmt = lambda v: _common.fmt_int(int(v or 0))
    dpp_tbl = _common.markdown_table([_pct_row("docs/project", stats_dpp, int_fmt)], headers=headers)
    doc_tbl = _common.markdown_table([_pct_row("doc_length", stats_doc_len, _fmt_chars)], headers=headers)
    prompt_tbl = _common.markdown_table(
        [_pct_row("prompt_template (dolu olanlar)", stats_prompt, int_fmt)],
        headers=headers,
    )

    def _bucket_rows(bucket_defs, counts, total, pct_digits=1) -> list[list[str]]:
        out = []
        for (lbl, _lo, _hi), c in zip(bucket_defs, counts):
            p = (100.0 * c / total) if total else 0.0
            out.append([lbl, _common.fmt_int(c), f"%{p:.{pct_digits}f}"])
        return out

    dpp_rows = _bucket_rows(DOCS_PER_PROJECT_BUCKETS, dpp_counts, n_projects)
    doc_len_rows = _bucket_rows(DOC_LEN_BUCKETS, doc_len_counts, int(doc_lens.size))
    prompt_rows = _bucket_rows(PROMPT_LEN_BUCKETS, prompt_counts, n_projects)

    # ---------- summary ----------
    summary = {
        "n_projects": n_projects,
        "n_starter": n_starter,
        "n_private": n_private,
        "n_with_description": n_desc,
        "n_with_prompt_template": n_prompt,
        "n_docs": n_docs,
        "n_projects_with_doc": n_projects_with_doc,
        "docs_per_project": stats_dpp,
        "doc_content_length": stats_doc_len,
        "prompt_template_length": stats_prompt,
        "headline": (
            f"{n_projects} project, {n_docs} doc; "
            f"project başına doc medyan **{int(stats_dpp['p50'])}**, p95 **{int(stats_dpp['p95'])}**; "
            f"doc uzunluk medyan **{_fmt_chars(stats_doc_len['p50'])}**, "
            f"max **{_fmt_chars(stats_doc_len['max'])}**; "
            f"prompt_template dolu: **{n_prompt}** / **{n_projects}**."
        ),
    }
    _common.write_json(out_dir, "summary.json", summary)

    pct_starter = 100.0 * n_starter / n_projects if n_projects else 0.0
    pct_private = 100.0 * n_private / n_projects if n_projects else 0.0
    pct_desc = 100.0 * n_desc / n_projects if n_projects else 0.0
    pct_prompt = 100.0 * n_prompt / n_projects if n_projects else 0.0
    pct_doc = 100.0 * n_projects_with_doc / n_projects if n_projects else 0.0

    # ---------- structured block helpers ----------
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
    bucket_cols = lambda k: [
        {"key": "bucket", "label": k, "align": "left"},
        {"key": "count",  "label": "count",  "align": "right"},
        {"key": "pct",    "label": "pct",    "align": "right"},
    ]

    def _to_buckets(defs, counts, total):
        return [
            {"label": lbl, "count": int(c), "pct": (100.0 * c / total) if total else 0.0}
            for (lbl, _lo, _hi), c in zip(defs, counts)
        ]

    dpp_buckets_data = _to_buckets(DOCS_PER_PROJECT_BUCKETS, dpp_counts, n_projects)
    doc_len_buckets_data = _to_buckets(DOC_LEN_BUCKETS, doc_len_counts, int(doc_lens.size))
    prompt_buckets_data = _to_buckets(PROMPT_LEN_BUCKETS, prompt_counts, n_projects)

    # aylık proje — bucket_bar olarak
    monthly_buckets = [
        {"label": m, "count": int(c), "pct": 0.0}
        for m, c in monthly
    ]
    monthly_total = sum(int(c) for _m, c in monthly) if monthly else 0
    if monthly_total:
        for b in monthly_buckets:
            b["pct"] = 100.0 * b["count"] / monthly_total

    pct_starter = 100.0 * n_starter / n_projects if n_projects else 0.0
    pct_private = 100.0 * n_private / n_projects if n_projects else 0.0
    pct_desc = 100.0 * n_desc / n_projects if n_projects else 0.0
    pct_prompt = 100.0 * n_prompt / n_projects if n_projects else 0.0
    pct_doc = 100.0 * n_projects_with_doc / n_projects if n_projects else 0.0

    sections = [
        _common.Section(
            "Öne çıkanlar",
            f"- Toplam project: **{_common.fmt_int(n_projects)}** "
            f"(starter: **{n_starter}** / %{pct_starter:.1f}, "
            f"private: **{n_private}** / %{pct_private:.1f})\n"
            f"- `description` dolu: **{n_desc}** (**%{pct_desc:.1f}**)\n"
            f"- `prompt_template` (custom instruction) dolu: **{n_prompt}** (**%{pct_prompt:.1f}**)\n"
            f"- Knowledge doc'u olan project: **{n_projects_with_doc}** (**%{pct_doc:.1f}**)\n"
            f"- Toplam knowledge doc: **{_common.fmt_int(n_docs)}**; "
            f"medyan boy **{_fmt_chars(stats_doc_len['p50'])}**, "
            f"p95 **{_fmt_chars(stats_doc_len['p95'])}**, "
            f"max **{_fmt_chars(stats_doc_len['max'])}**",
            blocks=[
                _common.block_bullets([
                    f"Toplam project: **{_common.fmt_int(n_projects)}** "
                    f"(starter: **{n_starter}** / %{pct_starter:.1f}, "
                    f"private: **{n_private}** / %{pct_private:.1f})",
                    f"`description` dolu: **{n_desc}** (**%{pct_desc:.1f}**)",
                    f"`prompt_template` (custom instruction) dolu: **{n_prompt}** (**%{pct_prompt:.1f}**)",
                    f"Knowledge doc'u olan project: **{n_projects_with_doc}** (**%{pct_doc:.1f}**)",
                    f"Toplam knowledge doc: **{_common.fmt_int(n_docs)}**; "
                    f"medyan boy **{_fmt_chars(stats_doc_len['p50'])}**, "
                    f"p95 **{_fmt_chars(stats_doc_len['p95'])}**, "
                    f"max **{_fmt_chars(stats_doc_len['max'])}**",
                ]),
            ],
        ),
        _common.Section(
            "Project başına doc sayısı",
            dpp_tbl,
            blocks=[
                _common.block_table(pct_cols, [[
                    "docs/project",
                    _common.fmt_int(stats_dpp["n"]),
                    int_fmt(stats_dpp["min"]),
                    int_fmt(stats_dpp["p50"]),
                    int_fmt(stats_dpp["p90"]),
                    int_fmt(stats_dpp["p95"]),
                    int_fmt(stats_dpp["p99"]),
                    int_fmt(stats_dpp["max"]),
                ]]),
            ],
        ),
        _common.Section(
            "Doc sayısı kovaları",
            _common.markdown_table(dpp_rows, headers=["doc aralığı", "project", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    "Project başına doc sayısı",
                    dpp_buckets_data,
                    image="bucket_docs_per_project.png",
                    xlabel="doc aralığı",
                ),
                _common.block_table(
                    bucket_cols("doc aralığı"),
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"] for b in dpp_buckets_data],
                ),
            ],
        ),
        _common.Section(
            "Doc content_length (karakter)",
            doc_tbl,
            blocks=[
                _common.block_table(pct_cols, [[
                    "doc_length",
                    _common.fmt_int(stats_doc_len["n"]),
                    _fmt_chars(stats_doc_len["min"]),
                    _fmt_chars(stats_doc_len["p50"]),
                    _fmt_chars(stats_doc_len["p90"]),
                    _fmt_chars(stats_doc_len["p95"]),
                    _fmt_chars(stats_doc_len["p99"]),
                    _fmt_chars(stats_doc_len["max"]),
                ]]),
            ],
        ),
        _common.Section(
            "Doc uzunluk kovaları",
            _common.markdown_table(doc_len_rows, headers=["uzunluk aralığı", "doc", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    "Doc content_length kovaları (karakter)",
                    doc_len_buckets_data,
                    image="bucket_doc_length.png",
                    xlabel="uzunluk aralığı",
                ),
                _common.block_table(
                    bucket_cols("uzunluk aralığı"),
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"] for b in doc_len_buckets_data],
                ),
            ],
        ),
        _common.Section(
            "prompt_template uzunluğu (dolu olanlar)",
            prompt_tbl,
            blocks=[
                _common.block_table(pct_cols, [[
                    "prompt_template (dolu)",
                    _common.fmt_int(stats_prompt["n"]),
                    int_fmt(stats_prompt["min"]),
                    int_fmt(stats_prompt["p50"]),
                    int_fmt(stats_prompt["p90"]),
                    int_fmt(stats_prompt["p95"]),
                    int_fmt(stats_prompt["p99"]),
                    int_fmt(stats_prompt["max"]),
                ]]),
            ],
        ),
        _common.Section(
            "prompt_template kovaları",
            _common.markdown_table(prompt_rows, headers=["uzunluk aralığı", "project", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    "prompt_template uzunluğu (karakter)",
                    prompt_buckets_data,
                    image="bucket_prompt_template.png",
                    xlabel="uzunluk aralığı",
                ),
                _common.block_table(
                    bucket_cols("uzunluk aralığı"),
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"] for b in prompt_buckets_data],
                ),
            ],
        ),
        _common.Section(
            "Ay bazında yeni project oluşturma",
            "![](monthly_projects.png)",
            blocks=[
                _common.block_bucket_chart(
                    "Ay bazında yeni project oluşturma",
                    monthly_buckets,
                    image="monthly_projects.png",
                    xlabel="ay",
                )
            ] if monthly_buckets else [],
        ),
        _common.Section(
            "Grafikler",
            "- `bucket_docs_per_project.png` — project başına doc sayısı kovaları\n"
            "- `bucket_doc_length.png` — doc content_length kovaları\n"
            "- `bucket_prompt_template.png` — prompt_template uzunluğu kovaları\n"
            "- `monthly_projects.png` — ay bazında yeni project oluşturma",
            blocks=[],
        ),
        _common.Section(
            "Notlar",
            "- **Conversation ↔ project bağlantısı export'ta yer almıyor**: "
            "`conversations.json` içinde `project_uuid` ya da benzeri alan yok. "
            "Bu yüzden \"project'li vs project'siz konuşma\" türü kıyaslar yapılamaz.\n"
            "- `docs` = project knowledge dokümanları (`projects.json` altında, içerik DB'de saklı).\n"
            "- `prompt_template` = projenin custom instruction'ı (sistem prompt'u).\n"
            "- Uzunluklar karakter cinsindendir (LENGTH SQL fonksiyonu); "
            "tokenize edilmez, byte da değildir.\n"
            "- `is_starter_project` Anthropic'in hazır şablon projelerini işaretler.",
            blocks=[
                _common.block_bullets([
                    "**Conversation ↔ project bağlantısı export'ta yer almıyor**: "
                    "`conversations.json` içinde `project_uuid` ya da benzeri alan yok. "
                    "Bu yüzden \"project'li vs project'siz konuşma\" türü kıyaslar yapılamaz.",
                    "`docs` = project knowledge dokümanları (`projects.json` altında, içerik DB'de saklı).",
                    "`prompt_template` = projenin custom instruction'ı (sistem prompt'u).",
                    "Uzunluklar karakter cinsindendir (`LENGTH` SQL fonksiyonu); "
                    "tokenize edilmez, byte da değildir.",
                    "`is_starter_project` Anthropic'in hazır şablon projelerini işaretler.",
                ]),
            ],
        ),
    ]
    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)
    return summary
