"""
m11 — Citation ve kaynak profili

- Fenced citation sayımı (content_blocks.raw.citations[])
- Konuşma başına citation dağılımı
- Top domain'ler (regex ile url→host)
- Citation × web tool çağrı ilişkisi
- Citation'lı vs citationsız konuşma kıyası
- Aylık trend

Kaynak: content_blocks.raw.citations (sadece type='text' blokları).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from analysis import _common
from analysis._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "11-citations"
TITLE = "Citation ve kaynak profili"

PER_CONVO_BUCKETS = [
    ("1-4",    1,   5),
    ("5-19",   5,   20),
    ("20-49",  20,  50),
    ("50-99",  50,  100),
    ("100+",   100, 10 ** 9),
]


def _bucket_counts(values: np.ndarray, buckets) -> list[int]:
    return [int(((values >= lo) & (values < hi)).sum()) for _lbl, lo, hi in buckets]


def _bucket_bar(ax, labels, counts, color, ylabel, title) -> None:
    ax.bar(labels, counts, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=0)
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(i, c, f"{c}", ha="center", va="bottom", fontsize=9)


def run(con: duckdb.DuckDBPyConnection, out_dir: Path, cfg: dict) -> dict:
    # ---------- citation item tablosu ----------
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _tmp_citations AS
        SELECT cb.conversation_uuid,
               cb.message_uuid,
               cb.start_timestamp,
               json_extract_string(c.value, '$.details.type') AS cite_type,
               json_extract_string(c.value, '$.details.url')  AS url,
               regexp_extract(json_extract_string(c.value, '$.details.url'),
                              'https?://([^/]+)', 1) AS domain
        FROM content_blocks cb,
             UNNEST(CAST(json_extract(cb.raw, '$.citations') AS JSON[])) AS c(value)
        WHERE cb.type='text'
          AND json_array_length(json_extract(cb.raw, '$.citations')) > 0
    """)

    totals = con.execute("""
        SELECT COUNT(*) AS n_cites,
               COUNT(DISTINCT message_uuid) AS n_msgs,
               COUNT(DISTINCT conversation_uuid) AS n_convos,
               COUNT(DISTINCT domain) AS n_domains,
               COUNT(DISTINCT url) AS n_urls
        FROM _tmp_citations
    """).fetchone()
    n_cites = int(totals[0] or 0)
    n_msgs = int(totals[1] or 0)
    n_convos = int(totals[2] or 0)
    n_domains = int(totals[3] or 0)
    n_urls = int(totals[4] or 0)

    # ---------- citation type dağılımı ----------
    type_rows = con.execute("""
        SELECT cite_type, COUNT(*) FROM _tmp_citations
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()

    # ---------- per-convo dağılımı ----------
    per_convo_rows = con.execute("""
        SELECT conversation_uuid, COUNT(*) AS n
        FROM _tmp_citations GROUP BY 1
    """).fetchall()
    per_convo_arr = np.array([r[1] for r in per_convo_rows], dtype=float)
    stats_per_convo = _common.percentiles(per_convo_arr)
    per_convo_counts = _bucket_counts(per_convo_arr, PER_CONVO_BUCKETS)

    # ---------- top domain ----------
    domain_rows = con.execute("""
        SELECT domain,
               COUNT(*) AS citations,
               COUNT(DISTINCT message_uuid) AS messages,
               COUNT(DISTINCT conversation_uuid) AS convos
        FROM _tmp_citations
        WHERE domain IS NOT NULL AND domain != ''
        GROUP BY 1 ORDER BY citations DESC
    """).fetchall()

    top20 = domain_rows[:20]

    # ---------- web tool çağrı sayımı per convo ----------
    web_tool_rows = con.execute("""
        SELECT conversation_uuid,
               SUM(CASE WHEN name='web_search' THEN 1 ELSE 0 END) AS web_search_calls,
               SUM(CASE WHEN name='web_fetch'  THEN 1 ELSE 0 END) AS web_fetch_calls,
               SUM(CASE WHEN name IN ('web_search','web_fetch') THEN 1 ELSE 0 END) AS web_total
        FROM tool_calls
        WHERE name IN ('web_search','web_fetch')
        GROUP BY 1
    """).fetchall()
    web_map = {r[0]: (int(r[1]), int(r[2]), int(r[3])) for r in web_tool_rows}

    # ---------- citation × web tool scatter data ----------
    per_convo_map = {r[0]: int(r[1]) for r in per_convo_rows}
    scatter_x: list[int] = []  # web tool call count
    scatter_y: list[int] = []  # citation count
    cites_without_web = 0
    for uuid, n_c in per_convo_map.items():
        web = web_map.get(uuid, (0, 0, 0))[2]
        scatter_x.append(web)
        scatter_y.append(n_c)
        if web == 0:
            cites_without_web += 1
    # Pearson r
    if len(scatter_x) >= 2:
        r_matrix = np.corrcoef(scatter_x, scatter_y)
        pearson_r = float(r_matrix[0, 1])
    else:
        pearson_r = float("nan")

    # ---------- citationlı vs citationsız konuşma kıyası ----------
    with_cites = set(per_convo_map.keys())
    compare_stats = con.execute("""
        SELECT
            (CASE WHEN sc.conversation_uuid IN ({placeholders}) THEN 'citation var' ELSE 'citation yok' END) AS grp,
            COUNT(*) AS n,
            quantile_cont(sc.message_count, 0.5) AS msg_p50,
            quantile_cont(sc.message_count, 0.95) AS msg_p95,
            quantile_cont(COALESCE(sc.tokens_human + sc.tokens_assistant, 0), 0.5) AS tok_p50,
            quantile_cont(COALESCE(sc.tokens_human + sc.tokens_assistant, 0), 0.95) AS tok_p95,
            quantile_cont(COALESCE(sc.tool_calls_total, 0), 0.5) AS tool_p50,
            quantile_cont(COALESCE(sc.tool_calls_total, 0), 0.95) AS tool_p95,
            quantile_cont(COALESCE(sc.lifetime_seconds, 0), 0.5) AS life_p50
        FROM _stats_conversation sc
        WHERE sc.message_count > 0
        GROUP BY 1
        ORDER BY 1
    """.format(placeholders=",".join([f"'{u}'" for u in with_cites]) or "''")).fetchall()

    # ---------- aylık trend ----------
    monthly_rows = con.execute("""
        SELECT strftime(start_timestamp, '%Y-%m') AS ym,
               COUNT(*) AS cites,
               COUNT(DISTINCT conversation_uuid) AS convos
        FROM _tmp_citations
        WHERE start_timestamp IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetchall()

    # ---------- grafikler ----------
    if HAS_MPL:
        # 1) per-convo bucket bar
        fig, ax = plt.subplots()
        _bucket_bar(
            ax, [b[0] for b in PER_CONVO_BUCKETS], per_convo_counts,
            "#4C72B0", "konuşma sayısı", "Konuşma başına citation kovaları",
        )
        _common.save_fig(fig, out_dir / "bucket_per_conversation.png")

        # 2) top 20 domain bar
        fig, ax = plt.subplots(figsize=(10, 8))
        if top20:
            names = [r[0] for r in top20]
            counts = [r[1] for r in top20]
            y = np.arange(len(names))
            ax.barh(y, counts, color="#55A868")
            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel("citation sayısı")
            ax.set_title("Top 20 domain (citation sayısına göre)")
            for i, c in enumerate(counts):
                ax.text(c, i, f" {c}", va="center", fontsize=9)
        _common.save_fig(fig, out_dir / "top_domains.png")

        # 3) scatter citation vs web tool
        fig, ax = plt.subplots()
        ax.scatter(scatter_x, scatter_y, alpha=0.5, color="#C44E52", s=20)
        ax.set_xlabel("web_search + web_fetch çağrı sayısı (konuşma başına)")
        ax.set_ylabel("citation sayısı (konuşma başına)")
        ax.set_title(f"Citation vs web tool çağrısı (Pearson r={pearson_r:.2f})")
        _common.save_fig(fig, out_dir / "scatter_citations_vs_tools.png")

        # 4) aylık trend
        fig, ax = plt.subplots()
        if monthly_rows:
            months = [r[0] for r in monthly_rows]
            cites = [r[1] for r in monthly_rows]
            convos_m = [r[2] for r in monthly_rows]
            x = np.arange(len(months))
            ax.bar(x - 0.2, cites, width=0.4, label="citation", color="#4C72B0")
            ax.bar(x + 0.2, convos_m, width=0.4, label="citation'lı konuşma", color="#55A868")
            ax.set_xticks(x)
            ax.set_xticklabels(months, rotation=45, ha="right")
            ax.set_ylabel("sayı")
            ax.set_title("Aylık citation ve citation'lı konuşma")
            ax.legend()
        _common.save_fig(fig, out_dir / "monthly_trend.png")

    # ---------- CSV ----------
    _common.write_csv(
        out_dir, "domains.csv",
        domain_rows,
        headers=["domain", "citations", "messages", "conversations"],
    )
    _common.write_csv(
        out_dir, "per_conversation.csv",
        per_convo_rows,
        headers=["conversation_uuid", "citation_count"],
    )

    # ---------- tablolar ----------
    # top 20 domain tablo
    top_domain_rows = []
    for r in top20:
        dom, cit, msg, con_n = r
        pct = 100.0 * cit / n_cites if n_cites else 0.0
        top_domain_rows.append([
            dom,
            _common.fmt_int(cit),
            _common.fmt_int(msg),
            _common.fmt_int(con_n),
            f"%{pct:.1f}",
        ])

    # per-convo bucket tablo
    bpc_rows = []
    for (lbl, _lo, _hi), c in zip(PER_CONVO_BUCKETS, per_convo_counts):
        pct = 100.0 * c / n_convos if n_convos else 0.0
        bpc_rows.append([lbl, _common.fmt_int(c), f"%{pct:.1f}"])

    # citation type
    ctype_rows = [[t or "(boş)", _common.fmt_int(n)] for t, n in type_rows]

    # per-convo percentile table (mean yok)
    per_convo_pct = _common.markdown_table(
        [[
            "citation/konuşma",
            _common.fmt_int(stats_per_convo["n"]),
            _common.fmt_int(int(stats_per_convo["min"] or 0)),
            _common.fmt_int(int(stats_per_convo["p50"] or 0)),
            _common.fmt_int(int(stats_per_convo["p90"] or 0)),
            _common.fmt_int(int(stats_per_convo["p95"] or 0)),
            _common.fmt_int(int(stats_per_convo["p99"] or 0)),
            _common.fmt_int(int(stats_per_convo["max"] or 0)),
        ]],
        headers=["metric", "n", "min", "p50", "p90", "p95", "p99", "max"],
    )

    # kıyas tablosu (citation var/yok)
    cmp_rows = []
    for grp, n, m50, m95, t50, t95, tc50, tc95, l50 in compare_stats:
        cmp_rows.append([
            grp,
            _common.fmt_int(n),
            _common.fmt_int(m50), _common.fmt_int(m95),
            _common.fmt_int(t50), _common.fmt_int(t95),
            _common.fmt_int(tc50), _common.fmt_int(tc95),
            _common.fmt_int(l50) + "sn",
        ])

    # aylık trend tablo (kompakt — ilk 6 ay + son 6 ay)
    monthly_table_rows = []
    for ym, cites_m, convos_m in monthly_rows:
        monthly_table_rows.append([ym, _common.fmt_int(cites_m), _common.fmt_int(convos_m)])

    # ---------- summary ----------
    top_domain_json = [
        {"domain": r[0], "citations": int(r[1]), "messages": int(r[2]), "convos": int(r[3])}
        for r in top20[:10]
    ]
    summary = {
        "n_citations": n_cites,
        "n_messages_with_citation": n_msgs,
        "n_conversations_with_citation": n_convos,
        "n_unique_domains": n_domains,
        "n_unique_urls": n_urls,
        "per_conversation_percentiles": stats_per_convo,
        "citation_types": [{"type": t, "n": int(n)} for t, n in type_rows],
        "top_domains": top_domain_json,
        "pearson_r_citations_vs_web_tools": pearson_r,
        "conversations_with_citation_no_web_tool": cites_without_web,
        "headline": (
            f"{_common.fmt_int(n_cites)} citation / "
            f"{_common.fmt_int(n_msgs)} mesaj / "
            f"{_common.fmt_int(n_convos)} konuşma; "
            f"{_common.fmt_int(n_domains)} domain; "
            f"en sık: {top20[0][0] if top20 else '—'} ({top20[0][1] if top20 else 0})."
        ),
    }
    _common.write_json(out_dir, "summary.json", summary)

    # ---------- structured data ----------
    pct_cols_no_mean = [
        {"key": "metric", "label": "metric", "align": "left"},
        {"key": "n",      "label": "n",      "align": "right"},
        {"key": "min",    "label": "min",    "align": "right"},
        {"key": "p50",    "label": "p50",    "align": "right"},
        {"key": "p90",    "label": "p90",    "align": "right"},
        {"key": "p95",    "label": "p95",    "align": "right"},
        {"key": "p99",    "label": "p99",    "align": "right"},
        {"key": "max",    "label": "max",    "align": "right"},
    ]

    bpc_buckets_data = [
        {"label": lbl, "count": int(c), "pct": (100.0 * c / n_convos) if n_convos else 0.0}
        for (lbl, _lo, _hi), c in zip(PER_CONVO_BUCKETS, per_convo_counts)
    ]
    top_domain_buckets = [
        {"label": r[0], "count": int(r[1]), "pct": (100.0 * r[1] / n_cites) if n_cites else 0.0}
        for r in top20
    ]
    monthly_cites_buckets = [
        {"label": ym, "count": int(cites_m), "pct": 0.0}
        for ym, cites_m, _cv in monthly_rows
    ]
    monthly_convos_buckets = [
        {"label": ym, "count": int(cv), "pct": 0.0}
        for ym, _c, cv in monthly_rows
    ]
    type_buckets = [
        {"label": (t or "(boş)"), "count": int(n), "pct": (100.0 * n / n_cites) if n_cites else 0.0}
        for t, n in type_rows
    ]

    # ---------- report ----------
    sections = [
        _common.Section(
            "Öne çıkanlar",
            f"- Toplam citation: **{_common.fmt_int(n_cites)}**\n"
            f"- Citation'lı mesaj: **{_common.fmt_int(n_msgs)}**, citation'lı konuşma: **{_common.fmt_int(n_convos)}**\n"
            f"- Unique domain: **{_common.fmt_int(n_domains)}**, unique URL: **{_common.fmt_int(n_urls)}**\n"
            f"- Konuşma başına citation: medyan **{int(stats_per_convo['p50'] or 0)}**, "
            f"p95 **{int(stats_per_convo['p95'] or 0)}**, "
            f"max **{int(stats_per_convo['max'] or 0)}**\n"
            f"- En çok citation: **{top20[0][0] if top20 else '—'}** "
            f"({_common.fmt_int(top20[0][1] if top20 else 0)} citation)\n"
            f"- Citation × web tool çağrı Pearson r: **{pearson_r:.2f}**\n"
            f"- Web tool çağrısı olmayan ama citation'lı konuşma: **{cites_without_web}**",
            blocks=[
                _common.block_bullets([
                    f"Toplam citation: **{_common.fmt_int(n_cites)}**",
                    f"Citation'lı mesaj: **{_common.fmt_int(n_msgs)}**, "
                    f"citation'lı konuşma: **{_common.fmt_int(n_convos)}**",
                    f"Unique domain: **{_common.fmt_int(n_domains)}**, "
                    f"unique URL: **{_common.fmt_int(n_urls)}**",
                    f"Konuşma başına citation: medyan **{int(stats_per_convo['p50'] or 0)}**, "
                    f"p95 **{int(stats_per_convo['p95'] or 0)}**, "
                    f"max **{int(stats_per_convo['max'] or 0)}**",
                    f"En çok citation: **{top20[0][0] if top20 else '—'}** "
                    f"({_common.fmt_int(top20[0][1] if top20 else 0)} citation)",
                    f"Citation × web tool çağrı Pearson r: **{pearson_r:.2f}**",
                    f"Web tool çağrısı olmayan ama citation'lı konuşma: **{cites_without_web}**",
                ]),
            ],
        ),
        _common.Section(
            "Konuşma başına citation dağılımı",
            per_convo_pct,
            blocks=[
                _common.block_table(pct_cols_no_mean, [[
                    "citation/konuşma",
                    _common.fmt_int(stats_per_convo["n"]),
                    _common.fmt_int(int(stats_per_convo["min"] or 0)),
                    _common.fmt_int(int(stats_per_convo["p50"] or 0)),
                    _common.fmt_int(int(stats_per_convo["p90"] or 0)),
                    _common.fmt_int(int(stats_per_convo["p95"] or 0)),
                    _common.fmt_int(int(stats_per_convo["p99"] or 0)),
                    _common.fmt_int(int(stats_per_convo["max"] or 0)),
                ]]),
            ],
        ),
        _common.Section(
            "Konuşma başına kova",
            _common.markdown_table(bpc_rows, headers=["citation aralığı", "konuşma", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    "Konuşma başına citation kovaları",
                    bpc_buckets_data,
                    image="bucket_per_conversation.png",
                    xlabel="citation aralığı",
                ),
                _common.block_table(
                    [
                        {"key": "bucket", "label": "citation aralığı", "align": "left"},
                        {"key": "count",  "label": "konuşma",          "align": "right"},
                        {"key": "pct",    "label": "oran",             "align": "right"},
                    ],
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"] for b in bpc_buckets_data],
                ),
            ],
        ),
        _common.Section(
            "Citation tipi dağılımı",
            _common.markdown_table(ctype_rows, headers=["cite_type", "n"]),
            blocks=[
                _common.block_table(
                    [
                        {"key": "cite_type", "label": "cite_type", "align": "left"},
                        {"key": "count",     "label": "n",         "align": "right"},
                        {"key": "pct",       "label": "oran",      "align": "right"},
                    ],
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"] for b in type_buckets],
                ),
            ] if type_buckets else [],
        ),
        _common.Section(
            "Top 20 domain",
            _common.markdown_table(
                top_domain_rows,
                headers=["domain", "citation", "mesaj", "konuşma", "oran"],
            ),
            blocks=[
                _common.block_bucket_chart(
                    "Top 20 domain (citation sayısına göre)",
                    top_domain_buckets,
                    image="top_domains.png",
                    xlabel="domain",
                ),
                _common.block_table(
                    [
                        {"key": "domain",   "label": "domain",   "align": "left"},
                        {"key": "citation", "label": "citation", "align": "right"},
                        {"key": "msg",      "label": "mesaj",    "align": "right"},
                        {"key": "convo",    "label": "konuşma",  "align": "right"},
                        {"key": "pct",      "label": "oran",     "align": "right"},
                    ],
                    top_domain_rows,
                ),
            ] if top_domain_rows else [],
        ),
        _common.Section(
            "Citation × web tool çağrı ilişkisi",
            f"Pearson korelasyon (konuşma başına citation ve web_search+web_fetch çağrısı): "
            f"**r = {pearson_r:.2f}**. "
            f"Web tool çağrısı yapılmamış ama citation'lı konuşma sayısı: "
            f"**{cites_without_web}** — scatter grafiğinde x=0 noktasındakiler.\n\n"
            f"Grafik: `scatter_citations_vs_tools.png`.",
            blocks=[
                _common.block_paragraph(
                    f"Pearson korelasyon (konuşma başına citation ve web_search+web_fetch çağrısı): "
                    f"**r = {pearson_r:.2f}**. "
                    f"Web tool çağrısı yapılmamış ama citation'lı konuşma sayısı: "
                    f"**{cites_without_web}** — scatter grafiğinde x=0 noktasındakiler."
                ),
                _common.block_scatter_chart(
                    f"Citation vs web tool çağrısı (Pearson r={pearson_r:.2f})",
                    [
                        {"web_tools": int(x), "citations": int(y), "size": max(1, int(y))}
                        for x, y in zip(scatter_x, scatter_y)
                    ],
                    x_key="web_tools",
                    y_key="citations",
                    size_key="size",
                    xlabel="web tool çağrısı",
                    ylabel="citation sayısı",
                ),
            ],
        ),
        _common.Section(
            "Citation'lı vs citationsız konuşma kıyası",
            _common.markdown_table(
                cmp_rows,
                headers=[
                    "kitle", "n",
                    "mesaj p50", "mesaj p95",
                    "token p50", "token p95",
                    "tool çağrı p50", "tool çağrı p95",
                    "ömür p50",
                ],
            ),
            blocks=[
                _common.block_table(
                    [
                        {"key": "grp",        "label": "kitle",         "align": "left"},
                        {"key": "n",          "label": "n",             "align": "right"},
                        {"key": "m50",        "label": "mesaj p50",     "align": "right"},
                        {"key": "m95",        "label": "mesaj p95",     "align": "right"},
                        {"key": "t50",        "label": "token p50",     "align": "right"},
                        {"key": "t95",        "label": "token p95",     "align": "right"},
                        {"key": "tool50",     "label": "tool çağrı p50","align": "right"},
                        {"key": "tool95",     "label": "tool çağrı p95","align": "right"},
                        {"key": "life50",     "label": "ömür p50",      "align": "right"},
                    ],
                    cmp_rows,
                ),
            ] if cmp_rows else [],
        ),
        _common.Section(
            "Aylık trend",
            _common.markdown_table(
                monthly_table_rows,
                headers=["ay", "citation", "citation'lı konuşma"],
            ),
            blocks=[
                _common.block_bucket_chart(
                    "Aylık citation",
                    monthly_cites_buckets,
                    image="monthly_trend.png",
                    xlabel="ay",
                ),
                _common.block_bucket_chart(
                    "Aylık citation'lı konuşma",
                    monthly_convos_buckets,
                    xlabel="ay",
                ),
                _common.block_table(
                    [
                        {"key": "month",  "label": "ay",                  "align": "left"},
                        {"key": "cites",  "label": "citation",            "align": "right"},
                        {"key": "convos", "label": "citation'lı konuşma", "align": "right"},
                    ],
                    monthly_table_rows,
                ),
            ] if monthly_rows else [],
        ),
        _common.Section(
            "Grafikler",
            "- `bucket_per_conversation.png` — konuşma başına citation kovaları\n"
            "- `top_domains.png` — en çok atıf yapılan 20 domain\n"
            "- `scatter_citations_vs_tools.png` — web tool çağrı vs citation scatter\n"
            "- `monthly_trend.png` — aylık citation ve citation'lı konuşma",
            blocks=[],
        ),
        _common.Section(
            "Notlar",
            "- Kaynak: `content_blocks.raw.citations[]` — yalnız `type='text'` blokları. "
            "Citation item yapısı: `{uuid, start_index, end_index, details: {type, url}}`.\n"
            "- Domain çıkarımı: `regexp_extract(url, 'https?://([^/]+)', 1)`.\n"
            "- JSON unnest: `UNNEST(CAST(json_extract(raw, '$.citations') AS JSON[]))`.\n"
            "- Örnekteki tüm citation'lar `web_search_citation` tipindedir (web_fetch de aynı tip'i üretiyor); "
            "ham `tool_calls.name` ayrımı için m05'e bakılmalı.\n"
            "- Citation sayısı ≠ referanslanan kaynak sayısı: aynı URL bir konuşmada birden çok "
            "cümlede atıf olarak geçebiliyor (3 096 citation, 1 669 unique URL).",
            blocks=[
                _common.block_bullets([
                    "Kaynak: `content_blocks.raw.citations[]` — yalnız `type='text'` blokları. "
                    "Citation item yapısı: `{uuid, start_index, end_index, details: {type, url}}`.",
                    "Domain çıkarımı: `regexp_extract(url, 'https?://([^/]+)', 1)`.",
                    "JSON unnest: `UNNEST(CAST(json_extract(raw, '$.citations') AS JSON[]))`.",
                    "Örnekteki tüm citation'lar `web_search_citation` tipindedir (web_fetch de aynı tip'i üretiyor); "
                    "ham `tool_calls.name` ayrımı için m05'e bakılmalı.",
                    "Citation sayısı ≠ referanslanan kaynak sayısı: aynı URL bir konuşmada birden çok "
                    "cümlede atıf olarak geçebiliyor.",
                ]),
            ],
        ),
    ]
    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)
    return summary
