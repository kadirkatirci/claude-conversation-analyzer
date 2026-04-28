"""
m06 — Konuşma ömrü ve ritmi

- Konuşma ömrü (lifetime): last_msg_at - first_msg_at, message_count >= 2
- Mesaj-düzeyi ardışık gap: LAG(created_at) ile gerçek aralık dağılımı
- İlk-yanıt gecikmesi: human → assistant ardışık geçişlerin süresi
- Çok-oturumlu konuşma: içinde en az bir gap > 1sa olan konuşma
- Ömür × mesaj-sayısı çapraz tablosu
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from analysis import _common
from analysis._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "06-conversation-lifetime"
TITLE = "Konuşma ömrü ve ritmi"

LIFETIME_BUCKETS = [
    ("0sn (aynı timestamp)", 0, 1),
    ("<1dk",                 1, 60),
    ("1–10dk",               60, 10 * 60),
    ("10dk–1sa",             10 * 60, 60 * 60),
    ("1–24sa",               60 * 60, 24 * 60 * 60),
    ("1–7gün",               24 * 60 * 60, 7 * 24 * 60 * 60),
    (">7gün",                7 * 24 * 60 * 60, 10 ** 12),
]

GAP_BUCKETS = [
    ("<10sn",     0,             10),
    ("10–60sn",   10,            60),
    ("1–10dk",    60,            10 * 60),
    ("10dk–1sa",  10 * 60,       60 * 60),
    ("1–24sa",    60 * 60,       24 * 60 * 60),
    (">1gün",     24 * 60 * 60,  10 ** 12),
]

RESP_BUCKETS = GAP_BUCKETS  # aynı kova seti ilk-yanıt için de mantıklı

MSG_BUCKETS = [
    ("2–5",    2,   6),
    ("6–15",   6,   16),
    ("16–50",  16,  51),
    ("51+",    51,  10 ** 9),
]


def _fmt_duration(seconds: float) -> str:
    if seconds is None or seconds != seconds:
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s}sn"
    if s < 3600:
        return f"{s // 60}dk {s % 60}sn"
    if s < 86400:
        h, rem = divmod(s, 3600)
        return f"{h}sa {rem // 60}dk"
    d, rem = divmod(s, 86400)
    h = rem // 3600
    return f"{d}gn {h}sa"


def _bucket_counts(values: np.ndarray, buckets: list[tuple[str, float, float]]) -> list[int]:
    return [int(((values >= lo) & (values < hi)).sum()) for _lbl, lo, hi in buckets]


def _bucket_bar(ax, labels: list[str], counts: list[int], color: str, ylabel: str, title: str) -> None:
    ax.bar(labels, counts, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(i, c, f"{c}", ha="center", va="bottom", fontsize=9)


def run(con: duckdb.DuckDBPyConnection, out_dir: Path, cfg: dict) -> dict:
    # -------- lifetime + message count --------
    rows = con.execute("""
        SELECT message_count, lifetime_seconds
        FROM _stats_conversation
        WHERE message_count >= 2
    """).fetchall()
    data = np.array(rows, dtype=float)
    n = int(data.shape[0])

    msg_count = data[:, 0] if n else np.array([])
    lifetime = data[:, 1] if n else np.array([])

    zero_lifetime = int((lifetime == 0).sum()) if n else 0
    positive_lifetime = lifetime[lifetime > 0]
    stats_lifetime = _common.percentiles(positive_lifetime)

    life_counts = _bucket_counts(lifetime, LIFETIME_BUCKETS)
    life_rows = []
    for (label, _lo, _hi), cnt in zip(LIFETIME_BUCKETS, life_counts):
        pct = (100.0 * cnt / n) if n else 0.0
        life_rows.append([label, _common.fmt_int(cnt), f"%{pct:.1f}"])

    # -------- mesaj-düzeyi ardışık gap --------
    gap_rows = con.execute("""
        WITH ordered AS (
            SELECT conversation_uuid, created_at,
                LAG(created_at) OVER (PARTITION BY conversation_uuid ORDER BY created_at) AS prev_at
            FROM messages
            WHERE sender IN ('human','assistant')
        )
        SELECT epoch(created_at - prev_at) AS gap_sec
        FROM ordered
        WHERE prev_at IS NOT NULL
    """).fetchall()
    gaps = np.array([r[0] for r in gap_rows if r[0] is not None and r[0] >= 0], dtype=float)
    gap_positive = gaps[gaps > 0]
    stats_gap = _common.percentiles(gap_positive)
    # Kovalar ve öne-çıkanlar stat'ıyla tutarlı olsun diye pozitif gap'ler üzerinden sayılır
    # (0sn gap'ler "ardışık ama aynı timestamp" artefaktıdır).
    gap_bucket_counts = _bucket_counts(gap_positive, GAP_BUCKETS)
    gap_total = int(gap_positive.size)
    gap_bucket_rows = []
    for (label, _lo, _hi), cnt in zip(GAP_BUCKETS, gap_bucket_counts):
        pct = (100.0 * cnt / gap_total) if gap_total else 0.0
        gap_bucket_rows.append([label, _common.fmt_int(cnt), f"%{pct:.1f}"])

    # -------- ilk-yanıt gecikmesi (human → assistant) --------
    resp_rows = con.execute("""
        WITH ordered AS (
            SELECT conversation_uuid, sender, created_at,
                LAG(created_at) OVER (PARTITION BY conversation_uuid ORDER BY created_at) AS prev_at,
                LAG(sender)     OVER (PARTITION BY conversation_uuid ORDER BY created_at) AS prev_sender
            FROM messages
            WHERE sender IN ('human','assistant')
        )
        SELECT epoch(created_at - prev_at) AS gap_sec
        FROM ordered
        WHERE prev_sender = 'human' AND sender = 'assistant' AND prev_at IS NOT NULL
    """).fetchall()
    resp = np.array([r[0] for r in resp_rows if r[0] is not None and r[0] >= 0], dtype=float)
    resp_positive = resp[resp > 0]
    stats_resp = _common.percentiles(resp_positive)
    resp_bucket_counts = _bucket_counts(resp_positive, RESP_BUCKETS)
    resp_total = int(resp_positive.size)
    resp_bucket_rows = []
    for (label, _lo, _hi), cnt in zip(RESP_BUCKETS, resp_bucket_counts):
        pct = (100.0 * cnt / resp_total) if resp_total else 0.0
        resp_bucket_rows.append([label, _common.fmt_int(cnt), f"%{pct:.1f}"])

    # -------- çok-oturumlu konuşma (gap > 1sa) --------
    multi_rows = con.execute("""
        WITH ordered AS (
            SELECT conversation_uuid,
                epoch(created_at - LAG(created_at) OVER (PARTITION BY conversation_uuid ORDER BY created_at)) AS gap
            FROM messages
            WHERE sender IN ('human','assistant')
        )
        SELECT COUNT(DISTINCT conversation_uuid)
        FROM ordered
        WHERE gap > 3600
    """).fetchall()
    n_multi_session = int(multi_rows[0][0]) if multi_rows else 0
    pct_multi = (100.0 * n_multi_session / n) if n else 0.0

    # -------- ömür × mesaj-sayısı çapraz tablosu --------
    # satırlar = ömür kovaları, sütunlar = mesaj kovaları
    cross_counts = np.zeros((len(LIFETIME_BUCKETS), len(MSG_BUCKETS)), dtype=int)
    for i, (_lbl, lo, hi) in enumerate(LIFETIME_BUCKETS):
        life_mask = (lifetime >= lo) & (lifetime < hi)
        mc_subset = msg_count[life_mask]
        for j, (_mlbl, mlo, mhi) in enumerate(MSG_BUCKETS):
            cross_counts[i, j] = int(((mc_subset >= mlo) & (mc_subset < mhi)).sum())

    cross_rows = []
    for i, (lbl, _lo, _hi) in enumerate(LIFETIME_BUCKETS):
        row = [lbl] + [_common.fmt_int(int(cross_counts[i, j])) for j in range(len(MSG_BUCKETS))]
        row.append(_common.fmt_int(int(cross_counts[i, :].sum())))
        cross_rows.append(row)
    cross_headers = ["ömür \\ mesaj"] + [m[0] for m in MSG_BUCKETS] + ["toplam"]

    # -------- grafikler --------
    if HAS_MPL:
        # 1) lifetime bucket bar
        fig, ax = plt.subplots()
        _bucket_bar(
            ax,
            [b[0] for b in LIFETIME_BUCKETS],
            life_counts,
            "#55A868",
            "konuşma sayısı",
            "Konuşma ömrü kovaları",
        )
        _common.save_fig(fig, out_dir / "bucket_lifetime.png")

        # 2) ardışık gap bucket bar
        fig, ax = plt.subplots()
        _bucket_bar(
            ax,
            [b[0] for b in GAP_BUCKETS],
            gap_bucket_counts,
            "#4C72B0",
            "ardışık mesaj çifti sayısı",
            "Mesaj-arası gerçek gap dağılımı",
        )
        _common.save_fig(fig, out_dir / "bucket_message_gap.png")

        # 3) ilk-yanıt gecikmesi bucket bar
        fig, ax = plt.subplots()
        _bucket_bar(
            ax,
            [b[0] for b in RESP_BUCKETS],
            resp_bucket_counts,
            "#C44E52",
            "human→assistant geçiş sayısı",
            "İlk-yanıt gecikmesi (human → assistant)",
        )
        _common.save_fig(fig, out_dir / "bucket_first_response.png")

        # 4) ömür × mesaj-sayısı heatmap
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(cross_counts, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(MSG_BUCKETS)))
        ax.set_xticklabels([m[0] for m in MSG_BUCKETS])
        ax.set_yticks(range(len(LIFETIME_BUCKETS)))
        ax.set_yticklabels([b[0] for b in LIFETIME_BUCKETS])
        ax.set_xlabel("mesaj sayısı")
        ax.set_ylabel("ömür")
        ax.set_title("Ömür × mesaj sayısı çapraz tablosu")
        for i in range(cross_counts.shape[0]):
            for j in range(cross_counts.shape[1]):
                v = int(cross_counts[i, j])
                if v > 0:
                    ax.text(j, i, f"{v}", ha="center", va="center",
                            color="white" if v > cross_counts.max() * 0.5 else "black", fontsize=8)
        fig.colorbar(im, ax=ax)
        _common.save_fig(fig, out_dir / "cross_lifetime_msgcount.png")

    # -------- CSV --------
    _common.write_csv(
        out_dir, "data.csv",
        [(int(m), float(l)) for m, l in zip(msg_count, lifetime)],
        headers=["message_count", "lifetime_seconds"],
    )
    _common.write_csv(
        out_dir, "message_gaps.csv",
        [(float(g),) for g in gaps],
        headers=["gap_seconds"],
    )
    _common.write_csv(
        out_dir, "first_response_gaps.csv",
        [(float(g),) for g in resp],
        headers=["response_gap_seconds"],
    )

    # -------- summary --------
    summary = {
        "n_conversations": n,
        "zero_lifetime": zero_lifetime,
        "n_multi_session_conversations": n_multi_session,
        "pct_multi_session": round(pct_multi, 2),
        "lifetime_seconds": stats_lifetime,
        "lifetime_human": {
            k: _fmt_duration(v) for k, v in stats_lifetime.items()
            if k in ("p50", "p90", "p95", "p99", "max", "mean")
        },
        "message_gap_seconds": stats_gap,
        "message_gap_human": {
            k: _fmt_duration(v) for k, v in stats_gap.items()
            if k in ("p50", "p90", "p95", "p99", "max", "mean")
        },
        "first_response_seconds": stats_resp,
        "first_response_human": {
            k: _fmt_duration(v) for k, v in stats_resp.items()
            if k in ("p50", "p90", "p95", "p99", "max", "mean")
        },
        "lifetime_buckets": {b[0]: int(c) for b, c in zip(LIFETIME_BUCKETS, life_counts)},
        "headline": (
            f"{n} konuşma (≥2 mesaj); medyan ömür **{_fmt_duration(stats_lifetime['p50'])}**, "
            f"mesaj-arası medyan **{_fmt_duration(stats_gap['p50'])}**, "
            f"ilk-yanıt medyan **{_fmt_duration(stats_resp['p50'])}**. "
            f"Çok-oturumlu konuşma: %{pct_multi:.1f}."
        ),
    }
    _common.write_json(out_dir, "summary.json", summary)

    # -------- tablolar --------
    def _dur_row(label: str, st: dict) -> list[str]:
        return [
            label,
            _common.fmt_int(st["n"]),
            _fmt_duration(st["min"]),
            _fmt_duration(st["p50"]),
            _fmt_duration(st["p90"]),
            _fmt_duration(st["p95"]),
            _fmt_duration(st["p99"]),
            _fmt_duration(st["max"]),
        ]

    headers = ["metric", "n", "min", "p50", "p90", "p95", "p99", "max"]
    life_tbl = _common.markdown_table([_dur_row("lifetime", stats_lifetime)], headers=headers)
    gap_tbl  = _common.markdown_table([_dur_row("gap",      stats_gap)], headers=headers)
    resp_tbl = _common.markdown_table([_dur_row("ilk-yanıt", stats_resp)], headers=headers)

    # ── Structured blocks ──
    highlight_items = [
        f"Kapsam: **{_common.fmt_int(n)}** konuşma (≥2 mesajlı)",
        f"0 saniyelik ömür (aynı timestamp): **{zero_lifetime}**",
        f"Medyan ömür: **{_fmt_duration(stats_lifetime['p50'])}**, "
        f"p95 **{_fmt_duration(stats_lifetime['p95'])}**, "
        f"max **{_fmt_duration(stats_lifetime['max'])}**",
        f"Mesaj-arası gerçek gap (tüm ardışık çiftler, n=**{_common.fmt_int(gap_total)}**): "
        f"medyan **{_fmt_duration(stats_gap['p50'])}**, p95 **{_fmt_duration(stats_gap['p95'])}**",
        f"İlk-yanıt gecikmesi — human→assistant geçişleri, n=**{_common.fmt_int(resp_total)}**: "
        f"medyan **{_fmt_duration(stats_resp['p50'])}**, p95 **{_fmt_duration(stats_resp['p95'])}**",
        f"Çok-oturumlu konuşma (içinde >1sa gap olan): "
        f"**{_common.fmt_int(n_multi_session)}** (**%{pct_multi:.1f}**)",
    ]

    duration_columns = [
        {"key": "metric", "label": "metric", "align": "left"},
        {"key": "n",      "label": "n",      "align": "right"},
        {"key": "min",    "label": "min",    "align": "right"},
        {"key": "p50",    "label": "p50",    "align": "right"},
        {"key": "p90",    "label": "p90",    "align": "right"},
        {"key": "p95",    "label": "p95",    "align": "right"},
        {"key": "p99",    "label": "p99",    "align": "right"},
        {"key": "max",    "label": "max",    "align": "right"},
    ]
    def _dur_row_fmt(label: str, st: dict) -> list[str]:
        return [
            label,
            _common.fmt_int(st["n"]),
            _fmt_duration(st["min"]),
            _fmt_duration(st["p50"]),
            _fmt_duration(st["p90"]),
            _fmt_duration(st["p95"]),
            _fmt_duration(st["p99"]),
            _fmt_duration(st["max"]),
        ]

    def _bucket_chart_payload(buckets_def, counts, total):
        return [
            {"label": lbl,
             "count": int(c),
             "pct": (100.0 * c / total) if total else 0.0}
            for (lbl, _lo, _hi), c in zip(buckets_def, counts)
        ]

    lifetime_bucket_cols = [
        {"key": "range", "label": "ömür aralığı", "align": "left"},
        {"key": "n",     "label": "konuşma",       "align": "right"},
        {"key": "pct",   "label": "oran",          "align": "right"},
    ]
    gap_bucket_cols = [
        {"key": "range", "label": "gap aralığı",   "align": "left"},
        {"key": "n",     "label": "çift sayısı",   "align": "right"},
        {"key": "pct",   "label": "oran",          "align": "right"},
    ]
    resp_bucket_cols = [
        {"key": "range", "label": "gecikme aralığı", "align": "left"},
        {"key": "n",     "label": "geçiş sayısı",    "align": "right"},
        {"key": "pct",   "label": "oran",            "align": "right"},
    ]

    # Cross table
    cross_columns = [{"key": "life", "label": "ömür \\ mesaj", "align": "left"}] + [
        {"key": f"msg_{i}", "label": m[0], "align": "right"}
        for i, m in enumerate(MSG_BUCKETS)
    ] + [{"key": "total", "label": "toplam", "align": "right"}]

    notes_items = [
        "`lifetime = last_msg_at - first_msg_at`; tek mesajlı konuşmalar dışlanır.",
        "Mesaj-arası gap `LAG(created_at)` pencere fonksiyonuyla ardışık "
        "`sender IN ('human','assistant')` mesaj çiftleri üzerinden hesaplanır; "
        "tool-only mesajlar sıralamaya girmez.",
        "İlk-yanıt gecikmesi yalnızca `prev_sender='human' AND sender='assistant'` "
        "geçişlerinde ölçülür.",
        "Çok-oturumlu konuşma = içinde en az bir ardışık gap > 1 saat olan konuşma.",
        "Çarpık dağılımlar için ortalama (mean) yanıltıcı; tablolarda yalnız quantile verilir.",
    ]

    sections = [
        _common.Section(
            "Öne çıkanlar",
            f"- Kapsam: **{_common.fmt_int(n)}** konuşma (≥2 mesajlı)\n"
            f"- 0 saniyelik ömür (aynı timestamp): **{zero_lifetime}**\n"
            f"- Medyan ömür: **{_fmt_duration(stats_lifetime['p50'])}**, "
            f"p95 **{_fmt_duration(stats_lifetime['p95'])}**, "
            f"max **{_fmt_duration(stats_lifetime['max'])}**\n"
            f"- Mesaj-arası gerçek gap (tüm ardışık çiftler, n=**{_common.fmt_int(gap_total)}**): "
            f"medyan **{_fmt_duration(stats_gap['p50'])}**, p95 **{_fmt_duration(stats_gap['p95'])}**\n"
            f"- İlk-yanıt gecikmesi — human→assistant geçişleri, n=**{_common.fmt_int(resp_total)}**: "
            f"medyan **{_fmt_duration(stats_resp['p50'])}**, p95 **{_fmt_duration(stats_resp['p95'])}**\n"
            f"- Çok-oturumlu konuşma (içinde >1sa gap olan): "
            f"**{_common.fmt_int(n_multi_session)}** (**%{pct_multi:.1f}**)",
            blocks=[_common.block_bullets(highlight_items)],
        ),
        _common.Section(
            "Ömür dağılımı (konuşma başına)",
            life_tbl,
            blocks=[_common.block_table(duration_columns,
                                        [_dur_row_fmt("lifetime", stats_lifetime)])],
        ),
        _common.Section(
            "Ömür kovaları",
            _common.markdown_table(life_rows, headers=["ömür aralığı", "konuşma", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    label="Konuşma ömrü kovaları",
                    buckets=_bucket_chart_payload(LIFETIME_BUCKETS, life_counts, n),
                    image="bucket_lifetime.png",
                    xlabel="ömür aralığı",
                ),
                _common.block_table(lifetime_bucket_cols, life_rows),
            ],
        ),
        _common.Section(
            "Mesaj-arası gerçek gap",
            f"Ardışık human/assistant mesaj çiftleri arası saniye.\n\n" + gap_tbl,
            blocks=[
                _common.block_paragraph("Ardışık human/assistant mesaj çiftleri arası saniye."),
                _common.block_table(duration_columns, [_dur_row_fmt("gap", stats_gap)]),
            ],
        ),
        _common.Section(
            "Gap kovaları",
            _common.markdown_table(gap_bucket_rows, headers=["gap aralığı", "çift sayısı", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    label="Mesaj-arası gerçek gap dağılımı",
                    buckets=_bucket_chart_payload(GAP_BUCKETS, gap_bucket_counts, gap_total),
                    image="bucket_message_gap.png",
                    xlabel="gap aralığı",
                ),
                _common.block_table(gap_bucket_cols, gap_bucket_rows),
            ],
        ),
        _common.Section(
            "İlk-yanıt gecikmesi (human → assistant)",
            f"Human mesajından hemen sonraki assistant mesajına kadar geçen süre.\n\n" + resp_tbl,
            blocks=[
                _common.block_paragraph("Human mesajından hemen sonraki assistant mesajına kadar geçen süre."),
                _common.block_table(duration_columns, [_dur_row_fmt("ilk-yanıt", stats_resp)]),
            ],
        ),
        _common.Section(
            "İlk-yanıt kovaları",
            _common.markdown_table(resp_bucket_rows, headers=["gecikme aralığı", "geçiş sayısı", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    label="İlk-yanıt gecikmesi dağılımı",
                    buckets=_bucket_chart_payload(RESP_BUCKETS, resp_bucket_counts, resp_total),
                    image="bucket_first_response.png",
                    xlabel="gecikme aralığı",
                ),
                _common.block_table(resp_bucket_cols, resp_bucket_rows),
            ],
        ),
        _common.Section(
            "Ömür × mesaj sayısı",
            _common.markdown_table(cross_rows, headers=cross_headers),
            blocks=[
                _common.block_table(cross_columns, cross_rows),
                _common.block_heatmap_chart(
                    "Ömür × mesaj sayısı heatmap",
                    [
                        {"x": MSG_BUCKETS[j][0], "y": LIFETIME_BUCKETS[i][0], "value": int(cross_counts[i, j])}
                        for i in range(len(LIFETIME_BUCKETS))
                        for j in range(len(MSG_BUCKETS))
                    ],
                    x_labels=[m[0] for m in MSG_BUCKETS],
                    y_labels=[b[0] for b in LIFETIME_BUCKETS],
                ),
            ],
        ),
        _common.Section(
            "Grafikler",
            "- `bucket_lifetime.png` — konuşma ömrü kovaları\n"
            "- `bucket_message_gap.png` — ardışık mesaj arası gerçek gap dağılımı\n"
            "- `bucket_first_response.png` — human→assistant ilk-yanıt gecikmesi\n"
            "- `cross_lifetime_msgcount.png` — ömür × mesaj sayısı çapraz tablosu",
            blocks=[],
        ),
        _common.Section(
            "Notlar",
            "- `lifetime = last_msg_at - first_msg_at`; tek mesajlı konuşmalar dışlanır.\n"
            "- Mesaj-arası gap `LAG(created_at)` pencere fonksiyonuyla ardışık "
            "`sender IN ('human','assistant')` mesaj çiftleri üzerinden hesaplanır; "
            "tool-only mesajlar sıralamaya girmez.\n"
            "- İlk-yanıt gecikmesi yalnızca `prev_sender='human' AND sender='assistant'` "
            "geçişlerinde ölçülür.\n"
            "- Çok-oturumlu konuşma = içinde en az bir ardışık gap > 1 saat olan konuşma.\n"
            "- Çarpık dağılımlar için ortalama (mean) yanıltıcı; tablolarda yalnız quantile verilir.",
            blocks=[_common.block_bullets(notes_items)],
        ),
    ]
    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)
    return summary
