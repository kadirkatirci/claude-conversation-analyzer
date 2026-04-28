"""
m01 — Conversation length distribution

Metrics:
  - message_count (total messages per conversation)
  - tokens_total  (tokens_human + tokens_assistant)  [cl100k_base approximation]

Quantile table + bucket bar for each metric.
Empty conversations (message_count=0) are reported separately but excluded from distributions.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from analysis import _common
from analysis._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "01-conversation-length"
TITLE = "Konuşma uzunluğu dağılımı"


MSG_BUCKETS: list[tuple[str, float, float]] = [
    ("1",       1,   2),
    ("2–5",     2,   6),
    ("6–15",    6,   16),
    ("16–50",   16,  51),
    ("51–100",  51,  101),
    ("101+",    101, 1e9),
]

TURN_BUCKETS: list[tuple[str, float, float]] = [
    ("1",      1,   2),
    ("2–5",    2,   6),
    ("6–15",   6,   16),
    ("16–50",  16,  51),
    ("51+",    51,  1e9),
]

TOKEN_BUCKETS: list[tuple[str, float, float]] = [
    ("<500",       0,      500),
    ("500–2K",     500,    2_000),
    ("2K–10K",     2_000,  10_000),
    ("10K–50K",    10_000, 50_000),
    ("50K+",       50_000, 1e12),
]


def _load(con: duckdb.DuckDBPyConnection) -> dict[str, np.ndarray]:
    rows = con.execute("""
        SELECT
          message_count,
          human_turn_count,
          (tokens_human + tokens_assistant) AS tokens_total
        FROM _stats_conversation
    """).fetchall()
    arr = np.array(rows, dtype=float)
    return {
        "message_count": arr[:, 0],
        "human_turn_count": arr[:, 1],
        "tokens_total": arr[:, 2],
    }


def _bucket_counts(values: np.ndarray, buckets: list[tuple[str, float, float]]) -> list[tuple[str, int, float]]:
    total = values.size
    out = []
    for label, lo, hi in buckets:
        mask = (values >= lo) & (values < hi)
        n = int(mask.sum())
        pct = (100.0 * n / total) if total else 0.0
        out.append((label, n, pct))
    return out


def _bucket_bar(
    counts: list[tuple[str, int, float]],
    title: str,
    xlabel: str,
    path: Path,
    color: str = "#4C72B0",
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
    ax.set_ylabel("konuşma sayısı")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    _common.save_fig(fig, path)


def _bucket_table(counts: list[tuple[str, int, float]]) -> list[list]:
    return [[label, _common.fmt_int(n), f"%{pct:.1f}"] for label, n, pct in counts]


def _fmt_token(n: float) -> str:
    """1502 → "1.5K", 188266 → "188K"."""
    if n is None or n != n:
        return "—"
    if n < 1_000:
        return f"{int(round(n))}"
    if n < 10_000:
        return f"{n/1000:.1f}K"
    if n < 1_000_000:
        return f"{int(round(n/1000))}K"
    return f"{n/1_000_000:.1f}M"


def run(con: duckdb.DuckDBPyConnection, out_dir: Path, cfg: dict) -> dict:
    data = _load(con)

    n_total = int(data["message_count"].size)
    n_empty = int((data["message_count"] == 0).sum())
    n_nonempty = n_total - n_empty

    mc = data["message_count"][data["message_count"] > 0]
    tk = data["tokens_total"][data["message_count"] > 0]
    tn = data["human_turn_count"][data["message_count"] > 0]
    tn = tn[tn > 0]  # human turn içermeyen konuşma yoksa no-op

    sum_mc = _common.percentiles(mc)
    sum_tk = _common.percentiles(tk)
    sum_tn = _common.percentiles(tn)

    mc_counts = _bucket_counts(mc, MSG_BUCKETS)
    tk_counts = _bucket_counts(tk, TOKEN_BUCKETS)
    tn_counts = _bucket_counts(tn, TURN_BUCKETS)

    # Grafikler — başlıklar nötr; yorum rapor metninde
    _bucket_bar(
        mc_counts,
        title=f"Konuşma başına mesaj sayısı — kova dağılımı (n={_common.fmt_int(n_nonempty)})",
        xlabel="mesaj sayısı (aralık)",
        path=out_dir / "bucket_messages.png",
        color="#4C72B0",
    )
    _bucket_bar(
        tn_counts,
        title=f"Konuşma başına human turn sayısı — kova dağılımı (n={_common.fmt_int(int(tn.size))})",
        xlabel="human turn sayısı (aralık)",
        path=out_dir / "bucket_turns.png",
        color="#DD8452",
    )
    _bucket_bar(
        tk_counts,
        title=f"Konuşma başına toplam token — kova dağılımı (n={_common.fmt_int(n_nonempty)})",
        xlabel="toplam token (aralık)",
        path=out_dir / "bucket_tokens.png",
        color="#55A868",
    )

    # CSV
    _common.write_csv(
        out_dir, "data.csv",
        list(zip(
            [int(x) for x in data["message_count"]],
            [int(x) for x in data["human_turn_count"]],
            [int(x) for x in data["tokens_total"]],
        )),
        headers=["message_count", "human_turn_count", "tokens_total"],
    )

    summary = {
        "n_conversations": n_total,
        "n_empty": n_empty,
        "n_nonempty": n_nonempty,
        "message_count": sum_mc,
        "human_turn_count": sum_tn,
        "tokens_total": sum_tk,
        "buckets_messages": {lbl: n for lbl, n, _ in mc_counts},
        "buckets_turns": {lbl: n for lbl, n, _ in tn_counts},
        "buckets_tokens": {lbl: n for lbl, n, _ in tk_counts},
        "headline": (
            f"{n_nonempty} konuşma (boş: {n_empty}); medyan {int(sum_mc['p50'])} mesaj, "
            f"{int(sum_tn['p50'])} human turn, p95 {int(sum_mc['p95'])} mesaj, "
            f"max {int(sum_mc['max'])}. Medyan ~{_fmt_token(sum_tk['p50'])} token."
        ),
    }
    _common.write_json(out_dir, "summary.json", summary)

    # Öne çıkanlar — sadece sayı interpolation; semantik iskele yok
    highlights = (
        f"- Toplam: **{_common.fmt_int(n_nonempty)}** konuşma (boş: {n_empty})\n"
        f"- Medyan: **{int(sum_mc['p50'])} mesaj**, "
        f"**{int(sum_tn['p50'])} human turn**, "
        f"**~{_fmt_token(sum_tk['p50'])} token**\n"
        f"- p95: **{int(sum_mc['p95'])} mesaj**, "
        f"**{int(sum_tn['p95'])} human turn**, "
        f"**~{_fmt_token(sum_tk['p95'])} token**\n"
        f"- Max: **{int(sum_mc['max'])} mesaj**, "
        f"**{int(sum_tn['max'])} human turn**, "
        f"**~{_fmt_token(sum_tk['max'])} token**"
    )

    # Highlight maddeleri — structured bullet listesi için ayrı tutuyoruz,
    # markdown'a yine "\n".join ile döküyoruz (geriye uyum + report.md).
    highlight_items = [
        f"Toplam: **{_common.fmt_int(n_nonempty)}** konuşma (boş: {n_empty})",
        (
            f"Medyan: **{int(sum_mc['p50'])} mesaj**, "
            f"**{int(sum_tn['p50'])} human turn**, "
            f"**~{_fmt_token(sum_tk['p50'])} token**"
        ),
        (
            f"p95: **{int(sum_mc['p95'])} mesaj**, "
            f"**{int(sum_tn['p95'])} human turn**, "
            f"**~{_fmt_token(sum_tk['p95'])} token**"
        ),
        (
            f"Max: **{int(sum_mc['max'])} mesaj**, "
            f"**{int(sum_tn['max'])} human turn**, "
            f"**~{_fmt_token(sum_tk['max'])} token**"
        ),
    ]

    # Bucket block helpers — each bucket row is {label, count, pct}.
    def _bucket_payload(counts):
        return [{"label": lbl, "count": int(n), "pct": float(pct)} for lbl, n, pct in counts]

    # Structured kova tablosu sütunları (hem data hem rapor tablosu aynı veri).
    def _bucket_columns(first_header: str):
        return [
            {"key": "label", "label": first_header, "align": "left"},
            {"key": "count", "label": "n",          "align": "right"},
            {"key": "pct",   "label": "oran",       "align": "right"},
        ]

    def _bucket_rows(counts):
        return [[lbl, _common.fmt_int(n), f"%{pct:.1f}"] for lbl, n, pct in counts]

    notes_items = [
        (
            "`message_count` tüm mesajları (human + assistant + tool-only) sayar; "
            "`human_turn_count` yalnız `sender='human'` mesajları — yani saf talep sayısı."
        ),
        (
            "Token sayımı `tiktoken.cl100k_base` (OpenAI) ile yapılmıştır; Claude'un "
            "gerçek tokenizer'ı değildir — **yaklaşık üst sınır** olarak okunmalıdır."
        ),
        "Boş konuşmalar quantile hesaplamasından dışlanmıştır.",
    ]

    sections = [
        _common.Section(
            "Öne çıkanlar",
            highlights,
            blocks=[_common.block_bullets(highlight_items)],
        ),
        _common.Section(
            "Dağılım — quantile",
            "\n".join([
                _common.percentile_table(sum_mc, "mesaj/konuşma"),
                "",
                _common.percentile_table(sum_tn, "human turn/konuşma"),
                "",
                _common.percentile_table(sum_tk, "token/konuşma (≈cl100k)"),
            ]),
            blocks=[
                _common.block_percentile_table(sum_mc, "mesaj/konuşma"),
                _common.block_percentile_table(sum_tn, "human turn/konuşma"),
                _common.block_percentile_table(sum_tk, "token/konuşma (≈cl100k)"),
            ],
        ),
        _common.Section(
            "Mesaj sayısı kovaları",
            _common.markdown_table(_bucket_table(mc_counts), headers=["mesaj aralığı", "n", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    label="Konuşma başına mesaj sayısı",
                    buckets=_bucket_payload(mc_counts),
                    image="bucket_messages.png",
                    xlabel="mesaj sayısı (aralık)",
                ),
                _common.block_table(_bucket_columns("mesaj aralığı"), _bucket_rows(mc_counts)),
            ],
        ),
        _common.Section(
            "Human turn kovaları",
            _common.markdown_table(_bucket_table(tn_counts), headers=["turn aralığı", "n", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    label="Konuşma başına human turn sayısı",
                    buckets=_bucket_payload(tn_counts),
                    image="bucket_turns.png",
                    xlabel="human turn sayısı (aralık)",
                ),
                _common.block_table(_bucket_columns("turn aralığı"), _bucket_rows(tn_counts)),
            ],
        ),
        _common.Section(
            "Token kovaları",
            _common.markdown_table(_bucket_table(tk_counts), headers=["token aralığı", "n", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    label="Konuşma başına toplam token",
                    buckets=_bucket_payload(tk_counts),
                    image="bucket_tokens.png",
                    xlabel="toplam token (aralık)",
                ),
                _common.block_table(_bucket_columns("token aralığı"), _bucket_rows(tk_counts)),
            ],
        ),
        _common.Section(
            "Grafikler",
            "- `bucket_messages.png` — konuşma başına mesaj sayısı, kova dağılımı\n"
            "- `bucket_turns.png` — konuşma başına human turn sayısı, kova dağılımı\n"
            "- `bucket_tokens.png` — konuşma başına toplam token, kova dağılımı",
            # Yapısal render için chart bloklarını bucket section'larında
            # zaten verdik; burada "Grafikler" başlığını frontend gizleyebilsin
            # diye boş blocks listesi veriyoruz (dual-render tarafı bu
            # section'ı structured modda atlayacak).
            blocks=[],
        ),
        _common.Section(
            "Notlar",
            "- " + "\n- ".join(notes_items),
            blocks=[_common.block_bullets(notes_items)],
        ),
    ]
    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)
    return summary
