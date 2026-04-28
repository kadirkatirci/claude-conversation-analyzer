"""
m03 — Thinking bloğu kullanımı

(a) Konuşma bazında `has_thinking` oranı + ay bazlı trend.
(b) Assistant cevap uzunluğu: thinking yok vs thinking var (tokens quantile + kova).
(c) Thinking bloğu uzunluğu dağılımı (yalnız thinking>0 mesajlar).
(d) Assistant cevabında thinking payı: tokens_thinking / (tokens_thinking + tokens_text).
(e) Konuşma bazında toplam thinking token yükü.
(f) Thinking × tool_use çakışması (kontenjans).
(g) İlk assistant mesajı vs sonraki assistant mesajları — thinking oranı.
(h) Thinking kovası → cevap token medyanı (kondense kıyas).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from analysis import _common
from analysis._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "03-thinking-usage"
TITLE = "Thinking bloğu kullanımı"


RESPONSE_BUCKETS: list[tuple[str, float, float]] = [
    ("<200",      0,      200),
    ("200–1K",    200,    1_000),
    ("1K–3K",     1_000,  3_000),
    ("3K–10K",    3_000,  10_000),
    ("10K+",      10_000, 1e12),
]

THINKING_BUCKETS: list[tuple[str, float, float]] = [
    ("<200",      0,      200),
    ("200–1K",    200,    1_000),
    ("1K–3K",     1_000,  3_000),
    ("3K–10K",    3_000,  10_000),
    ("10K+",      10_000, 1e12),
]

THINK_SHARE_BUCKETS: list[tuple[str, float, float]] = [
    ("%0 (thinking yok)", 0,     1e-9),
    (">0–%20",            1e-9,  20),
    ("%20–%50",           20,    50),
    ("%50–%80",           50,    80),
    ("%80+",              80,    100.0001),
]

CONVO_THINKING_BUCKETS: list[tuple[str, float, float]] = [
    ("<1K",      0,       1_000),
    ("1K–5K",    1_000,   5_000),
    ("5K–20K",   5_000,   20_000),
    ("20K–50K",  20_000,  50_000),
    ("50K+",     50_000,  1e12),
]


def _fmt_token(n: float) -> str:
    if n is None or n != n:
        return "—"
    if n < 1_000:
        return f"{int(round(n))}"
    if n < 10_000:
        return f"{n/1000:.1f}K"
    if n < 1_000_000:
        return f"{int(round(n/1000))}K"
    return f"{n/1_000_000:.1f}M"


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


def _bucket_pair_table(
    labels: list[str],
    counts_a: list[tuple[str, int, float]],
    counts_b: list[tuple[str, int, float]],
    side_a: str,
    side_b: str,
) -> str:
    """İki kova sayımını yan yana tek tabloya koyar."""
    rows = []
    for (la, na, pa), (lb, nb, pb) in zip(counts_a, counts_b):
        assert la == lb, f"bucket labels must match: {la} vs {lb}"
        rows.append([
            la,
            _common.fmt_int(na), f"%{pa:.1f}",
            _common.fmt_int(nb), f"%{pb:.1f}",
        ])
    return _common.markdown_table(
        rows,
        headers=["token aralığı",
                 f"{side_a} n", f"{side_a} oran",
                 f"{side_b} n", f"{side_b} oran"],
    )


def _bucket_bar_pair(
    counts_top: list[tuple[str, int, float]],
    counts_bottom: list[tuple[str, int, float]],
    label_top: str,
    label_bottom: str,
    xlabel: str,
    color_top: str,
    color_bottom: str,
    path: Path,
) -> None:
    if not HAS_MPL:
        return
    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, counts, color, label in (
        (ax_t, counts_top, color_top, label_top),
        (ax_b, counts_bottom, color_bottom, label_bottom),
    ):
        labels = [c[0] for c in counts]
        ns = [c[1] for c in counts]
        pcts = [c[2] for c in counts]
        bars = ax.bar(labels, ns, color=color)
        ymax = max(ns) if ns else 1
        ax.set_ylim(0, ymax * 1.18)
        for bar, n, pct in zip(bars, ns, pcts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{n:,}\n%{pct:.1f}".replace(",", " "),
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylabel("mesaj sayısı")
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
    ax_b.set_xlabel(xlabel)
    _common.save_fig(fig, path)


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


def run(con: duckdb.DuckDBPyConnection, out_dir: Path, cfg: dict) -> dict:
    # ---------- Konuşma bazında oran + ay trend ----------
    convo_stats = con.execute("""
        SELECT
          COUNT(*) AS n,
          SUM(CASE WHEN has_thinking THEN 1 ELSE 0 END) AS n_thinking
        FROM _stats_conversation
        WHERE message_count > 0
    """).fetchone()
    n_convo, n_convo_thinking = convo_stats
    convo_rate = n_convo_thinking / n_convo if n_convo else 0.0

    monthly = con.execute("""
        SELECT
          strftime(date_trunc('month', created_at), '%Y-%m') AS month,
          COUNT(*) AS total,
          SUM(CASE WHEN has_thinking THEN 1 ELSE 0 END) AS with_thinking
        FROM _stats_conversation
        WHERE message_count > 0
        GROUP BY 1
        ORDER BY 1
    """).fetchall()

    # ---------- Assistant mesaj tabanlı: tokens_text, tokens_thinking, tool_use_count ----------
    asst_rows = con.execute("""
        SELECT
          message_uuid,
          conversation_uuid,
          tokens_text,
          tokens_thinking,
          tool_use_count,
          ROW_NUMBER() OVER (PARTITION BY conversation_uuid ORDER BY created_at) AS rn_asst
        FROM _stats_message
        WHERE sender = 'assistant'
    """).fetchall()
    # sütunlar: uuid, convo, tokens_text, tokens_thinking, tool_use_count, rn_asst
    if asst_rows:
        tokens_text = np.array([r[2] for r in asst_rows], dtype=float)
        tokens_thinking = np.array([r[3] for r in asst_rows], dtype=float)
        tool_use_count = np.array([r[4] for r in asst_rows], dtype=float)
        rn_asst = np.array([r[5] for r in asst_rows], dtype=int)
    else:
        tokens_text = tokens_thinking = tool_use_count = np.empty(0, dtype=float)
        rn_asst = np.empty(0, dtype=int)

    has_thinking_mask = tokens_thinking > 0
    n_asst = int(tokens_text.size)
    n_asst_with_thinking = int(has_thinking_mask.sum())

    # Cevap uzunluğu kıyası
    tk_with = tokens_text[has_thinking_mask]
    tk_with = tk_with[tk_with > 0]
    tk_without = tokens_text[~has_thinking_mask]
    tk_without = tk_without[tk_without > 0]

    stats_tk_with = _common.percentiles(tk_with)
    stats_tk_without = _common.percentiles(tk_without)

    counts_response_without = _bucket_counts(tk_without, RESPONSE_BUCKETS)
    counts_response_with = _bucket_counts(tk_with, RESPONSE_BUCKETS)

    # Thinking bloğu uzunluğu (yalnız thinking>0)
    thinking_tokens = tokens_thinking[has_thinking_mask]
    stats_thinking = _common.percentiles(thinking_tokens)
    counts_thinking = _bucket_counts(thinking_tokens, THINKING_BUCKETS)

    # Cevapta thinking payı — text>0 olan tüm assistant mesajları
    text_pos_mask = tokens_text > 0
    tt = tokens_text[text_pos_mask]
    th = tokens_thinking[text_pos_mask]
    denom = tt + th
    share_pct = np.where(denom > 0, 100.0 * th / denom, 0.0)
    counts_share = _bucket_counts(share_pct, THINK_SHARE_BUCKETS)

    total_tok_text = int(tt.sum())
    total_tok_think = int(th.sum())
    overall_think_share = (
        100.0 * total_tok_think / (total_tok_text + total_tok_think)
        if (total_tok_text + total_tok_think) else 0.0
    )

    # ---------- (E) Konuşma bazında thinking token yükü ----------
    convo_thinking_rows = con.execute("""
        SELECT tokens_thinking_total
        FROM _stats_conversation
        WHERE has_thinking = TRUE AND message_count > 0
    """).fetchall()
    convo_thinking_load = np.array([r[0] for r in convo_thinking_rows], dtype=float)
    stats_convo_load = _common.percentiles(convo_thinking_load)
    counts_convo_load = _bucket_counts(convo_thinking_load, CONVO_THINKING_BUCKETS)

    # ---------- (F) Thinking × tool_use çakışması (konuşma bazında) ----------
    contingency_rows = con.execute("""
        SELECT
          has_thinking,
          (tool_calls_total > 0) AS has_tool,
          COUNT(*) AS n
        FROM _stats_conversation
        WHERE message_count > 0
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).fetchall()
    contingency = {(bool(ht), bool(tu)): int(n) for ht, tu, n in contingency_rows}
    c_tt = contingency.get((True,  True),  0)
    c_tf = contingency.get((True,  False), 0)
    c_ft = contingency.get((False, True),  0)
    c_ff = contingency.get((False, False), 0)
    n_thinking_convo = c_tt + c_tf
    n_tool_convo = c_tt + c_ft
    # Koşullu oranlar
    rate_tool_given_thinking = (100.0 * c_tt / n_thinking_convo) if n_thinking_convo else 0.0
    rate_tool_given_nothinking = (100.0 * c_ft / (c_ft + c_ff)) if (c_ft + c_ff) else 0.0
    rate_thinking_given_tool = (100.0 * c_tt / n_tool_convo) if n_tool_convo else 0.0
    rate_thinking_given_notool = (100.0 * c_tf / (c_tf + c_ff)) if (c_tf + c_ff) else 0.0

    # ---------- (G) İlk assistant mesajı vs sonraki ----------
    first_mask = rn_asst == 1
    rest_mask = rn_asst > 1
    n_first = int(first_mask.sum())
    n_rest = int(rest_mask.sum())
    n_first_thinking = int((first_mask & has_thinking_mask).sum())
    n_rest_thinking = int((rest_mask & has_thinking_mask).sum())
    rate_first = (100.0 * n_first_thinking / n_first) if n_first else 0.0
    rate_rest = (100.0 * n_rest_thinking / n_rest) if n_rest else 0.0

    # ---------- (H) Thinking kovası → cevap token medyanı ----------
    # Thinking>0 olan mesajlarda her thinking-kovasının cevap medyanı nedir?
    resp_by_thinking_bucket = []
    for label, lo, hi in THINKING_BUCKETS:
        in_bucket = has_thinking_mask & (tokens_thinking >= lo) & (tokens_thinking < hi)
        n_b = int(in_bucket.sum())
        if n_b == 0:
            resp_by_thinking_bucket.append({
                "label": label, "n": 0,
                "resp_p50": None, "resp_p90": None, "resp_max": None,
            })
            continue
        resp_vals = tokens_text[in_bucket]
        resp_vals = resp_vals[resp_vals > 0]
        st = _common.percentiles(resp_vals)
        resp_by_thinking_bucket.append({
            "label": label, "n": n_b,
            "resp_p50": st["p50"], "resp_p90": st["p90"], "resp_max": st["max"],
        })

    # ---------- Grafikler ----------
    months = [r[0] for r in monthly]
    totals = np.array([r[1] for r in monthly], dtype=float)
    withs = np.array([r[2] for r in monthly], dtype=float)
    rates = np.where(totals > 0, withs / totals * 100.0, 0.0)

    if HAS_MPL:
        fig, ax = plt.subplots()
        ax.plot(range(len(months)), rates, marker="o", color="#C44E52", linewidth=1.5)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha="right")
        ax.set_ylabel("thinking içeren konuşma (%)")
        ax.set_title("Thinking kullanımı — ay bazlı oran")
        ax.set_ylim(0, 100)
        _common.save_fig(fig, out_dir / "monthly_rate.png")


    _bucket_bar_pair(
        counts_response_without, counts_response_with,
        label_top=f"thinking yok — cevap token kova dağılımı (n={_common.fmt_int(int(tk_without.size))})",
        label_bottom=f"thinking var — cevap token kova dağılımı (n={_common.fmt_int(int(tk_with.size))})",
        xlabel="cevap token (aralık)",
        color_top="#4C72B0",
        color_bottom="#C44E52",
        path=out_dir / "bucket_response_tokens.png",
    )

    _bucket_bar_single(
        counts_thinking,
        title=f"Thinking bloğu uzunluğu — kova dağılımı (n={_common.fmt_int(int(thinking_tokens.size))})",
        xlabel="thinking token (aralık)",
        ylabel="mesaj sayısı",
        color="#8172B3",
        path=out_dir / "bucket_thinking_tokens.png",
    )

    _bucket_bar_single(
        counts_share,
        title=f"Cevapta thinking payı — kova dağılımı (n={_common.fmt_int(int(tt.size))})",
        xlabel="thinking / (thinking + text) oranı",
        ylabel="mesaj sayısı",
        color="#C44E52",
        path=out_dir / "bucket_thinking_share.png",
    )

    _bucket_bar_single(
        counts_convo_load,
        title=f"Konuşma başına thinking token yükü — kova dağılımı "
              f"(n={_common.fmt_int(int(convo_thinking_load.size))} thinking'li konuşma)",
        xlabel="toplam thinking token (aralık)",
        ylabel="konuşma sayısı",
        color="#8172B3",
        path=out_dir / "bucket_convo_thinking_load.png",
    )

    # ---------- CSV ----------
    _common.write_csv(
        out_dir, "monthly.csv",
        [(m, int(t), int(w), round(r, 2)) for m, t, w, r in zip(months, totals.astype(int), withs.astype(int), rates)],
        headers=["month", "conversations", "with_thinking", "rate_pct"],
    )

    # ---------- Summary ----------
    summary = {
        "convo_rate_pct": round(convo_rate * 100, 2),
        "n_conversations": int(n_convo),
        "n_conversations_with_thinking": int(n_convo_thinking),
        "assistant_messages": {
            "total": n_asst,
            "with_thinking": n_asst_with_thinking,
            "without_thinking": n_asst - n_asst_with_thinking,
        },
        "response_tokens_with": stats_tk_with,
        "response_tokens_without": stats_tk_without,
        "thinking_tokens": stats_thinking,
        "buckets_response_without": {lbl: n for lbl, n, _ in counts_response_without},
        "buckets_response_with":    {lbl: n for lbl, n, _ in counts_response_with},
        "buckets_thinking":         {lbl: n for lbl, n, _ in counts_thinking},
        "thinking_share": {
            "total_tokens_text":     total_tok_text,
            "total_tokens_thinking": total_tok_think,
            "overall_share_pct":     round(overall_think_share, 2),
            "buckets":               {lbl: n for lbl, n, _ in counts_share},
        },
        "convo_thinking_load": {
            "n": int(convo_thinking_load.size),
            "tokens": stats_convo_load,
            "buckets": {lbl: n for lbl, n, _ in counts_convo_load},
        },
        "contingency": {
            "thinking_and_tool":       c_tt,
            "thinking_no_tool":        c_tf,
            "no_thinking_and_tool":    c_ft,
            "no_thinking_no_tool":     c_ff,
            "pct_tool_given_thinking":   round(rate_tool_given_thinking, 2),
            "pct_tool_given_nothinking": round(rate_tool_given_nothinking, 2),
            "pct_thinking_given_tool":   round(rate_thinking_given_tool, 2),
            "pct_thinking_given_notool": round(rate_thinking_given_notool, 2),
        },
        "first_vs_rest_assistant": {
            "first": {"n": n_first, "with_thinking": n_first_thinking, "rate_pct": round(rate_first, 2)},
            "rest":  {"n": n_rest,  "with_thinking": n_rest_thinking,  "rate_pct": round(rate_rest, 2)},
        },
        "response_by_thinking_bucket": resp_by_thinking_bucket,
        "headline": (
            f"konuşmaların %{convo_rate*100:.1f}'inde en az bir thinking bloğu; "
            f"thinking'li cevap medyan ~{_fmt_token(stats_tk_with['p50'])} token, "
            f"thinking'siz ~{_fmt_token(stats_tk_without['p50'])}; "
            f"thinking bloğu medyan ~{_fmt_token(stats_thinking['p50'])} token."
        ),
    }
    _common.write_json(out_dir, "summary.json", summary)

    # ---------- Öne çıkanlar ----------
    highlights = (
        f"- Toplam konuşma (non-empty): **{_common.fmt_int(n_convo)}**; "
        f"thinking içeren: **{_common.fmt_int(n_convo_thinking)}** "
        f"(**%{convo_rate*100:.1f}**)\n"
        f"- Assistant mesajı: **{_common.fmt_int(n_asst)}**; "
        f"thinking bloğu olan: **{_common.fmt_int(n_asst_with_thinking)}** "
        f"(**%{100*n_asst_with_thinking/max(n_asst,1):.1f}**)\n"
        f"- Cevap medyanı: thinking yok **~{_fmt_token(stats_tk_without['p50'])} token**, "
        f"thinking var **~{_fmt_token(stats_tk_with['p50'])} token**\n"
        f"- Cevap p95: thinking yok **~{_fmt_token(stats_tk_without['p95'])} token**, "
        f"thinking var **~{_fmt_token(stats_tk_with['p95'])} token**\n"
        f"- Thinking bloğu: medyan **~{_fmt_token(stats_thinking['p50'])} token**, "
        f"p95 **~{_fmt_token(stats_thinking['p95'])} token**, "
        f"max **~{_fmt_token(stats_thinking['max'])} token**\n"
        f"- Toplam assistant token'ının **%{overall_think_share:.1f}**'i thinking "
        f"(görünen text dışı)\n"
        f"- Thinking'li konuşma başına toplam thinking: medyan "
        f"**~{_fmt_token(stats_convo_load['p50'])} token**, "
        f"max **~{_fmt_token(stats_convo_load['max'])} token**\n"
        f"- Thinking-li konuşmaların **%{rate_tool_given_thinking:.1f}**'i tool da kullanıyor; "
        f"thinking-siz konuşmalarda bu oran **%{rate_tool_given_nothinking:.1f}**\n"
        f"- İlk assistant mesajında thinking oranı **%{rate_first:.1f}**; "
        f"sonraki mesajlarda **%{rate_rest:.1f}**"
    )

    # Cevap token kova kıyası (tek tablo)
    response_pair_table = _bucket_pair_table(
        ["<200", "200–1K", "1K–3K", "3K–10K", "10K+"],
        counts_response_without, counts_response_with,
        side_a="yok", side_b="var",
    )

    # Kontenjans tablosu
    contingency_table = _common.markdown_table(
        [
            ["thinking var, tool var",  _common.fmt_int(c_tt),
             f"%{100*c_tt/max(n_convo,1):.1f}"],
            ["thinking var, tool yok",  _common.fmt_int(c_tf),
             f"%{100*c_tf/max(n_convo,1):.1f}"],
            ["thinking yok, tool var",  _common.fmt_int(c_ft),
             f"%{100*c_ft/max(n_convo,1):.1f}"],
            ["thinking yok, tool yok",  _common.fmt_int(c_ff),
             f"%{100*c_ff/max(n_convo,1):.1f}"],
        ],
        headers=["kesişim", "n konuşma", "tüm konuşmaların oranı"],
    )
    conditional_table = _common.markdown_table(
        [
            ["thinking-li konuşmalarda tool kullanımı",
             f"%{rate_tool_given_thinking:.1f}",
             f"{c_tt} / {n_thinking_convo}"],
            ["thinking-siz konuşmalarda tool kullanımı",
             f"%{rate_tool_given_nothinking:.1f}",
             f"{c_ft} / {c_ft + c_ff}"],
            ["tool kullanan konuşmalarda thinking oranı",
             f"%{rate_thinking_given_tool:.1f}",
             f"{c_tt} / {n_tool_convo}"],
            ["tool kullanmayan konuşmalarda thinking oranı",
             f"%{rate_thinking_given_notool:.1f}",
             f"{c_tf} / {c_tf + c_ff}"],
        ],
        headers=["koşullu oran", "değer", "pay"],
    )

    # İlk vs sonraki
    first_vs_rest_table = _common.markdown_table(
        [
            ["ilk assistant mesajı",
             _common.fmt_int(n_first),
             _common.fmt_int(n_first_thinking),
             f"%{rate_first:.1f}"],
            ["sonraki assistant mesajları",
             _common.fmt_int(n_rest),
             _common.fmt_int(n_rest_thinking),
             f"%{rate_rest:.1f}"],
        ],
        headers=["kitle", "n", "thinking'li", "oran"],
    )

    # Thinking kovası → cevap medyanı
    resp_by_thinking_table = _common.markdown_table(
        [
            [
                row["label"],
                _common.fmt_int(row["n"]),
                _fmt_token(row["resp_p50"]) if row["resp_p50"] is not None else "—",
                _fmt_token(row["resp_p90"]) if row["resp_p90"] is not None else "—",
                _fmt_token(row["resp_max"]) if row["resp_max"] is not None else "—",
            ]
            for row in resp_by_thinking_bucket
        ],
        headers=["thinking kovası", "n", "cevap p50", "cevap p90", "cevap max"],
    )

    # ── Structured blocks ──
    def _buckets_payload(counts: list[tuple[str, int, float]]) -> list[dict]:
        return [{"label": lbl, "count": n, "pct": pct} for lbl, n, pct in counts]

    bucket_table_columns_3 = [
        {"key": "range", "label": "aralık", "align": "left"},
        {"key": "n",     "label": "n",      "align": "right"},
        {"key": "pct",   "label": "oran",   "align": "right"},
    ]

    highlight_items = [
        f"Toplam konuşma (non-empty): **{_common.fmt_int(n_convo)}**; "
        f"thinking içeren: **{_common.fmt_int(n_convo_thinking)}** "
        f"(**%{convo_rate*100:.1f}**)",
        f"Assistant mesajı: **{_common.fmt_int(n_asst)}**; "
        f"thinking bloğu olan: **{_common.fmt_int(n_asst_with_thinking)}** "
        f"(**%{100*n_asst_with_thinking/max(n_asst,1):.1f}**)",
        f"Cevap medyanı: thinking yok **~{_fmt_token(stats_tk_without['p50'])} token**, "
        f"thinking var **~{_fmt_token(stats_tk_with['p50'])} token**",
        f"Cevap p95: thinking yok **~{_fmt_token(stats_tk_without['p95'])} token**, "
        f"thinking var **~{_fmt_token(stats_tk_with['p95'])} token**",
        f"Thinking bloğu: medyan **~{_fmt_token(stats_thinking['p50'])} token**, "
        f"p95 **~{_fmt_token(stats_thinking['p95'])} token**, "
        f"max **~{_fmt_token(stats_thinking['max'])} token**",
        f"Toplam assistant token'ının **%{overall_think_share:.1f}**'i thinking "
        f"(görünen text dışı)",
        f"Thinking'li konuşma başına toplam thinking: medyan "
        f"**~{_fmt_token(stats_convo_load['p50'])} token**, "
        f"max **~{_fmt_token(stats_convo_load['max'])} token**",
        f"Thinking-li konuşmaların **%{rate_tool_given_thinking:.1f}**'i tool da kullanıyor; "
        f"thinking-siz konuşmalarda bu oran **%{rate_tool_given_nothinking:.1f}**",
        f"İlk assistant mesajında thinking oranı **%{rate_first:.1f}**; "
        f"sonraki mesajlarda **%{rate_rest:.1f}**",
    ]

    # Response-length quantile: tek tabloda 2 satır
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
    response_quantile_rows = [
        _qrow("thinking yok", stats_tk_without),
        _qrow("thinking var", stats_tk_with),
    ]

    # Response pair (yok vs var) — structured table
    response_pair_columns = [
        {"key": "range",  "label": "token aralığı", "align": "left"},
        {"key": "n_no",   "label": "yok n",         "align": "right"},
        {"key": "pct_no", "label": "yok oran",      "align": "right"},
        {"key": "n_yes",  "label": "var n",         "align": "right"},
        {"key": "pct_yes","label": "var oran",      "align": "right"},
    ]
    response_pair_rows = [
        [la, _common.fmt_int(na), f"%{pa:.1f}", _common.fmt_int(nb), f"%{pb:.1f}"]
        for (la, na, pa), (_, nb, pb) in zip(counts_response_without, counts_response_with)
    ]

    # Contingency
    contingency_columns = [
        {"key": "kesişim", "label": "kesişim",           "align": "left"},
        {"key": "n",       "label": "n konuşma",         "align": "right"},
        {"key": "pct",     "label": "tüm konuşmaların oranı", "align": "right"},
    ]
    contingency_rows = [
        ["thinking var, tool var",  _common.fmt_int(c_tt), f"%{100*c_tt/max(n_convo,1):.1f}"],
        ["thinking var, tool yok",  _common.fmt_int(c_tf), f"%{100*c_tf/max(n_convo,1):.1f}"],
        ["thinking yok, tool var",  _common.fmt_int(c_ft), f"%{100*c_ft/max(n_convo,1):.1f}"],
        ["thinking yok, tool yok",  _common.fmt_int(c_ff), f"%{100*c_ff/max(n_convo,1):.1f}"],
    ]
    conditional_columns = [
        {"key": "metric", "label": "koşullu oran", "align": "left"},
        {"key": "value",  "label": "değer",        "align": "right"},
        {"key": "ratio",  "label": "pay",          "align": "right"},
    ]
    conditional_rows = [
        ["thinking-li konuşmalarda tool kullanımı",
         f"%{rate_tool_given_thinking:.1f}", f"{c_tt} / {n_thinking_convo}"],
        ["thinking-siz konuşmalarda tool kullanımı",
         f"%{rate_tool_given_nothinking:.1f}", f"{c_ft} / {c_ft + c_ff}"],
        ["tool kullanan konuşmalarda thinking oranı",
         f"%{rate_thinking_given_tool:.1f}", f"{c_tt} / {n_tool_convo}"],
        ["tool kullanmayan konuşmalarda thinking oranı",
         f"%{rate_thinking_given_notool:.1f}", f"{c_tf} / {c_tf + c_ff}"],
    ]

    # First vs rest
    first_vs_rest_columns = [
        {"key": "kitle",    "label": "kitle",       "align": "left"},
        {"key": "n",        "label": "n",           "align": "right"},
        {"key": "thinking", "label": "thinking'li", "align": "right"},
        {"key": "rate",     "label": "oran",        "align": "right"},
    ]
    first_vs_rest_rows = [
        ["ilk assistant mesajı",
         _common.fmt_int(n_first), _common.fmt_int(n_first_thinking), f"%{rate_first:.1f}"],
        ["sonraki assistant mesajları",
         _common.fmt_int(n_rest), _common.fmt_int(n_rest_thinking), f"%{rate_rest:.1f}"],
    ]

    # Thinking bucket → response
    resp_by_thinking_columns = [
        {"key": "bucket", "label": "thinking kovası", "align": "left"},
        {"key": "n",      "label": "n",               "align": "right"},
        {"key": "p50",    "label": "cevap p50",       "align": "right"},
        {"key": "p90",    "label": "cevap p90",       "align": "right"},
        {"key": "max",    "label": "cevap max",       "align": "right"},
    ]
    resp_by_thinking_rows = [
        [
            row["label"],
            _common.fmt_int(row["n"]),
            _fmt_token(row["resp_p50"]) if row["resp_p50"] is not None else "—",
            _fmt_token(row["resp_p90"]) if row["resp_p90"] is not None else "—",
            _fmt_token(row["resp_max"]) if row["resp_max"] is not None else "—",
        ]
        for row in resp_by_thinking_bucket
    ]

    # Ay bazlı oran — structured bucket_bar (label = month, count = rate_pct, pct = total)
    # Bar height = oran (%), yardımcı bilgi = toplam konuşma.
    monthly_buckets = [
        {"label": m, "count": round(r, 1), "pct": None}
        for m, r in zip(months, rates)
    ]

    notes_items = [
        "`has_thinking`: konuşma içinde en az bir assistant mesajının "
        "`chars_thinking > 0` olması.",
        "Cevap uzunluğu hesabı yalnız `tokens_text > 0` assistant mesajları üzerindendir.",
        "Thinking payı = `tokens_thinking / (tokens_thinking + tokens_text)`; "
        "paydada assistant mesajının text'i sıfırsa (yalnız tool-use / tool-result) "
        "hesap dışı kalır.",
        "İlk assistant mesajı = konuşmada `sender='assistant'` mesajlar arasında "
        "`created_at`'e göre sıralanınca 1. sırada olan.",
        "Token sayımı `tiktoken.cl100k_base` (OpenAI) ile yapılmıştır; Claude'un "
        "gerçek tokenizer'ı değildir — **yaklaşık üst sınır** olarak okunmalıdır.",
    ]

    sections = [
        _common.Section(
            "Öne çıkanlar",
            highlights,
            blocks=[_common.block_bullets(highlight_items)],
        ),
        _common.Section(
            "Aylık thinking oranı (konuşma bazında)",
            "",
            blocks=[
                _common.block_bucket_chart(
                    label="Thinking içeren konuşma oranı — ay bazlı (%)",
                    buckets=monthly_buckets,
                    image="monthly_rate.png",
                    xlabel="ay",
                ),
            ],
        ),
        _common.Section(
            "Cevap uzunluğu — tokens (thinking yok vs var)",
            "\n".join([
                _common.percentile_table(stats_tk_without, "thinking yok"),
                "",
                _common.percentile_table(stats_tk_with, "thinking var"),
                "",
                response_pair_table,
            ]),
            blocks=[
                _common.block_table(quantile_columns, response_quantile_rows),
                _common.block_bucket_chart(
                    label=f"thinking yok — cevap token kova dağılımı (n={_common.fmt_int(int(tk_without.size))})",
                    buckets=_buckets_payload(counts_response_without),
                    image="bucket_response_tokens.png",
                    xlabel="cevap token (aralık)",
                ),
                _common.block_bucket_chart(
                    label=f"thinking var — cevap token kova dağılımı (n={_common.fmt_int(int(tk_with.size))})",
                    buckets=_buckets_payload(counts_response_with),
                    xlabel="cevap token (aralık)",
                ),
                _common.block_table(response_pair_columns, response_pair_rows),
            ],
        ),
        _common.Section(
            "Thinking bloğu uzunluğu",
            _common.percentile_table(stats_thinking, "thinking tokens")
            + "\n\n"
            + _common.markdown_table(
                _bucket_table(counts_thinking),
                headers=["thinking token aralığı", "n", "oran"],
            ),
            blocks=[
                _common.block_percentile_table(stats_thinking, "thinking tokens"),
                _common.block_bucket_chart(
                    label=f"Thinking bloğu uzunluğu — kova dağılımı (n={_common.fmt_int(int(thinking_tokens.size))})",
                    buckets=_buckets_payload(counts_thinking),
                    image="bucket_thinking_tokens.png",
                    xlabel="thinking token (aralık)",
                ),
                _common.block_table(
                    bucket_table_columns_3,
                    [[lbl, _common.fmt_int(n), f"%{pct:.1f}"] for lbl, n, pct in counts_thinking],
                ),
            ],
        ),
        _common.Section(
            "Cevapta thinking payı",
            f"- Toplam thinking token: **{_common.fmt_int(total_tok_think)}** / "
            f"toplam text + thinking token: "
            f"**{_common.fmt_int(total_tok_text + total_tok_think)}** "
            f"→ **%{overall_think_share:.1f}**\n\n"
            + _common.markdown_table(
                _bucket_table(counts_share),
                headers=["thinking payı", "n", "oran"],
            ),
            blocks=[
                _common.block_bullets([
                    f"Toplam thinking token: **{_common.fmt_int(total_tok_think)}** / "
                    f"toplam text + thinking token: "
                    f"**{_common.fmt_int(total_tok_text + total_tok_think)}** "
                    f"→ **%{overall_think_share:.1f}**"
                ]),
                _common.block_bucket_chart(
                    label=f"Cevapta thinking payı — kova dağılımı (n={_common.fmt_int(int(tt.size))})",
                    buckets=_buckets_payload(counts_share),
                    image="bucket_thinking_share.png",
                    xlabel="thinking / (thinking + text) oranı",
                ),
                _common.block_table(
                    bucket_table_columns_3,
                    [[lbl, _common.fmt_int(n), f"%{pct:.1f}"] for lbl, n, pct in counts_share],
                ),
            ],
        ),
        _common.Section(
            "Konuşma başına thinking token yükü",
            _common.percentile_table(stats_convo_load, "thinking/konuşma")
            + "\n\n"
            + _common.markdown_table(
                _bucket_table(counts_convo_load),
                headers=["thinking aralığı", "n konuşma", "oran"],
            ),
            blocks=[
                _common.block_percentile_table(stats_convo_load, "thinking/konuşma"),
                _common.block_bucket_chart(
                    label=f"Konuşma başına thinking token yükü — kova dağılımı "
                          f"(n={_common.fmt_int(int(convo_thinking_load.size))} thinking'li konuşma)",
                    buckets=_buckets_payload(counts_convo_load),
                    image="bucket_convo_thinking_load.png",
                    xlabel="toplam thinking token (aralık)",
                ),
                _common.block_table(
                    bucket_table_columns_3,
                    [[lbl, _common.fmt_int(n), f"%{pct:.1f}"] for lbl, n, pct in counts_convo_load],
                ),
            ],
        ),
        _common.Section(
            "Thinking × tool kullanımı çakışması (konuşma bazında)",
            contingency_table + "\n\n" + conditional_table,
            blocks=[
                _common.block_table(contingency_columns, contingency_rows),
                _common.block_table(conditional_columns, conditional_rows),
            ],
        ),
        _common.Section(
            "İlk assistant mesajı vs sonraki mesajlar — thinking oranı",
            first_vs_rest_table,
            blocks=[_common.block_table(first_vs_rest_columns, first_vs_rest_rows)],
        ),
        _common.Section(
            "Thinking kovası → cevap token (kondense)",
            resp_by_thinking_table,
            blocks=[_common.block_table(resp_by_thinking_columns, resp_by_thinking_rows)],
        ),
        _common.Section(
            "Grafikler",
            "- `monthly_rate.png` — ay bazlı thinking kullanım oranı (konuşma bazında)\n"
            "- `bucket_response_tokens.png` — cevap token dağılımı "
            "(üstte thinking yok, altta thinking var)\n"
            "- `bucket_thinking_tokens.png` — thinking bloğu uzunluk dağılımı\n"
            "- `bucket_thinking_share.png` — cevapta thinking payı dağılımı\n"
            "- `bucket_convo_thinking_load.png` — thinking'li konuşmalarda konuşma "
            "başına toplam thinking token yükü",
            blocks=[],
        ),
        _common.Section(
            "Notlar",
            "- `has_thinking`: konuşma içinde en az bir assistant mesajının "
            "`chars_thinking > 0` olması.\n"
            "- Cevap uzunluğu hesabı yalnız `tokens_text > 0` assistant mesajları "
            "üzerindendir.\n"
            "- Thinking payı = `tokens_thinking / (tokens_thinking + tokens_text)`; "
            "paydada assistant mesajının text'i sıfırsa (yalnız tool-use / tool-result) "
            "hesap dışı kalır.\n"
            "- İlk assistant mesajı = konuşmada `sender='assistant'` mesajlar arasında "
            "`created_at`'e göre sıralanınca 1. sırada olan.\n"
            "- Token sayımı `tiktoken.cl100k_base` (OpenAI) ile yapılmıştır; Claude'un "
            "gerçek tokenizer'ı değildir — **yaklaşık üst sınır** olarak okunmalıdır.",
            blocks=[_common.block_bullets(notes_items)],
        ),
    ]
    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)
    return summary
