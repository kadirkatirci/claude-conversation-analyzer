"""
m02 — Message length distribution (by sender)

Each message's tokens_text (excluding thinking, pure generated text).
Human vs assistant comparison:
  - Quantile table (token)
  - Bucket distribution (two subplots: top human, bottom assistant)
  - First human message vs follow-up human message bucket distribution
  - Code block share in assistant response (tokens_code / tokens_text)
Tool-only messages (chars_text=0) are reported separately.
Voice_note-containing message count is reported (if few, just a note; if many, separate analysis suggested).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from analysis import _common
from analysis._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "02-message-length"
TITLE = "Mesaj uzunluğu dağılımı (human vs assistant)"


TOKEN_BUCKETS: list[tuple[str, float, float]] = [
    ("<50",       0,     50),
    ("50–200",    50,    200),
    ("200–1K",    200,   1_000),
    ("1K–5K",     1_000, 5_000),
    ("5K+",       5_000, 1e12),
]

CODE_SHARE_BUCKETS: list[tuple[str, float, float]] = [
    ("%0 (kod yok)",  0,    1e-9),
    (">0–%10",        1e-9, 10),
    ("%10–%30",       10,   30),
    ("%30–%60",       30,   60),
    ("%60+",          60,   100.0001),
]


def _load(con: duckdb.DuckDBPyConnection) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for sender in ("human", "assistant"):
        rows = con.execute(f"""
            SELECT chars_text, tokens_text, tokens_code
            FROM _stats_message
            WHERE sender = '{sender}'
        """).fetchall()
        out[sender] = np.array(rows, dtype=float) if rows else np.empty((0, 3))
    return out


def _load_human_first_vs_followup(con: duckdb.DuckDBPyConnection) -> dict[str, np.ndarray]:
    rows = con.execute("""
        WITH ranked AS (
          SELECT
            sm.tokens_text,
            ROW_NUMBER() OVER (PARTITION BY sm.conversation_uuid ORDER BY sm.created_at) AS rn
          FROM _stats_message sm
          WHERE sm.sender = 'human' AND sm.tokens_text > 0
        )
        SELECT CASE WHEN rn = 1 THEN 'first' ELSE 'followup' END AS kind, tokens_text
        FROM ranked
    """).fetchall()
    first = np.array([t for k, t in rows if k == "first"], dtype=float)
    followup = np.array([t for k, t in rows if k == "followup"], dtype=float)
    return {"first": first, "followup": followup}


def _voice_note_count(con: duckdb.DuckDBPyConnection) -> int:
    return int(con.execute("""
        SELECT COUNT(DISTINCT cb.message_uuid)
        FROM content_blocks cb
        WHERE cb.type = 'voice_note'
    """).fetchone()[0])


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
    ax.set_ylabel("mesaj sayısı")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    _common.save_fig(fig, path)


def run(con: duckdb.DuckDBPyConnection, out_dir: Path, cfg: dict) -> dict:
    data = _load(con)

    stats_tokens = {
        s: _common.percentiles(data[s][:, 1][data[s][:, 1] > 0])
        for s in ("human", "assistant")
    }

    tool_only = {
        s: int((data[s][:, 0] == 0).sum()) for s in ("human", "assistant")
    }
    totals = {s: int(data[s].shape[0]) for s in ("human", "assistant")}

    tk_h = data["human"][:, 1][data["human"][:, 1] > 0]
    tk_a = data["assistant"][:, 1][data["assistant"][:, 1] > 0]

    counts_h = _bucket_counts(tk_h, TOKEN_BUCKETS)
    counts_a = _bucket_counts(tk_a, TOKEN_BUCKETS)

    _bucket_bar_pair(
        counts_h, counts_a,
        label_top=f"human — kova dağılımı (n={_common.fmt_int(tk_h.size)})",
        label_bottom=f"assistant — kova dağılımı (n={_common.fmt_int(tk_a.size)})",
        xlabel="token (aralık)",
        color_top="#55A868",
        color_bottom="#4C72B0",
        path=out_dir / "bucket_tokens.png",
    )

    # Token volume ratio
    total_tok_h = int(data["human"][:, 1].sum())
    total_tok_a = int(data["assistant"][:, 1].sum())
    total_tok = total_tok_h + total_tok_a
    share_h = 100.0 * total_tok_h / total_tok if total_tok else 0.0
    share_a = 100.0 * total_tok_a / total_tok if total_tok else 0.0
    ratio_a_to_h = (total_tok_a / total_tok_h) if total_tok_h else float("inf")

    # First vs follow-up human message
    fvf = _load_human_first_vs_followup(con)
    counts_first = _bucket_counts(fvf["first"], TOKEN_BUCKETS)
    counts_followup = _bucket_counts(fvf["followup"], TOKEN_BUCKETS)
    stats_first = _common.percentiles(fvf["first"])
    stats_followup = _common.percentiles(fvf["followup"])

    _bucket_bar_pair(
        counts_first, counts_followup,
        label_top=f"human ilk mesaj — kova dağılımı (n={_common.fmt_int(fvf['first'].size)})",
        label_bottom=f"human takip mesajı — kova dağılımı (n={_common.fmt_int(fvf['followup'].size)})",
        xlabel="token (aralık)",
        color_top="#DD8452",
        color_bottom="#55A868",
        path=out_dir / "bucket_tokens_first_vs_followup.png",
    )

    # Assistant code share
    asst_nonzero = data["assistant"][data["assistant"][:, 1] > 0]
    code_share_pct = 100.0 * asst_nonzero[:, 2] / asst_nonzero[:, 1]
    counts_code = _bucket_counts(code_share_pct, CODE_SHARE_BUCKETS)
    asst_total_tok_text = int(asst_nonzero[:, 1].sum())
    asst_total_tok_code = int(asst_nonzero[:, 2].sum())
    asst_code_share = 100.0 * asst_total_tok_code / asst_total_tok_text if asst_total_tok_text else 0.0
    n_asst_with_code = int((asst_nonzero[:, 2] > 0).sum())
    pct_asst_with_code = 100.0 * n_asst_with_code / asst_nonzero.shape[0] if asst_nonzero.shape[0] else 0.0

    _bucket_bar_single(
        counts_code,
        title=f"Assistant cevabında kod bloğu payı (n={_common.fmt_int(asst_nonzero.shape[0])})",
        xlabel="cevabın ne kadarı kod (token oranı)",
        color="#4C72B0",
        path=out_dir / "assistant_code_share.png",
    )

    # Voice note
    n_voice = _voice_note_count(con)

    # CSV: sender,chars,tokens,tokens_code
    csv_rows = []
    for s in ("human", "assistant"):
        for r in data[s]:
            csv_rows.append((s, int(r[0]), int(r[1]), int(r[2])))
    _common.write_csv(
        out_dir, "data.csv", csv_rows,
        headers=["sender", "chars_text", "tokens_text", "tokens_code"],
    )

    summary = {
        "totals": totals,
        "tool_only_messages": tool_only,
        "tokens": stats_tokens,
        "buckets_tokens_human": {lbl: n for lbl, n, _ in counts_h},
        "buckets_tokens_assistant": {lbl: n for lbl, n, _ in counts_a},
        "token_volume": {
            "human": total_tok_h,
            "assistant": total_tok_a,
            "total": total_tok,
            "human_share_pct": round(share_h, 2),
            "assistant_share_pct": round(share_a, 2),
            "assistant_to_human_ratio": round(ratio_a_to_h, 2),
        },
        "human_first_vs_followup": {
            "first":    {"p50": stats_first["p50"], "p95": stats_first["p95"], "n": stats_first["n"]},
            "followup": {"p50": stats_followup["p50"], "p95": stats_followup["p95"], "n": stats_followup["n"]},
        },
        "assistant_code_share": {
            "total_tokens_text": asst_total_tok_text,
            "total_tokens_code": asst_total_tok_code,
            "overall_code_share_pct": round(asst_code_share, 2),
            "messages_with_code": n_asst_with_code,
            "pct_messages_with_code": round(pct_asst_with_code, 2),
            "buckets": {lbl: n for lbl, n, _ in counts_code},
        },
        "voice_note_messages": n_voice,
        "headline": (
            f"human medyan ~{_fmt_token(stats_tokens['human']['p50'])} token, "
            f"assistant medyan ~{_fmt_token(stats_tokens['assistant']['p50'])} token; "
            f"toplam token hacminin %{share_a:.1f}'i assistant; "
            f"assistant cevabının %{asst_code_share:.1f}'i kod bloğu."
        ),
    }
    _common.write_json(out_dir, "summary.json", summary)

    token_table = "\n".join([
        _common.percentile_table(stats_tokens["human"], "human tokens"),
        "",
        _common.percentile_table(stats_tokens["assistant"], "assistant tokens"),
    ])

    # Öne çıkanlar — sadece sayı interpolation
    highlights = (
        f"- Human: **{_common.fmt_int(totals['human'])}** mesaj "
        f"(tool-only: {_common.fmt_int(tool_only['human'])})\n"
        f"- Assistant: **{_common.fmt_int(totals['assistant'])}** mesaj "
        f"(tool-only: {_common.fmt_int(tool_only['assistant'])})\n"
        f"- Medyan: human **~{_fmt_token(stats_tokens['human']['p50'])} token**, "
        f"assistant **~{_fmt_token(stats_tokens['assistant']['p50'])} token**\n"
        f"- p95: human **~{_fmt_token(stats_tokens['human']['p95'])} token**, "
        f"assistant **~{_fmt_token(stats_tokens['assistant']['p95'])} token**\n"
        f"- Max: human **~{_fmt_token(stats_tokens['human']['max'])} token**, "
        f"assistant **~{_fmt_token(stats_tokens['assistant']['max'])} token**\n"
        f"- Token hacmi: human **~{_fmt_token(total_tok_h)}** (%{share_h:.1f}), "
        f"assistant **~{_fmt_token(total_tok_a)}** (%{share_a:.1f}) "
        f"— oran **{ratio_a_to_h:.1f}×**\n"
        f"- Assistant cevabının **%{asst_code_share:.1f}**'i kod bloğu; "
        f"mesajların **%{pct_asst_with_code:.1f}**'i ({_common.fmt_int(n_asst_with_code)}) "
        f"en az bir kod bloğu içeriyor"
    )

    # İlk vs takip kıyas tablosu
    fvf_table = _common.markdown_table(
        [
            ["human ilk mesaj",
             _common.fmt_int(stats_first["n"]),
             _fmt_token(stats_first["p50"]),
             _fmt_token(stats_first["p95"]),
             _fmt_token(stats_first["max"])],
            ["human takip mesajı",
             _common.fmt_int(stats_followup["n"]),
             _fmt_token(stats_followup["p50"]),
             _fmt_token(stats_followup["p95"]),
             _fmt_token(stats_followup["max"])],
        ],
        headers=["kitle", "n", "p50", "p95", "max"],
    )

    # ── Structured blocks ──
    highlight_items = [
        f"Human: **{_common.fmt_int(totals['human'])}** mesaj "
        f"(tool-only: {_common.fmt_int(tool_only['human'])})",
        f"Assistant: **{_common.fmt_int(totals['assistant'])}** mesaj "
        f"(tool-only: {_common.fmt_int(tool_only['assistant'])})",
        f"Medyan: human **~{_fmt_token(stats_tokens['human']['p50'])} token**, "
        f"assistant **~{_fmt_token(stats_tokens['assistant']['p50'])} token**",
        f"p95: human **~{_fmt_token(stats_tokens['human']['p95'])} token**, "
        f"assistant **~{_fmt_token(stats_tokens['assistant']['p95'])} token**",
        f"Max: human **~{_fmt_token(stats_tokens['human']['max'])} token**, "
        f"assistant **~{_fmt_token(stats_tokens['assistant']['max'])} token**",
        f"Token hacmi: human **~{_fmt_token(total_tok_h)}** (%{share_h:.1f}), "
        f"assistant **~{_fmt_token(total_tok_a)}** (%{share_a:.1f}) "
        f"— oran **{ratio_a_to_h:.1f}×**",
        f"Assistant cevabının **%{asst_code_share:.1f}**'i kod bloğu; "
        f"mesajların **%{pct_asst_with_code:.1f}**'i ({_common.fmt_int(n_asst_with_code)}) "
        f"en az bir kod bloğu içeriyor",
    ]

    # Quantile — human + assistant in a single table, two rows
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
    def _quantile_row(label: str, s: dict) -> list:
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
    quantile_rows = [
        _quantile_row("human tokens", stats_tokens["human"]),
        _quantile_row("assistant tokens", stats_tokens["assistant"]),
    ]

    def _buckets_payload(counts: list[tuple[str, int, float]]) -> list[dict]:
        return [{"label": lbl, "count": n, "pct": pct} for lbl, n, pct in counts]

    bucket_table_columns = [
        {"key": "range", "label": "token aralığı", "align": "left"},
        {"key": "n",     "label": "n",             "align": "right"},
        {"key": "pct",   "label": "oran",          "align": "right"},
    ]

    # İlk vs takip — özet tablo + iki bucket tablosu tek section'da
    fvf_columns = [
        {"key": "kitle", "label": "kitle", "align": "left"},
        {"key": "n",     "label": "n",     "align": "right"},
        {"key": "p50",   "label": "p50",   "align": "right"},
        {"key": "p95",   "label": "p95",   "align": "right"},
        {"key": "max",   "label": "max",   "align": "right"},
    ]
    fvf_rows = [
        ["human ilk mesaj",
         _common.fmt_int(stats_first["n"]),
         _fmt_token(stats_first["p50"]),
         _fmt_token(stats_first["p95"]),
         _fmt_token(stats_first["max"])],
        ["human takip mesajı",
         _common.fmt_int(stats_followup["n"]),
         _fmt_token(stats_followup["p50"]),
         _fmt_token(stats_followup["p95"]),
         _fmt_token(stats_followup["max"])],
    ]

    code_share_columns = [
        {"key": "band", "label": "kod payı", "align": "left"},
        {"key": "n",    "label": "n",        "align": "right"},
        {"key": "pct",  "label": "oran",     "align": "right"},
    ]

    notes_items = [
        "Token sayımı **sadece** `content_blocks.type='text'` (ve voice_note) için "
        "yapılır; thinking blokları m03'te ele alınır.",
        "Tool-only mesaj = `chars_text=0` — yalnız tool_use / tool_result bloğu "
        "taşıyan mesajlar; dağılım hesabından dışlanır.",
        "Kod bloğu = üçlü-backtick ile çevrili fenced block; kapanışı olmayan "
        "bloklar metin sonuna kadar sayılır. Inline backtick (`` ` ``) dahil değildir.",
        f"Voice_note içeren mesaj sayısı: **{n_voice}** — örnekleme istatistik "
        "için çok küçük; ayrı analiz yapılmamıştır.",
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
            "Token dağılımı — quantile (tokens_text>0)",
            token_table,
            blocks=[_common.block_table(quantile_columns, quantile_rows)],
        ),
        _common.Section(
            "Token kovaları — human",
            _common.markdown_table(_bucket_table(counts_h), headers=["token aralığı", "n", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    label=f"human — kova dağılımı (n={_common.fmt_int(tk_h.size)})",
                    buckets=_buckets_payload(counts_h),
                    image="bucket_tokens.png",
                    xlabel="token (aralık)",
                ),
                _common.block_table(
                    bucket_table_columns,
                    [[lbl, _common.fmt_int(n), f"%{pct:.1f}"] for lbl, n, pct in counts_h],
                ),
            ],
        ),
        _common.Section(
            "Token kovaları — assistant",
            _common.markdown_table(_bucket_table(counts_a), headers=["token aralığı", "n", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    label=f"assistant — kova dağılımı (n={_common.fmt_int(tk_a.size)})",
                    buckets=_buckets_payload(counts_a),
                    image="bucket_tokens.png",
                    xlabel="token (aralık)",
                ),
                _common.block_table(
                    bucket_table_columns,
                    [[lbl, _common.fmt_int(n), f"%{pct:.1f}"] for lbl, n, pct in counts_a],
                ),
            ],
        ),
        _common.Section(
            "İlk mesaj vs takip mesajı (human)",
            fvf_table
            + "\n\n"
            + _common.markdown_table(_bucket_table(counts_first),
                                     headers=["token aralığı", "n (ilk)", "oran"])
            + "\n\n"
            + _common.markdown_table(_bucket_table(counts_followup),
                                     headers=["token aralığı", "n (takip)", "oran"]),
            blocks=[
                _common.block_table(fvf_columns, fvf_rows),
                _common.block_bucket_chart(
                    label=f"human ilk mesaj — kova dağılımı (n={_common.fmt_int(fvf['first'].size)})",
                    buckets=_buckets_payload(counts_first),
                    image="bucket_tokens_first_vs_followup.png",
                    xlabel="token (aralık)",
                ),
                _common.block_bucket_chart(
                    label=f"human takip mesajı — kova dağılımı (n={_common.fmt_int(fvf['followup'].size)})",
                    buckets=_buckets_payload(counts_followup),
                    xlabel="token (aralık)",
                ),
            ],
        ),
        _common.Section(
            "Assistant cevabında kod bloğu payı",
            f"- Toplam kod token: **{_common.fmt_int(asst_total_tok_code)}** / "
            f"toplam text token: **{_common.fmt_int(asst_total_tok_text)}** "
            f"→ **%{asst_code_share:.1f}**\n"
            f"- Kod bloğu içeren assistant mesajı: "
            f"**{_common.fmt_int(n_asst_with_code)}** (**%{pct_asst_with_code:.1f}**)\n\n"
            + _common.markdown_table(_bucket_table(counts_code),
                                     headers=["kod payı", "n", "oran"]),
            blocks=[
                _common.block_bullets([
                    f"Toplam kod token: **{_common.fmt_int(asst_total_tok_code)}** / "
                    f"toplam text token: **{_common.fmt_int(asst_total_tok_text)}** "
                    f"→ **%{asst_code_share:.1f}**",
                    f"Kod bloğu içeren assistant mesajı: "
                    f"**{_common.fmt_int(n_asst_with_code)}** (**%{pct_asst_with_code:.1f}**)",
                ]),
                _common.block_bucket_chart(
                    label=f"Assistant cevabında kod bloğu payı (n={_common.fmt_int(asst_nonzero.shape[0])})",
                    buckets=_buckets_payload(counts_code),
                    image="assistant_code_share.png",
                    xlabel="cevabın ne kadarı kod (token oranı)",
                ),
                _common.block_table(
                    code_share_columns,
                    [[lbl, _common.fmt_int(n), f"%{pct:.1f}"] for lbl, n, pct in counts_code],
                ),
            ],
        ),
        _common.Section(
            "Grafikler",
            "- `bucket_tokens.png` — sender bazında mesaj token kova dağılımı "
            "(üstte human, altta assistant)\n"
            "- `bucket_tokens_first_vs_followup.png` — human ilk mesaj vs takip mesajı "
            "kova dağılımı\n"
            "- `assistant_code_share.png` — assistant cevaplarında kod bloğu payı "
            "kova dağılımı",
            blocks=[],
        ),
        _common.Section(
            "Notlar",
            "- Token sayımı **sadece** `content_blocks.type='text'` (ve voice_note) için "
            "yapılır; thinking blokları m03'te ele alınır.\n"
            "- Tool-only mesaj = `chars_text=0` — yalnız tool_use / tool_result bloğu "
            "taşıyan mesajlar; dağılım hesabından dışlanır.\n"
            "- Kod bloğu = üçlü-backtick ile çevrili fenced block; kapanışı olmayan "
            "bloklar metin sonuna kadar sayılır. Inline backtick (`` ` ``) dahil değildir.\n"
            f"- Voice_note içeren mesaj sayısı: **{n_voice}** — örnekleme istatistik "
            "için çok küçük; ayrı analiz yapılmamıştır.\n"
            "- Token sayımı `tiktoken.cl100k_base` (OpenAI) ile yapılmıştır; Claude'un "
            "gerçek tokenizer'ı değildir — **yaklaşık üst sınır** olarak okunmalıdır.",
            blocks=[_common.block_bullets(notes_items)],
        ),
    ]
    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)
    return summary
