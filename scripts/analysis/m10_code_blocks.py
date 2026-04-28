"""
m10 — Kod bloğu profili

- Fenced block (triple-backtick) sayımı, dil dağılımı, blok boyut dağılımı
- Human vs assistant kod kullanım ayrımı
- Inline backtick kullanımı
- Konuşma başına kod bloğu sayısı + kod token yoğunluğu
- Kod-yoğun konuşma profili (konuşma kod payı × mesaj sayısı çapraz)

Kaynak: _stats_code_block (her fenced block bir satır) + _stats_message.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from analysis import _common
from analysis._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "10-code-blocks"
TITLE = "Kod bloğu profili"

BLOCK_LEN_BUCKETS = [
    ("<100",      0,     100),
    ("100–500",   100,   500),
    ("500–2 K",   500,   2_000),
    ("2–10 K",    2_000, 10_000),
    ("10 K+",     10_000, 10 ** 9),
]

BLOCKS_PER_CONVO_BUCKETS = [
    ("1",      1,  2),
    ("2–5",    2,  6),
    ("6–20",   6,  21),
    ("21–100", 21, 101),
    ("100+",   101, 10 ** 9),
]

CODE_SHARE_BUCKETS = [  # konuşma-bazında tokens_code / tokens_text payı
    ("%0",          0.0,   1e-9),   # kod yok
    (">%0-%10",     1e-9,  0.10),
    ("%10-%30",     0.10,  0.30),
    ("%30-%60",     0.30,  0.60),
    ("%60+",        0.60,  1.01),
]

# Dil etiketlerini normalize et (case + yaygın alias'lar)
LANG_ALIAS = {
    "ts": "typescript",
    "js": "javascript",
    "py": "python",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "cpp": "c++",
    "c#": "csharp",
    "cs": "csharp",
    "yml": "yaml",
    "md": "markdown",
    "txt": "(plain)",
    "plaintext": "(plain)",
    "text": "(plain)",
}


def _normalize_lang(raw: str) -> str:
    if raw is None:
        return "(boş)"
    s = raw.strip().lower()
    if not s:
        return "(boş)"
    # Bazı kayıtlarda "language=python" / "python info-string" geliyor;
    # ilk kelimeye düş
    first = s.split()[0]
    return LANG_ALIAS.get(first, first)


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
    # ---------- genel blok sayımı ----------
    totals = con.execute("""
        SELECT COUNT(*) n_blocks,
               COUNT(DISTINCT message_uuid) n_msgs_with_block,
               SUM(chars) total_chars,
               SUM(tokens) total_tokens
        FROM _stats_code_block
    """).fetchone()
    n_blocks = int(totals[0] or 0)
    n_msgs_with_block = int(totals[1] or 0)
    total_code_chars = int(totals[2] or 0)
    total_code_tokens = int(totals[3] or 0)

    # ---------- human vs assistant blok ayrımı ----------
    sender_breakdown = con.execute("""
        SELECT m.sender,
               COUNT(*) AS n_blocks,
               SUM(cb.tokens) AS tokens,
               COUNT(DISTINCT cb.message_uuid) AS n_msgs
        FROM _stats_code_block cb
        JOIN _stats_message m ON m.message_uuid = cb.message_uuid
        WHERE m.sender IN ('human','assistant')
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    sender_map: dict[str, tuple[int, int, int]] = {
        r[0]: (int(r[1] or 0), int(r[2] or 0), int(r[3] or 0))
        for r in sender_breakdown
    }
    h_blocks, h_tokens, h_msgs = sender_map.get("human", (0, 0, 0))
    a_blocks, a_tokens, a_msgs = sender_map.get("assistant", (0, 0, 0))

    # Sender başına toplam mesaj (non-empty text)
    msg_totals = con.execute("""
        SELECT sender, COUNT(*) AS n
        FROM _stats_message
        WHERE sender IN ('human','assistant') AND chars_text > 0
        GROUP BY 1
    """).fetchall()
    msg_total_map = {r[0]: int(r[1]) for r in msg_totals}
    h_total_msg = msg_total_map.get("human", 0)
    a_total_msg = msg_total_map.get("assistant", 0)

    # ---------- blok boyut dağılımı ----------
    chars_rows = con.execute("""
        SELECT chars FROM _stats_code_block WHERE chars > 0
    """).fetchall()
    block_chars = np.array([r[0] for r in chars_rows], dtype=float)
    stats_block_chars = _common.percentiles(block_chars)
    block_len_counts = _bucket_counts(block_chars, BLOCK_LEN_BUCKETS)

    # ---------- dil dağılımı ----------
    lang_raw = con.execute("""
        SELECT language, COUNT(*) AS n, SUM(tokens) AS toks
        FROM _stats_code_block
        GROUP BY 1
    """).fetchall()
    # Normalize + aggregate
    from collections import Counter
    lang_counter: Counter[str] = Counter()
    lang_tokens: Counter[str] = Counter()
    for raw, n, toks in lang_raw:
        norm = _normalize_lang(raw)
        lang_counter[norm] += int(n)
        lang_tokens[norm] += int(toks or 0)

    lang_ranked = lang_counter.most_common(20)

    # ---------- konuşma başına blok sayısı ----------
    per_convo_rows = con.execute("""
        SELECT m.conversation_uuid, COUNT(*) AS n
        FROM _stats_code_block cb
        JOIN _stats_message m ON m.message_uuid = cb.message_uuid
        GROUP BY 1
    """).fetchall()
    per_convo_arr = np.array([r[1] for r in per_convo_rows], dtype=float)
    stats_per_convo = _common.percentiles(per_convo_arr)
    blocks_per_convo_counts = _bucket_counts(per_convo_arr, BLOCKS_PER_CONVO_BUCKETS)
    n_convos_with_code = int(per_convo_arr.size)

    # ---------- kod-yoğun konuşma (konuşma bazında tokens_code / tokens_text) ----------
    # _stats_message'da tokens_code var → konuşma bazında SUM
    share_rows = con.execute("""
        SELECT
            m.conversation_uuid,
            COALESCE(SUM(m.tokens_code), 0) AS code_toks,
            COALESCE(SUM(m.tokens_text), 0) AS text_toks,
            c.message_count
        FROM _stats_message m
        JOIN conversations c ON c.uuid = m.conversation_uuid
        WHERE m.sender = 'assistant'
        GROUP BY m.conversation_uuid, c.message_count
    """).fetchall()
    share_vals = []
    msg_count_by_bucket: dict[int, list[int]] = {i: [] for i in range(len(CODE_SHARE_BUCKETS))}
    for _uuid, code_t, text_t, msg_cnt in share_rows:
        if text_t <= 0:
            continue
        share = code_t / text_t
        share_vals.append(share)
        # hangi kova
        for i, (_lbl, lo, hi) in enumerate(CODE_SHARE_BUCKETS):
            if lo <= share < hi:
                msg_count_by_bucket[i].append(int(msg_cnt))
                break
    share_arr = np.array(share_vals, dtype=float)
    share_bucket_rows = []
    for i, (lbl, _lo, _hi) in enumerate(CODE_SHARE_BUCKETS):
        bucket_msgs = msg_count_by_bucket[i]
        n = len(bucket_msgs)
        pct = 100.0 * n / share_arr.size if share_arr.size else 0.0
        if bucket_msgs:
            p50 = int(np.median(bucket_msgs))
            p95 = int(np.quantile(bucket_msgs, 0.95))
        else:
            p50, p95 = 0, 0
        share_bucket_rows.append([
            lbl, _common.fmt_int(n), f"%{pct:.1f}",
            _common.fmt_int(p50), _common.fmt_int(p95),
        ])

    # ---------- inline backtick ----------
    inline_totals = con.execute("""
        SELECT sender,
               SUM(inline_code_count) AS total_inline,
               SUM(CASE WHEN inline_code_count > 0 THEN 1 ELSE 0 END) AS msgs_with_inline,
               COUNT(*) AS total_msgs
        FROM _stats_message
        WHERE sender IN ('human','assistant')
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    inline_rows = []
    for sender, total_inline, msgs_with_inline, total_msgs in inline_totals:
        pct = 100.0 * (msgs_with_inline or 0) / (total_msgs or 1)
        inline_rows.append([
            sender,
            _common.fmt_int(total_msgs),
            _common.fmt_int(msgs_with_inline),
            f"%{pct:.1f}",
            _common.fmt_int(total_inline),
        ])

    # ---------- grafikler ----------
    if HAS_MPL:
        # 1) block length buckets
        fig, ax = plt.subplots()
        _bucket_bar(
            ax, [b[0] for b in BLOCK_LEN_BUCKETS], block_len_counts,
            "#4C72B0", "blok sayısı", "Fenced kod bloğu boyut kovaları (karakter)",
        )
        _common.save_fig(fig, out_dir / "bucket_block_length.png")

        # 2) top 15 dil bar
        fig, ax = plt.subplots()
        top15 = lang_counter.most_common(15)
        if top15:
            names = [n for n, _ in top15]
            counts = [c for _, c in top15]
            y = np.arange(len(names))
            ax.barh(y, counts, color="#55A868")
            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel("blok sayısı")
            ax.set_title("Kod bloğu dil dağılımı (top 15)")
            for i, c in enumerate(counts):
                ax.text(c, i, f" {c}", va="center", fontsize=9)
        _common.save_fig(fig, out_dir / "top_languages.png")

        # 3) konuşma başına blok kova
        fig, ax = plt.subplots()
        _bucket_bar(
            ax, [b[0] for b in BLOCKS_PER_CONVO_BUCKETS], blocks_per_convo_counts,
            "#8172B2", "konuşma sayısı", "Konuşma başına kod bloğu sayısı",
        )
        _common.save_fig(fig, out_dir / "bucket_blocks_per_conversation.png")

        # 4) kod payı kova × mesaj
        fig, ax = plt.subplots()
        labels_s = [r[0] for r in share_bucket_rows]
        counts_s = [int(r[1].replace(" ", "")) for r in share_bucket_rows]
        _bucket_bar(
            ax, labels_s, counts_s,
            "#C44E52", "konuşma sayısı",
            "Konuşma kod payı kovaları (assistant tokens_code / tokens_text)",
        )
        _common.save_fig(fig, out_dir / "bucket_code_share.png")

        # 5) sender kıyas — basit 2 bar
        fig, ax = plt.subplots()
        senders = ["human", "assistant"]
        blocks_by_sender = [h_blocks, a_blocks]
        msgs_by_sender = [h_msgs, a_msgs]
        x = np.arange(2)
        w = 0.38
        ax.bar(x - w/2, blocks_by_sender, w, label="blok sayısı", color="#4C72B0")
        ax.bar(x + w/2, msgs_by_sender, w, label="bloklu mesaj", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(senders)
        ax.set_ylabel("adet")
        ax.set_title("Human vs assistant — kod bloğu kullanımı")
        ax.legend()
        for i, v in enumerate(blocks_by_sender):
            ax.text(i - w/2, v, f"{v}", ha="center", va="bottom", fontsize=9)
        for i, v in enumerate(msgs_by_sender):
            ax.text(i + w/2, v, f"{v}", ha="center", va="bottom", fontsize=9)
        _common.save_fig(fig, out_dir / "human_vs_assistant.png")

    # ---------- CSV ----------
    _common.write_csv(
        out_dir, "languages.csv",
        [(n, int(c), int(lang_tokens[n])) for n, c in lang_ranked],
        headers=["language", "blocks", "tokens"],
    )
    _common.write_csv(
        out_dir, "blocks_per_conversation.csv",
        [(uuid, int(n)) for uuid, n in per_convo_rows],
        headers=["conversation_uuid", "code_blocks"],
    )
    _common.write_csv(
        out_dir, "block_sizes.csv",
        [(int(c),) for c in block_chars],
        headers=["chars"],
    )

    # ---------- tablolar ----------
    block_len_rows = []
    for (lbl, _lo, _hi), c in zip(BLOCK_LEN_BUCKETS, block_len_counts):
        p = 100.0 * c / n_blocks if n_blocks else 0.0
        block_len_rows.append([lbl, _common.fmt_int(c), f"%{p:.1f}"])

    bpc_rows = []
    for (lbl, _lo, _hi), c in zip(BLOCKS_PER_CONVO_BUCKETS, blocks_per_convo_counts):
        p = 100.0 * c / n_convos_with_code if n_convos_with_code else 0.0
        bpc_rows.append([lbl, _common.fmt_int(c), f"%{p:.1f}"])

    lang_rows = []
    top15_tokens_total = sum(lang_tokens[n] for n, _ in lang_ranked[:15]) or 1
    for name, c in lang_ranked[:15]:
        toks = lang_tokens[name]
        pct_blocks = 100.0 * c / n_blocks if n_blocks else 0.0
        pct_tokens = 100.0 * toks / total_code_tokens if total_code_tokens else 0.0
        lang_rows.append([
            name,
            _common.fmt_int(c),
            f"%{pct_blocks:.1f}",
            _common.fmt_int(toks),
            f"%{pct_tokens:.1f}",
        ])

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

    def _int_row(label: str, st: dict) -> list[str]:
        return [
            label,
            _common.fmt_int(st["n"]),
            _common.fmt_int(int(st["min"] or 0)),
            _common.fmt_int(int(st["p50"] or 0)),
            _common.fmt_int(int(st["p90"] or 0)),
            _common.fmt_int(int(st["p95"] or 0)),
            _common.fmt_int(int(st["p99"] or 0)),
            _common.fmt_int(int(st["max"] or 0)),
        ]

    headers_pct = ["metric", "n", "min", "p50", "p90", "p95", "p99", "max"]
    block_len_tbl = _common.markdown_table([_chars_row("blok_chars", stats_block_chars)], headers=headers_pct)
    per_convo_tbl = _common.markdown_table([_int_row("blok/konuşma", stats_per_convo)], headers=headers_pct)

    sender_rows = []
    for sender, n_tot, n_with, pct, n_msg in inline_rows:
        sender_rows.append([sender, n_tot, n_with, pct, n_msg])

    # Block-sender kıyas tablosu (fenced)
    fenced_sender_rows = [
        [
            "human",
            _common.fmt_int(h_total_msg),
            _common.fmt_int(h_msgs),
            f"%{100.0 * h_msgs / h_total_msg:.1f}" if h_total_msg else "—",
            _common.fmt_int(h_blocks),
            _common.fmt_int(h_tokens),
        ],
        [
            "assistant",
            _common.fmt_int(a_total_msg),
            _common.fmt_int(a_msgs),
            f"%{100.0 * a_msgs / a_total_msg:.1f}" if a_total_msg else "—",
            _common.fmt_int(a_blocks),
            _common.fmt_int(a_tokens),
        ],
    ]

    # ---------- summary ----------
    summary = {
        "n_fenced_blocks": n_blocks,
        "n_messages_with_block": n_msgs_with_block,
        "total_code_chars": total_code_chars,
        "total_code_tokens": total_code_tokens,
        "human": {"blocks": h_blocks, "tokens": h_tokens, "msgs": h_msgs},
        "assistant": {"blocks": a_blocks, "tokens": a_tokens, "msgs": a_msgs},
        "n_conversations_with_code": n_convos_with_code,
        "block_chars_percentiles": stats_block_chars,
        "blocks_per_conversation_percentiles": stats_per_convo,
        "top_languages": [
            {"language": n, "blocks": int(c), "tokens": int(lang_tokens[n])}
            for n, c in lang_ranked[:15]
        ],
        "headline": (
            f"{_common.fmt_int(n_blocks)} fenced blok / "
            f"{_common.fmt_int(n_msgs_with_block)} mesaj / "
            f"{_common.fmt_int(n_convos_with_code)} konuşma; "
            f"blok medyan {_fmt_chars(stats_block_chars['p50'])}, "
            f"p95 {_fmt_chars(stats_block_chars['p95'])}; "
            f"en sık dil: {lang_ranked[0][0] if lang_ranked else '—'}."
        ),
    }
    _common.write_json(out_dir, "summary.json", summary)

    # ---------- report ----------
    pct_a_with = 100.0 * a_msgs / a_total_msg if a_total_msg else 0.0
    pct_h_with = 100.0 * h_msgs / h_total_msg if h_total_msg else 0.0

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

    block_len_buckets_data = [
        {"label": lbl, "count": int(c), "pct": (100.0 * c / n_blocks) if n_blocks else 0.0}
        for (lbl, _lo, _hi), c in zip(BLOCK_LEN_BUCKETS, block_len_counts)
    ]
    bpc_buckets_data = [
        {"label": lbl, "count": int(c), "pct": (100.0 * c / n_convos_with_code) if n_convos_with_code else 0.0}
        for (lbl, _lo, _hi), c in zip(BLOCKS_PER_CONVO_BUCKETS, blocks_per_convo_counts)
    ]
    lang_buckets_data = [
        {"label": name, "count": int(c), "pct": (100.0 * c / n_blocks) if n_blocks else 0.0}
        for name, c in lang_ranked[:15]
    ]
    share_buckets_data = []
    for i, (lbl, _lo, _hi) in enumerate(CODE_SHARE_BUCKETS):
        bucket_msgs = msg_count_by_bucket[i]
        n = len(bucket_msgs)
        pct = 100.0 * n / share_arr.size if share_arr.size else 0.0
        share_buckets_data.append({"label": lbl, "count": n, "pct": pct})

    sections = [
        _common.Section(
            "Öne çıkanlar",
            f"- Toplam fenced kod bloğu: **{_common.fmt_int(n_blocks)}** "
            f"(**{_common.fmt_int(n_msgs_with_block)}** mesajda, "
            f"**{_common.fmt_int(n_convos_with_code)}** konuşmada)\n"
            f"- Toplam kod token: **{_common.fmt_int(total_code_tokens)}**, "
            f"toplam kod karakter: **{_fmt_chars(total_code_chars)}**\n"
            f"- Assistant: **{_common.fmt_int(a_blocks)}** blok / "
            f"**{_common.fmt_int(a_msgs)}** mesaj ("
            f"assistant mesajlarının **%{pct_a_with:.1f}**'inde en az bir blok)\n"
            f"- Human: **{_common.fmt_int(h_blocks)}** blok / "
            f"**{_common.fmt_int(h_msgs)}** mesaj ("
            f"human mesajlarının **%{pct_h_with:.1f}**'inde en az bir blok)\n"
            f"- Blok boyutu: medyan **{_fmt_chars(stats_block_chars['p50'])}**, "
            f"p95 **{_fmt_chars(stats_block_chars['p95'])}**, "
            f"max **{_fmt_chars(stats_block_chars['max'])}**\n"
            f"- Konuşma başına blok: medyan **{int(stats_per_convo['p50'] or 0)}**, "
            f"p95 **{int(stats_per_convo['p95'] or 0)}**, "
            f"max **{int(stats_per_convo['max'] or 0)}**\n"
            f"- En sık dil: **{lang_ranked[0][0]}** ({_common.fmt_int(lang_ranked[0][1])} blok)"
            if lang_ranked else "- Dil: veri yok",
            blocks=[
                _common.block_bullets([
                    f"Toplam fenced kod bloğu: **{_common.fmt_int(n_blocks)}** "
                    f"(**{_common.fmt_int(n_msgs_with_block)}** mesajda, "
                    f"**{_common.fmt_int(n_convos_with_code)}** konuşmada)",
                    f"Toplam kod token: **{_common.fmt_int(total_code_tokens)}**, "
                    f"toplam kod karakter: **{_fmt_chars(total_code_chars)}**",
                    f"Assistant: **{_common.fmt_int(a_blocks)}** blok / "
                    f"**{_common.fmt_int(a_msgs)}** mesaj "
                    f"(assistant mesajlarının **%{pct_a_with:.1f}**'inde en az bir blok)",
                    f"Human: **{_common.fmt_int(h_blocks)}** blok / "
                    f"**{_common.fmt_int(h_msgs)}** mesaj "
                    f"(human mesajlarının **%{pct_h_with:.1f}**'inde en az bir blok)",
                    f"Blok boyutu: medyan **{_fmt_chars(stats_block_chars['p50'])}**, "
                    f"p95 **{_fmt_chars(stats_block_chars['p95'])}**, "
                    f"max **{_fmt_chars(stats_block_chars['max'])}**",
                    f"Konuşma başına blok: medyan **{int(stats_per_convo['p50'] or 0)}**, "
                    f"p95 **{int(stats_per_convo['p95'] or 0)}**, "
                    f"max **{int(stats_per_convo['max'] or 0)}**",
                    (f"En sık dil: **{lang_ranked[0][0]}** ({_common.fmt_int(lang_ranked[0][1])} blok)"
                     if lang_ranked else "Dil: veri yok"),
                ]),
            ],
        ),
        _common.Section(
            "Human vs assistant — fenced blok",
            _common.markdown_table(
                fenced_sender_rows,
                headers=["sender", "toplam text-mesaj", "bloklu mesaj", "oran", "blok sayısı", "kod token"],
            ),
            blocks=[
                _common.block_table(
                    [
                        {"key": "sender",    "label": "sender",           "align": "left"},
                        {"key": "total",     "label": "toplam text-mesaj","align": "right"},
                        {"key": "with",      "label": "bloklu mesaj",     "align": "right"},
                        {"key": "rate",      "label": "oran",             "align": "right"},
                        {"key": "blocks",    "label": "blok sayısı",      "align": "right"},
                        {"key": "tokens",    "label": "kod token",        "align": "right"},
                    ],
                    fenced_sender_rows,
                ),
                _common.block_grouped_bar_chart(
                    "Human vs assistant — kod bloğu kullanımı",
                    [
                        {"sender": "human", "bloklu_mesaj": h_msgs, "blok_sayisi": h_blocks, "kod_token": h_tokens},
                        {"sender": "assistant", "bloklu_mesaj": a_msgs, "blok_sayisi": a_blocks, "kod_token": a_tokens},
                    ],
                    [
                        {"key": "bloklu_mesaj", "label": "bloklu mesaj"},
                        {"key": "blok_sayisi", "label": "blok sayısı"},
                        {"key": "kod_token", "label": "kod token"},
                    ],
                    x_key="sender",
                ),
            ],
        ),
        _common.Section(
            "Fenced blok boyut dağılımı (karakter)",
            block_len_tbl,
            blocks=[
                _common.block_table(pct_cols, [_chars_row("blok_chars", stats_block_chars)]),
            ],
        ),
        _common.Section(
            "Blok boyut kovaları",
            _common.markdown_table(block_len_rows, headers=["boyut aralığı", "blok", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    "Fenced kod bloğu boyut kovaları (karakter)",
                    block_len_buckets_data,
                    image="bucket_block_length.png",
                    xlabel="boyut aralığı",
                ),
                _common.block_table(
                    [
                        {"key": "bucket", "label": "boyut aralığı", "align": "left"},
                        {"key": "count",  "label": "blok",         "align": "right"},
                        {"key": "pct",    "label": "oran",         "align": "right"},
                    ],
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"] for b in block_len_buckets_data],
                ),
            ],
        ),
        _common.Section(
            "Dil dağılımı (top 15)",
            _common.markdown_table(
                lang_rows,
                headers=["language", "blok", "blok oran", "kod token", "token oran"],
            ),
            blocks=[
                _common.block_bucket_chart(
                    "Kod bloğu dil dağılımı (top 15)",
                    lang_buckets_data,
                    image="top_languages.png",
                    xlabel="language",
                ),
                _common.block_table(
                    [
                        {"key": "language",   "label": "language",   "align": "left"},
                        {"key": "blocks",     "label": "blok",       "align": "right"},
                        {"key": "block_pct",  "label": "blok oran",  "align": "right"},
                        {"key": "tokens",     "label": "kod token",  "align": "right"},
                        {"key": "token_pct",  "label": "token oran", "align": "right"},
                    ],
                    lang_rows,
                ),
            ] if lang_rows else [],
        ),
        _common.Section(
            "Konuşma başına fenced blok sayısı",
            per_convo_tbl,
            blocks=[
                _common.block_table(pct_cols, [_int_row("blok/konuşma", stats_per_convo)]),
            ],
        ),
        _common.Section(
            "Konuşma kovaları",
            _common.markdown_table(bpc_rows, headers=["blok aralığı", "konuşma", "oran"]),
            blocks=[
                _common.block_bucket_chart(
                    "Konuşma başına kod bloğu sayısı",
                    bpc_buckets_data,
                    image="bucket_blocks_per_conversation.png",
                    xlabel="blok aralığı",
                ),
                _common.block_table(
                    [
                        {"key": "bucket", "label": "blok aralığı", "align": "left"},
                        {"key": "count",  "label": "konuşma",     "align": "right"},
                        {"key": "pct",    "label": "oran",        "align": "right"},
                    ],
                    [[b["label"], _common.fmt_int(b["count"]), f"%{b['pct']:.1f}"] for b in bpc_buckets_data],
                ),
            ],
        ),
        _common.Section(
            "Konuşma kod payı × mesaj sayısı",
            _common.markdown_table(
                share_bucket_rows,
                headers=["kod payı", "konuşma", "oran", "mesaj p50", "mesaj p95"],
            ),
            blocks=[
                _common.block_bucket_chart(
                    "Konuşma kod payı kovaları (assistant tokens_code / tokens_text)",
                    share_buckets_data,
                    image="bucket_code_share.png",
                    xlabel="kod payı",
                ),
                _common.block_table(
                    [
                        {"key": "share",   "label": "kod payı",  "align": "left"},
                        {"key": "count",   "label": "konuşma",   "align": "right"},
                        {"key": "pct",     "label": "oran",      "align": "right"},
                        {"key": "msg_p50", "label": "mesaj p50", "align": "right"},
                        {"key": "msg_p95", "label": "mesaj p95", "align": "right"},
                    ],
                    share_bucket_rows,
                ),
            ],
        ),
        _common.Section(
            "Inline backtick kullanımı",
            _common.markdown_table(
                sender_rows,
                headers=["sender", "toplam mesaj", "inline'lı mesaj", "oran", "toplam inline"],
            ),
            blocks=[
                _common.block_table(
                    [
                        {"key": "sender",  "label": "sender",         "align": "left"},
                        {"key": "total",   "label": "toplam mesaj",   "align": "right"},
                        {"key": "with",    "label": "inline'lı mesaj","align": "right"},
                        {"key": "rate",    "label": "oran",           "align": "right"},
                        {"key": "inline",  "label": "toplam inline",  "align": "right"},
                    ],
                    sender_rows,
                ),
            ] if sender_rows else [],
        ),
        _common.Section(
            "Grafikler",
            "- `bucket_block_length.png` — blok boyut kovaları\n"
            "- `top_languages.png` — top 15 dil\n"
            "- `bucket_blocks_per_conversation.png` — konuşma başına blok kovaları\n"
            "- `bucket_code_share.png` — konuşma kod payı kovaları\n"
            "- `human_vs_assistant.png` — sender kıyası",
            blocks=[],
        ),
        _common.Section(
            "Notlar",
            "- Kaynak: `_stats_code_block` (her fenced block bir satır) + "
            "`_stats_message.inline_code_count` (inline backtick sayımı).\n"
            "- Fenced blok = satır başında `\\`\\`\\`[dil]` ile başlayan üçlü backtick bloğu; "
            "kapanışı olmayan bloklar metin sonuna kadar alınır.\n"
            "- Dil etiketi `\\`\\`\\`` sonrası açılış satırından alınır; boş kalırsa `(boş)` kovasına düşer. "
            f"Yaygın alias'lar ({len(LANG_ALIAS)} tane) normalize edilir: "
            "ts→typescript, py→python, sh/shell/zsh→bash, vs.\n"
            "- Inline backtick = tek-tırnaklı `foo`. Üçlü-backtick açılışıyla karışmasın diye "
            "`(?<!`)`(?!`)` lookaround kullanılır.\n"
            "- \"Konuşma kod payı\" sadece `sender='assistant'` mesajlarındaki "
            "`tokens_code / tokens_text` oranından hesaplanır — insan mesajları hariçtir.\n"
            "- Token sayımı `tiktoken.cl100k_base` (OpenAI); Claude tokenizer'ı değildir.",
            blocks=[
                _common.block_bullets([
                    "Kaynak: `_stats_code_block` (her fenced block bir satır) + "
                    "`_stats_message.inline_code_count` (inline backtick sayımı).",
                    "Fenced blok = satır başında ` ``` `[dil] ile başlayan üçlü backtick bloğu; "
                    "kapanışı olmayan bloklar metin sonuna kadar alınır.",
                    "Dil etiketi ` ``` ` sonrası açılış satırından alınır; boş kalırsa `(boş)` kovasına düşer. "
                    f"Yaygın alias'lar ({len(LANG_ALIAS)} tane) normalize edilir: "
                    "ts→typescript, py→python, sh/shell/zsh→bash, vs.",
                    "Inline backtick = tek-tırnaklı `foo`. Üçlü-backtick açılışıyla karışmasın diye "
                    "`(?<!`)`(?!`)` lookaround kullanılır.",
                    "\"Konuşma kod payı\" sadece `sender='assistant'` mesajlarındaki "
                    "`tokens_code / tokens_text` oranından hesaplanır — insan mesajları hariçtir.",
                    "Token sayımı `tiktoken.cl100k_base` (OpenAI); Claude tokenizer'ı değildir.",
                ]),
            ],
        ),
    ]
    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)
    return summary
