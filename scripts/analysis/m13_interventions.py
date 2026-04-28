"""
m13 — Konuşma müdahaleleri (prompt yeniden yazma, cevap yeniden üretme,
cevap in-place düzenleme, çok-root konuşmalar).

Kaynak: `_stats_message` (intervention flag kolonları), `messages`, `_stats_conversation`.
Çıktı: reports/quantitative/13-interventions/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import _common
from ._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "13-interventions"
TITLE = "Konuşma müdahaleleri (edit, fork, retry)"

FORK_PER_CONVO_BUCKETS: list[tuple[str, int, int]] = [
    ("1",      1, 1),
    ("2-5",    2, 5),
    ("6-20",   6, 20),
    ("20+",   21, 10**9),
]

INPLACE_DELTA_BUCKETS: list[tuple[str, float, float]] = [
    ("1-5dk",    60,        5*60),
    ("5-60dk",   5*60,      60*60),
    ("1-24sa",   60*60,     24*60*60),
    ("1-7gün",   24*60*60,  7*24*60*60),
    (">7gün",    7*24*60*60, float("inf")),
]


def _ensure_columns(con) -> None:
    cols = {r[0] for r in con.execute("DESCRIBE _stats_message").fetchall()}
    required = {"inplace_edit_flag", "fork_parent_flag", "fork_child_flag", "fork_child_continued", "edit_delta_seconds"}
    missing = required - cols
    if missing:
        raise SystemExit(
            f"m13: _stats_message eksik kolonlar: {sorted(missing)}. "
            "`--skip-prepare` olmadan çalıştır (yeni prepare gerekiyor)."
        )


def _bucketize_int(values: list[int], buckets: list[tuple[str, int, int]]) -> list[tuple[str, int]]:
    out = []
    for label, lo, hi in buckets:
        n = sum(1 for v in values if lo <= v <= hi)
        out.append((label, n))
    return out


def _bucketize_float(values: list[float], buckets: list[tuple[str, float, float]]) -> list[tuple[str, int]]:
    out = []
    for label, lo, hi in buckets:
        n = sum(1 for v in values if lo <= v < hi)
        out.append((label, n))
    return out


def run(con, out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    _ensure_columns(con)
    out_dir.mkdir(parents=True, exist_ok=True)
    sections: list[_common.Section] = []

    # --- Toplam rakamlar
    # NOT: fork_parent_flag parent mesajın sender'ına bakar. Children'ın sender'ı her zaman
    # parent'ın zıttı (mixed-sender doğrulanmış 0). Yani:
    #   parent=assistant → children=human  → kullanıcı prompt'u yeniden yazmış (human fork)
    #   parent=human     → children=assistant → kullanıcı retry basmış (assistant fork)
    # "human fork" etiketi çocukların eylemini (prompt rewrite) ifade eder.
    human_fork_points = con.execute(
        "SELECT COUNT(*) FROM _stats_message WHERE fork_parent_flag AND sender='assistant'"
    ).fetchone()[0]
    asst_fork_points = con.execute(
        "SELECT COUNT(*) FROM _stats_message WHERE fork_parent_flag AND sender='human'"
    ).fetchone()[0]
    inplace_edits = con.execute(
        "SELECT COUNT(*) FROM _stats_message WHERE inplace_edit_flag"
    ).fetchone()[0]

    human_fork_convs = con.execute(
        "SELECT COUNT(*) FROM _stats_conversation WHERE human_fork_count > 0"
    ).fetchone()[0]
    asst_fork_convs = con.execute(
        "SELECT COUNT(*) FROM _stats_conversation WHERE assistant_fork_count > 0"
    ).fetchone()[0]
    inplace_convs = con.execute(
        "SELECT COUNT(*) FROM _stats_conversation WHERE inplace_edit_count > 0"
    ).fetchone()[0]
    multi_root_convs = con.execute(
        "SELECT COUNT(*) FROM _stats_conversation WHERE root_count > 1"
    ).fetchone()[0]
    total_nonempty_convs = con.execute(
        "SELECT COUNT(*) FROM _stats_conversation WHERE message_count > 0"
    ).fetchone()[0]

    # Ekstra child sayımı: fork parent başına child-1 toplamı.
    # human_fork = children'ın sender'ı human olan fork'lar → parent sender = assistant
    extra_children_human = con.execute("""
        WITH ch AS (
          SELECT m.parent_message_uuid AS puuid, COUNT(*) AS c
          FROM messages m
          WHERE m.parent_message_uuid IS NOT NULL
            AND m.parent_message_uuid <> '00000000-0000-4000-8000-000000000000'
          GROUP BY 1
        )
        SELECT COALESCE(SUM(ch.c - 1), 0)
        FROM _stats_message sm
        JOIN ch ON ch.puuid = sm.message_uuid
        WHERE sm.fork_parent_flag AND sm.sender='assistant'
    """).fetchone()[0]

    extra_children_asst = con.execute("""
        WITH ch AS (
          SELECT m.parent_message_uuid AS puuid, COUNT(*) AS c
          FROM messages m
          WHERE m.parent_message_uuid IS NOT NULL
            AND m.parent_message_uuid <> '00000000-0000-4000-8000-000000000000'
          GROUP BY 1
        )
        SELECT COALESCE(SUM(ch.c - 1), 0)
        FROM _stats_message sm
        JOIN ch ON ch.puuid = sm.message_uuid
        WHERE sm.fork_parent_flag AND sm.sender='human'
    """).fetchone()[0]

    # --- Öne çıkanlar
    head = [
        f"- Prompt yeniden yazma (human fork): **{_common.fmt_int(extra_children_human)}** edit / **{_common.fmt_int(human_fork_points)}** edit noktası / **{_common.fmt_int(human_fork_convs)}** konuşma",
        f"- Cevap yeniden üretme (assistant fork / retry): **{_common.fmt_int(extra_children_asst)}** retry / **{_common.fmt_int(asst_fork_points)}** retry noktası / **{_common.fmt_int(asst_fork_convs)}** konuşma",
        f"- Cevap in-place düzenleme (assistant): **{_common.fmt_int(inplace_edits)}** mesaj / **{_common.fmt_int(inplace_convs)}** konuşma",
        f"- Çok-root konuşma (root_count > 1): **{_common.fmt_int(multi_root_convs)}** (toplam {_common.fmt_int(total_nonempty_convs)} boş olmayan konuşmadan)",
        "",
        "*Ayrım:* **edit noktası** = düzenlenen benzersiz prompt konumu; **edit** = toplam düzenleme eylemi (aynı noktada 3 kez yeniden yazıldıysa 3 edit sayılır). Retry için de aynı: **retry noktası** vs toplam **retry** eylemi.",
    ]
    sections.append(_common.Section(
        "Öne çıkanlar",
        "\n".join(head),
        blocks=[
            _common.block_bullets([
                f"Prompt yeniden yazma (human fork): **{_common.fmt_int(extra_children_human)}** edit / "
                f"**{_common.fmt_int(human_fork_points)}** edit noktası / "
                f"**{_common.fmt_int(human_fork_convs)}** konuşma",
                f"Cevap yeniden üretme (assistant fork / retry): **{_common.fmt_int(extra_children_asst)}** retry / "
                f"**{_common.fmt_int(asst_fork_points)}** retry noktası / "
                f"**{_common.fmt_int(asst_fork_convs)}** konuşma",
                f"Cevap in-place düzenleme (assistant): **{_common.fmt_int(inplace_edits)}** mesaj / "
                f"**{_common.fmt_int(inplace_convs)}** konuşma",
                f"Çok-root konuşma (root_count > 1): **{_common.fmt_int(multi_root_convs)}** "
                f"(toplam {_common.fmt_int(total_nonempty_convs)} boş olmayan konuşmadan)",
            ]),
            _common.block_paragraph(
                "*Ayrım:* **edit noktası** = düzenlenen benzersiz prompt konumu; "
                "**edit** = toplam düzenleme eylemi (aynı noktada 3 kez yeniden yazıldıysa 3 edit sayılır). "
                "Retry için de aynı: **retry noktası** vs toplam **retry** eylemi."
            ),
        ],
    ))

    how_body = (
        "Claude.ai'de bir konuşmaya üç farklı müdahale biçimi uygulanabilir. Export'ta bunlar "
        "birbirinden farklı izler bırakır — tek bir \"edit\" kolonu yoktur:\n\n"
        "- **Prompt yeniden yazma** (*human fork*): kullanıcı önceki asistan cevabının altındaki "
        "prompt'unu düzenler. Export'a **yeni `message_uuid`** yazılır, aynı asistan cevabı "
        "(`parent_message_uuid`) altına ikinci bir human child eklenir. Orijinal prompt silinmez; "
        "`updated_at` değişmez. Fork'un **parent**'ı bir asistan mesajıdır, **children** human.\n"
        "- **Cevap yeniden üretme** (*assistant fork* / retry): kullanıcı bir prompt'a verilen "
        "asistan cevabını yeniden üretmek için retry basar. Yine **yeni `message_uuid`**, aynı "
        "human prompt (`parent_message_uuid`) altına ikinci bir assistant child eklenir. Fork'un "
        "**parent**'ı bir human mesajıdır, **children** assistant.\n"
        "- **Cevap in-place düzenleme** (*assistant in-place edit*): aynı `message_uuid` içinde "
        "`updated_at` güncellenir ve `content` değişir. Streaming snapshot'ları eleme için "
        "`updated_at - created_at > 60s` eşiği uygulanır. Örneklem **yalnız assistant** mesajında bu "
        "davranışı gösteriyor; human mesajında (streaming dışında) ölçülmedi."
    )
    sections.append(_common.Section(
        "Nasıl çalışır — Claude.ai müdahale mekaniği",
        how_body,
        blocks=[
            _common.block_paragraph(
                "Claude.ai'de bir konuşmaya üç farklı müdahale biçimi uygulanabilir. "
                "Export'ta bunlar birbirinden farklı izler bırakır — tek bir \"edit\" kolonu yoktur:"
            ),
            _common.block_bullets([
                "**Prompt yeniden yazma** (*human fork*): kullanıcı önceki asistan cevabının altındaki "
                "prompt'unu düzenler. Export'a **yeni `message_uuid`** yazılır, aynı asistan cevabı "
                "(`parent_message_uuid`) altına ikinci bir human child eklenir. Orijinal prompt silinmez; "
                "`updated_at` değişmez. Fork'un **parent**'ı bir asistan mesajıdır, **children** human.",
                "**Cevap yeniden üretme** (*assistant fork* / retry): kullanıcı bir prompt'a verilen "
                "asistan cevabını yeniden üretmek için retry basar. Yine **yeni `message_uuid`**, aynı "
                "human prompt (`parent_message_uuid`) altına ikinci bir assistant child eklenir. "
                "Fork'un **parent**'ı bir human mesajıdır, **children** assistant.",
                "**Cevap in-place düzenleme** (*assistant in-place edit*): aynı `message_uuid` içinde "
                "`updated_at` güncellenir ve `content` değişir. Streaming snapshot'ları eleme için "
                "`updated_at - created_at > 60s` eşiği uygulanır. Örneklem **yalnız assistant** mesajında "
                "bu davranışı gösteriyor; human mesajında (streaming dışında) ölçülmedi.",
            ]),
        ],
    ))

    # --- Prompt yeniden yazma
    human_per_conv = [
        r[0] for r in con.execute(
            "SELECT human_fork_count FROM _stats_conversation WHERE human_fork_count > 0"
        ).fetchall()
    ]
    hbucks = _bucketize_int(human_per_conv, FORK_PER_CONVO_BUCKETS)
    hb_rows = [[lbl, _common.fmt_int(n), f"%{100.0*n/max(1,len(human_per_conv)):.1f}"] for lbl, n in hbucks]
    h_table = _common.markdown_table(hb_rows, headers=["edit noktası", "konuşma", "oran"])
    h_pct = _common.percentiles(human_per_conv)
    h_dist = _common.markdown_table(
        [[
            "edit noktası/konuşma",
            _common.fmt_int(h_pct["n"]),
            _common.fmt_int(h_pct["min"]),
            _common.fmt_int(h_pct["p50"]),
            _common.fmt_int(h_pct["p90"]),
            _common.fmt_int(h_pct["p95"]),
            _common.fmt_int(h_pct["p99"]),
            _common.fmt_int(h_pct["max"]),
        ]],
        headers=["metric", "n", "min", "p50", "p90", "p95", "p99", "max"],
    )

    # Human fork child continuation: hangi pozisyondaki child devam etmiş?
    # human fork = children sender=human → parent sender=assistant
    hfc_rows = con.execute("""
        WITH parents AS (
          SELECT sm.message_uuid AS puuid
          FROM _stats_message sm
          WHERE sm.fork_parent_flag AND sm.sender='assistant'
        ),
        children_ranked AS (
          SELECT m.parent_message_uuid AS puuid,
                 m.uuid AS cuuid,
                 ROW_NUMBER() OVER (PARTITION BY m.parent_message_uuid ORDER BY m.created_at, m.uuid) AS rn_first,
                 ROW_NUMBER() OVER (PARTITION BY m.parent_message_uuid ORDER BY m.created_at DESC, m.uuid DESC) AS rn_last,
                 COUNT(*)    OVER (PARTITION BY m.parent_message_uuid) AS ntot
          FROM messages m
          JOIN parents p ON p.puuid = m.parent_message_uuid
        ),
        continued AS (
          SELECT cr.*, CASE WHEN EXISTS (
                           SELECT 1 FROM messages m2
                           WHERE m2.parent_message_uuid = cr.cuuid
                         ) THEN TRUE ELSE FALSE END AS cont
          FROM children_ranked cr
        )
        SELECT
          SUM(CASE WHEN cont AND rn_first = 1 THEN 1 ELSE 0 END) AS first_cont,
          SUM(CASE WHEN cont AND rn_last  = 1 THEN 1 ELSE 0 END) AS last_cont,
          SUM(CASE WHEN cont AND rn_first <> 1 AND rn_last <> 1 THEN 1 ELSE 0 END) AS mid_cont,
          SUM(CASE WHEN cont THEN 1 ELSE 0 END)                   AS total_cont,
          COUNT(*)                                                AS total_children
        FROM continued
    """).fetchone()

    # Human fork: ilk vs son child chars_text farkı
    hfdelta = con.execute("""
        WITH parents AS (
          SELECT sm.message_uuid AS puuid
          FROM _stats_message sm
          WHERE sm.fork_parent_flag AND sm.sender='assistant'
        ),
        ranked AS (
          SELECT m.parent_message_uuid AS puuid,
                 m.uuid AS cuuid,
                 sm.chars_text,
                 ROW_NUMBER() OVER (PARTITION BY m.parent_message_uuid ORDER BY m.created_at, m.uuid) AS rn_first,
                 ROW_NUMBER() OVER (PARTITION BY m.parent_message_uuid ORDER BY m.created_at DESC, m.uuid DESC) AS rn_last
          FROM messages m
          JOIN parents p ON p.puuid = m.parent_message_uuid
          JOIN _stats_message sm ON sm.message_uuid = m.uuid
        ),
        firsts AS (SELECT puuid, chars_text AS first_chars FROM ranked WHERE rn_first = 1),
        lasts  AS (SELECT puuid, chars_text AS last_chars  FROM ranked WHERE rn_last  = 1)
        SELECT
          COUNT(*) AS n,
          quantile_cont(f.first_chars, 0.5) AS first_p50,
          quantile_cont(l.last_chars,  0.5) AS last_p50,
          quantile_cont(f.first_chars, 0.95) AS first_p95,
          quantile_cont(l.last_chars,  0.95) AS last_p95,
          SUM(CASE WHEN l.last_chars > f.first_chars THEN 1 ELSE 0 END) AS lengthened,
          SUM(CASE WHEN l.last_chars < f.first_chars THEN 1 ELSE 0 END) AS shortened,
          SUM(CASE WHEN l.last_chars = f.first_chars THEN 1 ELSE 0 END) AS unchanged
        FROM firsts f JOIN lasts l USING (puuid)
    """).fetchone()
    n_hfd, fp50, lp50, fp95, lp95, length, shortn, unch = hfdelta

    body_h = (
        f"**Temel sayılar**\n\n"
        f"- {_common.fmt_int(extra_children_human)} toplam edit eylemi, "
        f"{_common.fmt_int(human_fork_points)} edit noktasında (benzersiz prompt konumu), "
        f"{_common.fmt_int(human_fork_convs)} konuşmada. "
        f"Bir konuşmadaki max edit noktası: {_common.fmt_int((max(human_per_conv) if human_per_conv else 0))}; "
        f"tek bir prompt'un kaç kez yeniden yazıldığı ayrı ölçüm (aşağıdaki child-sayısı dağılımı).\n\n"
        f"**Konuşma başına edit noktası dağılımı**\n\n{h_dist}\n\n{h_table}\n\n"
        f"Grafik: `human_fork_per_conversation.png`.\n\n"
        f"**Edit sonrası hangi child devam etmiş?** ({_common.fmt_int(hfc_rows[3])}/{_common.fmt_int(hfc_rows[4])} child devam ediyor)\n\n"
    )
    body_h += _common.markdown_table(
        [
            ["ilk child devam",  _common.fmt_int(hfc_rows[0])],
            ["son child devam",  _common.fmt_int(hfc_rows[1])],
            ["orta child devam", _common.fmt_int(hfc_rows[2])],
        ],
        headers=["pozisyon", "n"],
    )
    body_h += (
        "\n\n**İlk vs son prompt uzunluğu** — her edit noktasında ilk yazılan prompt ile son yazılan prompt kıyaslanır.\n\n"
        + _common.markdown_table(
            [
                [
                    _common.fmt_int(n_hfd),
                    _common.fmt_int(fp50), _common.fmt_int(lp50),
                    _common.fmt_int(fp95), _common.fmt_int(lp95),
                    _common.fmt_int(length), _common.fmt_int(shortn), _common.fmt_int(unch),
                ]
            ],
            headers=[
                "edit noktası",
                "ilk p50 kar", "son p50 kar",
                "ilk p95 kar", "son p95 kar",
                "uzayan", "kısalan", "değişmeyen",
            ],
        )
    )
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
    h_bucket_buckets = [
        {"label": lbl, "count": int(n),
         "pct": (100.0 * n / max(1, len(human_per_conv)))}
        for lbl, n in hbucks
    ]
    sections.append(_common.Section(
        "Prompt yeniden yazma (human fork)",
        body_h,
        blocks=[
            _common.block_bullets([
                f"{_common.fmt_int(extra_children_human)} toplam edit eylemi, "
                f"{_common.fmt_int(human_fork_points)} edit noktasında (benzersiz prompt konumu), "
                f"{_common.fmt_int(human_fork_convs)} konuşmada.",
                f"Bir konuşmadaki max edit noktası: "
                f"{_common.fmt_int((max(human_per_conv) if human_per_conv else 0))}; "
                f"tek bir prompt'un kaç kez yeniden yazıldığı ayrı ölçüm "
                f"(aşağıdaki child-sayısı dağılımı).",
            ]),
            _common.block_paragraph("**Konuşma başına edit noktası dağılımı**"),
            _common.block_table(pct_cols, [[
                "edit noktası/konuşma",
                _common.fmt_int(h_pct["n"]),
                _common.fmt_int(h_pct["min"]),
                _common.fmt_int(h_pct["p50"]),
                _common.fmt_int(h_pct["p90"]),
                _common.fmt_int(h_pct["p95"]),
                _common.fmt_int(h_pct["p99"]),
                _common.fmt_int(h_pct["max"]),
            ]]),
            _common.block_bucket_chart(
                "Konuşma başına edit noktası",
                h_bucket_buckets,
                image="human_fork_per_conversation.png",
                xlabel="edit noktası",
            ),
            _common.block_table(
                [
                    {"key": "bucket", "label": "edit noktası", "align": "left"},
                    {"key": "count",  "label": "konuşma",     "align": "right"},
                    {"key": "pct",    "label": "oran",        "align": "right"},
                ],
                hb_rows,
            ),
            _common.block_paragraph(
                f"**Edit sonrası hangi child devam etmiş?** "
                f"({_common.fmt_int(hfc_rows[3])}/{_common.fmt_int(hfc_rows[4])} child devam ediyor)"
            ),
            _common.block_table(
                [
                    {"key": "pos", "label": "pozisyon", "align": "left"},
                    {"key": "n",   "label": "n",        "align": "right"},
                ],
                [
                    ["ilk child devam",  _common.fmt_int(hfc_rows[0])],
                    ["son child devam",  _common.fmt_int(hfc_rows[1])],
                    ["orta child devam", _common.fmt_int(hfc_rows[2])],
                ],
            ),
            _common.block_paragraph(
                "**İlk vs son prompt uzunluğu** — her edit noktasında ilk yazılan prompt "
                "ile son yazılan prompt kıyaslanır."
            ),
            _common.block_table(
                [
                    {"key": "np",          "label": "edit noktası",  "align": "right"},
                    {"key": "first_p50",   "label": "ilk p50 kar",   "align": "right"},
                    {"key": "last_p50",    "label": "son p50 kar",   "align": "right"},
                    {"key": "first_p95",   "label": "ilk p95 kar",   "align": "right"},
                    {"key": "last_p95",    "label": "son p95 kar",   "align": "right"},
                    {"key": "lengthened",  "label": "uzayan",        "align": "right"},
                    {"key": "shortened",   "label": "kısalan",       "align": "right"},
                    {"key": "unchanged",   "label": "değişmeyen",    "align": "right"},
                ],
                [[
                    _common.fmt_int(n_hfd),
                    _common.fmt_int(fp50), _common.fmt_int(lp50),
                    _common.fmt_int(fp95), _common.fmt_int(lp95),
                    _common.fmt_int(length), _common.fmt_int(shortn), _common.fmt_int(unch),
                ]],
            ),
        ],
    ))

    # --- Cevap yeniden üretme (assistant fork)
    asst_per_conv = [
        r[0] for r in con.execute(
            "SELECT assistant_fork_count FROM _stats_conversation WHERE assistant_fork_count > 0"
        ).fetchall()
    ]
    abucks = _bucketize_int(asst_per_conv, FORK_PER_CONVO_BUCKETS)
    ab_rows = [[lbl, _common.fmt_int(n), f"%{100.0*n/max(1,len(asst_per_conv)):.1f}"] for lbl, n in abucks]
    a_table = _common.markdown_table(ab_rows, headers=["retry noktası", "konuşma", "oran"])
    a_pct = _common.percentiles(asst_per_conv) if asst_per_conv else None

    # assistant fork = children sender=assistant → parent sender=human
    afc_rows = con.execute("""
        WITH parents AS (
          SELECT sm.message_uuid AS puuid
          FROM _stats_message sm
          WHERE sm.fork_parent_flag AND sm.sender='human'
        ),
        children_ranked AS (
          SELECT m.parent_message_uuid AS puuid,
                 m.uuid AS cuuid,
                 ROW_NUMBER() OVER (PARTITION BY m.parent_message_uuid ORDER BY m.created_at, m.uuid) AS rn_first,
                 ROW_NUMBER() OVER (PARTITION BY m.parent_message_uuid ORDER BY m.created_at DESC, m.uuid DESC) AS rn_last
          FROM messages m
          JOIN parents p ON p.puuid = m.parent_message_uuid
        ),
        continued AS (
          SELECT cr.*, CASE WHEN EXISTS (
                           SELECT 1 FROM messages m2
                           WHERE m2.parent_message_uuid = cr.cuuid
                         ) THEN TRUE ELSE FALSE END AS cont
          FROM children_ranked cr
        )
        SELECT
          SUM(CASE WHEN cont AND rn_first = 1 THEN 1 ELSE 0 END),
          SUM(CASE WHEN cont AND rn_last  = 1 THEN 1 ELSE 0 END),
          SUM(CASE WHEN cont AND rn_first <> 1 AND rn_last <> 1 THEN 1 ELSE 0 END),
          SUM(CASE WHEN cont THEN 1 ELSE 0 END),
          COUNT(*)
        FROM continued
    """).fetchone()

    # Retry sonrası asistan cevap uzunluğu: first vs last child
    afdelta = con.execute("""
        WITH parents AS (
          SELECT sm.message_uuid AS puuid
          FROM _stats_message sm
          WHERE sm.fork_parent_flag AND sm.sender='human'
        ),
        ranked AS (
          SELECT m.parent_message_uuid AS puuid,
                 sm.chars_text,
                 ROW_NUMBER() OVER (PARTITION BY m.parent_message_uuid ORDER BY m.created_at, m.uuid) AS rn_first,
                 ROW_NUMBER() OVER (PARTITION BY m.parent_message_uuid ORDER BY m.created_at DESC, m.uuid DESC) AS rn_last
          FROM messages m
          JOIN parents p ON p.puuid = m.parent_message_uuid
          JOIN _stats_message sm ON sm.message_uuid = m.uuid
        ),
        firsts AS (SELECT puuid, chars_text AS first_chars FROM ranked WHERE rn_first = 1),
        lasts  AS (SELECT puuid, chars_text AS last_chars  FROM ranked WHERE rn_last  = 1)
        SELECT COUNT(*),
               quantile_cont(f.first_chars, 0.5),
               quantile_cont(l.last_chars,  0.5),
               SUM(CASE WHEN l.last_chars > f.first_chars THEN 1 ELSE 0 END),
               SUM(CASE WHEN l.last_chars < f.first_chars THEN 1 ELSE 0 END),
               SUM(CASE WHEN l.last_chars = f.first_chars THEN 1 ELSE 0 END)
        FROM firsts f JOIN lasts l USING (puuid)
    """).fetchone()

    body_a = (
        f"**Temel sayılar**\n\n"
        f"- {_common.fmt_int(extra_children_asst)} toplam retry eylemi, "
        f"{_common.fmt_int(asst_fork_points)} retry noktasında (benzersiz asistan cevap konumu), "
        f"{_common.fmt_int(asst_fork_convs)} konuşmada. "
        f"Bir konuşmadaki max retry noktası: {_common.fmt_int((max(asst_per_conv) if asst_per_conv else 0))}.\n\n"
        f"**Konuşma başına retry noktası dağılımı**\n\n"
    )
    if a_pct:
        body_a += _common.markdown_table(
            [[
                "retry noktası/konuşma",
                _common.fmt_int(a_pct["n"]),
                _common.fmt_int(a_pct["min"]),
                _common.fmt_int(a_pct["p50"]),
                _common.fmt_int(a_pct["p90"]),
                _common.fmt_int(a_pct["p95"]),
                _common.fmt_int(a_pct["p99"]),
                _common.fmt_int(a_pct["max"]),
            ]],
            headers=["metric", "n", "min", "p50", "p90", "p95", "p99", "max"],
        ) + "\n\n" + a_table + "\n\nGrafik: `assistant_fork_per_conversation.png`.\n\n"

    body_a += (
        f"**Retry sonrası hangi child devam etmiş?** ({_common.fmt_int(afc_rows[3])}/{_common.fmt_int(afc_rows[4])} child devam ediyor)\n\n"
        + _common.markdown_table(
            [
                ["ilk child devam",  _common.fmt_int(afc_rows[0])],
                ["son child devam",  _common.fmt_int(afc_rows[1])],
                ["orta child devam", _common.fmt_int(afc_rows[2])],
            ],
            headers=["pozisyon", "n"],
        )
        + "\n\n**İlk vs son retry cevap uzunluğu** — her retry noktasında ilk üretilen cevap ile son üretilen cevap kıyaslanır.\n\n"
        + _common.markdown_table(
            [[
                _common.fmt_int(afdelta[0]),
                _common.fmt_int(afdelta[1]), _common.fmt_int(afdelta[2]),
                _common.fmt_int(afdelta[3]), _common.fmt_int(afdelta[4]), _common.fmt_int(afdelta[5]),
            ]],
            headers=["retry noktası", "ilk p50 kar", "son p50 kar", "uzayan", "kısalan", "değişmeyen"],
        )
        + "\n\nGrafik: `retry_length_delta.png`."
    )
    a_bucket_buckets = [
        {"label": lbl, "count": int(n),
         "pct": (100.0 * n / max(1, len(asst_per_conv)))}
        for lbl, n in abucks
    ]
    a_blocks: list[_common.Block] = [
        _common.block_bullets([
            f"{_common.fmt_int(extra_children_asst)} toplam retry eylemi, "
            f"{_common.fmt_int(asst_fork_points)} retry noktasında (benzersiz asistan cevap konumu), "
            f"{_common.fmt_int(asst_fork_convs)} konuşmada.",
            f"Bir konuşmadaki max retry noktası: "
            f"{_common.fmt_int((max(asst_per_conv) if asst_per_conv else 0))}.",
        ]),
        _common.block_paragraph("**Konuşma başına retry noktası dağılımı**"),
    ]
    if a_pct:
        a_blocks.append(_common.block_table(pct_cols, [[
            "retry noktası/konuşma",
            _common.fmt_int(a_pct["n"]),
            _common.fmt_int(a_pct["min"]),
            _common.fmt_int(a_pct["p50"]),
            _common.fmt_int(a_pct["p90"]),
            _common.fmt_int(a_pct["p95"]),
            _common.fmt_int(a_pct["p99"]),
            _common.fmt_int(a_pct["max"]),
        ]]))
        a_blocks.append(_common.block_bucket_chart(
            "Konuşma başına retry noktası",
            a_bucket_buckets,
            image="assistant_fork_per_conversation.png",
            xlabel="retry noktası",
        ))
        a_blocks.append(_common.block_table(
            [
                {"key": "bucket", "label": "retry noktası", "align": "left"},
                {"key": "count",  "label": "konuşma",      "align": "right"},
                {"key": "pct",    "label": "oran",         "align": "right"},
            ],
            ab_rows,
        ))
    a_blocks.extend([
        _common.block_paragraph(
            f"**Retry sonrası hangi child devam etmiş?** "
            f"({_common.fmt_int(afc_rows[3])}/{_common.fmt_int(afc_rows[4])} child devam ediyor)"
        ),
        _common.block_table(
            [
                {"key": "pos", "label": "pozisyon", "align": "left"},
                {"key": "n",   "label": "n",        "align": "right"},
            ],
            [
                ["ilk child devam",  _common.fmt_int(afc_rows[0])],
                ["son child devam",  _common.fmt_int(afc_rows[1])],
                ["orta child devam", _common.fmt_int(afc_rows[2])],
            ],
        ),
        _common.block_paragraph(
            "**İlk vs son retry cevap uzunluğu** — her retry noktasında ilk üretilen cevap "
            "ile son üretilen cevap kıyaslanır."
        ),
        _common.block_table(
            [
                {"key": "np",         "label": "retry noktası", "align": "right"},
                {"key": "first_p50",  "label": "ilk p50 kar",   "align": "right"},
                {"key": "last_p50",   "label": "son p50 kar",   "align": "right"},
                {"key": "lengthened", "label": "uzayan",        "align": "right"},
                {"key": "shortened",  "label": "kısalan",       "align": "right"},
                {"key": "unchanged",  "label": "değişmeyen",    "align": "right"},
            ],
            [[
                _common.fmt_int(afdelta[0]),
                _common.fmt_int(afdelta[1]), _common.fmt_int(afdelta[2]),
                _common.fmt_int(afdelta[3]), _common.fmt_int(afdelta[4]), _common.fmt_int(afdelta[5]),
            ]],
        ),
        _common.block_bucket_chart(
            "Retry sonrası cevap uzunluk değişimi",
            [
                {"label": "uzayan", "count": int(afdelta[3]), "pct": None},
                {"label": "kısalan", "count": int(afdelta[4]), "pct": None},
                {"label": "değişmeyen", "count": int(afdelta[5]), "pct": None},
            ],
            xlabel="uzunluk değişimi",
        ),
    ])
    sections.append(_common.Section(
        "Cevap yeniden üretme (assistant fork / retry)",
        body_a,
        blocks=a_blocks,
    ))

    # --- Cevap in-place düzenleme
    inplace_delta_vals = [
        r[0] for r in con.execute(
            "SELECT edit_delta_seconds FROM _stats_message WHERE inplace_edit_flag AND edit_delta_seconds IS NOT NULL"
        ).fetchall()
    ]
    ibucks = _bucketize_float(inplace_delta_vals, INPLACE_DELTA_BUCKETS)
    ib_rows = [[lbl, _common.fmt_int(n), f"%{100.0*n/max(1,len(inplace_delta_vals)):.1f}"] for lbl, n in ibucks]
    i_table = _common.markdown_table(ib_rows, headers=["delta", "mesaj", "oran"])

    # Inplace edit'li mesaj vs edit'siz assistant mesajı karakter dağılımı
    cmp_rows = con.execute("""
        SELECT
          CASE WHEN inplace_edit_flag THEN 'in-place edit' ELSE 'edit yok' END AS grp,
          COUNT(*) AS n,
          quantile_cont(chars_text, 0.5) AS p50,
          quantile_cont(chars_text, 0.9) AS p90,
          quantile_cont(chars_text, 0.95) AS p95,
          MAX(chars_text) AS maxc
        FROM _stats_message
        WHERE sender='assistant' AND chars_text > 0
        GROUP BY 1 ORDER BY 1
    """).fetchall()

    body_i = (
        f"**Kriter**: aynı `message_uuid` içinde `updated_at > created_at + 60s` ve `chars_text > 0`. "
        f"Örneklem yalnız **assistant** mesajlarında bu davranışı gösteriyor; human mesajlarında "
        f"(60s eşiğini geçen) in-place düzenleme ölçülmedi.\n\n"
        f"**Delta dağılımı** (n = {_common.fmt_int(len(inplace_delta_vals))})\n\n{i_table}\n\n"
        f"Grafik: `inplace_edit_delta.png`.\n\n"
        f"**In-place edit'li vs edit'siz asistan mesajı karakter dağılımı**\n\n"
        + _common.markdown_table(
            [
                [g, _common.fmt_int(n), _common.fmt_int(p50), _common.fmt_int(p90), _common.fmt_int(p95), _common.fmt_int(mx)]
                for g, n, p50, p90, p95, mx in cmp_rows
            ],
            headers=["kitle", "n", "p50", "p90", "p95", "max"],
        )
    )
    inplace_bucket_buckets = [
        {"label": lbl, "count": int(n),
         "pct": (100.0 * n / max(1, len(inplace_delta_vals)))}
        for lbl, n in ibucks
    ]
    sections.append(_common.Section(
        "Cevap in-place düzenleme (assistant)",
        body_i,
        blocks=[
            _common.block_paragraph(
                "**Kriter**: aynı `message_uuid` içinde `updated_at > created_at + 60s` ve "
                "`chars_text > 0`. Örneklem yalnız **assistant** mesajlarında bu davranışı gösteriyor; "
                "human mesajlarında (60s eşiğini geçen) in-place düzenleme ölçülmedi."
            ),
            _common.block_paragraph(
                f"**Delta dağılımı** (n = {_common.fmt_int(len(inplace_delta_vals))})"
            ),
            _common.block_bucket_chart(
                "In-place edit delta süresi dağılımı",
                inplace_bucket_buckets,
                image="inplace_edit_delta.png",
                xlabel="delta",
            ),
            _common.block_table(
                [
                    {"key": "bucket", "label": "delta",   "align": "left"},
                    {"key": "count",  "label": "mesaj",   "align": "right"},
                    {"key": "pct",    "label": "oran",    "align": "right"},
                ],
                ib_rows,
            ),
            _common.block_paragraph(
                "**In-place edit'li vs edit'siz asistan mesajı karakter dağılımı**"
            ),
            _common.block_table(
                [
                    {"key": "grp",  "label": "kitle", "align": "left"},
                    {"key": "n",    "label": "n",     "align": "right"},
                    {"key": "p50",  "label": "p50",   "align": "right"},
                    {"key": "p90",  "label": "p90",   "align": "right"},
                    {"key": "p95",  "label": "p95",   "align": "right"},
                    {"key": "max",  "label": "max",   "align": "right"},
                ],
                [
                    [g, _common.fmt_int(n), _common.fmt_int(p50), _common.fmt_int(p90), _common.fmt_int(p95), _common.fmt_int(mx)]
                    for g, n, p50, p90, p95, mx in cmp_rows
                ],
            ),
        ],
    ))

    # --- Çok-root konuşmalar
    mr_rows = con.execute("""
        SELECT root_count, COUNT(*) AS n,
               quantile_cont(message_count, 0.5) AS msg_p50,
               quantile_cont(lifetime_seconds, 0.5) AS life_p50
        FROM _stats_conversation
        WHERE root_count > 1
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    mr_table = _common.markdown_table(
        [
            [_common.fmt_int(rc), _common.fmt_int(n), _common.fmt_int(mp50), f"{int(lp50)}sn" if lp50 is not None else "—"]
            for rc, n, mp50, lp50 in mr_rows
        ],
        headers=["root_count", "konuşma", "mesaj p50", "ömür p50"],
    )
    body_mr = (
        f"Aynı `conversation_uuid` altında `parent_message_uuid IS NULL` veya root sentinel "
        f"(`00000000-…`) olan mesaj sayısı >1 olan konuşmalar. Muhtemel neden: UI'da "
        f"\"continue from previous state\" benzeri bir etkileşim.\n\n"
        f"**Dağılım** — toplam {_common.fmt_int(multi_root_convs)} konuşma\n\n{mr_table}"
    )
    mr_table_rows = [
        [_common.fmt_int(rc), _common.fmt_int(n), _common.fmt_int(mp50),
         f"{int(lp50)}sn" if lp50 is not None else "—"]
        for rc, n, mp50, lp50 in mr_rows
    ]
    sections.append(_common.Section(
        "Çok-root konuşmalar",
        body_mr,
        blocks=[
            _common.block_paragraph(
                "Aynı `conversation_uuid` altında `parent_message_uuid IS NULL` veya root sentinel "
                "(`00000000-…`) olan mesaj sayısı >1 olan konuşmalar. Muhtemel neden: UI'da "
                "\"continue from previous state\" benzeri bir etkileşim."
            ),
            _common.block_paragraph(
                f"**Dağılım** — toplam {_common.fmt_int(multi_root_convs)} konuşma"
            ),
            _common.block_table(
                [
                    {"key": "root",   "label": "root_count", "align": "right"},
                    {"key": "n",      "label": "konuşma",    "align": "right"},
                    {"key": "msg50",  "label": "mesaj p50",  "align": "right"},
                    {"key": "life50", "label": "ömür p50",   "align": "right"},
                ],
                mr_table_rows,
            ),
        ] if mr_table_rows else [],
    ))

    # --- Müdahale yoğunluğu — bayrak seti kombinasyonları
    combo_rows = con.execute("""
        SELECT
          CASE WHEN human_fork_count     > 0 THEN 1 ELSE 0 END AS h,
          CASE WHEN assistant_fork_count > 0 THEN 1 ELSE 0 END AS a,
          CASE WHEN inplace_edit_count   > 0 THEN 1 ELSE 0 END AS i,
          CASE WHEN root_count           > 1 THEN 1 ELSE 0 END AS r,
          COUNT(*) AS n
        FROM _stats_conversation
        WHERE message_count > 0
        GROUP BY 1,2,3,4
        ORDER BY n DESC
    """).fetchall()
    combo_labels = []
    combo_counts = []
    combo_table_rows = []
    for h, a, i, r, n in combo_rows:
        tags = []
        if h: tags.append("human-fork")
        if a: tags.append("asst-fork")
        if i: tags.append("in-place")
        if r: tags.append("multi-root")
        lbl = " + ".join(tags) if tags else "(hiçbiri)"
        combo_labels.append(lbl)
        combo_counts.append(n)
        combo_table_rows.append([lbl, _common.fmt_int(n), f"%{100.0*n/max(1,total_nonempty_convs):.2f}"])
    body_c = (
        "Her konuşmaya dört bayrak atanır: `human-fork` (prompt yeniden yazma), `asst-fork` "
        "(cevap yeniden üretme), `in-place` (asistan in-place düzenleme), `multi-root`. Tablo "
        "bayrak kombinasyonlarını gösterir.\n\n"
        + _common.markdown_table(combo_table_rows, headers=["kombinasyon", "konuşma", "oran"])
        + "\n\nGrafik: `intervention_combinations.png`."
    )
    combo_buckets = [
        {"label": lbl, "count": int(n),
         "pct": (100.0 * n / max(1, total_nonempty_convs))}
        for lbl, n in zip(combo_labels, combo_counts)
    ][:12]
    sections.append(_common.Section(
        "Müdahale yoğunluğu — bayrak kombinasyonları",
        body_c,
        blocks=[
            _common.block_paragraph(
                "Her konuşmaya dört bayrak atanır: `human-fork` (prompt yeniden yazma), "
                "`asst-fork` (cevap yeniden üretme), `in-place` (asistan in-place düzenleme), "
                "`multi-root`. Tablo bayrak kombinasyonlarını gösterir."
            ),
            _common.block_bucket_chart(
                "Müdahale bayrak kombinasyonları (top 12)",
                combo_buckets,
                image="intervention_combinations.png",
                xlabel="kombinasyon",
            ),
            _common.block_table(
                [
                    {"key": "combo", "label": "kombinasyon", "align": "left"},
                    {"key": "count", "label": "konuşma",    "align": "right"},
                    {"key": "pct",   "label": "oran",       "align": "right"},
                ],
                combo_table_rows,
            ),
        ],
    ))

    # --- Top konuşmalar (her mekaniğin en yoğun 3 örneği)
    TOP_N = 3

    top_human = con.execute(f"""
        WITH pc AS (
          SELECT m.parent_message_uuid AS puuid, m.conversation_uuid, COUNT(*) AS c
          FROM messages m
          WHERE m.sender='human'
            AND m.parent_message_uuid IS NOT NULL
            AND m.parent_message_uuid <> '00000000-0000-4000-8000-000000000000'
          GROUP BY 1, 2
          HAVING COUNT(*) > 1
        )
        SELECT sc.name,
               SUM(pc.c - 1) AS edit_count,
               COUNT(*)       AS fork_points,
               MAX(pc.c)      AS max_children,
               sc.message_count
        FROM pc
        JOIN _stats_conversation sc ON sc.conversation_uuid = pc.conversation_uuid
        GROUP BY sc.name, sc.message_count
        ORDER BY edit_count DESC
        LIMIT {TOP_N}
    """).fetchall()

    top_retry = con.execute(f"""
        WITH pc AS (
          SELECT m.parent_message_uuid AS puuid, m.conversation_uuid, COUNT(*) AS c
          FROM messages m
          WHERE m.sender='assistant'
            AND m.parent_message_uuid IS NOT NULL
            AND m.parent_message_uuid <> '00000000-0000-4000-8000-000000000000'
          GROUP BY 1, 2
          HAVING COUNT(*) > 1
        )
        SELECT sc.name,
               SUM(pc.c - 1) AS retry_count,
               COUNT(*)       AS fork_points,
               MAX(pc.c)      AS max_children,
               sc.message_count
        FROM pc
        JOIN _stats_conversation sc ON sc.conversation_uuid = pc.conversation_uuid
        GROUP BY sc.name, sc.message_count
        ORDER BY retry_count DESC
        LIMIT {TOP_N}
    """).fetchall()

    top_inplace = con.execute(f"""
        SELECT name, inplace_edit_count, message_count
        FROM _stats_conversation
        WHERE inplace_edit_count > 0
        ORDER BY inplace_edit_count DESC
        LIMIT {TOP_N}
    """).fetchall()

    top_fork = con.execute(f"""
        SELECT name,
               human_fork_count,
               assistant_fork_count,
               (human_fork_count + assistant_fork_count) AS total_forks,
               message_count
        FROM _stats_conversation
        WHERE (human_fork_count + assistant_fork_count) > 0
        ORDER BY total_forks DESC, human_fork_count DESC
        LIMIT {TOP_N}
    """).fetchall()

    def _trim(s: str, n: int = 60) -> str:
        s = (s or "").strip() or "(adı yok)"
        return s if len(s) <= n else s[: n - 1] + "…"

    body_top = (
        "Her mekaniği adıyla listeleyen top 3. \"Konuşma\" sütunu `name` (CSV'lerde "
        f"`conversation_uuid` yer alır).\n\n"
        "**Prompt yeniden yazma (human fork) — en çok edit**\n\n"
        + _common.markdown_table(
            [[_trim(nm), _common.fmt_int(ec), _common.fmt_int(fp), _common.fmt_int(mc), _common.fmt_int(msg)]
             for nm, ec, fp, mc, msg in top_human],
            headers=["konuşma", "edit", "edit noktası", "tek noktada max yazım", "mesaj"],
        )
        + "\n\n**Cevap yeniden üretme (retry) — en çok retry**\n\n"
        + _common.markdown_table(
            [[_trim(nm), _common.fmt_int(rc), _common.fmt_int(fp), _common.fmt_int(mc), _common.fmt_int(msg)]
             for nm, rc, fp, mc, msg in top_retry],
            headers=["konuşma", "retry", "retry noktası", "tek noktada max yeniden üretim", "mesaj"],
        )
        + "\n\n**In-place edit — en çok düzenlenen asistan mesajına sahip**\n\n"
        + _common.markdown_table(
            [[_trim(nm), _common.fmt_int(ie), _common.fmt_int(msg)]
             for nm, ie, msg in top_inplace],
            headers=["konuşma", "in-place edit", "mesaj"],
        )
        + "\n\n**Toplam müdahale noktası — en çok edit + retry noktası**\n\n"
        + _common.markdown_table(
            [[_trim(nm), _common.fmt_int(tot), _common.fmt_int(hf), _common.fmt_int(af), _common.fmt_int(msg)]
             for nm, hf, af, tot, msg in top_fork],
            headers=["konuşma", "toplam nokta", "edit noktası", "retry noktası", "mesaj"],
        )
        + "\n\nNot: **edit** = Σ(child-1) her fork noktasında — aynı noktada 3 child varsa 2 edit sayılır. "
        "**edit noktası** = ≥2 child'a sahip benzersiz parent. **tek noktada max yazım** = bir prompt'un kaç kez yeniden yazıldığı."
    )
    top_human_rows = [
        [_trim(nm), _common.fmt_int(ec), _common.fmt_int(fp), _common.fmt_int(mc), _common.fmt_int(msg)]
        for nm, ec, fp, mc, msg in top_human
    ]
    top_retry_rows = [
        [_trim(nm), _common.fmt_int(rc), _common.fmt_int(fp), _common.fmt_int(mc), _common.fmt_int(msg)]
        for nm, rc, fp, mc, msg in top_retry
    ]
    top_inplace_rows = [
        [_trim(nm), _common.fmt_int(ie), _common.fmt_int(msg)]
        for nm, ie, msg in top_inplace
    ]
    top_fork_rows = [
        [_trim(nm), _common.fmt_int(tot), _common.fmt_int(hf), _common.fmt_int(af), _common.fmt_int(msg)]
        for nm, hf, af, tot, msg in top_fork
    ]
    sections.append(_common.Section(
        "Top konuşmalar — her mekaniğin en yoğun 3 örneği",
        body_top,
        blocks=[
            _common.block_paragraph(
                "Her mekaniği adıyla listeleyen top 3. \"Konuşma\" sütunu `name` (CSV'lerde "
                "`conversation_uuid` yer alır)."
            ),
            _common.block_paragraph("**Prompt yeniden yazma (human fork) — en çok edit**"),
            _common.block_table(
                [
                    {"key": "name",  "label": "konuşma",               "align": "left"},
                    {"key": "edit",  "label": "edit",                  "align": "right"},
                    {"key": "fp",    "label": "edit noktası",          "align": "right"},
                    {"key": "max",   "label": "tek noktada max yazım", "align": "right"},
                    {"key": "msg",   "label": "mesaj",                 "align": "right"},
                ],
                top_human_rows,
            ),
            _common.block_paragraph("**Cevap yeniden üretme (retry) — en çok retry**"),
            _common.block_table(
                [
                    {"key": "name",  "label": "konuşma",                           "align": "left"},
                    {"key": "retry", "label": "retry",                             "align": "right"},
                    {"key": "fp",    "label": "retry noktası",                     "align": "right"},
                    {"key": "max",   "label": "tek noktada max yeniden üretim",    "align": "right"},
                    {"key": "msg",   "label": "mesaj",                             "align": "right"},
                ],
                top_retry_rows,
            ),
            _common.block_paragraph("**In-place edit — en çok düzenlenen asistan mesajına sahip**"),
            _common.block_table(
                [
                    {"key": "name", "label": "konuşma",      "align": "left"},
                    {"key": "ie",   "label": "in-place edit","align": "right"},
                    {"key": "msg",  "label": "mesaj",        "align": "right"},
                ],
                top_inplace_rows,
            ),
            _common.block_paragraph("**Toplam müdahale noktası — en çok edit + retry noktası**"),
            _common.block_table(
                [
                    {"key": "name", "label": "konuşma",       "align": "left"},
                    {"key": "tot",  "label": "toplam nokta",  "align": "right"},
                    {"key": "hf",   "label": "edit noktası",  "align": "right"},
                    {"key": "af",   "label": "retry noktası", "align": "right"},
                    {"key": "msg",  "label": "mesaj",         "align": "right"},
                ],
                top_fork_rows,
            ),
            _common.block_paragraph(
                "Not: **edit** = Σ(child-1) her fork noktasında — aynı noktada 3 child varsa 2 edit sayılır. "
                "**edit noktası** = ≥2 child'a sahip benzersiz parent. "
                "**tek noktada max yazım** = bir prompt'un kaç kez yeniden yazıldığı."
            ),
        ],
    ))

    # --- Notlar
    notes = [
        "- Kaynak: `_stats_message` kolonları `inplace_edit_flag`, `fork_parent_flag`, `fork_child_flag`, `fork_child_continued`, `edit_delta_seconds`.",
        "- Claude.ai'de üç müdahale tipinin export izi farklı: prompt/cevap yeniden üretimi **yeni message_uuid** (fork), in-place düzenleme **aynı uuid + updated_at**.",
        "- Fork = aynı non-root parent'ın birden çok child'ı olması. Root parent (`NULL` veya `00000000-…`) dahil edilmez.",
        "- **Fork sender semantiği**: fork parent'ın sender'ı, fork'u *tetikleyen eylemi yapan*'ın tersidir. Parent=assistant → children=human → **prompt yeniden yazma**. Parent=human → children=assistant → **retry**. Örneklemde mixed-sender fork yok — bir parent'ın child'ları her zaman tek sender tipinde.",
        "- `fork_child_continued` = bu child'ın descendant'ı var mı. Kullanıcının hangi dalı sürdürdüğünün göstergesidir; ama kullanıcı bir dalda gezinip hiç mesaj yazmadıysa oradaki seçim izi tutulmaz.",
        "- In-place edit eşiği 60s: streaming snapshot'larını eliminate etmek içindir; streaming sırasında `updated_at - created_at` birkaç saniyedir.",
        "- Çok-root konuşmalarda `root_count` alanı, `parent_message_uuid IN (NULL, '00000000-0000-4000-8000-000000000000')` olan mesaj sayısıdır.",
    ]
    note_bullets = [n.lstrip("- ").strip() for n in notes]
    sections.append(_common.Section(
        "Notlar",
        "\n".join(notes),
        blocks=[_common.block_bullets(note_bullets)],
    ))

    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)

    # --- Grafikler

    if HAS_MPL:
        def _bar_buckets(values_bucket, path_name, title, color):
            labels = [r[0] for r in values_bucket]
            vals = [r[1] for r in values_bucket]
            fig, ax = plt.subplots()
            ax.bar(labels, vals, color=color)
            ax.set_ylabel("konuşma sayısı")
            ax.set_title(title)
            _common.save_fig(fig, out_dir / path_name)

        if human_per_conv:
            _bar_buckets(hbucks, "human_fork_per_conversation.png",
                         "Konuşma başına edit noktası", "#2c3e50")
        if asst_per_conv:
            _bar_buckets(abucks, "assistant_fork_per_conversation.png",
                         "Konuşma başına retry noktası", "#34495e")
        if inplace_delta_vals:
            labels = [r[0] for r in ibucks]
            vals = [r[1] for r in ibucks]
            fig, ax = plt.subplots()
            ax.bar(labels, vals, color="#7f8c8d")
            ax.set_ylabel("mesaj sayısı")
            ax.set_title("In-place edit delta süresi dağılımı")
            _common.save_fig(fig, out_dir / "inplace_edit_delta.png")

        # Fork devam pozisyonu grafiği: human ve assistant'ı grouped bar'la göster
        fig, ax = plt.subplots()
        positions = ["ilk", "orta", "son"]
        h_vals = [hfc_rows[0] or 0, hfc_rows[2] or 0, hfc_rows[1] or 0]
        a_vals = [afc_rows[0] or 0, afc_rows[2] or 0, afc_rows[1] or 0]
        x = np.arange(len(positions))
        w = 0.38
        ax.bar(x - w/2, h_vals, w, label="human fork", color="#2c3e50")
        ax.bar(x + w/2, a_vals, w, label="asst fork", color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(positions)
        ax.set_ylabel("devam eden child sayısı")
        ax.set_title("Fork sonrası devam eden child pozisyonu")
        ax.legend()
        _common.save_fig(fig, out_dir / "fork_active_branch.png")

        # Retry length delta
        if afdelta and afdelta[0]:
            labels = ["uzayan", "kısalan", "değişmeyen"]
            vals = [afdelta[3] or 0, afdelta[4] or 0, afdelta[5] or 0]
            fig, ax = plt.subplots()
            ax.bar(labels, vals, color="#34495e")
            ax.set_ylabel("nokta sayısı")
            ax.set_title("Retry sonrası cevap uzunluk değişimi")
            _common.save_fig(fig, out_dir / "retry_length_delta.png")

        # Intervention combinations
        top_combos = combo_table_rows[:12]
        if top_combos:
            labels = [r[0] for r in top_combos]
            vals = [int(r[1].replace(" ", "")) for r in top_combos]
            fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(labels))))
            y = np.arange(len(labels))
            ax.barh(y, vals, color="#7f8c8d")
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("konuşma sayısı")
            ax.set_title("Müdahale bayrak kombinasyonları (top 12)")
            _common.save_fig(fig, out_dir / "intervention_combinations.png")

    # --- CSV çıktıları
    # human_forks.csv: parent mesajı (assistant sender) + children human
    _common.write_csv(
        out_dir, "human_forks.csv",
        con.execute("""
          SELECT sm.conversation_uuid, sm.message_uuid AS parent_message_uuid,
                 sm.sender AS parent_sender, sm.chars_text AS parent_chars,
                 (SELECT COUNT(*) FROM messages m2 WHERE m2.parent_message_uuid = sm.message_uuid) AS child_count
          FROM _stats_message sm
          WHERE sm.fork_parent_flag AND sm.sender='assistant'
          ORDER BY sm.created_at
        """).fetchall(),
        headers=["conversation_uuid", "parent_message_uuid", "parent_sender", "parent_chars", "child_count"],
    )
    # assistant_forks.csv: parent mesajı (human sender) + children assistant
    _common.write_csv(
        out_dir, "assistant_forks.csv",
        con.execute("""
          SELECT sm.conversation_uuid, sm.message_uuid AS parent_message_uuid,
                 sm.sender AS parent_sender, sm.chars_text AS parent_chars,
                 (SELECT COUNT(*) FROM messages m2 WHERE m2.parent_message_uuid = sm.message_uuid) AS child_count
          FROM _stats_message sm
          WHERE sm.fork_parent_flag AND sm.sender='human'
          ORDER BY sm.created_at
        """).fetchall(),
        headers=["conversation_uuid", "parent_message_uuid", "parent_sender", "parent_chars", "child_count"],
    )
    _common.write_csv(
        out_dir, "inplace_edits.csv",
        con.execute("""
          SELECT sm.conversation_uuid, sm.message_uuid, sm.sender, sm.chars_text,
                 sm.edit_delta_seconds
          FROM _stats_message sm
          WHERE sm.inplace_edit_flag
          ORDER BY sm.edit_delta_seconds DESC
        """).fetchall(),
        headers=["conversation_uuid", "message_uuid", "sender", "chars_text", "edit_delta_seconds"],
    )
    _common.write_csv(
        out_dir, "multi_root.csv",
        con.execute("""
          SELECT conversation_uuid, name, root_count, message_count, lifetime_seconds
          FROM _stats_conversation
          WHERE root_count > 1
          ORDER BY root_count DESC, message_count DESC
        """).fetchall(),
        headers=["conversation_uuid", "name", "root_count", "message_count", "lifetime_seconds"],
    )

    # --- JSON summary
    top_heavy = con.execute("""
        SELECT conversation_uuid, name,
               human_fork_count, assistant_fork_count, inplace_edit_count, root_count, message_count
        FROM _stats_conversation
        WHERE message_count > 0
        ORDER BY (human_fork_count + assistant_fork_count + inplace_edit_count + (CASE WHEN root_count>1 THEN root_count ELSE 0 END)) DESC
        LIMIT 10
    """).fetchall()
    summary = {
        "headline": (
            f"{_common.fmt_int(human_fork_points)} prompt yeniden yazma / "
            f"{_common.fmt_int(asst_fork_points)} retry / "
            f"{_common.fmt_int(inplace_edits)} in-place edit / "
            f"{_common.fmt_int(multi_root_convs)} çok-root"
        ),
        "human_fork_points": int(human_fork_points),
        "assistant_fork_points": int(asst_fork_points),
        "inplace_edits": int(inplace_edits),
        "multi_root_conversations": int(multi_root_convs),
        "extra_children_human": int(extra_children_human),
        "extra_children_assistant": int(extra_children_asst),
        "human_fork_conversations": int(human_fork_convs),
        "assistant_fork_conversations": int(asst_fork_convs),
        "inplace_edit_conversations": int(inplace_convs),
        "top_intervention_heavy": [
            {
                "conversation_uuid": u, "name": nm,
                "human_fork": int(hf or 0), "assistant_fork": int(af or 0),
                "inplace_edit": int(ie or 0), "root_count": int(rc or 0),
                "message_count": int(mc or 0),
            }
            for u, nm, hf, af, ie, rc, mc in top_heavy
        ],
    }
    _common.write_json(out_dir, "summary.json", summary)
    return summary
