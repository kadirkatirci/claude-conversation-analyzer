"""
m12 — Tool hata tipolojisi.

Kaynak: `tool_results` (is_error, content), `tool_calls`, `messages`.
Çıktı: reports/quantitative/12-tool-errors/
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from . import _common
from ._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "12-tool-errors"
TITLE = "Tool hata tipolojisi"

MIN_CALLS_FOR_RATE = 20  # tool başına hata oranı için asgari çağrı sayısı

CATEGORY_PATTERNS = [
    ("NOT_FOUND", [r"not found", r"\b404\b", r"does not exist", r"no such"]),
    ("PERMISSIONS", [r"permission", r"forbidden", r"\b403\b", r"blocked"]),
    ("TIMEOUT", [r"timeout", r"timed out", r"deadline"]),
    ("RATE_LIMIT", [r"rate.?limit", r"\b429\b", r"too many"]),
    ("ROBOTS", [r"robots", r"disallow"]),
    ("CLIENT_ERROR", [r"client_error", r"\b400\b", r"bad request", r"cannot be fetched"]),
    ("VALIDATION", [r"validation", r"invalid input", r"field required", r"schema"]),
    ("NO_RESULT", [r"no result received", r"empty response"]),
    ("SYNTAX", [r"syntax", r"parse error", r"malformed"]),
    ("FETCH_UNKNOWN", [r"unknown error", r"error while fetching"]),
]


def categorize(text: str) -> str:
    t = (text or "").lower()
    for cat, patterns in CATEGORY_PATTERNS:
        for p in patterns:
            if re.search(p, t):
                return cat
    return "OTHER"


def normalize_msg(text: str, limit: int = 200) -> str:
    t = (text or "").strip()
    t = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<UUID>", t)
    t = re.sub(r"/[A-Za-z0-9._/\-]+", "<PATH>", t)
    t = re.sub(r"https?://\S+", "<URL>", t)
    t = re.sub(r"\s+", " ", t)
    return t[:limit]


def run(con, out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sections: list[_common.Section] = []
    summary: dict[str, Any] = {}

    # --- Toplam rakamlar
    total_calls, total_errs = con.execute(
        "SELECT COUNT(*), SUM(CASE WHEN is_error THEN 1 ELSE 0 END) FROM tool_results"
    ).fetchone()
    err_conv_count = con.execute(
        "SELECT COUNT(DISTINCT conversation_uuid) FROM tool_results WHERE is_error=true"
    ).fetchone()[0]
    err_msg_count = con.execute(
        "SELECT COUNT(DISTINCT message_uuid) FROM tool_results WHERE is_error=true"
    ).fetchone()[0]
    overall_rate = 100.0 * total_errs / total_calls if total_calls else 0.0

    # --- Tool başına hata oranı
    tool_rates = con.execute(
        """
        SELECT name,
               COUNT(*) AS calls,
               SUM(CASE WHEN is_error THEN 1 ELSE 0 END) AS errs
        FROM tool_results
        GROUP BY name
        ORDER BY errs DESC
        """
    ).fetchall()

    # --- Hata mesajları (kategori + örnek) çıkar
    err_rows = con.execute(
        """
        SELECT tr.name, tr.conversation_uuid, tr.message_uuid,
               json_extract_string(c.value, '$.text') AS msg
        FROM tool_results tr,
             UNNEST(CAST(tr.content AS JSON[])) AS c(value)
        WHERE tr.is_error = true
        """
    ).fetchall()

    cat_counter: Counter[str] = Counter()
    cat_by_tool: dict[str, Counter[str]] = {}
    samples_by_cat: dict[str, Counter[str]] = {}
    for name, _cuuid, _muuid, msg in err_rows:
        cat = categorize(msg or "")
        cat_counter[cat] += 1
        cat_by_tool.setdefault(name, Counter())[cat] += 1
        samples_by_cat.setdefault(cat, Counter())[normalize_msg(msg or "")] += 1

    highest_rate_tool = None
    for name, calls, errs in tool_rates:
        if calls >= MIN_CALLS_FOR_RATE:
            highest_rate_tool = (name, calls, errs, 100.0 * errs / calls)
            break
    if highest_rate_tool is None and tool_rates:
        n, c, e = tool_rates[0]
        highest_rate_tool = (n, c, e, 100.0 * e / c if c else 0.0)
    most_errs_tool = tool_rates[0] if tool_rates else None

    # --- Post-error behavior: aynı konuşmada sonraki human mesajı + sonraki assistant davranışı
    # Semantik:
    #   A_err      = is_error=true üreten assistant mesajı (tool_results.message_uuid)
    #   hata_tools = A_err'de hataya düşen tool adlarının seti
    #   H_next     = aynı conversation'da A_err'den sonraki ilk human mesajı
    #   A_next     = H_next'ten sonraki ilk assistant mesajı
    # Kategoriler:
    #   retry_same_tool  → A_next'te hata_tools'tan en az birinin tool_call'u var
    #   diff_tool        → A_next'in tool_call'u var ama hata_tools'tan hiçbirini kullanmıyor
    #   no_tool          → A_next var, tool_call'u yok
    #   no_followup      → A_err sonrası başka mesaj yok
    #   ended_on_human   → H_next var ama A_next yok
    post_rows = con.execute(
        """
        WITH err_msgs AS (
          SELECT conversation_uuid,
                 message_uuid,
                 LIST(DISTINCT name) AS err_tools
          FROM tool_results
          WHERE is_error = true
          GROUP BY conversation_uuid, message_uuid
        ),
        err_pos AS (
          SELECT e.conversation_uuid, e.message_uuid AS err_msg, e.err_tools,
                 m.position AS err_pos
          FROM err_msgs e
          JOIN messages m ON m.uuid = e.message_uuid
        ),
        next_human AS (
          SELECT ep.conversation_uuid, ep.err_msg, ep.err_tools,
                 MIN(m.position) AS h_pos
          FROM err_pos ep
          JOIN messages m
            ON m.conversation_uuid = ep.conversation_uuid
           AND m.sender = 'human'
           AND m.position > ep.err_pos
          GROUP BY 1,2,3
        ),
        next_assistant AS (
          SELECT nh.conversation_uuid, nh.err_msg, nh.err_tools, nh.h_pos,
                 MIN(m.position) AS a_pos
          FROM next_human nh
          JOIN messages m
            ON m.conversation_uuid = nh.conversation_uuid
           AND m.sender = 'assistant'
           AND m.position > nh.h_pos
          GROUP BY 1,2,3,4
        ),
        a_next_tools AS (
          SELECT na.conversation_uuid, na.err_msg, na.err_tools, na.a_pos,
                 LIST(DISTINCT tc.name) AS a_tools
          FROM next_assistant na
          JOIN messages m
            ON m.conversation_uuid = na.conversation_uuid
           AND m.position = na.a_pos
          LEFT JOIN tool_calls tc
            ON tc.message_uuid = m.uuid
          GROUP BY 1,2,3,4
        )
        SELECT ep.conversation_uuid, ep.err_msg,
               ep.err_tools,
               nh.h_pos,
               ant.a_pos,
               ant.a_tools
        FROM err_pos ep
        LEFT JOIN next_human nh
          ON nh.conversation_uuid = ep.conversation_uuid AND nh.err_msg = ep.err_msg
        LEFT JOIN a_next_tools ant
          ON ant.conversation_uuid = ep.conversation_uuid AND ant.err_msg = ep.err_msg
        """
    ).fetchall()

    # Basitleştir: her err_msg için tek satıra indirgeyelim
    seen: set[tuple[str, str]] = set()
    post_counts = Counter()
    for conv, err_msg, err_tools, h_pos, a_pos, a_tools in post_rows:
        key = (conv, err_msg)
        if key in seen:
            continue
        seen.add(key)
        err_set = set(err_tools or [])
        if h_pos is None:
            post_counts["no_followup"] += 1
            continue
        if a_pos is None:
            post_counts["ended_on_human"] += 1
            continue
        a_set = set(x for x in (a_tools or []) if x is not None)
        if not a_set:
            post_counts["no_tool"] += 1
        elif a_set & err_set:
            post_counts["retry_same_tool"] += 1
        else:
            post_counts["diff_tool"] += 1

    # --- Hatalı vs hatasız konuşma kıyası
    comp_rows = con.execute(
        """
        WITH err_convs AS (
          SELECT DISTINCT conversation_uuid FROM tool_results WHERE is_error=true
        )
        SELECT
          CASE WHEN ec.conversation_uuid IS NOT NULL THEN 'hata var' ELSE 'hata yok' END AS grp,
          COUNT(*) AS n,
          quantile_cont(sc.message_count, 0.5) AS msg_p50,
          quantile_cont(sc.message_count, 0.95) AS msg_p95,
          quantile_cont(sc.tool_calls_total, 0.5) AS tc_p50,
          quantile_cont(sc.tool_calls_total, 0.95) AS tc_p95,
          quantile_cont(sc.lifetime_seconds, 0.5) AS life_p50
        FROM _stats_conversation sc
        LEFT JOIN err_convs ec USING (conversation_uuid)
        WHERE sc.message_count > 0
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()

    # --- Bölüm 1: Öne çıkanlar
    head_lines = [
        f"- Toplam tool çağrısı: **{_common.fmt_int(total_calls)}**, hatalı: **{_common.fmt_int(total_errs)}** (genel oran **%{overall_rate:.1f}**)",
        f"- Hatalı mesaj: **{_common.fmt_int(err_msg_count)}**, hatalı konuşma: **{_common.fmt_int(err_conv_count)}**",
    ]
    head_bullets = [
        f"Toplam tool çağrısı: **{_common.fmt_int(total_calls)}**, hatalı: **{_common.fmt_int(total_errs)}** (genel oran **%{overall_rate:.1f}**)",
        f"Hatalı mesaj: **{_common.fmt_int(err_msg_count)}**, hatalı konuşma: **{_common.fmt_int(err_conv_count)}**",
    ]
    if highest_rate_tool:
        n, c, e, r = highest_rate_tool
        line = f"En yüksek hata oranı (n≥{MIN_CALLS_FOR_RATE}): **{n}** — {_common.fmt_int(e)}/{_common.fmt_int(c)} (%{r:.1f})"
        head_lines.append("- " + line)
        head_bullets.append(line)
    if most_errs_tool:
        n, c, e = most_errs_tool
        r = 100.0 * e / c if c else 0.0
        line = f"En çok hata üreten: **{n}** — {_common.fmt_int(e)} hata ({_common.fmt_int(c)} çağrı, %{r:.1f})"
        head_lines.append("- " + line)
        head_bullets.append(line)
    top_cat = cat_counter.most_common(1)
    if top_cat:
        line = f"En yaygın kategori: **{top_cat[0][0]}** ({_common.fmt_int(top_cat[0][1])} hata)"
        head_lines.append("- " + line)
        head_bullets.append(line)
    sections.append(_common.Section(
        "Öne çıkanlar",
        "\n".join(head_lines),
        blocks=[_common.block_bullets(head_bullets)],
    ))

    # --- Bölüm 2: Tool başına hata oranı
    rows_filt = [(n, c, e) for n, c, e in tool_rates if c >= MIN_CALLS_FOR_RATE]
    rows_filt.sort(key=lambda x: x[2] / x[1] if x[1] else 0, reverse=True)
    rows_filt = rows_filt[:20]
    tool_table_rows = [
        [n, _common.fmt_int(c), _common.fmt_int(e), f"%{100.0*e/c:.1f}"]
        for n, c, e in rows_filt
    ]
    table = _common.markdown_table(tool_table_rows, headers=["tool", "çağrı", "hata", "oran"])
    tool_rate_buckets = [
        {"label": n, "count": int(round(100.0 * e / c)), "pct": (100.0 * e / c) if c else 0.0}
        for n, c, e in rows_filt
    ]
    sections.append(_common.Section(
        f"Tool başına hata oranı (n≥{MIN_CALLS_FOR_RATE}, top 20)",
        table + f"\n\nGrafik: `top_tools_error_rate.png`.",
        blocks=[
            _common.block_bucket_chart(
                f"Tool başına hata oranı (n≥{MIN_CALLS_FOR_RATE})",
                tool_rate_buckets,
                image="top_tools_error_rate.png",
                xlabel="tool",
            ),
            _common.block_table(
                [
                    {"key": "tool",  "label": "tool",  "align": "left"},
                    {"key": "calls", "label": "çağrı", "align": "right"},
                    {"key": "errs",  "label": "hata",  "align": "right"},
                    {"key": "rate",  "label": "oran",  "align": "right"},
                ],
                tool_table_rows,
            ),
        ] if rows_filt else [],
    ))

    # --- Bölüm 3: Hata kategori dağılımı
    cat_total = sum(cat_counter.values())
    cat_rows = [
        [cat, _common.fmt_int(n), f"%{100.0*n/cat_total:.1f}"]
        for cat, n in cat_counter.most_common()
    ]
    cat_table = _common.markdown_table(cat_rows, headers=["kategori", "hata", "oran"])
    cat_buckets = [
        {"label": cat, "count": int(n), "pct": (100.0 * n / cat_total) if cat_total else 0.0}
        for cat, n in cat_counter.most_common()
    ]
    sections.append(_common.Section(
        "Hata kategori dağılımı",
        cat_table + "\n\nGrafik: `error_categories.png`.",
        blocks=[
            _common.block_bucket_chart(
                "Hata kategori dağılımı",
                cat_buckets,
                image="error_categories.png",
                xlabel="kategori",
            ),
            _common.block_table(
                [
                    {"key": "cat",   "label": "kategori", "align": "left"},
                    {"key": "count", "label": "hata",    "align": "right"},
                    {"key": "pct",   "label": "oran",    "align": "right"},
                ],
                cat_rows,
            ),
        ] if cat_buckets else [],
    ))

    # --- Bölüm 4: Kategori × tool çaprazı
    top_err_tools = [n for n, _, _ in tool_rates[:10] if n is not None]
    cats_order = [c for c, _ in cat_counter.most_common()]
    cross_headers = ["tool"] + cats_order + ["toplam"]
    cross_rows = []
    for t in top_err_tools:
        row = [t]
        total = 0
        for c in cats_order:
            n = cat_by_tool.get(t, Counter()).get(c, 0)
            row.append(_common.fmt_int(n) if n else "—")
            total += n
        row.append(_common.fmt_int(total))
        cross_rows.append(row)
    cross_cols = [{"key": "tool", "label": "tool", "align": "left"}]
    cross_cols.extend({"key": c, "label": c, "align": "right"} for c in cats_order)
    cross_cols.append({"key": "total", "label": "toplam", "align": "right"})
    sections.append(_common.Section(
        "Kategori × tool (top 10 hata üreten tool)",
        _common.markdown_table(cross_rows, headers=cross_headers),
        blocks=[_common.block_table(cross_cols, cross_rows)] if cross_rows else [],
    ))

    # --- Bölüm 5: Hata sonrası davranış
    post_total = sum(post_counts.values())
    post_order = ["retry_same_tool", "diff_tool", "no_tool", "ended_on_human", "no_followup"]
    post_rows_t = [
        [k, _common.fmt_int(post_counts.get(k, 0)), f"%{100.0*post_counts.get(k,0)/post_total:.1f}" if post_total else "—"]
        for k in post_order
    ]
    post_desc = {
        "retry_same_tool": "sonraki asistan mesajı aynı tool'u tekrar çağırdı",
        "diff_tool": "sonraki asistan mesajı farklı tool kullandı",
        "no_tool": "sonraki asistan mesajı tool kullanmadan yanıtladı",
        "ended_on_human": "kullanıcı prompt attı, asistan cevabı yok",
        "no_followup": "hatadan sonra konuşma bitti",
    }
    post_table_rows = [[k, post_desc[k], n, p] for k, n, p in post_rows_t]
    post_table = _common.markdown_table(post_table_rows, headers=["davranış", "açıklama", "hata mesajı", "oran"])
    post_buckets = [
        {"label": k, "count": int(post_counts.get(k, 0)),
         "pct": (100.0 * post_counts.get(k, 0) / post_total) if post_total else 0.0}
        for k in post_order
    ]
    sections.append(_common.Section(
        "Hata sonrası davranış (sonraki human → asistan)",
        post_table + "\n\nGrafik: `post_error_behavior.png`.",
        blocks=[
            _common.block_bucket_chart(
                "Hata sonrası sonraki human→asistan davranışı",
                post_buckets,
                image="post_error_behavior.png",
                xlabel="davranış",
            ),
            _common.block_table(
                [
                    {"key": "behavior", "label": "davranış",   "align": "left"},
                    {"key": "desc",     "label": "açıklama",   "align": "left"},
                    {"key": "count",    "label": "hata mesajı","align": "right"},
                    {"key": "pct",      "label": "oran",       "align": "right"},
                ],
                post_table_rows,
            ),
        ] if post_total else [],
    ))

    # --- Bölüm 6: Hatalı vs hatasız konuşma kıyası
    comp_table_rows = [
        [
            g,
            _common.fmt_int(n),
            _common.fmt_int(msg_p50), _common.fmt_int(msg_p95),
            _common.fmt_int(tc_p50), _common.fmt_int(tc_p95),
            f"{int(life_p50)}sn" if life_p50 is not None else "—",
        ]
        for g, n, msg_p50, msg_p95, tc_p50, tc_p95, life_p50 in comp_rows
    ]
    comp_table = _common.markdown_table(
        comp_table_rows,
        headers=["kitle", "n", "mesaj p50", "mesaj p95", "tool p50", "tool p95", "ömür p50"],
    )
    sections.append(_common.Section(
        "Hatalı vs hatasız konuşma kıyası",
        comp_table,
        blocks=[
            _common.block_table(
                [
                    {"key": "grp",      "label": "kitle",     "align": "left"},
                    {"key": "n",        "label": "n",         "align": "right"},
                    {"key": "msg_p50",  "label": "mesaj p50", "align": "right"},
                    {"key": "msg_p95",  "label": "mesaj p95", "align": "right"},
                    {"key": "tc_p50",   "label": "tool p50",  "align": "right"},
                    {"key": "tc_p95",   "label": "tool p95",  "align": "right"},
                    {"key": "life_p50", "label": "ömür p50",  "align": "right"},
                ],
                comp_table_rows,
            ),
        ] if comp_table_rows else [],
    ))

    # --- Bölüm 7: Örnek hata metinleri (kategori başına en sık 3 normalized metin)
    sample_rows = []
    for cat in cats_order:
        for text, cnt in samples_by_cat.get(cat, Counter()).most_common(3):
            sample_rows.append([cat, _common.fmt_int(cnt), text])
    sections.append(_common.Section(
        "Örnek hata metinleri (normalize, kategori başına top 3)",
        _common.markdown_table(sample_rows, headers=["kategori", "n", "örnek"]),
        blocks=[
            _common.block_table(
                [
                    {"key": "cat",    "label": "kategori", "align": "left"},
                    {"key": "n",      "label": "n",        "align": "right"},
                    {"key": "sample", "label": "örnek",    "align": "left"},
                ],
                sample_rows,
            ),
        ] if sample_rows else [],
    ))

    # --- Bölüm 8: Notlar
    notes = [
        "- Kaynak: `tool_results.is_error` ve `tool_results.content` (JSON). `UNNEST(CAST(content AS JSON[]))` ile text blokları açılır.",
        "- Kategorizasyon regex tabanlı (yapısal anahtar kelimeler, dilden bağımsız İngilizce hata literatürüne göre). Eşleşme sırası yukarıdan aşağıya; ilk eşleşen kategori atanır. Eşleşmeyen → OTHER.",
        "- Normalize: UUID'ler `<UUID>`, yollar `<PATH>`, URL'ler `<URL>` ile değiştirilir; metin 200 karaktere kesilir.",
        "- `tool_results.stop_timestamp` tüm kayıtlarda NULL — tool latency ölçülemez, raporda hesaplanmadı.",
        "- `retry_same_tool`/`diff_tool`/`no_tool` kararı: hatalı mesajdan sonraki ilk human mesajını izleyen ilk assistant mesajının `tool_calls.name` listesi kullanılır.",
        f"- Asgari çağrı eşiği: tool başına oran tablosunda n≥{MIN_CALLS_FOR_RATE}.",
    ]
    note_bullets = [n.lstrip("- ").strip() for n in notes]
    sections.append(_common.Section(
        "Notlar",
        "\n".join(notes),
        blocks=[_common.block_bullets(note_bullets)],
    ))

    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)

    # --- Grafik 1: top tools error rate
    if HAS_MPL and rows_filt:
        names = [r[0] for r in rows_filt]
        rates = [100.0 * r[2] / r[1] for r in rows_filt]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(names))))
        y = np.arange(len(names))
        ax.barh(y, rates, color="#c0392b")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("hata oranı (%)")
        ax.set_title(f"Tool başına hata oranı (n≥{MIN_CALLS_FOR_RATE})")
        _common.save_fig(fig, out_dir / "top_tools_error_rate.png")

    # --- Grafik 2: kategori dağılımı
    if HAS_MPL and cat_counter:
        cats = [c for c, _ in cat_counter.most_common()]
        vals = [cat_counter[c] for c in cats]
        fig, ax = plt.subplots()
        ax.bar(cats, vals, color="#7f8c8d")
        ax.set_ylabel("hata sayısı")
        ax.set_title("Hata kategori dağılımı")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        _common.save_fig(fig, out_dir / "error_categories.png")

    # --- Grafik 3: post-error behavior
    if HAS_MPL and post_total:
        labels = post_order
        vals = [post_counts.get(k, 0) for k in labels]
        fig, ax = plt.subplots()
        ax.bar(labels, vals, color="#34495e")
        ax.set_ylabel("hata sonrası davranış sayısı")
        ax.set_title("Hata sonrası sonraki human→asistan davranışı")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        _common.save_fig(fig, out_dir / "post_error_behavior.png")

    # --- CSV çıktıları
    _common.write_csv(
        out_dir, "tool_error_rates.csv",
        [[n, c, e, f"{100.0*e/c:.2f}" if c else "0"] for n, c, e in tool_rates],
        headers=["tool", "calls", "errors", "err_pct"],
    )
    err_samples_rows: list[list[Any]] = []
    for cat in cats_order:
        for text, cnt in samples_by_cat.get(cat, Counter()).most_common(10):
            err_samples_rows.append([cat, cnt, text])
    _common.write_csv(
        out_dir, "error_samples.csv",
        err_samples_rows,
        headers=["category", "count", "normalized_text"],
    )

    # --- JSON summary
    summary = {
        "headline": (
            f"{_common.fmt_int(total_errs)} hata / {_common.fmt_int(total_calls)} çağrı "
            f"(%{overall_rate:.1f}); {_common.fmt_int(err_conv_count)} konuşma; "
            f"en yüksek oran: {highest_rate_tool[0] if highest_rate_tool else '—'} "
            f"(%{highest_rate_tool[3]:.1f})" if highest_rate_tool else ""
        ),
        "total_calls": int(total_calls or 0),
        "total_errors": int(total_errs or 0),
        "error_rate_pct": round(overall_rate, 2),
        "error_conversations": int(err_conv_count),
        "error_messages": int(err_msg_count),
        "highest_rate_tool": {
            "name": highest_rate_tool[0],
            "calls": int(highest_rate_tool[1]),
            "errors": int(highest_rate_tool[2]),
            "rate_pct": round(highest_rate_tool[3], 2),
        } if highest_rate_tool else None,
        "categories": [{"name": c, "count": n} for c, n in cat_counter.most_common()],
        "post_error_behavior": {k: int(post_counts.get(k, 0)) for k in post_order},
    }
    _common.write_json(out_dir, "summary.json", summary)
    return summary
