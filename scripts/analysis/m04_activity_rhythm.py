"""
m04 — Etkinlik ritmi (saat × gün, haftalık seri, aylık human/assistant)

- Heatmap: hour_of_day × day_of_week (mesaj sayısı, yerel TZ)
- Haftalık mesaj zaman serisi (iso_week başlangıcına göre)
- Aylık human vs assistant mesaj bar
- Yeni konuşma açılış saati dağılımı (hour_of_day, konuşma bazında)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from analysis import _common
from analysis._common import HAS_MPL
if HAS_MPL:
    import matplotlib.pyplot as plt

SLUG = "04-activity-rhythm"
TITLE = "Etkinlik ritmi — saat / gün / hafta"

DOW_LABELS = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]


def run(con: duckdb.DuckDBPyConnection, out_dir: Path, cfg: dict) -> dict:
    # Heatmap için: saat × dow (ISO DOW: 1=Pzt, 7=Paz)
    hm_rows = con.execute("""
        SELECT
          CAST(extract(hour FROM created_local) AS INTEGER)  AS hour,
          CAST(extract(isodow FROM created_local) AS INTEGER) AS dow,
          COUNT(*) AS n
        FROM _stats_message
        GROUP BY 1, 2
    """).fetchall()

    heat = np.zeros((7, 24), dtype=float)
    for hour, dow, n in hm_rows:
        heat[dow - 1, hour] = n

    # Haftalık zaman serisi (tüm mesajlar)
    weekly = con.execute("""
        SELECT
          date_trunc('week', created_local) AS week,
          COUNT(*) AS n
        FROM _stats_message
        GROUP BY 1
        ORDER BY 1
    """).fetchall()

    # Aylık toplam mesaj
    monthly = con.execute("""
        SELECT
          strftime(date_trunc('month', created_local), '%Y-%m') AS month,
          COUNT(*) AS total
        FROM _stats_message
        GROUP BY 1
        ORDER BY 1
    """).fetchall()

    # Yeni konuşma açılış saati
    convo_hour_rows = con.execute("""
        SELECT CAST(extract(hour FROM created_local) AS INTEGER) AS hour, COUNT(*) AS n
        FROM _stats_conversation
        WHERE message_count > 0
        GROUP BY 1
        ORDER BY 1
    """).fetchall()
    convo_hours = np.zeros(24, dtype=int)
    for h, n in convo_hour_rows:
        convo_hours[h] = n

    # Yeni konuşma açılış heatmap (saat × dow)
    convo_hm_rows = con.execute("""
        SELECT
          CAST(extract(hour FROM created_local) AS INTEGER)    AS hour,
          CAST(extract(isodow FROM created_local) AS INTEGER)  AS dow,
          COUNT(*) AS n
        FROM _stats_conversation
        WHERE message_count > 0
        GROUP BY 1, 2
    """).fetchall()
    convo_heat = np.zeros((7, 24), dtype=float)
    for hour, dow, n in convo_hm_rows:
        convo_heat[dow - 1, hour] = n

    # Günlük yeni konuşma sayısı (zaman serisi)
    daily_convo = con.execute("""
        SELECT
          date_trunc('day', created_local) AS day,
          COUNT(*) AS n
        FROM _stats_conversation
        WHERE message_count > 0
        GROUP BY 1
        ORDER BY 1
    """).fetchall()

    # En yoğun saat ve gün
    msg_per_hour = heat.sum(axis=0)
    msg_per_dow = heat.sum(axis=1)
    peak_hour = int(msg_per_hour.argmax())
    peak_dow = int(msg_per_dow.argmax())
    total_msgs = int(heat.sum())

    if HAS_MPL:
        # Grafik 1: heatmap
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(heat, aspect="auto", cmap="YlOrRd", origin="upper")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)])
        ax.set_yticks(range(7))
        ax.set_yticklabels(DOW_LABELS)
        ax.set_xlabel("saat (yerel, Europe/Istanbul)")
        ax.set_ylabel("gün")
        ax.set_title("Mesaj heatmap — saat × gün")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("mesaj sayısı")
        ax.grid(False)
        _common.save_fig(fig, out_dir / "heatmap_hour_dow.png")

        # Grafik 2: haftalık seri
        fig, ax = plt.subplots()
        if weekly:
            weeks = [r[0] for r in weekly]
            counts = [r[1] for r in weekly]
            ax.plot(weeks, counts, color="#4C72B0", linewidth=1.2, marker="o", markersize=3)
            ax.set_xlabel("hafta")
            ax.set_ylabel("mesaj sayısı")
            ax.set_title("Haftalık mesaj hacmi")
            fig.autofmt_xdate()
        _common.save_fig(fig, out_dir / "weekly_timeseries.png")

        # Grafik 3: aylık toplam mesaj
        fig, ax = plt.subplots(figsize=(12, 5))
        if monthly:
            months = [r[0] for r in monthly]
            totals_m = np.array([r[1] for r in monthly], dtype=float)
            x = np.arange(len(months))
            ax.bar(x, totals_m, color="#4C72B0")
            ax.set_xticks(x)
            ax.set_xticklabels(months, rotation=45, ha="right")
            ax.set_ylabel("mesaj sayısı")
            ax.set_title("Ay bazlı toplam mesaj")
        _common.save_fig(fig, out_dir / "monthly_total.png")

        # Grafik 4: yeni konuşma açılış saati
        fig, ax = plt.subplots()
        ax.bar(range(24), convo_hours, color="#C44E52", alpha=0.8)
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)])
        ax.set_xlabel("saat (yerel)")
        ax.set_ylabel("yeni konuşma")
        ax.set_title("Yeni konuşma açılış saati dağılımı")
        _common.save_fig(fig, out_dir / "new_conversation_hour.png")

        # Grafik 5: yeni konuşma heatmap (saat × gün)
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(convo_heat, aspect="auto", cmap="PuRd", origin="upper")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)])
        ax.set_yticks(range(7))
        ax.set_yticklabels(DOW_LABELS)
        ax.set_xlabel("saat (yerel, Europe/Istanbul)")
        ax.set_ylabel("gün")
        ax.set_title("Yeni konuşma heatmap — saat × gün")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("yeni konuşma sayısı")
        ax.grid(False)
        _common.save_fig(fig, out_dir / "heatmap_new_convo_hour_dow.png")

    total_m = msg_per_hour.sum()
    total_c = convo_hours.sum()
    msg_pct = (msg_per_hour / total_m * 100.0) if total_m else np.zeros(24)
    convo_pct = (convo_hours / total_c * 100.0) if total_c else np.zeros(24)

    if HAS_MPL:
        fig, ax = plt.subplots()
        ax.plot(range(24), msg_pct, marker="o", color="#4C72B0",
                linewidth=1.5, label="mesaj (%)")
        ax.plot(range(24), convo_pct, marker="s", color="#C44E52",
                linewidth=1.5, label="yeni konuşma (%)")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)])
        ax.set_xlabel("saat (yerel)")
        ax.set_ylabel("saatlik pay (%)")
        ax.set_title("Saat bazında mesaj vs yeni konuşma — % dağılımı")
        ax.legend()
        _common.save_fig(fig, out_dir / "hourly_messages_vs_new_convos.png")

        # Grafik 7: günlük yeni konuşma zaman serisi
        fig, ax = plt.subplots()
        if daily_convo:
            days = [r[0] for r in daily_convo]
            counts_d = [r[1] for r in daily_convo]
            ax.plot(days, counts_d, color="#C44E52", linewidth=0.9)
            ax.set_xlabel("gün")
            ax.set_ylabel("yeni konuşma sayısı")
            ax.set_title("Günlük yeni konuşma sayısı")
            fig.autofmt_xdate()
        _common.save_fig(fig, out_dir / "daily_new_conversations.png")

    # CSV: heatmap (long form)
    heat_csv = []
    for d in range(7):
        for h in range(24):
            heat_csv.append((DOW_LABELS[d], h, int(heat[d, h])))
    _common.write_csv(out_dir, "heatmap.csv", heat_csv, headers=["dow", "hour", "n"])

    _common.write_csv(
        out_dir, "monthly.csv",
        [(m, int(t)) for m, t in monthly],
        headers=["month", "messages"],
    )

    _common.write_csv(
        out_dir, "weekly.csv",
        [(str(w), int(n)) for w, n in weekly],
        headers=["week", "messages"],
    )

    _common.write_csv(
        out_dir, "daily_new_conversations.csv",
        [(str(d), int(n)) for d, n in daily_convo],
        headers=["day", "new_conversations"],
    )

    _common.write_csv(
        out_dir, "new_conversation_heatmap.csv",
        [(DOW_LABELS[d], h, int(convo_heat[d, h])) for d in range(7) for h in range(24)],
        headers=["dow", "hour", "new_conversations"],
    )

    # Konuşma açılış saati tablosu
    convo_hour_table_rows = []
    total_convos = int(convo_hours.sum())
    for h in range(24):
        n = int(convo_hours[h])
        pct = (100.0 * n / total_convos) if total_convos else 0.0
        convo_hour_table_rows.append([f"{h:02d}", _common.fmt_int(n), f"{pct:.1f}%"])

    summary = {
        "total_messages": total_msgs,
        "peak_hour": peak_hour,
        "peak_hour_messages": int(msg_per_hour[peak_hour]),
        "peak_dow": DOW_LABELS[peak_dow],
        "peak_dow_messages": int(msg_per_dow[peak_dow]),
        "messages_per_hour": {f"{h:02d}": int(msg_per_hour[h]) for h in range(24)},
        "messages_per_dow": {DOW_LABELS[d]: int(msg_per_dow[d]) for d in range(7)},
        "new_conversations_per_hour": {f"{h:02d}": int(convo_hours[h]) for h in range(24)},
        "headline": (
            f"en yoğun saat {peak_hour:02d}:00 ({_common.fmt_int(msg_per_hour[peak_hour])} mesaj), "
            f"en yoğun gün {DOW_LABELS[peak_dow]} ({_common.fmt_int(msg_per_dow[peak_dow])} mesaj); "
            f"hafta sonu payı "
            f"%{100*(msg_per_dow[5]+msg_per_dow[6])/max(total_msgs,1):.1f}."
        ),
    }
    _common.write_json(out_dir, "summary.json", summary)

    # Raporlar
    hour_table = _common.markdown_table(
        [[f"{h:02d}", _common.fmt_int(msg_per_hour[h])] for h in range(24)],
        headers=["saat", "mesaj"],
    )
    dow_table = _common.markdown_table(
        [[DOW_LABELS[d], _common.fmt_int(msg_per_dow[d])] for d in range(7)],
        headers=["gün", "mesaj"],
    )
    convo_hour_table = _common.markdown_table(
        convo_hour_table_rows, headers=["saat", "yeni konuşma", "oran"]
    )

    # ── Structured blocks ──
    highlight_items = [
        f"Toplam mesaj: **{_common.fmt_int(total_msgs)}**",
        f"En yoğun saat: **{peak_hour:02d}:00** "
        f"({_common.fmt_int(msg_per_hour[peak_hour])} mesaj)",
        f"En yoğun gün: **{DOW_LABELS[peak_dow]}** "
        f"({_common.fmt_int(msg_per_dow[peak_dow])} mesaj)",
        f"Hafta sonu payı: "
        f"**%{100*(msg_per_dow[5]+msg_per_dow[6])/max(total_msgs,1):.1f}**",
    ]

    # Saat bazında mesaj — 24-slot bucket_bar (pct yardımcı olsun)
    hour_buckets = [
        {"label": f"{h:02d}", "count": int(msg_per_hour[h]),
         "pct": (100.0 * msg_per_hour[h] / total_msgs) if total_msgs else 0.0}
        for h in range(24)
    ]
    hour_columns = [
        {"key": "hour", "label": "saat",  "align": "left"},
        {"key": "n",    "label": "mesaj", "align": "right"},
    ]
    hour_rows = [[f"{h:02d}", _common.fmt_int(msg_per_hour[h])] for h in range(24)]

    # Gün bazında mesaj — 7-slot
    dow_buckets = [
        {"label": DOW_LABELS[d], "count": int(msg_per_dow[d]),
         "pct": (100.0 * msg_per_dow[d] / total_msgs) if total_msgs else 0.0}
        for d in range(7)
    ]
    dow_columns = [
        {"key": "dow", "label": "gün",   "align": "left"},
        {"key": "n",   "label": "mesaj", "align": "right"},
    ]
    dow_rows = [[DOW_LABELS[d], _common.fmt_int(msg_per_dow[d])] for d in range(7)]

    # Yeni konuşma açılış saati
    new_convo_hour_buckets = [
        {"label": f"{h:02d}", "count": int(convo_hours[h]),
         "pct": (100.0 * convo_hours[h] / total_convos) if total_convos else 0.0}
        for h in range(24)
    ]
    new_convo_hour_columns = [
        {"key": "hour", "label": "saat",         "align": "left"},
        {"key": "n",    "label": "yeni konuşma", "align": "right"},
        {"key": "pct",  "label": "oran",         "align": "right"},
    ]

    # Ay bazlı toplam mesaj
    monthly_buckets = [
        {"label": m, "count": int(t), "pct": None}
        for m, t in monthly
    ]
    hours = [f"{h:02d}" for h in range(24)]
    heat_cells = [
        {"x": f"{h:02d}", "y": DOW_LABELS[d], "value": int(heat[d, h])}
        for d in range(7) for h in range(24)
    ]
    convo_heat_cells = [
        {"x": f"{h:02d}", "y": DOW_LABELS[d], "value": int(convo_heat[d, h])}
        for d in range(7) for h in range(24)
    ]
    hourly_compare = [
        {
            "hour": f"{h:02d}",
            "messages_pct": float(msg_pct[h]),
            "new_conversations_pct": float(convo_pct[h]),
        }
        for h in range(24)
    ]
    weekly_series = [
        {"week": str(w), "messages": int(n)}
        for w, n in weekly
    ]
    daily_series = [
        {"day": str(d), "new_conversations": int(n)}
        for d, n in daily_convo
    ]

    notes_items = [
        f"Tüm saat/gün bilgisi yerel saate (**{cfg.get('tz', _common.DEFAULT_TZ)}**) "
        "çevrilerek hesaplanmıştır.",
        "ISO day-of-week: 1=Pazartesi, 7=Pazar.",
    ]

    sections = [
        _common.Section(
            "Özet",
            f"- Toplam mesaj: **{_common.fmt_int(total_msgs)}**\n"
            f"- En yoğun saat: **{peak_hour:02d}:00** "
            f"({_common.fmt_int(msg_per_hour[peak_hour])} mesaj)\n"
            f"- En yoğun gün: **{DOW_LABELS[peak_dow]}** "
            f"({_common.fmt_int(msg_per_dow[peak_dow])} mesaj)\n"
            f"- Hafta sonu payı: "
            f"**%{100*(msg_per_dow[5]+msg_per_dow[6])/max(total_msgs,1):.1f}**",
            blocks=[_common.block_bullets(highlight_items)],
        ),
        _common.Section(
            "Saat × gün heatmap",
            "",
            blocks=[_common.block_heatmap_chart(
                "Mesaj heatmap — saat × gün",
                heat_cells,
                x_labels=hours,
                y_labels=DOW_LABELS,
            )],
        ),
        _common.Section(
            "Saat bazında mesaj",
            hour_table,
            blocks=[
                _common.block_bucket_chart(
                    label="Saat bazında mesaj dağılımı",
                    buckets=hour_buckets,
                    xlabel="saat (yerel)",
                ),
                _common.block_table(hour_columns, hour_rows),
            ],
        ),
        _common.Section(
            "Gün bazında mesaj",
            dow_table,
            blocks=[
                _common.block_bucket_chart(
                    label="Gün bazında mesaj dağılımı",
                    buckets=dow_buckets,
                    xlabel="gün (ISO Pzt–Paz)",
                ),
                _common.block_table(dow_columns, dow_rows),
            ],
        ),
        _common.Section(
            "Ay bazlı toplam mesaj",
            "",
            blocks=[
                _common.block_bucket_chart(
                    label="Ay bazlı toplam mesaj",
                    buckets=monthly_buckets,
                    image="monthly_total.png",
                    xlabel="ay",
                ),
            ],
        ),
        _common.Section(
            "Yeni konuşma açılış saati",
            convo_hour_table,
            blocks=[
                _common.block_bucket_chart(
                    label="Yeni konuşma açılış saati dağılımı",
                    buckets=new_convo_hour_buckets,
                    image="new_conversation_hour.png",
                    xlabel="saat (yerel)",
                ),
                _common.block_table(new_convo_hour_columns, convo_hour_table_rows),
            ],
        ),
        _common.Section(
            "Yeni konuşma heatmap (saat × gün)",
            "",
            blocks=[_common.block_heatmap_chart(
                "Yeni konuşma heatmap — saat × gün",
                convo_heat_cells,
                x_labels=hours,
                y_labels=DOW_LABELS,
            )],
        ),
        _common.Section(
            "Saat bazında mesaj vs yeni konuşma",
            "",
            blocks=[_common.block_line_chart(
                "Saat bazında mesaj vs yeni konuşma — % dağılımı",
                hourly_compare,
                x_key="hour",
                series=[
                    {"key": "messages_pct", "label": "mesaj (%)"},
                    {"key": "new_conversations_pct", "label": "yeni konuşma (%)"},
                ],
            )],
        ),
        _common.Section(
            "Haftalık mesaj hacmi",
            "",
            blocks=[_common.block_line_chart(
                "Haftalık mesaj hacmi",
                weekly_series,
                x_key="week",
                series=[{"key": "messages", "label": "mesaj"}],
            )],
        ),
        _common.Section(
            "Günlük yeni konuşma",
            "",
            blocks=[_common.block_line_chart(
                "Günlük yeni konuşma sayısı",
                daily_series,
                x_key="day",
                series=[{"key": "new_conversations", "label": "yeni konuşma"}],
            )],
        ),
        _common.Section(
            "Grafikler",
            "- `heatmap_hour_dow.png` — saat × gün heatmap (mesaj)\n"
            "- `weekly_timeseries.png` — haftalık mesaj hacmi\n"
            "- `monthly_total.png` — ay bazlı toplam mesaj\n"
            "- `new_conversation_hour.png` — yeni konuşma açılış saati\n"
            "- `heatmap_new_convo_hour_dow.png` — saat × gün heatmap (yeni konuşma)\n"
            "- `hourly_messages_vs_new_convos.png` — saat bazında mesaj vs yeni "
            "konuşma payı (% normalize)\n"
            "- `daily_new_conversations.png` — günlük yeni konuşma sayısı",
            blocks=[],
        ),
        _common.Section(
            "Notlar",
            f"- Tüm saat/gün bilgisi yerel saate (**{cfg.get('tz', _common.DEFAULT_TZ)}**) "
            "çevrilerek hesaplanmıştır.\n"
            "- ISO day-of-week: 1=Pazartesi, 7=Pazar.",
            blocks=[_common.block_bullets(notes_items)],
        ),
    ]
    _common.write_report(out_dir, TITLE, sections)
    _common.write_sections(out_dir, TITLE, sections)
    return summary
