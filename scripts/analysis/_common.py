"""
Common helpers: DB connection, chart style, markdown table,
percentile, report writing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import numpy as np
from tabulate import tabulate

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    plt = None  # type: ignore[assignment]
    HAS_MPL = False

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "build" / "claude.duckdb"
REPORTS_DIR = ROOT / "reports" / "quantitative"
DEFAULT_TZ = "Europe/Istanbul"

_LOCALE_KEYS = {"tr", "en"}
_SKIP_LOCALIZE_KEYS = {"body_md"}

_REPORT_HEADERS_EN = {
    "Öne çıkanlar": "Highlights",
    "Özet": "Summary",
    "Dağılım — quantile": "Distribution — quantile",
    "Mesaj sayısı kovaları": "Message count buckets",
    "Human turn kovaları": "Human turn buckets",
    "Token kovaları": "Token buckets",
    "Token kovaları — human": "Token buckets — human",
    "Token kovaları — assistant": "Token buckets — assistant",
    "Token dağılımı — quantile (tokens_text>0)": "Token distribution — quantile (tokens_text>0)",
    "İlk mesaj vs takip mesajı (human)": "First vs follow-up message (human)",
    "Assistant cevabında kod bloğu payı": "Code-block share in assistant responses",
    "Grafikler": "Charts",
    "Notlar": "Notes",
    "Cevap uzunluğu — tokens (thinking yok vs var)": "Response length — tokens (without vs with thinking)",
    "Thinking bloğu uzunluğu": "Thinking-block length",
    "Cevapta thinking payı": "Thinking share within response",
    "Konuşma başına thinking token yükü": "Per-conversation thinking-token load",
    "Thinking × tool kullanımı çakışması (konuşma bazında)": "Thinking × tool-use overlap (per conversation)",
    "İlk assistant mesajı vs sonraki mesajlar — thinking oranı": "First assistant message vs the rest — thinking rate",
    "Thinking kovası → cevap token (kondense)": "Thinking bucket → response tokens (condensed)",
    "Saat bazında mesaj": "Messages per hour",
    "Gün bazında mesaj": "Messages per day-of-week",
    "Yeni konuşma açılış saati": "New-conversation start hour",
    "Konuşma başına tool sayısı — quantile": "Tool calls per conversation — quantile",
    "Konuşma kovaları": "Conversation buckets",
    "Tool kullanan vs kullanmayan — konuşma uzunluğu": "Tool users vs non-users — conversation length",
    "Konuşma başına tool çeşitliliği": "Tool diversity per conversation",
    "Tool kategori dağılımı": "Tool category breakdown",
    "En çok kullanılan 15 tool": "Top 15 tools",
    "Tool sonuç hata oranı": "Tool-result error rate",
    "Tool çağrı süresi": "Tool-call duration",
    "Ömür dağılımı (konuşma başına)": "Lifetime distribution (per conversation)",
    "Ömür kovaları": "Lifetime buckets",
    "Mesaj-arası gerçek gap": "Real inter-message gap",
    "Gap kovaları": "Gap buckets",
    "İlk-yanıt gecikmesi (human → assistant)": "First-response latency (human → assistant)",
    "İlk-yanıt kovaları": "First-response buckets",
    "Ömür × mesaj sayısı": "Lifetime × message count",
    "Attachment boyut dağılımı": "Attachment size distribution",
    "Boyut kovaları": "Size buckets",
    "Konuşma başına attachment/file sayısı": "Attachments/files per conversation",
    "Konuşma başına kovalar — attachment": "Per-conversation buckets — attachment",
    "Konuşma başına kovalar — file": "Per-conversation buckets — file",
    "Attachment/file'lı vs yok — konuşma uzunluğu": "With vs without attachment/file — conversation length",
    "En yaygın file_type (top 8)": "Most common file_type (top 8)",
    "file_type × boyut (top 6)": "file_type × size (top 6)",
    "Project başına doc sayısı": "Docs per project",
    "Doc content_length (karakter)": "Doc content_length (characters)",
    "prompt_template uzunluğu (dolu olanlar)": "prompt_template length (non-empty)",
    "Doc sayısı kovaları": "Doc count buckets",
    "Doc uzunluk kovaları": "Doc length buckets",
    "prompt_template kovaları": "prompt_template buckets",
    "Konuşma başına artifact çağrısı": "Artifact calls per conversation",
    "Konuşma başına iterasyon (update + rewrite)": "Iterations per conversation (update + rewrite)",
    "Artifact content uzunluğu": "Artifact content length",
    "Type dağılımı": "Type breakdown",
    "Command dağılımı": "Command breakdown",
    "Type × command çapraz": "Type × command cross",
    "Code artifact'lerinde language (top 10)": "Language in code artifacts (top 10)",
    "Content uzunluğu kovaları": "Content length buckets",
    "Artifact'li vs yok — konuşma uzunluğu": "With vs without artifact — conversation length",
    "Fenced blok boyut dağılımı (karakter)": "Fenced-block size distribution (characters)",
    "Konuşma başına fenced blok sayısı": "Fenced blocks per conversation",
    "Human vs assistant — fenced blok": "Human vs assistant — fenced block",
    "Blok boyut kovaları": "Block size buckets",
    "Dil dağılımı (top 15)": "Language breakdown (top 15)",
    "Konuşma kod payı × mesaj sayısı": "Conversation code-share × message count",
    "Inline backtick kullanımı": "Inline backtick usage",
    "Konuşma başına citation dağılımı": "Citation distribution per conversation",
    "Konuşma başına kova": "Per-conversation buckets",
    "Citation tipi dağılımı": "Citation type breakdown",
    "Top 20 domain": "Top 20 domains",
    "Citation × web tool çağrı ilişkisi": "Citation × web-tool call relationship",
    "Citation'lı vs citationsız konuşma kıyası": "With vs without citation — conversation comparison",
    "Aylık trend": "Monthly trend",
    "Hata kategori dağılımı": "Error category breakdown",
    "Kategori × tool (top 10 hata üreten tool)": "Category × tool (top 10 error-producing tools)",
    "Hata sonrası davranış (sonraki human → asistan)": "Post-error behavior (next human → assistant)",
    "Hatalı vs hatasız konuşma kıyası": "Error vs error-free conversation comparison",
    "Örnek hata metinleri (normalize, kategori başına top 3)": "Sample error texts (normalized, top 3 per category)",
    "Nasıl çalışır — Claude.ai müdahale mekaniği": "How it works — Claude.ai intervention mechanics",
    "Prompt yeniden yazma (human fork)": "Prompt rewriting (human fork)",
    "Cevap yeniden üretme (assistant fork / retry)": "Response regeneration (assistant fork / retry)",
    "Cevap in-place düzenleme (assistant)": "Response in-place edit (assistant)",
    "Çok-root konuşmalar": "Multi-root conversations",
    "Müdahale yoğunluğu — bayrak kombinasyonları": "Intervention density — flag combinations",
    "Top konuşmalar — her mekaniğin en yoğun 3 örneği": "Top conversations — 3 heaviest per mechanic",
    "Ay bazlı toplam mesaj": "Total messages per month",
    "Ay bazında attachment + file": "Monthly attachment + file",
    "Ay bazında attachment'lı konuşma oranı": "Monthly share of attachment-bearing conversations",
    "Ay bazında yeni project oluşturma": "Monthly new project creation",
    "Aylık thinking oranı (konuşma bazında)": "Monthly thinking rate (per conversation)",
    "Günlük yeni konuşma": "Daily new conversations",
    "Haftalık mesaj hacmi": "Weekly message volume",
    "Saat bazında mesaj vs yeni konuşma": "Messages vs new conversations by hour",
    "Saat × gün heatmap": "Hour × day heatmap",
    "Tool başına hata oranı (n≥20, top 20)": "Per-tool error rate (n≥20, top 20)",
    "Top-4 file_type — aylık trend": "Top-4 file_type — monthly trend",
    "Yeni konuşma heatmap (saat × gün)": "New-conversation heatmap (hour × day)",
    "Üst-5 tool — aylık trend": "Top-5 tools — monthly trend",
}

_ANALYSIS_TEXT_EN = {
    "Pazartesi": "Monday",
    "Salı": "Tuesday",
    "Çarşamba": "Wednesday",
    "Perşembe": "Thursday",
    "Cuma": "Friday",
    "Cumartesi": "Saturday",
    "Pazar": "Sunday",
    "Pzt": "Mon",
    "Sal": "Tue",
    "Çar": "Wed",
    "Per": "Thu",
    "Cum": "Fri",
    "Cmt": "Sat",
    "Paz": "Sun",
    "_Hata kaydı yok._": "_No error records._",
    "code artifact yok": "No code artifact",
    "Öne çıkanlar": "Highlights",
    "Notlar": "Notes",
    "Grafikler": "Charts",
    "Metrik": "Metric",
    "metrik": "metric",
    "Ort": "Mean",
    "ort": "mean",
    "Konuşma": "Conversation",
    "konuşma": "conversation",
    "Mesaj": "Message",
    "mesaj": "message",
    "Oran": "Share",
    "oran": "share",
    "Oranı": "Rate",
    "oranı": "rate",
    "Aralık": "Range",
    "aralık": "range",
    "Adet": "Count",
    "adet": "count",
    "Sayı": "Count",
    "sayı": "count",
    "Boy": "Size",
    "boy": "size",
    "Boyut": "Size",
    "boyut": "size",
    "Saat": "Hour",
    "saat": "hour",
    "Gün": "Day",
    "gün": "day",
    "Ay": "Month",
    "ay": "month",
    "Hafta": "Week",
    "hafta": "week",
    "Çağrı": "Calls",
    "çağrı": "calls",
    "Hata": "Errors",
    "hata": "errors",
    "Kategori": "Category",
    "kategori": "category",
    "Davranış": "Behavior",
    "davranış": "behavior",
    "Açıklama": "Description",
    "açıklama": "description",
    "Örnek": "Sample",
    "örnek": "sample",
    "Değer": "Value",
    "değer": "value",
    "saat (yerel)": "hour (local time)",
    "gün (ISO Pzt–Paz)": "day (ISO Mon–Sun)",
    "Mesaj sayısı": "Message count",
    "mesaj sayısı": "message count",
    "Konuşma sayısı": "Conversation count",
    "konuşma sayısı": "conversation count",
    "Citation sayısı": "Citation count",
    "citation sayısı": "citation count",
    "Süre": "Duration",
    "süre": "duration",
    "Saniye": "Seconds",
    "saniye": "seconds",
    "Dakika": "Minutes",
    "dakika": "minutes",
    "Yüzde": "Percent",
    "yüzde": "percent",
    "blok sayısı": "block count",
    "bloklu mesaj": "messages with block",
    "toplam mesaj": "total messages",
    "Attachment — aylık sayı": "Attachments — monthly count",
    "File — aylık sayı": "Files — monthly count",
    "Attachment/file içeren konuşma oranı (%)": "Share of conversations with attachment/file (%)",
    "Attachment boyut kovaları": "Attachment-size buckets",
    "Attachment file_type dağılımı (top 8)": "Attachment file_type breakdown (top 8)",
    "Konuşma ömrü kovaları": "Conversation-lifetime buckets",
    "Mesaj-arası gerçek gap dağılımı": "Real inter-message gap distribution",
    "İlk-yanıt gecikmesi dağılımı": "First-response latency distribution",
    "Saat bazında mesaj dağılımı": "Message distribution by hour",
    "Gün bazında mesaj dağılımı": "Message distribution by day",
    "Ay bazlı toplam mesaj": "Monthly total messages",
    "Yeni konuşma açılış saati dağılımı": "New-conversation start-hour distribution",
    "Thinking içeren konuşma oranı — ay bazlı (%)": "Monthly share of conversations with thinking (%)",
    "Tool kategori dağılımı": "Tool category breakdown",
    "**Hata üreten üst-15 tool:**": "**Top 15 error-producing tools:**",
    "**Konuşma başına edit noktası dağılımı**": "**Edit points per conversation**",
    "**Konuşma başına retry noktası dağılımı**": "**Retry points per conversation**",
    "**Prompt yeniden yazma (human fork) — en çok edit**": "**Prompt rewriting (human fork) — most edits**",
    "**Cevap yeniden üretme (retry) — en çok retry**": "**Response regeneration (retry) — most retries**",
    "**In-place edit — en çok düzenlenen asistan mesajına sahip**": "**In-place edit — most edited assistant message**",
    "**Toplam müdahale noktası — en çok edit + retry noktası**": "**Total intervention points — most edits + retries**",
    "pay": "ratio",
    "kitle": "cohort",
    "kesişim": "intersection",
    "koşullu oran": "conditional rate",
    "n konuşma": "conversations",
    "tüm konuşmaların oranı": "share of all conversations",
    "çift sayısı": "pair count",
    "geçiş sayısı": "transition count",
    "toplam": "total",
    "toplamdaki oran": "share of total",
    "farklı konuşma": "distinct conversations",
    "üyeler": "members",
    "token aralığı": "token range",
    "mesaj aralığı": "message range",
    "turn aralığı": "turn range",
    "tool aralığı": "tool range",
    "thinking aralığı": "thinking range",
    "thinking kovası": "thinking bucket",
    "süre aralığı": "duration range",
    "ömür aralığı": "lifetime range",
    "gap aralığı": "gap range",
    "gecikme aralığı": "latency range",
    "boyut aralığı": "size range",
    "adet aralığı": "count range",
    "citation aralığı": "citation range",
    "mesaj sayısı (aralık)": "message-count range",
    "human turn sayısı (aralık)": "human-turn count range",
    "toplam token (aralık)": "total-token range",
    "token (aralık)": "token range",
    "cevap token (aralık)": "response-token range",
    "thinking token (aralık)": "thinking-token range",
    "thinking / (thinking + text) oranı": "thinking / (thinking + text) ratio",
    "toplam thinking token (aralık)": "total thinking-token range",
    "toplam çağrı": "total calls",
    "web tool çağrısı": "web-tool calls",
    "kod payı": "code share",
    "mesaj p50": "message p50",
    "mesaj p95": "message p95",
    "token p50": "token p50",
    "token p95": "token p95",
    "tool p50": "tool p50",
    "tool p95": "tool p95",
    "ömür p50": "lifetime p50",
    "boy p50": "size p50",
    "boy p95": "size p95",
    "boy max": "size max",
    "cevap p50": "response p50",
    "cevap p90": "response p90",
    "cevap max": "response max",
    "edit noktası": "edit points",
    "retry noktası": "retry points",
    "tek noktada max yazım": "max rewrites at one point",
    "tek noktada max yeniden üretim": "max regenerations at one point",
    "toplam nokta": "total points",
    "uzunluk değişimi": "length change",
    "uzayan": "longer",
    "kısalan": "shorter",
    "değişmeyen": "unchanged",
    "delta": "delta",
    "pozisyon": "position",
    "blok": "blocks",
    "blok oran": "block share",
    "toplam text-mesaj": "total text messages",
    "kod token": "code tokens",
    "inline'lı mesaj": "messages with inline code",
    "toplam inline": "total inline",
    "human ilk mesaj": "human first message",
    "human takip mesajı": "human follow-up message",
    "thinking yok": "without thinking",
    "thinking var": "with thinking",
    "thinking var, tool var": "thinking yes, tool yes",
    "thinking var, tool yok": "thinking yes, tool no",
    "thinking yok, tool var": "thinking no, tool yes",
    "thinking yok, tool yok": "thinking no, tool no",
    "yok n": "without n",
    "yok oran": "without share",
    "var n": "with n",
    "var oran": "with share",
    "ilk assistant mesajı": "first assistant message",
    "sonraki assistant mesajları": "later assistant messages",
    "hafta sonu payı": "weekend share",
    "Hafta sonu payı:": "Weekend share:",
    "diğer": "other",
    "0sn (aynı timestamp)": "0s (same timestamp)",
    "1–7gün": "1–7 days",
    ">7gün": ">7 days",
    ">1gün": ">1 day",
    "ardışık ama aynı timestamp": "consecutive but same timestamp",
    "human→assistant geçiş sayısı": "human→assistant transitions",
    "ilk-yanıt": "first response",
    "süre (saniye)": "duration (seconds)",
    "farklı tool": "distinct tools",
    "thinking token aralığı": "thinking-token range",
    "Boş konuşmalar quantile hesaplamasından dışlanmıştır.": "Empty conversations are excluded from quantile calculations.",
    "Top 15 tool": "Top 15 tools",
    "`message_count` tüm mesajları (human + assistant + tool-only) sayar; `human_turn_count` yalnız `sender='human'` mesajları — yani saf talep sayısı.": "`message_count` counts all messages (human + assistant + tool-only); `human_turn_count` counts only `sender='human'` messages, i.e. pure prompt count.",
    "Token sayımı `tiktoken.cl100k_base` (OpenAI) ile yapılmıştır; Claude'un gerçek tokenizer'ı değildir — **yaklaşık üst sınır** olarak okunmalıdır.": "Token counting uses `tiktoken.cl100k_base` (OpenAI); it is not Claude's real tokenizer and should be read as an **approximate upper bound**.",
    "cevabın ne kadarı kod (token rate)": "how much of the response is code (token share)",
    "Voice_note içeren messages sayısı: **0** — örnekleme istatistik için çok küçük; ayrı analiz yapılmamıştır.": "Messages with voice_note: **0** — the sample is too small for a separate statistical analysis.",
    "Token sayımı **sadece** `content_blocks.type='text'` (ve voice_note) için yapılır; thinking blokları m03'te ele alınır.": "Token counting is performed **only** for `content_blocks.type='text'` (and voice_note); thinking blocks are covered in m03.",
    "Tool-only messages = `chars_text=0` — yalnız tool_use / tool_result bloğu taşıyan mesajlar; dağılım hesabından dışlanır.": "Tool-only messages = `chars_text=0` — messages that contain only tool_use / tool_result blocks; they are excluded from the distribution calculation.",
    "Code block = üçlü-backtick ile çevrili fenced block; kapanışı olmayan bloklar metin sonuna kadar sayılır. Inline backtick (`` ` ``) dahil değildir.": "Code block = a fenced block delimited by triple backticks; unterminated blocks are counted until the end of the text. Inline backticks (`` ` ``) are excluded.",
    "thinking-li konuşmalarda tool use": "tool use in conversations with thinking",
    "thinking-siz konuşmalarda tool use": "tool use in conversations without thinking",
    "with thinking konuşmalarda tool use": "tool use in conversations with thinking",
    "without thinking konuşmalarda tool use": "tool use in conversations without thinking",
    "tool kullanan konuşmalarda thinking rate": "thinking rate in tool-using conversations",
    "tool kullanmayan konuşmalarda thinking rate": "thinking rate in non-tool conversations",
    "`has_thinking`: conversations içinde en az bir assistant mesajının `chars_thinking > 0` olması.": "`has_thinking`: at least one assistant message in the conversation has `chars_thinking > 0`.",
    "Cevap length hesabı yalnız `tokens_text > 0` assistant mesajları üzerindendir.": "Response-length calculations use only assistant messages with `tokens_text > 0`.",
    "Thinking share = `tokens_thinking / (tokens_thinking + tokens_text)`; paydada assistant mesajının text'i sıfırsa (yalnız tool-use / tool-result) hesap dışı kalır.": "Thinking share = `tokens_thinking / (tokens_thinking + tokens_text)`; if the assistant message text is zero in the denominator (tool-use / tool-result only), it is excluded.",
    "İlk assistant mesajı = konuşmada `sender='assistant'` mesajlar arasında `created_at`'e göre sıralanınca 1. sırada olan.": "First assistant message = the one ranked first by `created_at` among `sender='assistant'` messages in the conversation.",
    "Mesaj heatmap — hour × day": "Message heatmap — hour × day",
    "tool call (aralık)": "tool-call range",
    "doc aralığı": "doc range",
    "length aralığı": "length range",
    "calls aralığı": "call range",
    "blok aralığı": "block range",
    "Ömür × messages sayısı heatmap": "Lifetime × message-count heatmap",
    "Saat bazında messages vs yeni conversations — % distribution": "Messages vs new conversations by hour — % distribution",
    "Günlük yeni conversations sayısı": "Daily new conversation count",
    "Konuşma başına distinct tool count — bucket distribution (n=4)": "Distinct tools per conversation — bucket distribution (n=4)",
    "tool_calls/conversations (tümü)": "tool calls/conversation (all)",
    "farklı tool/conversations (tool users)": "distinct tools/conversation (tool users)",
    "dosya yazma": "file write",
    "dosya okuma": "file read",
    "Kategori eşleme, MCP prefix (`filesystem:` gibi) atılıp temel ada bakar; kategorize edilemeyen tool'lar `diğer`'e düşer.": "Category mapping strips the MCP prefix (such as `filesystem:`) and uses the base name; uncategorized tools fall into `other`.",
    "Tool çeşitliliği = konuşmadaki distinct `tool_calls.name` sayısı. Tool kullanmayan konuşmalar `0` kovasında.": "Tool diversity = the count of distinct `tool_calls.name` values in the conversation. Conversations without tool use fall into the `0` bucket.",
    "Tool süresi = `stop_timestamp − start_timestamp`. Timestamp'i eksik çağrılar hesap dışı; negatif süreli kayıtlar da atılır.": "Tool duration = `stop_timestamp - start_timestamp`. Calls with missing timestamps are excluded; records with negative duration are dropped as well.",
    "Hata rate `tool_results.is_error=true` üzerinden hesaplanır.": "Error rate is computed from `tool_results.is_error=true`.",
    "`lifetime = last_msg_at - first_msg_at`; tek mesajlı konuşmalar dışlanır.": "`lifetime = last_msg_at - first_msg_at`; single-message conversations are excluded.",
    "Inter-message gap `LAG(created_at)` pencere fonksiyonuyla ardışık `sender IN ('human','assistant')` messages çiftleri üzerinden hesaplanır; tool-only mesajlar sıralamaya girmez.": "Inter-message gap is computed with the `LAG(created_at)` window function over consecutive `sender IN ('human','assistant')` message pairs; tool-only messages do not enter the sequence.",
    "First-response latency yalnızca `prev_sender='human' AND sender='assistant'` geçişlerinde ölçülür.": "First-response latency is measured only on `prev_sender='human' AND sender='assistant'` transitions.",
    "Çok-oturumlu conversations = içinde en az bir ardışık gap > 1 hour olan conversations.": "Multi-session conversations = conversations with at least one consecutive gap > 1 hour.",
    "Çarpık dağılımlar için ortalama (mean) yanıltıcı; tablolarda yalnız quantile verilir.": "For skewed distributions the mean is misleading; tables report quantiles only.",
    "Boyut bilgisi eksik kayıt sayısı: **0** (total içinde).": "Rows with missing size metadata: **0** (included in the total).",
    "`attachments` = ekran yapıştırması / yüklenen metin dosyası (içerik DB'de).": "`attachments` = pasted screen content / uploaded text file (content lives in the DB).",
    "`files` = sadece referans (uuid + ad); içerik veya size yoktur.": "`files` = references only (uuid + name); there is no content or size.",
    "`attachments.file_type` bazı kayıtlarda empty — `(unknown)` olarak etiketlendi.": "`attachments.file_type` is empty in some rows, labeled as `(unknown)`.",
    "`file_size` ile `content_length` paste metni için neredeyse aynı değerleri verir; ayrı tablo olarak sunulmaz.": "`file_size` and `content_length` are almost the same for pasted text; they are not shown as separate tables.",
    "**Conversation ↔ project bağlantısı export'ta yer almıyor**: `conversations.json` içinde `project_uuid` ya da benzeri alan yok. Bu yüzden \"project'li vs project'siz conversations\" türü kıyaslar yapılamaz.": "**The export does not include a conversation ↔ project link**: there is no `project_uuid` or similar field in `conversations.json`. That is why comparisons such as \"conversations with a project vs without a project\" cannot be made.",
    "`docs` = project knowledge dokümanları (`projects.json` altında, içerik DB'de saklı).": "`docs` = project knowledge documents (under `projects.json`, with content stored in the DB).",
    "`prompt_template` = projenin custom instruction'ı (sistem prompt'u).": "`prompt_template` = the project's custom instruction (system prompt).",
    "Uzunluklar karakter cinsindendir (`LENGTH` SQL fonksiyonu); tokenize edilmez, byte da değildir.": "Lengths are measured in characters (`LENGTH` SQL function); they are not tokenized and they are not byte counts.",
    "`is_starter_project` Anthropic'in hazır şablon projelerini işaretler.": "`is_starter_project` marks Anthropic's starter-template projects.",
    "Kaynak: `tool_calls.name='artifacts'` kayıtlarının `input` JSON'u.": "Source: the `input` JSON of rows where `tool_calls.name='artifacts'`.",
    "`update` komutunda input'ta genelde `type` ve `content` bulunmaz (yalnız `id` + patch); bu yüzden type dağılımında `(tipsiz)` satırı görülür ve content length yalnız create/rewrite için raporlanır.": "For the `update` command the input usually does not include `type` or `content` (only `id` + patch); that is why a `(typeless)` row appears in the type breakdown and content length is reported only for create/rewrite.",
    "İterasyon rate = (update + rewrite) / create; 1.0 üstü, aynı artifact'in ortalama birden çok kez düzenlendiğini gösterir.": "Iteration rate = (update + rewrite) / create; above 1.0 means the same artifact is edited more than once on average.",
    "Artifact'li conversations = içinde en az bir `artifacts` tool call olan conversations.": "Artifact-bearing conversations = conversations with at least one `artifacts` tool call.",
    "**Önemli:** Bu analiz yalnızca `artifacts` tool'unu kapsar. Claude.ai'de bash_tool, create_file, str_replace gibi araçlarla da dosya üretilir; bunlar arayüzde artifact gibi görünse de export'ta farklı tool adıyla kayıt düşer ve bu modülün kapsamı dışındadır. Claude.ai'nin araç yönlendirme davranışı zaman içinde değişebilir — eski konuşmalarda aynı içerik `artifacts` ile üretilmişken yeni konuşmalarda `create_file` veya `bash_tool` tercih edilmiş olabilir.": "**Important:** This analysis only covers the `artifacts` tool. Claude.ai also produces files via bash_tool, create_file, str_replace, etc.; these may look like artifacts in the UI but are logged under different tool names in the export and fall outside this module's scope. Claude.ai's tool-routing behavior may change over time — content that was once produced via `artifacts` may later be routed through `create_file` or `bash_tool` instead.",
    "Human vs assistant — code block kullanımı": "Human vs assistant — code-block usage",
    "Konuşma başına code block sayısı": "Code blocks per conversation",
    "Konuşma code share buckets (assistant tokens_code / tokens_text)": "Conversation code-share buckets (assistant `tokens_code / tokens_text`)",
    "Kaynak: `_stats_code_block` (her fenced block bir satır) + `_stats_message.inline_code_count` (inline backtick sayımı).": "Source: `_stats_code_block` (one row per fenced block) + `_stats_message.inline_code_count` (inline backtick count).",
    "Fenced blok = satır başında ` ``` `[dil] ile başlayan üçlü backtick bloğu; kapanışı olmayan bloklar metin sonuna kadar alınır.": "Fenced block = a triple-backtick block starting with ` ``` `[language] at the line start; unterminated blocks are taken until the end of the text.",
    "Dil etiketi ` ``` ` sonrası açılış satırından alınır; empty kalırsa `(empty)` kovasına düşer. Yaygın alias'lar (14 tane) normalize edilir: ts→typescript, py→python, sh/shell/zsh→bash, vs.": "The language label is read from the opening line after ` ``` `; if it stays empty, it falls into the `(empty)` bucket. Common aliases (14 of them) are normalized: ts→typescript, py→python, sh/shell/zsh→bash, etc.",
    "Inline backtick = tek-tırnaklı `foo`. Üçlü-backtick açılışıyla karışmasın diye `(?<!`)`(?!`)` lookaround kullanılır.": "Inline backtick = a single-backtick span like `foo`. `(?<!`)`(?!`)` lookarounds are used so it does not collide with triple-backtick openings.",
    "\"Konuşma code share\" sadece `sender='assistant'` mesajlarındaki `tokens_code / tokens_text` oranından hesaplanır — insan mesajları hariçtir.": "\"Conversation code share\" is computed only from `tokens_code / tokens_text` in `sender='assistant'` messages — human messages are excluded.",
    "Token sayımı `tiktoken.cl100k_base` (OpenAI); Claude tokenizer'ı değildir.": "Token counting uses `tiktoken.cl100k_base` (OpenAI); it is not Claude's tokenizer.",
    "Aylık citation": "Monthly citations",
    "Aylık citation'lı conversations": "Monthly conversations with citations",
    "Aylık with citations conversations": "Monthly conversations with citations",
    "citation'lı conversations": "conversations with citations",
    "Konuşma başına citation buckets": "Citation buckets per conversation",
    "Top 20 domain (citation sayısına göre)": "Top 20 domains (by citation count)",
    "Kaynak: `content_blocks.raw.citations[]` — yalnız `type='text'` blokları. Citation item yapısı: `{uuid, start_index, end_index, details: {type, url}}`.": "Source: `content_blocks.raw.citations[]` — only `type='text'` blocks. Citation item shape: `{uuid, start_index, end_index, details: {type, url}}`.",
    "Domain çıkarımı: `regexp_extract(url, 'https?://([^/]+)', 1)`.": "Domain extraction: `regexp_extract(url, 'https?://([^/]+)', 1)`.",
    "Örnekteki tüm citation'lar `web_search_citation` tipindedir (web_fetch de aynı tip'i üretiyor); ham `tool_calls.name` ayrımı için m05'e bakılmalı.": "All citations in the sample are of type `web_search_citation` (web_fetch produces the same type as well); for raw `tool_calls.name` distinctions, see m05.",
    "Citation sayısı ≠ referanslanan kaynak sayısı: aynı URL bir konuşmada birden çok cümlede atıf olarak geçebiliyor.": "Citation count is not equal to the number of referenced sources: the same URL can be cited in multiple sentences within one conversation.",
    "Kaynak: `tool_results.is_error` ve `tool_results.content` (JSON). `UNNEST(CAST(content AS JSON[]))` ile text blokları açılır.": "Source: `tool_results.is_error` and `tool_results.content` (JSON). Text blocks are expanded with `UNNEST(CAST(content AS JSON[]))`.",
    "Kategorizasyon regex tabanlı (yapısal anahtar kelimeler, dilden bağımsız İngilizce errors literatürüne göre). Eşleşme sırası yukarıdan aşağıya; ilk eşleşen kategori atanır. Eşleşmeyen → OTHER.": "Categorization is regex-based (structural keywords, language-agnostic English error terminology). Matching runs top to bottom; the first matching category is assigned. Unmatched rows go to OTHER.",
    "Normalize: UUID'ler `<UUID>`, yollar `<PATH>`, URL'ler `<URL>` ile değiştirilir; metin 200 karaktere kesilir.": "Normalization replaces UUIDs with `<UUID>`, paths with `<PATH>`, URLs with `<URL>`; text is truncated to 200 characters.",
    "`tool_results.stop_timestamp` tüm kayıtlarda NULL — tool latency ölçülemez, raporda hesaplanmadı.": "`tool_results.stop_timestamp` is NULL in all rows — tool latency cannot be measured and is omitted from the report.",
    "`retry_same_tool`/`diff_tool`/`no_tool` kararı: hatalı mesajdan sonraki ilk human mesajını izleyen ilk assistant mesajının `tool_calls.name` listesi kullanılır.": "The `retry_same_tool` / `diff_tool` / `no_tool` decision uses the `tool_calls.name` list from the first assistant message after the first human message following the error.",
    "Asgari calls eşiği: tool başına share tablosunda n≥20.": "Minimum calls threshold: n>=20 in the per-tool share table.",
    "sonraki asistan mesajı aynı tool'u tekrar çağırdı": "the next assistant message called the same tool again",
    "sonraki asistan mesajı farklı tool kullandı": "the next assistant message used a different tool",
    "sonraki asistan mesajı tool kullanmadan yanıtladı": "the next assistant message replied without using a tool",
    "kullanıcı prompt attı, asistan cevabı yok": "the user sent another prompt; there is no assistant reply",
    "Hata sonrası sonraki human→asistan davranışı": "Next human→assistant behavior after an error",
    "errors mesajı": "error message",
    "edit noktası/conversations": "edit points/conversation",
    "retry noktası/conversations": "retry points/conversation",
    "Konuşma başına edit noktası": "Edit points per conversation",
    "Konuşma başına retry noktası": "Retry points per conversation",
    "**Edit sonrası hangi child devam etmiş?** (2/2 child devam ediyor)": "**Which child continued after the edit?** (2/2 children continue)",
    "**İlk vs son prompt length** — her edit noktasında ilk yazılan prompt ile son yazılan prompt kıyaslanır.": "**First vs last prompt length** — at each edit point, the first written prompt is compared with the last prompt.",
    "**Retry sonrası hangi child devam etmiş?** (1/2 child devam ediyor)": "**Which child continued after retry?** (1/2 children continue)",
    "**İlk vs son retry cevap length** — her retry noktasında ilk üretilen cevap ile son üretilen cevap kıyaslanır.": "**First vs last retry response length** — at each retry point, the first generated response is compared with the last response.",
    "Retry sonrası cevap length değişimi": "Response-length change after retry",
    "**Kriter**: aynı `message_uuid` içinde `updated_at > created_at + 60s` ve `chars_text > 0`. Örneklem yalnız **assistant** mesajlarında bu davranışı gösteriyor; human mesajlarında (60s eşiğini geçen) in-place düzenleme ölçülmedi.": "**Criterion**: within the same `message_uuid`, `updated_at > created_at + 60s` and `chars_text > 0`. In this sample the behavior appears only in **assistant** messages; in-place edits were not measured on human messages (crossing the 60s threshold).",
    "In-place edit delta süresi distribution": "In-place edit delta-time distribution",
    "**In-place edit'li vs edit'siz asistan mesajı karakter distribution**": "**Assistant-message character distribution: with vs without in-place edit**",
    "Aynı `conversation_uuid` altında `parent_message_uuid IS NULL` veya root sentinel (`00000000-…`) olan messages sayısı >1 olan konuşmalar. Muhtemel neden: UI'da \"continue from previous state\" benzeri bir etkileşim.": "Conversations where the count of messages under the same `conversation_uuid` with `parent_message_uuid IS NULL` or the root sentinel (`00000000-…`) is >1. A likely cause is a UI interaction similar to \"continue from previous state\".",
    "**Dağılım** — total 1 conversations": "**Distribution** — total 1 conversations",
    "Her konuşmaya dört bayrak atanır: `human-fork` (prompt yeniden yazma), `asst-fork` (cevap yeniden üretme), `in-place` (asistan in-place düzenleme), `multi-root`. Tablo bayrak kombinasyonlarını gösterir.": "Each conversation gets four flags: `human-fork` (prompt rewriting), `asst-fork` (response regeneration), `in-place` (assistant in-place edit), `multi-root`. The table shows flag combinations.",
    "Müdahale bayrak kombinasyonları (top 12)": "Intervention-flag combinations (top 12)",
    "(hiçbiri)": "(none)",
    "Her mekaniği adıyla listeleyen top 3. \"Konuşma\" sütunu `name` (CSV'lerde `conversation_uuid` yer alır).": "Top 3 rows listing each mechanic by name. The \"Conversation\" column uses `name` (`conversation_uuid` appears in CSV files).",
    "Not: **edit** = Σ(child-1) her fork noktasında — aynı noktada 3 child varsa 2 edit sayılır. **edit noktası** = ≥2 child'a sahip benzersiz parent. **tek noktada max yazım** = bir prompt'un kaç kez yeniden yazıldığı.": "Note: **edit** = Σ(child-1) at each fork point — if the same point has 3 children, that counts as 2 edits. **edit point** = a unique parent with >=2 children. **max rewrites at one point** = how many times one prompt was rewritten.",
    "Kaynak: `_stats_message` kolonları `inplace_edit_flag`, `fork_parent_flag`, `fork_child_flag`, `fork_child_continued`, `edit_delta_seconds`.": "Source: `_stats_message` columns `inplace_edit_flag`, `fork_parent_flag`, `fork_child_flag`, `fork_child_continued`, `edit_delta_seconds`.",
    "Claude.ai'de üç müdahale tipinin export izi farklı: prompt/cevap yeniden üretimi **yeni message_uuid** (fork), in-place düzenleme **aynı uuid + updated_at**.": "In Claude.ai the export traces of the three intervention types differ: prompt/response regeneration produces a **new message_uuid** (fork), while in-place edit keeps the **same uuid + updated_at**.",
    "*Ayrım:* **edit noktası** = düzenlenen benzersiz prompt konumu; **edit** = total düzenleme eylemi (aynı noktada 3 kez yeniden yazıldıysa 3 edit sayılır). Retry için de aynı: **retry noktası** vs total **retry** eylemi.": "*Distinction:* **edit point** = the unique prompt position that was edited; **edit** = the total number of edit actions (if the same point was rewritten 3 times, that counts as 3 edits). The same applies to retry: **retry point** vs total **retry** actions.",
    "Claude.ai'de bir konuşmaya üç farklı müdahale biçimi uygulanabilir. Export'ta bunlar birbirinden farklı izler bırakır — tek bir \"edit\" kolonu yoktur:": "Three different intervention mechanics can be applied to a Claude.ai conversation. In the export they leave different traces — there is no single \"edit\" column:",
    "**Prompt rewriting** (*human fork*): kullanıcı önceki asistan cevabının altındaki prompt'unu düzenler. Export'a **yeni `message_uuid`** yazılır, aynı asistan cevabı (`parent_message_uuid`) altına ikinci bir human child eklenir. Orijinal prompt silinmez; `updated_at` değişmez. Fork'un **parent**'ı bir asistan mesajıdır, **children** human.": "**Prompt rewriting** (*human fork*): the user edits their prompt under the previous assistant reply. The export writes a **new `message_uuid`**, adding a second human child under the same assistant reply (`parent_message_uuid`). The original prompt is not deleted; `updated_at` does not change. The fork **parent** is an assistant message, the **children** are human.",
    "**Response regeneration** (*assistant fork* / retry): kullanıcı bir prompt'a verilen asistan cevabını yeniden üretmek için retry basar. Yine **yeni `message_uuid`**, aynı human prompt (`parent_message_uuid`) altına ikinci bir assistant child eklenir. Fork'un **parent**'ı bir human mesajıdır, **children** assistant.": "**Response regeneration** (*assistant fork* / retry): the user hits retry to regenerate the assistant reply to a prompt. Again a **new `message_uuid`** is written, adding a second assistant child under the same human prompt (`parent_message_uuid`). The fork **parent** is a human message, the **children** are assistant.",
    "**Response in-place edit** (*assistant in-place edit*): aynı `message_uuid` içinde `updated_at` güncellenir ve `content` değişir. Streaming snapshot'ları eleme için `updated_at - created_at > 60s` eşiği uygulanır. Örneklem **yalnız assistant** mesajında bu davranışı gösteriyor; human mesajında (streaming dışında) ölçülmedi.": "**Response in-place edit** (*assistant in-place edit*): `updated_at` is updated and `content` changes within the same `message_uuid`. A threshold of `updated_at - created_at > 60s` is applied to filter streaming snapshots. In this sample the behavior appears **only in assistant** messages; it was not measured on human messages (outside streaming).",
    "Fork = aynı non-root parent'ın birden çok child'ı olması. Root parent (`NULL` veya `00000000-…`) dahil edilmez.": "Fork = a non-root parent having multiple children. The root parent (`NULL` or `00000000-…`) is excluded.",
    "**Fork sender semantiği**: fork parent'ın sender'ı, fork'u *tetikleyen eylemi yapan*'ın tersidir. Parent=assistant → children=human → **prompt yeniden yazma**. Parent=human → children=assistant → **retry**. Örneklemde mixed-sender fork yok — bir parent'ın child'ları her zaman tek sender tipinde.": "**Fork sender semantics**: the fork parent's sender is the inverse of the actor who *triggered* the fork. Parent=assistant -> children=human -> **prompt rewriting**. Parent=human -> children=assistant -> **retry**. In this sample there is no mixed-sender fork — a parent's children always have a single sender type.",
    "`fork_child_continued` = bu child'ın descendant'ı var mı. Kullanıcının hangi dalı sürdürdüğünün göstergesidir; ama kullanıcı bir dalda gezinip hiç messages yazmadıysa oradaki seçim izi tutulmaz.": "`fork_child_continued` = whether this child has descendants. It indicates which branch the user continued, but if the user explored a branch and never wrote any messages there, that choice leaves no trace.",
    "In-place edit eşiği 60s: streaming snapshot'larını eliminate etmek içindir; streaming sırasında `updated_at - created_at` birkaç saniyedir.": "The 60s in-place edit threshold exists to eliminate streaming snapshots; during streaming, `updated_at - created_at` is only a few seconds.",
    "Çok-root konuşmalarda `root_count` alanı, `parent_message_uuid IN (NULL, '00000000-0000-4000-8000-000000000000')` olan messages sayısıdır.": "In multi-root conversations, the `root_count` field is the number of messages where `parent_message_uuid IN (NULL, '00000000-0000-4000-8000-000000000000')`.",
}

_TRANSLATION_RULES: list[tuple[re.Pattern[str], str | Any]] = [
    (re.compile(r"Tüm saat/gün bilgisi yerel saate \(\*\*(.+?)\*\*\) çevrilerek hesaplanmıştır\.", re.U),
     r"All hour/day values are computed in local time (**\1**)."),
    (re.compile(r"^(.+?) konuşma \(boş: (.+?)\); medyan (.+?) mesaj, (.+?) human turn, p95 (.+?) mesaj, max (.+?)\. Medyan ~(.+?) token\.$", re.U),
     r"\1 conversations (empty: \2); median \3 messages, \4 human turns, p95 \5 messages, max \6. Median ~\7 tokens."),
    (re.compile(r"Toplam: \*\*(.+?)\*\* konuşma \(boş: (.+?)\)", re.U), r"Total: **\1** conversations (empty: \2)"),
    (re.compile(r"^human medyan ~(.+?) token, assistant medyan ~(.+?) token; toplam token hacminin %(.+?)'i assistant; assistant cevabının %(.+?)'i kod bloğu\.$", re.U),
     r"Human median ~\1 tokens, assistant median ~\2 tokens; \3% of the total token volume belongs to the assistant; \4% of assistant replies is code blocks."),
    (re.compile(r"^Human: \*\*(.+?)\*\* mesaj \(tool-only: (.+?)\)$", re.U), r"Human: **\1** messages (tool-only: \2)"),
    (re.compile(r"^Assistant: \*\*(.+?)\*\* mesaj \(tool-only: (.+?)\)$", re.U), r"Assistant: **\1** messages (tool-only: \2)"),
    (re.compile(r"^Medyan: human \*\*~(.+?) token\*\*, assistant \*\*~(.+?) token\*\*$", re.U), r"Median: human **~\1 tokens**, assistant **~\2 tokens**"),
    (re.compile(r"^p95: human \*\*~(.+?) token\*\*, assistant \*\*~(.+?) token\*\*$", re.U), r"p95: human **~\1 tokens**, assistant **~\2 tokens**"),
    (re.compile(r"^Max: human \*\*~(.+?) token\*\*, assistant \*\*~(.+?) token\*\*$", re.U), r"Max: human **~\1 tokens**, assistant **~\2 tokens**"),
    (re.compile(r"^Token hacmi: human \*\*~(.+?)\*\* \(%(.+?)\), assistant \*\*~(.+?)\*\* \(%(.+?)\) — oran \*\*(.+?)×\*\*$", re.U),
     r"Token volume: human **~\1** (\2%), assistant **~\3** (\4%) — ratio **\5x**"),
    (re.compile(r"^Assistant cevabının \*\*%(.+?)\*\*'i kod bloğu; mesajların \*\*%(.+?)\*\*'i \((.+?)\) en az bir kod bloğu içeriyor$", re.U),
     r"Code blocks make up **\1%** of assistant replies; **\2%** of messages (\3) contain at least one code block"),
    (re.compile(r"^konuşmaların %(.+?)'inde en az bir thinking bloğu; thinking'li cevap medyan ~(.+?) token, thinking'siz ~(.+?); thinking bloğu medyan ~(.+?) token\.$", re.U),
     r"At least one thinking block appears in \1% of conversations; median reply with thinking is ~\2 tokens, without thinking ~\3; median thinking block is ~\4 tokens."),
    (re.compile(r"Toplam konuşma \(non-empty\): \*\*(.+?)\*\*", re.U), r"Total conversations (non-empty): **\1**"),
    (re.compile(r"^Toplam konuşma \(non-empty\): \*\*(.+?)\*\*; thinking içeren: \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)$", re.U),
     r"Total conversations (non-empty): **\1**; with thinking: **\2** (**\3**)"),
    (re.compile(r"^Assistant mesajı: \*\*(.+?)\*\*; thinking bloğu olan: \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)$", re.U),
     r"Assistant messages: **\1**; with thinking block: **\2** (**\3**)"),
    (re.compile(r"^Cevap medyanı: thinking yok \*\*~(.+?) token\*\*, thinking var \*\*~(.+?) token\*\*$", re.U),
     r"Response median: without thinking **~\1 tokens**, with thinking **~\2 tokens**"),
    (re.compile(r"^Cevap p95: thinking yok \*\*~(.+?) token\*\*, thinking var \*\*~(.+?) token\*\*$", re.U),
     r"Response p95: without thinking **~\1 tokens**, with thinking **~\2 tokens**"),
    (re.compile(r"^Thinking bloğu: medyan \*\*~(.+?) token\*\*, p95 \*\*~(.+?) token\*\*, max \*\*~(.+?) token\*\*$", re.U),
     r"Thinking block: median **~\1 tokens**, p95 **~\2 tokens**, max **~\3 tokens**"),
    (re.compile(r"^Toplam assistant token'ının \*\*%(.+?)\*\*'i thinking \(görünen text dışı\)$", re.U),
     r"\1% of assistant tokens is thinking (outside visible text)"),
    (re.compile(r"^Thinking'li konuşma başına toplam thinking: medyan \*\*~(.+?) token\*\*, max \*\*~(.+?) token\*\*$", re.U),
     r"Total thinking per conversation with thinking: median **~\1 tokens**, max **~\2 tokens**"),
    (re.compile(r"^Thinking-li konuşmaların \*\*%(.+?)\*\*'i tool da kullanıyor; thinking-siz konuşmalarda bu oran \*\*%(.+?)\*\*$", re.U),
     r"\1% of conversations with thinking also use tools; in conversations without thinking this rate is **\2%**"),
    (re.compile(r"^İlk assistant mesajında thinking oranı \*\*%(.+?)\*\*; sonraki mesajlarda \*\*%(.+?)\*\*$", re.U),
     r"Thinking rate is **\1%** in the first assistant message and **\2%** in later messages"),
    (re.compile(r"^en yoğun saat (.+?) \((.+?) mesaj\), en yoğun gün (.+?) \((.+?) mesaj\); hafta sonu payı %(.+?)\.$", re.U), None),
    (re.compile(r"En az 1 tool çağrısı olan: \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)", re.U), r"With at least 1 tool call: **\1** (**\2**)"),
    (re.compile(r"Toplam tool çağrısı: \*\*(.+?)\*\*", re.U), r"Total tool calls: **\1**"),
    (re.compile(r"Tool sonuç hata oranı: \*\*(.+?)\*\* / \*\*(.+?)\*\* → \*\*(.+?)\*\*", re.U), r"Tool-result error rate: **\1** / **\2** -> **\3**"),
    (re.compile(r"Tool çağrısı süresi: medyan \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*", re.U), r"Tool-call duration: median **\1**, p95 **\2**, max **\3**"),
    (re.compile(r"^(.+?)/(.+?) konuşmada tool kullanılmış \(%(.+?)\); tool kullanan konuşmalarda medyan (.+?), p95 (.+?), max (.+?) tool; toplam (.+?) çağrı, %(.+?) hata, medyan süre (.+?)\.$", re.U),
     r"\1/\2 conversations used tools (\3%); among tool-using conversations the median is \4, p95 \5, max \6 tools; \7 total calls, \8% errors, median duration \9."),
    (re.compile(r"^(.+?) konuşma \(≥2 mesaj\); medyan ömür \*\*(.+?)\*\*, mesaj-arası medyan \*\*(.+?)\*\*, ilk-yanıt medyan \*\*(.+?)\*\*\. Çok-oturumlu konuşma: %(.+?)\.$", re.U),
     r"\1 conversations (>=2 messages); median lifetime **\2**, inter-message median **\3**, first-response median **\4**. Multi-session conversations: \5%."),
    (re.compile(r"Hafta sonu payı: \*\*(.+?)\*\*", re.U), r"Weekend share: **\1**"),
    (re.compile(r"En yoğun saat: \*\*(.+?)\*\* \((.+?) mesaj\)", re.U), r"Peak hour: **\1** (\2 messages)"),
    (re.compile(r"En yoğun gün: \*\*(.+?)\*\* \((.+?) mesaj\)", re.U), None),
    (re.compile(r"Medyan: \*\*(.+?)\*\* mesaj, \*\*(.+?)\*\* human turn, \*\*~(.+?) token\*\*", re.U), r"Median: **\1** messages, **\2** human turns, **~\3 tokens**"),
    (re.compile(r"p95: \*\*(.+?)\*\* mesaj, \*\*(.+?)\*\* human turn, \*\*~(.+?) token\*\*", re.U), r"p95: **\1** messages, **\2** human turns, **~\3 tokens**"),
    (re.compile(r"Max: \*\*(.+?)\*\* mesaj, \*\*(.+?)\*\* human turn, \*\*~(.+?) token\*\*", re.U), r"Max: **\1** messages, **\2** human turns, **~\3 tokens**"),
    (re.compile(r"^(.+?) attachment \+ (.+?) file; medyan boy (.+?), p95 (.+?); ilk human mesajında attachment/file oranı %(.+?)\.$", re.U),
     r"\1 attachments + \2 files; median size \3, p95 \4; attachment/file rate in the first human message is \5%."),
    (re.compile(r"^Toplam attachment: (.+)$", re.U), r"Total attachments: \1"),
    (re.compile(r"^Toplam file \(referans\): (.+)$", re.U), r"Total files (references): \1"),
    (re.compile(r"^Attachment içeren konuşma: (.+)$", re.U), r"Conversations with attachment: \1"),
    (re.compile(r"^File referansı içeren konuşma: (.+)$", re.U), r"Conversations with file reference: \1"),
    (re.compile(r"^İlk human mesajında attachment/file: (.+)$", re.U), r"Attachment/file in the first human message: \1"),
    (re.compile(r"^En sık file_type: (.+)$", re.U), r"Most common file_type: \1"),
    (re.compile(r"^Toplam messages: \*\*(.+?)\*\*$", re.U), r"Total messages: **\1**"),
    (re.compile(r"^Toplam code tokens: \*\*(.+?)\*\* / total text token: \*\*(.+?)\*\* → \*\*(.+?)\*\*$", re.U), r"Total code tokens: **\1** / total text tokens: **\2** -> **\3**"),
    (re.compile(r"^Code block içeren assistant mesajı: \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)$", re.U), r"Assistant messages with a code block: **\1** (**\2**)"),
    (re.compile(r"^Toplam thinking token: \*\*(.+?)\*\* / total text \+ thinking token: \*\*(.+?)\*\* → \*\*(.+?)\*\*$", re.U), r"Total thinking tokens: **\1** / total text + thinking tokens: **\2** -> **\3**"),
    (re.compile(r"^Tool-only messages = `chars_text=0` — yalnız tool_use / tool_result bloğu taşıyan mesajlar; dağılım hesabından dışlanır\.$", re.U),
     r"Tool-only messages = `chars_text=0` — messages that contain only tool_use / tool_result blocks; they are excluded from the distribution calculation."),
    (re.compile(r"^Code block = üçlü-backtick ile çevrili fenced block; kapanışı olmayan bloklar metin sonuna kadar sayılır\. Inline backtick \(`` ` ``\) dahil değildir\.$", re.U),
     r"Code block = a fenced block delimited by triple backticks; unterminated blocks are counted until the end of the text. Inline backticks (`` ` ``) are excluded."),
    (re.compile(r"^Toplam citation: \*\*(.+?)\*\*$", re.U), r"Total citations: **\1**"),
    (re.compile(r"^Toplam artifact çağrısı: \*\*(.+?)\*\* \(parse edilebilen input'lu\)$", re.U), r"Total artifact calls: **\1** (with parseable input)"),
    (re.compile(r"^Artifact üretilen conversations: \*\*(.+?)\*\*$", re.U), r"Conversations with artifacts: **\1**"),
    (re.compile(r"^Toplam project: \*\*(.+?)\*\* \(starter: \*\*(.+?)\*\* / %(.+?), private: \*\*(.+?)\*\* / %(.+?)\)$", re.U), r"Total projects: **\1** (starter: **\2** / \3%, private: **\4** / \5%)"),
    (re.compile(r"^Toplam knowledge doc: \*\*(.+?)\*\*; median size \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*$", re.U), r"Total knowledge docs: **\1**; median size **\2**, p95 **\3**, max **\4**"),
    (re.compile(r"^(.+?) project, (.+?) doc; project başına doc medyan \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*; doc uzunluk medyan \*\*(.+?)\*\*, max \*\*(.+?)\*\*; prompt_template dolu: \*\*(.+?)\*\* / \*\*(.+?)\*\*\.$", re.U),
     r"\1 projects, \2 docs; median docs per project **\3**, p95 **\4**; median doc length **\5**, max **\6**; prompt_template populated: **\7** / **\8**."),
    (re.compile(r"^(.+?) artifact çağrısı, (.+?) konuşmada; en sık tip: \*\*(.+?)\*\* \(\*\*(.+?)\*\*\); iterasyon oranı \(update\+rewrite\)/create = \*\*(.+?)\*\*\.$", re.U),
     r"\1 artifact calls across \2 conversations; most common type: **\3** (**\4**); iteration ratio (update+rewrite)/create = **\5**."),
    (re.compile(r"^(.+?) fenced blok / (.+?) mesaj / (.+?) konuşma; blok medyan (.+?), p95 (.+?); en sık dil: (.+?)\.$", re.U),
     r"\1 fenced blocks / \2 messages / \3 conversations; median block size \4, p95 \5; most common language: \6."),
    (re.compile(r"^(.+?) citation / (.+?) mesaj / (.+?) konuşma; (.+?) domain; en sık: (.+?) \((.+?)\)\.$", re.U),
     r"\1 citations / \2 messages / \3 conversations; \4 domains; most common: \5 (\6)."),
    (re.compile(r"^(.+?) hata / (.+?) çağrı \(%(.+?)\); (.+?) konuşma; en yüksek oran: (.+?) \(%(.+?)\)$", re.U),
     r"\1 errors / \2 calls (\3%); \4 conversations; highest rate: \5 (\6%)"),
    (re.compile(r"^(.+?) prompt yeniden yazma / (.+?) retry / (.+?) in-place edit / (.+?) çok-root$", re.U),
     r"\1 prompt rewrites / \2 retries / \3 in-place edits / \4 multi-root"),
    (re.compile(r"^Toplam tool calls: \*\*(.+?)\*\*, hatalı: \*\*(.+?)\*\* \(genel share \*\*(.+?)\*\*\)$", re.U), r"Total tool calls: **\1**, errors: **\2** (overall share **\3**)"),
    (re.compile(r"^Toplam tool_result: \*\*(.+?)\*\*$", re.U), r"Total tool results: **\1**"),
    (re.compile(r"^Hata \(`is_error=true`\): \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)$", re.U), r"Errors (`is_error=true`): **\1** (**\2**)"),
    (re.compile(r"^Total tool calls: \*\*(.+?)\*\*, hatalı: \*\*(.+?)\*\* \(genel share \*\*(.+?)\*\*\)$", re.U), r"Total tool calls: **\1**, errors: **\2** (overall share **\3**)"),
    (re.compile(r"^Hatalı messages: \*\*(.+?)\*\*, hatalı conversations: \*\*(.+?)\*\*$", re.U), r"Error messages: **\1**, error conversations: **\2**"),
    (re.compile(r"^En yüksek errors rate \(n≥20\): \*\*(.+?)\*\* — (.+?)/(.+?) \((.+?)\)$", re.U), r"Highest error rate (n>=20): **\1** — \2/\3 (\4)"),
    (re.compile(r"^En çok errors üreten: \*\*(.+?)\*\* — (.+?) errors \((.+?) calls, (.+?)\)$", re.U), r"Most error-producing tool: **\1** — \2 errors (\3 calls, \4)"),
    (re.compile(r"^En yaygın kategori: \*\*(.+?)\*\* \((.+?) errors\)$", re.U), r"Most common category: **\1** (\2 errors)"),
    (re.compile(r"^(.+?)/(.+?) konuşmada tool kullanılmış \(%(.+?)\); tool kullanan konuşmalarda median (.+?)\s+p95 (.+?)\s+max (.+?) tool; total (.+?) calls\s+%(.+?) errors\s+median süre (.+?)\.$", re.U),
     r"\1/\2 conversations used tools (\3%); among tool-using conversations the median is \4, p95 \5, max \6 tools; \7 total calls, \8% errors, median duration \9."),
    (re.compile(r"^(.+?)/(.+?) konuşmada tool kullanılmış \(%(.+?)\); in tool-using conversations median (.+?)\s+p95 (.+?)\s+max (.+?) tool; total (.+?) calls\s+%(.+?) errors\s+median süre (.+?)\.$", re.U),
     r"\1/\2 conversations used tools (\3%); among tool-using conversations the median is \4, p95 \5, max \6 tools; \7 total calls, \8% errors, median duration \9."),
    (re.compile(r"^Tool kullanan konuşmalarda medyan \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\* tool$", re.U), r"Among tool-using conversations: median **\1**, p95 **\2**, max **\3** tools"),
    (re.compile(r"^Tool kullanan konuşmalar ortalama \*\*(.+?)\*\* messages / tool kullanmayanlar \*\*(.+?)\*\* \(median\)$", re.U), r"Conversation length comparison: tool-using **\1** messages / non-tool-using **\2** (median)"),
    (re.compile(r"^Tool kullanan konuşmalar ortalama \*\*(.+?)\*\* messages / non-tool users \*\*(.+?)\*\* \(median\)$", re.U), r"Conversation length comparison: tool-using **\1** messages / non-tool users **\2** (median)"),
    (re.compile(r"^Konuşma başına farklı tool \(tool kullananlarda\): median \*\*(.+?)\*\*, max \*\*(.+?)\*\*$", re.U), r"Distinct tools per conversation (tool users): median **\1**, max **\2**"),
    (re.compile(r"^Konuşma başına farklı tool \(among tool users\): median \*\*(.+?)\*\*, max \*\*(.+?)\*\*$", re.U), r"Distinct tools per conversation (tool users): median **\1**, max **\2**"),
    (re.compile(r"^Kapsam: \*\*(.+?)\*\* conversations \(≥2 mesajlı\)$", re.U), r"Coverage: **\1** conversations (>=2 messages)"),
    (re.compile(r"^0 saniyelik lifetime \(aynı timestamp\): \*\*(.+?)\*\*$", re.U), r"Zero-second lifetime (same timestamp): **\1**"),
    (re.compile(r"^Inter-message gerçek gap \(tüm ardışık çiftler, n=\*\*(.+?)\*\*\): median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*$", re.U),
     r"Real inter-message gap (all consecutive pairs, n=**\1**): median **\2**, p95 **\3**"),
    (re.compile(r"^First-response latency — human→assistant geçişleri, n=\*\*(.+?)\*\*: median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*$", re.U),
     r"First-response latency — human→assistant transitions, n=**\1**: median **\2**, p95 **\3**"),
    (re.compile(r"^Çok-oturumlu conversations \(içinde >1h gap olan\): \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)$", re.U),
     r"Multi-session conversations (with a >1h gap inside): **\1** (**\2**)"),
    (re.compile(r"^Komut distribution: create \*\*(.+?)\*\* · update \*\*(.+?)\*\* · rewrite \*\*(.+?)\*\* · diğer \*\*(.+?)\*\*$", re.U),
     r"Command distribution: create **\1** · update **\2** · rewrite **\3** · other **\4**"),
    (re.compile(r"^İterasyon rate \(update\+rewrite\) / create = \*\*(.+?)\*\*$", re.U),
     r"Iteration rate (update+rewrite) / create = **\1**"),
    (re.compile(r"^Tipsiz kayıt: \*\*(.+?)\*\* \(input'ta `type` alanı yok — büyük ihtimalle `update` komutu\)$", re.U),
     r"Records without type: **\1** (`type` is missing in the input — most likely an `update` command)"),
    (re.compile(r"^content length \(create/rewrite için, n=\*\*(.+?)\*\*\): median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*$", re.U),
     r"Content length (create/rewrite, n=**\1**): median **\2**, p95 **\3**, max **\4**"),
    (re.compile(r"^Toplam fenced code block: \*\*(.+?)\*\* \(\*\*(.+?)\*\* mesajda, \*\*(.+?)\*\* konuşmada\)$", re.U),
     r"Total fenced code blocks: **\1** (across **\2** messages, **\3** conversations)"),
    (re.compile(r"^Toplam code tokens: \*\*(.+?)\*\*, total kod karakter: \*\*(.+?)\*\*$", re.U),
     r"Total code tokens: **\1**, total code characters: **\2**"),
    (re.compile(r"^Assistant: \*\*(.+?)\*\* blok / \*\*(.+?)\*\* messages \(assistant mesajlarının \*\*(.+?)\*\*'inde en az bir blok\)$", re.U),
     r"Assistant: **\1** blocks / **\2** messages (**\3** of assistant messages include at least one block)"),
    (re.compile(r"^Human: \*\*(.+?)\*\* blok / \*\*(.+?)\*\* messages \(human mesajlarının \*\*(.+?)\*\*'inde en az bir blok\)$", re.U),
     r"Human: **\1** blocks / **\2** messages (**\3** of human messages include at least one block)"),
    (re.compile(r"^Konuşma başına blok: median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*$", re.U),
     r"Blocks per conversation: median **\1**, p95 **\2**, max **\3**"),
    (re.compile(r"^En sık dil: \*\*(.+?)\*\* \((.+?) blok\)$", re.U), r"Most common language: **\1** (\2 blocks)"),
    (re.compile(r"^Citation'lı messages: \*\*(.+?)\*\*, with citations conversations: \*\*(.+?)\*\*$", re.U),
     r"Messages with citations: **\1**, conversations with citations: **\2**"),
    (re.compile(r"^Konuşma başına citation: median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*$", re.U),
     r"Citations per conversation: median **\1**, p95 **\2**, max **\3**"),
    (re.compile(r"^En çok citation: \*\*(.+?)\*\* \((.+?) citation\)$", re.U),
     r"Most citations: **\1** (\2 citations)"),
    (re.compile(r"^Pearson korelasyon \(conversations başına citation ve web_search\+web_fetch çağrısı\): \*\*r = (.+?)\*\*\. Web tool call yapılmamış ama with citations conversations sayısı: \*\*(.+?)\*\* — scatter grafiğinde x=0 noktasındakiler\.$", re.U),
     r"Pearson correlation (citations per conversation vs web_search+web_fetch calls): **r = \1**. Conversations with citations but no web-tool call: **\2** — the points at x=0 in the scatter plot."),
    (re.compile(r"^Prompt rewriting \(human fork\): \*\*(.+?)\*\* edit / \*\*(.+?)\*\* edit noktası / \*\*(.+?)\*\* conversations$", re.U), r"Prompt rewriting (human fork): **\1** edits / **\2** edit points / **\3** conversations"),
    (re.compile(r"^Response regeneration \(assistant fork / retry\): \*\*(.+?)\*\* retry / \*\*(.+?)\*\* retry noktası / \*\*(.+?)\*\* conversations$", re.U), r"Response regeneration (assistant fork / retry): **\1** retries / **\2** retry points / **\3** conversations"),
    (re.compile(r"^Çok-root conversations \(root_count > 1\): \*\*(.+?)\*\* \(total (.+?) empty olmayan konuşmadan\)$", re.U), r"Multi-root conversations (root_count > 1): **\1** (out of \2 non-empty conversations)"),
    (re.compile(r"^(\d+) total edit eylemi, (\d+) edit noktasında \(benzersiz prompt konumu\), (\d+) konuşmada\.$", re.U),
     r"\1 total edit actions across \2 edit points (unique prompt positions), \3 conversations."),
    (re.compile(r"^Bir konuşmadaki max edit noktası: (.+?); tek bir prompt'un kaç kez yeniden yazıldığı ayrı ölçüm \(aşağıdaki child-sayısı distribution\)\.$", re.U),
     r"Max edit points in one conversation: \1; how many times a single prompt was rewritten is measured separately (child-count distribution below)."),
    (re.compile(r"^(\d+) total retry eylemi, (\d+) retry noktasında \(benzersiz asistan cevap konumu\), (\d+) konuşmada\.$", re.U),
     r"\1 total retry actions across \2 retry points (unique assistant-response positions), \3 conversations."),
    (re.compile(r"^Bir konuşmadaki max retry noktası: (.+?)\.$", re.U), r"Max retry points in one conversation: \1."),
    (re.compile(r"^(.+?) — kova dağılımı \(n=(.+?)\)$", re.U), None),
    (re.compile(r"^(.+?) \(n=(.+?)\)$", re.U), None),
    (re.compile(r"^En çok kullanılan 15 tool \(toplam (.+?) çağrı\)$", re.U), r"Top 15 tools (\1 total calls)"),
    (re.compile(r"Ardışık human/assistant mesaj çiftleri arası saniye\.", re.U), "Seconds between consecutive human/assistant message pairs."),
    (re.compile(r"Human mesajından hemen sonraki assistant mesajına kadar geçen süre\.", re.U), "Elapsed time until the assistant message immediately following a human message."),
]

_TERM_TRANSLATIONS = [
    ("Thinking × tool kullanımı çakışması", "Thinking × tool-use overlap"),
    ("Attachment/file içeren konuşma oranı", "Share of conversations with attachment/file"),
    ("Toplam assistant token'ının", "Of total assistant tokens"),
    ("Thinking'li konuşma başına toplam thinking", "Total thinking per conversation with thinking"),
    ("İlk assistant mesajında thinking oranı", "Thinking rate in the first assistant message"),
    ("Konuşma başına human turn sayısı", "Human turns per conversation"),
    ("Konuşma başına toplam token", "Total tokens per conversation"),
    ("Konuşma başına mesaj sayısı", "Messages per conversation"),
    ("Konuşma başına thinking token yükü", "Thinking-token load per conversation"),
    ("Konuşma başına attachment/file sayısı", "Attachment/file count per conversation"),
    ("Konuşma başına attachment sayısı", "Attachment count per conversation"),
    ("Konuşma başına artifact çağrısı", "Artifact calls per conversation"),
    ("Konuşma başına iterasyon", "Iterations per conversation"),
    ("Konuşma başına tool çeşitliliği", "Tool diversity per conversation"),
    ("Konuşma başına tool sayısı", "Tool calls per conversation"),
    ("Konuşma başına file sayısı", "File count per conversation"),
    ("Prompt yeniden yazma", "Prompt rewriting"),
    ("Cevap yeniden üretme", "Response regeneration"),
    ("Cevap in-place düzenleme", "Response in-place edit"),
    ("thinking içeren", "with thinking"),
    ("thinking bloğu", "thinking block"),
    ("Thinking bloğu", "Thinking block"),
    ("thinking payı", "thinking share"),
    ("Thinking payı", "Thinking share"),
    ("tool çağrısı", "tool call"),
    ("Tool çağrısı", "Tool call"),
    ("tool kullanımı", "tool use"),
    ("tool çeşitliliği", "tool diversity"),
    ("farklı tool sayısı", "distinct tool count"),
    ("thinking-li", "with thinking"),
    ("thinking-siz", "without thinking"),
    ("tool-kullananlar", "tool users"),
    ("tool kullananlarda", "among tool users"),
    ("tool kullanan konuşmalarda", "in tool-using conversations"),
    ("tool kullanmayanlarda", "among non-tool users"),
    ("tool kullanmayanlar", "non-tool users"),
    ("empty olmayan", "non-empty"),
    ("dosya yazma", "file write"),
    ("dosya okuma", "file read"),
    ("citation'lı", "with citations"),
    ("artifact'li", "with artifacts"),
    ("attachment'lı", "with attachments"),
    ("tool sayısı", "tool-call count"),
    ("kategori dağılımı", "category breakdown"),
    ("kova dağılımı", "bucket distribution"),
    ("Mesaj-arası", "Inter-message"),
    ("İlk-yanıt", "First-response"),
    ("konuşma ömrü", "conversation lifetime"),
    ("Konuşma ömrü", "Conversation lifetime"),
    ("fenced blok", "fenced block"),
    ("kod bloğu", "code block"),
    ("Kod bloğu", "Code block"),
    ("kod payı", "code share"),
    ("kod token", "code tokens"),
    ("token hacmi", "token volume"),
    ("attachment/file", "attachment/file"),
    ("file referansı", "file reference"),
    ("boyut bilgisi olan", "with size metadata"),
    ("file_type", "file_type"),
    ("ömür", "lifetime"),
    ("gecikmesi", "latency"),
    ("gecikme", "latency"),
    ("yoğunluğu", "intensity"),
    ("uzunluğu", "length"),
    ("uzunluk", "length"),
    ("boyutu", "size"),
    ("boyut", "size"),
    ("boy", "size"),
    ("dağılımı", "distribution"),
    ("kovaları", "buckets"),
    ("çağrı", "calls"),
    ("hata", "errors"),
    ("mesaj", "messages"),
    ("konuşma", "conversations"),
    ("oranı", "rate"),
    ("oran", "share"),
    ("medyan", "median"),
    ("toplam", "total"),
    ("aylık", "monthly"),
    ("haftalık", "weekly"),
    ("gün", "day"),
    ("saat", "hour"),
    ("boş", "empty"),
    ("dolu", "non-empty"),
    ("adet", "count"),
    ("tümü", "all"),
]


def setup_matplotlib() -> None:
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.bbox": "tight",
    })


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise SystemExit(
            f"DB not found: {DB_PATH}\nRun first: uv run python scripts/etl.py"
        )
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def save_fig(fig, path: Path) -> None:
    if not HAS_MPL:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def percentiles(values: Iterable[float], qs: Sequence[float] = (0.5, 0.9, 0.95, 0.99)) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs} | {"mean": float("nan"), "max": float("nan"), "min": float("nan"), "n": 0}
    out = {f"p{int(q*100)}": float(np.quantile(arr, q)) for q in qs}
    out.update({
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "n": int(arr.size),
    })
    return out


def fmt_int(n: float | int | None) -> str:
    if n is None:
        return "—"
    if isinstance(n, float) and (n != n):  # NaN
        return "—"
    return f"{int(round(n)):,}".replace(",", " ")


def fmt_float(n: float | None, digits: int = 2) -> str:
    if n is None:
        return "—"
    if n != n:
        return "—"
    return f"{n:.{digits}f}"


def markdown_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    return tabulate(rows, headers=headers, tablefmt="github")


def percentile_table(summary: dict[str, Any], label: str) -> str:
    """Single-metric table with p50/p90/p95/p99/max + mean."""
    return markdown_table(
        [[
            label,
            fmt_int(summary["n"]),
            fmt_int(summary["min"]),
            fmt_float(summary["mean"], 1),
            fmt_int(summary["p50"]),
            fmt_int(summary["p90"]),
            fmt_int(summary["p95"]),
            fmt_int(summary["p99"]),
            fmt_int(summary["max"]),
        ]],
        headers=["metric", "n", "min", "mean", "p50", "p90", "p95", "p99", "max"],
    )


@dataclass
class Block:
    # type: "paragraph" | "bullets" | "table" | "chart" | "legacy_asset" | "kv_list"
    type: str
    payload: dict[str, Any]


@dataclass
class Section:
    heading: str | dict[str, str]
    body: str
    # Block list for structured rendering. The frontend does not use markdown fallback;
    # new modules should populate blocks for visible UI.
    blocks: list[Block] | None = None


def write_report(out_dir: Path, title: str | dict[str, str], sections: Sequence[Section]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # report.md is still written in Turkish; allow bilingual payload inputs so
    # structured sections.json can evolve independently without breaking markdown.
    title_str = title if isinstance(title, str) else title.get("tr", "")

    lines = [f"# {title_str}", ""]
    for s in sections:
        h_str = s.heading if isinstance(s.heading, str) else s.heading.get("tr", "")
        lines.append(f"## {h_str}")
        lines.append("")
        lines.append(s.body.rstrip())
        lines.append("")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def sections_to_payload(sections: Sequence[Section]) -> list[dict[str, Any]]:
    """Convert a Section list to the JSON shape the server payload expects.

    Each section: {heading, body_md, blocks?: [{type, payload}]}. `body_md`
    is preserved for downloadable report.md compatibility; the frontend UI
    renders only the structured `blocks` tree."""
    out: list[dict[str, Any]] = []
    for s in sections:
        entry: dict[str, Any] = {"heading": s.heading, "body_md": s.body}
        # blocks is None  → no visible content in the UI; report.md/body_md is still preserved.
        # blocks == []     → "structured but no content"; the frontend hides the section.
        # blocks non-empty → structured render.
        if s.blocks is not None:
            entry["blocks"] = [{"type": b.type, "payload": b.payload} for b in s.blocks]
        out.append(entry)
    return out


# ───────────────── Block constructors ─────────────────
# Using these helpers instead of raw Block(...) in modules:
#   - prevents typos in type names,
#   - keeps the payload contract in one place.


def block_bullets(items: Sequence[str | dict[str, str]]) -> Block:
    """Highlights / bullet list items. Each item supports markdown inline (bold, code)."""
    return Block(type="bullets", payload={"items": list(items)})


def block_paragraph(text: str | dict[str, str]) -> Block:
    """Single paragraph / free text. Markdown inline is supported."""
    return Block(type="paragraph", payload={"text": text})


def block_table(
    columns: Sequence[dict[str, Any]],
    rows: Sequence[Sequence[Any]],
    caption: str | dict[str, str] | None = None,
) -> Block:
    """Structured table.
    columns: [{"key": "p50", "label": "p50", "align": "right"?}]
    rows:    [[val, val, ...], ...] (same order as columns).
    """
    payload: dict[str, Any] = {"columns": list(columns), "rows": [list(r) for r in rows]}
    if caption:
        payload["caption"] = caption
    return Block(type="table", payload=payload)


def block_bucket_chart(
    label: str | dict[str, str],
    buckets: Sequence[dict[str, Any]],
    image: str | None = None,
    xlabel: str | dict[str, str] | None = None,
) -> Block:
    """Bucket-bar chart: each element is {label, count, pct}.
    image: PNG filename showing the same data (fallback / archival)."""
    payload: dict[str, Any] = {
        "kind": "bucket_bar",
        "label": label,
        "buckets": list(buckets),
    }
    if image:
        payload["image"] = image
    if xlabel:
        payload["xlabel"] = xlabel
    return Block(type="chart", payload=payload)


def block_bar_chart(
    label: str | dict[str, str],
    data: Sequence[dict[str, Any]],
    x_key: str = "label",
    y_key: str = "count",
    xlabel: str | dict[str, str] | None = None,
    ylabel: str | dict[str, str] | None = None,
) -> Block:
    """Single-series vertical bar chart."""
    payload: dict[str, Any] = {
        "kind": "bar",
        "label": label,
        "data": list(data),
        "x_key": x_key,
        "y_key": y_key,
    }
    if xlabel:
        payload["xlabel"] = xlabel
    if ylabel:
        payload["ylabel"] = ylabel
    return Block(type="chart", payload=payload)


def block_grouped_bar_chart(
    label: str | dict[str, str],
    data: Sequence[dict[str, Any]],
    series: Sequence[dict[str, Any]],
    x_key: str = "label",
) -> Block:
    """Multi-series bar chart. series: [{"key": "human", "label": "Human"}]."""
    return Block(type="chart", payload={
        "kind": "grouped_bar",
        "label": label,
        "data": list(data),
        "series": list(series),
        "x_key": x_key,
    })


def block_line_chart(
    label: str | dict[str, str],
    data: Sequence[dict[str, Any]],
    series: Sequence[dict[str, Any]],
    x_key: str = "label",
) -> Block:
    """Single or multi-series line chart."""
    return Block(type="chart", payload={
        "kind": "multi_line" if len(series) > 1 else "line",
        "label": label,
        "data": list(data),
        "series": list(series),
        "x_key": x_key,
    })


def block_heatmap_chart(
    label: str | dict[str, str],
    cells: Sequence[dict[str, Any]],
    x_labels: Sequence[Any],
    y_labels: Sequence[Any],
) -> Block:
    """Heatmap. cells: [{"x": x, "y": y, "value": n}]."""
    return Block(type="chart", payload={
        "kind": "heatmap",
        "label": label,
        "cells": list(cells),
        "x_labels": list(x_labels),
        "y_labels": list(y_labels),
    })


def block_scatter_chart(
    label: str | dict[str, str],
    data: Sequence[dict[str, Any]],
    x_key: str = "x",
    y_key: str = "y",
    size_key: str = "size",
    xlabel: str | dict[str, str] | None = None,
    ylabel: str | dict[str, str] | None = None,
) -> Block:
    """Scatter/bubble chart."""
    payload: dict[str, Any] = {
        "kind": "scatter",
        "label": label,
        "data": list(data),
        "x_key": x_key,
        "y_key": y_key,
        "size_key": size_key,
    }
    if xlabel:
        payload["xlabel"] = xlabel
    if ylabel:
        payload["ylabel"] = ylabel
    return Block(type="chart", payload=payload)


def block_histogram_chart(
    label: str | dict[str, str],
    buckets: Sequence[dict[str, Any]],
    xlabel: str | dict[str, str] | None = None,
) -> Block:
    """Histogram-style bucket chart; the frontend uses the same renderer as bucket_bar."""
    payload: dict[str, Any] = {
        "kind": "histogram",
        "label": label,
        "buckets": list(buckets),
    }
    if xlabel:
        payload["xlabel"] = xlabel
    return Block(type="chart", payload=payload)


def block_delta_chart(
    label: str | dict[str, str],
    buckets: Sequence[dict[str, Any]],
    xlabel: str | dict[str, str] | None = None,
) -> Block:
    """Chart payload for positive/negative delta buckets."""
    payload: dict[str, Any] = {
        "kind": "delta",
        "label": label,
        "buckets": list(buckets),
    }
    if xlabel:
        payload["xlabel"] = xlabel
    return Block(type="chart", payload=payload)


def block_image(name: str, caption: str | dict[str, str] | None = None) -> Block:
    """Deprecated: does not render the PNG in the UI; listed as a legacy asset in the Data tab.
    New code should use dynamic chart helpers instead of this one."""
    payload: dict[str, Any] = {"name": name, "kind": "legacy_png"}
    if caption:
        payload["caption"] = caption
    return Block(type="legacy_asset", payload=payload)


def block_percentile_table(summary: dict[str, Any], label: str | dict[str, str]) -> Block:
    """Structured equivalent of the percentile_table() markdown."""
    columns = [
        {"key": "metric", "label": {"tr": "metrik", "en": "metric"}, "align": "left"},
        {"key": "n",      "label": "n",                              "align": "right"},
        {"key": "min",    "label": "min",                            "align": "right"},
        {"key": "mean",   "label": {"tr": "ort", "en": "mean"},      "align": "right"},
        {"key": "p50",    "label": "p50",                            "align": "right"},
        {"key": "p90",    "label": "p90",                            "align": "right"},
        {"key": "p95",    "label": "p95",                            "align": "right"},
        {"key": "p99",    "label": "p99",                            "align": "right"},
        {"key": "max",    "label": "max",                            "align": "right"},
    ]
    row = [
        label,
        fmt_int(summary.get("n")),
        fmt_int(summary.get("min")),
        fmt_float(summary.get("mean"), 1),
        fmt_int(summary.get("p50")),
        fmt_int(summary.get("p90")),
        fmt_int(summary.get("p95")),
        fmt_int(summary.get("p99")),
        fmt_int(summary.get("max")),
    ]
    return block_table(columns, [row])


def write_sections(out_dir: Path, title: str | dict[str, str], sections: Sequence[Section]) -> Path:
    """Write the Section list to sections.json. The server passes this file
    as payload["sections"] to the frontend. If not written, the payload
    sections field stays empty and the frontend falls back to markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sections.json"
    payload = {"title": title, "sections": sections_to_payload(sections)}
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def write_json(out_dir: Path, name: str, obj: Any) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def write_csv(out_dir: Path, name: str, rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> Path:
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return path


def _lookup_en(text: str) -> str | None:
    return _ANALYSIS_TEXT_EN.get(text) or _REPORT_HEADERS_EN.get(text)


def _replace_whole_term(text: str, source: str, target: str) -> str:
    pattern = re.compile(rf"(?<![\w]){re.escape(source)}(?![\w])", re.U)
    return pattern.sub(target, text)


def _apply_translation_rules(text: str) -> str:
    out = text
    for pattern, replacement in _TRANSLATION_RULES:
        if replacement is None:
            if pattern.pattern == r"^en yoğun saat (.+?) \((.+?) mesaj\), en yoğun gün (.+?) \((.+?) mesaj\); hafta sonu payı %(.+?)\.$":
                def _peak_full(m: re.Match[str]) -> str:
                    day = translate_text_en(m.group(3))
                    return f"Peak hour {m.group(1)} ({m.group(2)} messages), peak day {day} ({m.group(4)} messages); weekend share {m.group(5)}%."
                out = pattern.sub(_peak_full, out)
                continue
            if pattern.pattern == r"En yoğun gün: \*\*(.+?)\*\* \((.+?) mesaj\)":
                def _peak_day(m: re.Match[str]) -> str:
                    day = translate_text_en(m.group(1))
                    return f"Peak day: **{day}** ({m.group(2)} messages)"
                out = pattern.sub(_peak_day, out)
                continue
            if pattern.pattern == r"^(.+?) — kova dağılımı \(n=(.+?)\)$":
                def _bucket_dist(m: re.Match[str]) -> str:
                    return f"{translate_text_en(m.group(1))} — bucket distribution (n={m.group(2)})"
                out = pattern.sub(_bucket_dist, out)
                continue
            if pattern.pattern == r"^(.+?) \(n=(.+?)\)$":
                def _with_n(m: re.Match[str]) -> str:
                    return f"{translate_text_en(m.group(1))} (n={m.group(2)})"
                out = pattern.sub(_with_n, out)
                continue
            continue
        out = pattern.sub(replacement, out)
    return out


def translate_text_en(text: str) -> str:
    out = text
    for _ in range(3):
        prev = out
        exact = _lookup_en(out)
        if exact is not None:
            out = exact

        out = _apply_translation_rules(out)

        for source, target in sorted(_TERM_TRANSLATIONS, key=lambda item: len(item[0]), reverse=True):
            out = _replace_whole_term(out, source, target)

        out = re.sub(r"(\d+(?:[.,]\d+)?)dk\b", r"\1m", out, flags=re.U)
        out = re.sub(r"(\d+(?:[.,]\d+)?)sn\b", r"\1s", out, flags=re.U)
        out = re.sub(r"(\d+(?:[.,]\d+)?)sa\b", r"\1h", out, flags=re.U)
        out = re.sub(r"(\d+(?:[.,]\d+)?)gün\b", r"\1d", out, flags=re.U)

        if out == prev:
            break
    return out


def localize_text(text: str) -> str | dict[str, str]:
    translated = translate_text_en(text)
    if translated == text:
        return text
    return {"tr": text, "en": translated}


def localize_ui_payload(value: Any, *, key: str | None = None) -> Any:
    if key in _SKIP_LOCALIZE_KEYS:
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return localize_text(value)
    if isinstance(value, dict):
        if _LOCALE_KEYS.issubset(value.keys()):
            return value
        return {k: localize_ui_payload(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [localize_ui_payload(item) for item in value]
    return value
