// i18n runtime. Tüm UI metinleri `I18N.t(key, params?)` üzerinden çağrılır.
//
//   I18N.t("result.charts", { n: 6 })
//   I18N.setLang("tr"); I18N.getLang()
//   I18N.subscribe(fn)        → dil değişince callback
//   I18N.pick({tr, en})       → backend {tr, en} objesini aktif dile indirge
//   I18N.pick("Öne çıkanlar") → legacy/plain backend metnini exact lookup ile çevir
//   I18N.reportHeader(tr)     → legacy rapor başlığı TR→EN
//   I18N.summaryLabel(field)  → summary.json alan adı → etiket
//
// Sözlükler ayrı dosyalarda: dict-ui.js, dict-summary.js. Bu dosya onların
// window'a yazdığı __I18N_* globallerini tüketir.

(function () {
  const LS_KEY = "cca:lang";
  const SUPPORTED = ["en", "tr"];
  const DEFAULT = "en";

  const DICT            = window.__I18N_DICT            || { en: {}, tr: {} };
  const REPORT_HEADERS  = window.__I18N_REPORT_HEADERS  || { en: {} };
  const SUMMARY_LABELS  = window.__I18N_SUMMARY_LABELS  || { en: {}, tr: {} };
  const ANALYSIS_TEXT   = window.__I18N_ANALYSIS_TEXT   || { en: {}, tr: {} };
  const TR_CHAR_RE = /[ÇĞİÖŞÜçğıöşü]/u;
  const TR_WORD_RE = /\b(konuşma(?:lar|sız|lı)?|mesaj(?:lar|lık)?|hata(?:lı|sız)?|çağrı(?:lar|sı|ları)?|özet|öne|çıkanlar|aylık|haftalık|günlük|gün|saat|hafta|ay|ömür|gecikme|dağılımı|kovaları|boyut|aralığı|sayısı|davranış|müdahale|yeniden|düzenleme|uzunluğu|oranı|boş|dolu|farklı|toplam|içeren|asistan|insan|doküman|proje|dosya|çok-root|çok|edit noktası|retry noktası|başına)\b/iu;

  // ─── Aktif dil durumu ───
  let current = readLang();
  const subs = new Set();

  function readLang() {
    try {
      const v = localStorage.getItem(LS_KEY);
      if (v && SUPPORTED.includes(v)) return v;
    } catch (e) { /* LS erişilmez olabilir */ }
    return DEFAULT;
  }

  // İlk render için <html lang> etiketini senkronla.
  if (typeof document !== "undefined") {
    document.documentElement.setAttribute("lang", current);
  }

  // ─── Format ───
  function format(str, params) {
    if (!params) return str;
    return str.replace(/\{(\w+)\}/g, (m, k) =>
      Object.prototype.hasOwnProperty.call(params, k) ? String(params[k]) : m
    );
  }

  // ─── Ana lookup ───
  function t(key, params) {
    const table = DICT[current] || DICT[DEFAULT];
    const raw = table[key];
    if (raw === undefined) {
      // Aktif dilde yoksa default dile düş
      const other = DICT[DEFAULT][key];
      if (other === undefined) return key;
      return format(other, params);
    }
    return format(raw, params);
  }

  function setLang(lang) {
    if (!SUPPORTED.includes(lang)) return;
    if (lang === current) return;
    current = lang;
    try { localStorage.setItem(LS_KEY, lang); } catch (e) { /* ignore */ }
    document.documentElement.setAttribute("lang", lang);
    subs.forEach((fn) => {
      try { fn(lang); } catch (e) { /* ignore */ }
    });
  }

  function getLang() { return current; }
  function supported() { return SUPPORTED.slice(); }

  function subscribe(fn) {
    subs.add(fn);
    return () => subs.delete(fn);
  }

  function lookupPlainString(text) {
    if (typeof text !== "string") return undefined;
    return (
      (ANALYSIS_TEXT[current] || {})[text] ??
      (REPORT_HEADERS[current] || {})[text] ??
      (SUMMARY_LABELS[current] || {})[text]
    );
  }

  function escapeRegExp(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function replaceWholeTerm(text, source, target) {
    const pattern = new RegExp(`(^|[^\\p{L}\\p{N}_])(${escapeRegExp(source)})(?=$|[^\\p{L}\\p{N}_])`, "giu");
    return text.replace(pattern, (_, prefix) => `${prefix}${target}`);
  }

  function hasLegacyTranslationSignals(text) {
    return TR_CHAR_RE.test(text) || TR_WORD_RE.test(text);
  }

  function translateFragments(text) {
    let out = String(text);
    const rules = [
      [/Tüm saat\/gün bilgisi yerel saate \(\*\*(.+?)\*\*\) çevrilerek hesaplanmıştır\./gu, "All hour/day values are computed in local time (**$1**)."],
      [/^(.+?) konuşma \(boş: (.+?)\); medyan (.+?) mesaj, (.+?) human turn, p95 (.+?) mesaj, max (.+?)\. Medyan ~(.+?) token\.$/u, "$1 conversations (empty: $2); median $3 messages, $4 human turns, p95 $5 messages, max $6. Median ~$7 tokens."],
      [/Toplam: \*\*(.+?)\*\* konuşma \(boş: (.+?)\)/gu, "Total: **$1** conversations (empty: $2)"],
      [/^human medyan ~(.+?) token, assistant medyan ~(.+?) token; toplam token hacminin %(.+?)'i assistant; assistant cevabının %(.+?)'i kod bloğu\.$/u, "Human median ~$1 tokens, assistant median ~$2 tokens; $3% of the total token volume belongs to the assistant; $4% of assistant replies is code blocks."],
      [/^Human: \*\*(.+?)\*\* mesaj \(tool-only: (.+?)\)$/u, "Human: **$1** messages (tool-only: $2)"],
      [/^Assistant: \*\*(.+?)\*\* mesaj \(tool-only: (.+?)\)$/u, "Assistant: **$1** messages (tool-only: $2)"],
      [/^Medyan: human \*\*~(.+?) token\*\*, assistant \*\*~(.+?) token\*\*$/u, "Median: human **~$1 tokens**, assistant **~$2 tokens**"],
      [/^p95: human \*\*~(.+?) token\*\*, assistant \*\*~(.+?) token\*\*$/u, "p95: human **~$1 tokens**, assistant **~$2 tokens**"],
      [/^Max: human \*\*~(.+?) token\*\*, assistant \*\*~(.+?) token\*\*$/u, "Max: human **~$1 tokens**, assistant **~$2 tokens**"],
      [/^Token hacmi: human \*\*~(.+?)\*\* \(%(.+?)\), assistant \*\*~(.+?)\*\* \(%(.+?)\) — oran \*\*(.+?)×\*\*$/u, "Token volume: human **~$1** ($2%), assistant **~$3** ($4%) — ratio **$5x**"],
      [/^Assistant cevabının \*\*%(.+?)\*\*'i kod bloğu; mesajların \*\*%(.+?)\*\*'i \((.+?)\) en az bir kod bloğu içeriyor$/u, "Code blocks make up **$1%** of assistant replies; **$2%** of messages ($3) contain at least one code block"],
      [/^konuşmaların %(.+?)'inde en az bir thinking bloğu; thinking'li cevap medyan ~(.+?) token, thinking'siz ~(.+?); thinking bloğu medyan ~(.+?) token\.$/u, "At least one thinking block appears in $1% of conversations; median reply with thinking is ~$2 tokens, without thinking ~$3; median thinking block is ~$4 tokens."],
      [/Toplam konuşma \(non-empty\): \*\*(.+?)\*\*/gu, "Total conversations (non-empty): **$1**"],
      [/^Toplam konuşma \(non-empty\): \*\*(.+?)\*\*; thinking içeren: \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)$/u, "Total conversations (non-empty): **$1**; with thinking: **$2** (**$3**)"],
      [/^Assistant mesajı: \*\*(.+?)\*\*; thinking bloğu olan: \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)$/u, "Assistant messages: **$1**; with thinking block: **$2** (**$3**)"],
      [/^Cevap medyanı: thinking yok \*\*~(.+?) token\*\*, thinking var \*\*~(.+?) token\*\*$/u, "Response median: without thinking **~$1 tokens**, with thinking **~$2 tokens**"],
      [/^Cevap p95: thinking yok \*\*~(.+?) token\*\*, thinking var \*\*~(.+?) token\*\*$/u, "Response p95: without thinking **~$1 tokens**, with thinking **~$2 tokens**"],
      [/^Thinking bloğu: medyan \*\*~(.+?) token\*\*, p95 \*\*~(.+?) token\*\*, max \*\*~(.+?) token\*\*$/u, "Thinking block: median **~$1 tokens**, p95 **~$2 tokens**, max **~$3 tokens**"],
      [/^Toplam assistant token'ının \*\*%(.+?)\*\*'i thinking \(görünen text dışı\)$/u, "$1% of assistant tokens is thinking (outside visible text)"],
      [/^Thinking'li konuşma başına toplam thinking: medyan \*\*~(.+?) token\*\*, max \*\*~(.+?) token\*\*$/u, "Total thinking per conversation with thinking: median **~$1 tokens**, max **~$2 tokens**"],
      [/^Thinking-li konuşmaların \*\*%(.+?)\*\*'i tool da kullanıyor; thinking-siz konuşmalarda bu oran \*\*%(.+?)\*\*$/u, "$1% of conversations with thinking also use tools; in conversations without thinking this rate is **$2%**"],
      [/^İlk assistant mesajında thinking oranı \*\*%(.+?)\*\*; sonraki mesajlarda \*\*%(.+?)\*\*$/u, "Thinking rate is **$1%** in the first assistant message and **$2%** in later messages"],
      [/^en yoğun saat (.+?) \((.+?) mesaj\), en yoğun gün (.+?) \((.+?) mesaj\); hafta sonu payı %(.+?)\.$/u, (_, hour, hourCount, day, dayCount, pct) => `Peak hour ${hour} (${hourCount} messages), peak day ${localizePlainString(day) || day} (${dayCount} messages); weekend share ${pct}%.`],
      [/En az 1 tool çağrısı olan: \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)/gu, "With at least 1 tool call: **$1** (**$2**)"],
      [/Toplam tool çağrısı: \*\*(.+?)\*\*/gu, "Total tool calls: **$1**"],
      [/Tool sonuç hata oranı: \*\*(.+?)\*\* \/ \*\*(.+?)\*\* → \*\*(.+?)\*\*/gu, "Tool-result error rate: **$1** / **$2** -> **$3**"],
      [/Tool çağrısı süresi: medyan \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*/gu, "Tool-call duration: median **$1**, p95 **$2**, max **$3**"],
      [/^(.+?)\/(.+?) konuşmada tool kullanılmış \(%(.+?)\); tool kullanan konuşmalarda medyan (.+?), p95 (.+?), max (.+?) tool; toplam (.+?) çağrı, %(.+?) hata, medyan süre (.+?)\.$/u, "$1/$2 conversations used tools ($3%); among tool-using conversations the median is $4, p95 $5, max $6 tools; $7 total calls, $8% errors, median duration $9."],
      [/^(.+?) konuşma \(≥2 mesaj\); medyan ömür \*\*(.+?)\*\*, mesaj-arası medyan \*\*(.+?)\*\*, ilk-yanıt medyan \*\*(.+?)\*\*\. Çok-oturumlu konuşma: %(.+?)\.$/u, "$1 conversations (>=2 messages); median lifetime **$2**, inter-message median **$3**, first-response median **$4**. Multi-session conversations: $5%."],
      [/Hafta sonu payı: \*\*(.+?)\*\*/gu, "Weekend share: **$1**"],
      [/En yoğun saat: \*\*(.+?)\*\* \((.+?) mesaj\)/gu, "Peak hour: **$1** ($2 messages)"],
      [/En yoğun gün: \*\*(.+?)\*\* \((.+?) mesaj\)/gu, (_, day, count) => `Peak day: **${localizePlainString(day) || day}** (${count} messages)`],
      [/Medyan: \*\*(.+?)\*\* mesaj, \*\*(.+?)\*\* human turn, \*\*~(.+?) token\*\*/gu, "Median: **$1** messages, **$2** human turns, **~$3 tokens**"],
      [/p95: \*\*(.+?)\*\* mesaj, \*\*(.+?)\*\* human turn, \*\*~(.+?) token\*\*/gu, "p95: **$1** messages, **$2** human turns, **~$3 tokens**"],
      [/Max: \*\*(.+?)\*\* mesaj, \*\*(.+?)\*\* human turn, \*\*~(.+?) token\*\*/gu, "Max: **$1** messages, **$2** human turns, **~$3 tokens**"],
      [/^(.+?) attachment \+ (.+?) file; medyan boy (.+?), p95 (.+?); ilk human mesajında attachment\/file oranı %(.+?)\.$/u, "$1 attachments + $2 files; median size $3, p95 $4; attachment/file rate in the first human message is $5%."],
      [/^Toplam attachment: (.+)$/u, "Total attachments: $1"],
      [/^Toplam file \(referans\): (.+)$/u, "Total files (references): $1"],
      [/^Attachment içeren konuşma: (.+)$/u, "Conversations with attachment: $1"],
      [/^File referansı içeren konuşma: (.+)$/u, "Conversations with file reference: $1"],
      [/^İlk human mesajında attachment\/file: (.+)$/u, "Attachment/file in the first human message: $1"],
      [/^En sık file_type: (.+)$/u, "Most common file_type: $1"],
      [/^Toplam messages: \*\*(.+?)\*\*$/u, "Total messages: **$1**"],
      [/^Toplam code tokens: \*\*(.+?)\*\* \/ total text token: \*\*(.+?)\*\* → \*\*(.+?)\*\*$/u, "Total code tokens: **$1** / total text tokens: **$2** -> **$3**"],
      [/^Code block içeren assistant mesajı: \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)$/u, "Assistant messages with a code block: **$1** (**$2**)"],
      [/^Toplam thinking token: \*\*(.+?)\*\* \/ total text \+ thinking token: \*\*(.+?)\*\* → \*\*(.+?)\*\*$/u, "Total thinking tokens: **$1** / total text + thinking tokens: **$2** -> **$3**"],
      [/^Tool-only messages = `chars_text=0` — yalnız tool_use \/ tool_result bloğu taşıyan mesajlar; dağılım hesabından dışlanır\.$/u, "Tool-only messages = `chars_text=0` — messages that contain only tool_use / tool_result blocks; they are excluded from the distribution calculation."],
      [/^Code block = üçlü-backtick ile çevrili fenced block; kapanışı olmayan bloklar metin sonuna kadar sayılır\. Inline backtick \(`` ` ``\) dahil değildir\.$/u, "Code block = a fenced block delimited by triple backticks; unterminated blocks are counted until the end of the text. Inline backticks (`` ` ``) are excluded."],
      [/^Toplam citation: \*\*(.+?)\*\*$/u, "Total citations: **$1**"],
      [/^Toplam artifact çağrısı: \*\*(.+?)\*\* \(parse edilebilen input'lu\)$/u, "Total artifact calls: **$1** (with parseable input)"],
      [/^Artifact üretilen conversations: \*\*(.+?)\*\*$/u, "Conversations with artifacts: **$1**"],
      [/^Toplam project: \*\*(.+?)\*\* \(starter: \*\*(.+?)\*\* \/ %(.+?), private: \*\*(.+?)\*\* \/ %(.+?)\)$/u, "Total projects: **$1** (starter: **$2** / $3%, private: **$4** / $5%)"],
      [/^Toplam knowledge doc: \*\*(.+?)\*\*; median size \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*$/u, "Total knowledge docs: **$1**; median size **$2**, p95 **$3**, max **$4**"],
      [/^Komut (?:dağılımı|distribution): create \*\*(.+?)\*\* · update \*\*(.+?)\*\* · rewrite \*\*(.+?)\*\* · diğer \*\*(.+?)\*\*$/u, "Command distribution: create **$1** · update **$2** · rewrite **$3** · other **$4**"],
      [/^İterasyon (?:oranı|rate) \(update\+rewrite\) \/ create = \*\*(.+?)\*\*$/u, "Iteration rate (update+rewrite) / create = **$1**"],
      [/^Tipsiz kayıt: \*\*(.+?)\*\* \(input'ta `type` alanı yok — büyük ihtimalle `update` komutu\)$/u, "Records without type: **$1** (`type` is missing in the input — most likely an `update` command)"],
      [/^content length \(create\/rewrite için, n=\*\*(.+?)\*\*\): medyan \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*$/u, "Content length (create/rewrite, n=**$1**): median **$2**, p95 **$3**, max **$4**"],
      [/^Toplam fenced code block: \*\*(.+?)\*\* \(\*\*(.+?)\*\* mesajda, \*\*(.+?)\*\* konuşmada\)$/u, "Total fenced code blocks: **$1** (across **$2** messages, **$3** conversations)"],
      [/^Toplam code tokens: \*\*(.+?)\*\*, total kod karakter: \*\*(.+?)\*\*$/u, "Total code tokens: **$1**, total code characters: **$2**"],
      [/^Assistant: \*\*(.+?)\*\* blok \/ \*\*(.+?)\*\* messages \(assistant mesajlarının \*\*(.+?)\*\*'inde en az bir blok\)$/u, "Assistant: **$1** blocks / **$2** messages (**$3** of assistant messages include at least one block)"],
      [/^Human: \*\*(.+?)\*\* blok \/ \*\*(.+?)\*\* messages \(human mesajlarının \*\*(.+?)\*\*'inde en az bir blok\)$/u, "Human: **$1** blocks / **$2** messages (**$3** of human messages include at least one block)"],
      [/^En sık dil: \*\*(.+?)\*\* \((.+?) blok\)$/u, "Most common language: **$1** ($2 blocks)"],
      [/^Citation'lı messages: \*\*(.+?)\*\*, citation'lı conversations: \*\*(.+?)\*\*$/u, "Messages with citations: **$1**, conversations with citations: **$2**"],
      [/^En çok citation: \*\*(.+?)\*\* \((.+?) citation\)$/u, "Most citations: **$1** ($2 citations)"],
      [/^Web tool call olmayan ama citation'lı conversations: \*\*(.+?)\*\*$/u, "Conversations with citations but no web-tool call: **$1**"],
      [/^Kapsam: \*\*(.+?)\*\* conversations \(≥2 mesajlı\)$/u, "Coverage: **$1** conversations (>=2 messages)"],
      [/^0 saniyelik ömür \(aynı timestamp\): \*\*(.+?)\*\*$/u, "Zero-second lifetime (same timestamp): **$1**"],
      [/^Inter-message gerçek gap \(tüm ardışık çiftler, n=\*\*(.+?)\*\*\): medyan \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*$/u, "Real inter-message gap (all consecutive pairs, n=**$1**): median **$2**, p95 **$3**"],
      [/^First-response latency — human→assistant geçişleri, n=\*\*(.+?)\*\*: medyan \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*$/u, "First-response latency — human→assistant transitions, n=**$1**: median **$2**, p95 **$3**"],
      [/^Çok-oturumlu conversations \(içinde >1sa gap olan\): \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)$/u, "Multi-session conversations (with a >1h gap inside): **$1** (**$2**)"],
      [/^Toplam tool calls: \*\*(.+?)\*\*, hatalı: \*\*(.+?)\*\* \(genel share \*\*(.+?)\*\*\)$/u, "Total tool calls: **$1**, errors: **$2** (overall share **$3**)"],
      [/^Toplam tool_result: \*\*(.+?)\*\*$/u, "Total tool results: **$1**"],
      [/^Hata \(`is_error=true`\): \*\*(.+?)\*\* \(\*\*(.+?)\*\*\)$/u, "Errors (`is_error=true`): **$1** (**$2**)"],
      [/^Total tool calls: \*\*(.+?)\*\*, hatalı: \*\*(.+?)\*\* \(genel share \*\*(.+?)\*\*\)$/u, "Total tool calls: **$1**, errors: **$2** (overall share **$3**)"],
      [/^Hatalı messages: \*\*(.+?)\*\*, hatalı conversations: \*\*(.+?)\*\*$/u, "Error messages: **$1**, error conversations: **$2**"],
      [/^Hatalı messages: \*\*(.+?)\*\*, hatalı conversations: \*\*(.+?)\*\*$/u, "Error messages: **$1**, error conversations: **$2**"],
      [/^En yüksek errors rate \(n≥20\): \*\*(.+?)\*\* — (.+?)\/(.+?) \((.+?)\)$/u, "Highest error rate (n>=20): **$1** — $2/$3 ($4)"],
      [/^En çok errors üreten: \*\*(.+?)\*\* — (.+?) errors \((.+?) calls, (.+?)\)$/u, "Most error-producing tool: **$1** — $2 errors ($3 calls, $4)"],
      [/^En yaygın kategori: \*\*(.+?)\*\* \((.+?) errors\)$/u, "Most common category: **$1** ($2 errors)"],
      [/^Tool kullanan konuşmalarda medyan \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\* tool$/u, "Among tool-using conversations: median **$1**, p95 **$2**, max **$3** tools"],
      [/^Tool kullanan konuşmalar ortalama \*\*(.+?)\*\* mesaj \/ tool kullanmayanlar \*\*(.+?)\*\* \(medyan\)$/u, "Conversation length comparison: tool-using **$1** messages / non-tool-using **$2** (median)"],
      [/^(.+?)\/(.+?) konuşmada tool kullanılmış \(%(.+?)\); tool kullanan konuşmalarda median (.+?)\s+p95 (.+?)\s+max (.+?) tool; total (.+?) calls\s+%(.+?) errors\s+median süre (.+?)\.$/u, "$1/$2 conversations used tools ($3%); among tool-using conversations the median is $4, p95 $5, max $6 tools; $7 total calls, $8% errors, median duration $9."],
      [/^(.+?)\/(.+?) konuşmada tool kullanılmış \(%(.+?)\); in tool-using conversations median (.+?)\s+p95 (.+?)\s+max (.+?) tool; total (.+?) calls\s+%(.+?) errors\s+median süre (.+?)\.$/u, "$1/$2 conversations used tools ($3%); among tool-using conversations the median is $4, p95 $5, max $6 tools; $7 total calls, $8% errors, median duration $9."],
      [/^Konuşma başına farklı tool \(tool kullananlarda\): medyan \*\*(.+?)\*\*, max \*\*(.+?)\*\*$/u, "Distinct tools per conversation (tool users): median **$1**, max **$2**"],
      [/^Tool kullanan konuşmalar ortalama \*\*(.+?)\*\* messages \/ non-tool users \*\*(.+?)\*\* \(median\)$/u, "Conversation length comparison: tool-using **$1** messages / non-tool users **$2** (median)"],
      [/^Konuşma başına farklı tool \(among tool users\): median \*\*(.+?)\*\*, max \*\*(.+?)\*\*$/u, "Distinct tools per conversation (tool users): median **$1**, max **$2**"],
      [/^conversations başına farklı tool \(among tool users\): median \*\*(.+?)\*\*, max \*\*(.+?)\*\*$/u, "Distinct tools per conversation (tool users): median **$1**, max **$2**"],
      [/^conversations per farklı tool \(among tool users\): median \*\*(.+?)\*\*, max \*\*(.+?)\*\*$/u, "Distinct tools per conversation (tool users): median **$1**, max **$2**"],
      [/^conversations başına blok: median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*$/u, "Blocks per conversation: median **$1**, p95 **$2**, max **$3**"],
      [/^conversations başına citation: median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*$/u, "Citations per conversation: median **$1**, p95 **$2**, max **$3**"],
      [/^Prompt yeniden yazma \(human fork\): \*\*(.+?)\*\* edit \/ \*\*(.+?)\*\* edit noktası \/ \*\*(.+?)\*\* konuşma$/u, "Prompt rewriting (human fork): **$1** edits / **$2** edit points / **$3** conversations"],
      [/^Cevap yeniden üretme \(assistant fork \/ retry\): \*\*(.+?)\*\* retry \/ \*\*(.+?)\*\* retry noktası \/ \*\*(.+?)\*\* konuşma$/u, "Response regeneration (assistant fork / retry): **$1** retries / **$2** retry points / **$3** conversations"],
      [/^Cevap in-place düzenleme \(assistant\): \*\*(.+?)\*\* mesaj \/ \*\*(.+?)\*\* konuşma$/u, "Response in-place edit (assistant): **$1** messages / **$2** conversations"],
      [/^Çok-root konuşma \(root_count > 1\): \*\*(.+?)\*\* \(toplam (.+?) boş olmayan konuşmadan\)$/u, "Multi-root conversations (root_count > 1): **$1** (out of $2 non-empty conversations)"],
      [/^Prompt rewriting \(human fork\): \*\*(.+?)\*\* edit \/ \*\*(.+?)\*\* edit noktası \/ \*\*(.+?)\*\* conversations$/u, "Prompt rewriting (human fork): **$1** edits / **$2** edit points / **$3** conversations"],
      [/^Response regeneration \(assistant fork \/ retry\): \*\*(.+?)\*\* retry \/ \*\*(.+?)\*\* retry noktası \/ \*\*(.+?)\*\* conversations$/u, "Response regeneration (assistant fork / retry): **$1** retries / **$2** retry points / **$3** conversations"],
      [/^Çok-root conversations \(root_count > 1\): \*\*(.+?)\*\* \(total (.+?) empty olmayan konuşmadan\)$/u, "Multi-root conversations (root_count > 1): **$1** (out of $2 non-empty conversations)"],
      [/^0 saniyelik lifetime \(aynı timestamp\): \*\*(.+?)\*\*$/u, "Zero-second lifetime (same timestamp): **$1**"],
      [/^Inter-message gerçek gap \(tüm ardışık çiftler, n=\*\*(.+?)\*\*\): median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*$/u, "Real inter-message gap (all consecutive pairs, n=**$1**): median **$2**, p95 **$3**"],
      [/^First-response latency — human→assistant geçişleri, n=\*\*(.+?)\*\*: median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*$/u, "First-response latency — human→assistant transitions, n=**$1**: median **$2**, p95 **$3**"],
      [/^content length \(create\/rewrite için, n=\*\*(.+?)\*\*\): median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*$/u, "Content length (create/rewrite, n=**$1**): median **$2**, p95 **$3**, max **$4**"],
      [/^conversations başına blok: median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*$/u, "Blocks per conversation: median **$1**, p95 **$2**, max **$3**"],
      [/^conversations başına citation: median \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*, max \*\*(.+?)\*\*$/u, "Citations per conversation: median **$1**, p95 **$2**, max **$3**"],
      [/^Pearson korelasyon \(conversations başına citation ve web_search\+web_fetch çağrısı\): \*\*r = (.+?)\*\*\. Web tool call yapılmamış ama with citations conversations sayısı: \*\*(.+?)\*\* — scatter grafiğinde x=0 noktasındakiler\.$/u, "Pearson correlation (citations per conversation vs web_search+web_fetch calls): **r = $1**. Conversations with citations but no web-tool call: **$2** — the points at x=0 in the scatter plot."],
      [/^Thinking share = `tokens_thinking \/ \(tokens_thinking \+ tokens_text\)`; paydada assistant mesajının text'i sıfırsa \(yalnız tool-use \/ tool-result\) hesap dışı kalır\.$/u, "Thinking share = `tokens_thinking / (tokens_thinking + tokens_text)`; if the assistant message text is zero in the denominator (tool-use / tool-result only), it is excluded."],
      [/^(\d+) total edit eylemi, (\d+) edit noktasında \(benzersiz prompt konumu\), (\d+) konuşmada\.$/u, "$1 total edit actions across $2 edit points (unique prompt positions), $3 conversations."],
      [/^Bir konuşmadaki max edit noktası: (.+?); tek bir prompt'un kaç kez yeniden yazıldığı ayrı ölçüm \(aşağıdaki child-sayısı distribution\)\.$/u, "Max edit points in one conversation: $1; how many times a single prompt was rewritten is measured separately (child-count distribution below)."],
      [/^(\d+) total retry eylemi, (\d+) retry noktasında \(benzersiz asistan cevap konumu\), (\d+) konuşmada\.$/u, "$1 total retry actions across $2 retry points (unique assistant-response positions), $3 conversations."],
      [/^Bir konuşmadaki max retry noktası: (.+?)\.$/u, "Max retry points in one conversation: $1."],
      [/^Toplam thinking token: \*\*(.+?)\*\* \/ total text \+ thinking token: \*\*(.+?)\*\* → \*\*(.+?)\*\*$/u, "Total thinking tokens: **$1** / total text + thinking tokens: **$2** -> **$3**"],
      [/^Voice_note içeren mesaj sayısı: \*\*(.+?)\*\* — örnekleme istatistik için çok küçük; ayrı analiz yapılmamıştır\.$/u, "Messages with voice_note: **$1** — the sample is too small for a separate statistical analysis."],
      [/^(.+?) project, (.+?) doc; project başına doc medyan \*\*(.+?)\*\*, p95 \*\*(.+?)\*\*; doc uzunluk medyan \*\*(.+?)\*\*, max \*\*(.+?)\*\*; prompt_template dolu: \*\*(.+?)\*\* \/ \*\*(.+?)\*\*\.$/u, "$1 projects, $2 docs; median docs per project **$3**, p95 **$4**; median doc length **$5**, max **$6**; prompt_template populated: **$7** / **$8**."],
      [/^(.+?) artifact çağrısı, (.+?) konuşmada; en sık tip: \*\*(.+?)\*\* \(\*\*(.+?)\*\*\); iterasyon oranı \(update\+rewrite\)\/create = \*\*(.+?)\*\*\.$/u, "$1 artifact calls across $2 conversations; most common type: **$3** (**$4**); iteration ratio (update+rewrite)/create = **$5**."],
      [/^(.+?) fenced blok \/ (.+?) mesaj \/ (.+?) konuşma; blok medyan (.+?), p95 (.+?); en sık dil: (.+?)\.$/u, "$1 fenced blocks / $2 messages / $3 conversations; median block size $4, p95 $5; most common language: $6."],
      [/^(.+?) citation \/ (.+?) mesaj \/ (.+?) konuşma; (.+?) domain; en sık: (.+?) \((.+?)\)\.$/u, "$1 citations / $2 messages / $3 conversations; $4 domains; most common: $5 ($6)."],
      [/^(.+?) hata \/ (.+?) çağrı \(%(.+?)\); (.+?) konuşma; en yüksek oran: (.+?) \(%(.+?)\)$/u, "$1 errors / $2 calls ($3%); $4 conversations; highest rate: $5 ($6%)"],
      [/^(.+?) prompt yeniden yazma \/ (.+?) retry \/ (.+?) in-place edit \/ (.+?) çok-root$/u, "$1 prompt rewrites / $2 retries / $3 in-place edits / $4 multi-root"],
      [/^(.+?) — kova dağılımı \(n=(.+?)\)$/u, (_, label, n) => `${localizePlainString(label) || label} — bucket distribution (n=${n})`],
      [/^(.+?) \(n=(.+?)\)$/u, (_, label, n) => `${localizePlainString(label) || label} (n=${n})`],
      [/^En çok kullanılan 15 tool \(toplam (.+?) çağrı\)$/u, "Top 15 tools ($1 total calls)"],
      [/Ardışık human\/assistant mesaj çiftleri arası saniye\./gu, "Seconds between consecutive human/assistant message pairs."],
      [/Human mesajından hemen sonraki assistant mesajına kadar geçen süre\./gu, "Elapsed time until the assistant message immediately following a human message."],
    ];
    for (const [pattern, replacement] of rules) {
      out = out.replace(pattern, replacement);
    }
    out = out.replace(/(\d+(?:[.,]\d+)?)dk\b/gu, "$1m");
    out = out.replace(/(\d+(?:[.,]\d+)?)sn\b/gu, "$1s");
    out = out.replace(/(\d+(?:[.,]\d+)?)sa\b/gu, "$1h");
    out = out.replace(/(\d+(?:[.,]\d+)?)gün\b/gu, "$1d");

    const terms = [
      ["Thinking × tool kullanımı çakışması", "Thinking × tool-use overlap"],
      ["Attachment/file içeren konuşma oranı", "Share of conversations with attachment/file"],
      ["Toplam assistant token'ının", "Of total assistant tokens"],
      ["Thinking'li konuşma başına toplam thinking", "Total thinking per conversation with thinking"],
      ["İlk assistant mesajında thinking oranı", "Thinking rate in the first assistant message"],
      ["Konuşma başına human turn sayısı", "Human turns per conversation"],
      ["Konuşma başına toplam token", "Total tokens per conversation"],
      ["Konuşma başına mesaj sayısı", "Messages per conversation"],
      ["Konuşma başına thinking token yükü", "Thinking-token load per conversation"],
      ["Konuşma başına attachment/file sayısı", "Attachment/file count per conversation"],
      ["Konuşma başına attachment sayısı", "Attachment count per conversation"],
      ["Konuşma başına artifact çağrısı", "Artifact calls per conversation"],
      ["Konuşma başına iterasyon", "Iterations per conversation"],
      ["Konuşma başına tool çeşitliliği", "Tool diversity per conversation"],
      ["Konuşma başına tool sayısı", "Tool calls per conversation"],
      ["Konuşma başına file sayısı", "File count per conversation"],
      ["Prompt yeniden yazma", "Prompt rewriting"],
      ["Cevap yeniden üretme", "Response regeneration"],
      ["Cevap in-place düzenleme", "Response in-place edit"],
      ["thinking içeren", "with thinking"],
      ["thinking bloğu", "thinking block"],
      ["Thinking bloğu", "Thinking block"],
      ["thinking payı", "thinking share"],
      ["Thinking payı", "Thinking share"],
      ["tool çağrısı", "tool call"],
      ["Tool çağrısı", "Tool call"],
      ["tool kullanımı", "tool use"],
      ["tool çeşitliliği", "tool diversity"],
      ["farklı tool sayısı", "distinct tool count"],
      ["thinking-li", "with thinking"],
      ["thinking-siz", "without thinking"],
      ["konuşmalarda", "conversations"],
      ["tool-kullananlar", "tool users"],
      ["tool kullananlarda", "among tool users"],
      ["tool kullanan konuşmalarda", "in tool-using conversations"],
      ["tool kullanmayanlarda", "among non-tool users"],
      ["tool kullanmayanlar", "non-tool users"],
      ["empty olmayan", "non-empty"],
      ["dosya yazma", "file write"],
      ["dosya okuma", "file read"],
      ["citation'lı", "with citations"],
      ["artifact'li", "with artifacts"],
      ["attachment'lı", "with attachments"],
      ["tool sayısı", "tool-call count"],
      ["kategori dağılımı", "category breakdown"],
      ["kova dağılımı", "bucket distribution"],
      ["Mesaj-arası", "Inter-message"],
      ["İlk-yanıt", "First-response"],
      ["konuşma ömrü", "conversation lifetime"],
      ["Konuşma ömrü", "Conversation lifetime"],
      ["fenced blok", "fenced block"],
      ["kod bloğu", "code block"],
      ["Kod bloğu", "Code block"],
      ["kod payı", "code share"],
      ["kod token", "code tokens"],
      ["kod karakter", "code characters"],
      ["token hacmi", "token volume"],
      ["attachment/file", "attachment/file"],
      ["file referansı", "file reference"],
      ["boyut bilgisi olan", "with size metadata"],
      ["file_type", "file_type"],
      ["ömür", "lifetime"],
      ["gecikmesi", "latency"],
      ["gecikme", "latency"],
      ["yoğunluğu", "intensity"],
      ["uzunluğu", "length"],
      ["uzunluk", "length"],
      ["boyutu", "size"],
      ["boyut", "size"],
      ["boy", "size"],
      ["dağılımı", "distribution"],
      ["kovaları", "buckets"],
      ["çağrı", "calls"],
      ["hata", "errors"],
      ["mesaj", "messages"],
      ["konuşma", "conversations"],
      ["oranı", "rate"],
      ["oran", "share"],
      ["medyan", "median"],
      ["toplam", "total"],
      ["aylık", "monthly"],
      ["haftalık", "weekly"],
      ["gün", "day"],
      ["saat", "hour"],
      ["boş", "empty"],
      ["dolu", "non-empty"],
      ["adet", "count"],
      ["başına", "per"],
      ["karakter", "characters"],
    ];
    for (const [src, dst] of terms.sort((a, b) => b[0].length - a[0].length)) {
      out = replaceWholeTerm(out, src, dst);
    }
    return out;
  }

  function localizePlainString(text) {
    if (typeof text !== "string" || current === "tr") return text;
    const exact = lookupPlainString(text);
    if (exact !== undefined) return exact;
    if (!hasLegacyTranslationSignals(text)) return text;
    return translateFragments(text);
  }

  // Backend'den gelen {tr, en} metin objesini aktif dile indirger.
  // Plain string path yalnız legacy payload için exact lookup fallback'idir.
  function pick(obj) {
    if (obj == null) return "";
    if (typeof obj === "string") return localizePlainString(obj);
    const chosen = obj[current] || obj[DEFAULT] || Object.values(obj)[0] || "";
    return typeof chosen === "string" ? localizePlainString(chosen) : chosen;
  }

  // Python tarafında üretilen TR rapor başlığını aktif dile çevirir.
  // TR modda identity. Sözlükte olmayanlar olduğu gibi döner (safe fallback).
  function reportHeader(tr) {
    return localizePlainString(tr);
  }

  // snake_case alan adını okunaklı label'a çevirir (sözlük → humanize fallback).
  function humanize(field) {
    return field
      .split(/[_.]/)
      .filter(Boolean)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");
  }

  function summaryLabel(field) {
    if (typeof field === "object" && field != null) return pick(field);
    const table = SUMMARY_LABELS[current] || SUMMARY_LABELS[DEFAULT] || {};
    return table[field] || humanize(field);
  }

  window.I18N = {
    t, setLang, getLang, supported, subscribe, pick,
    reportHeader, summaryLabel,
    analysisText: localizePlainString,
  };
})();
