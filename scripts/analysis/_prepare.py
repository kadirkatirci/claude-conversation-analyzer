"""
Materialized tables for analysis: _stats_message, _stats_conversation.

Idempotent: DROP TABLE IF EXISTS + CREATE TABLE AS.
Token counting uses cl100k_base when tiktoken is available, otherwise falls back
to a char/4 approximation (see _tokenizer.py). Both are approximate upper bounds.

ensure(con, tz="Europe/Istanbul") is the single entry point.
"""

from __future__ import annotations

import re
import sys
import time

import duckdb

from analysis import _tokenizer

# Fenced code block delimited by triple backticks.
# Line-start `\`\`\`[lang]` -> content -> line-start `\`\`\``.
# Unterminated blocks are accepted until end of text.
_CODE_BLOCK_RE = re.compile(
    r"```[^\n]*\n(.*?)(?:```|\Z)",
    re.DOTALL,
)

# Each fenced block opening captures the language tag and body separately.
# group(1) = language identifier (may be empty), group(2) = body.
_CODE_BLOCK_LANG_RE = re.compile(
    r"```([^\n]*)\n(.*?)(?:```|\Z)",
    re.DOTALL,
)

# Inline backtick: single `...` pattern. Uses lookarounds to avoid colliding
# with triple-backtick (3+) fenced block openings.
_INLINE_CODE_RE = re.compile(
    r"(?<!`)`(?!`)([^`\n]+?)`(?!`)",
)


def _drop_stats(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS _stats_code_block")
    con.execute("DROP TABLE IF EXISTS _stats_message")
    con.execute("DROP TABLE IF EXISTS _stats_conversation")


def _build_stats_message(con: duckdb.DuckDBPyConnection, tz: str) -> None:
    """
    _stats_message: combined text length and tool counts for each message.

    The tokens_text and tokens_thinking columns are populated later by Python.
    """
    con.execute(f"""
        CREATE TABLE _stats_message AS
        SELECT
          m.uuid                                           AS message_uuid,
          m.conversation_uuid,
          m.sender,
          m.created_at,
          m.created_at AT TIME ZONE '{tz}'                 AS created_local,
          COALESCE(SUM(CASE WHEN cb.type IN ('text','voice_note')
                            THEN cb.text_length END), 0)   AS chars_text,
          COALESCE(SUM(CASE WHEN cb.type = 'thinking'
                            THEN cb.text_length END), 0)   AS chars_thinking,
          SUM(CASE WHEN cb.type='tool_use'    THEN 1 ELSE 0 END) AS tool_use_count,
          SUM(CASE WHEN cb.type='tool_result' THEN 1 ELSE 0 END) AS tool_result_count,
          m.attachment_count,
          m.file_count,
          CAST(NULL AS INTEGER)                             AS tokens_text,
          CAST(NULL AS INTEGER)                             AS tokens_thinking,
          CAST(NULL AS INTEGER)                             AS chars_code,
          CAST(NULL AS INTEGER)                             AS tokens_code
        FROM messages m
        LEFT JOIN content_blocks cb ON cb.message_uuid = m.uuid
        GROUP BY ALL;
    """)


def _extract_text_per_message(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    """
    Returns (message_uuid, text_concat, thinking_concat) per message.
    Only rows with chars_text>0 or chars_thinking>0 — empty ones are 0 tokens anyway.
    """
    rows = con.execute("""
        WITH texts AS (
          SELECT
            cb.message_uuid,
            string_agg(
              CASE WHEN cb.type IN ('text','voice_note')
                   THEN json_extract_string(cb.raw, '$.text')
                   WHEN cb.type = 'voice_note'
                   THEN json_extract_string(cb.raw, '$.text')
              END,
              '\n' ORDER BY cb.position
            ) AS text_concat,
            string_agg(
              CASE WHEN cb.type = 'thinking'
                   THEN json_extract_string(cb.raw, '$.thinking')
              END,
              '\n' ORDER BY cb.position
            ) AS thinking_concat
          FROM content_blocks cb
          WHERE cb.type IN ('text','voice_note','thinking')
          GROUP BY cb.message_uuid
        )
        SELECT message_uuid,
               COALESCE(text_concat, ''),
               COALESCE(thinking_concat, '')
        FROM texts
    """).fetchall()
    return rows


def _extract_code_blocks(text: str) -> str:
    """Joins all fenced code block bodies in the text with `\n\n`.

    Strips the triple-backtick opening and closing lines — only the code body
    is counted. Unterminated blocks are taken until end of text.
    """
    if not text or "```" not in text:
        return ""
    return "\n\n".join(m.group(1) for m in _CODE_BLOCK_RE.finditer(text))


def _tokenize_messages(rows: list[tuple[str, str, str]]) -> list[tuple[int, int, int, int, str]]:
    """
    Produces a list of (tokens_text, tokens_thinking, chars_code, tokens_code,
    message_uuid). chars_code and tokens_code are computed only from
    fenced code blocks in the text body (thinking is excluded).
    """
    out: list[tuple[int, int, int, int, str]] = []
    t0 = time.time()
    text_corpus = [r[1] for r in rows]
    thinking_corpus = [r[2] for r in rows]
    code_corpus = [_extract_code_blocks(t) for t in text_corpus]

    text_tokens = _tokenizer.encode_batch(text_corpus)
    thinking_tokens = _tokenizer.encode_batch(thinking_corpus)
    code_tokens = _tokenizer.encode_batch(code_corpus)

    for i, r in enumerate(rows):
        out.append((
            len(text_tokens[i]),
            len(thinking_tokens[i]),
            len(code_corpus[i]),
            len(code_tokens[i]),
            r[0],
        ))
    print(f"  tokenized {len(rows)} messages in {time.time() - t0:.1f}s", file=sys.stderr)
    return out


def _apply_tokens(con: duckdb.DuckDBPyConnection, tokens: list[tuple[int, int, int, int, str]]) -> None:
    """
    Fills the tokens_text / tokens_thinking / chars_code / tokens_code columns in
    _stats_message. Strategy: CREATE a temp table, then UPDATE ... FROM in one pass.
    """
    con.execute("""
        CREATE TEMP TABLE _tok(
          tokens_text INTEGER,
          tokens_thinking INTEGER,
          chars_code INTEGER,
          tokens_code INTEGER,
          message_uuid VARCHAR
        )
    """)
    con.executemany(
        "INSERT INTO _tok VALUES (?, ?, ?, ?, ?)",
        tokens,
    )
    con.execute("""
        UPDATE _stats_message AS s
        SET tokens_text      = COALESCE(t.tokens_text, 0),
            tokens_thinking  = COALESCE(t.tokens_thinking, 0),
            chars_code       = COALESCE(t.chars_code, 0),
            tokens_code      = COALESCE(t.tokens_code, 0)
        FROM _tok AS t
        WHERE s.message_uuid = t.message_uuid
    """)
    # Set remaining NULLs to 0 (messages with no text/thinking at all)
    con.execute("""
        UPDATE _stats_message
        SET tokens_text     = COALESCE(tokens_text, 0),
            tokens_thinking = COALESCE(tokens_thinking, 0),
            chars_code      = COALESCE(chars_code, 0),
            tokens_code     = COALESCE(tokens_code, 0)
    """)
    con.execute("DROP TABLE _tok")


def _extract_code_blocks_detailed(text: str) -> list[tuple[str, str]]:
    r"""Returns fenced blocks in the text as a list of (language_tag, body).

    The language tag is read from the opening line after ```. Whitespace is
    trimmed; if empty it stays an empty string. Body is the content between
    the backtick delimiters (boundaries excluded).
    """
    if not text or "```" not in text:
        return []
    out: list[tuple[str, str]] = []
    for m in _CODE_BLOCK_LANG_RE.finditer(text):
        lang = (m.group(1) or "").strip()
        body = m.group(2) or ""
        out.append((lang, body))
    return out


def _count_inline_code(text: str) -> int:
    if not text or "`" not in text:
        return 0
    return sum(1 for _ in _INLINE_CODE_RE.finditer(text))


def _apply_intervention_flags(con: duckdb.DuckDBPyConnection) -> None:
    """
    Adds intervention flags per message:
      inplace_edit_flag     — updated_at > created_at + 60s within the same uuid and chars_text>0
      edit_delta_seconds    — epoch(updated_at - created_at) (all messages)
      fork_parent_flag      — the message uuid is a parent with >1 children (excluding root parent)
      fork_child_flag       — the message's parent has >1 children (excluding root parent)
      fork_child_continued  — whether this fork child has descendants (did the user continue that branch)

    Root parent: messages.parent_message_uuid IS NULL or '00000000-0000-4000-8000-000000000000'.
    In multi-root conversations each root message stays as a separate "conversation root";
    fork flags are computed only for NON-ROOT parents.
    """
    con.execute("ALTER TABLE _stats_message ADD COLUMN IF NOT EXISTS inplace_edit_flag BOOLEAN")
    con.execute("ALTER TABLE _stats_message ADD COLUMN IF NOT EXISTS edit_delta_seconds DOUBLE")
    con.execute("ALTER TABLE _stats_message ADD COLUMN IF NOT EXISTS fork_parent_flag BOOLEAN")
    con.execute("ALTER TABLE _stats_message ADD COLUMN IF NOT EXISTS fork_child_flag BOOLEAN")
    con.execute("ALTER TABLE _stats_message ADD COLUMN IF NOT EXISTS fork_child_continued BOOLEAN")

    con.execute("""
        UPDATE _stats_message AS s
        SET edit_delta_seconds =
              CASE WHEN m.updated_at IS NOT NULL AND m.created_at IS NOT NULL
                   THEN epoch(m.updated_at - m.created_at) END,
            inplace_edit_flag =
              CASE WHEN m.updated_at IS NOT NULL AND m.created_at IS NOT NULL
                        AND s.chars_text > 0
                        AND epoch(m.updated_at - m.created_at) > 60 THEN TRUE
                   ELSE FALSE END
        FROM messages AS m
        WHERE s.message_uuid = m.uuid
    """)

    con.execute("""
        CREATE TEMP TABLE _pc AS
        SELECT parent_message_uuid AS puuid,
               COUNT(*) AS child_count
        FROM messages
        WHERE parent_message_uuid IS NOT NULL
          AND parent_message_uuid <> '00000000-0000-4000-8000-000000000000'
        GROUP BY 1
    """)
    con.execute("""
        UPDATE _stats_message AS s
        SET fork_parent_flag =
              CASE WHEN pc.child_count IS NOT NULL AND pc.child_count > 1 THEN TRUE ELSE FALSE END
        FROM _pc AS pc
        WHERE s.message_uuid = pc.puuid
    """)
    con.execute("UPDATE _stats_message SET fork_parent_flag = COALESCE(fork_parent_flag, FALSE)")

    con.execute("""
        UPDATE _stats_message AS s
        SET fork_child_flag =
              CASE WHEN pc.child_count > 1 THEN TRUE ELSE FALSE END
        FROM messages AS m
        LEFT JOIN _pc AS pc ON pc.puuid = m.parent_message_uuid
        WHERE s.message_uuid = m.uuid
          AND m.parent_message_uuid IS NOT NULL
          AND m.parent_message_uuid <> '00000000-0000-4000-8000-000000000000'
    """)
    con.execute("UPDATE _stats_message SET fork_child_flag = COALESCE(fork_child_flag, FALSE)")

    con.execute("""
        UPDATE _stats_message AS s
        SET fork_child_continued =
              CASE WHEN s.fork_child_flag
                        AND EXISTS (
                          SELECT 1 FROM messages m2
                          WHERE m2.parent_message_uuid = s.message_uuid
                        )
                   THEN TRUE ELSE FALSE END
    """)

    con.execute("DROP TABLE _pc")


def _build_stats_code_block(
    con: duckdb.DuckDBPyConnection,
    text_per_msg: list[tuple[str, str, str]],
) -> None:
    """
    _stats_code_block: one row per fenced code block.

    Columns: message_uuid, conversation_uuid, sender, position (0-based block
    index within the message), language (normalized), chars, tokens, inline_count.
    inline_count is duplicated at the message level (convenience in a single row;
    carries the same value per message — not relevant in block-level reports).

    A separate `_stats_message.inline_code_count` column is also added (via
    ALTER TABLE) for message-level inline code counting.
    """
    t0 = time.time()

    # Extract text body from the (uuid, text, _) list and parse
    rows_out: list[tuple[str, int, str, int, int]] = []  # (uuid, pos, lang, chars, tokens)
    inline_counts: list[tuple[int, str]] = []  # (count, uuid)

    all_bodies: list[str] = []
    body_refs: list[tuple[str, int, str]] = []  # (uuid, pos, lang)

    for uuid, text, _thinking in text_per_msg:
        blocks = _extract_code_blocks_detailed(text)
        for i, (lang, body) in enumerate(blocks):
            body_refs.append((uuid, i, lang))
            all_bodies.append(body)
        inline_counts.append((_count_inline_code(text), uuid))

    # Batch tokenize
    if all_bodies:
        tokens_batch = _tokenizer.encode_batch(all_bodies)
    else:
        tokens_batch = []

    for (uuid, pos, lang), body, toks in zip(body_refs, all_bodies, tokens_batch):
        rows_out.append((uuid, pos, lang, len(body), len(toks)))

    # Create table
    con.execute("""
        CREATE TABLE _stats_code_block(
            message_uuid      VARCHAR,
            position          INTEGER,
            language          VARCHAR,
            chars             INTEGER,
            tokens            INTEGER
        )
    """)
    if rows_out:
        con.executemany(
            "INSERT INTO _stats_code_block VALUES (?, ?, ?, ?, ?)",
            rows_out,
        )

    # Sender + conversation join can be done later via a view on _stats_message;
    # we keep it simple here and avoid denormalization.
    # Also add + update the message-level inline code count column
    con.execute("ALTER TABLE _stats_message ADD COLUMN IF NOT EXISTS inline_code_count INTEGER")
    con.execute("CREATE TEMP TABLE _inl(cnt INTEGER, message_uuid VARCHAR)")
    con.executemany("INSERT INTO _inl VALUES (?, ?)", inline_counts)
    con.execute("""
        UPDATE _stats_message AS s
        SET inline_code_count = COALESCE(i.cnt, 0)
        FROM _inl AS i
        WHERE s.message_uuid = i.message_uuid
    """)
    con.execute("""
        UPDATE _stats_message
        SET inline_code_count = COALESCE(inline_code_count, 0)
    """)
    con.execute("DROP TABLE _inl")

    n_blocks = len(rows_out)
    print(f"  extracted {n_blocks} fenced code blocks in {time.time() - t0:.1f}s", file=sys.stderr)


def _build_stats_conversation(con: duckdb.DuckDBPyConnection, tz: str) -> None:
    con.execute(f"""
        CREATE TABLE _stats_conversation AS
        WITH root_counts AS (
          SELECT conversation_uuid,
                 SUM(CASE WHEN parent_message_uuid IS NULL
                               OR parent_message_uuid = '00000000-0000-4000-8000-000000000000'
                          THEN 1 ELSE 0 END) AS root_count
          FROM messages
          GROUP BY conversation_uuid
        ),
        sender_forks AS (
          -- parent sender assistant → children human → prompt edit (human_fork_count)
          -- parent sender human     → children assistant → retry (assistant_fork_count)
          SELECT sm.conversation_uuid,
                 SUM(CASE WHEN sm.fork_parent_flag AND sm.sender='assistant' THEN 1 ELSE 0 END) AS human_fork_count,
                 SUM(CASE WHEN sm.fork_parent_flag AND sm.sender='human'     THEN 1 ELSE 0 END) AS assistant_fork_count,
                 SUM(CASE WHEN sm.inplace_edit_flag THEN 1 ELSE 0 END) AS inplace_edit_count
          FROM _stats_message sm
          GROUP BY 1
        )
        SELECT
          c.uuid AS conversation_uuid,
          c.name,
          c.created_at,
          c.created_at AT TIME ZONE '{tz}'                                     AS created_local,
          c.message_count,
          MIN(sm.created_at)                                                    AS first_msg_at,
          MAX(sm.created_at)                                                    AS last_msg_at,
          COALESCE(epoch(MAX(sm.created_at) - MIN(sm.created_at)), 0)          AS lifetime_seconds,
          COALESCE(SUM(CASE WHEN sm.sender='human'     THEN sm.chars_text END), 0)      AS chars_human,
          COALESCE(SUM(CASE WHEN sm.sender='assistant' THEN sm.chars_text END), 0)      AS chars_assistant,
          COALESCE(SUM(CASE WHEN sm.sender='human'     THEN sm.tokens_text END), 0)     AS tokens_human,
          COALESCE(SUM(CASE WHEN sm.sender='assistant' THEN sm.tokens_text END), 0)     AS tokens_assistant,
          COALESCE(SUM(sm.chars_thinking), 0)                                            AS chars_thinking_total,
          COALESCE(SUM(sm.tokens_thinking), 0)                                           AS tokens_thinking_total,
          COALESCE(SUM(sm.tool_use_count), 0)                                            AS tool_calls_total,
          COALESCE(SUM(sm.attachment_count), 0)                                          AS attachment_total,
          COALESCE(SUM(sm.file_count), 0)                                                AS file_total,
          COALESCE(BOOL_OR(sm.chars_thinking > 0), FALSE)                                AS has_thinking,
          COALESCE(SUM(CASE WHEN sm.sender='human'     THEN 1 ELSE 0 END), 0)            AS human_turn_count,
          COALESCE(SUM(CASE WHEN sm.sender='assistant' THEN 1 ELSE 0 END), 0)            AS assistant_turn_count,
          COALESCE(BOOL_OR(sm.fork_parent_flag), FALSE)                                  AS has_fork,
          COALESCE(SUM(CASE WHEN sm.fork_parent_flag THEN 1 ELSE 0 END), 0)              AS fork_count,
          COALESCE(sf.human_fork_count, 0)                                               AS human_fork_count,
          COALESCE(sf.assistant_fork_count, 0)                                           AS assistant_fork_count,
          COALESCE(sf.inplace_edit_count, 0)                                             AS inplace_edit_count,
          COALESCE(rc.root_count, 0)                                                     AS root_count
        FROM conversations c
        LEFT JOIN _stats_message sm ON sm.conversation_uuid = c.uuid
        LEFT JOIN sender_forks sf   ON sf.conversation_uuid = c.uuid
        LEFT JOIN root_counts  rc   ON rc.conversation_uuid = c.uuid
        GROUP BY c.uuid, c.name, c.created_at, c.message_count,
                 sf.human_fork_count, sf.assistant_fork_count, sf.inplace_edit_count,
                 rc.root_count;
    """)


def _sanity(con: duckdb.DuckDBPyConnection) -> None:
    avg = con.execute("SELECT AVG(message_count) FROM _stats_conversation").fetchone()[0]
    n = con.execute("SELECT COUNT(*) FROM _stats_message").fetchone()[0]
    assert avg is not None and n is not None, "stats tables empty"
    n_convo = con.execute("SELECT COUNT(*) FROM _stats_conversation").fetchone()[0]
    print(f"  sanity ok: {n_convo} convos (avg msg={avg:.3f}), {n} messages", file=sys.stderr)


def ensure(con: duckdb.DuckDBPyConnection, tz: str = "Europe/Istanbul") -> None:
    t0 = time.time()
    print("prepare: dropping old _stats_*", file=sys.stderr)
    _drop_stats(con)
    print("prepare: _stats_message (sql)", file=sys.stderr)
    _build_stats_message(con, tz)
    print("prepare: extracting text per message", file=sys.stderr)
    rows = _extract_text_per_message(con)
    print(f"  {len(rows)} messages with text/thinking", file=sys.stderr)
    print("prepare: tiktoken (cl100k_base)", file=sys.stderr)
    tokens = _tokenize_messages(rows)
    print("prepare: writing token columns", file=sys.stderr)
    _apply_tokens(con, tokens)
    print("prepare: intervention flags (fork / in-place edit)", file=sys.stderr)
    _apply_intervention_flags(con)
    print("prepare: _stats_code_block (fenced blocks + inline)", file=sys.stderr)
    _build_stats_code_block(con, rows)
    print("prepare: _stats_conversation", file=sys.stderr)
    _build_stats_conversation(con, tz)
    _sanity(con)
    con.execute("CHECKPOINT")
    print(f"prepare done in {time.time() - t0:.1f}s", file=sys.stderr)
