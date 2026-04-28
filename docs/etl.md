# ETL: JSON → DuckDB

Transforms raw export files into normalized tables in
[`build/claude.duckdb`](../build/). Uses streaming parse (`ijson`) so the
potentially large `conversations.json` is never loaded entirely into memory.

## Running

```sh
uv run python scripts/etl.py
```

Each run **recreates `build/claude.duckdb` from scratch**. Idempotent.

Typical duration: ~2 minutes on macOS Apple Silicon. Output DB size depends on
the export — raw block JSON is stored in `content_blocks.raw`.

## Tables

The full schema is in the `init_schema()` function inside
[scripts/etl.py](../scripts/etl.py). Summary:

| Table            | Notes                                                                  |
| ---------------- | ---------------------------------------------------------------------- |
| `users`          | From `users.json`, typically one row                                   |
| `projects`       | Top-level entries from `projects.json`                                 |
| `docs`           | `projects[*].docs[*]` — linked via `project_uuid` FK                  |
| `conversations`  | Top-level entries from `conversations.json`; `account_uuid` FK         |
| `messages`       | `chat_messages`; `conversation_uuid` FK, `parent_message_uuid` self-FK |
| `content_blocks` | All blocks (text/thinking/tool_use/tool_result/token_budget/voice_note). `raw` JSON column stores the full block. |
| `tool_calls`     | Denormalized view of `type='tool_use'` blocks                          |
| `tool_results`   | Denormalized view of `type='tool_result'` blocks                       |
| `attachments`    | Pasted text; stores `extracted_content` length, not the text itself    |
| `files`          | File references (uuid + name); no content                              |

### Why both `content_blocks` and `tool_calls`/`tool_results`?

`content_blocks` is a single wide view; convenient for chronological iteration.
`tool_calls` and `tool_results` are denormalized — this keeps SQL joins simple
and indexes hot columns. Both derive from the same source, so they stay
consistent:

```sql
SELECT
  (SELECT COUNT(*) FROM content_blocks WHERE type='tool_use') AS via_blocks,
  (SELECT COUNT(*) FROM tool_calls)                            AS via_calls;
-- both return the same count
```

### Why is `attachments.extracted_content` not stored?

Pasted text can range from tiny to tens of MB. For size control we store only
the length. When content analysis is needed, fall back to `content_blocks.raw`
or the raw JSON files.

**Note:** This decision makes full-text search over attachment text impossible
via this table. If full-text search is needed, either add a new table
(`attachment_contents`) or search `data/conversations.json` directly.

## Timezone

All timestamp columns are `TIMESTAMPTZ`. Raw JSON arrives in UTC (`Z` suffix);
DuckDB applies the local timezone on the read side (Python side needs `pytz` —
see [pyproject.toml](../pyproject.toml)).

To convert back to UTC if needed: `SELECT created_at AT TIME ZONE 'UTC' ...`.

## Reproducibility

This script + `data/` + `pyproject.toml` + `uv.lock` produce the same DB. On
every change, `uv.lock` is committed (pinned dependencies).

## Known limitations

1. `attachments.extracted_content` is not stored (see above).
2. `content_blocks.raw` is stored as JSON, which significantly increases DB
   size. Once hot queries are identified, the `raw` column can be dropped or
   split into Parquet.
3. There is still no link between `conversations` and `projects` (an export
   limitation; see [schema.md](schema.md#conversations--projects-relationship)).
