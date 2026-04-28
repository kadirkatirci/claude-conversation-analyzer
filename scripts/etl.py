"""
ETL: data/{conversations,projects,users}.json -> build/claude.duckdb

Idempotent: re-running recreates build/claude.duckdb from scratch.

Tables:
    users
    projects
    docs               (projects.docs)
    conversations
    messages
    content_blocks     (all block types)
    tool_calls         (denormalized view of type='tool_use' blocks)
    tool_results       (denormalized view of type='tool_result' blocks)
    attachments        (including paste text)
    files              (uuid + name references only)

Usage:
    uv run python scripts/etl.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BUILD = ROOT / "build"
DB_PATH = BUILD / "claude.duckdb"

CONVERSATIONS_JSON = DATA / "conversations.json"
PROJECTS_JSON = DATA / "projects.json"
USERS_JSON = DATA / "users.json"

# Module-level constants for CLI flow; programmatic use goes through `run_etl()`.
_CURRENT_CONVOS: Path = CONVERSATIONS_JSON
_CURRENT_PROJECTS: Path = PROJECTS_JSON
_CURRENT_USERS: Path = USERS_JSON


def as_json(obj) -> str | None:
    """For DuckDB JSON columns. Returns NULL when None."""
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE users (
            uuid VARCHAR PRIMARY KEY,
            full_name VARCHAR,
            email_address VARCHAR,
            verified_phone_number VARCHAR
        );

        CREATE TABLE projects (
            uuid VARCHAR PRIMARY KEY,
            name VARCHAR,
            description VARCHAR,
            prompt_template VARCHAR,
            is_private BOOLEAN,
            is_starter_project BOOLEAN,
            created_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ,
            creator_uuid VARCHAR,
            creator_full_name VARCHAR
        );

        CREATE TABLE docs (
            uuid VARCHAR PRIMARY KEY,
            project_uuid VARCHAR,
            filename VARCHAR,
            content VARCHAR,
            content_length INTEGER,
            created_at TIMESTAMPTZ
        );

        CREATE TABLE conversations (
            uuid VARCHAR PRIMARY KEY,
            name VARCHAR,
            summary VARCHAR,
            account_uuid VARCHAR,
            created_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ,
            message_count INTEGER
        );

        CREATE TABLE messages (
            uuid VARCHAR PRIMARY KEY,
            conversation_uuid VARCHAR,
            parent_message_uuid VARCHAR,
            sender VARCHAR,
            position INTEGER,                -- index within chat_messages array
            created_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ,
            text_length INTEGER,             -- length of legacy .text field
            content_block_count INTEGER,
            attachment_count INTEGER,
            file_count INTEGER
        );

        CREATE TABLE content_blocks (
            block_id VARCHAR PRIMARY KEY,    -- synthetic: message_uuid#position
            message_uuid VARCHAR,
            conversation_uuid VARCHAR,
            position INTEGER,
            type VARCHAR,                    -- text | thinking | tool_use | tool_result | token_budget | voice_note
            start_timestamp TIMESTAMPTZ,
            stop_timestamp TIMESTAMPTZ,
            text_length INTEGER,             -- meaningful for type=text/thinking/voice_note
            raw JSON                         -- full raw JSON of the block (for denormalization)
        );

        CREATE TABLE tool_calls (
            block_id VARCHAR PRIMARY KEY,
            message_uuid VARCHAR,
            conversation_uuid VARCHAR,
            tool_use_id VARCHAR,
            name VARCHAR,
            integration_name VARCHAR,
            is_mcp_app BOOLEAN,
            mcp_server_url VARCHAR,
            input JSON,
            start_timestamp TIMESTAMPTZ,
            stop_timestamp TIMESTAMPTZ
        );

        CREATE TABLE tool_results (
            block_id VARCHAR PRIMARY KEY,
            message_uuid VARCHAR,
            conversation_uuid VARCHAR,
            tool_use_id VARCHAR,             -- corresponding tool_calls.tool_use_id
            name VARCHAR,
            is_error BOOLEAN,
            content JSON,
            structured_content JSON,
            start_timestamp TIMESTAMPTZ,
            stop_timestamp TIMESTAMPTZ
        );

        CREATE TABLE attachments (
            message_uuid VARCHAR,
            position INTEGER,
            file_name VARCHAR,
            file_size BIGINT,
            file_type VARCHAR,
            content_length INTEGER,
            PRIMARY KEY (message_uuid, position)
        );

        CREATE TABLE files (
            message_uuid VARCHAR,
            position INTEGER,
            file_uuid VARCHAR,
            file_name VARCHAR,
            PRIMARY KEY (message_uuid, position)
        );
    """)


def load_users(con: duckdb.DuckDBPyConnection) -> int:
    with _CURRENT_USERS.open("r", encoding="utf-8") as f:
        users = json.load(f)
    rows = [
        (u["uuid"], u.get("full_name"), u.get("email_address"), u.get("verified_phone_number"))
        for u in users
    ]
    con.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", rows)
    return len(rows)


def load_projects(con: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    with _CURRENT_PROJECTS.open("r", encoding="utf-8") as f:
        projects = json.load(f)

    proj_rows = []
    doc_rows = []
    for p in projects:
        creator = p.get("creator") or {}
        proj_rows.append((
            p["uuid"],
            p.get("name"),
            p.get("description"),
            p.get("prompt_template"),
            p.get("is_private"),
            p.get("is_starter_project"),
            p.get("created_at"),
            p.get("updated_at"),
            creator.get("uuid"),
            creator.get("full_name"),
        ))
        for d in p.get("docs") or []:
            content = d.get("content") or ""
            doc_rows.append((
                d["uuid"],
                p["uuid"],
                d.get("filename"),
                content,
                len(content),
                d.get("created_at"),
            ))

    con.executemany(
        "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        proj_rows,
    )
    con.executemany(
        "INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?)",
        doc_rows,
    )
    return len(proj_rows), len(doc_rows)


def load_conversations(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Parse conversations.json (a top-level JSON array) and load into DuckDB."""
    counts = {
        "conversations": 0,
        "messages": 0,
        "content_blocks": 0,
        "tool_calls": 0,
        "tool_results": 0,
        "attachments": 0,
        "files": 0,
    }

    # Using an appender would be faster, but heterogeneous types make batched
    # executemany preferable. Batches also keep memory usage bounded.
    BATCH = 2000
    buffers: dict[str, list] = {k: [] for k in counts}

    INSERTS = {
        "conversations":
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?)",
        "messages":
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        "content_blocks":
            "INSERT INTO content_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        "tool_calls":
            "INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        "tool_results":
            "INSERT INTO tool_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        "attachments":
            "INSERT INTO attachments VALUES (?, ?, ?, ?, ?, ?)",
        "files":
            "INSERT INTO files VALUES (?, ?, ?, ?)",
    }

    def flush(table: str, force: bool = False) -> None:
        buf = buffers[table]
        if not buf:
            return
        if len(buf) < BATCH and not force:
            return
        con.executemany(INSERTS[table], buf)
        counts[table] += len(buf)
        buf.clear()

    t0 = time.time()
    seen_convos = 0
    with _CURRENT_CONVOS.open("r", encoding="utf-8") as f:
        all_convos = json.load(f)
    for convo in all_convos:
        seen_convos += 1
        convo_uuid = convo["uuid"]
        messages = convo.get("chat_messages") or []

        buffers["conversations"].append((
            convo_uuid,
            convo.get("name"),
            convo.get("summary"),
            (convo.get("account") or {}).get("uuid"),
            convo.get("created_at"),
            convo.get("updated_at"),
            len(messages),
        ))
        flush("conversations")

        for m_pos, msg in enumerate(messages):
            msg_uuid = msg["uuid"]
            content = msg.get("content") or []
            attachments = msg.get("attachments") or []
            files = msg.get("files") or []
            text_len = len(msg.get("text") or "")

            buffers["messages"].append((
                msg_uuid,
                convo_uuid,
                msg.get("parent_message_uuid"),
                msg.get("sender"),
                m_pos,
                msg.get("created_at"),
                msg.get("updated_at"),
                text_len,
                len(content),
                len(attachments),
                len(files),
            ))
            flush("messages")

            for c_pos, block in enumerate(content):
                block_id = f"{msg_uuid}#{c_pos}"
                btype = block.get("type")
                # type-specific text length
                if btype == "text":
                    tl = len(block.get("text") or "")
                elif btype == "thinking":
                    tl = len(block.get("thinking") or "")
                elif btype == "voice_note":
                    tl = len(block.get("text") or "")
                else:
                    tl = None

                buffers["content_blocks"].append((
                    block_id,
                    msg_uuid,
                    convo_uuid,
                    c_pos,
                    btype,
                    block.get("start_timestamp"),
                    block.get("stop_timestamp"),
                    tl,
                    as_json(block),
                ))
                flush("content_blocks")

                if btype == "tool_use":
                    buffers["tool_calls"].append((
                        block_id,
                        msg_uuid,
                        convo_uuid,
                        block.get("id"),
                        block.get("name"),
                        block.get("integration_name"),
                        block.get("is_mcp_app"),
                        block.get("mcp_server_url"),
                        as_json(block.get("input")),
                        block.get("start_timestamp"),
                        block.get("stop_timestamp"),
                    ))
                    flush("tool_calls")
                elif btype == "tool_result":
                    buffers["tool_results"].append((
                        block_id,
                        msg_uuid,
                        convo_uuid,
                        block.get("tool_use_id"),
                        block.get("name"),
                        block.get("is_error"),
                        as_json(block.get("content")),
                        as_json(block.get("structured_content")),
                        block.get("start_timestamp"),
                        block.get("stop_timestamp"),
                    ))
                    flush("tool_results")

            for a_pos, att in enumerate(attachments):
                extracted = att.get("extracted_content") or ""
                buffers["attachments"].append((
                    msg_uuid,
                    a_pos,
                    att.get("file_name"),
                    att.get("file_size"),
                    att.get("file_type"),
                    len(extracted),
                ))
                flush("attachments")

            for fpos, fil in enumerate(files):
                buffers["files"].append((
                    msg_uuid,
                    fpos,
                    fil.get("file_uuid"),
                    fil.get("file_name"),
                ))
                flush("files")

        if seen_convos % 500 == 0:
            print(
                f"  ... {seen_convos:>5d} conversations "
                f"({time.time() - t0:5.1f}s)",
                file=sys.stderr,
            )

    for table in buffers:
        flush(table, force=True)

    return counts


def run_etl(
    conversations_path: Path,
    projects_path: Path,
    users_path: Path,
    db_path: Path,
    *,
    on_progress=None,
) -> dict[str, int]:
    """
    Programmatic ETL entry point. on_progress(stage, detail) is an optional
    callback used by the server to report progress.
    """
    global _CURRENT_CONVOS, _CURRENT_PROJECTS, _CURRENT_USERS
    _CURRENT_CONVOS = conversations_path
    _CURRENT_PROJECTS = projects_path
    _CURRENT_USERS = users_path

    for p in (conversations_path, projects_path, users_path):
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    counts_out: dict[str, int] = {}
    try:
        init_schema(con)
        if on_progress:
            on_progress("users", "")
        load_users(con)
        if on_progress:
            on_progress("projects", "")
        n_p, n_d = load_projects(con)
        counts_out["projects"] = n_p
        counts_out["docs"] = n_d
        if on_progress:
            on_progress("conversations", "parsing")
        c = load_conversations(con)
        counts_out.update(c)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return counts_out


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    print(f"→ {DB_PATH}")
    t0 = time.time()
    counts = run_etl(CONVERSATIONS_JSON, PROJECTS_JSON, USERS_JSON, DB_PATH)
    for k, v in counts.items():
        print(f"  {k}={v}")
    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"done in {time.time() - t0:.1f}s — db size {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
