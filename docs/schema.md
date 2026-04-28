# Data Schema

Detailed schema for the three files from Claude.ai's "Export data" feature.

- [conversations.json](#conversationsjson) — all conversations
- [projects.json](#projectsjson) — "Projects" and uploaded documents
- [users.json](#usersjson) — user profile

---

## `conversations.json`

**Top level:** JSON array. Each element is a **Conversation** object. The file is
a single line (not pretty-printed).

### Conversation

| Field             | Type      | Description                                               |
| ----------------- | --------- | --------------------------------------------------------- |
| `uuid`            | string    | Conversation ID (matches the ID in Claude.ai URLs).       |
| `name`            | string    | Auto-generated title (e.g. "Tailwind CSS Input Styles Explained"). |
| `summary`         | string    | Always `""` — not populated in the export.                |
| `created_at`      | ISO-8601  | UTC, ends with `Z`. E.g. `2024-08-15T12:41:58.189064Z`.  |
| `updated_at`      | ISO-8601  | Last message or metadata update.                          |
| `account`         | object    | `{ "uuid": "<user_uuid>" }`. Same in every record.       |
| `chat_messages`   | array     | Message objects (see below).                              |

**Note:** There is **no foreign key** between Conversation and Project in this
export. Even if a conversation belongs to a project in the UI,
`conversations.json` does not contain a `project_uuid` field.

### Message (`chat_messages[*]`)

| Field                  | Type      | Description                                           |
| ---------------------- | --------- | ----------------------------------------------------- |
| `uuid`                 | string    | Message ID.                                           |
| `parent_message_uuid`  | string    | UUID of the previous message. Root messages use `00000000-0000-4000-8000-000000000000`. The chain is built from this field. |
| `sender`               | string    | `"human"` or `"assistant"`.                           |
| `created_at`           | ISO-8601  |                                                       |
| `updated_at`           | ISO-8601  |                                                       |
| `text`                 | string    | Legacy plain-text field. In newer messages it is populated alongside `content`; not reliable on its own. |
| `content`              | array     | **The actual message payload.** Array of content blocks. May be empty. |
| `attachments`          | array     | Pasted attachments (typically `paste.txt`).            |
| `files`                | array     | Uploaded file references (uuid + name).               |

The `human` and `assistant` sender counts are nearly equal — the assistant chain
sometimes produces multiple blocks (thinking + text) but these count as a single
"message."

### Content block types (`chat_messages[*].content[*]`)

Every block has common fields: `start_timestamp`, `stop_timestamp`, `flags`,
`type`. Type-specific fields vary.

Content block types observed in a typical export:

| `type`          | Where                                   |
| --------------- | --------------------------------------- |
| `text`          | Both human and assistant (most common)  |
| `tool_use`      | Assistant only                          |
| `tool_result`   | Assistant only (tool call response)     |
| `thinking`      | Assistant only (extended thinking)      |
| `token_budget`  | System / telemetry                      |
| `voice_note`    | Human voice note                        |

#### `text` block

```json
{
  "type": "text",
  "start_timestamp": "2024-08-15T12:42:05.206517Z",
  "stop_timestamp":  "2024-08-15T12:42:05.206517Z",
  "flags": null,
  "text": "...",
  "citations": []
}
```

`citations` is usually empty; it may be populated in assistant responses after
web_search/web_fetch (citation sources).

#### `thinking` block

Extended thinking output. Contains both raw text and intermediate summaries.

| Field                        | Notes                                         |
| ---------------------------- | --------------------------------------------- |
| `thinking`                   | Full thinking text.                           |
| `summaries`                  | `[{"summary": "..."}]` array; short titles shown in the UI. |
| `cut_off`, `truncated`       | Boolean flags.                                |
| `signature`                  | Verifiable-thinking signature; usually `null`. |
| `alternative_display_type`   | Optional rendering hint.                      |

#### `tool_use` block

A tool invoked by the assistant (both built-in and MCP).

| Field                 | Notes                                                   |
| --------------------- | ------------------------------------------------------- |
| `id`                  | Tool-use ID; `tool_result.tool_use_id` links to this.   |
| `name`                | Tool name. For MCP tools: `"<Server>:<tool>"` format.   |
| `input`               | Argument object (free-form JSON).                       |
| `integration_name`    | MCP / integration name.                                 |
| `integration_icon_url`| Icon URL.                                               |
| `mcp_server_url`      | MCP server URL (if applicable).                         |
| `is_mcp_app`          | Boolean.                                                |
| `approval_key`, `approval_options`, `context` | User approval dialog metadata.    |
| `display_content`     | UI rendering preference.                                |
| `message`             | Label shown to the user.                                |

Typical exports contain many distinct tool names. The most frequently used
include: `artifacts`, `web_search`, `bash_tool`, `write_file`, `repl`, `view`,
`filesystem:write_file`, `str_replace`.

#### `tool_result` block

The response to a `tool_use`.

| Field                 | Notes                                                   |
| --------------------- | ------------------------------------------------------- |
| `tool_use_id`         | The related `tool_use.id`.                              |
| `name`                | Tool name.                                              |
| `content`             | Block array of `[{type:"text"|"image"..., text/..., uuid}]`. |
| `structured_content`  | Optional; structured tool result.                       |
| `is_error`            | Boolean.                                                |
| `meta`                | Free-form metadata.                                     |

#### `token_budget` block

Contains a `remaining` field (usually `null`); a placeholder for "budget
exhausted" warnings in the UI. Can be ignored for analysis purposes.

#### `voice_note` block

Voice note transcript.

| Field    | Notes                                 |
| -------- | ------------------------------------- |
| `title`  | Title (provided by the user).         |
| `text`   | Transcript text.                      |

### Additional structures

#### `attachments[*]` (pasted text)

```json
{
  "file_name": "paste.txt",
  "file_size": 7927,
  "file_type": "txt",
  "extracted_content": "..."
}
```

Content is embedded in the export. Common file types include `txt` (most
frequent), empty string, `pdf`, `docx`, and occasional `text/*` or
`application/json`.

#### `files[*]` (uploaded file)

```json
{
  "file_uuid": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "file_name": "example-image.jpeg"
}
```

**Reference only** — file content is not included in the export. There is
content loss for images, PDFs, and audio files.

---

## `projects.json`

**Top level:** JSON array. Each element is a **Project**.

### Project

| Field                  | Type      | Notes                                             |
| ---------------------- | --------- | ------------------------------------------------- |
| `uuid`                 | string    | Project ID.                                       |
| `name`                 | string    | Project name as shown in the UI.                  |
| `description`          | string    | Usually empty.                                    |
| `prompt_template`      | string    | The project's custom instructions.                |
| `is_private`           | boolean   | Most projects are private.                        |
| `is_starter_project`   | boolean   | Indicates Claude-provided starter projects.       |
| `created_at`           | ISO-8601  | Timezone-aware: `2025-04-03T14:37:35.165453+00:00`. |
| `updated_at`           | ISO-8601  |                                                   |
| `creator`              | object    | `{ "uuid", "full_name" }`.                        |
| `docs`                 | array     | Uploaded project knowledge documents.             |

### Doc (`docs[*]`)

| Field        | Notes                                          |
| ------------ | ---------------------------------------------- |
| `uuid`       | Doc ID.                                        |
| `filename`   | E.g. `example-doc.md`.                         |
| `content`    | Raw text content (fully embedded).             |
| `created_at` | ISO-8601.                                      |

### Conversations ↔ Projects relationship

There is **no direct link** in the export. To establish this relationship you
would need to either:
- (a) Pull additional data from the Claude.ai API, or
- (b) Use heuristic matching (e.g. overlap between the first prompt's content
  and project documents, or time/topic proximity).

---

## `users.json`

Single-element array:

```json
[{
  "uuid": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "full_name": "Your Name",
  "email_address": "user@example.com",
  "verified_phone_number": "+1234567890"
}]
```

`conversations[*].account.uuid` and `projects[*].creator.uuid` match this UUID.

---

## Known gaps / losses

1. **File contents** — `files[*]` contains only uuid + name; no image/PDF
   content.
2. **Project ↔ Conversation mapping** — explained above.
3. **Artifact contents** — `artifacts` tool_use entries exist but the resulting
   HTML/JSX content is embedded in `tool_result.content`; there is no full
   artifact versioning.
4. **`summary` field** — always empty.
5. **Model information** — which model produced a given message is not in the
   export.
6. **Empty conversations** — records with `chat_messages` length of 0 exist.
