# Claude Conversation Analyzer

Analyze your Claude.ai conversation exports — entirely in your browser.
No server, no data upload. Everything runs locally via [Pyodide](https://pyodide.org/) + [DuckDB-WASM](https://duckdb.org/docs/api/wasm).

**[Live Demo →](https://claude-conversation-analyzer.pages.dev)**

## How it works

1. Export your data from Claude.ai (Settings → Account → Export Data)
2. Open the app and drop the ZIP file
3. Browse 13 analysis modules covering conversation patterns, tool usage, thinking blocks, and more

Your data never leaves your device. All processing happens in a Web Worker using Python (Pyodide) and DuckDB compiled to WebAssembly.

## Features

- **13 analysis modules** — conversation length, message length, thinking usage, activity rhythm, tool intensity, conversation lifetime, attachments, projects, artifacts, code blocks, citations, tool errors, interventions
- **Bilingual UI** — English and Turkish
- **System / light / dark theme**
- **Downloadable results** — CSV, JSON, Markdown reports
- **Zero server dependency** — deployable as static files

## Analysis modules

| Module | Description |
|--------|-------------|
| m01 | Conversation length distribution |
| m02 | Message length (human vs assistant) |
| m03 | Thinking block usage |
| m04 | Activity rhythm (hour / day / week) |
| m05 | Tool-call intensity |
| m06 | Conversation lifetime |
| m07 | Attachments & files |
| m08 | Projects & knowledge docs |
| m09 | Artifact production profile |
| m10 | Code block profile |
| m11 | Citations & source profile |
| m12 | Tool error typology |
| m13 | Conversation interventions |

## Architecture

```
Browser
├── React UI (Vite build)
├── Web Worker
│   ├── Pyodide (Python 3.12 in WebAssembly)
│   ├── DuckDB (SQL engine in WebAssembly)
│   └── 13 Python analysis modules
└── fflate (ZIP decompression)
```

The Python analysis scripts (`scripts/analysis/`) are bundled into the Web Worker at build time via Vite's raw import. On first visit, Pyodide (~10 MB) and DuckDB (~5 MB) are loaded from CDN and cached by the browser.

## Local development

```sh
cd frontend
npm install
npm run dev
```

This starts a Vite dev server with hot reload. No Python or backend setup needed.

To build for production:

```sh
cd frontend
npm run build     # outputs to frontend/dist/
npx serve dist -s # test locally
```

## Deploy

The `frontend/dist/` directory is a fully self-contained static site. Deploy to any static host:

**Cloudflare Pages:**
- Build command: `cd frontend && npm install && npm run build`
- Output directory: `frontend/dist`

**Any static host:** Upload the contents of `frontend/dist/`.

## Docs

- [Data schema](docs/schema.md) — structure of the Claude.ai export files
- [ETL pipeline](docs/etl.md) — how raw JSON is transformed into DuckDB tables

## License

[MIT](LICENSE)
