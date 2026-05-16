# Anamnesis · 搜魂术

> A user-sovereign memory layer that imports, unifies, and serves agent memory across Claude Code, Codex, mem0, and any MCP-aware tool.

**Status:** 🚧 Active development (Phase 2 — pre-release). Three adapters, hybrid local RAG, MCP server, all working end-to-end.

Your agent memories are valuable. Today they're locked inside each tool — switch from Claude Code to Cursor, and you start from zero. Anamnesis is a small, local-first binary that:

1. **Imports** memories from each agent (adapters for Claude Code, Codex, mem0; more coming).
2. **Unifies** them under a single normalized schema with full provenance.
3. **Serves** them back to *any* agent over [MCP](https://modelcontextprotocol.io), so what you taught one tool, you've taught them all.

No cloud. No telemetry. All RAG runs locally with your own embeddings.

## Three sentences

- Your memories are *yours* — Anamnesis is the bridge, not the owner.
- One binary, one local SQLite, one MCP endpoint — every agent reads from the same well.
- The RAG stack (chunker, embedder, FTS5, vector search, rerank) is **Anamnesis-owned**; source-system vectors stay as provenance metadata and never enter retrieval.

## Quick start

```bash
# Build (cargo install once 0.1.0 ships)
git clone https://github.com/Trapezohe/Anamnesis
cd Anamnesis
cargo install --path crates/cli

# Initialize: creates DB + pins default embedding model
anamnesis init

# Discover known memory sources at default locations
anamnesis discover

# Register and import (or just `anamnesis import claude-code`)
anamnesis source add claude-code --path ~/.claude/projects
anamnesis import claude-code

# Search across all imported records (Hybrid: FTS5 + vector + RRF)
anamnesis search "what does the user prefer for testing?"

# Or expose to any MCP-aware agent over stdio
anamnesis serve
```

## What's in 0.1.0 (current `main`)

### Adapters
- **claude-code** — `~/.claude/projects/*/memory/*.md` (typed by frontmatter) + `*.jsonl` sessions
- **mem0** — self-hosted SQLite at `~/.mem0/db.sqlite` (schema-flexible; tolerates mem0 version drift)
- **codex** — OpenAI Codex CLI session files under `~/.codex/`

Every adapter goes through the same `MemoryAdapter` contract: `detector → scanner → parser → normalizer → AnamnesisRecord`. Source vectors stay in `raw_artifacts.source_embedding` as provenance only.

### Storage + RAG (all Anamnesis-owned, local)
- SQLite + FTS5 + sqlite-vec (BLOB fallback in current build)
- Chunker: script-aware token estimation (CJK + Latin), paragraph → sentence → word boundary descent
- Embedding: `fastembed-rs` local provider with 4 built-in models (multilingual-e5-small default, plus tiny / english / multi-strong)
- Optional: Voyage AI cloud embedding (`--features cloud-voyage`, opt-in)
- Hybrid retrieval: FTS5 BM25 + vector kNN merged via Reciprocal Rank Fusion (K=60)
- `ContextPacker` aggregates chunks → records, applies source-diversity caps, bounds token budget

### MCP server
Wire-compatible JSON-RPC 2.0 over stdio. Exposes:
- **5 tools**: `search_memories`, `get_record`, `list_sources`, `import_source`, `trace_provenance`
- **3 resources**: `anamnesis://record/{id}`, `anamnesis://source/{adapter}`, `anamnesis://timeline/{YYYY-MM-DD}`
- **2 prompts**: `summarize_my_preferences`, `find_related`

Wire it into Claude Desktop / ghast / any MCP-aware tool by pointing them at `anamnesis-mcp` (stdio).

### CLI (all working today)
```
anamnesis init [--model KEY]
anamnesis status [--json]
anamnesis discover
anamnesis source add/list/remove
anamnesis import <adapter>[:instance] [--full] [--dry-run] [--no-embed] [--path PATH]
anamnesis search <query> [--source X] [--kind K] [--scope S] [--limit N] [--mode hybrid|fulltext|vector] [--json]
anamnesis export [--format jsonl|csv] [--out FILE] [--source X]
anamnesis verify [--repair]
anamnesis serve
anamnesis model list/use/install/rebuild
anamnesis migrate
```

### Other niceties
- Config file at `$XDG_CONFIG_HOME/anamnesis/config.toml` (override with `--config PATH` or `ANAMNESIS_CONFIG`)
- Append-only audit log at `$DATA_DIR/audit.log` (one JSONL per import/search/export/serve)
- Adapter contract test framework (`anamnesis_core::contract::AdapterContract`) — every adapter passes 7 shared invariants

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  ghast │ Claude Code │ Cursor │ Zed │ CLI │ your scripts     │
└──────────┬──────────────────────────┬─────────────────────────┘
           │ MCP (stdio)              │ CLI
           ▼                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     anamnesis (Rust binary)                  │
│   mcp-server  /  cli                                         │
│       ↓                                                       │
│    search / pack ← Hybrid RAG (FTS5 + vec + RRF)             │
│       ↓                          ↑                            │
│    store ←─── importer ←─── adapters/*  ←─── embedder        │
│   (SQLite                                  (fastembed / voyage)│
│   + FTS5)                                                     │
└──────────┬────────────┬────────────┬───────────┬─────────────┘
           ▼            ▼            ▼           ▼
      claude-code     mem0        codex      hermes (TODO)
```

See [docs/BLUEPRINT.md](docs/BLUEPRINT.md) for the full design and decisions.

## Repository layout

```
anamnesis/
├── crates/
│   ├── core/                   # Domain types, traits, contract harness
│   ├── store/                  # SQLite + FTS5 + migrations + typed API
│   ├── embedder/               # EmbeddingProvider trait + local + voyage
│   ├── importer/               # scan → normalize → chunk → upsert pipeline
│   ├── search/                 # Hybrid RAG + ContextPacker
│   ├── cli/                    # `anamnesis` CLI binary
│   ├── mcp-server/             # `anamnesis-mcp` MCP stdio server
│   ├── adapter-claude-code/    # Adapter: Claude Code
│   ├── adapter-mem0/           # Adapter: mem0 SQLite
│   └── adapter-codex/          # Adapter: OpenAI Codex
├── docs/
│   └── BLUEPRINT.md            # Full design document (with decisions §16)
├── CHANGELOG.md
├── CONTRIBUTING.md             # How to add a new adapter
└── .github/workflows/          # CI
```

## Development

```bash
git clone https://github.com/Trapezohe/Anamnesis
cd Anamnesis

cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings

# Without the heavy fastembed dep (fast iteration):
cargo test --workspace --no-default-features
```

Rust 1.85+ (managed via `rust-toolchain.toml`).

The `local-fastembed` feature is on by default in `anamnesis-cli` and `anamnesis-mcp-server`; turn it off (`--no-default-features`) when iterating on non-embedding code to avoid the ~5-minute ONNX runtime first-compile.

## Roadmap

- **Phase 0 ✅** — Repo scaffolding, schema lock, CI.
- **Phase 1 ✅** — Core engine + Claude Code adapter + working CLI + local RAG.
- **Phase 2 ✅** — mem0 adapter + Codex adapter + MCP server + cross-source unified search.
- **Phase 3 (now)** — Polish: MCP SSE transport, generic MCP adapter, Hermes adapter, ghast integration, Homebrew + cargo install release packaging.
- **Phase 4+** — Reverse mode (Anamnesis as MCP memory provider to other agents), embedding profile switcher, telemetry-opt-in, RFC for Agent Memory Interchange Format.

Detailed plan: [docs/BLUEPRINT.md](docs/BLUEPRINT.md).

## Contributing

We welcome adapter contributions especially — adding a new memory source is the most leveraged way to help. See [CONTRIBUTING.md](CONTRIBUTING.md) for the 5-step recipe.

## License

[Apache License 2.0](LICENSE). Your memory data is *not* covered by this license — it remains yours, always.

---

**Anamnesis** (n.) — the Platonic concept of recollection: the soul remembering knowledge it already possesses, drawn back from forgetting.
