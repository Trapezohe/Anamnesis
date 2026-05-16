# Anamnesis · 搜魂术

> A user-sovereign memory layer that imports, unifies, and serves agent memory across Claude Code, Codex, mem0, Hermes, and any MCP-aware tool.

**Status:** 🚧 Early development (Phase 0 — scaffolding).

Your agent memories are valuable. Today they're locked inside each tool — switch from Claude Code to Cursor, and you start from zero. Anamnesis is a small, local-first binary that:

1. **Imports** memories from each agent (adapters for Claude Code, mem0, Codex, Hermes, …).
2. **Unifies** them under a single schema with full provenance.
3. **Serves** them back to *any* agent over [MCP](https://modelcontextprotocol.io), so what you taught one tool, you've taught them all.

No cloud. No telemetry. Apache 2.0.

## Three sentences

- Your memories are *yours* — Anamnesis is the bridge, not the owner.
- One binary, one local SQLite, one MCP endpoint — every agent reads from the same well.
- Adapters are pluggable; if your tool stores memories somewhere, it can join.

## Quick start (planned, not yet shipped)

```bash
# Install (Phase 3+)
brew install anamnesis            # or: cargo install anamnesis

# Initialize and add your first source
anamnesis init
anamnesis source add claude-code --path ~/.claude/projects

# Import
anamnesis import claude-code

# Search across all sources
anamnesis search "what does the user prefer for testing?"

# Or serve over MCP — every MCP-aware agent can read from you
anamnesis serve --stdio
```

## Architecture (one diagram)

```
┌──────────────────────────────────────────────────────────────┐
│  ghast │ Claude Code │ Cursor │ Zed │ CLI │ your scripts      │
└──────────┬──────────────────────────┬─────────────────────────┘
           │ MCP (stdio/SSE)          │ CLI
           ▼                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     anamnesis (Rust binary)                  │
│   mcp-server  /  cli  →  core  →  store (SQLite+FTS+vec)     │
│                       ↑                                       │
│                  adapter-*                                    │
└──────────┬────────────┬────────────┬───────────┬─────────────┘
           ▼            ▼            ▼           ▼
      claude-code     mem0        codex       hermes …
```

See [docs/BLUEPRINT.md](docs/BLUEPRINT.md) for the full design.

## Repository layout

```
anamnesis/
├── crates/
│   ├── core/                   # Domain types & traits, no IO
│   ├── store/                  # SQLite + FTS5 + sqlite-vec
│   ├── cli/                    # `anamnesis` CLI
│   ├── mcp-server/             # MCP server (stdio + SSE)
│   ├── adapter-claude-code/    # Adapter: Claude Code
│   └── adapter-mem0/           # Adapter: mem0
├── docs/
│   └── BLUEPRINT.md            # Full design document
└── .github/workflows/          # CI
```

## Development

```bash
git clone https://github.com/Trapezohe/Anamnesis
cd Anamnesis
cargo build
cargo test
cargo clippy --workspace -- -D warnings
```

Rust 1.75+ required (managed via `rust-toolchain.toml`).

## Roadmap

- **Phase 0 (now):** Repo scaffolding, schema lock, CI.
- **Phase 1:** Core engine + Claude Code adapter + working CLI.
- **Phase 2:** mem0 adapter + MCP server.
- **Phase 3:** ghast integration + first public release (0.1.0).
- **Phase 4+:** Codex, Hermes, generic MCP adapter, reverse-serve as MCP memory provider.

Detailed plan: [docs/BLUEPRINT.md](docs/BLUEPRINT.md).

## Contributing

We welcome adapter contributions especially — adding a new memory source is the most leveraged way to help. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE). Your memory data is *not* covered by this license — it remains yours, always.

---

**Anamnesis** (n.) — the Platonic concept of recollection: the soul remembering knowledge it already possesses, drawn back from forgetting.
