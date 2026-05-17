# Changelog

All notable changes to Anamnesis are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

_Nothing yet — open the next iteration here._

## [0.0.2] — 2026-05-18 — Phase 2 → 3 transition

First **tagged release with pre-built binaries** for macOS / Linux /
Windows (see [Releases](https://github.com/Trapezohe/Anamnesis/releases/tag/v0.0.2)).

### Added — North-Star iteration (§-1.5 PR-1 through PR-5)

- **§-1.5 PR-5 — Full filter surface on MCP / CLI search.** `tools/list`
  schema for `search_memories` now advertises `instance`, `kind` (enum),
  `scope` (enum), `since` / `until` (RFC3339). Handler reads them into
  `SearchFilter.{instance,time_from,time_to}` (previously hardcoded to
  `None`). CLI `anamnesis search --instance / --since / --until` flags.
  (#30)
- **§-1.5 PR-2 — Cursor pagination for `resources/list`.** Store
  exposes `list_record_ids_paged(cursor, limit)` with stable
  lexicographic ordering and `MAX_LIST_LIMIT = 1000`. MCP returns
  `nextCursor`. `generic-mcp` adapter follows the cursor up to
  `MAX_LIST_PAGES = 1000`. End-to-end 250-record migration test
  asserts no records lost. (#29)
- **§-1.5 PR-4b — Remaining adapters honor `ScanOpts`.** codex
  (file-mtime filter, true streaming); mem0 (Rust-side row filter on
  `updated_at ?? created_at` parsing both RFC3339 and stringified
  epoch); generic-mcp (one-shot warning until PR-2 upstream
  timestamps exist). (#28)
- **§-1.5 PR-4a — `ScanOpts` pushdown end-to-end.** `--since` /
  `--full` CLI flags reach `ImportRunner` and `adapter.scan(opts)`;
  CLI auto-fills `since` from `sources.last_import_at`. claude-code
  adapter does true streaming + per-file mtime filter. New
  `ImportRunner::run_with_opts` API. (#27)
- **§-1.5 PR-3 — Unified `ImportService`.** CLI and MCP admin import
  produce identical system-state deltas (registry, `last_import_at`,
  `audit.log`). MCP `import_source` rejects `path` / `url` args
  (clients must register sources via CLI `source add` first). (#26)
- **§-1.5 PR-1 — `generic-mcp` is a first-class CLI source.**
  `anamnesis source add generic-mcp --url <upstream> --token-env
  <ENV>` + `anamnesis import generic-mcp:<instance>`. Token name lives
  in registry; value resolved from operator's env at import time.
  (#25)
- **§-2.6 Demo path.** End-to-end loopback migration demonstrated:
  Anamnesis instance A → MCP HTTP → generic-mcp adapter → Anamnesis
  instance B, with provenance preserved across the boundary.

### Added — Release Engineering (Phase 3 prep)

- **Pre-built release binaries** for macOS (aarch64 + x86_64), Linux
  (x86_64 + aarch64), Windows (x86_64) — triggered by `v*` tag push.
  Each release ships `anamnesis-<version>-<target>.{tar.gz,zip}` with
  `anamnesis` + `anamnesis-mcp` binaries, README, LICENSE, and a
  SHA-256 sidecar. (See `.github/workflows/release.yml`.)
- README install path: pre-built binary install instructions
  alongside `cargo install --path`.

### Added

#### CLI
- `anamnesis serve` — embeds the MCP stdio server in the CLI process (no separate binary needed).
- `anamnesis export [--format jsonl|csv] [--out FILE] [--source X]` — full record export with RFC4180 CSV quoting.
- `anamnesis verify [--repair]` — SQLite `PRAGMA integrity_check`, orphan-record detection, FTS index drift detection, missing-embedding counters; `--repair` rebuilds FTS in place and re-queues embedding jobs.
- `anamnesis status --json` — structured output for scripts and dashboards.
- `anamnesis search --kind K --scope S` — metadata filters (in addition to existing `--source`).
- `anamnesis migrate` — re-opens the store (idempotent).
- Global flag `--config PATH` (or `ANAMNESIS_CONFIG` env) overrides the TOML config location.

#### Config
- TOML config file at `$XDG_CONFIG_HOME/anamnesis/config.toml` (macOS uses Application Support).
- `[embedding]` (model, provider, batch_size, cache_dir), `[server]` (allowed_clients, require_token), `[[sources]]` blocks.
- Precedence: CLI flag > config file > defaults. Every section/field has a serde default so older configs keep parsing.

#### Adapters
- **mem0** — self-hosted SQLite adapter. Schema-flexible (PRAGMA table_info introspection tolerates mem0 version drift). Required cols: `id`, `memory`. Optional cols: `user_id`, `agent_id`, `run_id`, `metadata` (JSON), `created_at`, `updated_at`. Unknown cols captured to `extra` for provenance.
- **codex** — OpenAI Codex CLI adapter. Permissive walker — every `.json` and `.jsonl` under `~/.codex/` becomes one Kind::Episode record.

#### MCP server
- **2 prompts**: `summarize_my_preferences` (renders top user-scope records for LLM summarization) and `find_related` (Hybrid-searches a free-text description, returns top-N as a prompt template).
- Resource handlers for `anamnesis://record/{id}`, `anamnesis://source/{adapter}[:instance]`, `anamnesis://timeline/{YYYY-MM-DD}`.

#### Embedder
- **Voyage cloud provider** behind `cloud-voyage` feature flag. Reads `VOYAGE_API_KEY`. voyage-3 model (1024-dim). Asymmetric task handling (query vs document `input_type`). Never invoked without explicit opt-in.

#### Audit
- Append-only audit log at `$DATA_DIR/audit.log` (JSONL, one entry per import/search/export/serve). Best-effort writes — never blocks the user's command if disk fails.

#### Tests
- Adapter contract test framework `anamnesis_core::contract::AdapterContract` — 7 invariants every adapter passes.
- Cross-source E2E (`crates/cli/tests/e2e_cross_source.rs`) proving claude-code + mem0 hits surface in one unified result set.
- Stdio E2E for the MCP binary (spawns subprocess, exchanges real JSON-RPC frames).

### Changed
- `HybridSearcher<P>` bound relaxed to `P: EmbeddingProvider + ?Sized` so the MCP server can pass `Box<dyn EmbeddingProvider>`.
- MSRV bumped from 1.75 → 1.85 to accommodate `fastembed-rs`'s dep tree.
- `mcp-server` switched from `std::sync::Mutex` to `tokio::sync::Mutex` (await-holding-lock fix).

### Fixed
- `tmp_dir()` helpers in mem0 and codex adapter tests now use an atomic counter + pid instead of a nanosecond timestamp (parallel-test race).
- mem0 detector tolerates SQLite files that exist but lack a `memories` table (Confidence::Low instead of Err).

---

## [0.0.1] — Phase 0 / Phase 1 (initial public scaffold)

### Added — Phase 0
- Cargo workspace skeleton, schema v1 (`AnamnesisRecord`), CI (fmt + clippy + test), Apache 2.0 LICENSE.

### Added — Phase 1
- **Core types**: `AnamnesisRecord`, `RecordId` (blake3-derived), `Kind`, `Scope`, `Provenance`, `Embedding`.
- **Store**: SQLite with migrations 0001 (initial) + 0002 (chunks/embeddings/jobs/sources/raw_artifacts).
- **EmbeddingProvider** trait + `EmbeddingTask` for asymmetric models + `ModelId`.
- **Curated 5-model registry**: `default` (multilingual-e5-small, 120MB), `tiny` (MiniLM-L6-v2 quantized, 90MB), `en` (BGE-small-EN, 130MB), `multi-strong` (multilingual-e5-base, 280MB), `cloud-voyage` (cloud).
- **Local fastembed provider** behind `local-fastembed` feature.
- **Chunker** — script-aware token estimation (CJK + Latin), boundary descent (paragraph → line → sentence → word → char).
- **Discovery / SourceDetector** trait + `Discovery` orchestrator.
- **claude-code adapter** — `memory/*.md` frontmatter typing + `*.jsonl` sessions.
- **Importer** — scan → normalize → chunk → upsert pipeline.
- **EmbeddingWorker** — drains `embedding_jobs`.
- **Hybrid search** — FTS5 BM25 + vector kNN + RRF (K=60).
- **ContextPacker** — record aggregation + provenance + diversity cap + token budget.
- **CLI** — init, status, discover, source, import, search, model commands.
- **MCP server** — stdio JSON-RPC, 5 tools, 3 resources.
- **Adapter contract test framework** — `anamnesis_core::contract::AdapterContract`.
- **E2E tests** — fixture-driven for the full discover → import → search loop.

[Unreleased]: https://github.com/Trapezohe/Anamnesis/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/Trapezohe/Anamnesis/releases/tag/v0.0.2
[0.0.1]: https://github.com/Trapezohe/Anamnesis/releases/tag/v0.0.1
