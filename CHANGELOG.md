# Changelog

All notable changes to Anamnesis are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`import <target> --reconcile-export-*` derives the format too** — the
  post-import drift hook's `--reconcile-export-format` is now optional. The
  hook is always `only-left` (imported side = left), so the lagging adapter
  is always `--reconcile-export-against`; omit the format and it derives
  that adapter's canonical round-trip writer, erroring (before the import
  commits) if it has none. Explicit mismatch still runs with a `warning`.
  Audit gains `format_source` / `lagging_adapter` /
  `canonical_round_trip_format`. Brings the post-import hook to parity with
  the standalone `reconcile-export` (every reconcile-export surface now
  derives).

### Fixed
- **`import generic-mcp --reconcile-export-*` now runs the post-import drift
  hook.** The generic-mcp path early-returned before the shared hook, so the
  requested drift artifact was silently skipped. Routed through the same
  `run_post_import_reconcile_export` tail as every other adapter.
- **`reconcile-export` derives the round-trip format from the lagging
  adapter** (CLI `--format` and MCP `reconcile_export_bucket.format` are now
  optional). Omit it and the lagging side of the bucket (`only-left` → right,
  `only-right` → left) selects its canonical writer
  (`mem0`→`mem0-sqlite`, `letta`→`letta-sqlite`, `memos`→`memos-dir`); a
  lagging adapter with no round-trip target errors clearly. An explicit
  format that disagrees with the canonical one still runs but emits a
  `warning`. Responses/audit gain `format_source`, `lagging_adapter`, and
  `canonical_round_trip_format`. The adapter→format map now lives in
  `anamnesis-export::round_trip_format_for_adapter`, the single source of
  truth shared with `discover_adapters`.

### Infrastructure
- **Crates.io publish prep**: workspace `categories` (`command-line-utilities`,
  `database`, `development-tools`) + `keywords` (`memory`, `agent`, `mcp`,
  `rag`) added at `[workspace.package]`; every crate now inherits
  `readme.workspace`, `categories.workspace`, `keywords.workspace` so the
  full 22-crate workspace is publish-ready.
- **`.github/workflows/publish-crates.yml`**: manual `workflow_dispatch`
  publish ladder in topological order (core → store/embedder →
  search/extractor/importer → 14 adapters → mcp-server → cli), with a
  `dry_run` toggle and 30 s sleep between real uploads. Requires a
  `CARGO_REGISTRY_TOKEN` repo secret before the first real run.

## [0.1.0] — 2026-05-18 — First stable adapter-contract release

The "every adapter is contract-validated" milestone. With 14 first-class
adapters all passing the shared `MemoryAdapter` invariant suite — plus
the host-client on-ramp (`anamnesis mcp config`) and the per-source
health surface (`doctor`) wired through both CLI and MCP — Anamnesis
crosses from "useful in principle" to "safe to wire into a Claude
Desktop / Cursor / ghast / Continue / Windsurf install." That is what
0.1.0 marks.

### Added — Client on-ramp & ops

- **`anamnesis mcp config [--name --transport --sse-port --token-env --binary]`**
  emits the standard `mcpServers` JSON every host client consumes.
  Defaults to stdio + the absolute path of the running binary so GUI
  hosts that don't inherit shell PATH find the binary. SSE mode emits
  `${env:NAME}` placeholders so tokens never land in any config file.
  (#63)
- **`docs/INTEGRATIONS.md`** — copy-pasteable setup walk-through for
  Claude Desktop, Cursor, ghast, Continue, Windsurf, and SSE mode. (#63)
- **`docs/demo/quickstart.sh`** — POSIX E2E smoke test. `init` →
  `mcp config` → stdio handshake (`initialize` + `tools/list`) →
  5-tool catalogue assertion (`search_memories`, `get_record`,
  `list_sources`, `trace_provenance`, `doctor`). (#63)
- **`doctor` MCP tool** — per-source health + staleness probe over
  JSON-RPC, complementing the existing `anamnesis doctor` CLI. (#62)

### Added — Quality / coverage

- **`AdapterContract` coverage for every first-class adapter.** Before
  0.1.0 only 3 adapters (claude-code, codex, mem0) ran the shared
  `MemoryAdapter` contract suite. After: all 14 do — 22 new tests
  covering descriptor stability, scan idempotency, native_id presence,
  pure normalize, schema_version correctness, raw_hash non-triviality,
  instance → RecordId propagation, and health-message contract. (#64)

### Notes

- No breaking API changes from 0.0.2.
- No new runtime dependencies; all new tests use existing dev-deps.

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
