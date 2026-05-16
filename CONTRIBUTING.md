# Contributing to Anamnesis

Most user value comes from new **adapters** — the per-agent connectors that turn a third-party memory store into Anamnesis records. This guide walks through adding one in 5 steps.

The fastest read is [the existing adapters](crates/) under `crates/adapter-*`. They all follow the same shape.

---

## Ground rules

1. **User sovereignty first.** Anamnesis never sends user memory data anywhere by default. No telemetry. No phone-home.
2. **Local-first.** All persistence lives in the user's own SQLite. Cloud is opt-in only.
3. **Read-only on third-party data.** Adapters import; they never write back. The source-of-truth stays in the original tool.
4. **Anamnesis runs its own RAG.** Source-system vectors stay in `raw_artifacts.source_embedding` for provenance. They never enter the retrieval path. (See `docs/BLUEPRINT.md §6.6.1`.)
5. **Detect ≠ Import.** `detect()` only reads metadata (paths, schemas, counts). Reading content happens during `import` and only after the user opts in.
6. **Adapters never write to the store.** They produce `RawRecord`s; the `importer` crate writes.
7. **Adapters never own the chunker or embedder.** The pipeline handles them.
8. **Every adapter passes `AdapterContract::run_all()`** (7 invariants).

---

## Getting started

```bash
git clone https://github.com/Trapezohe/Anamnesis
cd Anamnesis

cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check

# Skip the heavy fastembed ONNX runtime first compile during iteration:
cargo test --workspace --no-default-features
```

Required: Rust 1.85+ (managed by `rust-toolchain.toml`).

---

## 5-step recipe for a new adapter

Pick a short adapter id, e.g. `obsidian`. Files below use that placeholder.

### 1. Scaffold the crate

```bash
mkdir -p crates/adapter-obsidian/{src,tests}
```

`crates/adapter-obsidian/Cargo.toml`:
```toml
[package]
name = "anamnesis-adapter-obsidian"
description = "Anamnesis adapter for Obsidian vaults"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors.workspace = true
repository.workspace = true

[dependencies]
anamnesis-core = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
chrono = { workspace = true }
async-trait = { workspace = true }
futures = { workspace = true }
tracing = { workspace = true }
blake3 = { workspace = true }
# Add adapter-specific deps here (e.g. `pulldown-cmark`, `notify`)

[dev-dependencies]
tokio = { workspace = true }
```

Add to workspace root `Cargo.toml`:
```toml
[workspace]
members = [
    # ... existing crates ...
    "crates/adapter-obsidian",
]

[workspace.dependencies]
# ... existing entries ...
anamnesis-adapter-obsidian = { path = "crates/adapter-obsidian", version = "0.0.1" }
```

### 2. Implement the detector

`src/detector.rs`:
```rust
use std::path::PathBuf;
use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

pub struct ObsidianDetector {
    pub override_root: Option<PathBuf>,
}

#[async_trait]
impl SourceDetector for ObsidianDetector {
    fn adapter_id(&self) -> &'static str { crate::ADAPTER_ID }

    async fn detect(&self, opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let root = resolve_root(self.override_root.as_deref(), opts);
        if !root.exists() {
            return Ok(Vec::new());     // missing → no DetectedSource at all
        }
        // metadata-only probe: count vault folders, files, etc. NEVER read content.
        let count = count_md_files(&root)?;
        Ok(vec![DetectedSource {
            adapter: crate::ADAPTER_ID.into(),
            instance: Some("default".into()),
            location: root.display().to_string(),
            local_path: Some(root),
            confidence: if count > 0 { Confidence::High } else { Confidence::Medium },
            estimated_records: Some(count),
            note: Some(format!("{count} note(s) found")),
        }])
    }
}
```

The detector **must** honour `DetectOpts.home_override` so contract tests can fixture-stub `$HOME` without touching the real user directory.

### 3. Implement the scanner + normalizer

`src/scanner.rs`:
- Walks the filesystem (or queries the DB / API).
- Returns a list of file paths (or rows / API IDs).
- Reads metadata only — content reads happen in `normalize`.

`src/normalizer.rs`:
- `raw_*()` helpers build `RawRecord { native_id, native_path, payload, captured_at }`.
- `normalize(raw, instance)` produces one or more `AnamnesisRecord`s. **Must be pure** — same input → same output.

Mapping decisions:
- `Kind`: pick from `Fact | Preference | Feedback | Reference | Episode | Skill | Unknown`. If the source has typed memory (like Claude Code's frontmatter), map; otherwise default to `Episode` for conversation-shaped data, `Fact` for structured rows.
- `Scope`: `User | Project | Session | Ephemeral`. Default to `User` for stable facts, `Session` for episodes.
- `provenance.raw_hash = blake3(raw payload)`.
- `provenance.native_id` should be deterministic across runs — `format!("{instance}|{kind}|{path}")` is the typical pattern.

### 4. Wire the `MemoryAdapter` impl

`src/lib.rs`:
```rust
pub const ADAPTER_ID: &str = "obsidian";

pub struct ObsidianAdapter { config: Arc<ObsidianConfig> }

#[async_trait]
impl MemoryAdapter for ObsidianAdapter {
    fn descriptor(&self) -> SourceDescriptor { ... }
    fn scan<'a>(&'a self, _: ScanOpts) -> BoxStream<'a, Result<RawRecord>> { ... }
    fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>> { ... }
    async fn health(&self) -> HealthStatus { ... }
}

pub fn obsidian_adapter(root: impl Into<PathBuf>, instance: Option<&str>) -> ObsidianAdapter { ... }
```

### 5. Run the contract test

`tests/contract.rs`:
```rust
use anamnesis_adapter_obsidian::{obsidian_adapter, ObsidianAdapter};
use anamnesis_core::contract::AdapterContract;

#[tokio::test]
async fn obsidian_satisfies_adapter_contract() {
    let root = build_fixture_vault();
    let contract = AdapterContract::new(move || -> ObsidianAdapter {
        obsidian_adapter(root.clone(), Some("default"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn obsidian_no_instance_satisfies_contract() {
    let root = build_fixture_vault();
    let contract = AdapterContract::new(move || -> ObsidianAdapter {
        obsidian_adapter(root.clone(), None)
    });
    contract.run_all().await;
}
```

The harness asserts 7 invariants automatically:
1. `descriptor()` is stable across calls.
2. Two independent scans yield the same `(native_id, path)` sets.
3. All raw records have non-empty `native_id`.
4. `normalize()` is pure.
5. Every record has `schema_version == SCHEMA_VERSION`.
6. `provenance.raw_hash` is populated and non-trivial.
7. Different `instance` values produce different `RecordId`s (the "SQLite-NULL-UNIQUE pitfall" guard).

If your adapter trips any of these, fix the adapter — never the contract.

### 6. Wire into the CLI + MCP server

In `crates/cli/Cargo.toml` add `anamnesis-adapter-obsidian = { workspace = true }`.
In `crates/cli/src/main.rs`:
- Add `ObsidianDetector` to the `Discovery` registration in `cmd_discover`.
- Add a `anamnesis_adapter_obsidian::ADAPTER_ID => { ... }` arm to `cmd_import`.

Do the same in `crates/mcp-server/Cargo.toml` and `crates/mcp-server/src/server.rs::tool_import_source`.

---

## Tempdir tests: avoid the nanosecond race

If your tests use timestamps for unique tempdir paths, you'll eventually hit a race where two parallel tests get the same nanosecond and step on each other. Use a per-binary atomic counter instead:

```rust
static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn tmp_dir() -> PathBuf {
    let n = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id();
    let p = std::env::temp_dir().join(format!("my-adapter-{pid}-{n}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}
```

---

## Schema changes

Changes to `AnamnesisRecord` need an ADR (one short doc in `docs/adr/`) and a migration in `crates/store/src/migrations/`. Minor schema changes must preserve read-compatibility with the previous version.

## Style

- Run `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings` before pushing.
- Default to **no comments**. Add one when the *why* is non-obvious (an invariant, a workaround for a specific bug, behaviour that would surprise a reader). Don't explain *what* the code does — well-named identifiers already do that.
- No `unwrap()` outside tests and one-shot CLI paths — return `Result` everywhere else.
- Public items in `core` need at least a one-line doc comment.

## Commits

- Conventional commits (`feat(adapter-obsidian):`, `fix:`, `docs:`, `refactor:`, `chore:`).
- One commit per logical change. If clippy yells at you mid-PR, fix in a follow-up `fix(scope): satisfy clippy …` commit rather than rewriting history.
- Reference issues when relevant.

## Security

If you find a vulnerability, **do not** open a public issue. Email the maintainer (see repo profile) or use GitHub Security Advisories.

## License

By contributing, you agree your contributions are licensed under [Apache 2.0](LICENSE).
