# Contributing to Anamnesis

Thanks for considering a contribution! Anamnesis is early-stage — the easiest way to make a real impact is to add or improve an **adapter** for a memory source we don't yet support.

## Ground rules

1. **User sovereignty first.** Anamnesis never sends user memory data anywhere by default. No telemetry. No phone-home.
2. **Local-first.** All persistence lives in the user's own SQLite. Cloud is opt-in only.
3. **Read-only on third-party data.** Adapters import; they never write back. The source-of-truth stays in the original tool.
4. **Provenance is sacred.** Every record must trace back to its source — `native_id`, `native_path`, `captured_at` are required.

## Getting started

```bash
git clone https://github.com/Trapezohe/Anamnesis
cd Anamnesis
cargo build
cargo test
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

Required: Rust 1.75+ (managed by `rust-toolchain.toml`).

## Writing a new adapter

The easiest way to help is to add support for another memory source. The pattern:

```
crates/adapter-<name>/
├── Cargo.toml             # depends only on anamnesis-core
├── src/
│   ├── lib.rs             # impl MemoryAdapter for YourAdapter
│   └── normalize.rs       # map raw payload → AnamnesisRecord
├── tests/contract.rs      # runs the shared adapter contract suite
└── fixtures/              # one anonymized sample per behavior
```

Every adapter must:
- Implement `anamnesis_core::adapter::MemoryAdapter`.
- Pass the shared contract test suite (idempotent imports, dedup, schema_version handling).
- Ship anonymized fixtures (never check in real user data).
- Document its config schema in its `lib.rs` doc comment.

Open an issue with `[adapter]` in the title before starting — happy to sketch the mapping with you.

## Schema changes

Changes to `AnamnesisRecord` need an ADR (one short doc in `docs/adr/`) and a migration in `crates/store/src/migrations/`. Minor schema changes must preserve read-compatibility with the previous version.

## Commit style

- Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
- Reference issues when relevant.
- One logical change per commit; PRs can stack commits but should tell a coherent story.

## Code style

- `cargo fmt` (settings in `rustfmt.toml`).
- `cargo clippy --workspace -- -D warnings` must be clean.
- No `unwrap()` outside tests and one-shot CLI paths — return `Result` everywhere else.
- Public items in `core` need at least a one-line doc comment.

## Security

If you find a vulnerability, **do not** open a public issue. Email the maintainer (see repo profile) or use GitHub Security Advisories.

## License

By contributing, you agree your contributions are licensed under [Apache 2.0](LICENSE).
