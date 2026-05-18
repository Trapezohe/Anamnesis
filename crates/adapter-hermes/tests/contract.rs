//! `hermes` adapter satisfies the shared `anamnesis_core::contract`
//! invariants. Mirrors the pattern from `adapter-codex` / `adapter-mem0`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_hermes::{hermes_adapter, HermesAdapter};
use anamnesis_core::contract::AdapterContract;
use rusqlite::Connection;

static NONCE: AtomicU64 = AtomicU64::new(0);

fn fixture_data_dir() -> PathBuf {
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("anamnesis-hermes-contract-{pid}-{n}"));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("MEMORY.md"), "system on macOS").unwrap();
    fs::write(dir.join("USER.md"), "prefers Rust").unwrap();
    let conn = Connection::open(dir.join("sessions.db")).unwrap();
    conn.execute_batch(
        r#"CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            role TEXT,
            content TEXT NOT NULL,
            created_at TEXT
        );
        INSERT INTO messages VALUES
          ('m1', 'user',      'hi',         '2024-01-01T00:00:00Z'),
          ('m2', 'assistant', 'hello back', '2026-04-01T00:00:00Z');"#,
    )
    .unwrap();
    dir
}

#[tokio::test]
async fn hermes_satisfies_adapter_contract() {
    let root = fixture_data_dir();
    let contract = AdapterContract::new(move || -> HermesAdapter {
        hermes_adapter(root.clone(), Some("default"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn hermes_no_instance_satisfies_contract() {
    let root = fixture_data_dir();
    let contract =
        AdapterContract::new(move || -> HermesAdapter { hermes_adapter(root.clone(), None) });
    contract.run_all().await;
}
