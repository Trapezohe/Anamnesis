//! `letta` adapter satisfies the shared `anamnesis_core::contract`
//! invariants. Mirrors the pattern from `adapter-codex` / `adapter-mem0`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_letta::{letta_adapter, LettaAdapter};
use anamnesis_core::contract::AdapterContract;
use rusqlite::Connection;

static NONCE: AtomicU64 = AtomicU64::new(0);

fn fixture_db() -> PathBuf {
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("anamnesis-letta-contract-{pid}-{n}"));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("letta.db");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
        r#"CREATE TABLE block (
            id TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            label TEXT,
            description TEXT,
            metadata_ TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        INSERT INTO block VALUES
          ('p', 'I am Sam.',         'persona', 'self-view',  NULL,        '2024-01-01T00:00:00Z', NULL),
          ('h', 'User likes Rust.',  'human',   'user model', NULL,        '2026-04-01T00:00:00Z', '2026-04-15T00:00:00Z'),
          ('c', 'Custom block.',     'note',    NULL,         '{"v":1}',   '2025-06-01T00:00:00Z', NULL);
        "#,
    )
    .unwrap();
    path
}

#[tokio::test]
async fn letta_satisfies_adapter_contract() {
    let db = fixture_db();
    let contract = AdapterContract::new(move || -> LettaAdapter {
        letta_adapter(db.clone(), Some("self-hosted"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn letta_no_instance_satisfies_contract() {
    let db = fixture_db();
    let contract =
        AdapterContract::new(move || -> LettaAdapter { letta_adapter(db.clone(), None) });
    contract.run_all().await;
}
