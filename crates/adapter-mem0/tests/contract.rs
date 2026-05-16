//! `mem0` adapter satisfies the shared `anamnesis_core::contract`
//! invariants (instance dedup, idempotent scan, pure normalize, …).

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_mem0::{sqlite_adapter, Mem0Adapter};
use anamnesis_core::contract::AdapterContract;
use rusqlite::Connection;

static NONCE: AtomicU64 = AtomicU64::new(0);

fn fixture_db() -> PathBuf {
    // Atomic counter + pid avoids same-nanosecond collisions between
    // parallel test threads in this binary.
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("anamnesis-mem0-contract-{pid}-{n}"));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db.sqlite");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
        "CREATE TABLE memories(id TEXT PRIMARY KEY, memory TEXT NOT NULL, user_id TEXT, created_at TEXT);",
    )
    .unwrap();
    for (id, mem) in [
        ("a", "user prefers vim"),
        ("b", "do not mock the database"),
        ("c", "Friday deploys forbidden"),
    ] {
        conn.execute(
            "INSERT INTO memories(id, memory, user_id, created_at) VALUES(?1,?2,?3,?4)",
            rusqlite::params![id, mem, "u1", "1700000000"],
        )
        .unwrap();
    }
    path
}

#[tokio::test]
async fn mem0_sqlite_satisfies_adapter_contract() {
    let db = fixture_db();
    let contract = AdapterContract::new(move || -> Mem0Adapter {
        sqlite_adapter(db.clone(), Some("self-hosted"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn mem0_no_instance_satisfies_contract() {
    let db = fixture_db();
    let contract =
        AdapterContract::new(move || -> Mem0Adapter { sqlite_adapter(db.clone(), None) });
    contract.run_all().await;
}
