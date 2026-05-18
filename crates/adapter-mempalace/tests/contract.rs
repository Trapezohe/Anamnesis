//! `mempalace` adapter satisfies the shared `anamnesis_core::contract`
//! invariants. Mirrors the pattern from `adapter-codex` / `adapter-mem0`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_mempalace::{mempalace_adapter, MempalaceAdapter};
use anamnesis_core::contract::AdapterContract;
use rusqlite::{params, Connection};

static NONCE: AtomicU64 = AtomicU64::new(0);

// Mirrors the (private) constants in `adapter-mempalace::scanner` —
// these are the chromaDB collection names the scanner queries for.
const COLLECTION_DRAWERS: &str = "mempalace_drawers";
const COLLECTION_CLOSETS: &str = "mempalace_closets";

fn fixture_home() -> PathBuf {
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let home = std::env::temp_dir().join(format!("anamnesis-mempalace-contract-{pid}-{n}"));
    fs::create_dir_all(home.join("palace")).unwrap();
    fs::write(home.join("identity.txt"), "I am a senior engineer.").unwrap();

    let db_path = home.join("palace/chroma.sqlite3");
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, dimension INTEGER);
         CREATE TABLE segments (id TEXT PRIMARY KEY, collection TEXT, scope TEXT);
         CREATE TABLE embeddings (
             id INTEGER PRIMARY KEY,
             segment_id TEXT,
             embedding_id TEXT,
             seq_id BLOB,
             created_at INTEGER
         );
         CREATE TABLE embedding_metadata (
             id INTEGER,
             key TEXT,
             string_value TEXT,
             int_value INTEGER,
             float_value REAL,
             bool_value INTEGER
         );",
    )
    .unwrap();

    // Drawers collection.
    conn.execute(
        "INSERT INTO collections (id, name, dimension) VALUES (?, ?, ?)",
        params!["coll-drawers", COLLECTION_DRAWERS, 384],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO segments (id, collection, scope) VALUES (?, ?, ?)",
        params!["seg-drawers", "coll-drawers", "METADATA"],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO embeddings (id, segment_id, embedding_id, created_at) \
         VALUES (?, ?, ?, ?)",
        params![
            1,
            "seg-drawers",
            "drawer_default_general_aaa",
            1_730_000_000_i64
        ],
    )
    .unwrap();
    for (k, sv) in [
        (
            "chroma:document",
            "user prefers dark mode and tabs over spaces",
        ),
        ("wing", "default"),
        ("room", "general"),
        ("source_file", "/repo/CLAUDE.md"),
        ("filed_at", "2026-05-01T10:00:00Z"),
    ] {
        conn.execute(
            "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
            params![1, k, sv],
        )
        .unwrap();
    }

    // Closets collection.
    conn.execute(
        "INSERT INTO collections (id, name, dimension) VALUES (?, ?, ?)",
        params!["coll-closets", COLLECTION_CLOSETS, 384],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO segments (id, collection, scope) VALUES (?, ?, ?)",
        params!["seg-closets", "coll-closets", "METADATA"],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO embeddings (id, segment_id, embedding_id, created_at) \
         VALUES (?, ?, ?, ?)",
        params![
            2,
            "seg-closets",
            "closet_default_general_zzz",
            1_730_000_002_i64
        ],
    )
    .unwrap();
    for (k, sv) in [
        ("chroma:document", "rooms in wing default: general, work"),
        ("wing", "default"),
    ] {
        conn.execute(
            "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
            params![2, k, sv],
        )
        .unwrap();
    }

    home
}

#[tokio::test]
async fn mempalace_satisfies_adapter_contract() {
    let home = fixture_home();
    let contract = AdapterContract::new(move || -> MempalaceAdapter {
        mempalace_adapter(home.clone(), Some("default"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn mempalace_no_instance_satisfies_contract() {
    let home = fixture_home();
    let contract =
        AdapterContract::new(move || -> MempalaceAdapter { mempalace_adapter(home.clone(), None) });
    contract.run_all().await;
}
