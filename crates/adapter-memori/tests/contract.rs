//! `memori` adapter satisfies the shared `anamnesis_core::contract`
//! invariants. Mirrors the pattern from `adapter-codex` / `adapter-mem0`.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anamnesis_adapter_memori::{memori_adapter, MemoriAdapter};
use anamnesis_core::contract::AdapterContract;
use rusqlite::{params, Connection};

static NONCE: AtomicU64 = AtomicU64::new(0);

fn fixture_db() -> PathBuf {
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("anamnesis-memori-contract-{pid}-{n}"));
    fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("memori.db");
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE memori_entity (id INTEGER PRIMARY KEY, uuid TEXT, external_id TEXT);
         CREATE TABLE memori_process (id INTEGER PRIMARY KEY, uuid TEXT, external_id TEXT);
         CREATE TABLE memori_session (id INTEGER PRIMARY KEY, uuid TEXT, entity_id INTEGER, process_id INTEGER);
         CREATE TABLE memori_conversation (id INTEGER PRIMARY KEY, uuid TEXT, session_id INTEGER, summary TEXT, date_created TEXT);
         CREATE TABLE memori_conversation_message (
             id INTEGER PRIMARY KEY, uuid TEXT, conversation_id INTEGER,
             role TEXT, type TEXT, content TEXT, date_created TEXT
         );
         CREATE TABLE memori_entity_fact (
             id INTEGER PRIMARY KEY, uuid TEXT, entity_id INTEGER,
             content TEXT, num_times INTEGER, date_last_time TEXT, date_created TEXT
         );
         CREATE TABLE memori_process_attribute (
             id INTEGER PRIMARY KEY, uuid TEXT, process_id INTEGER,
             content TEXT, num_times INTEGER, date_last_time TEXT, date_created TEXT
         );
         CREATE TABLE memori_subject (id INTEGER PRIMARY KEY, uuid TEXT, name TEXT, type TEXT);
         CREATE TABLE memori_predicate (id INTEGER PRIMARY KEY, uuid TEXT, content TEXT);
         CREATE TABLE memori_object (id INTEGER PRIMARY KEY, uuid TEXT, name TEXT, type TEXT);
         CREATE TABLE memori_knowledge_graph (
             id INTEGER PRIMARY KEY, uuid TEXT, entity_id INTEGER,
             subject_id INTEGER, predicate_id INTEGER, object_id INTEGER,
             num_times INTEGER, date_last_time TEXT, date_created TEXT
         );",
    )
    .unwrap();

    conn.execute(
        "INSERT INTO memori_entity (id, uuid, external_id) VALUES (?, ?, ?)",
        params![1, "ent-uuid", "user-123"],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memori_process (id, uuid, external_id) VALUES (?, ?, ?)",
        params![10, "proc-uuid", "my-app"],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memori_session (id, uuid, entity_id, process_id) VALUES (?, ?, ?, ?)",
        params![100, "sess-uuid", 1, 10],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memori_conversation (id, uuid, session_id, summary, date_created) \
         VALUES (?, ?, ?, ?, ?)",
        params![
            1000,
            "conv-uuid",
            100,
            "User asked about colors and cities.",
            "2026-05-01 10:00:00",
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memori_conversation_message \
         (uuid, conversation_id, role, type, content, date_created) \
         VALUES (?, ?, ?, ?, ?, ?)",
        params![
            "msg1",
            1000,
            "user",
            "text",
            "My favorite color is blue",
            "2026-05-01 10:00:00",
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memori_conversation_message \
         (uuid, conversation_id, role, type, content, date_created) \
         VALUES (?, ?, ?, ?, ?, ?)",
        params![
            "msg2",
            1000,
            "assistant",
            "text",
            "Got it.",
            "2026-05-01 10:00:01",
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memori_entity_fact \
         (uuid, entity_id, content, num_times, date_last_time, date_created) \
         VALUES (?, ?, ?, ?, ?, ?)",
        params![
            "fact-uuid",
            1,
            "user lives in Paris",
            3,
            "2026-05-01 10:00:00",
            "2026-04-01 10:00:00",
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memori_process_attribute \
         (uuid, process_id, content, num_times, date_last_time, date_created) \
         VALUES (?, ?, ?, ?, ?, ?)",
        params![
            "attr-uuid",
            10,
            "app prefers JSON responses",
            5,
            "2026-05-01 10:00:00",
            "2026-04-01 10:00:00",
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memori_subject (id, uuid, name, type) VALUES (?, ?, ?, ?)",
        params![1, "subj-uuid", "user", "Person"],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memori_predicate (id, uuid, content) VALUES (?, ?, ?)",
        params![1, "pred-uuid", "lives_in"],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memori_object (id, uuid, name, type) VALUES (?, ?, ?, ?)",
        params![1, "obj-uuid", "Paris", "City"],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memori_knowledge_graph \
         (uuid, entity_id, subject_id, predicate_id, object_id, \
          num_times, date_last_time, date_created) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            "kg-uuid",
            1,
            1,
            1,
            1,
            2,
            "2026-05-01 10:00:00",
            "2026-04-01 10:00:00",
        ],
    )
    .unwrap();
    db_path
}

#[tokio::test]
async fn memori_satisfies_adapter_contract() {
    let db = fixture_db();
    let contract = AdapterContract::new(move || -> MemoriAdapter {
        memori_adapter(db.clone(), Some("default"))
    });
    contract.run_all().await;
}

#[tokio::test]
async fn memori_no_instance_satisfies_contract() {
    let db = fixture_db();
    let contract =
        AdapterContract::new(move || -> MemoriAdapter { memori_adapter(db.clone(), None) });
    contract.run_all().await;
}
