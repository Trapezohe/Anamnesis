//! Cross-source E2E: claude-code + mem0 → unified Hybrid search.
//!
//! This is the proof of the project's central promise (BLUEPRINT §2 core
//! principle): records from two completely different memory frameworks
//! end up in one normalized schema, indexed by Anamnesis's own RAG stack,
//! and findable in a single `anamnesis search` call.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;
use rusqlite::Connection;

fn cli() -> Command {
    Command::cargo_bin("anamnesis").expect("cargo bin")
}

fn tmp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

fn seed_claude_fixture(home: &Path) {
    let proj = home.join(".claude").join("projects").join("p");
    fs::create_dir_all(proj.join("memory")).unwrap();
    fs::write(
        proj.join("memory").join("from_claude_code.md"),
        "---\n\
         name: claude-code-anchor\n\
         description: distinct phrase only the claude file has\n\
         metadata:\n  type: user\n\
         ---\n\n\
         This memory comes from claude-code and mentions the marker phrase \
         crocodileTeaRocket.\n",
    )
    .unwrap();
}

fn seed_mem0_fixture(mem0_db: &Path) {
    let conn = Connection::open(mem0_db).unwrap();
    conn.execute_batch(
        "CREATE TABLE memories(id TEXT PRIMARY KEY, memory TEXT NOT NULL, user_id TEXT);",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memories(id, memory, user_id) VALUES(\
            'm-only-mem0', \
            'This memory comes from mem0 and mentions the marker phrase platypusBanjoComet.', \
            'alice')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memories(id, memory, user_id) VALUES(\
            'm-shared-topic', \
            'Both sources agree the team uses Rust for systems work.', \
            'alice')",
        [],
    )
    .unwrap();
}

#[test]
fn unified_search_returns_hits_from_both_adapters() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_claude_fixture(home.path());
    let mem0_db = home.path().join(".mem0").join("db.sqlite");
    fs::create_dir_all(mem0_db.parent().unwrap()).unwrap();
    seed_mem0_fixture(&mem0_db);

    // 1. init + discover should see both adapters.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["discover"])
        .assert()
        .success()
        .stdout(contains("claude-code").and(contains("mem0")));

    // 2. Import both adapters.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["import", "claude-code", "--no-embed"])
        .assert()
        .success();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["import", "mem0", "--no-embed"])
        .assert()
        .success();

    // 3. Status should show records from BOTH sources (≥ 3: 1 claude-code
    //    memory + 2 mem0 rows; no jsonl session in this fixture).
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["status"])
        .assert()
        .success()
        .stdout(contains("sources").and(contains("records         : 3")));

    // 4. Distinct-phrase searches: each marker must surface only its own
    //    adapter, proving normalization carried provenance through.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "search",
            "crocodileTeaRocket",
            "--mode",
            "fulltext",
            "--limit",
            "5",
        ])
        .assert()
        .success()
        .stdout(contains("claude-code"));

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "search",
            "platypusBanjoComet",
            "--mode",
            "fulltext",
            "--limit",
            "5",
        ])
        .assert()
        .success()
        .stdout(contains("mem0"));

    // 5. The crux: a search that matches BOTH sources returns BOTH
    //    adapters in one ranked result set. This is the "unified RAG"
    //    promise (BLUEPRINT §2 core principle).
    let combined = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "search",
            "marker phrase",
            "--mode",
            "fulltext",
            "--limit",
            "10",
            "--json",
        ])
        .output()
        .expect("run cross-source search");
    assert!(combined.status.success());
    let payload: serde_json::Value =
        serde_json::from_slice(&combined.stdout).expect("parseable json");
    let results = payload["results"].as_array().expect("results array");
    assert!(
        results.len() >= 2,
        "expected ≥2 hits from cross-source query"
    );
    let adapters: std::collections::HashSet<&str> = results
        .iter()
        .filter_map(|r| r["adapter"].as_str())
        .collect();
    assert!(
        adapters.contains("claude-code"),
        "claude-code adapter missing from cross-source result set: {adapters:?}"
    );
    assert!(
        adapters.contains("mem0"),
        "mem0 adapter missing from cross-source result set: {adapters:?}"
    );
}

#[test]
fn source_filter_restricts_to_one_adapter() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_claude_fixture(home.path());
    let mem0_db = home.path().join(".mem0").join("db.sqlite");
    fs::create_dir_all(mem0_db.parent().unwrap()).unwrap();
    seed_mem0_fixture(&mem0_db);

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["import", "claude-code", "--no-embed"])
        .assert()
        .success();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["import", "mem0", "--no-embed"])
        .assert()
        .success();

    // --source filter restricts to the selected adapter only.
    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "search",
            "marker phrase",
            "--mode",
            "fulltext",
            "--source",
            "claude-code",
            "--json",
        ])
        .output()
        .expect("run source-filtered search");
    let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let results = payload["results"].as_array().unwrap();
    assert!(!results.is_empty());
    assert!(
        results.iter().all(|r| r["adapter"] == "claude-code"),
        "--source claude-code should suppress mem0 hits"
    );
}
