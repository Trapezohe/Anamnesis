//! Round-77 PR-77 end-to-end: `anamnesis dedupe`.
//!
//! These tests reach into the store at setup time to plant
//! records with controlled `raw_hash` values — the CLI dedupe
//! detector is pure-SQL grouping on that column, so seeding
//! through the Store API gives deterministic fixtures without
//! depending on which adapter happens to produce a collision in
//! the wild.

use std::path::Path;

use assert_cmd::Command;
use predicates::str::contains;

fn cli() -> Command {
    Command::cargo_bin("anamnesis").expect("cargo bin")
}

fn tmp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

fn init_db(data_dir: &Path) {
    cli()
        .env("ANAMNESIS_DATA_DIR", data_dir)
        .args(["init"])
        .assert()
        .success();
}

/// Seed two records sharing `raw_hash = h-shared` (different
/// adapters) and one unique record. Done directly via the Store
/// API so the test doesn't need a real adapter fixture.
fn seed_duplicates(data_dir: &Path) {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::Store;
    use chrono::Utc;

    let db = data_dir.join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");
    let make = |adapter: &str, native: &str, hash: &str| AnamnesisRecord {
        id: RecordId::from_parts(adapter, None, native),
        source: SourceDescriptor {
            adapter: adapter.into(),
            instance: None,
            version: "0".into(),
        },
        content: format!("{adapter}|{native} content"),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: native.into(),
            native_path: Some(format!("/tmp/{adapter}/{native}.md")),
            captured_at: Utc::now(),
            raw_hash: hash.into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    for r in [
        make("claude-code", "alpha", "h-shared"),
        make("mem0", "beta", "h-shared"),
        make("claude-code", "gamma", "h-singleton"),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
}

#[test]
fn dedupe_empty_store_says_so() {
    let data = tmp_dir();
    init_db(data.path());
    cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe"])
        .assert()
        .success()
        .stdout(contains("no duplicate raw_hash groups"));
}

#[test]
fn dedupe_default_json_redacts_sensitive_fields() {
    let data = tmp_dir();
    init_db(data.path());
    seed_duplicates(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--json"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["count"], 1);
    assert_eq!(v["sensitive_included"], false);
    let groups = v["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0]["record_count"], 2);
    // No `raw_hash` at the group level.
    assert!(
        groups[0].get("raw_hash").is_none(),
        "raw_hash must be redacted by default; got {}",
        groups[0]
    );
    // No `native_path` on any row.
    let rows = groups[0]["records"].as_array().unwrap();
    for row in rows {
        assert_eq!(row["has_native_path"], true);
        assert!(row.get("native_path").is_none());
    }
    // Marker leak check.
    assert!(
        !stdout.contains("h-shared"),
        "raw_hash marker must not appear in redacted output: {stdout}"
    );
    assert!(
        !stdout.contains("/tmp/claude-code/alpha.md"),
        "native_path must not appear in redacted output: {stdout}"
    );
}

#[test]
fn dedupe_include_sensitive_reveals_fields() {
    let data = tmp_dir();
    init_db(data.path());
    seed_duplicates(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--json", "--include-sensitive"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["sensitive_included"], true);
    let group = &v["groups"][0];
    assert_eq!(group["raw_hash"], "h-shared");
    let rows = group["records"].as_array().unwrap();
    let paths: Vec<&str> = rows
        .iter()
        .filter_map(|r| r["native_path"].as_str())
        .collect();
    assert_eq!(paths.len(), 2);
    assert!(
        paths.iter().any(|p| p.ends_with("alpha.md"))
            && paths.iter().any(|p| p.ends_with("beta.md"))
    );
}
