//! `anamnesis reconcile-export` — derive-format behaviour (R150).
//! Seeds two mem0 records letta lacks so `only-left` has a 2-record
//! drift bucket whose lagging side (letta) has a canonical round-trip
//! format. Mirrors the MCP `reconcile_export_bucket` parity tests.

use std::path::Path;

use assert_cmd::Command;
use predicates::str::contains;

fn cli() -> Command {
    Command::cargo_bin("anamnesis").expect("cargo bin")
}

fn init_db(data_dir: &Path) {
    cli()
        .env("ANAMNESIS_DATA_DIR", data_dir)
        .args(["init"])
        .assert()
        .success();
}

fn seed(data_dir: &Path) {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::Store;
    use chrono::Utc;

    let store = Store::open(data_dir.join("anamnesis.sqlite")).expect("open store");
    let make = |adapter: &str, native: &str| AnamnesisRecord {
        id: RecordId::from_parts(adapter, None, native),
        source: SourceDescriptor {
            adapter: adapter.into(),
            instance: None,
            version: "0".into(),
        },
        content: format!("{adapter}|{native} body"),
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
            raw_hash: format!("raw-{adapter}-{native}"),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    for r in [make("mem0", "left-A"), make("mem0", "left-B")] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
}

#[test]
fn only_left_derives_lagging_letta_format_when_omitted() {
    let dir = tempfile::tempdir().unwrap();
    init_db(dir.path());
    seed(dir.path());
    let out = dir.path().join("derived.db");
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "reconcile-export",
            "--left",
            "mem0",
            "--right",
            "letta",
            "--bucket",
            "only-left",
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("lagging=letta"))
        .stdout(contains("format=letta-sqlite (derived)"));
    assert!(out.is_file());
}

#[test]
fn only_right_derives_lagging_mem0_format_when_omitted() {
    let dir = tempfile::tempdir().unwrap();
    init_db(dir.path());
    seed(dir.path());
    let out = dir.path().join("right.db");
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "reconcile-export",
            "--left",
            "mem0",
            "--right",
            "letta",
            "--bucket",
            "only-right",
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("lagging=mem0"))
        .stdout(contains("format=mem0-sqlite (derived)"));
}

#[test]
fn omitted_format_errors_when_lagging_has_no_round_trip_target() {
    let dir = tempfile::tempdir().unwrap();
    init_db(dir.path());
    seed(dir.path());
    let out = dir.path().join("nope.db");
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "reconcile-export",
            "--left",
            "mem0",
            "--right",
            "claude-code",
            "--bucket",
            "only-left",
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("no round-trip export format"))
        .stderr(contains("claude-code"));
    assert!(!out.exists());
}

#[test]
fn explicit_mismatch_succeeds_with_warning() {
    let dir = tempfile::tempdir().unwrap();
    init_db(dir.path());
    seed(dir.path());
    let out = dir.path().join("forced.jsonl");
    let assert = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "reconcile-export",
            "--left",
            "mem0",
            "--right",
            "letta",
            "--bucket",
            "only-left",
            "--format",
            "jsonl",
            "--out",
            out.to_str().unwrap(),
            "--json",
        ])
        .assert()
        .success()
        .stderr(contains("differs"));
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let payload: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(payload["format"], "jsonl");
    assert_eq!(payload["format_source"], "explicit");
    assert_eq!(payload["canonical_round_trip_format"], "letta-sqlite");
    assert!(payload["warning"].as_str().unwrap().contains("differs"));
}
