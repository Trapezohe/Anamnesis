//! `anamnesis reconcile` — R152 round_trip diagnostic hint. Asserts the
//! drift output surfaces, per direction, the lagging side and the format
//! reconcile-export would derive. Parity with the MCP `reconcile_sources`
//! `round_trip` subtree.

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
    for r in [make("mem0", "m1"), make("letta", "l1")] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
}

#[test]
fn reconcile_json_surfaces_round_trip_hint() {
    let dir = tempfile::tempdir().unwrap();
    init_db(dir.path());
    seed(dir.path());
    let assert = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["reconcile", "--left", "mem0", "--right", "letta", "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let payload: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let rt = &payload["round_trip"];
    // only_left lags right (letta); only_right lags left (mem0).
    assert_eq!(rt["only_left"]["lagging"]["adapter"], "letta");
    assert_eq!(rt["only_left"]["export_format"], "letta-sqlite");
    assert_eq!(rt["only_right"]["lagging"]["adapter"], "mem0");
    assert_eq!(rt["only_right"]["export_format"], "mem0-sqlite");
}

#[test]
fn reconcile_json_round_trip_null_for_no_target_adapter() {
    let dir = tempfile::tempdir().unwrap();
    init_db(dir.path());
    seed(dir.path());
    let assert = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "reconcile",
            "--left",
            "mem0",
            "--right",
            "claude-code",
            "--json",
        ])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let payload: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let rt = &payload["round_trip"];
    assert!(rt["only_left"]["export_format"].is_null());
    assert_eq!(rt["only_right"]["export_format"], "mem0-sqlite");
}

#[test]
fn reconcile_human_output_includes_round_trip_line() {
    let dir = tempfile::tempdir().unwrap();
    init_db(dir.path());
    seed(dir.path());
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["reconcile", "--left", "mem0", "--right", "letta"])
        .assert()
        .success()
        .stdout(contains(
            "round_trip: only_left -> lagging=letta format=letta-sqlite",
        ))
        .stdout(contains("only_right -> lagging=mem0 format=mem0-sqlite"));
}
