//! Round-135 PR-78bd end-to-end: `anamnesis conflicts`.
//!
//! Validates the cross-adapter `native_id` content disagreement
//! detector through the CLI binary. Same fixture pattern as the
//! R77 dedupe tests — seed via the Store API so we don't have to
//! drive a real adapter just to plant a conflict.
//!
//! Acceptance:
//!   1. Only cross-adapter `native_id` groups with disagreeing
//!      `content` surface. Singletons drop. Same-content
//!      cross-adapter pairs drop.
//!   2. Default output redacts `content_preview` and
//!      `native_path`. `--include-content` opts in to the short
//!      preview but still redacts `native_path`.
//!   3. JSON shape carries `summary`, `count`, `groups[]`,
//!      and per-record `content_variant` indices.

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

/// Plant a cross-adapter conflict on `shared-1` (mem0 says A,
/// claude-code says B) plus a control on `shared-2` (identical
/// content across adapters, must not group). Seeded via the
/// Store API so the test doesn't need to drive a real adapter.
fn seed_conflict_fixture(data_dir: &Path) -> String {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::Store;
    use chrono::Utc;

    let db = data_dir.join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");

    let mk = |adapter: &str, native: &str, content: &str, raw: &str| AnamnesisRecord {
        id: RecordId::from_parts(adapter, None, native),
        source: SourceDescriptor {
            adapter: adapter.into(),
            instance: None,
            version: "0".into(),
        },
        content: content.into(),
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
            raw_hash: raw.into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };

    for r in [
        mk("mem0", "shared-1", "Variant A body for shared-1", "raw-a"),
        mk(
            "claude-code",
            "shared-1",
            "Variant B body for shared-1",
            "raw-b",
        ),
        // Identical content across adapters → must NOT group.
        mk("mem0", "shared-2", "Identical body", "raw-c"),
        mk("claude-code", "shared-2", "Identical body", "raw-d"),
        // Singleton → must NOT group.
        mk("codex", "solo-3", "Solo record", "raw-e"),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
    "shared-1".to_string()
}

#[test]
fn conflicts_empty_store_says_so() {
    let data = tmp_dir();
    init_db(data.path());
    cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["conflicts"])
        .assert()
        .success()
        .stdout(contains("no cross-adapter `native_id` content conflicts"));
}

#[test]
fn conflicts_default_json_redacts_content_and_paths() {
    let data = tmp_dir();
    init_db(data.path());
    let nid = seed_conflict_fixture(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["conflicts", "--json"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["count"], 1, "must isolate the one true conflict group");
    assert_eq!(v["format"], "json");
    assert_eq!(v["content_included"], false);
    let summary = v["summary"].as_str().unwrap();
    assert!(summary.contains("content conflict group(s)"));
    assert!(summary.contains("content_preview: redacted"));

    let g = &v["groups"][0];
    assert_eq!(g["native_id"], nid);
    assert_eq!(g["record_count"], 2);
    assert_eq!(g["content_variant_count"], 2);

    let rows = g["records"].as_array().unwrap();
    let variants: std::collections::BTreeSet<u64> = rows
        .iter()
        .map(|r| r["content_variant"].as_u64().unwrap())
        .collect();
    assert_eq!(variants, [1u64, 2].into_iter().collect());

    // Privacy: no content_preview, no native_path; has_native_path
    // is reported as boolean.
    for row in rows {
        assert!(
            row.get("content_preview").is_none(),
            "default must NOT include content_preview: {row}"
        );
        assert!(row.get("native_path").is_none());
        assert_eq!(row["has_native_path"], true);
    }

    // Marker leaks: no raw_hash, no content bodies, no paths.
    for forbidden in [
        "Variant A body for shared-1",
        "Variant B body for shared-1",
        "Identical body",
        "raw-a",
        "raw-b",
        "/tmp/mem0/shared-1.md",
    ] {
        assert!(
            !stdout.contains(forbidden),
            "default conflicts --json must not leak {forbidden:?}: {stdout}"
        );
    }
}

#[test]
fn conflicts_include_content_attaches_short_preview() {
    let data = tmp_dir();
    init_db(data.path());
    seed_conflict_fixture(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["conflicts", "--json", "--include-content"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["count"], 1);
    assert_eq!(v["content_included"], true);
    let rows = v["groups"][0]["records"].as_array().unwrap();
    let bodies: Vec<&str> = rows
        .iter()
        .map(|r| r["content_preview"].as_str().expect("preview present"))
        .collect();
    assert!(bodies.iter().any(|b| b.contains("Variant A body")));
    assert!(bodies.iter().any(|b| b.contains("Variant B body")));
    // native_path still redacted even with --include-content.
    for row in rows {
        assert!(row.get("native_path").is_none());
    }
}

#[test]
fn conflicts_source_filter_keeps_full_group_visible() {
    let data = tmp_dir();
    init_db(data.path());
    seed_conflict_fixture(data.path());

    // Filter on `mem0` — the cross-adapter group still surfaces
    // with the claude-code sibling so the operator can compare.
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["conflicts", "--source", "mem0", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["count"], 1);
    let adapters: std::collections::BTreeSet<String> = v["groups"][0]["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["adapter"].as_str().unwrap().to_string())
        .collect();
    assert!(adapters.contains("mem0"));
    assert!(
        adapters.contains("claude-code"),
        "sibling must stay visible under filter"
    );
}

#[test]
fn conflicts_unknown_source_filter_returns_zero_groups() {
    let data = tmp_dir();
    init_db(data.path());
    seed_conflict_fixture(data.path());

    cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["conflicts", "--source", "hermes"])
        .assert()
        .success()
        .stdout(contains("no cross-adapter"));
}
