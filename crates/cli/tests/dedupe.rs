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

// ─── Round-80: --source / --instance filter ─────────────────────────

/// Seed two duplicate groups so the filter has something to
/// narrow:
///   * h-mixed: mem0/claude-code (filter target)
///   * h-other: two claude-code records (irrelevant)
fn seed_two_groups(data_dir: &Path) {
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
        make("mem0", "m1", "h-mixed"),
        make("claude-code", "c1", "h-mixed"),
        make("claude-code", "x1", "h-other"),
        make("claude-code", "x2", "h-other"),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
}

/// `--source mem0` returns only the mixed group, but with the
/// full sibling set (mem0 + claude-code) — the operator needs to
/// see what they'd be choosing between.
#[test]
fn dedupe_source_filter_scopes_groups_keeps_siblings_whole() {
    let data = tmp_dir();
    init_db(data.path());
    seed_two_groups(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "dedupe",
            "--source",
            "mem0",
            "--json",
            "--include-sensitive",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["count"], 1, "h-other group filtered out: {stdout}");
    assert_eq!(v["filter"]["source"], "mem0");
    assert!(v["filter"]["instance"].is_null());
    let group = &v["groups"][0];
    assert_eq!(group["raw_hash"], "h-mixed");
    // Both adapters visible.
    let adapters: std::collections::BTreeSet<String> = group["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["adapter"].as_str().unwrap().to_string())
        .collect();
    assert!(adapters.contains("mem0"));
    assert!(adapters.contains("claude-code"));
}

/// `--source` that nobody matches returns an empty report and
/// the filter is echoed back so the operator can see "ok, you
/// said `letta`, that's why it's empty."
#[test]
fn dedupe_source_filter_unknown_adapter_human_shows_filter() {
    let data = tmp_dir();
    init_db(data.path());
    seed_two_groups(data.path());

    cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--source", "letta"])
        .assert()
        .success()
        .stdout(contains("no duplicate raw_hash groups"))
        .stdout(contains("filter: source=letta"));
}

/// Limit-before-filter regression guard: with `--limit 1`, the
/// 2-row mem0-containing group must win even though the 2-row
/// pure-claude-code group has the same size. The filter narrows
/// eligibility *before* the LIMIT clause.
#[test]
fn dedupe_source_filter_limit_picks_filtered_group() {
    let data = tmp_dir();
    init_db(data.path());
    seed_two_groups(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "dedupe",
            "--source",
            "mem0",
            "--limit",
            "1",
            "--json",
            "--include-sensitive",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["count"], 1);
    assert_eq!(v["groups"][0]["raw_hash"], "h-mixed");
}

// ─── Round-97 PR-78s: dedupe --include-counts ──────────────────────

/// Default `dedupe --json` has no `counts` block — every R77
/// consumer keeps working verbatim.
#[test]
fn dedupe_default_json_has_no_counts_block() {
    let data = tmp_dir();
    init_db(data.path());
    seed_duplicates(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        v.get("counts").is_none(),
        "default dedupe must not carry counts; got {v}"
    );
}

/// `--include-counts` attaches the filter-scoped aggregate.
/// `limit` doesn't affect counts; they always reflect the full
/// matching set.
#[test]
fn dedupe_include_counts_reflects_full_set_ignoring_limit() {
    let data = tmp_dir();
    init_db(data.path());
    seed_two_groups(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--limit", "1", "--json", "--include-counts"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["count"], 1, "rows still respect --limit");
    let counts = &v["counts"];
    // seed_two_groups builds h-mixed (mem0 + claude-code) and
    // h-other (claude-code + claude-code) — 2 groups, 4 records.
    assert_eq!(counts["total_groups"], 2);
    assert_eq!(counts["total_records"], 4);
    let by_source = counts["by_source"].as_array().unwrap();
    let cc = by_source
        .iter()
        .find(|b| b["adapter"] == "claude-code")
        .unwrap();
    let mem = by_source.iter().find(|b| b["adapter"] == "mem0").unwrap();
    assert_eq!(cc["duplicate_record_count"], 3);
    assert_eq!(mem["duplicate_record_count"], 1);
    assert!(cc["instance"].is_null());
    assert!(mem["instance"].is_null());
}

/// Counts block carries only numerics — no `raw_hash`, no
/// `native_path`, no `native_id`. Stays inside the existing
/// dedupe redaction boundary.
#[test]
fn dedupe_counts_block_carries_no_sensitive_fields() {
    let data = tmp_dir();
    init_db(data.path());
    seed_two_groups(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--json", "--include-counts"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let counts_str = serde_json::to_string(&v["counts"]).unwrap();
    for forbidden in ["h-mixed", "h-other", "raw_hash", "native_path", "native_id"] {
        assert!(
            !counts_str.contains(forbidden),
            "counts must not leak {forbidden:?}: {counts_str}"
        );
    }
}
