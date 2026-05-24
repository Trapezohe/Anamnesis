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
    // Round 108 (PR-78ad): `format: "json"` marker pairs
    // with R107's `format: "csv"` so a script that supports
    // both shapes can branch on `payload.format` instead of
    // probing for `groups[]` vs `csv`.
    assert_eq!(v["format"], "json");
    // Round 125 (PR-78at): top-level redacted summary
    // mirrors MCP R117 dedupe summary. Must surface count
    // and default filter/sensitive/counts state. Must NEVER
    // contain raw_hash (`h-shared`) or native_path.
    let summary = v["summary"]
        .as_str()
        .expect("dedupe --json must carry top-level `summary`");
    assert!(
        summary.contains("1 duplicate group(s) returned"),
        "summary must declare count: {summary}"
    );
    assert!(
        summary.contains("source filter: all sources"),
        "no-filter summary must say `all sources`: {summary}"
    );
    assert!(
        summary.contains("sensitive: redacted"),
        "default sensitive state must surface: {summary}"
    );
    assert!(
        summary.contains("counts: omitted"),
        "default counts state must surface: {summary}"
    );
    assert!(
        !summary.contains("h-shared"),
        "summary must not leak raw_hash: {summary}"
    );
    assert!(
        !summary.contains("/tmp/claude-code/alpha.md"),
        "summary must not leak native_path: {summary}"
    );
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

// ─── Round-104 PR-78z: dedupe --source multi-value OR ─────────────

/// Build a 3-group fixture with adapter-distinct groups so the
/// OR filter is unambiguous: `h-mem` (mem0), `h-cc`
/// (claude-code), `h-cx` (codex). `--source mem0,claude-code`
/// must return groups 1 and 2 and drop group 3.
fn seed_three_groups(data_dir: &std::path::Path) {
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
        make("mem0", "m1", "h-mem"),
        make("mem0", "m2", "h-mem"),
        make("claude-code", "c1", "h-cc"),
        make("claude-code", "c2", "h-cc"),
        make("codex", "x1", "h-cx"),
        make("codex", "x2", "h-cx"),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
}

fn seed_instance_groups(data_dir: &std::path::Path) {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::Store;
    use chrono::Utc;

    let db = data_dir.join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");
    let make = |instance: &str, native: &str, hash: &str| AnamnesisRecord {
        id: RecordId::from_parts("mem0", Some(instance), native),
        source: SourceDescriptor {
            adapter: "mem0".into(),
            instance: Some(instance.into()),
            version: "0".into(),
        },
        content: format!("mem0:{instance}|{native} content"),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: native.into(),
            native_path: Some(format!("/tmp/mem0/{instance}/{native}.md")),
            captured_at: Utc::now(),
            raw_hash: hash.into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    for r in [
        make("prod", "p1", "h-prod"),
        make("prod", "p2", "h-prod"),
        make("dev", "d1", "h-dev"),
        make("qa", "q1", "h-dev"),
        make("qa", "q2", "h-qa"),
        make("qa", "q3", "h-qa"),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
}

/// `--source mem0,claude-code` is the OR filter: both
/// adapter-specific groups survive, the codex group drops.
/// `filter.source` echoes the raw input string (R97/R80 wire
/// shape unchanged) so downstream scripts still see what the
/// operator typed. The new multi-value capability lives in the
/// store's filter logic, not in the JSON wire format.
#[test]
fn dedupe_source_multi_value_or_narrows_to_listed_adapters() {
    let data = tmp_dir();
    init_db(data.path());
    seed_three_groups(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "dedupe",
            "--source",
            "mem0, , claude-code",
            "--json",
            "--include-sensitive",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["count"], 2, "codex group must drop: {stdout}");
    let hashes: std::collections::BTreeSet<String> = v["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["raw_hash"].as_str().unwrap().to_string())
        .collect();
    assert!(
        hashes.contains("h-mem"),
        "mem0 group must survive: {hashes:?}"
    );
    assert!(
        hashes.contains("h-cc"),
        "claude-code group must survive: {hashes:?}"
    );
    assert!(
        !hashes.contains("h-cx"),
        "codex group must drop under multi-value OR: {hashes:?}"
    );
    // `filter.source` keeps the raw operator-supplied string —
    // R97 wire shape, no break.
    assert_eq!(v["filter"]["source"], "mem0, , claude-code");
}

/// `--include-counts` honours the same multi-source eligibility:
/// `total_groups` reflects the eligible-only set, and `by_source[]`
/// reports records only from surviving groups.
#[test]
fn dedupe_source_multi_value_or_counts_respect_filter() {
    let data = tmp_dir();
    init_db(data.path());
    seed_three_groups(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "dedupe",
            "--source",
            "mem0,claude-code",
            "--json",
            "--include-counts",
        ])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["counts"]["total_groups"], 2);
    assert_eq!(v["counts"]["total_records"], 4);
    let by_source = v["counts"]["by_source"].as_array().unwrap();
    let adapters: std::collections::BTreeSet<&str> = by_source
        .iter()
        .map(|b| b["adapter"].as_str().unwrap())
        .collect();
    assert!(adapters.contains("mem0"));
    assert!(adapters.contains("claude-code"));
    assert!(
        !adapters.contains("codex"),
        "codex must be excluded from by_source under filter: {adapters:?}"
    );
}

// ─── Round-115 PR-78ak: dedupe --instance multi-value OR ────────────

#[test]
fn dedupe_instance_multi_value_or_narrows_to_listed_instances() {
    let data = tmp_dir();
    init_db(data.path());
    seed_instance_groups(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "dedupe",
            "--source",
            "mem0",
            "--instance",
            "prod, , dev",
            "--json",
            "--include-sensitive",
        ])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["count"], 2);
    let hashes: std::collections::BTreeSet<&str> = v["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["raw_hash"].as_str().unwrap())
        .collect();
    assert_eq!(hashes, ["h-dev", "h-prod"].into_iter().collect());
    assert_eq!(v["filter"]["instance"], "prod, , dev");
}

#[test]
fn dedupe_instance_multi_value_or_counts_respect_filter() {
    let data = tmp_dir();
    init_db(data.path());
    seed_instance_groups(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "dedupe",
            "--source",
            "mem0",
            "--instance",
            "prod,dev",
            "--json",
            "--include-counts",
        ])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["counts"]["total_groups"], 2);
    assert_eq!(v["counts"]["total_records"], 4);
    let has_qa_sibling = v["counts"]["by_source"]
        .as_array()
        .unwrap()
        .iter()
        .any(|b| b["instance"] == "qa" && b["duplicate_record_count"] == 1);
    assert!(
        has_qa_sibling,
        "whole-group counts must keep qa sibling: {v}"
    );
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

// ─── Round-107 PR-78ac: dedupe --csv ────────────────────────────────

/// Empty store + `--csv` still emits the fixed header so
/// downstream scripts can branch uniformly. Same contract as
/// R91 `audit tail --csv` and R106 `list-forgotten --csv`.
#[test]
fn dedupe_csv_empty_emits_header_only() {
    let data = tmp_dir();
    init_db(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--csv"])
        .output()
        .unwrap();
    assert!(out.status.success(), "csv on empty store must exit 0");
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        stdout.trim(),
        "group_index,record_id,adapter,instance,native_id,created_at,updated_at,has_native_path,record_count"
    );
}

/// `--csv` emits redacted summary rows. `raw_hash` (the
/// duplicate-grouping key) NEVER appears. `native_path` NEVER
/// appears. Rows in the same group share the same
/// `group_index` — operator can pivot by it without ever
/// seeing the underlying hash.
#[test]
fn dedupe_csv_returns_redacted_rows_with_group_index_membership() {
    let data = tmp_dir();
    init_db(data.path());
    seed_two_groups(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--csv"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    // Header + 4 rows (2 groups × 2 records each).
    assert_eq!(lines.len(), 5, "expected header + 4 rows: {stdout}");
    assert!(lines[0].starts_with("group_index,record_id,"));

    // raw_hash MUST NOT appear (load-bearing privacy contract).
    assert!(
        !stdout.contains("h-mixed"),
        "csv must not leak raw_hash `h-mixed`: {stdout}"
    );
    assert!(
        !stdout.contains("h-other"),
        "csv must not leak raw_hash `h-other`: {stdout}"
    );
    // native_path MUST NOT appear.
    assert!(
        !stdout.contains("/tmp/mem0/"),
        "csv must not leak native_path: {stdout}"
    );
    assert!(
        !stdout.contains("/tmp/claude-code/"),
        "csv must not leak native_path: {stdout}"
    );

    // group_index pivot: row[1] is the first row of the first
    // group, row[2] is the second record of the same group, so
    // they must share group_index `0`. row[3] starts the next
    // group with group_index `1`.
    let first_group_index = lines[1].split(',').next().unwrap();
    let second_row_group_index = lines[2].split(',').next().unwrap();
    let third_row_group_index = lines[3].split(',').next().unwrap();
    assert_eq!(first_group_index, "0");
    assert_eq!(
        second_row_group_index, "0",
        "second row of first group must share group_index"
    );
    assert_eq!(
        third_row_group_index, "1",
        "first row of second group must increment group_index"
    );
}

/// `--csv` is mutually exclusive with `--include-sensitive`
/// (runtime check). CSV is the redacted-summary form by
/// design; mixing them would either leak `raw_hash` /
/// `native_path` or pretend the CSV carried more shape than
/// it does.
#[test]
fn dedupe_csv_and_include_sensitive_are_mutually_exclusive() {
    let data = tmp_dir();
    init_db(data.path());
    seed_two_groups(data.path());

    cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--csv", "--include-sensitive"])
        .assert()
        .failure();
}

/// `--csv --json` (clap-rejected) and `--csv --include-counts`
/// (runtime-rejected). CSV is flat redacted rows — no nested
/// counts block, no structured form.
#[test]
fn dedupe_csv_and_json_are_mutually_exclusive() {
    let data = tmp_dir();
    init_db(data.path());
    seed_two_groups(data.path());

    cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--csv", "--json"])
        .assert()
        .failure();
    cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--csv", "--include-counts"])
        .assert()
        .failure();
}

// ─── Round 132 PR-78ba: dedupe --mode near ───────────────────────────

/// Plant a mem0+claude-code paraphrase pair (different `raw_hash`
/// but lexically near-identical), plus an unrelated codex record,
/// so the R131 algorithm has the canonical positive + negative
/// case under one fixture. Reuses the same prose family as the
/// store-side `near_dedupe_groups_cross_adapter_paraphrases` test
/// — important because the SimHash / Jaccard thresholds are
/// calibrated against this exact wording.
fn seed_near_dup_paraphrases(data_dir: &Path) {
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
        mk(
            "mem0",
            "rec-a",
            "The user prefers thorough error handling in Rust code and \
             writes comprehensive integration tests with real fixtures \
             and never uses mocks for critical paths.",
            "near-raw-mem0",
        ),
        mk(
            "claude-code",
            "rec-b",
            "User prefers thorough error handling in Rust code; \
             comprehensive integration tests with real fixtures; \
             no mocks on critical paths.",
            "near-raw-cc",
        ),
        mk(
            "codex",
            "rec-c",
            "Configure the database connection pool to recycle stale \
             connections after sixty seconds and enable TLS for all \
             outbound traffic through the corporate proxy.",
            "near-raw-codex-unrelated",
        ),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
}

/// Default `dedupe --json` (no `--mode`) stays at the exact path
/// and now carries the new `mode: "exact"` discriminator without
/// breaking any earlier wire shape. Back-compat canary.
#[test]
fn dedupe_default_mode_is_exact_and_carries_mode_field() {
    let data = tmp_dir();
    init_db(data.path());
    seed_duplicates(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["mode"], "exact", "default mode must be exact: {v}");
    // Old fields still present.
    assert_eq!(v["format"], "json");
    assert_eq!(v["count"], 1);
    // No new R132 fields leaking into exact response.
    assert!(
        v["filter"].get("require_cross_source").is_none(),
        "require_cross_source belongs to near-only filter"
    );
}

/// `dedupe --mode near --json` finds the cross-adapter paraphrase
/// pair and surfaces `min_similarity` + `max_distance` per group.
/// Unrelated codex record never joins. This is the core R132
/// happy-path: R131's algorithm reached the CLI surface.
#[test]
fn dedupe_near_json_groups_cross_adapter_paraphrases() {
    let data = tmp_dir();
    init_db(data.path());
    seed_near_dup_paraphrases(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--json"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["mode"], "near");
    assert_eq!(v["format"], "json");
    assert_eq!(
        v["count"], 1,
        "must find the mem0+claude-code near-dup group: {stdout}"
    );
    assert_eq!(v["filter"]["require_cross_source"], true);
    let summary = v["summary"]
        .as_str()
        .expect("near --json must carry top-level summary");
    assert!(
        summary.contains("near-duplicate group(s) returned (mode=near)"),
        "near summary must declare mode: {summary}"
    );
    let group = &v["groups"][0];
    assert_eq!(group["record_count"], 2);
    let sim = group["min_similarity"].as_f64().unwrap();
    assert!(
        sim >= 0.6,
        "min_similarity must be >= Jaccard threshold: {sim}"
    );
    let dist = group["max_distance"].as_u64().unwrap();
    assert!(
        dist <= 8,
        "max_distance must be within Hamming threshold: {dist}"
    );
    let adapters: std::collections::BTreeSet<String> = group["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["adapter"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        adapters,
        ["claude-code", "mem0"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    );
}

/// Privacy canary: `--mode near --json` must not leak any record
/// content, raw_hash, or native_path. The algorithm reads content
/// in-memory only; never returns it. Same discipline as R107/R125
/// summary surfaces.
#[test]
fn dedupe_near_json_never_leaks_content_or_paths() {
    let data = tmp_dir();
    init_db(data.path());
    seed_near_dup_paraphrases(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--json"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    for forbidden in [
        "thorough error handling",
        "integration tests",
        "no mocks",
        "near-raw-mem0",
        "near-raw-cc",
        "/tmp/mem0/rec-a.md",
        "/tmp/claude-code/rec-b.md",
        "raw_hash",
    ] {
        assert!(
            !stdout.contains(forbidden),
            "near payload must not leak {forbidden:?}: {stdout}"
        );
    }
    // has_native_path must still be reported as a boolean.
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let row = &v["groups"][0]["records"][0];
    assert!(row["has_native_path"].is_boolean());
    assert!(row.get("native_path").is_none());
}

/// `--mode near --csv` emits the R107-style flat header plus the
/// new `min_similarity` and `max_distance` columns. Empty store
/// still prints the header so scripts can branch uniformly.
#[test]
fn dedupe_near_csv_carries_similarity_columns() {
    let data = tmp_dir();
    init_db(data.path());

    // Empty case → header only.
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--csv"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let header = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        header.trim(),
        "group_index,record_id,adapter,instance,native_id,created_at,updated_at,has_native_path,record_count,min_similarity,max_distance"
    );

    seed_near_dup_paraphrases(data.path());
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--csv"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "header + 2 rows for the dup pair: {stdout}");
    // First row of the (only) group has group_index 0.
    assert!(lines[1].starts_with("0,"));
    // No raw_hash / native_path leak in CSV either.
    for forbidden in [
        "near-raw-mem0",
        "near-raw-cc",
        "/tmp/mem0/rec-a.md",
        "/tmp/claude-code/rec-b.md",
    ] {
        assert!(
            !stdout.contains(forbidden),
            "near CSV must not leak {forbidden:?}: {stdout}"
        );
    }
}

/// `--mode near --include-sensitive` is refused — the near surface
/// has no `raw_hash` / `native_path` to reveal, so a passing flag
/// would silently no-op and mislead the operator.
#[test]
fn dedupe_near_rejects_include_sensitive() {
    let data = tmp_dir();
    init_db(data.path());
    seed_near_dup_paraphrases(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--include-sensitive"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("mutually exclusive"),
        "must surface the conflict: {stderr}"
    );
}

/// `--mode near --include-counts` is refused — the counts
/// aggregate is exact-dedupe specific (raw_hash-by-source
/// breakdown). Group cardinality is already in each near group.
#[test]
fn dedupe_near_rejects_include_counts() {
    let data = tmp_dir();
    init_db(data.path());
    seed_near_dup_paraphrases(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--include-counts"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("mutually exclusive"),
        "must surface the conflict: {stderr}"
    );
}

/// `--include-near-self` only applies under `--mode near`. Passing
/// it on `--mode exact` is a fast-fail so an operator notices
/// they meant `--mode near`.
#[test]
fn dedupe_exact_rejects_include_near_self() {
    let data = tmp_dir();
    init_db(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--include-near-self"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("--mode near"),
        "must point at the right mode: {stderr}"
    );
}

/// `--mode near --include-near-self` opts out of the cross-source
/// filter — within-adapter near-dups surface too. Verifies the
/// R131 `require_cross_source` toggle is plumbed through.
#[test]
fn dedupe_near_include_near_self_surfaces_within_adapter_groups() {
    let data = tmp_dir();
    init_db(data.path());

    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::Store;
    use chrono::Utc;
    let db = data.path().join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");

    // Two mem0 records with effectively the same content — the
    // R131 default filter drops them because there's only one
    // adapter; `--include-near-self` should bring them back.
    let mk = |native: &str, content: &str, raw: &str| AnamnesisRecord {
        id: RecordId::from_parts("mem0", None, native),
        source: SourceDescriptor {
            adapter: "mem0".into(),
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
            native_path: None,
            captured_at: Utc::now(),
            raw_hash: raw.into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    for r in [
        mk(
            "self-a",
            "User prefers thorough error handling in Rust code with comprehensive integration tests and real fixtures.",
            "raw-self-a",
        ),
        mk(
            "self-b",
            "User prefers thorough error handling in Rust code with comprehensive integration tests and real fixtures.",
            "raw-self-b",
        ),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }

    // Default cross-source filter drops the pair.
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["count"], 0, "default filter must drop within-adapter");

    // With opt-out, the pair surfaces.
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--include-near-self", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["count"], 1, "opt-out must surface within-adapter pair");
    assert_eq!(v["filter"]["require_cross_source"], false);
}

/// Empty store + `--mode near` prints the no-groups headline so
/// operators don't think they hit a stack-trace path. Same UX
/// shape as the exact branch's "no duplicate raw_hash groups".
#[test]
fn dedupe_near_empty_store_says_so() {
    let data = tmp_dir();
    init_db(data.path());
    cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near"])
        .assert()
        .success()
        .stdout(contains("no near-duplicate groups"));
}

// ─── Round 141 PR-78bj: dedupe --mode near --merge-preview ──────────

/// Happy path: with the same paraphrase fixture R132 uses, the
/// merge-preview block proposes one keeper and one forget loser.
/// User tags on a record force it to win regardless of recency.
#[test]
fn dedupe_near_merge_preview_user_tag_wins_over_recency() {
    let data = tmp_dir();
    init_db(data.path());
    seed_near_dup_paraphrases(data.path());

    // First find the records via near-dedupe (no preview) so we
    // can tag one — then re-run WITH preview and confirm the
    // tagged record is the keeper.
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let group = &v["groups"][0];
    // Tag the mem0 record — but only if it's the OLDER one so the
    // preview must override recency to honour the tag.
    let mem0_id = group["records"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["adapter"] == "mem0")
        .unwrap()["record_id"]
        .as_str()
        .unwrap()
        .to_string();
    cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "tag-record",
            &mem0_id,
            "important",
            "keep-forever",
            "--json",
        ])
        .assert()
        .success();

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--merge-preview", "--json"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["merge_preview_included"], true);
    let group = &v["groups"][0];
    let preview = &group["merge_preview"];
    assert_eq!(
        preview["keep_record_id"], mem0_id,
        "tagged record must win: {preview}"
    );
    assert_eq!(preview["forget_record_ids"].as_array().unwrap().len(), 1);
    // proposed_derived_from edges point losers → keeper.
    let edges = preview["proposed_derived_from"].as_array().unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0]["to"], mem0_id);
    // ranking[] has 2 rows: rank 1 (keep) + rank 2 (forget).
    let ranking = preview["ranking"].as_array().unwrap();
    assert_eq!(ranking.len(), 2);
    assert_eq!(ranking[0]["rank"], 1);
    assert_eq!(ranking[0]["decision"], "keep");
    assert_eq!(ranking[0]["user_tag_count"], 2);
    assert_eq!(ranking[1]["rank"], 2);
    assert_eq!(ranking[1]["decision"], "forget");
    assert_eq!(ranking[1]["user_tag_count"], 0);
}

/// Privacy canary: --merge-preview must NOT leak content,
/// raw_hash, native_path, or user-tag names. Only counts +
/// metadata-derived ranking signals.
#[test]
fn dedupe_near_merge_preview_does_not_leak_sensitive_fields() {
    let data = tmp_dir();
    init_db(data.path());
    seed_near_dup_paraphrases(data.path());

    // Tag with a sensitive name to make sure the name itself
    // never appears in the merge-preview output.
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let any_id = v["groups"][0]["records"][0]["record_id"]
        .as_str()
        .unwrap()
        .to_string();
    cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "tag-record",
            &any_id,
            "private-vault-token-secret",
            "--json",
        ])
        .assert()
        .success();

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--merge-preview", "--json"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    for forbidden in [
        "thorough error handling",    // content body marker
        "integration tests",          // content body marker
        "near-raw-mem0",              // raw_hash marker
        "near-raw-cc",                // raw_hash marker
        "/tmp/mem0/rec-a.md",         // native_path marker
        "/tmp/claude-code/rec-b.md",  // native_path marker
        "private-vault-token-secret", // tag NAME marker
        "raw_hash",                   // field name leak
    ] {
        assert!(
            !stdout.contains(forbidden),
            "merge-preview must not leak {forbidden:?}: {stdout}"
        );
    }
}

/// `--merge-preview` without `--mode near` errors with a clear
/// pointer at the right mode.
#[test]
fn dedupe_merge_preview_requires_near_mode() {
    let data = tmp_dir();
    init_db(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--merge-preview"])
        .output()
        .unwrap();
    assert!(!out.status.success(), "must reject exact + merge-preview");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("--mode near"),
        "error must point at near: {stderr}"
    );
}

/// `--mode near --merge-preview --csv` errors: nested decision
/// draft doesn't flatten safely.
#[test]
fn dedupe_near_merge_preview_rejects_csv() {
    let data = tmp_dir();
    init_db(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--merge-preview", "--csv"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("mutually exclusive"),
        "error must surface the conflict: {stderr}"
    );
}

/// Default near dedupe (no `--merge-preview`) wire shape stays
/// back-compat — no `merge_preview` key on groups, top-level
/// `merge_preview_included: false`.
#[test]
fn dedupe_near_default_json_back_compat_without_merge_preview() {
    let data = tmp_dir();
    init_db(data.path());
    seed_near_dup_paraphrases(data.path());

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["dedupe", "--mode", "near", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["merge_preview_included"], false);
    let group = &v["groups"][0];
    assert!(
        group.get("merge_preview").is_none(),
        "no flag → no merge_preview block: {group}"
    );
}
