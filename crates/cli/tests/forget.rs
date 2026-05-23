//! Round-72 PR-72a end-to-end: `anamnesis forget <record_id>`.
//!
//! Two paths a unit test in `anamnesis-store` can't cover on its
//! own:
//!
//!   1. `forget` is exposed via the real binary (clap parsing,
//!      audit log integration, JSON / human rendering).
//!   2. The "stay forgotten" guarantee survives a *real* import
//!      cycle through the claude-code adapter — proving the
//!      tombstone gate actually fires when the importer reaches
//!      the store layer.
//!
//! Fixture: one claude-code memory file with a unique marker
//! phrase. We import it, look up the resulting record_id, forget
//! it, confirm search returns nothing, re-import the same source,
//! and confirm search *still* returns nothing.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;

fn cli() -> Command {
    Command::cargo_bin("anamnesis").expect("cargo bin")
}

fn tmp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

fn seed_fixture(home: &Path) {
    let memdir = home
        .join(".claude")
        .join("projects")
        .join("forget-proj")
        .join("memory");
    fs::create_dir_all(&memdir).unwrap();
    fs::write(
        memdir.join("doomed.md"),
        "---\n\
         name: doomed-record\n\
         description: anchored on the unique marker forgetMeChannel\n\
         metadata:\n  type: user\n\
         ---\n\n\
         This memory will be forgotten and must never resurrect: forgetMeChannel.\n",
    )
    .unwrap();
}

fn init_and_import(home: &Path, data: &Path) {
    cli()
        .env("HOME", home)
        .env("ANAMNESIS_DATA_DIR", data)
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("HOME", home)
        .env("ANAMNESIS_DATA_DIR", data)
        .args([
            "source",
            "add",
            "claude-code",
            "--instance",
            "default",
            "--path",
        ])
        .arg(home.join(".claude").join("projects"))
        .assert()
        .success();
    cli()
        .env("HOME", home)
        .env("ANAMNESIS_DATA_DIR", data)
        .args(["import", "claude-code", "--no-embed"])
        .assert()
        .success();
}

/// Pull the `record_id` of the first hit out of `search --json`'s
/// stdout. Lightweight enough to inline rather than import
/// `serde_json` just for tests.
fn record_id_for_query(home: &Path, data: &Path, query: &str) -> String {
    let out = cli()
        .env("HOME", home)
        .env("ANAMNESIS_DATA_DIR", data)
        .args(["search", query, "--mode", "fulltext", "--json"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    v["results"][0]["record_id"]
        .as_str()
        .expect("hit")
        .to_string()
}

fn search_hit_count(home: &Path, data: &Path, query: &str) -> usize {
    let out = cli()
        .env("HOME", home)
        .env("ANAMNESIS_DATA_DIR", data)
        .args(["search", query, "--mode", "fulltext", "--json"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    v["results"].as_array().map(|a| a.len()).unwrap_or(0)
}

#[test]
fn forget_record_tombstones_and_suppresses_reimport() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());

    // Pre-forget: search lands on the doomed record.
    assert_eq!(
        search_hit_count(home.path(), data.path(), "forgetMeChannel"),
        1,
        "doomed record should be searchable before forget"
    );
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");

    // Forget. Exit 0 + human-readable acknowledgement.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid, "--reason", "test wipe"])
        .assert()
        .success()
        .stdout(contains("forgotten").and(contains("claude-code")));

    // Post-forget: search returns nothing.
    assert_eq!(
        search_hit_count(home.path(), data.path(), "forgetMeChannel"),
        0,
        "record must be search-invisible after forget"
    );

    // Re-import the same source. The importer must respect the
    // tombstone — the search result stays empty.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["import", "claude-code", "--no-embed", "--full"])
        .assert()
        .success();
    assert_eq!(
        search_hit_count(home.path(), data.path(), "forgetMeChannel"),
        0,
        "tombstoned record must NOT resurrect on re-import"
    );
}

#[test]
fn forget_record_unknown_id_exits_nonzero() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", "no-such-record-id"])
        .assert()
        .failure()
        .stderr(contains("nothing to forget"));
}

#[test]
fn forget_record_second_call_is_idempotent_success() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid])
        .assert()
        .success();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid])
        .assert()
        .success()
        .stdout(contains("already forgotten"));
}

// ─── Round-74 PR-74: list-forgotten ─────────────────────────────────

/// Default `list-forgotten --json` includes the tombstone but
/// *redacts* `native_path`, `raw_hash`, and `reason`. Critical
/// behaviour — keeps the audit view safe for casual operator use.
#[test]
fn list_forgotten_default_json_redacts_sensitive_fields() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid, "--reason", "secretReasonMarkerCli"])
        .assert()
        .success();

    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--json"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["count"], 1);
    assert_eq!(v["sensitive_included"], false);
    let row = &v["rows"][0];
    assert_eq!(row["record_id"], rid);
    assert_eq!(row["has_reason"], true);
    assert_eq!(row["has_native_path"], true);
    assert!(row.get("reason").is_none(), "reason must be absent: {row}");
    assert!(row.get("native_path").is_none());
    assert!(row.get("raw_hash").is_none());
    assert!(
        !stdout.contains("secretReasonMarkerCli"),
        "reason marker must not leak into redacted output"
    );

    // Round 126 (PR-78au): top-level redacted summary on
    // `list-forgotten --json`. Mirrors MCP R117. NEVER reads
    // reason/native_path/raw_hash.
    let summary = v["summary"]
        .as_str()
        .expect("list-forgotten --json must carry top-level `summary`");
    assert!(
        summary.contains("1 tombstone row(s) returned"),
        "summary must declare count: {summary}"
    );
    assert!(
        summary.contains("source filter: all sources"),
        "default no-filter summary must say `all sources`: {summary}"
    );
    assert!(
        summary.contains("instance filter: all instances"),
        "default no-filter summary must say `all instances`: {summary}"
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
        !summary.contains("secretReasonMarkerCli"),
        "summary must not leak reason: {summary}"
    );
}

#[test]
fn list_forgotten_include_sensitive_reveals_fields() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid, "--reason", "secretReasonMarkerCli"])
        .assert()
        .success();

    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--json", "--include-sensitive"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["sensitive_included"], true);
    let row = &v["rows"][0];
    assert_eq!(row["reason"], "secretReasonMarkerCli");
    assert!(row["native_path"].is_string());
    assert!(row["raw_hash"].is_string());
}

// ─── Round-75 PR-75: unforget ───────────────────────────────────────

/// Full lifecycle: import → search hits → forget → search empty →
/// unforget → search STILL empty → re-import → search hits again.
/// The "STILL empty after unforget" step is load-bearing: it
/// proves `unforget` removes the gate without resurrecting the
/// record on its own.
#[test]
fn unforget_lifts_suppression_but_requires_reimport_to_resurrect() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());

    assert_eq!(
        search_hit_count(home.path(), data.path(), "forgetMeChannel"),
        1
    );
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid])
        .assert()
        .success();
    assert_eq!(
        search_hit_count(home.path(), data.path(), "forgetMeChannel"),
        0
    );

    // Unforget: tombstone gate removed.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["unforget", &rid])
        .assert()
        .success()
        .stdout(contains("unforgotten").and(contains("NOT resurrected")));

    // Still empty — unforget alone doesn't bring the record back.
    assert_eq!(
        search_hit_count(home.path(), data.path(), "forgetMeChannel"),
        0,
        "unforget must not resurrect the record by itself",
    );

    // Re-import: the source's own data brings the record back.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["import", "claude-code", "--no-embed", "--full"])
        .assert()
        .success();
    assert_eq!(
        search_hit_count(home.path(), data.path(), "forgetMeChannel"),
        1,
        "after unforget + re-import the record must be searchable again",
    );
}

#[test]
fn unforget_removes_tombstone_from_list_forgotten() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid])
        .assert()
        .success();

    let pre = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&pre.stdout).unwrap();
    assert_eq!(v["count"], 1);

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["unforget", &rid])
        .assert()
        .success();

    let post = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&post.stdout).unwrap();
    assert_eq!(v["count"], 0);
}

#[test]
fn unforget_unknown_id_exits_nonzero() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["unforget", "no-such-id"])
        .assert()
        .failure()
        .stderr(contains("nothing to unforget"));
}

#[test]
fn unforget_json_payload_makes_resurrection_semantics_explicit() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid])
        .assert()
        .success();
    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["unforget", &rid, "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["status"], "unforgotten");
    assert_eq!(v["record_id"], rid);
    assert_eq!(v["record_resurrected"], false);
    assert_eq!(v["requires_reimport"], true);
}

// ─── Round-95 PR-78q: unforget --dry-run ───────────────────────────

/// `--dry-run` reports the tombstone without removing it. The
/// real `unforget` would write 1 audit entry; dry-run writes 0.
#[test]
fn unforget_dry_run_reports_tombstone_without_mutating() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid])
        .assert()
        .success();

    let audit_path = data.path().join("audit.log");
    let audit_before = std::fs::metadata(&audit_path).ok().map(|m| m.len());

    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["unforget", &rid, "--dry-run", "--json"])
        .output()
        .unwrap();
    let audit_after = std::fs::metadata(&audit_path).ok().map(|m| m.len());
    assert_eq!(
        audit_before, audit_after,
        "dry-run must not append to audit.log"
    );
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["dry_run"], true);
    assert_eq!(v["status"], "would-unforget");
    assert_eq!(v["record_id"], rid);
    assert_eq!(v["record_resurrected"], false);
    assert_eq!(v["requires_reimport"], true);
    assert_eq!(v["would_delete"]["record_tombstones"], 1);
    assert_eq!(v["would_insert"]["audit_log_entries"], 1);

    // Tombstone still present.
    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["count"], 1, "dry-run must not delete the tombstone");
}

/// `--dry-run` on an unknown id exits non-zero — typo loud.
#[test]
fn unforget_dry_run_unknown_id_exits_nonzero() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["unforget", "phantom-id", "--dry-run"])
        .assert()
        .failure();
}

#[test]
fn list_forgotten_empty_store_says_so() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten"])
        .assert()
        .success()
        .stdout(contains("no forgotten records"));
}

// ─── Round-90 PR-78l: list-forgotten --include-counts ───────────────

/// Default `list-forgotten --json` has no `counts` field —
/// back-compat with every existing R74/R75 consumer.
#[test]
fn list_forgotten_default_json_has_no_counts_block() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid])
        .assert()
        .success();
    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["count"], 1);
    assert!(
        v.get("counts").is_none(),
        "default list-forgotten must not carry a counts block; got {v}"
    );
}

/// `--include-counts` attaches `counts.total` + `counts.by_source[]`
/// — operators see "137 tombstones, 120 claude-code + 17 mem0" in
/// one call without paging. Counts respect the same source/instance
/// filter as the row list but reflect the full matching set.
#[test]
fn list_forgotten_include_counts_attaches_total_and_by_source() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid, "--reason", "preview"])
        .assert()
        .success();

    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--json", "--include-counts"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let counts = &v["counts"];
    assert_eq!(counts["total"], 1);
    let by_source = counts["by_source"].as_array().unwrap();
    assert_eq!(by_source.len(), 1);
    assert_eq!(by_source[0]["adapter"], "claude-code");
    assert!(by_source[0]["instance"].is_null());
    assert_eq!(by_source[0]["forgotten_count"], 1);
    // Sensitive fields stay out of counts even when sensitive
    // mode isn't requested.
    let counts_str = serde_json::to_string(counts).unwrap();
    assert!(
        !counts_str.contains("preview"),
        "counts block must not leak forgot reason"
    );
}

// ─── Round-83 PR-78e: forget --dry-run ──────────────────────────────

/// `--dry-run` reports the cascade without mutating: the record
/// is still searchable, list-forgotten count stays 0, and
/// audit.log carries no new entry.
#[test]
fn forget_dry_run_reports_cascade_without_mutating() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());

    // Baseline: forgetMeChannel still hits.
    let pre_hits = search_hit_count(home.path(), data.path(), "forgetMeChannel");
    assert_eq!(pre_hits, 1);
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");

    // Capture audit.log size **immediately** before the dry-run
    // so subsequent `search` / `list-forgotten` calls don't get
    // counted against the dry-run's promise.
    let audit_path = data.path().join("audit.log");
    let audit_before = std::fs::metadata(&audit_path).ok().map(|m| m.len());

    // Dry-run.
    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid, "--dry-run", "--json", "--reason", "preview"])
        .output()
        .unwrap();
    // Capture audit.log immediately after the dry-run binary
    // returns — this is the only window where "dry-run did not
    // append" is a well-defined statement.
    let audit_after = std::fs::metadata(&audit_path).ok().map(|m| m.len());
    assert_eq!(
        audit_before, audit_after,
        "dry-run must not append to audit.log"
    );

    assert!(out.status.success(), "dry-run must succeed on live record");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["dry_run"], true);
    assert_eq!(v["status"], "would-forget");
    assert_eq!(v["record_id"], rid);
    assert_eq!(v["reason"], "preview");
    assert_eq!(v["would_delete"]["records"], 1);
    assert!(v["would_delete"]["record_chunks"].as_u64().unwrap() >= 1);
    assert_eq!(v["would_insert"]["record_tombstones"], 1);
    assert_eq!(v["would_insert"]["audit_log_entries"], 1);

    // Mutation guard: the record is still searchable + no
    // tombstone landed.
    assert_eq!(
        search_hit_count(home.path(), data.path(), "forgetMeChannel"),
        1,
        "dry-run must not delete the record"
    );
    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        v["count"], 0,
        "dry-run must not write a tombstone (list-forgotten count must stay 0)"
    );
}

/// `--dry-run` on an already-forgotten id reports
/// `already-forgotten` and exits 0 — same idempotency contract
/// as the real path. Mutation guard: no second tombstone.
#[test]
fn forget_dry_run_on_already_forgotten_is_idempotent() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");

    // Real forget first.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid])
        .assert()
        .success();
    // Now dry-run should report already-forgotten.
    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid, "--dry-run", "--json"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["dry_run"], true);
    assert_eq!(v["status"], "already-forgotten");
    // No additional cascade work would happen.
    assert_eq!(v["would_delete"]["records"], 0);
    assert_eq!(v["would_insert"]["record_tombstones"], 0);
}

/// `--dry-run` on a never-existed id exits non-zero — typo loud.
#[test]
fn forget_dry_run_not_found_exits_nonzero() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", "phantom-id-doesnt-exist", "--dry-run"])
        .assert()
        .failure();
}

// ─── Round-106 PR-78ab: list-forgotten --csv ────────────────────────
// Re-added after the R105 Windows stack-overflow fix landed in the
// same PR. CSV is the redacted-summary form; mutually exclusive
// with --json (clap) and --include-sensitive / --include-counts
// (runtime check).

/// Empty store still emits the fixed header so downstream
/// scripts can branch uniformly. Mirrors R91 `audit tail --csv`.
#[test]
fn list_forgotten_csv_empty_emits_header_only() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());

    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--csv"])
        .output()
        .unwrap();
    assert!(out.status.success(), "csv on empty store must exit 0");
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        stdout.trim(),
        "record_id,adapter,instance,native_id,forgotten_at,has_reason,has_native_path"
    );
}

/// `--csv` emits the redacted summary row even when the
/// tombstone carries a `reason`. Critical privacy contract:
/// `reason`, `native_path`, and `raw_hash` NEVER appear in CSV.
#[test]
fn list_forgotten_csv_returns_redacted_summary_rows() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "forgetMeChannel");
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["forget", &rid, "--reason", "secretCsvCanary106"])
        .assert()
        .success();

    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--csv"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "header + 1 row: {stdout}");
    assert_eq!(
        lines[0],
        "record_id,adapter,instance,native_id,forgotten_at,has_reason,has_native_path"
    );
    assert!(
        lines[1].contains(&rid),
        "csv row must reference the forgotten id: {}",
        lines[1]
    );
    assert!(
        !stdout.contains("secretCsvCanary106"),
        "csv must never leak `reason`: {stdout}"
    );
}

/// `--csv --include-sensitive` is rejected at runtime — CSV is
/// the redacted-summary form by design.
#[test]
fn list_forgotten_csv_and_include_sensitive_are_mutually_exclusive() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--csv", "--include-sensitive"])
        .assert()
        .failure();
}

/// `--csv --json` (clap-rejected) and `--csv --include-counts`
/// (runtime-rejected). CSV is flat redacted rows — no nested
/// counts block, no structured form.
#[test]
fn list_forgotten_csv_and_json_are_mutually_exclusive() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--csv", "--json"])
        .assert()
        .failure();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["list-forgotten", "--csv", "--include-counts"])
        .assert()
        .failure();
}
