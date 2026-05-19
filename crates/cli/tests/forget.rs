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
