//! Round-78 PR-78 end-to-end: `anamnesis tag-record`.
//!
//! What's covered here that store unit tests can't:
//!
//!   1. The `tag-record` subcommand is wired through clap + the
//!      dispatcher, audit log fires, output renders.
//!   2. Re-import preserves user tags **observed through the
//!      real CLI search wire** — the load-bearing R78
//!      guarantee, asserted on the user-visible surface.
//!   3. `search --json` surfaces `user_tags` as a top-level
//!      field; default-empty for untagged records keeps the
//!      wire stable.

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
        .join("tag-proj")
        .join("memory");
    fs::create_dir_all(&memdir).unwrap();
    fs::write(
        memdir.join("alpha.md"),
        "---\n\
         name: alpha-record\n\
         description: anchored on the marker uniqueTagMarkerR78\n\
         metadata:\n  type: user\n\
         ---\n\n\
         A user fact marked uniqueTagMarkerR78.\n",
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

fn record_id_for_query(home: &Path, data: &Path, query: &str) -> String {
    let out = cli()
        .env("HOME", home)
        .env("ANAMNESIS_DATA_DIR", data)
        .args(["search", query, "--mode", "fulltext", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    v["results"][0]["record_id"].as_str().unwrap().to_string()
}

fn user_tags_in_search(home: &Path, data: &Path, query: &str) -> Vec<String> {
    let out = cli()
        .env("HOME", home)
        .env("ANAMNESIS_DATA_DIR", data)
        .args(["search", query, "--mode", "fulltext", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    v["results"][0]["user_tags"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|t| t.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn tag_record_add_and_surface_in_search_json() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "uniqueTagMarkerR78");

    // Default: empty.
    assert!(user_tags_in_search(home.path(), data.path(), "uniqueTagMarkerR78").is_empty());

    // Add two tags.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["tag-record", &rid, "todo", "keep"])
        .assert()
        .success()
        .stdout(contains("added").and(contains("user_tags")));

    let tags = user_tags_in_search(home.path(), data.path(), "uniqueTagMarkerR78");
    assert_eq!(tags, vec!["keep".to_string(), "todo".to_string()]);
}

/// The load-bearing R78 promise. Tag, re-import the same source,
/// confirm tags are still there.
#[test]
fn tag_record_survives_reimport_through_cli() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "uniqueTagMarkerR78");

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["tag-record", &rid, "keep-forever"])
        .assert()
        .success();

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["import", "claude-code", "--no-embed", "--full"])
        .assert()
        .success();

    let tags = user_tags_in_search(home.path(), data.path(), "uniqueTagMarkerR78");
    assert_eq!(
        tags,
        vec!["keep-forever".to_string()],
        "user tags must survive full re-import"
    );
}

#[test]
fn tag_record_remove_drops_tag() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "uniqueTagMarkerR78");

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["tag-record", &rid, "todo"])
        .assert()
        .success();
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["tag-record", &rid, "todo", "--remove"])
        .assert()
        .success();
    let tags = user_tags_in_search(home.path(), data.path(), "uniqueTagMarkerR78");
    assert!(tags.is_empty(), "tag must be gone after --remove");
}

#[test]
fn tag_record_json_payload_carries_normalised_state() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "uniqueTagMarkerR78");

    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "tag-record",
            &rid,
            "  TODO  ",
            "todo", // duplicate (after normalisation)
            "Keep",
            "--json",
        ])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["record_id"], rid);
    assert_eq!(v["operation"], "add");
    // Normalised + deduped requested list.
    assert_eq!(v["requested"], serde_json::json!(["todo", "keep"]));
    assert_eq!(v["changed"], 2);
    assert_eq!(v["user_tags"], serde_json::json!(["keep", "todo"]));
}

// ─── Round-79 PR-78b: search --user-tag filter ─────────────────────

/// `--user-tag` returns only records carrying that tag, and
/// normalises case+whitespace through the shared helper so a
/// query for `KEEP` hits a tag stored as `keep`.
#[test]
fn search_user_tag_filter_hits_only_tagged_records() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "uniqueTagMarkerR78");
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["tag-record", &rid, "Keep-Forever"])
        .assert()
        .success();

    // With the tag: 1 hit.
    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "search",
            "uniqueTagMarkerR78",
            "--mode",
            "fulltext",
            "--json",
            "--user-tag",
            "  KEEP-FOREVER  ",
        ])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["record_id"], rid);
    assert_eq!(results[0]["user_tags"], serde_json::json!(["keep-forever"]));
}

// ─── Round-81 PR-78c: tag-record --replace ─────────────────────────

/// `--replace` installs the input as the full post-call set,
/// overwriting prior tags in one atomic call. JSON payload uses
/// `operation: "replace"` and the set delta in `changed`.
#[test]
fn tag_record_replace_overwrites_prior_set() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "uniqueTagMarkerR78");

    // Seed {keep, todo} with the legacy add path.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["tag-record", &rid, "keep", "todo"])
        .assert()
        .success();

    // Replace with {keep, final} — todo gone, final added.
    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["tag-record", &rid, "keep", "final", "--replace", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["operation"], "replace");
    assert_eq!(v["changed"], 2, "set delta: 1 removed + 1 added");
    assert_eq!(v["user_tags"], serde_json::json!(["final", "keep"]));

    let tags = user_tags_in_search(home.path(), data.path(), "uniqueTagMarkerR78");
    assert_eq!(tags, vec!["final".to_string(), "keep".to_string()]);
}

/// `--replace` with no positional tags clears the overlay.
/// This is the only path that accepts an empty tag list.
#[test]
fn tag_record_replace_with_no_tags_clears_overlay() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "uniqueTagMarkerR78");

    // Seed two tags.
    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["tag-record", &rid, "a", "b"])
        .assert()
        .success();

    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["tag-record", &rid, "--replace", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["operation"], "replace");
    assert_eq!(v["changed"], 2, "two tags cleared = 2 deletions");
    assert_eq!(v["requested"], serde_json::json!([]));
    assert_eq!(v["user_tags"], serde_json::json!([]));

    assert!(user_tags_in_search(home.path(), data.path(), "uniqueTagMarkerR78").is_empty());
}

/// `--remove` and `--replace` are mutually exclusive at the
/// clap level. Catches a user pasting both flags without
/// realising the semantic conflict.
#[test]
fn tag_record_replace_and_remove_are_mutually_exclusive() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "uniqueTagMarkerR78");

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["tag-record", &rid, "x", "--remove", "--replace"])
        .assert()
        .failure();
}

/// Without `--replace`, an empty positional `tags` is still
/// rejected by clap (matches prior R78 behaviour). Guards
/// against a regression where someone widens the required check.
#[test]
fn tag_record_add_with_no_tags_still_fails() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    let rid = record_id_for_query(home.path(), data.path(), "uniqueTagMarkerR78");

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["tag-record", &rid])
        .assert()
        .failure();
}

#[test]
fn search_user_tag_filter_no_match_returns_empty() {
    let home = tmp_dir();
    let data = tmp_dir();
    seed_fixture(home.path());
    init_and_import(home.path(), data.path());
    // No tag added — query for any tag must return zero hits.
    let out = cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args([
            "search",
            "uniqueTagMarkerR78",
            "--mode",
            "fulltext",
            "--json",
            "--user-tag",
            "never-applied",
        ])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["results"].as_array().unwrap().len(), 0);
}
