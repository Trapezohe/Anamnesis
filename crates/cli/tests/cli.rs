//! Integration tests for the `anamnesis` CLI.
//!
//! These exercise the binary end-to-end via assert_cmd. They are scoped
//! to scenarios that don't require downloading an embedding model so
//! they're fast on every `cargo test`. Full E2E with real embeddings
//! lives in Task #17.

use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;

fn cli() -> Command {
    Command::cargo_bin("anamnesis").expect("cargo bin")
}

fn tmp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

#[test]
fn init_creates_db_and_sets_active_model() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success()
        .stdout(contains("initialized at").and(contains("local:default:1")));
    assert!(dir.path().join("anamnesis.sqlite").exists());
}

#[test]
fn init_with_explicit_model_sets_it_active() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init", "--model", "en"])
        .assert()
        .success()
        .stdout(contains("local:en:1"));
}

#[test]
fn init_rejects_unknown_model() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init", "--model", "garbage"])
        .assert()
        .failure()
        .stderr(contains("unknown model key"));
}

#[test]
fn status_before_init_is_friendly() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["status"])
        .assert()
        .success()
        .stdout(contains("no database found"));
}

#[test]
fn status_after_init_prints_counters() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["status"])
        .assert()
        .success()
        .stdout(
            contains("records         : 0")
                .and(contains("chunks          : 0"))
                .and(contains("active model    : local:default:1")),
        );
}

#[test]
fn source_add_then_list_shows_entry() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "source",
            "add",
            "claude-code",
            "--instance",
            "default",
            "--path",
            "/tmp/some/place",
        ])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "list"])
        .assert()
        .success()
        .stdout(
            contains("claude-code")
                .and(contains("default"))
                .and(contains("/tmp/some/place")),
        );
}

#[test]
fn source_remove_drops_entry() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "add", "mem0"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "remove", "mem0"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "list"])
        .assert()
        .success()
        .stdout(contains("no sources registered"));
}

#[test]
fn model_list_shows_five_curated_with_active() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["model", "list"])
        .assert()
        .success()
        .stdout(
            contains("default")
                .and(contains("tiny"))
                .and(contains("en"))
                .and(contains("multi-strong"))
                .and(contains("cloud-voyage"))
                .and(contains("yes")), // marker on the active row
        );
}

#[test]
fn discover_returns_friendly_message_when_no_sources_found() {
    let dir = tmp_dir();
    // Point HOME at the empty tempdir → no .claude/projects exists.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .env("HOME", dir.path())
        .args(["discover"])
        .assert()
        .success()
        .stdout(contains("no known memory sources found"));
}

#[test]
fn import_rejects_unknown_adapter() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["import", "made-up-adapter", "--no-embed"])
        .assert()
        .failure()
        .stderr(contains("not wired"));
}

#[test]
fn import_supports_mem0_via_path_override() {
    use rusqlite::Connection;
    let dir = tmp_dir();
    let db = dir.path().join("mem0-test.sqlite");
    let conn = Connection::open(&db).unwrap();
    conn.execute_batch(
        "CREATE TABLE memories(id TEXT PRIMARY KEY, memory TEXT NOT NULL, user_id TEXT);",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memories(id, memory, user_id) VALUES('a', 'imported via cli mem0 path', 'u1')",
        [],
    )
    .unwrap();
    drop(conn);

    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "import",
            "mem0",
            "--no-embed",
            "--path",
            db.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("import done").and(contains("1 upserted")));
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["search", "imported via cli mem0", "--mode", "fulltext"])
        .assert()
        .success()
        .stdout(contains("imported via cli mem0").or(contains("mem0")));
}

#[test]
fn discover_lists_mem0_when_db_exists() {
    use rusqlite::Connection;
    let home = tmp_dir();
    let data = tmp_dir();
    let mem0_dir = home.path().join(".mem0");
    std::fs::create_dir_all(&mem0_dir).unwrap();
    let db = mem0_dir.join("db.sqlite");
    let conn = Connection::open(&db).unwrap();
    conn.execute_batch("CREATE TABLE memories(id TEXT PRIMARY KEY, memory TEXT NOT NULL);")
        .unwrap();
    conn.execute(
        "INSERT INTO memories(id, memory) VALUES('x','one'),('y','two')",
        [],
    )
    .unwrap();
    drop(conn);

    cli()
        .env("HOME", home.path())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .args(["discover"])
        .assert()
        .success()
        .stdout(
            contains("mem0")
                .and(contains("high"))
                .and(contains("2 row")),
        );
}

#[test]
fn search_with_empty_db_prints_no_results() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["search", "anything", "--mode", "fulltext"])
        .assert()
        .success()
        .stdout(contains("no results"));
}

#[test]
fn search_json_mode_emits_parseable_json() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    let output = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["search", "nothing here", "--mode", "fulltext", "--json"])
        .output()
        .expect("run cli");
    assert!(output.status.success());
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).expect("valid json");
    assert_eq!(parsed["query"], "nothing here");
    assert_eq!(parsed["mode"], "fulltext");
    assert!(parsed["results"].is_array());
}
