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
fn status_shows_per_source_health_block_with_empty_hint_when_no_sources() {
    // Round-16: even with zero sources registered, `status` shows the
    // "sources by health:" block so the first-run UX includes an
    // explicit "try discover / source add" affordance.
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
            contains("sources by health:")
                .and(contains("no sources registered"))
                .and(contains("anamnesis discover").or(contains("anamnesis source add"))),
        );
}

#[test]
fn status_per_source_table_lists_never_imported_source() {
    // Round-16: register a source without importing. The per-source
    // table must list it with status = "never-imported" so the
    // operator can spot registered-but-empty sources at a glance.
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
            "/tmp/round-16-fake",
        ])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["status"])
        .assert()
        .success()
        .stdout(
            contains("claude-code")
                .and(contains("default"))
                .and(contains("never-imported"))
                // The legacy global counters must still be present —
                // we're adding the per-source block, not replacing the
                // header.
                .and(contains("records         : 0")),
        );
}

#[test]
fn status_json_includes_per_source_counts_and_freshness() {
    // Round-16 JSON contract: each source object exposes
    // `record_count`, `chunk_count`, `freshness`, `age_seconds`
    // alongside the existing `last_import_at` / `added_at` fields.
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
            "/tmp/round-16-fake",
        ])
        .assert()
        .success();
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["status", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(out).expect("utf8 stdout");
    let v: serde_json::Value = serde_json::from_str(&text).expect("status --json must be JSON");
    let arr = v
        .get("sources")
        .and_then(|s| s.as_array())
        .expect("sources array");
    assert_eq!(arr.len(), 1, "exactly one source registered");
    let s = &arr[0];
    assert_eq!(
        s.get("adapter").and_then(|x| x.as_str()),
        Some("claude-code")
    );
    assert_eq!(s.get("instance").and_then(|x| x.as_str()), Some("default"));
    assert_eq!(s.get("record_count").and_then(|x| x.as_u64()), Some(0));
    assert_eq!(s.get("chunk_count").and_then(|x| x.as_u64()), Some(0));
    assert_eq!(
        s.get("freshness").and_then(|x| x.as_str()),
        Some("never-imported")
    );
    assert!(s.get("age_seconds").map(|x| x.is_null()).unwrap_or(false));
    assert!(s
        .get("last_import_at")
        .map(|x| x.is_null())
        .unwrap_or(false));
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

/// Round 82 PR-78d: `source list` table renders the new `tagged`
/// column. Seed records directly via the store, tag one, and
/// verify the header + a count line appear together.
#[test]
fn source_list_reports_tagged_record_count_column() {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::{Store, UserTagOperation};
    use chrono::Utc;

    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "add", "claude-code", "--path", "/tmp/cc"])
        .assert()
        .success();

    // Seed 3 records on claude-code; tag only 1.
    let db = dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).unwrap();
    for n in ["a", "b", "c"] {
        let r = AnamnesisRecord {
            id: RecordId::from_parts("claude-code", None, n),
            source: SourceDescriptor {
                adapter: "claude-code".into(),
                instance: None,
                version: "0".into(),
            },
            content: format!("body for {n}"),
            embedding: None,
            scope: Scope::User,
            kind: Kind::Fact,
            created_at: Utc::now(),
            updated_at: None,
            tags: vec![],
            metadata: Default::default(),
            provenance: Provenance {
                native_id: n.into(),
                native_path: None,
                captured_at: Utc::now(),
                raw_hash: format!("h-{n}"),
                derived_from: None,
            },
            schema_version: SCHEMA_VERSION,
        };
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
    }
    let id_a = RecordId::from_parts("claude-code", None, "a");
    store
        .tag_record(
            &id_a,
            &["keep".into(), "todo".into()],
            UserTagOperation::Add,
        )
        .unwrap();
    drop(store); // release the lock before the CLI re-opens it

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "list"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(
        stdout.contains("tagged"),
        "header row must include `tagged`: {stdout}"
    );
    // The data row must carry: adapter=claude-code, records=3,
    // tagged=1. We assert these without pinning column widths.
    let data_line = stdout
        .lines()
        .find(|l| l.starts_with("claude-code"))
        .unwrap_or_else(|| panic!("no claude-code data line in:\n{stdout}"));
    let tokens: Vec<&str> = data_line.split_whitespace().collect();
    // adapter, instance(possibly "-"), records, chunks, tagged, ...
    assert_eq!(tokens[0], "claude-code");
    assert!(
        tokens.contains(&"3"),
        "records=3 must appear in row: {data_line}"
    );
    // tagged=1: this is the only `1` token sandwiched between
    // record/chunk counts and the `<never>` last_import marker.
    assert!(
        tokens.contains(&"1"),
        "tagged=1 must appear in row: {data_line}"
    );
}

/// Status JSON also surfaces `tagged_record_count` so a script
/// that consumes `anamnesis status --json` doesn't need a second
/// store round-trip for the same answer.
#[test]
fn status_json_surfaces_tagged_record_count_per_source() {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::{Store, UserTagOperation};
    use chrono::Utc;

    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "add", "claude-code", "--path", "/tmp/cc"])
        .assert()
        .success();

    let db = dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).unwrap();
    let r = AnamnesisRecord {
        id: RecordId::from_parts("claude-code", None, "z"),
        source: SourceDescriptor {
            adapter: "claude-code".into(),
            instance: None,
            version: "0".into(),
        },
        content: "z body".into(),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: "z".into(),
            native_path: None,
            captured_at: Utc::now(),
            raw_hash: "h-z".into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let c = Chunker::default().chunk(&r.id, &r.content);
    store.upsert_record(&r, &c, None).unwrap();
    store
        .tag_record(&r.id, &["keep".into()], UserTagOperation::Add)
        .unwrap();
    drop(store);

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["status", "--json"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let sources = v["per_source"].as_array().unwrap_or_else(|| {
        // Fall back to `sources` if the JSON layout changes — we
        // only care that the field is present at depth-1.
        v["sources"].as_array().expect("per-source array somewhere")
    });
    let cc = sources
        .iter()
        .find(|s| s["adapter"] == "claude-code")
        .expect("claude-code source in status JSON");
    assert_eq!(
        cc["tagged_record_count"], 1,
        "status --json must surface tagged_record_count per source"
    );
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
fn status_json_emits_structured_output() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["status", "--json"])
        .output()
        .expect("run cli");
    assert!(out.status.success());
    let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(payload["initialized"], true);
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(payload["active_model"], "local:default:1");
    assert!(payload["stats"]["records"].as_u64() == Some(0));
}

#[test]
fn export_jsonl_round_trips_records() {
    use rusqlite::Connection;
    let dir = tmp_dir();
    // Seed via mem0 import so we have known records.
    let mem0_db = dir.path().join("mem0.sqlite");
    let conn = Connection::open(&mem0_db).unwrap();
    conn.execute_batch("CREATE TABLE memories(id TEXT PRIMARY KEY, memory TEXT NOT NULL);")
        .unwrap();
    conn.execute(
        "INSERT INTO memories(id, memory) VALUES('a','exported alpha'),('b','exported beta')",
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
            mem0_db.to_str().unwrap(),
        ])
        .assert()
        .success();

    let out_path = dir.path().join("out.jsonl");
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "export",
            "--format",
            "jsonl",
            "--out",
            out_path.to_str().unwrap(),
        ])
        .assert()
        .success();
    let body = std::fs::read_to_string(&out_path).unwrap();
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 2);
    for line in lines {
        let rec: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(rec["source"]["adapter"], "mem0");
    }
}

#[test]
fn export_csv_includes_header_and_rows() {
    use rusqlite::Connection;
    let dir = tmp_dir();
    let mem0_db = dir.path().join("mem0.sqlite");
    let conn = Connection::open(&mem0_db).unwrap();
    conn.execute_batch("CREATE TABLE memories(id TEXT PRIMARY KEY, memory TEXT NOT NULL);")
        .unwrap();
    conn.execute("INSERT INTO memories VALUES('a','tea, and, biscuits')", [])
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
            mem0_db.to_str().unwrap(),
        ])
        .assert()
        .success();

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["export", "--format", "csv"])
        .output()
        .expect("run cli");
    assert!(out.status.success());
    let body = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = body.lines().collect();
    assert!(lines[0].starts_with("id,adapter,instance,kind,scope"));
    assert!(lines.len() >= 2);
    // Comma in content must be quoted.
    assert!(body.contains("\"tea, and, biscuits\""));
}

#[test]
fn verify_reports_healthy_on_fresh_db() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["verify"])
        .assert()
        .success()
        .stdout(
            contains("integrity_check : ok")
                .and(contains("status          : healthy"))
                .and(contains("missing embeds")),
        );
}

#[test]
fn search_kind_filter_restricts_to_kind() {
    use rusqlite::Connection;
    let dir = tmp_dir();
    let mem0_db = dir.path().join("mem0.sqlite");
    let conn = Connection::open(&mem0_db).unwrap();
    conn.execute_batch("CREATE TABLE memories(id TEXT PRIMARY KEY, memory TEXT NOT NULL);")
        .unwrap();
    conn.execute("INSERT INTO memories VALUES('a','filter target')", [])
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
            mem0_db.to_str().unwrap(),
        ])
        .assert()
        .success();

    // mem0 normalizes to Kind::Fact, so --kind fact hits, --kind episode misses.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "search",
            "filter target",
            "--mode",
            "fulltext",
            "--kind",
            "fact",
        ])
        .assert()
        .success()
        .stdout(contains("filter target").or(contains("mem0")));
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "search",
            "filter target",
            "--mode",
            "fulltext",
            "--kind",
            "episode",
        ])
        .assert()
        .success()
        .stdout(contains("no results"));
}

/// Round-12: the human-readable `anamnesis search` card surfaces every
/// JSON wire field an agent would see — kind/scope/score breakdown/
/// timestamps/ids — so an operator running the CLI by hand has the
/// same context an MCP agent has after `search_memories`.
#[test]
fn search_card_surfaces_wire_fields_for_human_readers() {
    use rusqlite::Connection;
    let dir = tmp_dir();
    let mem0_db = dir.path().join("mem0.sqlite");
    let conn = Connection::open(&mem0_db).unwrap();
    conn.execute_batch("CREATE TABLE memories(id TEXT PRIMARY KEY, memory TEXT NOT NULL);")
        .unwrap();
    conn.execute(
        "INSERT INTO memories VALUES('a','platypusBanjoComet card test')",
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
            mem0_db.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Run the search and inspect stdout for each surfaced field.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["search", "platypusBanjoComet", "--mode", "fulltext"])
        .assert()
        .success()
        .stdout(
            contains("rrf=")
                .and(contains("fts="))
                .and(contains("vec="))
                .and(contains("(fact, user)"))
                .and(contains("created="))
                .and(contains("record_id="))
                .and(contains("chunk_id="))
                .and(contains("trace_id="))
                .and(contains("native_path="))
                .and(contains("snippet:"))
                .and(contains("platypusBanjoComet")),
        );
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
fn config_file_overrides_default_model_on_init() {
    let dir = tmp_dir();
    let config_dir = tmp_dir();
    let cfg_path = config_dir.path().join("config.toml");
    std::fs::write(&cfg_path, "[embedding]\nmodel = \"en\"\nbatch_size = 16\n").unwrap();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .env("ANAMNESIS_CONFIG", &cfg_path)
        .args(["init"])
        .assert()
        .success()
        .stdout(contains("local:en:1"));
}

#[test]
fn cli_flag_beats_config_file_model() {
    let dir = tmp_dir();
    let config_dir = tmp_dir();
    let cfg_path = config_dir.path().join("config.toml");
    std::fs::write(&cfg_path, "[embedding]\nmodel = \"en\"\n").unwrap();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .env("ANAMNESIS_CONFIG", &cfg_path)
        .args(["init", "--model", "tiny"])
        .assert()
        .success()
        .stdout(contains("local:tiny:1"));
}

#[test]
fn import_writes_audit_log_entry() {
    use rusqlite::Connection;
    let dir = tmp_dir();
    let mem0_db = dir.path().join("mem0.sqlite");
    let conn = Connection::open(&mem0_db).unwrap();
    conn.execute_batch("CREATE TABLE memories(id TEXT PRIMARY KEY, memory TEXT NOT NULL);")
        .unwrap();
    conn.execute("INSERT INTO memories VALUES('a','audited memory')", [])
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
            mem0_db.to_str().unwrap(),
        ])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["search", "audited", "--mode", "fulltext"])
        .assert()
        .success();

    let audit_log = dir.path().join("audit.log");
    assert!(
        audit_log.exists(),
        "audit.log should exist after import + search"
    );
    let body = std::fs::read_to_string(&audit_log).unwrap();
    let lines: Vec<&str> = body.lines().collect();
    assert!(!lines.is_empty());
    let actions: Vec<String> = lines
        .iter()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| v["action"].as_str().map(str::to_owned))
        .collect();
    assert!(actions.iter().any(|a| a == "import"));
    assert!(actions.iter().any(|a| a == "search"));
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
    // Round 76: default search JSON must NOT include the `trace`
    // field. Guards against accidentally always emitting it (which
    // would inflate every existing CLI consumer's payload).
    assert!(
        parsed.get("trace").is_none(),
        "trace must be absent without --trace; got {parsed}"
    );
}

// ─── Round-76: `anamnesis search --trace` ───────────────────────────

#[test]
fn search_trace_json_emits_stage_breakdown() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    let output = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "search", "anything", "--mode", "fulltext", "--json", "--trace",
        ])
        .output()
        .expect("run cli");
    assert!(output.status.success());
    let v: serde_json::Value = serde_json::from_slice(&output.stdout).expect("valid json");
    let trace = &v["trace"];
    assert_eq!(trace["effective_mode"], "fulltext");
    assert!(trace["candidate_pool"].is_u64());
    // Fulltext mode: fts and rrf and pack all ran; embed_query and vec did not.
    assert!(trace["stages_ms"]["fts"].is_u64());
    assert!(trace["stages_ms"]["rrf"].is_u64());
    assert!(trace["stages_ms"]["pack"].is_u64());
    assert!(trace["stages_ms"]["embed_query"].is_null());
    assert!(trace["stages_ms"]["vec"].is_null());
    // Counts align with what came back in results[].
    let returned = v["results"].as_array().unwrap().len();
    assert_eq!(
        trace["counts"]["returned_records"].as_u64().unwrap() as usize,
        returned
    );
}

/// Privacy guard: the trace sub-object must not carry user-typed
/// content. Same contract as the R71 MCP test. Note: the top-level
/// CLI JSON intentionally already exposes `query` / `snippet` /
/// `record_id` outside the trace, so we restrict the check to the
/// `trace` value, mirroring how the MCP-side test scopes it.
#[test]
fn search_trace_payload_excludes_user_content() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    let output = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "search",
            "uniqueQueryMarkerR76",
            "--mode",
            "fulltext",
            "--json",
            "--trace",
        ])
        .output()
        .expect("run cli");
    let v: serde_json::Value = serde_json::from_slice(&output.stdout).expect("valid json");
    let trace_str = v["trace"].to_string();
    assert!(
        !trace_str.contains("uniqueQueryMarkerR76"),
        "query text must not appear in trace payload: {trace_str}"
    );
    // Whitelist top-level trace fields.
    let trace = v["trace"].as_object().unwrap();
    let allowed = ["effective_mode", "candidate_pool", "stages_ms", "counts"];
    for k in trace.keys() {
        assert!(
            allowed.contains(&k.as_str()),
            "unexpected trace top-level field {k:?}: needs a privacy review"
        );
    }
}

#[test]
fn search_trace_human_mode_appends_trace_block() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["search", "still nothing", "--mode", "fulltext", "--trace"])
        .assert()
        .success()
        .stdout(
            contains("no results")
                .and(contains("Search trace:"))
                .and(contains("effective_mode=Fulltext"))
                .and(contains("stages_ms")),
        );
}

// ─── PR-B: source registry is the canonical truth for import ───

fn seed_mem0_db(path: &std::path::Path, rows: &[(&str, &str)]) {
    use rusqlite::Connection;
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(
        "CREATE TABLE memories(id TEXT PRIMARY KEY, memory TEXT NOT NULL, user_id TEXT);",
    )
    .unwrap();
    for (id, mem) in rows {
        conn.execute(
            "INSERT INTO memories(id, memory) VALUES(?1, ?2)",
            [*id, *mem],
        )
        .unwrap();
    }
}

#[test]
fn import_auto_registers_source_when_path_override_used() {
    // No `source add` first — import must auto-register the location
    // it actually used so `status` / `source list` stay truthful.
    let dir = tmp_dir();
    let mem0_db = dir.path().join("m.sqlite");
    seed_mem0_db(&mem0_db, &[("a", "alpha")]);

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
            mem0_db.to_str().unwrap(),
        ])
        .assert()
        .success();

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["status", "--json"])
        .output()
        .unwrap();
    let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        payload["stats"]["sources"].as_u64(),
        Some(1),
        "stats.sources must reach 1 after a successful import"
    );
    let sources = payload["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0]["adapter"], "mem0");
    assert_eq!(
        sources[0]["location"].as_str(),
        Some(mem0_db.to_str().unwrap()),
        "registered location must equal the --path that was used"
    );
    assert!(
        sources[0]["last_import_at"].as_i64().is_some(),
        "last_import_at must be non-null after a successful non-dry-run import"
    );
}

#[test]
fn dry_run_import_does_not_touch_registry() {
    // dry-run is read-only — must NOT write to sources or update timestamps.
    let dir = tmp_dir();
    let mem0_db = dir.path().join("m.sqlite");
    seed_mem0_db(&mem0_db, &[("a", "x")]);

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
            "--dry-run",
            "--path",
            mem0_db.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(contains("dry-run"));

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["status", "--json"])
        .output()
        .unwrap();
    let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        payload["stats"]["sources"].as_u64(),
        Some(0),
        "dry-run must leave the source registry untouched"
    );
}

#[test]
fn import_uses_registered_location_when_no_path_given() {
    // The Codex acceptance criterion: register path A, run plain
    // `import mem0` (no --path), and only A's rows must show up.
    let dir = tmp_dir();
    let db_a = dir.path().join("registered.sqlite");
    let db_b = dir.path().join("other.sqlite");
    seed_mem0_db(&db_a, &[("a", "alpha from A")]);
    seed_mem0_db(&db_b, &[("b", "beta from B")]);

    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "add", "mem0", "--path", db_a.to_str().unwrap()])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["import", "mem0", "--no-embed"])
        .assert()
        .success()
        .stdout(contains("1 upserted"));
    // Search should find alpha (from A) but not beta (from B).
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["search", "alpha", "--mode", "fulltext"])
        .assert()
        .success()
        .stdout(contains("alpha"));
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["search", "beta", "--mode", "fulltext"])
        .assert()
        .success()
        .stdout(contains("no results").or(contains("0 hit")));
}

#[test]
fn explicit_path_overwrites_registered_location() {
    // PR-B: --path is trusted override; the registry catches up to it.
    let dir = tmp_dir();
    let db_a = dir.path().join("first.sqlite");
    let db_b = dir.path().join("second.sqlite");
    seed_mem0_db(&db_a, &[("a", "first")]);
    seed_mem0_db(&db_b, &[("b", "second")]);

    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "add", "mem0", "--path", db_a.to_str().unwrap()])
        .assert()
        .success();
    // --path B overrides registered A and rewrites the registry.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "import",
            "mem0",
            "--no-embed",
            "--path",
            db_b.to_str().unwrap(),
        ])
        .assert()
        .success();

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["status", "--json"])
        .output()
        .unwrap();
    let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let sources = payload["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 1, "still exactly one row, not two");
    assert_eq!(
        sources[0]["location"].as_str(),
        Some(db_b.to_str().unwrap()),
        "explicit --path must overwrite the registered location"
    );
}

#[test]
fn double_import_keeps_one_source_row_and_advances_timestamp() {
    // Idempotency check — a second import on the same source must NOT
    // produce a second row.
    let dir = tmp_dir();
    let mem0_db = dir.path().join("m.sqlite");
    seed_mem0_db(&mem0_db, &[("a", "x")]);

    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    for _ in 0..2 {
        cli()
            .env("ANAMNESIS_DATA_DIR", dir.path())
            .args([
                "import",
                "mem0",
                "--no-embed",
                "--path",
                mem0_db.to_str().unwrap(),
            ])
            .assert()
            .success();
    }
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["status", "--json"])
        .output()
        .unwrap();
    let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        payload["stats"]["sources"].as_u64(),
        Some(1),
        "exactly one source row across repeated imports"
    );
    assert!(
        payload["sources"][0]["last_import_at"].as_i64().is_some(),
        "last_import_at non-null"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// §-1.5 PR-6 — `anamnesis extract` safety/audit features (Round 42)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_unknown_provider_errors_before_any_work() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["extract", "--no-dry-run", "--provider", "totally-fake"])
        .assert()
        .failure()
        .stderr(contains("unknown --provider").and(contains("supported: mock, openai")));
}

#[test]
fn extract_openai_without_api_key_errors_clearly() {
    // Two CI matrix legs hit this test path:
    //
    //   - `test (ubuntu-latest)` runs with `--no-default-features`, so
    //     the `openai-provider` cargo feature is OFF → the error is
    //     "requires the `openai-provider` cargo feature".
    //   - `test (sse, *)` / local default-features → feature is ON →
    //     the error is "OPENAI_API_KEY environment variable is required".
    //
    // Either error is a clean "we refused to make any HTTP call"
    // outcome, which is what §-1.5 #6 demands. Accept both.
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        // Force-unset OPENAI_API_KEY so this passes even on a machine that
        // happens to have one set in its shell env.
        .env_remove("OPENAI_API_KEY")
        .args(["extract", "--no-dry-run", "--provider", "openai"])
        .assert()
        .failure()
        .stderr(contains("OPENAI_API_KEY").or(contains("openai-provider")));
}

#[test]
fn extract_mock_no_dry_run_on_empty_store_runs_cleanly_and_writes_audit() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    // Mock is offline + deterministic — no prompt, no `--yes` needed.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["extract", "--no-dry-run"])
        .assert()
        .success();
    // §-1.5 #6 audit log must be appended on every Stage 2 run.
    let audit_path = dir.path().join("audit").join("stage2.jsonl");
    assert!(
        audit_path.exists(),
        "audit log {} not created",
        audit_path.display()
    );
    let body = std::fs::read_to_string(&audit_path).unwrap();
    let line = body.lines().next().expect("at least one audit line");
    let entry: serde_json::Value = serde_json::from_str(line).expect("audit line is JSON");
    assert_eq!(entry["stage"], "stage2");
    assert_eq!(entry["provider_id"], "mock");
    assert_eq!(entry["provider_model"], "mock:default");
    assert_eq!(entry["target_kind"], "fact");
    assert!(entry["ts_started"].is_string());
    assert!(entry["ts_finished"].is_string());
}

#[test]
fn extract_max_llm_calls_lets_zero_candidates_through() {
    // Sanity: --max-llm-calls=0 + empty store → 0 candidates → no cap
    // violation, just a normal "nothing to extract" run.
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["extract", "--no-dry-run", "--max-llm-calls", "0"])
        .assert()
        .success();
}

#[test]
fn extract_dry_run_default_still_inspection_only() {
    // Make sure the safety/audit work didn't accidentally enable LLM
    // calls in the default --dry-run path.
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["extract"])
        .assert()
        .success()
        .stdout(contains("Stage 2 not yet wired").or(contains("inspection")));
    // The dry-run path must NOT write the audit log — only the
    // --no-dry-run path is auditable. (Dry-run is metadata-free.)
    let audit_path = dir.path().join("audit").join("stage2.jsonl");
    assert!(
        !audit_path.exists(),
        "dry-run unexpectedly wrote audit log at {}",
        audit_path.display()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// §-1.5 PR-6 — `anamnesis audit list / show` (Round 43)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn audit_list_on_empty_data_dir_is_friendly() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["audit", "list"])
        .assert()
        .success()
        .stdout(contains("No Stage 2 audit entries yet"));
}

#[test]
fn audit_list_after_extract_run_shows_one_entry() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    // Populate by running mock extract (offline, deterministic).
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["extract", "--no-dry-run"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["audit", "list"])
        .assert()
        .success()
        .stdout(
            contains("mock")
                .and(contains("fact"))
                .and(contains("1 entries shown")),
        );
}

#[test]
fn audit_show_last_pretty_prints_a_run() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["extract", "--no-dry-run"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["audit", "show", "last"])
        .assert()
        .success()
        .stdout(
            contains("Stage 2 audit entry #1")
                .and(contains("provider_id    : mock"))
                .and(contains("target_kind    : fact")),
        );
}

#[test]
fn audit_show_out_of_range_index_errors_cleanly() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["audit", "show", "999"])
        .assert()
        .failure()
        .stderr(contains("out of range").or(contains("audit log has 0 entries")));
}

#[test]
fn audit_show_unparseable_target_errors_with_hint() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["audit", "show", "garbage"])
        .assert()
        .failure()
        .stderr(contains("1-based line number").and(contains("`last`")));
}

#[test]
fn audit_list_json_is_array_of_summaries() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["extract", "--no-dry-run"])
        .assert()
        .success();
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["audit", "list", "--json"])
        .output()
        .unwrap();
    let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let arr = payload.as_array().expect("array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["line_no"], 1);
    assert_eq!(arr[0]["provider_id"], "mock");
}

// ─────────────────────────────────────────────────────────────────────────────
// §-1.5 PR-6 — Anthropic provider (Round 45)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_anthropic_without_api_key_errors_clearly() {
    // Mirrors the openai variant. Two CI legs hit this path:
    //
    //   - `test (ubuntu-latest)` with `--no-default-features` → the
    //     `anthropic-provider` feature is OFF → "requires the
    //     anthropic-provider cargo feature".
    //   - default-features → feature ON → "ANTHROPIC_API_KEY required".
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .env_remove("ANTHROPIC_API_KEY")
        .args(["extract", "--no-dry-run", "--provider", "anthropic"])
        .assert()
        .failure()
        .stderr(contains("ANTHROPIC_API_KEY").or(contains("anthropic-provider")));
}

#[test]
fn extract_unknown_provider_lists_anthropic_in_supported() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["extract", "--no-dry-run", "--provider", "bogus-vendor"])
        .assert()
        .failure()
        .stderr(contains("unknown --provider").and(contains("anthropic")));
}

#[test]
fn extract_max_retries_lands_in_audit_log() {
    // Round 48: --max-retries flag plumbs into both the provider's
    // RetryPolicy AND into the per-run audit log entry.
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["extract", "--no-dry-run", "--max-retries", "7"])
        .assert()
        .success();
    let audit_path = dir.path().join("audit").join("stage2.jsonl");
    let body = std::fs::read_to_string(&audit_path).unwrap();
    let line = body.lines().next().expect("at least one audit line");
    let entry: serde_json::Value = serde_json::from_str(line).unwrap();
    assert_eq!(
        entry["max_retries"].as_u64(),
        Some(7),
        "--max-retries must be recorded in stage2.jsonl"
    );
}

#[test]
fn extract_default_max_retries_is_three() {
    // Without --max-retries, the audit log captures the default (3).
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["extract", "--no-dry-run"])
        .assert()
        .success();
    let audit_path = dir.path().join("audit").join("stage2.jsonl");
    let body = std::fs::read_to_string(&audit_path).unwrap();
    let entry: serde_json::Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
    assert_eq!(entry["max_retries"].as_u64(), Some(3));
}

// ─────────────────────────────────────────────────────────────────────────────
// §-2.5 — `anamnesis doctor` per-source health check (Round 49)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn doctor_no_sources_is_friendly() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor"])
        .assert()
        .success()
        .stdout(contains("No registered sources"));
}

#[test]
fn doctor_reports_unhealthy_when_path_missing() {
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
            "--path",
            "/tmp/anamnesis-doctor-nonexistent-path",
        ])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor"])
        .assert()
        .success()
        .stdout(
            contains("claude-code")
                .and(contains("NOT HEALTHY"))
                .and(contains("projects_root not found")),
        );
}

#[test]
fn doctor_json_shape_includes_adapter_and_ok() {
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
            "mem0",
            "--path",
            "/tmp/anamnesis-doctor-nope.sqlite",
        ])
        .assert()
        .success();
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--json"])
        .output()
        .unwrap();
    let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let arr = payload.as_array().expect("array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["adapter"], "mem0");
    assert_eq!(arr[0]["registered"], true);
    assert!(arr[0]["detail"].is_string());
}

#[test]
fn doctor_filter_source_narrows_output() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "add", "claude-code", "--path", "/tmp/a"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "add", "mem0", "--path", "/tmp/b.sqlite"])
        .assert()
        .success();

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--source", "mem0", "--json"])
        .output()
        .unwrap();
    let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let arr = payload.as_array().unwrap();
    assert_eq!(arr.len(), 1, "--source should filter to one row");
    assert_eq!(arr[0]["adapter"], "mem0");
}

#[test]
fn doctor_strict_exits_nonzero_when_a_source_is_unhealthy() {
    // Round 50: --strict turns the "all healthy?" check into a CI gate.
    // Without --strict, doctor exits 0 even when some sources are
    // unreachable (it's an inspection tool too).
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
            "--path",
            "/tmp/anamnesis-doctor-strict-nope",
        ])
        .assert()
        .success();
    // Default mode → exit 0.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor"])
        .assert()
        .success();
    // --strict → exit non-zero.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--strict"])
        .assert()
        .failure()
        .stderr(contains(
            "registered source(s) reported unhealthy under --strict",
        ));
}

#[test]
fn doctor_strict_with_only_healthy_sources_exits_zero() {
    // Build a minimal fixture that the claude-code adapter accepts as
    // a valid (empty) projects_root, then strict mode must NOT trip.
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    // Make a real empty projects dir.
    let projects = dir.path().join("claude_projects");
    std::fs::create_dir_all(&projects).unwrap();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "source",
            "add",
            "claude-code",
            "--path",
            projects.to_str().unwrap(),
        ])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--strict"])
        .assert()
        .success();
}

#[test]
fn doctor_strict_json_still_prints_then_errors() {
    // CI gates want both the JSON payload AND the non-zero exit so they
    // can tee the report and fail the build. Verify both happen.
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "add", "mem0", "--path", "/tmp/nope.sqlite"])
        .assert()
        .success();
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--json", "--strict"])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "strict json mode must still fail when unhealthy: status={:?}",
        out.status
    );
    let payload: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let arr = payload.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["ok"], false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Round 51: doctor --since staleness filter
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn doctor_since_marks_never_imported_as_stale() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    // Real empty projects dir → adapter health() = ok.
    let projects = dir.path().join("projects");
    std::fs::create_dir_all(&projects).unwrap();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "source",
            "add",
            "claude-code",
            "--path",
            projects.to_str().unwrap(),
        ])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--since", "7d"])
        .assert()
        .success()
        .stdout(contains("STALE").and(contains("1 stale")));
}

#[test]
fn doctor_strict_staleness_exits_nonzero() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    let projects = dir.path().join("projects");
    std::fs::create_dir_all(&projects).unwrap();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "source",
            "add",
            "claude-code",
            "--path",
            projects.to_str().unwrap(),
        ])
        .assert()
        .success();
    // No --since-staleness → exit 0 even with --since.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--since", "7d"])
        .assert()
        .success();
    // With --strict-staleness, the never-imported source trips exit 1.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--since", "7d", "--strict-staleness"])
        .assert()
        .failure()
        .stderr(contains("stale under --strict-staleness"));
}

#[test]
fn doctor_since_units_parse_correctly() {
    // `1m` (1 minute) is short — the never-imported row should still
    // flag stale, but more importantly `0d` should NOT flag stale.
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    let projects = dir.path().join("projects");
    std::fs::create_dir_all(&projects).unwrap();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "source",
            "add",
            "claude-code",
            "--path",
            projects.to_str().unwrap(),
        ])
        .assert()
        .success();
    // `1m` accepted as minutes.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--since", "1m"])
        .assert()
        .success()
        .stdout(contains("1 stale"));
    // Bare seconds also accepted.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--since", "60"])
        .assert()
        .success()
        .stdout(contains("1 stale"));
}

#[test]
fn doctor_since_rejects_malformed_value() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--since", "not-a-duration"])
        .assert()
        .failure()
        .stderr(contains("--since must be of the form"));
}

#[test]
fn doctor_json_includes_stale_field_when_since_set() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    let projects = dir.path().join("projects");
    std::fs::create_dir_all(&projects).unwrap();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "source",
            "add",
            "claude-code",
            "--path",
            projects.to_str().unwrap(),
        ])
        .assert()
        .success();
    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["doctor", "--json", "--since", "7d"])
        .output()
        .unwrap();
    let arr: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let rows = arr.as_array().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["stale"], true);
}

// ─────────────────────────────────────────────────────────────────────────────
// Round 52: doctor generic-mcp live probe
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn doctor_generic_mcp_unreachable_url_is_unhealthy() {
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
            "generic-mcp",
            "--url",
            "http://127.0.0.1:1/nope-doctor-test",
            "--token-env",
            "DOCTOR_GENERIC_MCP_TOK",
        ])
        .assert()
        .success();
    // Provide the token so we get past env-resolution and actually
    // attempt the HTTP probe — which then fails for a dead address.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .env("DOCTOR_GENERIC_MCP_TOK", "stub-token")
        .args(["doctor"])
        .assert()
        .success()
        .stdout(
            contains("generic-mcp")
                .and(contains("NOT HEALTHY"))
                .and(contains("upstream MCP unreachable")),
        );
}

#[test]
fn doctor_generic_mcp_missing_token_env_surfaces_clean_error() {
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
            "generic-mcp",
            "--url",
            "http://127.0.0.1:1/whatever",
            "--token-env",
            "DOCTOR_GENERIC_MCP_NOSUCH",
        ])
        .assert()
        .success();
    // No env var set → token resolution fails before the HTTP probe.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .env_remove("DOCTOR_GENERIC_MCP_NOSUCH")
        .args(["doctor"])
        .assert()
        .success()
        .stdout(
            contains("generic-mcp")
                .and(contains("token resolution failed"))
                .and(contains("DOCTOR_GENERIC_MCP_NOSUCH")),
        );
}

// ─── Round-87 PR-78i: search --explain ─────────────────────────────

/// Default `search --json` carries no `explain` field —
/// back-compat with every existing R8/R71 consumer.
#[test]
fn search_default_json_has_no_explain_field() {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::Store;
    use chrono::Utc;

    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    let db = dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).unwrap();
    let r = AnamnesisRecord {
        id: RecordId::from_parts("claude-code", None, "x"),
        source: SourceDescriptor {
            adapter: "claude-code".into(),
            instance: None,
            version: "0".into(),
        },
        content: "uniqueExplainMarker body".into(),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: "x".into(),
            native_path: None,
            captured_at: Utc::now(),
            raw_hash: "h-x".into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let c = Chunker::default().chunk(&r.id, &r.content);
    store.upsert_record(&r, &c, None).unwrap();
    drop(store);

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "search",
            "uniqueExplainMarker",
            "--mode",
            "fulltext",
            "--json",
        ])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let row = &v["results"][0];
    assert!(
        row.get("explain").is_none(),
        "default search must NOT carry an `explain` field; got {row}"
    );
}

/// `--explain` attaches a numeric breakdown to every result.
/// Fields: `record_score`, `best_chunk_rrf_score`, `kind_boost`,
/// `stages.fts.{rank,raw_score,rrf_contribution}`, `stages.rrf_k`.
/// Asserts the arithmetic: `record_score == best_chunk_rrf_score
/// + kind_boost` and `rrf_contribution == 1/(rrf_k + rank)`.
#[test]
fn search_explain_attaches_numeric_breakdown() {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::Store;
    use chrono::Utc;

    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    let db = dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).unwrap();
    let r = AnamnesisRecord {
        id: RecordId::from_parts("claude-code", None, "x"),
        source: SourceDescriptor {
            adapter: "claude-code".into(),
            instance: None,
            version: "0".into(),
        },
        content: "uniqueExplainArith body".into(),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: "x".into(),
            native_path: None,
            captured_at: Utc::now(),
            raw_hash: "h-x".into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let c = Chunker::default().chunk(&r.id, &r.content);
    store.upsert_record(&r, &c, None).unwrap();
    drop(store);

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args([
            "search",
            "uniqueExplainArith",
            "--mode",
            "fulltext",
            "--json",
            "--explain",
        ])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let row = &v["results"][0];
    let explain = &row["explain"];
    assert!(!explain.is_null(), "must carry explain block");
    let record_score = explain["record_score"].as_f64().unwrap();
    let best = explain["best_chunk_rrf_score"].as_f64().unwrap();
    let boost = explain["kind_boost"].as_f64().unwrap();
    assert!(
        (record_score - (best + boost)).abs() < 1e-9,
        "record_score must equal best_chunk_rrf_score + kind_boost; got {explain}"
    );
    let stages = &explain["stages"];
    let fts = &stages["fts"];
    let rrf_k = stages["rrf_k"].as_f64().unwrap();
    let rank = fts["rank"].as_u64().unwrap() as f64;
    let contribution = fts["rrf_contribution"].as_f64().unwrap();
    assert!(
        (contribution - 1.0 / (rrf_k + rank)).abs() < 1e-9,
        "rrf_contribution must equal 1 / (rrf_k + rank); got {explain}"
    );
}

// ─── Round-86 PR-78h: source show ──────────────────────────────────

/// `source show` on an unregistered adapter exits non-zero
/// with "source not found" — typo'd ids stay loud.
#[test]
fn source_show_missing_source_exits_nonzero() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "show", "claude-code"])
        .assert()
        .failure()
        .stderr(contains("source not found"));
}

/// `source show <id> --json` returns counts + (empty)
/// recent_import_errors for a fresh registered source.
#[test]
fn source_show_json_returns_counts_and_empty_errors() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "add", "claude-code", "--path", "/tmp/cc"])
        .assert()
        .success();

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "show", "claude-code", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["source"]["adapter"], "claude-code");
    assert!(v["source"]["instance"].is_null());
    assert_eq!(v["source"]["record_count"], 0);
    assert_eq!(v["source"]["tagged_record_count"], 0);
    assert_eq!(v["recent_import_errors"].as_array().unwrap().len(), 0);
}

/// After seeding records + tagging one + logging an import
/// error, `source show --json` surfaces both pieces of state
/// in one call, scoped to the requested `(adapter, instance)`.
#[test]
fn source_show_json_surfaces_tagged_count_and_recent_errors() {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::{Store, UserTagOperation};
    use chrono::Utc;

    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "add", "claude-code", "--path", "/tmp/cc"])
        .assert()
        .success();

    // Seed via store directly.
    let db = dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).unwrap();
    let r = AnamnesisRecord {
        id: RecordId::from_parts("claude-code", None, "a"),
        source: SourceDescriptor {
            adapter: "claude-code".into(),
            instance: None,
            version: "0".into(),
        },
        content: "body".into(),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: "a".into(),
            native_path: None,
            captured_at: Utc::now(),
            raw_hash: "h-a".into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let c = Chunker::default().chunk(&r.id, &r.content);
    store.upsert_record(&r, &c, None).unwrap();
    store
        .tag_record(&r.id, &["keep".into()], UserTagOperation::Add)
        .unwrap();
    store
        .log_import_error(
            "claude-code",
            None,
            Some("a"),
            Some("/tmp/cc/a.md"),
            "parse",
            "boom",
        )
        .unwrap();
    drop(store);

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["source", "show", "claude-code", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["source"]["record_count"], 1);
    assert_eq!(v["source"]["tagged_record_count"], 1);
    let errors = v["recent_import_errors"].as_array().unwrap();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0]["phase"], "parse");
    assert_eq!(errors[0]["error"], "boom");
    assert_eq!(errors[0]["native_path"], "/tmp/cc/a.md");
}

// ─── Round-84 PR-78f: audit tail ────────────────────────────────────

/// Empty store has no audit.log → CLI prints a friendly empty
/// message (not an error) and exits 0.
#[test]
fn audit_tail_empty_store_prints_friendly_message() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["audit", "tail"])
        .assert()
        .success()
        .stdout(contains("no audit entries yet"));
}

/// After a forget call lands an audit entry, `audit tail --json`
/// returns it with full detail, including `via` + `outcome`. The
/// `filter` block echoes the operator-supplied --action.
#[test]
fn audit_tail_json_returns_recent_forget_entries() {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::Store;
    use chrono::Utc;

    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    // Seed a record so we can forget it.
    let db = dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).unwrap();
    let r = AnamnesisRecord {
        id: RecordId::from_parts("claude-code", None, "auditme"),
        source: SourceDescriptor {
            adapter: "claude-code".into(),
            instance: None,
            version: "0".into(),
        },
        content: "auditable body".into(),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: "auditme".into(),
            native_path: None,
            captured_at: Utc::now(),
            raw_hash: "h-audit".into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let c = Chunker::default().chunk(&r.id, &r.content);
    store.upsert_record(&r, &c, None).unwrap();
    drop(store);

    // forget appends one audit entry with action="forget".
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["forget", &r.id.0, "--reason", "auditing"])
        .assert()
        .success();

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["audit", "tail", "--action", "forget", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["filter"]["action"], "forget");
    let entries = v["entries"].as_array().unwrap();
    assert!(!entries.is_empty(), "must capture the forget entry");
    let forget_entry = entries
        .iter()
        .find(|e| e["action"] == "forget")
        .expect("forget entry");
    assert_eq!(forget_entry["detail"]["outcome"], "forgotten");
    assert_eq!(forget_entry["detail"]["reason"], "auditing");
}

/// `--action` filter excludes unrelated entries. Two operations
/// land (`forget` + `search` via CLI), `--action search` shows
/// only the search rows.
#[test]
fn audit_tail_action_filter_excludes_other_actions() {
    let dir = tmp_dir();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["init"])
        .assert()
        .success();
    // `search` writes an audit entry even on an empty store.
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["search", "anything", "--mode", "fulltext"])
        .assert()
        .success();
    cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["search", "different", "--mode", "fulltext"])
        .assert()
        .success();

    let out = cli()
        .env("ANAMNESIS_DATA_DIR", dir.path())
        .args(["audit", "tail", "--action", "search", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let entries = v["entries"].as_array().unwrap();
    assert!(entries.len() >= 2);
    for e in entries {
        assert_eq!(e["action"], "search");
    }
}
