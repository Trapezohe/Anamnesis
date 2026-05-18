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
        .stderr(contains("OPENAI_API_KEY"));
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
