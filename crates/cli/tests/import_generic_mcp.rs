//! End-to-end test for `anamnesis import generic-mcp:<instance>`.
//!
//! Per BLUEPRINT §19.3 PR-1 acceptance: the operator must be able to
//! import from an upstream MCP memory provider using ONLY CLI commands
//! — no test-only construction code, no in-process adapter
//! instantiation, no special builders.
//!
//! Topology:
//!
//!   upstream tempdir A
//!     ├── anamnesis.sqlite (seeded with 1 synthetic record via Store)
//!     └── `anamnesis serve --sse 0 --token <UP_T>`  (subprocess A)
//!
//!   downstream tempdir B
//!     ├── `anamnesis source add generic-mcp --instance loopback \
//!     │      --url http://<addr> --token-env ANAMNESIS_TEST_TOKEN`
//!     └── `anamnesis import generic-mcp:loopback --no-embed`
//!         (the operator's ANAMNESIS_TEST_TOKEN env var = <UP_T>)
//!
//! Acceptance:
//!   - downstream store ends up with at least one record carrying the
//!     seeded sentinel content.
//!   - the source row's last_import_at is non-null after the import.
//!
//! This is the round-17 proof point for "Anamnesis 是记忆统一/迁移层":
//! historical memory in one MCP provider can be pulled into Anamnesis
//! with `source add` + `import`, the same surface as any other
//! adapter.

#![cfg(feature = "sse")]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

fn anamnesis_bin() -> std::path::PathBuf {
    assert_cmd::cargo::cargo_bin("anamnesis")
}

/// Spawn `anamnesis serve --sse 0 --token <t>` against `data_dir` and
/// block until the "listening on http://127.0.0.1:<port>" stderr line
/// appears (the round-14 contract). Returns the still-running child
/// plus its bound socket address.
fn spawn_upstream(data_dir: &std::path::Path, token: &str) -> (Child, std::net::SocketAddr) {
    let mut child = Command::new(anamnesis_bin())
        .env("ANAMNESIS_DATA_DIR", data_dir)
        .env("RUST_LOG", "error")
        .args(["serve", "--sse", "0", "--token", token])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn anamnesis serve");

    let stderr = child.stderr.take().expect("stderr pipe");
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            eprintln!("[upstream stderr] {line}");
            let _ = tx.send(line);
        }
    });

    let deadline = Instant::now() + Duration::from_secs(15);
    let prefix = "anamnesis-mcp HTTP — listening on http://";
    loop {
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("timed out waiting for upstream bound-address line");
        }
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(line) => {
                if let Some(rest) = line.strip_prefix(prefix) {
                    let addr: std::net::SocketAddr = rest
                        .trim()
                        .parse()
                        .unwrap_or_else(|e| panic!("parse upstream addr from {rest:?}: {e}"));
                    return (child, addr);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Ok(Some(status)) = child.try_wait() {
                    panic!("upstream exited early with status {status:?}");
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = child.kill();
                panic!("upstream stderr drain thread closed before bound-address line");
            }
        }
    }
}

/// RAII cleanup for the upstream subprocess.
struct Guard(Child);
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Seed one synthetic record + chunks into `data_dir/anamnesis.sqlite`
/// containing the given sentinel. Returns the sentinel for assertions.
///
/// No embedding model is installed; the upstream serve takes the
/// FTS-only path through hybrid search. Generic-mcp adapter's scan
/// uses `resources/list` + `resources/read`, neither of which requires
/// embeddings.
fn seed_upstream_record(data_dir: &std::path::Path) -> &'static str {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::Store;
    use chrono::Utc;

    let store = Store::open(data_dir.join("anamnesis.sqlite")).expect("open upstream store");
    store
        .register_source("claude-code", None, Some("/tmp/round-17-upstream"), None)
        .expect("register upstream source");

    let native_id = "round-17-upstream-1";
    let sentinel = "UniqueRound17GenericMcpFirstClassImportToken";
    let content = format!(
        "round-17 upstream sentinel: {sentinel} — this record is the proof point \
         that an MCP memory provider can be imported via `anamnesis source add \
         generic-mcp --url ...` + `anamnesis import generic-mcp:<i>` without any \
         test-only construction code."
    );

    let r = AnamnesisRecord {
        id: RecordId::from_parts("claude-code", None, native_id),
        source: SourceDescriptor {
            adapter: "claude-code".into(),
            instance: None,
            version: "0.0.1".into(),
        },
        content,
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: native_id.into(),
            native_path: Some(format!("/p/{native_id}")),
            captured_at: Utc::now(),
            raw_hash: format!("h-{native_id}"),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let chunks = Chunker::default().chunk(&r.id, &r.content);
    store
        .upsert_record(&r, &chunks, None)
        .expect("upsert upstream record");
    drop(store);
    sentinel
}

/// Read all records from `data_dir/anamnesis.sqlite` and return their
/// `content` strings. Used by the test to assert the seeded sentinel
/// reached the downstream after the import.
fn read_downstream_record_contents(data_dir: &std::path::Path) -> Vec<String> {
    let conn =
        rusqlite::Connection::open(data_dir.join("anamnesis.sqlite")).expect("open downstream db");
    let mut stmt = conn.prepare("SELECT content FROM records").unwrap();
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect::<Vec<_>>();
    rows
}

#[test]
fn cli_source_add_then_import_generic_mcp_pulls_from_upstream() {
    let up_dir = tempfile::tempdir().expect("upstream tempdir");
    let down_dir = tempfile::tempdir().expect("downstream tempdir");
    let token = "round-17-upstream-token-7f29";

    let sentinel = seed_upstream_record(up_dir.path());
    let (up_child, addr) = spawn_upstream(up_dir.path(), token);
    let _guard = Guard(up_child);

    // 1. `anamnesis init` on the downstream so an active model is set.
    //    (Not strictly required for the import path, but matches the
    //    real operator flow.)
    let status = Command::new(anamnesis_bin())
        .env("ANAMNESIS_DATA_DIR", down_dir.path())
        .args(["init"])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()
        .expect("spawn anamnesis init");
    assert!(status.success(), "init failed: {status:?}");

    // 2. `anamnesis source add generic-mcp --instance loopback \
    //          --url http://<addr> --token-env ANAMNESIS_R17_TOKEN`
    //
    //    Stored fields: location = URL, config_json = {"token_env": "ANAMNESIS_R17_TOKEN"}.
    let status = Command::new(anamnesis_bin())
        .env("ANAMNESIS_DATA_DIR", down_dir.path())
        .args([
            "source",
            "add",
            "generic-mcp",
            "--instance",
            "loopback",
            "--url",
            &format!("http://{addr}"),
            "--token-env",
            "ANAMNESIS_R17_TOKEN",
        ])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .expect("spawn anamnesis source add");
    assert!(status.success(), "source add failed: {status:?}");

    // 3. `anamnesis import generic-mcp:loopback --no-embed`
    //    The operator's env supplies the actual bearer token under the
    //    name referenced by --token-env.
    let status = Command::new(anamnesis_bin())
        .env("ANAMNESIS_DATA_DIR", down_dir.path())
        .env("ANAMNESIS_R17_TOKEN", token)
        .args(["import", "generic-mcp:loopback", "--no-embed"])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .expect("spawn anamnesis import");
    assert!(status.success(), "import failed: {status:?}");

    // 4. Assert the upstream's seeded sentinel reached the downstream.
    let contents = read_downstream_record_contents(down_dir.path());
    assert!(
        !contents.is_empty(),
        "downstream must hold at least one record after `import generic-mcp:loopback`"
    );
    let any_with_sentinel = contents.iter().any(|c| c.contains(sentinel));
    assert!(
        any_with_sentinel,
        "downstream record(s) must contain the upstream sentinel {sentinel:?}. \
         got contents: {contents:?}"
    );

    // 5. last_import_at must have been stamped on the downstream's
    //    generic-mcp source row — this proves run_import preserved the
    //    URL + token_env config when it re-registered (the round-17
    //    bug-fix point in run_import).
    let conn = rusqlite::Connection::open(down_dir.path().join("anamnesis.sqlite")).unwrap();
    let (location, config_json, last_import_at): (Option<String>, Option<String>, Option<i64>) =
        conn.query_row(
            "SELECT location, config_json, last_import_at \
             FROM sources WHERE adapter = 'generic-mcp' AND instance = 'loopback'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .expect("downstream source row");
    assert_eq!(
        location.as_deref(),
        Some(format!("http://{addr}").as_str()),
        "URL must survive re-registration in run_import"
    );
    let cfg = config_json.expect("config_json must survive re-registration");
    assert!(
        cfg.contains("ANAMNESIS_R17_TOKEN"),
        "token_env must survive re-registration; got config_json: {cfg}"
    );
    assert!(
        last_import_at.is_some(),
        "last_import_at must be set after a successful import"
    );
}
