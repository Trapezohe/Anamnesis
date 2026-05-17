//! End-to-end test for `anamnesis serve --sse <port>`.
//!
//! Spawns the real CLI binary as a subprocess against a temp data dir,
//! parses the bound port from stderr (since the test uses `--sse 0`
//! for an ephemeral port), and verifies the HTTP transport is reachable
//! with the expected auth contract:
//!
//!   GET /healthz                      → 200 ok
//!   POST /mcp  (no token)             → 401
//!   POST /mcp  (Bearer test-token)    → 200 with initialize result
//!
//! This is the round-14 acceptance test that unwires the
//! "use the dedicated anamnesis-mcp --sse binary instead" foot-gun.

#![cfg(feature = "sse")]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Path to the cargo-built `anamnesis` binary.
fn anamnesis_bin() -> std::path::PathBuf {
    assert_cmd::cargo::cargo_bin("anamnesis")
}

/// Spawn `anamnesis serve --sse 0 --token <token>` and block until
/// either the "listening on http://127.0.0.1:<port>" line shows up on
/// stderr (in which case we return the parsed `SocketAddr`) or the
/// child exits / we time out.
fn spawn_and_wait_for_port(
    data_dir: &std::path::Path,
    token: &str,
) -> (Child, std::net::SocketAddr) {
    let mut child = Command::new(anamnesis_bin())
        .env("ANAMNESIS_DATA_DIR", data_dir)
        // Keep noise low — only our own eprintlns + the bound-addr line.
        .env("RUST_LOG", "error")
        .args(["serve", "--sse", "0", "--token", token])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn anamnesis serve --sse 0");

    let stderr = child.stderr.take().expect("stderr pipe");
    let (tx, rx) = mpsc::channel::<String>();

    // Drain stderr on a background thread so the child doesn't block
    // on a full pipe — and forward every line back so we can parse it.
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            eprintln!("[anamnesis stderr] {line}");
            // Best effort — receiver may be gone if we already parsed.
            let _ = tx.send(line);
        }
    });

    let deadline = Instant::now() + Duration::from_secs(15);
    let prefix = "anamnesis-mcp HTTP — listening on http://";
    loop {
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("timed out waiting for bound-address line");
        }
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(line) => {
                if let Some(rest) = line.strip_prefix(prefix) {
                    let addr: std::net::SocketAddr = rest
                        .trim()
                        .parse()
                        .unwrap_or_else(|e| panic!("parse bound addr from {rest:?}: {e}"));
                    return (child, addr);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Ok(Some(status)) = child.try_wait() {
                    panic!("child exited early with status {status:?}");
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = child.kill();
                panic!("stderr drain thread closed before bound-address line");
            }
        }
    }
}

/// Best-effort cleanup so a leftover serve subprocess doesn't outlive
/// the test runner.
struct Guard(Child);
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Open the store at `data_dir/anamnesis.sqlite`, register a synthetic
/// claude-code source, and upsert one record + chunks containing a
/// unique sentinel token. Returns the sentinel + record id for the
/// caller's assertions.
///
/// We deliberately do NOT install an active embedding model — the
/// downstream `anamnesis serve --sse` process will see `provider =
/// None` and take the FTS-only path through hybrid search. Avoids the
/// ~500 MB model download in CI.
fn seed_synthetic_search_record(data_dir: &std::path::Path) -> (&'static str, String) {
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use anamnesis_store::Store;
    use chrono::Utc;

    let db = data_dir.join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store for seeding");
    store
        .register_source("claude-code", None, Some("/tmp/round-15"), None)
        .expect("register synthetic source");

    let native_id = "round-15-seed-1";
    // The sentinel must survive jieba+unicode61 tokenization. CamelCase
    // ASCII does; pick a token unique enough that it cannot collide with
    // any other test fixture.
    let sentinel = "UniqueRound15HttpSearchToken";
    let content = format!(
        "round-15 SSE integration sentinel: {sentinel} — this record \
         exists only so the over-HTTP search test has something to find."
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
        },
        schema_version: SCHEMA_VERSION,
    };
    let chunks = Chunker::default().chunk(&r.id, &r.content);
    store
        .upsert_record(&r, &chunks, None)
        .expect("upsert synthetic record");

    // Drop the store explicitly so its SQLite handle is closed before
    // the child server opens its own connection. SQLite in WAL mode
    // tolerates concurrent opens, but closing here keeps the test's
    // failure mode obvious if that ever changes.
    drop(store);
    (sentinel, r.id.0.clone())
}

#[tokio::test(flavor = "current_thread")]
async fn serve_sse_http_transport_round_trips_initialize() {
    let dir = tempfile::tempdir().expect("tempdir");
    let token = "round-14-test-token-9c1f";

    let (child, addr) = spawn_and_wait_for_port(dir.path(), token);
    let _guard = Guard(child);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();

    // 1. /healthz is open (no auth) and returns 200 "ok".
    let url = format!("http://{addr}");
    let resp = client
        .get(format!("{url}/healthz"))
        .send()
        .await
        .expect("GET /healthz");
    assert_eq!(resp.status(), 200, "/healthz should be 200");
    assert_eq!(resp.text().await.unwrap().trim(), "ok");

    // 2. /mcp without bearer → 401.
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "serve-sse-test", "version": "0" }
        }
    });
    let resp = client
        .post(format!("{url}/mcp"))
        .json(&body)
        .send()
        .await
        .expect("POST /mcp no-bearer");
    assert_eq!(
        resp.status(),
        401,
        "POST /mcp without bearer must be 401, was {}",
        resp.status()
    );

    // 3. /mcp with bearer + initialize → 200 + serverInfo.name == "anamnesis".
    let resp = client
        .post(format!("{url}/mcp"))
        .bearer_auth(token)
        .json(&body)
        .send()
        .await
        .expect("POST /mcp bearer");
    assert_eq!(
        resp.status(),
        200,
        "POST /mcp with bearer must be 200, was {}",
        resp.status()
    );
    let v: serde_json::Value = resp.json().await.expect("json initialize response");
    let server_name = v
        .get("result")
        .and_then(|r| r.get("serverInfo"))
        .and_then(|s| s.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("");
    assert_eq!(
        server_name, "anamnesis",
        "initialize must return serverInfo.name = anamnesis, got payload: {v}"
    );
}

/// Round-15: prove an MCP client can actually *call a tool* (not just
/// `initialize`) over the SSE/HTTP transport exposed by the unified
/// CLI. Seeds one record with a unique sentinel, then runs a
/// JSON-RPC `tools/call` for `search_memories` and checks the wire
/// format committed in PR-#16:
///
///   - HTTP 200, no JSON-RPC `error` member
///   - `result.structuredContent.results` is a non-empty array
///   - every hit has a non-empty `trace_id` (alias for `record_id`)
///   - at least one hit has `from_fts == true` (no embedding model is
///     installed → FTS path is the only one available, so this is the
///     load-bearing modality for the test)
///   - at least one hit's `snippet` contains the seeded sentinel
#[tokio::test(flavor = "current_thread")]
async fn serve_sse_search_memories_round_trips_over_http() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (sentinel, _record_id) = seed_synthetic_search_record(dir.path());

    let token = "round-15-search-token-3e5a";
    let (child, addr) = spawn_and_wait_for_port(dir.path(), token);
    let _guard = Guard(child);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "search_memories",
            "arguments": {
                "query": sentinel,
                "limit": 5,
                "mode": "fulltext"
            }
        }
    });
    let resp = client
        .post(format!("http://{addr}/mcp"))
        .bearer_auth(token)
        .json(&body)
        .send()
        .await
        .expect("POST /mcp tools/call");
    assert_eq!(
        resp.status(),
        200,
        "tools/call must be 200 over HTTP, was {}",
        resp.status()
    );

    let v: serde_json::Value = resp.json().await.expect("json tools/call response");
    assert!(
        v.get("error").is_none(),
        "JSON-RPC error from tools/call: {v}"
    );
    let results = v
        .pointer("/result/structuredContent/results")
        .and_then(|r| r.as_array())
        .unwrap_or_else(|| panic!("expected result.structuredContent.results array, got: {v}"));
    assert!(
        !results.is_empty(),
        "search_memories over HTTP must return at least one hit for the seeded sentinel \
         (FTS path is enough — no embedding model installed). payload: {v}"
    );

    for hit in results {
        let trace_id = hit
            .get("trace_id")
            .and_then(|t| t.as_str())
            .unwrap_or_default();
        assert!(
            !trace_id.is_empty(),
            "every search_memories hit must carry a non-empty trace_id; hit was: {hit}"
        );
    }

    let any_from_fts = results
        .iter()
        .any(|h| h.get("from_fts").and_then(|f| f.as_bool()).unwrap_or(false));
    assert!(
        any_from_fts,
        "at least one hit must have from_fts = true (FTS is the only \
         active modality in this test). hits: {results:?}"
    );

    let any_with_sentinel = results.iter().any(|h| {
        h.get("snippet")
            .and_then(|s| s.as_str())
            .map(|s| s.contains(sentinel))
            .unwrap_or(false)
    });
    assert!(
        any_with_sentinel,
        "at least one hit's snippet must contain the seeded sentinel {sentinel:?}. \
         hits: {results:?}"
    );
}
