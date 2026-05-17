//! Round-18 (§-1.5 PR-3): MCP `tool_import_source` end-to-end integration.
//!
//! Acceptance points (every one is a load-bearing change in PR-3):
//!
//!   1. **Path / URL rejected on MCP.** A client that previously passed
//!      `{adapter: "mem0", path: "/somewhere"}` now gets a clear error.
//!      This enforces the §-1.6.8 "MCP cannot introduce a new source
//!      location — go through CLI `source add` first" boundary.
//!
//!   2. **Unregistered source rejected.** `import_source` without a
//!      matching `(adapter, instance)` row in the registry refuses.
//!      No silent fallback to a default location.
//!
//!   3. **Registered source produces the same system-state delta as
//!      CLI import**: records appear, `last_import_at` is stamped on
//!      the source row, and a single `import` line is appended to
//!      `audit.log`.
//!
//!   4. **Admin gate respected.** Without `with_admin_tools(true)`,
//!      `tools/call import_source` is rejected even when the source is
//!      registered. This was PR-#10's gate; PR-3 preserves it.

use anamnesis_mcp_server::{server::ADMIN_TOOLS, AnamnesisServer};
use anamnesis_store::Store;
use serde_json::{json, Value};

/// A complete MCP server bundle for the test — owns the temp dir + DB
/// path, so callers can also reach into the store directly to verify
/// side effects.
struct TestBundle {
    server: AnamnesisServer,
    data_dir: tempfile::TempDir,
    db_path: std::path::PathBuf,
}

fn build_bundle(allow_admin: bool) -> TestBundle {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let db_path = data_dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db_path).expect("open store");
    let server = AnamnesisServer::new(store, None, data_dir.path().to_path_buf())
        .with_admin_tools(allow_admin);
    TestBundle {
        server,
        data_dir,
        db_path,
    }
}

/// Seed a fixture mem0 SQLite file at `path` with two `memories` rows.
/// Returns nothing; the test inspects the imported store after.
fn seed_mem0_fixture(path: &std::path::Path) {
    let conn = rusqlite::Connection::open(path).expect("open fixture sqlite");
    // Minimal schema mirroring what `adapter-mem0::Mem0SqliteScanner`
    // reads. See `crates/adapter-mem0/src/scanner.rs` for the exact
    // shape; columns we don't fill (`metadata`, etc.) are optional.
    conn.execute_batch(
        "CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            memory TEXT NOT NULL,
            user_id TEXT,
            created_at TEXT
        );
        INSERT INTO memories(id, memory, user_id, created_at) VALUES
          ('round18-a', 'round-18 first sentinel UniquePr3MemAlpha', 'u', '2026-05-18T00:00:00Z'),
          ('round18-b', 'round-18 second sentinel UniquePr3MemBeta', 'u', '2026-05-18T00:00:01Z');",
    )
    .expect("seed mem0 fixture");
}

fn tool_call(name: &str, arguments: Value) -> anamnesis_mcp_server::protocol::JsonRpcRequest {
    anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(7)),
        method: "tools/call".into(),
        params: json!({ "name": name, "arguments": arguments }),
    }
}

fn extract_text_payload(resp: &anamnesis_mcp_server::protocol::JsonRpcResponse) -> Value {
    // Successful `tools/call` results come back as
    // `{result: {content: [{type: "text", text: "<json-string>"}], structuredContent: <json>}}`.
    // For these assertions we want the structuredContent.
    let v = serde_json::to_value(resp).unwrap();
    v["result"]["structuredContent"].clone()
}

fn read_audit_lines(dir: &std::path::Path) -> Vec<Value> {
    let raw = std::fs::read_to_string(dir.join("audit.log")).unwrap_or_default();
    raw.lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str::<Value>(l).expect("audit line is json"))
        .collect()
}

#[tokio::test]
async fn import_source_is_admin_gated() {
    // PR-#10 guarantee. PR-3 must NOT loosen the gate.
    assert!(
        ADMIN_TOOLS.contains(&"import_source"),
        "import_source must stay tagged as admin"
    );

    let bundle = build_bundle(false); // admin OFF
    let req = tool_call("import_source", json!({"adapter": "mem0"}));
    let resp = bundle.server.handle(req).await;
    let v = serde_json::to_value(&resp).unwrap();
    let err_msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        err_msg.contains("admin"),
        "expected admin-gate rejection, got: {v}"
    );
}

#[tokio::test]
async fn import_source_rejects_path_argument() {
    // Round-18: MCP no longer accepts `path` overrides. This is the
    // §-1.6.8 / §-1.2.2 boundary we tightened.
    let bundle = build_bundle(true);
    let req = tool_call(
        "import_source",
        json!({"adapter": "mem0", "path": "/wherever"}),
    );
    let resp = bundle.server.handle(req).await;
    let v = serde_json::to_value(&resp).unwrap();
    let err_msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        err_msg.contains("path") && err_msg.contains("source add"),
        "expected 'path/url not accepted, use CLI source add' error, got: {v}"
    );
}

#[tokio::test]
async fn import_source_rejects_unregistered_source() {
    let bundle = build_bundle(true);
    let req = tool_call("import_source", json!({"adapter": "mem0"}));
    let resp = bundle.server.handle(req).await;
    let v = serde_json::to_value(&resp).unwrap();
    let err_msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        err_msg.contains("not registered"),
        "expected 'not registered' error, got: {v}"
    );
}

#[tokio::test]
async fn import_source_imports_records_writes_audit_and_stamps_last_import_at() {
    let bundle = build_bundle(true);

    // 1. Pre-register a mem0 source pointing at a seeded fixture.
    let fixture_dir = tempfile::tempdir().expect("fixture tempdir");
    let fixture_db = fixture_dir.path().join("mem0.sqlite");
    seed_mem0_fixture(&fixture_db);
    {
        // Need to register through the same store the server holds.
        // The server owns it by value, so we re-open with a fresh
        // handle — SQLite WAL allows concurrent opens.
        let store = Store::open(&bundle.db_path).expect("open store for register");
        store
            .register_source("mem0", None, Some(fixture_db.to_str().unwrap()), None)
            .expect("register mem0 source");
    }

    // 2. Call `tools/call import_source`.
    let req = tool_call("import_source", json!({"adapter": "mem0"}));
    let resp = bundle.server.handle(req).await;
    let payload = extract_text_payload(&resp);
    assert_eq!(
        payload["raw_seen"], 2,
        "should have scanned both seeded rows; payload was: {payload}"
    );
    assert_eq!(
        payload["records_upserted"], 2,
        "should have upserted both rows; payload was: {payload}"
    );

    // 3. Verify side effects.
    let store = Store::open(&bundle.db_path).expect("re-open store");

    // 3a. last_import_at stamped.
    let row = store.get_source("mem0", None).unwrap().expect("source row");
    assert!(
        row.last_import_at.is_some(),
        "last_import_at must be stamped by ImportService"
    );
    assert_eq!(
        row.location.as_deref(),
        Some(fixture_db.to_str().unwrap()),
        "location must survive re-registration"
    );

    // 3b. Records visible.
    let recent = store.list_recent_record_ids(10).unwrap();
    assert!(
        recent.len() >= 2,
        "expected ≥2 records after import, got {recent:?}"
    );

    // 3c. Audit line appended.
    let lines = read_audit_lines(bundle.data_dir.path());
    assert_eq!(
        lines.len(),
        1,
        "expected exactly one audit line, got: {lines:?}"
    );
    assert_eq!(lines[0]["action"], "import");
    assert_eq!(lines[0]["detail"]["adapter"], "mem0");
    assert_eq!(lines[0]["detail"]["records_upserted"], 2);
    assert_eq!(lines[0]["detail"]["source_was_explicit"], true);
}

#[tokio::test]
async fn import_source_dry_run_does_not_write_registry_or_audit() {
    let bundle = build_bundle(true);

    let fixture_dir = tempfile::tempdir().expect("fixture tempdir");
    let fixture_db = fixture_dir.path().join("mem0.sqlite");
    seed_mem0_fixture(&fixture_db);
    {
        let store = Store::open(&bundle.db_path).expect("open store for register");
        store
            .register_source("mem0", None, Some(fixture_db.to_str().unwrap()), None)
            .expect("register mem0 source");
    }

    // Capture the source row before dry-run.
    let store = Store::open(&bundle.db_path).expect("open store pre");
    let before = store.get_source("mem0", None).unwrap().expect("source row");
    assert!(before.last_import_at.is_none());

    let req = tool_call("import_source", json!({"adapter": "mem0", "dry_run": true}));
    let resp = bundle.server.handle(req).await;
    let payload = extract_text_payload(&resp);
    assert_eq!(payload["raw_seen"], 2);
    assert_eq!(payload["records_upserted"], 0);

    // last_import_at must NOT have been stamped.
    let store = Store::open(&bundle.db_path).expect("open store post");
    let after = store.get_source("mem0", None).unwrap().expect("source row");
    assert!(
        after.last_import_at.is_none(),
        "dry-run must not stamp last_import_at"
    );

    // Records must NOT have been written.
    let recent = store.list_recent_record_ids(10).unwrap();
    assert!(
        recent.is_empty(),
        "dry-run must not write records, got: {recent:?}"
    );

    // Audit log must be empty.
    let lines = read_audit_lines(bundle.data_dir.path());
    assert!(
        lines.is_empty(),
        "dry-run must not append audit, got: {lines:?}"
    );
}
