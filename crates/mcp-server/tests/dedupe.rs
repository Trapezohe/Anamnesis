//! Round-77 PR-77 MCP `dedupe` tool — admin-free read-only
//! diagnostic over `records.raw_hash` collisions.
//!
//! Acceptance points:
//!
//!   1. **Not admin-gated.** Appears in default `tools/list` and
//!      is callable without `allow_admin_tools`. The action half
//!      (`forget_record`) is still admin-gated; this is just the
//!      report.
//!   2. **Redaction by default.** `raw_hash` and `native_path` are
//!      absent unless `include_sensitive=true`. Same privacy
//!      discipline as `list_forgotten` (R74).
//!   3. **Group shape.** Each group reports `record_count` and a
//!      `records[]` array carrying the minimum operator-decision
//!      fields (record_id / adapter / instance / native_id).

use anamnesis_core::chunker::Chunker;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_mcp_server::AnamnesisServer;
use anamnesis_store::Store;
use chrono::Utc;
use serde_json::{json, Value};

fn build_bundle(allow_admin: bool) -> (AnamnesisServer, tempfile::TempDir) {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let db = data_dir.path().join("anamnesis.sqlite");
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
        make("claude-code", "alpha", "secretMarkerH"),
        make("mem0", "beta", "secretMarkerH"),
        make("claude-code", "gamma", "h-singleton"),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
    let server = AnamnesisServer::new(store, None, data_dir.path().to_path_buf())
        .with_admin_tools(allow_admin);
    (server, data_dir)
}

fn tool_call(name: &str, arguments: Value) -> anamnesis_mcp_server::protocol::JsonRpcRequest {
    anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(7)),
        method: "tools/call".into(),
        params: json!({ "name": name, "arguments": arguments }),
    }
}

fn extract_payload(resp: &anamnesis_mcp_server::protocol::JsonRpcResponse) -> Value {
    serde_json::to_value(resp).unwrap()["result"]["structuredContent"].clone()
}

#[tokio::test]
async fn dedupe_appears_in_default_tools_list_without_admin() {
    let (server, _data) = build_bundle(false);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(
        names.contains(&"dedupe"),
        "dedupe must be in non-admin tools/list; got {names:?}"
    );
}

#[tokio::test]
async fn dedupe_callable_without_admin() {
    let (server, _data) = build_bundle(false);
    let resp = server.handle(tool_call("dedupe", json!({}))).await;
    assert!(
        resp.error.is_none(),
        "dedupe must be callable without admin gate; got {:?}",
        resp.error
    );
    let payload = extract_payload(&resp);
    assert_eq!(payload["count"], 1);
    assert_eq!(payload["groups"][0]["record_count"], 2);
}

#[tokio::test]
async fn dedupe_default_payload_redacts_sensitive_fields() {
    let (server, _data) = build_bundle(false);
    let body = server.handle(tool_call("dedupe", json!({}))).await;
    let payload = extract_payload(&body);
    assert_eq!(payload["sensitive_included"], false);
    let group = &payload["groups"][0];
    assert!(
        group.get("raw_hash").is_none(),
        "raw_hash must be redacted: {group}"
    );
    let row = &group["records"][0];
    assert_eq!(row["has_native_path"], true);
    assert!(row.get("native_path").is_none());
    // Marker leak check against the full serialised payload.
    let serialised = serde_json::to_string(&payload).unwrap();
    assert!(
        !serialised.contains("secretMarkerH"),
        "raw_hash marker must not appear in redacted payload: {serialised}"
    );
    assert!(
        !serialised.contains("/tmp/claude-code/alpha.md"),
        "native_path must not appear in redacted payload: {serialised}"
    );
}

#[tokio::test]
async fn dedupe_include_sensitive_reveals_fields() {
    let (server, _data) = build_bundle(false);
    let body = server
        .handle(tool_call("dedupe", json!({"include_sensitive": true})))
        .await;
    let payload = extract_payload(&body);
    assert_eq!(payload["sensitive_included"], true);
    let group = &payload["groups"][0];
    assert_eq!(group["raw_hash"], "secretMarkerH");
    let rows = group["records"].as_array().unwrap();
    let paths: Vec<&str> = rows
        .iter()
        .filter_map(|r| r["native_path"].as_str())
        .collect();
    assert_eq!(paths.len(), 2);
}
