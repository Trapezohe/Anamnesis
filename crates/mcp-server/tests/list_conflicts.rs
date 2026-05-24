//! Round-135 PR-78bd MCP `list_conflicts` end-to-end.
//!
//! Mirrors the CLI fixture / acceptance points. NOT admin-gated.
//! Privacy contract: redacted by default, opt-in `content_preview`.

use anamnesis_core::chunker::Chunker;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_mcp_server::AnamnesisServer;
use anamnesis_store::Store;
use chrono::Utc;
use serde_json::{json, Value};

fn build_bundle() -> (AnamnesisServer, tempfile::TempDir) {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let db = data_dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");

    let mk = |adapter: &str, native: &str, content: &str, raw: &str| AnamnesisRecord {
        id: RecordId::from_parts(adapter, None, native),
        source: SourceDescriptor {
            adapter: adapter.into(),
            instance: None,
            version: "0".into(),
        },
        content: content.into(),
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
            raw_hash: raw.into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };

    for r in [
        mk("mem0", "shared-mcp", "MCP Variant A", "raw-mcp-a"),
        mk("claude-code", "shared-mcp", "MCP Variant B", "raw-mcp-b"),
        // Same content → no conflict.
        mk("mem0", "agree-mcp", "Agreed body", "raw-mcp-c"),
        mk("claude-code", "agree-mcp", "Agreed body", "raw-mcp-d"),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
    let server =
        AnamnesisServer::new(store, None, data_dir.path().to_path_buf()).with_admin_tools(false);
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
async fn list_conflicts_appears_in_default_tools_list_without_admin() {
    let (server, _data) = build_bundle();
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
        names.contains(&"list_conflicts"),
        "list_conflicts must appear in non-admin tools/list; got {names:?}"
    );
}

#[tokio::test]
async fn list_conflicts_callable_without_admin() {
    let (server, _data) = build_bundle();
    let resp = server.handle(tool_call("list_conflicts", json!({}))).await;
    assert!(
        resp.error.is_none(),
        "list_conflicts must be callable without admin gate: {:?}",
        resp.error
    );
    let payload = extract_payload(&resp);
    assert_eq!(payload["count"], 1);
    assert_eq!(payload["format"], "json");
    assert_eq!(payload["groups"][0]["native_id"], "shared-mcp");
    assert_eq!(payload["groups"][0]["content_variant_count"], 2);
}

#[tokio::test]
async fn list_conflicts_default_payload_redacts_content_and_paths() {
    let (server, _data) = build_bundle();
    let resp = server.handle(tool_call("list_conflicts", json!({}))).await;
    let payload = extract_payload(&resp);
    assert_eq!(payload["content_included"], false);
    let rows = payload["groups"][0]["records"].as_array().unwrap();
    for row in rows {
        assert!(row.get("content_preview").is_none());
        assert!(row.get("native_path").is_none());
        assert_eq!(row["has_native_path"], true);
    }
    // Serialised payload must not leak content body or raw_hash.
    let serialised = serde_json::to_string(&payload).unwrap();
    for forbidden in ["MCP Variant A", "MCP Variant B", "raw-mcp-a", "raw-mcp-b"] {
        assert!(
            !serialised.contains(forbidden),
            "default payload must not leak {forbidden:?}: {serialised}"
        );
    }
}

#[tokio::test]
async fn list_conflicts_include_content_attaches_preview() {
    let (server, _data) = build_bundle();
    let resp = server
        .handle(tool_call(
            "list_conflicts",
            json!({"include_content": true}),
        ))
        .await;
    let payload = extract_payload(&resp);
    assert_eq!(payload["content_included"], true);
    let rows = payload["groups"][0]["records"].as_array().unwrap();
    let bodies: Vec<&str> = rows
        .iter()
        .map(|r| r["content_preview"].as_str().expect("preview present"))
        .collect();
    assert!(bodies.iter().any(|b| b.contains("Variant A")));
    assert!(bodies.iter().any(|b| b.contains("Variant B")));
}

#[tokio::test]
async fn list_conflicts_tools_list_advertises_include_content_arg() {
    let (server, _data) = build_bundle();
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let entry = tools
        .iter()
        .find(|t| t["name"] == "list_conflicts")
        .expect("list_conflicts in tools/list");
    let props = &entry["inputSchema"]["properties"];
    assert_eq!(props["include_content"]["type"], "boolean");
    assert_eq!(props["include_content"]["default"], false);
    assert_eq!(props["source"]["type"], "string");
    assert_eq!(props["instance"]["type"], "string");
}
