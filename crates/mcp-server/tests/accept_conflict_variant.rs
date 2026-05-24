//! MCP `accept_conflict_variant` — admin-gated conflict resolver.
//!
//! Acceptance:
//!   - Admin-gated (hidden in non-admin `tools/list`, rejected in
//!     `tools/call` without `allow_admin_tools`).
//!   - `apply: false` (default) is a dry-run preview; store unchanged.
//!   - `apply: true` tombstones losers in one tx; conflict goes away.
//!   - Payload never leaks `content` / `raw_hash` / `native_path`.

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
        // Variant A: keeper if `keep_variant=1`.
        mk("claude-code", "shared-id", "Body variant A", "raw-secret-a"),
        // Variant A (same content) — sibling of the keeper.
        mk("mem0", "shared-id", "Body variant A", "raw-secret-b"),
        // Variant B: would be tombstoned if `keep_variant=1`.
        mk("codex", "shared-id", "Body variant B", "raw-secret-c"),
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
async fn accept_conflict_variant_is_admin_gated() {
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
        !names.contains(&"accept_conflict_variant"),
        "must be hidden without admin"
    );

    // tools/call must also be rejected.
    let resp = server
        .handle(tool_call(
            "accept_conflict_variant",
            json!({"native_id": "shared-id", "keep_variant": 1}),
        ))
        .await;
    assert!(resp.error.is_some(), "non-admin call must be rejected");
}

#[tokio::test]
async fn accept_conflict_variant_dry_run_does_not_mutate() {
    let (server, _data) = build_bundle(true);
    let resp = server
        .handle(tool_call(
            "accept_conflict_variant",
            json!({"native_id": "shared-id", "keep_variant": 1}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["dry_run"], true);
    assert_eq!(payload["status"], "would-accept");
    assert_eq!(payload["keep_variant"], 1);
    let keep = payload["keep_records"].as_array().unwrap();
    assert_eq!(keep.len(), 2, "two records share variant A");
    let forget = payload["forget_records"].as_array().unwrap();
    assert_eq!(forget.len(), 1, "one variant B record gets queued");

    // Conflict still present (no writes).
    let resp = server
        .handle(tool_call("list_conflicts", json!({})))
        .await;
    let payload = extract_payload(&resp);
    assert_eq!(payload["count"], 1, "dry-run must not resolve the conflict");
}

#[tokio::test]
async fn accept_conflict_variant_apply_tombstones_losers() {
    let (server, _data) = build_bundle(true);
    let resp = server
        .handle(tool_call(
            "accept_conflict_variant",
            json!({"native_id": "shared-id", "keep_variant": 1, "apply": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["dry_run"], false);
    assert_eq!(payload["status"], "accepted");

    // Conflict is gone.
    let resp = server
        .handle(tool_call("list_conflicts", json!({})))
        .await;
    let payload = extract_payload(&resp);
    assert_eq!(payload["count"], 0, "apply must resolve the conflict");
}

#[tokio::test]
async fn accept_conflict_variant_never_leaks_secrets() {
    let (server, _data) = build_bundle(true);
    let resp = server
        .handle(tool_call(
            "accept_conflict_variant",
            json!({"native_id": "shared-id", "keep_variant": 1}),
        ))
        .await;
    let payload = extract_payload(&resp);
    let serialised = serde_json::to_string(&payload).unwrap();
    for forbidden in [
        "Body variant A",
        "Body variant B",
        "raw-secret-a",
        "raw-secret-b",
        "raw-secret-c",
        "/tmp/claude-code/shared-id.md",
        "\"raw_hash\"",
        "\"native_path\"",
        "\"content\"",
    ] {
        assert!(
            !serialised.contains(forbidden),
            "payload must not leak {forbidden:?}: {serialised}"
        );
    }
}

#[tokio::test]
async fn accept_conflict_variant_keep_record_id_path_works() {
    let (server, _data) = build_bundle(true);
    let target = RecordId::from_parts("codex", None, "shared-id");
    let resp = server
        .handle(tool_call(
            "accept_conflict_variant",
            json!({
                "native_id": "shared-id",
                "keep_record_id": target.0,
                "apply": true,
            }),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["keep_variant"], 2);
    let keep_ids: std::collections::BTreeSet<&str> = payload["keep_records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["record_id"].as_str().unwrap())
        .collect();
    assert!(keep_ids.contains(target.0.as_str()));
}

#[tokio::test]
async fn accept_conflict_variant_rejects_dual_selectors() {
    let (server, _data) = build_bundle(true);
    let resp = server
        .handle(tool_call(
            "accept_conflict_variant",
            json!({"native_id": "shared-id", "keep_variant": 1, "keep_record_id": "anything"}),
        ))
        .await;
    assert!(resp.error.is_some());
    assert!(
        resp.error.unwrap().message.contains("exactly one"),
        "must reject both selectors set"
    );
}

#[tokio::test]
async fn accept_conflict_variant_rejects_no_selector() {
    let (server, _data) = build_bundle(true);
    let resp = server
        .handle(tool_call(
            "accept_conflict_variant",
            json!({"native_id": "shared-id"}),
        ))
        .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn accept_conflict_variant_tools_list_schema() {
    let (server, _data) = build_bundle(true);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let tool = tools
        .iter()
        .find(|t| t["name"] == "accept_conflict_variant")
        .expect("must surface with admin");
    let props = &tool["inputSchema"]["properties"];
    assert_eq!(props["native_id"]["type"], "string");
    assert_eq!(props["keep_variant"]["type"], "integer");
    assert_eq!(props["keep_record_id"]["type"], "string");
    assert_eq!(props["apply"]["default"], false);
    assert_eq!(props["cascade_derived"]["default"], false);
    let required: Vec<&str> = tool["inputSchema"]["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(required, vec!["native_id"]);
}
