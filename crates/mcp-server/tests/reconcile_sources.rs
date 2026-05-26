//! MCP `reconcile_sources` — cross-adapter drift diagnostic.
//!
//! Acceptance:
//!   - NOT admin-gated (read-only).
//!   - Counts always present; sample arrays capped at `limit`.
//!   - Payload never leaks `content` / `raw_hash` / `native_path`.
//!   - `identity_key` only surfaces under `include_identity: true`.
//!   - `identity_source` distinguishes round-trip (`anamnesis_native_id`)
//!     from per-adapter (`native_id`) matches.

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

    let mk = |adapter: &str, native: &str, content: &str, raw: &str, ana_native: Option<&str>| {
        let mut r = AnamnesisRecord {
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
        if let Some(key) = ana_native {
            r.metadata.insert("anamnesis_native_id".into(), json!(key));
        }
        r
    };

    for r in [
        // only_left: mem0-side record with no letta counterpart.
        mk("mem0", "left-only", "Only on mem0", "raw-secret-left", None),
        // only_right: letta-side record with no mem0 counterpart.
        mk(
            "letta",
            "right-only",
            "Only on letta",
            "raw-secret-right",
            None,
        ),
        // both: matched via round-trip provenance.
        mk(
            "mem0",
            "shared-agree",
            "Same body",
            "raw-secret-agree-l",
            None,
        ),
        mk(
            "letta",
            "letta-agree",
            "Same body",
            "raw-secret-agree-r",
            Some("shared-agree"),
        ),
        // conflict: matched + content differs.
        mk(
            "mem0",
            "shared-conflict",
            "Body A",
            "raw-secret-conflict-l",
            None,
        ),
        mk(
            "letta",
            "letta-conflict",
            "Body B",
            "raw-secret-conflict-r",
            Some("shared-conflict"),
        ),
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
async fn reconcile_sources_is_not_admin_gated() {
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
        names.contains(&"reconcile_sources"),
        "must appear without admin gate; got {names:?}"
    );
}

#[tokio::test]
async fn reconcile_sources_reports_drift_buckets() {
    let (server, _data) = build_bundle(false);
    let resp = server
        .handle(tool_call(
            "reconcile_sources",
            json!({
                "left":  {"adapter": "mem0"},
                "right": {"adapter": "letta"},
            }),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    let counts = &payload["counts"];
    assert_eq!(counts["only_left"], 1);
    assert_eq!(counts["only_right"], 1);
    assert_eq!(counts["both"], 2);
    assert_eq!(counts["conflicts"], 1);
    assert_eq!(counts["left_total"], 3);
    assert_eq!(counts["right_total"], 3);

    // R152: each drift direction surfaces the lagging side + the format
    // reconcile-export would derive. only_left lags right (letta);
    // only_right lags left (mem0).
    let rt = &payload["round_trip"];
    assert_eq!(rt["only_left"]["lagging"]["adapter"], "letta");
    assert_eq!(rt["only_left"]["export_format"], "letta-sqlite");
    assert_eq!(rt["only_right"]["lagging"]["adapter"], "mem0");
    assert_eq!(rt["only_right"]["export_format"], "mem0-sqlite");
}

#[tokio::test]
async fn reconcile_sources_round_trip_export_format_is_null_for_no_target_adapter() {
    let (server, _data) = build_bundle(false);
    let resp = server
        .handle(tool_call(
            "reconcile_sources",
            json!({
                "left":  {"adapter": "mem0"},
                "right": {"adapter": "claude-code"},
            }),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let rt = &extract_payload(&resp)["round_trip"];
    // only_left lags claude-code, which has no round-trip target.
    assert_eq!(rt["only_left"]["lagging"]["adapter"], "claude-code");
    assert!(rt["only_left"]["export_format"].is_null());
    // only_right lags mem0 → mem0-sqlite.
    assert_eq!(rt["only_right"]["export_format"], "mem0-sqlite");
}

#[tokio::test]
async fn reconcile_sources_never_leaks_secrets_default() {
    let (server, _data) = build_bundle(false);
    let resp = server
        .handle(tool_call(
            "reconcile_sources",
            json!({
                "left":  {"adapter": "mem0"},
                "right": {"adapter": "letta"},
            }),
        ))
        .await;
    let payload = extract_payload(&resp);
    let serialised = serde_json::to_string(&payload).unwrap();
    for forbidden in [
        "Body A",
        "Body B",
        "Same body",
        "Only on mem0",
        "Only on letta",
        "raw-secret-left",
        "raw-secret-right",
        "raw-secret-agree-l",
        "raw-secret-agree-r",
        "raw-secret-conflict-l",
        "raw-secret-conflict-r",
        "/tmp/mem0/left-only.md",
        "/tmp/letta/right-only.md",
        "\"raw_hash\"",
        "\"native_path\"",
        "\"content\"",
        "\"identity_key\"",
    ] {
        assert!(
            !serialised.contains(forbidden),
            "default payload must not leak {forbidden:?}: {serialised}"
        );
    }
}

#[tokio::test]
async fn reconcile_sources_include_identity_surfaces_key_and_source() {
    let (server, _data) = build_bundle(false);
    let resp = server
        .handle(tool_call(
            "reconcile_sources",
            json!({
                "left":  {"adapter": "mem0"},
                "right": {"adapter": "letta"},
                "include_identity": true,
            }),
        ))
        .await;
    let payload = extract_payload(&resp);
    let conf = payload["samples"]["conflicts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["identity_key"].as_str() == Some("shared-conflict"))
        .expect("conflict sample present");
    // Left-side mem0 row produced the conflict; it doesn't carry the
    // round-trip metadata (mem0 IS the upstream here), so the identity
    // source is the per-adapter native_id.
    assert_eq!(conf["identity_source"], "native_id");
    let only_left = &payload["samples"]["only_left"];
    assert_eq!(only_left.as_array().unwrap().len(), 1);
    assert!(only_left[0]["identity_key"].is_string());
}

#[tokio::test]
async fn reconcile_sources_rejects_missing_side() {
    let (server, _data) = build_bundle(false);
    let resp = server
        .handle(tool_call("reconcile_sources", json!({})))
        .await;
    assert!(resp.error.is_some());
    let msg = resp.error.unwrap().message;
    assert!(msg.contains("left"), "must point at missing field: {msg}");
}

#[tokio::test]
async fn reconcile_sources_rejects_missing_adapter() {
    let (server, _data) = build_bundle(false);
    let resp = server
        .handle(tool_call(
            "reconcile_sources",
            json!({"left": {}, "right": {"adapter": "letta"}}),
        ))
        .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn reconcile_sources_tools_list_schema() {
    let (server, _data) = build_bundle(false);
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
        .find(|t| t["name"] == "reconcile_sources")
        .expect("reconcile_sources in non-admin tools/list");
    let schema = &tool["inputSchema"];
    let required: Vec<&str> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(required, vec!["left", "right"]);
    let props = &schema["properties"];
    assert_eq!(props["limit"]["default"], 10);
    assert_eq!(props["include_identity"]["default"], false);
}
