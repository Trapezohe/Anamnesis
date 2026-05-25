//! MCP `reconcile_export_bucket` — admin-gated round-trip export of a
//! reconcile drift bucket. Pipes ids from `Store::reconcile_bucket_ids`
//! through the existing R138/R139/R145 round-trip writers.

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
        // only_left: 2 mem0 records letta doesn't have.
        mk("mem0", "left-A", "Left body A", "raw-secret-A", None),
        mk("mem0", "left-B", "Left body B", "raw-secret-B", None),
        // both: matched via round-trip.
        mk("mem0", "shared", "Same body", "raw-secret-shared-l", None),
        mk(
            "letta",
            "letta-shared",
            "Same body",
            "raw-secret-shared-r",
            Some("shared"),
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
async fn reconcile_export_bucket_is_admin_gated() {
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
    assert!(!names.contains(&"reconcile_export_bucket"));

    let resp = server
        .handle(tool_call(
            "reconcile_export_bucket",
            json!({
                "left":  {"adapter": "mem0"},
                "right": {"adapter": "letta"},
                "bucket": "only-left",
                "format": "jsonl",
                "out": "/tmp/nope.jsonl",
            }),
        ))
        .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn reconcile_export_bucket_only_left_writes_drift_via_memos_scanner() {
    use anamnesis_adapter_memos::scanner::scan_memos;
    let (server, data) = build_bundle(true);
    let out = data.path().join("memos_drift");
    let resp = server
        .handle(tool_call(
            "reconcile_export_bucket",
            json!({
                "left":  {"adapter": "mem0"},
                "right": {"adapter": "letta"},
                "bucket": "only-left",
                "format": "memos-dir",
                "out": out.to_str().unwrap(),
            }),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["records"], 2, "only_left bucket has 2 mem0 records");
    assert_eq!(payload["bucket"], "only-left");
    assert_eq!(payload["format"], "memos-dir");
    // MemOS scanner round-trips the export back to 2 items.
    let scan = scan_memos(&out);
    assert_eq!(scan.total(), 2);
}

#[tokio::test]
async fn reconcile_export_bucket_only_right_is_empty_when_left_dominates() {
    let (server, data) = build_bundle(true);
    let out = data.path().join("right_export.jsonl");
    let resp = server
        .handle(tool_call(
            "reconcile_export_bucket",
            json!({
                "left":  {"adapter": "mem0"},
                "right": {"adapter": "letta"},
                "bucket": "only-right",
                "format": "jsonl",
                "out": out.to_str().unwrap(),
            }),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["records"], 0);
    assert!(out.is_file(), "empty-bucket export still writes the file");
}

#[tokio::test]
async fn reconcile_export_bucket_refuses_to_overwrite_existing_target() {
    let (server, data) = build_bundle(true);
    let out = data.path().join("already.jsonl");
    std::fs::write(&out, b"existing").unwrap();
    let resp = server
        .handle(tool_call(
            "reconcile_export_bucket",
            json!({
                "left":  {"adapter": "mem0"},
                "right": {"adapter": "letta"},
                "bucket": "only-left",
                "format": "jsonl",
                "out": out.to_str().unwrap(),
            }),
        ))
        .await;
    assert!(resp.error.is_some());
    let msg = resp.error.unwrap().message;
    assert!(msg.contains("refusing to overwrite"), "{msg}");
    assert_eq!(std::fs::read(&out).unwrap(), b"existing");
}

#[tokio::test]
async fn reconcile_export_bucket_rejects_invalid_bucket() {
    let (server, data) = build_bundle(true);
    let out = data.path().join("nope.jsonl");
    let resp = server
        .handle(tool_call(
            "reconcile_export_bucket",
            json!({
                "left":  {"adapter": "mem0"},
                "right": {"adapter": "letta"},
                "bucket": "both",
                "format": "jsonl",
                "out": out.to_str().unwrap(),
            }),
        ))
        .await;
    assert!(resp.error.is_some());
    let msg = resp.error.unwrap().message;
    assert!(msg.contains("only-left"), "{msg}");
}

#[tokio::test]
async fn reconcile_export_bucket_tools_list_schema() {
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
        .find(|t| t["name"] == "reconcile_export_bucket")
        .expect("must surface with admin");
    let schema = &tool["inputSchema"];
    let required: Vec<&str> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(required, vec!["left", "right", "bucket", "format", "out"]);
    let bucket_enum: Vec<&str> = schema["properties"]["bucket"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(bucket_enum, vec!["only-left", "only-right"]);
}
