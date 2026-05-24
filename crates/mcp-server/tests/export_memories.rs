//! Round-140 PR-78bi MCP `export_memories` end-to-end.
//!
//! Acceptance:
//!   1. Admin-gated (hidden from non-admin `tools/list`, rejected
//!      by non-admin `tools/call`).
//!   2. Schema advertises the `format` enum + `out` required field
//!      + optional `source` / `instance` / `kind` filters.
//!   3. Happy path: writes a fresh JSONL file with the requested
//!      records, returns bounded metadata.
//!   4. SQLite formats round-trip through the canonical mem0 /
//!      Letta scanners.
//!   5. Missing `out`, unknown `format`, existing output file
//!      all error loudly.

use anamnesis_core::chunker::Chunker;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_mcp_server::AnamnesisServer;
use anamnesis_store::Store;
use chrono::Utc;
use serde_json::{json, Value};

fn build_server(allow_admin: bool) -> (AnamnesisServer, tempfile::TempDir) {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let store = Store::open(data_dir.path().join("anamnesis.sqlite")).expect("open store");
    let mk = |adapter: &str, native: &str, content: &str| AnamnesisRecord {
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
            native_path: None,
            captured_at: Utc::now(),
            raw_hash: format!("raw-{native}"),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    for r in [
        mk("mem0", "m1", "first mem0 memory"),
        mk("mem0", "m2", "second mem0 memory"),
        mk("claude-code", "cc1", "claude code memory"),
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
async fn export_memories_hidden_from_non_admin_tools_list() {
    let (server, _data) = build_server(false);
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
        !names.contains(&"export_memories"),
        "export_memories must be hidden without admin gate; got {names:?}"
    );
}

#[tokio::test]
async fn export_memories_rejected_when_admin_disabled() {
    let (server, data) = build_server(false);
    let out = data.path().join("nope.jsonl");
    let resp = server
        .handle(tool_call(
            "export_memories",
            json!({"format": "jsonl", "out": out.to_str().unwrap()}),
        ))
        .await;
    assert!(resp.error.is_some(), "must error without admin");
    let msg = resp.error.unwrap().message.to_lowercase();
    assert!(
        msg.contains("admin"),
        "error must mention admin gate: {msg}"
    );
}

#[tokio::test]
async fn export_memories_jsonl_happy_path_returns_bounded_metadata() {
    let (server, data) = build_server(true);
    let out = data.path().join("export.jsonl");
    let resp = server
        .handle(tool_call(
            "export_memories",
            json!({"format": "jsonl", "out": out.to_str().unwrap()}),
        ))
        .await;
    assert!(resp.error.is_none(), "happy path: {:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["format"], "jsonl");
    assert_eq!(payload["records"], 3);
    assert_eq!(payload["out"], out.to_str().unwrap());
    assert!(payload["bytes"].as_u64().unwrap() > 0);
    // Summary mentions counts + filter clauses.
    let summary = payload["summary"].as_str().unwrap();
    assert!(summary.contains("3 record"));
    assert!(summary.contains("source filter: all sources"));

    // Output file exists with 3 JSONL lines.
    let body = std::fs::read_to_string(&out).unwrap();
    assert_eq!(body.lines().count(), 3);
}

#[tokio::test]
async fn export_memories_source_filter_narrows_output() {
    let (server, data) = build_server(true);
    let out = data.path().join("mem0_only.jsonl");
    let resp = server
        .handle(tool_call(
            "export_memories",
            json!({
                "format": "jsonl",
                "out":    out.to_str().unwrap(),
                "source": "mem0",
            }),
        ))
        .await;
    let payload = extract_payload(&resp);
    assert_eq!(payload["records"], 2, "mem0 fixture has 2 records");
    assert_eq!(payload["filters"]["source"], "mem0");

    let body = std::fs::read_to_string(&out).unwrap();
    for line in body.lines() {
        let v: Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["source"]["adapter"], "mem0");
    }
}

#[tokio::test]
async fn export_memories_mem0_sqlite_writes_canonical_schema() {
    use rusqlite::Connection;
    let (server, data) = build_server(true);
    let out = data.path().join("mem0_export.sqlite");
    let resp = server
        .handle(tool_call(
            "export_memories",
            json!({"format": "mem0-sqlite", "out": out.to_str().unwrap()}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["format"], "mem0-sqlite");
    assert_eq!(payload["records"], 3);

    let conn = Connection::open(&out).unwrap();
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 3);
}

#[tokio::test]
async fn export_memories_letta_sqlite_is_letta_adapter_readable() {
    use anamnesis_adapter_letta::scanner::read_all_blocks;
    let (server, data) = build_server(true);
    let out = data.path().join("letta_export.sqlite");
    let resp = server
        .handle(tool_call(
            "export_memories",
            json!({"format": "letta-sqlite", "out": out.to_str().unwrap()}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);

    let rows = read_all_blocks(&out).expect("Letta scanner reads exported DB");
    assert_eq!(rows.len(), 3);
}

#[tokio::test]
async fn export_memories_memos_dir_is_memos_adapter_readable() {
    use anamnesis_adapter_memos::scanner::scan_memos;
    let (server, data) = build_server(true);
    let out = data.path().join("memos_cube");
    let resp = server
        .handle(tool_call(
            "export_memories",
            json!({"format": "memos-dir", "out": out.to_str().unwrap()}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["format"], "memos-dir");
    assert_eq!(payload["records"], 3);
    assert!(out.is_dir());
    assert!(out.join("textual_memory.json").is_file());

    // MemOS scanner round-trips the exported MemCube.
    let scan = scan_memos(&out);
    assert_eq!(scan.total(), 3, "memos adapter reads back the export");
    assert_eq!(
        scan.parse_errors.len(),
        0,
        "no parse errors: {:?}",
        scan.parse_errors
    );
}

#[tokio::test]
async fn export_memories_memos_dir_refuses_to_overwrite_existing_dir() {
    let (server, data) = build_server(true);
    let out = data.path().join("memos_existing");
    std::fs::create_dir_all(&out).unwrap();
    let resp = server
        .handle(tool_call(
            "export_memories",
            json!({"format": "memos-dir", "out": out.to_str().unwrap()}),
        ))
        .await;
    assert!(resp.error.is_some());
    let msg = resp.error.unwrap().message;
    assert!(msg.contains("refusing to overwrite"), "{msg}");
}

#[tokio::test]
async fn export_memories_missing_out_errors() {
    let (server, _data) = build_server(true);
    let resp = server
        .handle(tool_call("export_memories", json!({"format": "jsonl"})))
        .await;
    assert!(resp.error.is_some());
    let msg = resp.error.unwrap().message;
    assert!(msg.contains("out is required"));
}

#[tokio::test]
async fn export_memories_unknown_format_errors() {
    let (server, data) = build_server(true);
    let out = data.path().join("yaml.txt");
    let resp = server
        .handle(tool_call(
            "export_memories",
            json!({"format": "yaml", "out": out.to_str().unwrap()}),
        ))
        .await;
    assert!(resp.error.is_some());
    let msg = resp.error.unwrap().message;
    assert!(msg.contains("unsupported format"));
}

#[tokio::test]
async fn export_memories_refuses_to_overwrite_existing_file() {
    let (server, data) = build_server(true);
    let out = data.path().join("already.jsonl");
    std::fs::write(&out, b"existing").unwrap();
    let resp = server
        .handle(tool_call(
            "export_memories",
            json!({"format": "jsonl", "out": out.to_str().unwrap()}),
        ))
        .await;
    assert!(resp.error.is_some());
    let msg = resp.error.unwrap().message;
    assert!(msg.contains("refusing to overwrite"));
    // File untouched.
    assert_eq!(std::fs::read(&out).unwrap(), b"existing");
}

#[tokio::test]
async fn export_memories_schema_advertises_required_fields() {
    let (server, _data) = build_server(true);
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
        .find(|t| t["name"] == "export_memories")
        .expect("export_memories in admin tools/list");
    let schema = &entry["inputSchema"];
    let required: Vec<&str> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(required.contains(&"format"));
    assert!(required.contains(&"out"));
    let format_enum: Vec<&str> = schema["properties"]["format"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(
        format_enum,
        vec!["jsonl", "csv", "mem0-sqlite", "letta-sqlite", "memos-dir"]
    );
}
