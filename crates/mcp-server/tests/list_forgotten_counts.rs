//! Round-90 PR-78l MCP `list_forgotten { include_counts: true }`.
//!
//! Acceptance points:
//!
//!   1. **Admin-gated** (R74). Already covered by the existing
//!      forget_record test suite; we re-assert here so a future
//!      ACL change can't slip past unnoticed.
//!   2. **Back-compat default.** Without `include_counts`, the
//!      response has no `counts` block — every existing R74
//!      consumer keeps working.
//!   3. **`include_counts: true`** attaches the aggregation block:
//!      `counts.total`, `counts.by_source[]` with
//!      `(adapter, instance, forgotten_count)`. Default instance
//!      serialises as JSON `null`.
//!   4. **Filter respect.** The same `source` / `instance` arg
//!      narrows both the row list AND the counts; the counts
//!      reflect the full matching set, not the current page.
//!   5. **Privacy.** The counts block carries no `reason` /
//!      `native_path` / `raw_hash` — operator-friendly without
//!      `include_sensitive`.

use anamnesis_core::chunker::Chunker;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_mcp_server::AnamnesisServer;
use anamnesis_store::Store;
use chrono::Utc;
use serde_json::{json, Value};

fn build_bundle(allow_admin: bool) -> (AnamnesisServer, tempfile::TempDir, Store) {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let db = data_dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");
    let server = AnamnesisServer::new(
        Store::open(&db).expect("re-open store for server"),
        None,
        data_dir.path().to_path_buf(),
    )
    .with_admin_tools(allow_admin);
    (server, data_dir, store)
}

fn seed_and_forget(store: &Store, adapter: &str, native: &str, reason: &str) {
    let id = RecordId::from_parts(adapter, None, native);
    let r = AnamnesisRecord {
        id: id.clone(),
        source: SourceDescriptor {
            adapter: adapter.into(),
            instance: None,
            version: "0".into(),
        },
        content: format!("body {native}"),
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
            raw_hash: format!("h-{native}"),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let c = Chunker::default().chunk(&r.id, &r.content);
    store.upsert_record(&r, &c, None).unwrap();
    store.forget_record(&id, Some(reason)).unwrap();
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
async fn list_forgotten_default_has_no_counts_block() {
    let (server, _dir, store) = build_bundle(true);
    seed_and_forget(&store, "claude-code", "a", "secret-reason");

    let resp = server.handle(tool_call("list_forgotten", json!({}))).await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert!(
        payload.get("counts").is_none(),
        "default list_forgotten must not carry counts; got {payload}"
    );
}

#[tokio::test]
async fn list_forgotten_include_counts_attaches_total_and_by_source() {
    let (server, _dir, store) = build_bundle(true);
    // 3 claude-code tombstones + 2 mem0 tombstones.
    for n in ["a", "b", "c"] {
        seed_and_forget(&store, "claude-code", n, "no leak");
    }
    for n in ["d", "e"] {
        seed_and_forget(&store, "mem0", n, "no leak");
    }

    let resp = server
        .handle(tool_call("list_forgotten", json!({"include_counts": true})))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    let counts = &payload["counts"];
    assert_eq!(counts["total"], 5);
    let by_source = counts["by_source"].as_array().unwrap();
    let cc = by_source
        .iter()
        .find(|b| b["adapter"] == "claude-code")
        .unwrap();
    let mem = by_source.iter().find(|b| b["adapter"] == "mem0").unwrap();
    assert_eq!(cc["forgotten_count"], 3);
    assert_eq!(mem["forgotten_count"], 2);
    assert!(cc["instance"].is_null());
    assert!(mem["instance"].is_null());
}

#[tokio::test]
async fn list_forgotten_include_counts_respects_source_filter() {
    let (server, _dir, store) = build_bundle(true);
    seed_and_forget(&store, "claude-code", "a", "ok");
    seed_and_forget(&store, "claude-code", "b", "ok");
    seed_and_forget(&store, "mem0", "c", "ok");

    let resp = server
        .handle(tool_call(
            "list_forgotten",
            json!({"source": "mem0", "include_counts": true}),
        ))
        .await;
    let payload = extract_payload(&resp);
    // Page rows scoped to mem0...
    assert_eq!(payload["count"], 1);
    // ...and the counts block matches.
    assert_eq!(payload["counts"]["total"], 1);
    let by_source = payload["counts"]["by_source"].as_array().unwrap();
    assert_eq!(by_source.len(), 1);
    assert_eq!(by_source[0]["adapter"], "mem0");
}

#[tokio::test]
async fn list_forgotten_counts_block_does_not_leak_sensitive_fields() {
    let (server, _dir, store) = build_bundle(true);
    seed_and_forget(&store, "claude-code", "a", "do-not-leak-this-reason");

    let resp = server
        .handle(tool_call("list_forgotten", json!({"include_counts": true})))
        .await;
    let counts = &extract_payload(&resp)["counts"];
    let s = serde_json::to_string(counts).unwrap();
    for forbidden in ["do-not-leak-this-reason", "native_path", "raw_hash"] {
        assert!(
            !s.contains(forbidden),
            "counts must not leak {forbidden:?}: {s}"
        );
    }
}

#[tokio::test]
async fn list_forgotten_tools_list_schema_advertises_include_counts() {
    let (server, _dir, _store) = build_bundle(true);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let lf = tools
        .iter()
        .find(|t| t["name"] == "list_forgotten")
        .expect("list_forgotten must be in admin tools/list");
    let props = &lf["inputSchema"]["properties"];
    assert!(
        props.get("include_counts").is_some(),
        "list_forgotten must advertise include_counts: {lf}"
    );
    assert_eq!(props["include_counts"]["type"], "boolean");
}
