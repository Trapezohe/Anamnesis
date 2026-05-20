//! Round-85 PR-78g MCP `get_record { include_lineage: true }`.
//!
//! Acceptance points:
//!
//!   1. **Back-compat default.** Without `include_lineage`, the
//!      payload has no `lineage` key (or a null value). Existing
//!      MCP agents that don't know about R85 keep working
//!      verbatim.
//!   2. **Lineage walk.** With `include_lineage: true`, seed a
//!      2-deep chain (root → child) and assert the chain comes
//!      back leaf→root with depth=2 / complete=true.
//!   3. **Dangling parent.** If the chain references a missing
//!      ancestor, the response is `complete: false` +
//!      `missing_parent: <that id>` — exposing the gap rather
//!      than silently truncating.
//!   4. **Schema advertises arg.** tools/list shows
//!      `include_lineage` so MCP clients can introspect.

use anamnesis_core::chunker::Chunker;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_mcp_server::AnamnesisServer;
use anamnesis_store::Store;
use chrono::Utc;
use serde_json::{json, Value};

fn build_server() -> (AnamnesisServer, tempfile::TempDir, Store) {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let db = data_dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");
    let server = AnamnesisServer::new(
        Store::open(&db).expect("re-open store for server"),
        None,
        data_dir.path().to_path_buf(),
    );
    (server, data_dir, store)
}

fn seed_record(
    store: &Store,
    adapter: &str,
    native: &str,
    content: &str,
    derived_from: Option<&RecordId>,
) -> RecordId {
    let id = RecordId::from_parts(adapter, None, native);
    let record = AnamnesisRecord {
        id: id.clone(),
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
            raw_hash: format!("h-{adapter}-{native}"),
            derived_from: derived_from.cloned(),
        },
        schema_version: SCHEMA_VERSION,
    };
    let chunks = Chunker::default().chunk(&record.id, &record.content);
    store.upsert_record(&record, &chunks, None).unwrap();
    id
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
async fn get_record_default_payload_has_no_lineage_block() {
    let (server, _dir, store) = build_server();
    let id = seed_record(&store, "claude-code", "solo", "alpha beta", None);

    let resp = server
        .handle(tool_call("get_record", json!({"id": id.0})))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    // Either the field is absent OR explicitly null — both keep
    // existing agents' branching code stable. We assert "not an
    // object" so the back-compat contract is loose enough to
    // survive harmless serde shape evolution.
    let lineage = payload.get("lineage");
    assert!(
        lineage.is_none() || lineage.unwrap().is_null(),
        "default get_record must not carry a lineage object; got {payload}"
    );
}

#[tokio::test]
async fn get_record_with_include_lineage_returns_leaf_to_root_chain() {
    let (server, _dir, store) = build_server();
    // Root: claude-code:root1 (raw conversation).
    let root = seed_record(&store, "claude-code", "root1", "raw transcript", None);
    // Leaf: extractor:fact1 derived from root.
    let leaf = seed_record(&store, "extractor", "fact1", "distilled fact", Some(&root));

    let resp = server
        .handle(tool_call(
            "get_record",
            json!({"id": leaf.0, "include_lineage": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    let lineage = &payload["lineage"];
    assert_eq!(lineage["depth"], 2);
    assert_eq!(lineage["complete"], true);
    assert!(lineage["missing_parent"].is_null());
    let chain = lineage["chain"].as_array().expect("chain array");
    assert_eq!(chain.len(), 2);
    // Leaf first.
    assert_eq!(chain[0]["record_id"], leaf.0);
    assert_eq!(chain[0]["derived_from"], root.0);
    assert_eq!(chain[0]["adapter"], "extractor");
    // Root last.
    assert_eq!(chain[1]["record_id"], root.0);
    assert!(chain[1]["derived_from"].is_null());
    assert_eq!(chain[1]["adapter"], "claude-code");
    // Summary discipline: chain entries must NOT carry `content`
    // — agents that want it re-call get_record.
    assert!(
        chain[0].get("content").is_none(),
        "lineage chain entries must NOT carry `content`; got {chain:?}"
    );
}

#[tokio::test]
async fn get_record_lineage_dangling_parent_exposes_missing_id() {
    let (server, _dir, store) = build_server();
    // Make a phantom parent id that doesn't correspond to any
    // real record.
    let phantom = RecordId::from_parts("claude-code", None, "phantom-root");
    // Leaf points at the phantom — chain walk will hit a dead end.
    let leaf = seed_record(
        &store,
        "extractor",
        "orphan",
        "fact with no ancestor",
        Some(&phantom),
    );

    let resp = server
        .handle(tool_call(
            "get_record",
            json!({"id": leaf.0, "include_lineage": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    let lineage = &payload["lineage"];
    assert_eq!(lineage["complete"], false);
    assert_eq!(lineage["missing_parent"], phantom.0);
    let chain = lineage["chain"].as_array().expect("chain array");
    assert_eq!(chain.len(), 1, "only the leaf resolves");
    assert_eq!(chain[0]["record_id"], leaf.0);
}

#[tokio::test]
async fn get_record_tools_list_schema_advertises_include_lineage() {
    let (server, _dir, _store) = build_server();
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let gr = tools
        .iter()
        .find(|t| t["name"] == "get_record")
        .expect("get_record must appear in tools/list");
    let props = &gr["inputSchema"]["properties"];
    assert!(
        props.get("include_lineage").is_some(),
        "get_record must advertise include_lineage: {gr}"
    );
    assert_eq!(props["include_lineage"]["type"], "boolean");
}
