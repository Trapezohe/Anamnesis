//! Round-87 PR-78i MCP `search_memories({ explain: true })`.
//!
//! Acceptance points:
//!
//!   1. **Back-compat default.** Default response has no
//!      `explain` block per result — every existing R8/R71
//!      consumer keeps working byte-identical.
//!   2. **Opt-in attaches numeric block.** `explain: true`
//!      attaches `record_score / best_chunk_rrf_score /
//!      kind_boost / stages.fts.{rank, raw_score, rrf_contribution}`
//!      and `stages.rrf_k`.
//!   3. **Privacy contract.** The explain block carries only
//!      numerics — no chunk_id, no query, no snippet beyond
//!      what `results[]` already exposes outside of explain.
//!   4. **Orthogonal to `trace`.** `explain` is per-hit;
//!      `trace` is top-level stage timings. Both flags can be
//!      true at once.
//!   5. **Schema advertises arg.** tools/list shows `explain`.

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

fn seed(store: &Store, native: &str, content: &str) -> RecordId {
    let id = RecordId::from_parts("claude-code", None, native);
    let r = AnamnesisRecord {
        id: id.clone(),
        source: SourceDescriptor {
            adapter: "claude-code".into(),
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
            raw_hash: format!("h-{native}"),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let c = Chunker::default().chunk(&r.id, &r.content);
    store.upsert_record(&r, &c, None).unwrap();
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
async fn search_explain_default_response_has_no_explain_block() {
    let (server, _dir, store) = build_server();
    seed(&store, "x", "uniqueExplainBackcompat body");

    let resp = server
        .handle(tool_call(
            "search_memories",
            json!({"query": "uniqueExplainBackcompat", "mode": "fulltext"}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    let row = &payload["results"][0];
    assert!(
        row.get("explain").is_none(),
        "default search must NOT carry explain; got {row}"
    );
}

#[tokio::test]
async fn search_explain_opt_in_returns_numeric_breakdown() {
    let (server, _dir, store) = build_server();
    seed(&store, "x", "uniqueExplainMcp body");

    let resp = server
        .handle(tool_call(
            "search_memories",
            json!({
                "query": "uniqueExplainMcp",
                "mode": "fulltext",
                "explain": true,
            }),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    let row = &payload["results"][0];
    let explain = &row["explain"];
    assert!(!explain.is_null());
    let record_score = explain["record_score"].as_f64().unwrap();
    let best = explain["best_chunk_rrf_score"].as_f64().unwrap();
    let boost = explain["kind_boost"].as_f64().unwrap();
    assert!(
        (record_score - (best + boost)).abs() < 1e-9,
        "record_score = best + kind_boost; got {explain}"
    );
    let stages = &explain["stages"];
    let fts = &stages["fts"];
    let rrf_k = stages["rrf_k"].as_f64().unwrap();
    let rank = fts["rank"].as_u64().unwrap() as f64;
    let contribution = fts["rrf_contribution"].as_f64().unwrap();
    assert!(
        (contribution - 1.0 / (rrf_k + rank)).abs() < 1e-9,
        "rrf_contribution = 1/(rrf_k+rank); got {explain}"
    );
}

/// Privacy contract: serialise the `explain` block and verify
/// it doesn't smuggle forbidden text fields (record_id, chunk_id,
/// query, snippet content). The forbidden fields stay outside
/// `explain` — they live elsewhere in the result row.
#[tokio::test]
async fn search_explain_block_carries_only_numerics() {
    let (server, _dir, store) = build_server();
    seed(&store, "x", "uniqueExplainPrivacy body");

    let resp = server
        .handle(tool_call(
            "search_memories",
            json!({
                "query": "uniqueExplainPrivacy",
                "mode": "fulltext",
                "explain": true,
            }),
        ))
        .await;
    let payload = extract_payload(&resp);
    let explain = &payload["results"][0]["explain"];
    let s = serde_json::to_string(explain).unwrap();
    for forbidden in [
        "chunk_id",
        "record_id",
        "uniqueExplainPrivacy",
        "snippet",
        "content",
        "query",
        "native_path",
    ] {
        assert!(
            !s.contains(forbidden),
            "explain block must NOT contain {forbidden:?}: {s}"
        );
    }
}

/// `explain` and `trace` are orthogonal — both can be set, and
/// each populates its own block.
#[tokio::test]
async fn search_explain_and_trace_coexist() {
    let (server, _dir, store) = build_server();
    seed(&store, "x", "uniqueExplainTrace body");

    let resp = server
        .handle(tool_call(
            "search_memories",
            json!({
                "query": "uniqueExplainTrace",
                "mode": "fulltext",
                "trace": true,
                "explain": true,
            }),
        ))
        .await;
    let payload = extract_payload(&resp);
    assert!(payload.get("trace").is_some(), "trace block must be present");
    assert!(
        payload["results"][0].get("explain").is_some(),
        "explain block must be present on the first result"
    );
}

#[tokio::test]
async fn search_explain_tools_list_schema_advertises_explain() {
    let (server, _dir, _store) = build_server();
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let sm = tools
        .iter()
        .find(|t| t["name"] == "search_memories")
        .expect("search_memories must be in tools/list");
    let props = &sm["inputSchema"]["properties"];
    assert!(
        props.get("explain").is_some(),
        "search_memories must advertise `explain`: {sm}"
    );
    assert_eq!(props["explain"]["type"], "boolean");
}
