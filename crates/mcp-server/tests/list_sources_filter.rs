//! Round-96 PR-78r MCP `list_sources { source?, instance? }`.
//!
//! Acceptance points:
//!
//!   1. **No filter** — back-compat with R0+R82: returns every
//!      registered source.
//!   2. **`source`** narrows `sources[]` to one adapter id.
//!   3. **`source + instance`** narrows further.
//!   4. **`stats` block unchanged** — top-level totals always
//!      reflect the whole store, never the filtered subset.
//!   5. **Schema advertises both args.**

use anamnesis_core::chunker::Chunker;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_mcp_server::AnamnesisServer;
use anamnesis_store::Store;
use chrono::Utc;
use serde_json::{json, Value};

fn build_bundle() -> (AnamnesisServer, tempfile::TempDir, Store) {
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

fn seed(store: &Store, adapter: &str, instance: Option<&str>, native: &str) {
    store
        .register_source(adapter, instance, Some("/tmp/x"), None)
        .unwrap();
    let id = RecordId::from_parts(adapter, instance, native);
    let r = AnamnesisRecord {
        id: id.clone(),
        source: SourceDescriptor {
            adapter: adapter.into(),
            instance: instance.map(str::to_owned),
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
            native_path: None,
            captured_at: Utc::now(),
            raw_hash: format!("h-{adapter}-{native}"),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let c = Chunker::default().chunk(&r.id, &r.content);
    store.upsert_record(&r, &c, None).unwrap();
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
async fn list_sources_no_filter_returns_all_sources() {
    let (server, _dir, store) = build_bundle();
    seed(&store, "claude-code", None, "a");
    seed(&store, "mem0", Some("prod"), "b");
    seed(&store, "mem0", Some("dev"), "c");

    let resp = server.handle(tool_call("list_sources", json!({}))).await;
    let payload = extract_payload(&resp);
    assert_eq!(payload["sources"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn list_sources_source_filter_narrows_array() {
    let (server, _dir, store) = build_bundle();
    seed(&store, "claude-code", None, "a");
    seed(&store, "mem0", Some("prod"), "b");
    seed(&store, "mem0", Some("dev"), "c");

    let resp = server
        .handle(tool_call("list_sources", json!({"source": "mem0"})))
        .await;
    let payload = extract_payload(&resp);
    let sources = payload["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 2);
    for s in sources {
        assert_eq!(s["adapter"], "mem0");
    }
}

#[tokio::test]
async fn list_sources_source_plus_instance_narrows_to_one() {
    let (server, _dir, store) = build_bundle();
    seed(&store, "mem0", Some("prod"), "b");
    seed(&store, "mem0", Some("dev"), "c");

    let resp = server
        .handle(tool_call(
            "list_sources",
            json!({"source": "mem0", "instance": "prod"}),
        ))
        .await;
    let payload = extract_payload(&resp);
    let sources = payload["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0]["instance"], "prod");
}

/// Top-level `stats` reflects the whole store, regardless of
/// the filter — back-compat with every R0 client that pinned
/// `stats.records` to "all records".
#[tokio::test]
async fn list_sources_stats_block_reflects_whole_store_under_filter() {
    let (server, _dir, store) = build_bundle();
    seed(&store, "claude-code", None, "a");
    seed(&store, "mem0", Some("prod"), "b");

    let resp = server
        .handle(tool_call("list_sources", json!({"source": "mem0"})))
        .await;
    let payload = extract_payload(&resp);
    // stats.records = both records, even though sources[] is
    // filtered to 1.
    assert_eq!(payload["stats"]["records"], 2);
    assert_eq!(payload["sources"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn list_sources_tools_list_schema_advertises_filter_args() {
    let (server, _dir, _store) = build_bundle();
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let ls = tools
        .iter()
        .find(|t| t["name"] == "list_sources")
        .expect("list_sources in tools/list");
    let props = &ls["inputSchema"]["properties"];
    assert_eq!(props["source"]["type"], "string");
    assert_eq!(props["instance"]["type"], "string");
    // Round 103: schema must advertise the comma-separated OR
    // semantic so MCP clients discover the new capability.
    let source_desc = props["source"]["description"].as_str().unwrap();
    assert!(
        source_desc.contains("comma-separated"),
        "list_sources.source description must mention multi-value: {source_desc}"
    );
    let instance_desc = props["instance"]["description"].as_str().unwrap();
    assert!(
        instance_desc.contains("comma-separated"),
        "list_sources.instance description must mention multi-value: {instance_desc}"
    );
}

// ─── Round-103 PR-78y: list_sources source multi-value OR ───────────

/// Comma-separated `source` becomes an OR filter — both
/// adapters' rows survive, the third drops. Symmetric with R102
/// `audit_tail.action` multi-value. Top-level `stats` block
/// still reflects the whole store (back-compat with R0+R96
/// clients).
#[tokio::test]
async fn list_sources_source_multi_value_or_filters_matching_set() {
    let (server, _dir, store) = build_bundle();
    seed(&store, "claude-code", None, "a");
    seed(&store, "mem0", Some("prod"), "b");
    seed(&store, "codex", None, "c");

    let resp = server
        .handle(tool_call(
            "list_sources",
            json!({"source": "mem0, claude-code"}),
        ))
        .await;
    let payload = extract_payload(&resp);
    let sources = payload["sources"].as_array().unwrap();
    let adapters: std::collections::HashSet<&str> = sources
        .iter()
        .map(|s| s["adapter"].as_str().unwrap())
        .collect();
    assert_eq!(
        adapters,
        ["claude-code", "mem0"].into_iter().collect(),
        "expected only the two matching adapters; got {sources:?}"
    );
    // stats unchanged — still reports whole-store totals.
    assert_eq!(payload["stats"]["records"], 3);
}

/// Multi-value `source` combined with `instance` is AND of the
/// adapter OR-set: row matches iff `adapter ∈ source-set` AND
/// `instance == instance-arg`. `mem0:dev` survives because it's
/// in both subsets; `mem0:prod` drops because of the instance
/// mismatch; `claude-code` drops because (default) instance ≠
/// `"dev"`.
#[tokio::test]
async fn list_sources_source_multi_value_with_instance_is_and_filter() {
    let (server, _dir, store) = build_bundle();
    seed(&store, "mem0", Some("prod"), "a");
    seed(&store, "mem0", Some("dev"), "b");
    seed(&store, "claude-code", None, "c");

    let resp = server
        .handle(tool_call(
            "list_sources",
            json!({"source": "mem0,claude-code", "instance": "dev"}),
        ))
        .await;
    let payload = extract_payload(&resp);
    let sources = payload["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0]["adapter"], "mem0");
    assert_eq!(sources[0]["instance"], "dev");
}

/// Round 115: `instance: "prod,dev"` is OR on the source list
/// presentation path. Top-level stats still reflect the whole
/// store, same as the earlier `source` filter.
#[tokio::test]
async fn list_sources_instance_multi_value_or_filters_matching_set() {
    let (server, _dir, store) = build_bundle();
    seed(&store, "mem0", Some("prod"), "a");
    seed(&store, "mem0", Some("dev"), "b");
    seed(&store, "mem0", Some("qa"), "c");

    let resp = server
        .handle(tool_call(
            "list_sources",
            json!({"source": "mem0", "instance": "prod, dev"}),
        ))
        .await;
    let payload = extract_payload(&resp);
    let sources = payload["sources"].as_array().unwrap();
    let instances: std::collections::BTreeSet<&str> = sources
        .iter()
        .map(|s| s["instance"].as_str().unwrap())
        .collect();
    assert_eq!(instances, ["dev", "prod"].into_iter().collect());
    assert_eq!(payload["stats"]["records"], 3);
}
