//! Round-86 PR-78h MCP `source_show` end-to-end.
//!
//! Acceptance points:
//!
//!   1. **Admin-gated.** Hidden from non-admin tools/list, rejected
//!      by tools/call without admin. The `recent_import_errors`
//!      rows carry `native_path` + adapter-side error text — same
//!      sensitivity contract as `audit_tail`.
//!   2. **Listed when admin enabled.** Operator-facing surface
//!      surfaces through tools/list as expected.
//!   3. **Counts + errors.** Returns the source row, counts
//!      (records / chunks / tagged), and the most recent
//!      import_error rows scoped to that (adapter, instance).
//!   4. **Missing source errors loudly.** Typo'd ids become a
//!      tool error, not a `null` response.
//!   5. **No cross-instance leakage.** A mem0:self-hosted query
//!      doesn't return mem0:cloud errors.

use anamnesis_core::chunker::Chunker;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_mcp_server::{server::ADMIN_TOOLS, AnamnesisServer};
use anamnesis_store::{Store, UserTagOperation};
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
async fn source_show_is_listed_as_admin_tool() {
    assert!(
        ADMIN_TOOLS.contains(&"source_show"),
        "source_show must be admin-gated"
    );
}

#[tokio::test]
async fn source_show_hidden_from_tools_list_without_admin() {
    let (server, _dir, _store) = build_bundle(false);
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
        !names.contains(&"source_show"),
        "source_show must NOT appear in default tools/list; got {names:?}"
    );
}

#[tokio::test]
async fn source_show_rejected_when_admin_disabled() {
    let (server, _dir, _store) = build_bundle(false);
    let resp = server
        .handle(tool_call("source_show", json!({"source": "claude-code"})))
        .await;
    assert!(
        resp.error.is_some(),
        "source_show must error without admin gate; got {resp:?}"
    );
}

#[tokio::test]
async fn source_show_missing_source_returns_error_not_null() {
    let (server, _dir, _store) = build_bundle(true);
    let resp = server
        .handle(tool_call(
            "source_show",
            json!({"source": "no-such-adapter"}),
        ))
        .await;
    assert!(resp.error.is_some(), "must error on unknown source");
    let msg = resp.error.unwrap().message;
    assert!(
        msg.contains("source not found"),
        "error must say source not found; got {msg}"
    );
}

#[tokio::test]
async fn source_show_returns_counts_and_recent_errors_scoped_to_instance() {
    let (server, _dir, store) = build_bundle(true);
    // Register two mem0 instances; only seed errors in self-hosted.
    store
        .register_source("mem0", Some("self-hosted"), Some("/local"), None)
        .unwrap();
    store
        .register_source("mem0", Some("cloud"), Some("https://x"), None)
        .unwrap();
    // Seed a record on self-hosted + tag.
    let r = AnamnesisRecord {
        id: RecordId::from_parts("mem0", Some("self-hosted"), "rec"),
        source: SourceDescriptor {
            adapter: "mem0".into(),
            instance: Some("self-hosted".into()),
            version: "0".into(),
        },
        content: "body".into(),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: "rec".into(),
            native_path: None,
            captured_at: Utc::now(),
            raw_hash: "h-rec".into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let c = Chunker::default().chunk(&r.id, &r.content);
    store.upsert_record(&r, &c, None).unwrap();
    store
        .tag_record(&r.id, &["keep".into()], UserTagOperation::Add)
        .unwrap();
    // Log errors on both instances — but the self-hosted query
    // must only see its own.
    store
        .log_import_error(
            "mem0",
            Some("self-hosted"),
            Some("rec"),
            Some("/local/rec.json"),
            "parse",
            "self-error",
        )
        .unwrap();
    store
        .log_import_error(
            "mem0",
            Some("cloud"),
            Some("rec2"),
            None,
            "parse",
            "cloud-error",
        )
        .unwrap();

    let resp = server
        .handle(tool_call(
            "source_show",
            json!({"source": "mem0", "instance": "self-hosted"}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["source"]["adapter"], "mem0");
    assert_eq!(payload["source"]["instance"], "self-hosted");
    assert_eq!(payload["source"]["record_count"], 1);
    assert_eq!(payload["source"]["tagged_record_count"], 1);
    let errors = payload["recent_import_errors"].as_array().unwrap();
    assert_eq!(errors.len(), 1, "must NOT include the cloud-instance error");
    assert_eq!(errors[0]["error"], "self-error");
    assert_eq!(errors[0]["native_path"], "/local/rec.json");
    // Cross-check: cloud instance has its own error, isolated.
    let cloud_resp = server
        .handle(tool_call(
            "source_show",
            json!({"source": "mem0", "instance": "cloud"}),
        ))
        .await;
    let cloud_payload = extract_payload(&cloud_resp);
    let cloud_errors = cloud_payload["recent_import_errors"].as_array().unwrap();
    assert_eq!(cloud_errors.len(), 1);
    assert_eq!(cloud_errors[0]["error"], "cloud-error");
}

#[tokio::test]
async fn source_show_error_limit_clamps_and_caps() {
    let (server, _dir, store) = build_bundle(true);
    store
        .register_source("claude-code", None, Some("/cc"), None)
        .unwrap();
    // Log 15 errors — more than the default and the cap.
    for i in 0..15 {
        store
            .log_import_error(
                "claude-code",
                None,
                Some(&format!("n-{i}")),
                None,
                "parse",
                &format!("err-{i}"),
            )
            .unwrap();
    }

    // Default (no error_limit) = 5.
    let resp = server
        .handle(tool_call("source_show", json!({"source": "claude-code"})))
        .await;
    let payload = extract_payload(&resp);
    assert_eq!(payload["recent_import_errors"].as_array().unwrap().len(), 5);
    assert_eq!(payload["error_limit"], 5);

    // Explicit 100 must clamp to 10.
    let resp = server
        .handle(tool_call(
            "source_show",
            json!({"source": "claude-code", "error_limit": 100}),
        ))
        .await;
    let payload = extract_payload(&resp);
    assert_eq!(
        payload["recent_import_errors"].as_array().unwrap().len(),
        10
    );
    assert_eq!(payload["error_limit"], 10);
}

#[tokio::test]
async fn source_show_tools_list_schema_advertises_all_args() {
    let (server, _dir, _store) = build_bundle(true);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let ss = tools
        .iter()
        .find(|t| t["name"] == "source_show")
        .expect("source_show must be in admin tools/list");
    let props = &ss["inputSchema"]["properties"];
    for key in ["source", "instance", "error_limit"] {
        assert!(
            props.get(key).is_some(),
            "source_show must advertise {key:?}: {ss}"
        );
    }
}
