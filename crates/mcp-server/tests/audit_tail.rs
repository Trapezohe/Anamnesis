//! Round-84 PR-78f MCP `audit_tail` end-to-end.
//!
//! Acceptance points:
//!
//!   1. **Admin-gated.** Hidden from non-admin `tools/list` and
//!      rejected by `tools/call` without admin. The entries it
//!      surfaces (search queries, forget reasons, source
//!      locations) shouldn't be a back-door read for non-admin
//!      MCP agents.
//!   2. **Default redacted.** Even with admin enabled, the
//!      response carries only `line_no / timestamp / action /
//!      via / outcome` per entry. The full per-entry `detail`
//!      is opt-in via `include_detail: true`.
//!   3. **Filter parity with CLI.** `action` and `since`
//!      narrow the result set with the same grammar.
//!   4. **Schema advertises args.** tools/list (admin enabled)
//!      shows `limit`, `action`, `since`, `include_detail`.

use anamnesis_core::{Audit, AuditEntry};
use anamnesis_mcp_server::{server::ADMIN_TOOLS, AnamnesisServer};
use anamnesis_store::Store;
use serde_json::{json, Value};

struct TestBundle {
    server: AnamnesisServer,
    _data_dir: tempfile::TempDir,
    audit: Audit,
}

fn build_bundle(allow_admin: bool) -> TestBundle {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let db = data_dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");
    let server = AnamnesisServer::new(store, None, data_dir.path().to_path_buf())
        .with_admin_tools(allow_admin);
    let audit = Audit::new(data_dir.path());
    TestBundle {
        server,
        _data_dir: data_dir,
        audit,
    }
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
async fn audit_tail_is_listed_as_admin_tool() {
    assert!(
        ADMIN_TOOLS.contains(&"audit_tail"),
        "audit_tail must be admin-gated"
    );
}

#[tokio::test]
async fn audit_tail_hidden_from_tools_list_without_admin() {
    let bundle = build_bundle(false);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = bundle.server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(
        !names.contains(&"audit_tail"),
        "audit_tail must NOT appear in default tools/list; got {names:?}"
    );
}

#[tokio::test]
async fn audit_tail_rejected_when_admin_disabled() {
    let bundle = build_bundle(false);
    let resp = bundle
        .server
        .handle(tool_call("audit_tail", json!({})))
        .await;
    assert!(
        resp.error.is_some(),
        "audit_tail must error without admin gate; got {resp:?}"
    );
}

#[tokio::test]
async fn audit_tail_default_response_omits_detail() {
    let bundle = build_bundle(true);
    // Seed two entries directly via Audit (no MCP write needed
    // here — the test is about read shape).
    bundle.audit.record(AuditEntry::new(
        "forget",
        json!({"via": "cli", "outcome": "forgotten", "reason": "secret-reason"}),
    ));
    bundle.audit.record(AuditEntry::new(
        "search",
        json!({"via": "mcp", "query": "should-not-leak"}),
    ));

    let resp = bundle
        .server
        .handle(tool_call("audit_tail", json!({})))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["count"], 2);
    assert_eq!(payload["include_detail"], false);
    let entries = payload["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2);

    // Default response shape: line_no / timestamp / action / via
    // / outcome. NO `detail` key. NO `reason` / `query`.
    let forget = entries.iter().find(|e| e["action"] == "forget").unwrap();
    assert_eq!(forget["via"], "cli");
    assert_eq!(forget["outcome"], "forgotten");
    assert!(forget.get("detail").is_none());

    let serialised = serde_json::to_string(&payload).unwrap();
    assert!(
        !serialised.contains("secret-reason"),
        "default response must not leak `reason` field: {serialised}"
    );
    assert!(
        !serialised.contains("should-not-leak"),
        "default response must not leak `query` field: {serialised}"
    );
}

#[tokio::test]
async fn audit_tail_include_detail_returns_full_payload() {
    let bundle = build_bundle(true);
    bundle.audit.record(AuditEntry::new(
        "forget",
        json!({"via": "cli", "outcome": "forgotten", "reason": "opt-in-leak-ok"}),
    ));

    let resp = bundle
        .server
        .handle(tool_call(
            "audit_tail",
            json!({"include_detail": true, "action": "forget"}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["include_detail"], true);
    let entry = &payload["entries"][0];
    assert_eq!(entry["detail"]["reason"], "opt-in-leak-ok");
}

#[tokio::test]
async fn audit_tail_action_filter_narrows_results() {
    let bundle = build_bundle(true);
    for _ in 0..5 {
        bundle
            .audit
            .record(AuditEntry::new("search", json!({"via": "mcp"})));
    }
    bundle
        .audit
        .record(AuditEntry::new("forget", json!({"via": "cli"})));

    let resp = bundle
        .server
        .handle(tool_call("audit_tail", json!({"action": "forget"})))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    let entries = payload["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["action"], "forget");
    assert_eq!(payload["filter"]["action"], "forget");
}

#[tokio::test]
async fn audit_tail_invalid_since_returns_clear_error() {
    let bundle = build_bundle(true);
    let resp = bundle
        .server
        .handle(tool_call("audit_tail", json!({"since": "garbage"})))
        .await;
    assert!(resp.error.is_some(), "garbage --since must error");
    let msg = resp.error.unwrap().message;
    assert!(
        msg.contains("audit_tail.since"),
        "error must mention the parameter name; got {msg}"
    );
}

#[tokio::test]
async fn audit_tail_tools_list_schema_advertises_all_args() {
    let bundle = build_bundle(true);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = bundle.server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let at = tools
        .iter()
        .find(|t| t["name"] == "audit_tail")
        .expect("audit_tail must be in admin tools/list");
    let props = &at["inputSchema"]["properties"];
    for key in ["limit", "action", "since", "include_detail"] {
        assert!(
            props.get(key).is_some(),
            "audit_tail must advertise {key:?}: {at}"
        );
    }
}
