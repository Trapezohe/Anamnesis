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
    // Round 109 (PR-78ae): `format: "json"` marker pairs
    // with R92's `format: "csv"` on the CSV branch — MCP
    // clients can branch on `payload.format` without
    // probing for `entries[]` vs `csv`.
    assert_eq!(payload["format"], "json");
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
    for key in ["limit", "action", "since", "include_detail", "csv"] {
        assert!(
            props.get(key).is_some(),
            "audit_tail must advertise {key:?}: {at}"
        );
    }
    assert_eq!(props["csv"]["type"], "boolean");
}

// ─── Round-92 PR-78n: audit_tail csv (MCP parity with R91) ──────────

/// `csv: true` returns a `csv` string with the redacted summary
/// header + rows, NOT an `entries[]` array. Same field
/// discipline as the CLI `audit tail --csv`: never carries
/// `detail` / `reason` / `query`.
#[tokio::test]
async fn audit_tail_csv_returns_string_with_header_and_redacted_rows() {
    let bundle = build_bundle(true);
    bundle.audit.record(AuditEntry::new(
        "forget",
        json!({"via": "cli", "outcome": "forgotten", "reason": "secret-leak-canary"}),
    ));

    let resp = bundle
        .server
        .handle(tool_call(
            "audit_tail",
            json!({"action": "forget", "csv": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["format"], "csv");
    assert_eq!(payload["count"], 1);
    assert_eq!(payload["include_detail"], false);
    // `entries` absent — csv path uses a flat string instead.
    assert!(payload.get("entries").is_none());
    let csv = payload["csv"].as_str().unwrap();
    assert!(csv.starts_with("line_no,timestamp,action,via,outcome\n"));
    assert!(csv.contains(",forget,cli,forgotten\n"));
    assert!(
        !csv.contains("secret-leak-canary"),
        "csv must not leak reason: {csv}"
    );
}

/// `csv: true` + `include_detail: true` is a contradictory
/// operator intent — CSV is the redacted summary form, and
/// detail would either leak through or pretend the CSV was
/// full-detail. The handler returns a clear error.
#[tokio::test]
async fn audit_tail_csv_conflicts_with_include_detail() {
    let bundle = build_bundle(true);
    let resp = bundle
        .server
        .handle(tool_call(
            "audit_tail",
            json!({"csv": true, "include_detail": true}),
        ))
        .await;
    assert!(resp.error.is_some(), "must error on the conflict");
    let msg = resp.error.unwrap().message;
    assert!(
        msg.contains("mutually exclusive") || msg.contains("redacted-summary"),
        "error must explain the conflict; got {msg}"
    );
}

/// Empty audit log + `csv: true` still emits header-only — same
/// behaviour as the CLI, so scripts can branch uniformly.
#[tokio::test]
async fn audit_tail_csv_empty_log_emits_header_only() {
    let bundle = build_bundle(true);
    let resp = bundle
        .server
        .handle(tool_call("audit_tail", json!({"csv": true})))
        .await;
    assert!(resp.error.is_none());
    let payload = extract_payload(&resp);
    assert_eq!(payload["count"], 0);
    let csv = payload["csv"].as_str().unwrap();
    assert_eq!(csv.trim(), "line_no,timestamp,action,via,outcome");
}

// ─── Round-102 PR-78x: audit_tail action multi-value OR ────────────

/// Comma-separated `action` argument is an OR filter — `forget`
/// + `tag_record` rows survive, `search` rows drop. Symmetric
/// with the CLI's R102 `--action forget,search`. Response keeps
/// `filter.action` raw (back-compat with R84 / R91 / R92
/// clients) and adds the additive `filter.actions` array of
/// normalised tokens.
#[tokio::test]
async fn audit_tail_action_multi_value_or_filters_matching_set() {
    let bundle = build_bundle(true);
    // 6 noise + 1 forget + 1 tag_record so the OR target sits
    // squarely inside the file.
    for _ in 0..3 {
        bundle
            .audit
            .record(AuditEntry::new("search", json!({"via": "mcp"})));
    }
    bundle.audit.record(AuditEntry::new(
        "forget",
        json!({"via": "cli", "outcome": "forgotten"}),
    ));
    bundle.audit.record(AuditEntry::new(
        "tag_record",
        json!({"via": "mcp", "outcome": "tagged"}),
    ));
    for _ in 0..3 {
        bundle
            .audit
            .record(AuditEntry::new("search", json!({"via": "mcp"})));
    }

    let resp = bundle
        .server
        .handle(tool_call(
            "audit_tail",
            json!({"action": "forget, tag_record"}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    // Raw `action` echoed for back-compat.
    assert_eq!(payload["filter"]["action"], "forget, tag_record");
    // Normalised `actions` is the new source of truth.
    assert_eq!(
        payload["filter"]["actions"],
        json!(["forget", "tag_record"])
    );
    let entries = payload["entries"].as_array().unwrap();
    let actions: Vec<&str> = entries
        .iter()
        .map(|e| e["action"].as_str().unwrap())
        .collect();
    assert_eq!(actions, vec!["forget", "tag_record"]);
}

/// Multi-value OR is also honoured on the CSV path so scripts
/// can `audit_tail` with `csv: true, action: "forget,search"`
/// and get a flat header-plus-rows string of the union — same
/// redaction discipline as the single-action CSV (R92).
#[tokio::test]
async fn audit_tail_csv_action_multi_value_or_filters_matching_set() {
    let bundle = build_bundle(true);
    bundle.audit.record(AuditEntry::new(
        "forget",
        json!({"via": "cli", "outcome": "forgotten", "reason": "csv-canary"}),
    ));
    bundle
        .audit
        .record(AuditEntry::new("search", json!({"via": "mcp"})));
    bundle
        .audit
        .record(AuditEntry::new("import", json!({"via": "cli"})));

    let resp = bundle
        .server
        .handle(tool_call(
            "audit_tail",
            json!({"action": "forget,search", "csv": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["filter"]["actions"], json!(["forget", "search"]));
    assert_eq!(payload["count"], 2);
    let csv = payload["csv"].as_str().unwrap();
    assert!(csv.contains(",forget,cli,forgotten\n"));
    assert!(csv.contains(",search,mcp,"));
    assert!(!csv.contains(",import,"), "import row must be dropped");
    assert!(
        !csv.contains("csv-canary"),
        "csv must not leak reason even on multi-action: {csv}"
    );
}

/// Omitting `action` keeps `filter.action` null and
/// `filter.actions` an empty array — back-compat: no filter
/// returns every entry.
#[tokio::test]
async fn audit_tail_no_action_arg_emits_empty_actions_array() {
    let bundle = build_bundle(true);
    bundle
        .audit
        .record(AuditEntry::new("search", json!({"via": "mcp"})));

    let resp = bundle
        .server
        .handle(tool_call("audit_tail", json!({})))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert!(payload["filter"]["action"].is_null());
    assert_eq!(payload["filter"]["actions"], json!([]));
    assert_eq!(payload["count"], 1);
}
