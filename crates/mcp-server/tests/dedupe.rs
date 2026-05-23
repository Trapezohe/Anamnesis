//! Round-77 PR-77 MCP `dedupe` tool — admin-free read-only
//! diagnostic over `records.raw_hash` collisions.
//!
//! Acceptance points:
//!
//!   1. **Not admin-gated.** Appears in default `tools/list` and
//!      is callable without `allow_admin_tools`. The action half
//!      (`forget_record`) is still admin-gated; this is just the
//!      report.
//!   2. **Redaction by default.** `raw_hash` and `native_path` are
//!      absent unless `include_sensitive=true`. Same privacy
//!      discipline as `list_forgotten` (R74).
//!   3. **Group shape.** Each group reports `record_count` and a
//!      `records[]` array carrying the minimum operator-decision
//!      fields (record_id / adapter / instance / native_id).

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
    let make = |adapter: &str, native: &str, hash: &str| AnamnesisRecord {
        id: RecordId::from_parts(adapter, None, native),
        source: SourceDescriptor {
            adapter: adapter.into(),
            instance: None,
            version: "0".into(),
        },
        content: format!("{adapter}|{native} content"),
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
            raw_hash: hash.into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    for r in [
        make("claude-code", "alpha", "secretMarkerH"),
        make("mem0", "beta", "secretMarkerH"),
        make("claude-code", "gamma", "h-singleton"),
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
async fn dedupe_appears_in_default_tools_list_without_admin() {
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
        names.contains(&"dedupe"),
        "dedupe must be in non-admin tools/list; got {names:?}"
    );
}

#[tokio::test]
async fn dedupe_callable_without_admin() {
    let (server, _data) = build_bundle(false);
    let resp = server.handle(tool_call("dedupe", json!({}))).await;
    assert!(
        resp.error.is_none(),
        "dedupe must be callable without admin gate; got {:?}",
        resp.error
    );
    let payload = extract_payload(&resp);
    assert_eq!(payload["count"], 1);
    assert_eq!(payload["groups"][0]["record_count"], 2);
}

#[tokio::test]
async fn dedupe_default_payload_redacts_sensitive_fields() {
    let (server, _data) = build_bundle(false);
    let body = server.handle(tool_call("dedupe", json!({}))).await;
    let payload = extract_payload(&body);
    // Round 108 (PR-78ad): `format: "json"` marker pairs
    // with the R107 CSV path's `format: "csv"` so MCP
    // clients can branch on `payload.format` without probing
    // for `csv` vs `groups[]`.
    assert_eq!(payload["format"], "json");
    assert_eq!(payload["sensitive_included"], false);
    let group = &payload["groups"][0];
    assert!(
        group.get("raw_hash").is_none(),
        "raw_hash must be redacted: {group}"
    );
    let row = &group["records"][0];
    assert_eq!(row["has_native_path"], true);
    assert!(row.get("native_path").is_none());
    // Marker leak check against the full serialised payload.
    let serialised = serde_json::to_string(&payload).unwrap();
    assert!(
        !serialised.contains("secretMarkerH"),
        "raw_hash marker must not appear in redacted payload: {serialised}"
    );
    assert!(
        !serialised.contains("/tmp/claude-code/alpha.md"),
        "native_path must not appear in redacted payload: {serialised}"
    );
}

#[tokio::test]
async fn dedupe_include_sensitive_reveals_fields() {
    let (server, _data) = build_bundle(false);
    let body = server
        .handle(tool_call("dedupe", json!({"include_sensitive": true})))
        .await;
    let payload = extract_payload(&body);
    assert_eq!(payload["sensitive_included"], true);
    let group = &payload["groups"][0];
    assert_eq!(group["raw_hash"], "secretMarkerH");
    let rows = group["records"].as_array().unwrap();
    let paths: Vec<&str> = rows
        .iter()
        .filter_map(|r| r["native_path"].as_str())
        .collect();
    assert_eq!(paths.len(), 2);
}

// ─── Round-80: source / instance filter ────────────────────────────

/// Build a fixture with two duplicate groups so the filter has
/// something to narrow.
///   * h-mixed: mem0 + claude-code (filter target)
///   * h-cc: two claude-code records (irrelevant under
///     `source=mem0`)
fn build_bundle_two_groups() -> (AnamnesisServer, tempfile::TempDir) {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let db = data_dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");
    let make = |adapter: &str, native: &str, hash: &str| AnamnesisRecord {
        id: RecordId::from_parts(adapter, None, native),
        source: SourceDescriptor {
            adapter: adapter.into(),
            instance: None,
            version: "0".into(),
        },
        content: format!("{adapter}|{native} content"),
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
            raw_hash: hash.into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    for r in [
        make("mem0", "m1", "h-mixed"),
        make("claude-code", "c1", "h-mixed"),
        make("claude-code", "x1", "h-cc"),
        make("claude-code", "x2", "h-cc"),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
    let server =
        AnamnesisServer::new(store, None, data_dir.path().to_path_buf()).with_admin_tools(false);
    (server, data_dir)
}

/// `source` arg is callable without admin and echoed back in the
/// payload so an MCP client can render "filter: source=mem0" in
/// its UI without re-tracking the request.
#[tokio::test]
async fn dedupe_source_filter_scopes_groups_and_echoes_filter() {
    let (server, _data) = build_bundle_two_groups();
    let body = server
        .handle(tool_call(
            "dedupe",
            json!({"source": "mem0", "include_sensitive": true}),
        ))
        .await;
    assert!(body.error.is_none(), "dedupe with source must succeed");
    let payload = extract_payload(&body);
    assert_eq!(payload["count"], 1, "h-cc group filtered out");
    assert_eq!(payload["filter"]["source"], "mem0");
    assert!(payload["filter"]["instance"].is_null());
    let group = &payload["groups"][0];
    assert_eq!(group["raw_hash"], "h-mixed");
    // Both adapters surfaced in the matching group.
    let adapters: std::collections::BTreeSet<String> = group["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["adapter"].as_str().unwrap().to_string())
        .collect();
    assert!(adapters.contains("mem0"));
    assert!(adapters.contains("claude-code"));
}

/// Empty-string args are normalised to absent (so a client that
/// always sends `{"source": ""}` doesn't accidentally filter on
/// the empty source — there is no such adapter).
#[tokio::test]
async fn dedupe_empty_string_source_is_treated_as_absent() {
    let (server, _data) = build_bundle_two_groups();
    let body = server
        .handle(tool_call("dedupe", json!({"source": ""})))
        .await;
    assert!(body.error.is_none());
    let payload = extract_payload(&body);
    assert_eq!(
        payload["count"], 2,
        "empty source string must not filter; got {payload}"
    );
    assert!(payload["filter"]["source"].is_null());
}

/// tools/list schema advertises the new `source` and `instance`
/// optional string args.
#[tokio::test]
async fn dedupe_tools_list_advertises_source_and_instance_args() {
    let (server, _data) = build_bundle(false);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let dedupe = tools
        .iter()
        .find(|t| t["name"] == "dedupe")
        .expect("dedupe in tools/list");
    let props = &dedupe["inputSchema"]["properties"];
    assert_eq!(
        props["source"]["type"], "string",
        "dedupe must advertise `source` arg: {dedupe}"
    );
    assert_eq!(
        props["instance"]["type"], "string",
        "dedupe must advertise `instance` arg: {dedupe}"
    );
}

// ─── Round-98 PR-78t: MCP dedupe include_counts ────────────────────

#[tokio::test]
async fn dedupe_default_response_has_no_counts_block() {
    let (server, _data) = build_bundle(false);
    let resp = server.handle(tool_call("dedupe", json!({}))).await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert!(
        payload.get("counts").is_none(),
        "default dedupe must not carry counts; got {payload}"
    );
}

#[tokio::test]
async fn dedupe_include_counts_reflects_full_set_ignoring_limit() {
    let (server, _data) = build_bundle_two_groups();
    let resp = server
        .handle(tool_call(
            "dedupe",
            json!({"limit": 1, "include_counts": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["count"], 1, "rows respect limit");
    let counts = &payload["counts"];
    // seed_two_groups builds h-mixed (mem0 + claude-code) and
    // h-cc (2× claude-code) — 2 groups, 4 records.
    assert_eq!(counts["total_groups"], 2);
    assert_eq!(counts["total_records"], 4);
    let by_source = counts["by_source"].as_array().unwrap();
    let cc = by_source
        .iter()
        .find(|b| b["adapter"] == "claude-code")
        .unwrap();
    let mem = by_source.iter().find(|b| b["adapter"] == "mem0").unwrap();
    assert_eq!(cc["duplicate_record_count"], 3);
    assert_eq!(mem["duplicate_record_count"], 1);
}

#[tokio::test]
async fn dedupe_counts_block_carries_no_sensitive_fields() {
    let (server, _data) = build_bundle_two_groups();
    let resp = server
        .handle(tool_call("dedupe", json!({"include_counts": true})))
        .await;
    let payload = extract_payload(&resp);
    let counts_str = serde_json::to_string(&payload["counts"]).unwrap();
    for forbidden in ["h-mixed", "h-cc", "raw_hash", "native_path", "native_id"] {
        assert!(
            !counts_str.contains(forbidden),
            "counts must not leak {forbidden:?}: {counts_str}"
        );
    }
}

#[tokio::test]
async fn dedupe_tools_list_schema_advertises_include_counts() {
    let (server, _data) = build_bundle(false);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let dedupe = tools
        .iter()
        .find(|t| t["name"] == "dedupe")
        .expect("dedupe in tools/list");
    let props = &dedupe["inputSchema"]["properties"];
    assert_eq!(props["include_counts"]["type"], "boolean");
}

// ─── Round-104 PR-78z: dedupe source multi-value OR ────────────────

/// 3-group fixture with adapter-distinct groups (mem0 /
/// claude-code / codex) so the OR filter is unambiguous.
fn build_bundle_three_groups() -> (AnamnesisServer, tempfile::TempDir) {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let db = data_dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");
    let make = |adapter: &str, native: &str, hash: &str| AnamnesisRecord {
        id: RecordId::from_parts(adapter, None, native),
        source: SourceDescriptor {
            adapter: adapter.into(),
            instance: None,
            version: "0".into(),
        },
        content: format!("{adapter}|{native} content"),
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
            raw_hash: hash.into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    for r in [
        make("mem0", "m1", "h-mem"),
        make("mem0", "m2", "h-mem"),
        make("claude-code", "c1", "h-cc"),
        make("claude-code", "c2", "h-cc"),
        make("codex", "x1", "h-cx"),
        make("codex", "x2", "h-cx"),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
    let server =
        AnamnesisServer::new(store, None, data_dir.path().to_path_buf()).with_admin_tools(false);
    (server, data_dir)
}

fn build_bundle_instance_groups() -> (AnamnesisServer, tempfile::TempDir) {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let db = data_dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");
    let make = |instance: &str, native: &str, hash: &str| AnamnesisRecord {
        id: RecordId::from_parts("mem0", Some(instance), native),
        source: SourceDescriptor {
            adapter: "mem0".into(),
            instance: Some(instance.into()),
            version: "0".into(),
        },
        content: format!("mem0:{instance}|{native} content"),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: native.into(),
            native_path: Some(format!("/tmp/mem0/{instance}/{native}.md")),
            captured_at: Utc::now(),
            raw_hash: hash.into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    for r in [
        make("prod", "p1", "h-prod"),
        make("prod", "p2", "h-prod"),
        make("dev", "d1", "h-dev"),
        make("qa", "q1", "h-dev"),
        make("qa", "q2", "h-qa"),
        make("qa", "q3", "h-qa"),
    ] {
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
    let server =
        AnamnesisServer::new(store, None, data_dir.path().to_path_buf()).with_admin_tools(false);
    (server, data_dir)
}

/// `source: "mem0,claude-code"` is the OR filter on MCP —
/// mem0 + claude-code groups survive, codex drops. Symmetric
/// with R103 list_sources / R102 audit_tail.
#[tokio::test]
async fn dedupe_source_multi_value_or_filters_matching_groups() {
    let (server, _data) = build_bundle_three_groups();
    let resp = server
        .handle(tool_call(
            "dedupe",
            json!({"source": "mem0,claude-code", "include_sensitive": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["count"], 2);
    let hashes: std::collections::BTreeSet<&str> = payload["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["raw_hash"].as_str().unwrap())
        .collect();
    assert_eq!(hashes, ["h-cc", "h-mem"].into_iter().collect());
    // `filter.source` is back-compat — keeps the raw operator
    // input. The new multi-value behaviour lives in the store
    // filter, not in the wire shape.
    assert_eq!(payload["filter"]["source"], "mem0,claude-code");
}

/// `include_counts` honours the same multi-source eligibility:
/// `total_groups` is the eligible-only set; `by_source[]` only
/// reports records from surviving groups.
#[tokio::test]
async fn dedupe_source_multi_value_or_counts_respect_filter() {
    let (server, _data) = build_bundle_three_groups();
    let resp = server
        .handle(tool_call(
            "dedupe",
            json!({"source": "mem0,claude-code", "include_counts": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    let counts = &payload["counts"];
    assert_eq!(counts["total_groups"], 2);
    assert_eq!(counts["total_records"], 4);
    let adapters: std::collections::BTreeSet<&str> = counts["by_source"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["adapter"].as_str().unwrap())
        .collect();
    assert!(adapters.contains("mem0"));
    assert!(adapters.contains("claude-code"));
    assert!(
        !adapters.contains("codex"),
        "codex must be excluded from by_source: {adapters:?}"
    );
}

// ─── Round-115 PR-78ak: dedupe instance multi-value OR ──────────────

#[tokio::test]
async fn dedupe_instance_multi_value_or_filters_matching_groups() {
    let (server, _data) = build_bundle_instance_groups();
    let resp = server
        .handle(tool_call(
            "dedupe",
            json!({"source": "mem0", "instance": "prod,dev", "include_sensitive": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["count"], 2);
    let hashes: std::collections::BTreeSet<&str> = payload["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["raw_hash"].as_str().unwrap())
        .collect();
    assert_eq!(hashes, ["h-dev", "h-prod"].into_iter().collect());
    assert_eq!(payload["filter"]["instance"], "prod,dev");
}

#[tokio::test]
async fn dedupe_instance_multi_value_or_counts_respect_filter() {
    let (server, _data) = build_bundle_instance_groups();
    let resp = server
        .handle(tool_call(
            "dedupe",
            json!({"source": "mem0", "instance": "prod,dev", "include_counts": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    let counts = &payload["counts"];
    assert_eq!(counts["total_groups"], 2);
    assert_eq!(counts["total_records"], 4);
    let has_qa_sibling = counts["by_source"]
        .as_array()
        .unwrap()
        .iter()
        .any(|b| b["instance"] == "qa" && b["duplicate_record_count"] == 1);
    assert!(
        has_qa_sibling,
        "whole-group counts must keep qa sibling: {counts}"
    );
}

/// Schema advertises the multi-value filter capabilities so MCP
/// clients can surface it in autocomplete / docs.
#[tokio::test]
async fn dedupe_tools_list_schema_advertises_multi_value_filters() {
    let (server, _data) = build_bundle(false);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let dedupe = tools
        .iter()
        .find(|t| t["name"] == "dedupe")
        .expect("dedupe in tools/list");
    let source_desc = dedupe["inputSchema"]["properties"]["source"]["description"]
        .as_str()
        .unwrap();
    assert!(
        source_desc.contains("comma-separated"),
        "dedupe.source description must mention multi-value: {source_desc}"
    );
    let instance_desc = dedupe["inputSchema"]["properties"]["instance"]["description"]
        .as_str()
        .unwrap();
    assert!(
        instance_desc.contains("comma-separated"),
        "dedupe.instance description must mention multi-value: {instance_desc}"
    );
}

// ─── Round-107 PR-78ac: dedupe csv (MCP parity with CLI) ───────────

/// `csv: true` returns a flat `csv` string instead of `groups[]`
/// + `format: "csv"` marker. Empty result still emits header-
/// only so scripts can branch uniformly.
#[tokio::test]
async fn dedupe_csv_empty_returns_header_only() {
    let (server, _data) = build_bundle(false);
    // build_bundle already plants 3 records with two sharing
    // `secretMarkerH` — that's a 2-record group, so the empty
    // case needs its own minimal bundle.
    let data_dir = tempfile::tempdir().unwrap();
    let db = data_dir.path().join("anamnesis.sqlite");
    let _store = Store::open(&db).expect("open store");
    let empty_server = AnamnesisServer::new(
        Store::open(&db).expect("re-open store"),
        None,
        data_dir.path().to_path_buf(),
    );

    let resp = empty_server
        .handle(tool_call("dedupe", json!({"csv": true})))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["count"], 0);
    assert_eq!(payload["format"], "csv");
    assert!(payload.get("groups").is_none());
    assert!(payload.get("counts").is_none());
    let csv = payload["csv"].as_str().unwrap();
    assert_eq!(
        csv.trim(),
        "group_index,record_id,adapter,instance,native_id,created_at,updated_at,has_native_path,record_count"
    );
    // Drop unused server.
    let _ = server;
}

/// `csv: true` emits redacted summary rows. `raw_hash` never
/// appears (group membership via `group_index`); `native_path`
/// never appears (`has_native_path` boolean instead).
#[tokio::test]
async fn dedupe_csv_returns_redacted_rows_with_group_index_membership() {
    let (server, _data) = build_bundle(false);

    let resp = server
        .handle(tool_call("dedupe", json!({"csv": true})))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["format"], "csv");
    let csv = payload["csv"].as_str().unwrap();

    // raw_hash from the fixture (build_bundle plants
    // `secretMarkerH`) must NEVER appear in CSV.
    assert!(
        !csv.contains("secretMarkerH"),
        "csv must not leak raw_hash: {csv}"
    );
    // native_path must not appear either (build_bundle plants
    // `/tmp/claude-code/alpha.md` etc.).
    assert!(
        !csv.contains("/tmp/claude-code/"),
        "csv must not leak native_path: {csv}"
    );
    assert!(
        !csv.contains("/tmp/mem0/"),
        "csv must not leak native_path: {csv}"
    );

    // 2-record group with claude-code+mem0 records both
    // starting at group_index 0.
    let lines: Vec<&str> = csv.lines().collect();
    assert!(lines.len() >= 3, "header + 2 rows minimum: {csv}");
    assert!(lines[1].starts_with("0,"), "first row group_index=0");
    assert!(
        lines[2].starts_with("0,"),
        "sibling row shares group_index=0"
    );
}

/// `csv: true` + `include_sensitive: true` is contradictory.
/// Handler errors with a clear message.
#[tokio::test]
async fn dedupe_csv_conflicts_with_include_sensitive() {
    let (server, _data) = build_bundle(false);
    let resp = server
        .handle(tool_call(
            "dedupe",
            json!({"csv": true, "include_sensitive": true}),
        ))
        .await;
    assert!(resp.error.is_some(), "must error on the conflict");
    let msg = resp.error.unwrap().message;
    assert!(
        msg.contains("mutually exclusive") || msg.contains("redacted-summary"),
        "error must explain the conflict; got {msg}"
    );
}

/// `csv: true` + `include_counts: true` is also contradictory.
/// CSV is flat rows; counts have no place there.
#[tokio::test]
async fn dedupe_csv_conflicts_with_include_counts() {
    let (server, _data) = build_bundle(false);
    let resp = server
        .handle(tool_call(
            "dedupe",
            json!({"csv": true, "include_counts": true}),
        ))
        .await;
    assert!(resp.error.is_some(), "must error on the conflict");
}

/// Schema advertises the `csv` arg so MCP clients can discover
/// the new capability via tools/list.
#[tokio::test]
async fn dedupe_tools_list_schema_advertises_csv() {
    let (server, _data) = build_bundle(false);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let dedupe = tools
        .iter()
        .find(|t| t["name"] == "dedupe")
        .expect("dedupe in tools/list");
    let props = &dedupe["inputSchema"]["properties"];
    assert_eq!(props["csv"]["type"], "boolean");
}

// ─── Round-117 PR-78al: dedupe JSON summary line ───────────────────

/// `dedupe` JSON response carries top-level `summary` reporting
/// group count, limit, source/instance OR filter, sensitive +
/// counts state. CSV path doesn't carry it. `raw_hash` MUST
/// NOT appear in summary even when seeded raw_hash is present
/// in the fixture (`secretMarkerH`).
#[tokio::test]
async fn dedupe_json_carries_top_level_summary_line() {
    let (server, _data) = build_bundle(false);
    let resp = server.handle(tool_call("dedupe", json!({}))).await;
    let payload = extract_payload(&resp);
    let summary = payload["summary"]
        .as_str()
        .expect("dedupe JSON must carry `summary`");
    assert!(
        summary.contains("1 duplicate group(s) returned"),
        "summary must declare count: {summary}"
    );
    assert!(
        summary.contains("source filter: all sources"),
        "no-source summary must say `all sources`: {summary}"
    );
    assert!(
        summary.contains("sensitive: redacted"),
        "default sensitive state must surface: {summary}"
    );
    assert!(
        summary.contains("counts: omitted"),
        "default counts state must surface: {summary}"
    );
    // Privacy: `raw_hash` must NEVER appear.
    assert!(
        !summary.contains("secretMarkerH"),
        "summary must not leak raw_hash: {summary}"
    );
}

/// Multi-value `source` OR is reflected in the summary clause.
#[tokio::test]
async fn dedupe_json_summary_reflects_multi_value_source_filter() {
    let (server, _data) = build_bundle(false);
    let resp = server
        .handle(tool_call("dedupe", json!({"source": "mem0,claude-code"})))
        .await;
    let payload = extract_payload(&resp);
    let summary = payload["summary"].as_str().unwrap();
    assert!(
        summary.contains("source filter: mem0 OR claude-code"),
        "summary must echo OR source filter: {summary}"
    );
}

/// CSV branch must NOT carry `summary` (JSON-only contract).
#[tokio::test]
async fn dedupe_csv_response_has_no_summary_field() {
    let (server, _data) = build_bundle(false);
    let resp = server
        .handle(tool_call("dedupe", json!({"csv": true})))
        .await;
    let payload = extract_payload(&resp);
    assert_eq!(payload["format"], "csv");
    assert!(
        payload.get("summary").is_none(),
        "CSV payload must not carry `summary`; got {payload}"
    );
}
