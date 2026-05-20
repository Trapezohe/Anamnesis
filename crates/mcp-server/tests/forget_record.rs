//! Round-73 PR-72b: admin-gated MCP `forget_record` end-to-end.
//!
//! Acceptance points:
//!
//!   1. **Admin gate respected.** Without `with_admin_tools(true)`,
//!      `tools/call forget_record` is rejected and `tools/list`
//!      hides the tool. Same pattern as `import_source` (R0 PR-#10).
//!
//!   2. **Live record forget succeeds.** Admin-on client can forget
//!      a record by id; the live `records` row goes; `get_record`
//!      returns null afterward; payload carries the structured
//!      tombstone fields.
//!
//!   3. **Idempotent re-forget.** A second `forget_record` on the
//!      same id returns `"outcome": "already-forgotten"` without
//!      breaking, with the original tombstone metadata preserved.
//!
//!   4. **NotFound is an error.** Calling on an id that never
//!      existed errors loudly — silently returning success would
//!      let the operator believe a guarantee was made when it
//!      wasn't (no tombstone was written).
//!
//!   5. **Audit parity.** The MCP forget writes a `"forget"`
//!      audit entry with the same shape the CLI does, with
//!      `via: "mcp"` for entry-point provenance.

use anamnesis_core::chunker::Chunker;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_mcp_server::{server::ADMIN_TOOLS, AnamnesisServer};
use anamnesis_store::Store;
use chrono::Utc;
use serde_json::{json, Value};

struct TestBundle {
    server: AnamnesisServer,
    /// Holds the temp dir alive for the duration of the test.
    _data_dir: tempfile::TempDir,
    audit_dir: std::path::PathBuf,
}

/// Build a server + pre-seed a single forget-able record so each
/// test starts from a known live state.
fn build_bundle(allow_admin: bool) -> (TestBundle, RecordId) {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let db_path = data_dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db_path).expect("open store");

    let id = RecordId::from_parts("claude-code", None, "doomed-r73");
    let record = AnamnesisRecord {
        id: id.clone(),
        source: SourceDescriptor {
            adapter: "claude-code".into(),
            instance: None,
            version: "0".into(),
        },
        content: "secret content to be forgotten".into(),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: "doomed-r73".into(),
            native_path: Some("/tmp/doomed.md".into()),
            captured_at: Utc::now(),
            raw_hash: "h-doomed-r73".into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let chunks = Chunker::default().chunk(&record.id, &record.content);
    store.upsert_record(&record, &chunks, None).unwrap();

    let server = AnamnesisServer::new(store, None, data_dir.path().to_path_buf())
        .with_admin_tools(allow_admin);

    let audit_dir = data_dir.path().to_path_buf();
    (
        TestBundle {
            server,
            _data_dir: data_dir,
            audit_dir,
        },
        id,
    )
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

fn read_audit_lines(dir: &std::path::Path) -> Vec<Value> {
    let raw = std::fs::read_to_string(dir.join("audit.log")).unwrap_or_default();
    raw.lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str::<Value>(l).expect("audit line is json"))
        .collect()
}

#[tokio::test]
async fn forget_record_is_listed_as_admin_tool() {
    assert!(
        ADMIN_TOOLS.contains(&"forget_record"),
        "forget_record must be admin-gated"
    );
}

#[tokio::test]
async fn forget_record_hidden_from_tools_list_without_admin() {
    let (bundle, _id) = build_bundle(false);
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
        !names.contains(&"forget_record"),
        "forget_record must NOT appear in default tools/list; got {names:?}"
    );
}

#[tokio::test]
async fn forget_record_visible_in_tools_list_with_admin() {
    let (bundle, _id) = build_bundle(true);
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
        names.contains(&"forget_record"),
        "forget_record must appear in admin tools/list; got {names:?}"
    );
}

#[tokio::test]
async fn forget_record_rejected_when_admin_disabled() {
    let (bundle, id) = build_bundle(false);
    let resp = bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": id.0, "reason": "test"}),
        ))
        .await;
    assert!(
        resp.error.is_some(),
        "forget_record must error without admin gate; got {resp:?}"
    );
    let err = resp.error.unwrap();
    let msg = err.message.to_lowercase();
    assert!(
        msg.contains("admin"),
        "error must mention admin gate; got {msg}"
    );
}

#[tokio::test]
async fn forget_record_deletes_record_and_returns_structured_payload() {
    let (bundle, id) = build_bundle(true);
    let resp = bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": id.0, "reason": "MCP integration test"}),
        ))
        .await;
    assert!(
        resp.error.is_none(),
        "forget_record failed: {:?}",
        resp.error
    );

    let payload = extract_payload(&resp);
    assert_eq!(payload["outcome"], "forgotten");
    assert_eq!(payload["record_id"], id.0);
    assert_eq!(payload["adapter"], "claude-code");
    assert_eq!(payload["native_id"], "doomed-r73");
    assert_eq!(payload["raw_hash"], "h-doomed-r73");
    assert_eq!(payload["reason"], "MCP integration test");
    assert!(payload["forgotten_at"].is_i64());

    // get_record on the same id must now return null (the live row is gone).
    let get_resp = bundle
        .server
        .handle(tool_call("get_record", json!({"id": id.0})))
        .await;
    let get_payload = extract_payload(&get_resp);
    assert!(
        get_payload["record"].is_null(),
        "live record must be gone after forget; got {get_payload}"
    );

    // Audit log carries an entry-point-tagged forget entry.
    let audit = read_audit_lines(&bundle.audit_dir);
    let forget_entries: Vec<_> = audit.iter().filter(|e| e["action"] == "forget").collect();
    assert_eq!(forget_entries.len(), 1, "expected one forget audit entry");
    assert_eq!(forget_entries[0]["detail"]["via"], "mcp");
    assert_eq!(forget_entries[0]["detail"]["outcome"], "forgotten");
}

#[tokio::test]
async fn forget_record_second_call_returns_already_forgotten() {
    let (bundle, id) = build_bundle(true);
    let _first = bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": id.0, "reason": "first"}),
        ))
        .await;
    let second = bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": id.0, "reason": "second"}),
        ))
        .await;
    assert!(
        second.error.is_none(),
        "second forget failed: {:?}",
        second.error
    );
    let payload = extract_payload(&second);
    assert_eq!(payload["outcome"], "already-forgotten");
    assert_eq!(
        payload["reason"], "first",
        "original tombstone reason must be preserved across re-forget"
    );
}

#[tokio::test]
async fn forget_record_unknown_id_is_tool_error() {
    let (bundle, _id) = build_bundle(true);
    let resp = bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": "no-such-id-anywhere"}),
        ))
        .await;
    assert!(
        resp.error.is_some(),
        "forget_record on unknown id must error, not silently succeed"
    );
    let msg = resp.error.unwrap().message.to_lowercase();
    assert!(
        msg.contains("nothing to forget") || msg.contains("no record"),
        "error must explain why; got {msg}"
    );
}

// ─── Round-74 PR-74: list_forgotten ─────────────────────────────────

#[tokio::test]
async fn list_forgotten_is_listed_as_admin_tool() {
    assert!(
        ADMIN_TOOLS.contains(&"list_forgotten"),
        "list_forgotten must be admin-gated"
    );
}

#[tokio::test]
async fn list_forgotten_hidden_from_tools_list_without_admin() {
    let (bundle, _id) = build_bundle(false);
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
        !names.contains(&"list_forgotten"),
        "list_forgotten must NOT appear in default tools/list; got {names:?}"
    );
}

#[tokio::test]
async fn list_forgotten_rejected_when_admin_disabled() {
    let (bundle, _id) = build_bundle(false);
    let resp = bundle
        .server
        .handle(tool_call("list_forgotten", json!({})))
        .await;
    assert!(
        resp.error.is_some(),
        "list_forgotten must error without admin gate"
    );
}

/// Default payload must redact sensitive fields. We seed a tombstone
/// with distinctive markers in `reason` and `native_path`, list, and
/// then grep the serialised payload to be sure neither string leaked.
#[tokio::test]
async fn list_forgotten_redacts_sensitive_fields_by_default() {
    let (bundle, id) = build_bundle(true);
    // Forget it first with a known marker reason.
    let _ = bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": id.0, "reason": "secretReasonMarkerR74"}),
        ))
        .await;
    let body = bundle
        .server
        .handle(tool_call("list_forgotten", json!({})))
        .await
        .result
        .unwrap();
    let payload = body["structuredContent"].clone();
    assert_eq!(payload["count"], 1);
    assert_eq!(payload["sensitive_included"], false);
    let row = &payload["rows"][0];
    assert_eq!(row["record_id"], id.0);
    assert_eq!(row["adapter"], "claude-code");
    assert_eq!(row["native_id"], "doomed-r73");
    assert_eq!(row["has_reason"], true);
    assert_eq!(row["has_native_path"], true);
    // Sensitive fields must be absent (not just null).
    assert!(row.get("reason").is_none(), "reason must be absent: {row}");
    assert!(row.get("native_path").is_none());
    assert!(row.get("raw_hash").is_none());
    let serialised = serde_json::to_string(&payload).unwrap();
    assert!(
        !serialised.contains("secretReasonMarkerR74"),
        "reason marker must not appear in redacted payload"
    );
    assert!(
        !serialised.contains("/tmp/doomed.md"),
        "native_path must not appear in redacted payload"
    );
    assert!(
        !serialised.contains("h-doomed-r73"),
        "raw_hash must not appear in redacted payload"
    );
}

#[tokio::test]
async fn list_forgotten_reveals_sensitive_fields_when_opted_in() {
    let (bundle, id) = build_bundle(true);
    let _ = bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": id.0, "reason": "secretReasonMarkerR74"}),
        ))
        .await;
    let body = bundle
        .server
        .handle(tool_call(
            "list_forgotten",
            json!({"include_sensitive": true}),
        ))
        .await
        .result
        .unwrap();
    let payload = body["structuredContent"].clone();
    assert_eq!(payload["sensitive_included"], true);
    let row = &payload["rows"][0];
    assert_eq!(row["reason"], "secretReasonMarkerR74");
    assert_eq!(row["native_path"], "/tmp/doomed.md");
    assert_eq!(row["raw_hash"], "h-doomed-r73");
}

#[tokio::test]
async fn list_forgotten_filters_by_source() {
    let (bundle, id) = build_bundle(true);
    let _ = bundle
        .server
        .handle(tool_call("forget_record", json!({"record_id": id.0})))
        .await;
    // Wrong source → empty.
    let body = bundle
        .server
        .handle(tool_call("list_forgotten", json!({"source": "mem0"})))
        .await
        .result
        .unwrap();
    assert_eq!(body["structuredContent"]["count"], 0);
    // Right source → 1.
    let body = bundle
        .server
        .handle(tool_call(
            "list_forgotten",
            json!({"source": "claude-code"}),
        ))
        .await
        .result
        .unwrap();
    assert_eq!(body["structuredContent"]["count"], 1);
}

// ─── Round-75 PR-75: unforget_record ────────────────────────────────

#[tokio::test]
async fn unforget_record_is_listed_as_admin_tool() {
    assert!(
        ADMIN_TOOLS.contains(&"unforget_record"),
        "unforget_record must be admin-gated"
    );
}

#[tokio::test]
async fn unforget_record_hidden_from_tools_list_without_admin() {
    let (bundle, _id) = build_bundle(false);
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
        !names.contains(&"unforget_record"),
        "unforget_record must NOT appear in default tools/list; got {names:?}"
    );
}

#[tokio::test]
async fn unforget_record_rejected_when_admin_disabled() {
    let (bundle, id) = build_bundle(false);
    let resp = bundle
        .server
        .handle(tool_call("unforget_record", json!({"record_id": id.0})))
        .await;
    assert!(
        resp.error.is_some(),
        "unforget_record must error without admin gate"
    );
}

/// The full forget → unforget cycle through MCP. After unforget,
/// `list_forgotten` must drop to 0 and `get_record` must STILL
/// return null (unforget doesn't resurrect the live row).
#[tokio::test]
async fn unforget_record_removes_tombstone_but_keeps_record_absent() {
    let (bundle, id) = build_bundle(true);
    // Forget first so there's a tombstone to remove.
    let _ = bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": id.0, "reason": "test"}),
        ))
        .await;

    let unforget_resp = bundle
        .server
        .handle(tool_call("unforget_record", json!({"record_id": id.0})))
        .await;
    assert!(
        unforget_resp.error.is_none(),
        "unforget_record failed: {:?}",
        unforget_resp.error
    );
    let payload = extract_payload(&unforget_resp);
    assert_eq!(payload["outcome"], "unforgotten");
    assert_eq!(payload["record_id"], id.0);
    assert_eq!(payload["record_resurrected"], false);
    assert_eq!(payload["requires_reimport"], true);

    // list_forgotten count drops to 0.
    let list_payload = extract_payload(
        &bundle
            .server
            .handle(tool_call("list_forgotten", json!({})))
            .await,
    );
    assert_eq!(list_payload["count"], 0, "tombstone must be gone");

    // get_record stays null — unforget alone doesn't resurrect.
    let get_payload = extract_payload(
        &bundle
            .server
            .handle(tool_call("get_record", json!({"id": id.0})))
            .await,
    );
    assert!(
        get_payload["record"].is_null(),
        "unforget must NOT resurrect the live row: {get_payload}"
    );

    // Audit log carries an `unforget` entry with `via: "mcp"`.
    let audit = read_audit_lines(&bundle.audit_dir);
    let entries: Vec<_> = audit.iter().filter(|e| e["action"] == "unforget").collect();
    assert_eq!(entries.len(), 1, "expected one unforget audit entry");
    assert_eq!(entries[0]["detail"]["via"], "mcp");
    assert_eq!(entries[0]["detail"]["outcome"], "unforgotten");
}

#[tokio::test]
async fn unforget_record_unknown_id_is_tool_error() {
    let (bundle, _id) = build_bundle(true);
    let resp = bundle
        .server
        .handle(tool_call(
            "unforget_record",
            json!({"record_id": "no-tombstone-here"}),
        ))
        .await;
    assert!(
        resp.error.is_some(),
        "unforget on unknown id must error, not silently succeed"
    );
    let msg = resp.error.unwrap().message.to_lowercase();
    assert!(
        msg.contains("no tombstone") || msg.contains("nothing to unforget"),
        "error must explain why; got {msg}"
    );
}

// ─── Round-83 PR-78e: forget_record dry_run ──────────────────────────

/// `dry_run: true` returns the cascade preview, marks
/// `dry_run: true` in the payload, and does NOT mutate the
/// store (record still gettable) and does NOT append an audit
/// entry.
#[tokio::test]
async fn forget_record_dry_run_previews_without_mutating_or_auditing() {
    let (bundle, id) = build_bundle(true);

    // Capture audit baseline (file may not exist yet).
    let audit_path = bundle.audit_dir.join("audit.log");
    let audit_before = std::fs::metadata(&audit_path).ok().map(|m| m.len());

    let resp = bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": id.0, "reason": "preview", "dry_run": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["dry_run"], true);
    assert_eq!(payload["status"], "would-forget");
    assert_eq!(payload["record_id"], id.0);
    assert_eq!(payload["reason"], "preview");
    assert_eq!(payload["would_delete"]["records"], 1);
    assert!(payload["would_delete"]["record_chunks"].as_u64().unwrap() >= 1);
    assert_eq!(payload["would_insert"]["record_tombstones"], 1);
    assert_eq!(payload["would_insert"]["audit_log_entries"], 1);

    // Mutation guard: audit file size is unchanged after the
    // dry-run call returned. Capture this BEFORE running
    // anything else so subsequent calls (like get_record) don't
    // confound the size delta.
    assert_eq!(
        std::fs::metadata(&audit_path).ok().map(|m| m.len()),
        audit_before,
        "dry-run must not append to audit.log"
    );

    // The live record still resolves — get_record uses `id`, not
    // `record_id`, matching the existing schema.
    let still_there = bundle
        .server
        .handle(tool_call("get_record", json!({"id": id.0})))
        .await;
    assert!(still_there.error.is_none());
    let still_there_payload = extract_payload(&still_there);
    assert!(
        !still_there_payload.is_null(),
        "live record must still resolve after dry-run; got null"
    );
}

/// Admin gate is still enforced for dry-run — the preview
/// reveals raw_hash/native_path and destructive intent, so it
/// must require the same ACL as the real forget.
#[tokio::test]
async fn forget_record_dry_run_is_still_admin_gated() {
    let (bundle, id) = build_bundle(false);
    let resp = bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": id.0, "dry_run": true}),
        ))
        .await;
    assert!(resp.error.is_some(), "dry-run must not bypass admin gate");
}

/// `dry_run: true` on an already-forgotten record returns
/// `already-forgotten` with the existing tombstone and
/// `dry_run: true` flag. No second audit entry.
#[tokio::test]
async fn forget_record_dry_run_already_forgotten_echoes_tombstone() {
    let (bundle, id) = build_bundle(true);
    // Real forget first.
    bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": id.0, "reason": "init"}),
        ))
        .await;
    let audit_path = bundle.audit_dir.join("audit.log");
    let audit_size_after_real = std::fs::metadata(&audit_path).ok().map(|m| m.len());

    // Now dry-run.
    let resp = bundle
        .server
        .handle(tool_call(
            "forget_record",
            json!({"record_id": id.0, "dry_run": true}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["dry_run"], true);
    assert_eq!(payload["outcome"], "already-forgotten");
    assert_eq!(payload["reason"], "init");

    // No new audit entry appended.
    assert_eq!(
        std::fs::metadata(&audit_path).ok().map(|m| m.len()),
        audit_size_after_real,
        "dry-run on already-forgotten must not append to audit.log"
    );
}

/// tools/list schema advertises the `dry_run` arg so MCP
/// clients can introspect.
#[tokio::test]
async fn forget_record_tools_list_schema_advertises_dry_run() {
    let (bundle, _id) = build_bundle(true);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = bundle.server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let forget = tools
        .iter()
        .find(|t| t["name"] == "forget_record")
        .expect("forget_record must be in admin tools/list");
    let props = &forget["inputSchema"]["properties"];
    assert_eq!(
        props["dry_run"]["type"], "boolean",
        "forget_record must advertise dry_run arg: {forget}"
    );
}
