//! Round-78 PR-78 MCP `tag_record` end-to-end.
//!
//! Acceptance points:
//!
//!   1. **Write is admin-gated.** `tag_record` appears in
//!      admin-enabled `tools/list` only and is rejected by
//!      `tools/call` without admin.
//!   2. **Reads are not admin-gated.** `search_memories` and
//!      `get_record` return `user_tags` regardless of admin
//!      gating.
//!   3. **Set semantics + normalisation** match the CLI path
//!      and the store unit tests.
//!   4. **Audit parity.** MCP writes an `action: "tag_record"`
//!      entry with `via: "mcp"`, mirroring the CLI.

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
    _data_dir: tempfile::TempDir,
    audit_dir: std::path::PathBuf,
}

fn build_bundle(allow_admin: bool) -> (TestBundle, RecordId) {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let db = data_dir.path().join("anamnesis.sqlite");
    let store = Store::open(&db).expect("open store");

    let id = RecordId::from_parts("claude-code", None, "tag-r78");
    let record = AnamnesisRecord {
        id: id.clone(),
        source: SourceDescriptor {
            adapter: "claude-code".into(),
            instance: None,
            version: "0".into(),
        },
        content: "content to tag".into(),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: "tag-r78".into(),
            native_path: Some("/tmp/tag-r78.md".into()),
            captured_at: Utc::now(),
            raw_hash: "h-tag-r78".into(),
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

fn read_audit(dir: &std::path::Path) -> Vec<Value> {
    let raw = std::fs::read_to_string(dir.join("audit.log")).unwrap_or_default();
    raw.lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .collect()
}

#[tokio::test]
async fn tag_record_is_listed_as_admin_tool() {
    assert!(
        ADMIN_TOOLS.contains(&"tag_record"),
        "tag_record must be admin-gated for write"
    );
}

#[tokio::test]
async fn tag_record_hidden_from_tools_list_without_admin() {
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
        !names.contains(&"tag_record"),
        "tag_record must NOT appear in default tools/list"
    );
}

#[tokio::test]
async fn tag_record_rejected_when_admin_disabled() {
    let (bundle, id) = build_bundle(false);
    let resp = bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({"record_id": id.0, "tags": ["x"]}),
        ))
        .await;
    assert!(resp.error.is_some(), "must error without admin gate");
}

#[tokio::test]
async fn tag_record_adds_and_returns_normalised_state() {
    let (bundle, id) = build_bundle(true);
    let resp = bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({
                "record_id": id.0,
                "tags": ["  TODO  ", "Keep", "todo"],
                "operation": "add",
            }),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["record_id"], id.0);
    assert_eq!(payload["operation"], "add");
    assert_eq!(payload["requested"], json!(["todo", "keep"]));
    assert_eq!(payload["changed"], 2);
    assert_eq!(payload["user_tags"], json!(["keep", "todo"]));

    let audit = read_audit(&bundle.audit_dir);
    let entries: Vec<_> = audit
        .iter()
        .filter(|e| e["action"] == "tag_record")
        .collect();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["detail"]["via"], "mcp");
    assert_eq!(entries[0]["detail"]["changed"], 2);
}

/// Read paths surface `user_tags` regardless of admin gating.
/// Important: a non-admin agent should be able to see the
/// tags it can't write.
#[tokio::test]
async fn get_record_surfaces_user_tags_without_admin() {
    let (admin_bundle, id) = build_bundle(true);
    // Plant a tag via admin.
    let _ = admin_bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({"record_id": id.0, "tags": ["admin-tag"]}),
        ))
        .await;
    // Re-open the same DB through a non-admin server.
    let store2 = Store::open(admin_bundle._data_dir.path().join("anamnesis.sqlite")).unwrap();
    let no_admin = AnamnesisServer::new(store2, None, admin_bundle._data_dir.path().to_path_buf())
        .with_admin_tools(false);
    let resp = no_admin
        .handle(tool_call("get_record", json!({"id": id.0})))
        .await;
    let payload = extract_payload(&resp);
    assert_eq!(payload["user_tags"], json!(["admin-tag"]));
}

#[tokio::test]
async fn tag_record_remove_drops_tag() {
    let (bundle, id) = build_bundle(true);
    let _ = bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({"record_id": id.0, "tags": ["doomed"]}),
        ))
        .await;
    let resp = bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({"record_id": id.0, "tags": ["doomed"], "operation": "remove"}),
        ))
        .await;
    let payload = extract_payload(&resp);
    assert_eq!(payload["operation"], "remove");
    assert_eq!(payload["changed"], 1);
    assert_eq!(payload["user_tags"], json!([]));
}

// ─── Round-79 PR-78b: search_memories(user_tag = ...) ──────────────

#[tokio::test]
async fn search_memories_user_tag_arg_is_advertised_in_tools_list() {
    let (bundle, _id) = build_bundle(false);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = bundle.server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let search = tools
        .iter()
        .find(|t| t["name"] == "search_memories")
        .unwrap()
        .clone();
    let props = search["inputSchema"]["properties"]
        .as_object()
        .expect("inputSchema.properties");
    assert!(
        props.contains_key("user_tag"),
        "search_memories must advertise user_tag arg; got {:?}",
        props.keys().collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn search_memories_user_tag_filters_to_tagged_records_only() {
    let (bundle, id) = build_bundle(true);
    // Seed a second, untagged record with the same FTS marker so
    // "content to tag" matches both pre-filter; only the tagged
    // one should survive `user_tag = "..."`.
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use chrono::Utc;
    let store = Store::open(bundle._data_dir.path().join("anamnesis.sqlite")).unwrap();
    let extra = AnamnesisRecord {
        id: RecordId::from_parts("mem0", None, "untagged"),
        source: SourceDescriptor {
            adapter: "mem0".into(),
            instance: None,
            version: "0".into(),
        },
        content: "content to tag".into(),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: "untagged".into(),
            native_path: Some("/tmp/untagged.md".into()),
            captured_at: Utc::now(),
            raw_hash: "h-untagged".into(),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let chunks = Chunker::default().chunk(&extra.id, &extra.content);
    store.upsert_record(&extra, &chunks, None).unwrap();
    drop(store);

    // Tag the original `tag-r78` record only.
    let _ = bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({"record_id": id.0, "tags": ["keep"]}),
        ))
        .await;

    // user_tag = "keep" must return ONLY the tagged record.
    let resp = bundle
        .server
        .handle(tool_call(
            "search_memories",
            json!({
                "query": "content",
                "mode": "fulltext",
                "limit": 5,
                "user_tag": "Keep",  // case-insensitive
            }),
        ))
        .await;
    let payload = extract_payload(&resp);
    let results = payload["results"].as_array().unwrap();
    assert_eq!(results.len(), 1, "only tagged record must survive");
    assert_eq!(results[0]["record_id"], id.0);
    assert_eq!(results[0]["user_tags"], json!(["keep"]));
}

#[tokio::test]
async fn search_memories_user_tag_unknown_returns_empty() {
    let (bundle, _id) = build_bundle(false);
    let resp = bundle
        .server
        .handle(tool_call(
            "search_memories",
            json!({
                "query": "content",
                "mode": "fulltext",
                "limit": 5,
                "user_tag": "never-applied",
            }),
        ))
        .await;
    let payload = extract_payload(&resp);
    assert_eq!(payload["results"].as_array().unwrap().len(), 0);
}

// ─── Round-81 PR-78c: tag_record operation=replace ─────────────────

/// `operation=replace` installs the input as the full
/// post-call set in one atomic call, returns set delta, and
/// the audit log records `"operation": "replace"` so a
/// post-hoc audit can distinguish replace from add/remove.
#[tokio::test]
async fn tag_record_replace_overwrites_prior_set_with_delta() {
    let (bundle, id) = build_bundle(true);
    // Seed {keep, todo}.
    bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({"record_id": id.0, "tags": ["keep", "todo"], "operation": "add"}),
        ))
        .await;

    // Replace with {keep, final}.
    let resp = bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({
                "record_id": id.0,
                "tags": ["keep", "final"],
                "operation": "replace",
            }),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["operation"], "replace");
    assert_eq!(payload["changed"], 2, "set delta: 1 added + 1 removed");
    assert_eq!(payload["user_tags"], json!(["final", "keep"]));

    let audit = read_audit(&bundle.audit_dir);
    let replace_entries: Vec<_> = audit
        .iter()
        .filter(|e| e["action"] == "tag_record" && e["detail"]["operation"] == "replace")
        .collect();
    assert_eq!(replace_entries.len(), 1);
    assert_eq!(replace_entries[0]["detail"]["via"], "mcp");
    assert_eq!(replace_entries[0]["detail"]["changed"], 2);
}

/// `operation=replace` with empty `tags` is the only path that
/// accepts an empty list — it clears the overlay for this record.
#[tokio::test]
async fn tag_record_replace_with_empty_tags_clears_overlay() {
    let (bundle, id) = build_bundle(true);
    bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({"record_id": id.0, "tags": ["a", "b"], "operation": "add"}),
        ))
        .await;

    let resp = bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({"record_id": id.0, "tags": [], "operation": "replace"}),
        ))
        .await;
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let payload = extract_payload(&resp);
    assert_eq!(payload["operation"], "replace");
    assert_eq!(payload["changed"], 2, "two tags cleared = 2 deletions");
    assert_eq!(payload["requested"], json!([]));
    assert_eq!(payload["user_tags"], json!([]));
}

/// Empty `tags` with `add` or `remove` must still error — only
/// `replace` carries the "explicit clear" intent. Guards
/// against a regression in the handler-side validation.
#[tokio::test]
async fn tag_record_add_with_empty_tags_still_errors() {
    let (bundle, id) = build_bundle(true);
    let resp = bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({"record_id": id.0, "tags": [], "operation": "add"}),
        ))
        .await;
    assert!(resp.error.is_some(), "empty add must error");
}

/// Unknown `operation` value returns a clear error mentioning
/// all three valid choices. Guards against schema drift between
/// the handler and the advertised enum.
#[tokio::test]
async fn tag_record_unknown_operation_errors_with_three_options() {
    let (bundle, id) = build_bundle(true);
    let resp = bundle
        .server
        .handle(tool_call(
            "tag_record",
            json!({"record_id": id.0, "tags": ["x"], "operation": "toggle"}),
        ))
        .await;
    assert!(resp.error.is_some());
    let msg = resp.error.unwrap().message;
    assert!(msg.contains("add"), "error must mention add: {msg}");
    assert!(msg.contains("remove"), "error must mention remove: {msg}");
    assert!(msg.contains("replace"), "error must mention replace: {msg}");
}

/// tools/list schema advertises `replace` in the operation enum
/// so MCP clients can introspect the wire shape.
#[tokio::test]
async fn tag_record_tools_list_schema_advertises_replace() {
    let (bundle, _id) = build_bundle(true);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = bundle.server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let tag = tools
        .iter()
        .find(|t| t["name"] == "tag_record")
        .expect("tag_record must be in admin tools/list");
    let ops = tag["inputSchema"]["properties"]["operation"]["enum"]
        .as_array()
        .expect("operation must carry an enum");
    let labels: Vec<&str> = ops.iter().filter_map(|v| v.as_str()).collect();
    assert!(labels.contains(&"add"));
    assert!(labels.contains(&"remove"));
    assert!(
        labels.contains(&"replace"),
        "operation enum must advertise replace: {labels:?}"
    );
}
