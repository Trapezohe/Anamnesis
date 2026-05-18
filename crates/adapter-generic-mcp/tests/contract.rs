//! `generic-mcp` adapter satisfies the shared `anamnesis_core::contract`
//! invariants. Unlike the filesystem-/sqlite-backed adapters, generic-mcp
//! speaks HTTP to a live MCP upstream, so the fixture spins up a real
//! `anamnesis-mcp-server` on a loopback port and points the adapter at it.

use anamnesis_adapter_generic_mcp::{generic_mcp_adapter, GenericMcpAdapter};
use anamnesis_core::chunker::Chunker;
use anamnesis_core::contract::AdapterContract;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_mcp_server::{sse, AnamnesisServer};
use anamnesis_store::Store;
use chrono::Utc;

/// Seed a fresh upstream with one record so `resources/list` has
/// something concrete to enumerate.
fn build_upstream() -> AnamnesisServer {
    let data = tempfile::tempdir().expect("tempdir");
    let store = Store::open(data.path().join("anamnesis.sqlite")).unwrap();
    store
        .register_source("claude-code", None, Some("/tmp/x"), None)
        .unwrap();
    let native_id = "contract-seed-1";
    let r = AnamnesisRecord {
        id: RecordId::from_parts("claude-code", None, native_id),
        source: SourceDescriptor {
            adapter: "claude-code".into(),
            instance: None,
            version: "0.0.1".into(),
        },
        content: "contract-test sentinel content".to_string(),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: native_id.into(),
            native_path: Some(format!("/p/{native_id}")),
            captured_at: Utc::now(),
            raw_hash: format!("h-{native_id}"),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    };
    let chunks = Chunker::default().chunk(&r.id, &r.content);
    store.upsert_record(&r, &chunks, None).unwrap();
    // Tempdir lives for the test process — leak it so the SQLite file
    // stays around for the duration of the upstream server.
    Box::leak(Box::new(data));
    AnamnesisServer::new(store, None, std::path::PathBuf::from("/tmp"))
}

#[tokio::test]
async fn generic_mcp_satisfies_adapter_contract() {
    let server = build_upstream();
    let (listener, addr, app, token) = sse::bind(server, Some("contract-token".into()))
        .await
        .unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let url = format!("http://{addr}");
    let contract = AdapterContract::new(move || -> GenericMcpAdapter {
        generic_mcp_adapter(url.clone(), Some(&token), Some("upstream"))
    });
    contract.run_all().await;

    handle.abort();
}

#[tokio::test]
async fn generic_mcp_no_instance_satisfies_contract() {
    let server = build_upstream();
    let (listener, addr, app, token) = sse::bind(server, Some("contract-token".into()))
        .await
        .unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let url = format!("http://{addr}");
    let contract = AdapterContract::new(move || -> GenericMcpAdapter {
        generic_mcp_adapter(url.clone(), Some(&token), None)
    });
    contract.run_all().await;

    handle.abort();
}
