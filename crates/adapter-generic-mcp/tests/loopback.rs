//! Loopback test: spin up anamnesis-mcp HTTP, point generic-mcp adapter
//! at it, verify roundtrip.
//!
//! This is the "Anamnesis as memory provider to other agents" loop
//! (BLUEPRINT §11 Phase 4) — proves that a second Anamnesis instance
//! can consume from the first via the standard MCP HTTP surface.

use anamnesis_adapter_generic_mcp::{generic_mcp_adapter, GenericMcpAdapter, GenericMcpConfig};
use anamnesis_core::adapter::{MemoryAdapter, ScanOpts};
use anamnesis_core::chunker::Chunker;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_mcp_server::{sse, AnamnesisServer};
use anamnesis_store::Store;
use chrono::Utc;
use futures::StreamExt;

/// Build an upstream server with optional seed records. Each `(id, content)`
/// pair is upserted under the `claude-code` adapter so `resources/list`
/// has something concrete to enumerate.
fn build_upstream_with_seeds(seeds: &[(&str, &str)]) -> AnamnesisServer {
    let data = tempfile::tempdir().expect("tempdir");
    let store = Store::open(data.path().join("anamnesis.sqlite")).unwrap();
    store
        .register_source("claude-code", None, Some("/tmp/x"), None)
        .unwrap();
    for (native_id, content) in seeds {
        let r = AnamnesisRecord {
            id: RecordId::from_parts("claude-code", None, native_id),
            source: SourceDescriptor {
                adapter: "claude-code".into(),
                instance: None,
                version: "0.0.1".into(),
            },
            content: (*content).to_string(),
            embedding: None,
            scope: Scope::User,
            kind: Kind::Fact,
            created_at: Utc::now(),
            updated_at: None,
            tags: vec![],
            metadata: Default::default(),
            provenance: Provenance {
                native_id: (*native_id).into(),
                native_path: Some(format!("/p/{native_id}")),
                captured_at: Utc::now(),
                raw_hash: format!("h-{native_id}"),
            },
            schema_version: SCHEMA_VERSION,
        };
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
    }
    Box::leak(Box::new(data));
    AnamnesisServer::new(store, None, std::path::PathBuf::from("/tmp"))
}

fn build_upstream() -> AnamnesisServer {
    build_upstream_with_seeds(&[])
}

#[tokio::test]
async fn generic_mcp_lists_then_reads_upstream_resources() {
    // Round-13: seed a concrete record so resources/list emits a
    // resolvable URI (not just `{id}` templates that the adapter would
    // skip). Asserts the full loopback: upstream record → MCP →
    // generic-mcp adapter scan → RawRecord with the seed text.
    let server = build_upstream_with_seeds(&[(
        "seed-1",
        "loopback round-13 sentinel content uniqueLoopbackToken",
    )]);
    let (listener, addr, app, token) = sse::bind(server, Some("loopback-token".into()))
        .await
        .unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let adapter: GenericMcpAdapter =
        generic_mcp_adapter(format!("http://{addr}"), Some(&token), Some("loopback"));

    let h = adapter.health().await;
    assert!(h.ok, "upstream should be healthy: {}", h.detail);

    let raws: Vec<_> = adapter
        .scan(ScanOpts::default())
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    // The whole point of round-13: NO MORE permissive empty assertion.
    // The upstream has a real record, so the loopback MUST surface at
    // least one raw with the seeded content.
    assert!(
        !raws.is_empty(),
        "loopback must return at least 1 raw record now that the upstream has data \
         (previous empty-result was the round-13 bug we just fixed)"
    );

    let any_with_seed_content = raws.iter().any(|r| {
        let payload_text = r.payload.to_string();
        payload_text.contains("uniqueLoopbackToken")
    });
    assert!(
        any_with_seed_content,
        "at least one raw must carry the seeded content. \
         Got payloads: {:?}",
        raws.iter()
            .map(|r| r.payload.to_string())
            .collect::<Vec<_>>()
    );

    // native_path follows the generic-mcp convention: upstream URI.
    let any_with_record_uri = raws.iter().any(|r| {
        r.native_path
            .as_deref()
            .unwrap_or("")
            .starts_with("anamnesis://record/")
    });
    assert!(
        any_with_record_uri,
        "at least one raw must carry an anamnesis://record/<id> native_path"
    );

    handle.abort();
}

#[tokio::test]
async fn descriptor_and_id_are_stable() {
    let adapter: GenericMcpAdapter =
        generic_mcp_adapter("http://127.0.0.1:1", Some("token"), Some("upstream"));
    let d1 = adapter.descriptor();
    let d2 = adapter.descriptor();
    assert_eq!(d1, d2);
    assert_eq!(d1.adapter, "generic-mcp");
    assert_eq!(d1.instance.as_deref(), Some("upstream"));
}

#[tokio::test]
async fn health_returns_false_when_unreachable() {
    let adapter: GenericMcpAdapter = generic_mcp_adapter(
        "http://127.0.0.1:1", // nothing listening
        None,
        None,
    );
    let h = adapter.health().await;
    assert!(!h.ok);
}

#[tokio::test]
async fn detector_reports_healthy_upstream() {
    use anamnesis_adapter_generic_mcp::GenericMcpDetector;
    use anamnesis_core::discovery::{DetectOpts, SourceDetector};

    let server = build_upstream();
    let (listener, addr, app, token) = sse::bind(server, Some("token".into())).await.unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let detector = GenericMcpDetector::new(format!("http://{addr}"), Some(token));
    let found = detector.detect(&DetectOpts::default()).await.unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].adapter, "generic-mcp");

    handle.abort();
}

/// Round-21 (§-1.5 PR-2): a 250-record upstream catalogue must be
/// migrated **completely** by the generic-mcp adapter via the new
/// `cursor` / `nextCursor` protocol. Previously the adapter saw only
/// the first page (capped at `RESOURCES_LIST_PAGE = 100`) and silently
/// dropped 150 records.
#[tokio::test]
async fn loopback_paginates_250_records_without_loss() {
    const N: usize = 250;

    // Build a seed pool of (native_id, content) so we can build the
    // upstream and then ask "did everything survive the migration?".
    let seeds_owned: Vec<(String, String)> = (0..N)
        .map(|i| {
            (
                format!("pr2-seed-{i:04}"),
                format!("pr2 pagination sentinel {i}"),
            )
        })
        .collect();
    let seeds: Vec<(&str, &str)> = seeds_owned
        .iter()
        .map(|(id, c)| (id.as_str(), c.as_str()))
        .collect();

    let server = build_upstream_with_seeds(&seeds);
    let (listener, addr, app, token) = sse::bind(server, Some("pr2-token".into())).await.unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let adapter = generic_mcp_adapter(format!("http://{addr}"), Some(&token), Some("pr2"));
    let raws: Vec<_> = adapter
        .scan(ScanOpts::default())
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    // Every seeded record must come back. This is the §-1.1.2 contract:
    // complete migration without dropping records.
    assert_eq!(
        raws.len(),
        N,
        "expected all {N} upstream records, got {}",
        raws.len()
    );

    // And every distinct native_path must appear. (Set-equality across
    // the full corpus; a bug that dropped some records but added
    // duplicates would still fail this check.)
    let native_paths: std::collections::HashSet<String> =
        raws.iter().filter_map(|r| r.native_path.clone()).collect();
    assert_eq!(
        native_paths.len(),
        N,
        "all {N} native_paths must be distinct; some pages overlapped or duplicated"
    );

    handle.abort();
}

// Reference to suppress unused-import linting in tests.
#[allow(dead_code)]
fn _unused(_: GenericMcpConfig) {}
