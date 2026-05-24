//! HTTP transport integration test (Phase 3 `sse` feature).
//!
//! Spawns the axum server on an ephemeral port, sends real JSON-RPC
//! frames via reqwest, and asserts bearer-token enforcement.

#![cfg(feature = "sse")]

use anamnesis_mcp_server::{sse, AnamnesisServer};
use anamnesis_store::Store;

fn build_server() -> AnamnesisServer {
    let data = tempfile::tempdir().expect("tempdir");
    let store = Store::open(data.path().join("anamnesis.sqlite")).unwrap();
    store
        .register_source("claude-code", None, Some("/tmp/x"), None)
        .unwrap();
    // Keep the tempdir alive for the duration of the test by leaking it
    // (acceptable in a single-test process — the OS cleans up at exit).
    Box::leak(Box::new(data));
    AnamnesisServer::new(store, None, std::path::PathBuf::from("/tmp"))
}

#[tokio::test]
async fn http_endpoint_round_trips_initialize() {
    let server = build_server();
    let (listener, addr, app, token) = sse::bind(server, Some("test-token-abc".into()))
        .await
        .unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let url = format!("http://{addr}/mcp");

    // 1. Without bearer → 401.
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);

    // 2. Wrong bearer → 401.
    let resp = client
        .post(&url)
        .bearer_auth("wrong-token")
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);

    // 3. Correct bearer + initialize → 200 with serverInfo.
    let resp = client
        .post(&url)
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "initialize",
            "params": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["id"], 3);
    assert_eq!(body["result"]["serverInfo"]["name"], "anamnesis");

    // 4. tools/list works too.
    let resp = client
        .post(&url)
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/list",
            "params": {}
        }))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let tools = body["result"]["tools"].as_array().unwrap();
    // PR-A: admin tools (`import_source`) are hidden by default. The HTTP
    // test server is built without `with_admin_tools(true)`, so only the
    // read-only tools should show up — search_memories, get_record,
    // list_sources, trace_provenance, doctor (5 since round-54) +
    // dedupe (Round 77) + list_conflicts (Round 135) +
    // discover_adapters (Round 137) = 8.
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert_eq!(
        tools.len(),
        8,
        "expect 8 non-admin tools by default; got {names:?}"
    );
    assert!(!names.contains(&"import_source"));
    for expected in [
        "search_memories",
        "get_record",
        "list_sources",
        "trace_provenance",
        "doctor",
        "dedupe",
    ] {
        assert!(names.contains(&expected), "missing tool {expected}");
    }

    // 5. notification (no id) → 204 No Content.
    let resp = client
        .post(&url)
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NO_CONTENT);

    handle.abort();
}

#[tokio::test]
async fn healthz_endpoint_skips_auth() {
    let server = build_server();
    let (listener, addr, app, _token) = sse::bind(server, Some("nope".into())).await.unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let resp = reqwest::get(format!("http://{addr}/healthz"))
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(resp.text().await.unwrap(), "ok");

    handle.abort();
}
