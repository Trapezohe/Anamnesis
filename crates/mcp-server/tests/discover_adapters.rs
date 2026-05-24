//! Round-137 PR-78bf MCP `discover_adapters` end-to-end.
//!
//! Acceptance:
//!   1. Tool is non-admin (appears in default `tools/list`,
//!      callable without admin gate).
//!   2. Capability roster is always returned, even when no
//!      sources are detected on this machine.
//!   3. `home_override` (via `AnamnesisServer::with_home`) is
//!      threaded into the detection pass, so a planted fixture
//!      under a tempdir gets picked up.
//!   4. `tools/list` advertises the empty-input wire shape.

use anamnesis_mcp_server::AnamnesisServer;
use anamnesis_store::Store;
use serde_json::{json, Value};

fn build_server(home: Option<std::path::PathBuf>) -> (AnamnesisServer, tempfile::TempDir) {
    let data_dir = tempfile::tempdir().expect("data tempdir");
    let store = Store::open(data_dir.path().join("anamnesis.sqlite")).expect("open store");
    let mut s =
        AnamnesisServer::new(store, None, data_dir.path().to_path_buf()).with_admin_tools(false);
    if let Some(h) = home {
        s = s.with_home(h);
    }
    (s, data_dir)
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
async fn discover_adapters_is_visible_to_non_admin_clients() {
    let (server, _data) = build_server(None);
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
        names.contains(&"discover_adapters"),
        "discover_adapters must appear in non-admin tools/list; got {names:?}"
    );
}

#[tokio::test]
async fn discover_adapters_returns_capability_roster_even_when_nothing_detected() {
    // Fresh tempdir as $HOME → detectors find nothing, but the
    // capability roster must still be there.
    let tempdir = tempfile::tempdir().expect("home tempdir");
    let (server, _data) = build_server(Some(tempdir.path().to_path_buf()));
    let resp = server
        .handle(tool_call("discover_adapters", json!({})))
        .await;
    assert!(
        resp.error.is_none(),
        "discover_adapters must succeed: {:?}",
        resp.error
    );
    let payload = extract_payload(&resp);

    let stats = &payload["stats"];
    assert_eq!(stats["adapter_count"], 13);
    assert_eq!(stats["detector_count"], 12);
    assert_eq!(stats["detected_count"], 0);

    let adapters = payload["adapters"].as_array().unwrap();
    assert_eq!(adapters.len(), 13);
    // generic-mcp is the only non-detectable adapter; pinning the
    // invariant so a future bump can't silently flip it.
    let non_detectable: Vec<&str> = adapters
        .iter()
        .filter(|a| a["detectable"] == false)
        .map(|a| a["adapter"].as_str().unwrap())
        .collect();
    assert_eq!(non_detectable, vec!["generic-mcp"]);

    assert!(payload["detected"].as_array().unwrap().is_empty());
    let summary = payload["summary"].as_str().unwrap();
    assert!(summary.contains("13 adapters"));
    assert!(summary.contains("12 auto-detectable"));
}

#[tokio::test]
async fn discover_adapters_uses_server_home_override_for_detection() {
    // Seed a `~/.letta/letta.db` shape under a tempdir, point the
    // server's `home_override` there, and confirm the Letta
    // detector fires through the MCP surface.
    let tempdir = tempfile::tempdir().expect("home tempdir");
    let letta_dir = tempdir.path().join(".letta");
    std::fs::create_dir_all(&letta_dir).unwrap();
    std::fs::write(letta_dir.join("letta.db"), b"").unwrap();

    let (server, _data) = build_server(Some(tempdir.path().to_path_buf()));
    let resp = server
        .handle(tool_call("discover_adapters", json!({})))
        .await;
    let payload = extract_payload(&resp);
    let detected = payload["detected"].as_array().unwrap();
    let hits: Vec<&str> = detected
        .iter()
        .map(|d| d["adapter"].as_str().unwrap())
        .collect();
    assert!(
        hits.contains(&"letta"),
        "letta detector must pick up the planted fixture: detected={detected:?}"
    );
    // The detected entry must carry confidence + location and
    // never expose memory content (the detector contract).
    let letta_row = detected
        .iter()
        .find(|d| d["adapter"] == "letta")
        .expect("letta row");
    assert!(letta_row["confidence"].is_string());
    assert!(letta_row["location"].is_string());
    // `note` is optional; when present it must be a string.
    if !letta_row["note"].is_null() {
        assert!(letta_row["note"].is_string());
    }
}

#[tokio::test]
async fn discover_adapters_tools_list_schema_advertises_empty_input() {
    let (server, _data) = build_server(None);
    let req = anamnesis_mcp_server::protocol::JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "tools/list".into(),
        params: Value::Null,
    };
    let resp = server.handle(req).await;
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let entry = tools
        .iter()
        .find(|t| t["name"] == "discover_adapters")
        .expect("discover_adapters in tools/list");
    let schema = &entry["inputSchema"];
    assert_eq!(schema["type"], "object");
    // No required input — agents call this tool with `{}`.
    let props = schema["properties"].as_object().unwrap();
    assert!(
        props.is_empty(),
        "discover_adapters takes no args: {props:?}"
    );
}
