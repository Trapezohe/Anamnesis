//! End-to-end test of the stdio MCP binary.
//!
//! Spawns `anamnesis-mcp`, sends JSON-RPC requests over stdin one line at
//! a time, and parses responses from stdout. Verifies initialize +
//! tools/list works against a real subprocess.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn binary() -> PathBuf {
    let exe = if cfg!(windows) {
        "anamnesis-mcp.exe"
    } else {
        "anamnesis-mcp"
    };
    PathBuf::from(env!("CARGO_BIN_EXE_anamnesis-mcp")).with_file_name(exe)
}

fn seed_store(data_dir: &std::path::Path) {
    use anamnesis_store::Store;
    let store = Store::open(data_dir.join("anamnesis.sqlite")).unwrap();
    store
        .register_source("claude-code", None, Some("/tmp/x"), None)
        .unwrap();
    // No active model so the binary doesn't try to load fastembed.
}

#[test]
fn binary_responds_to_initialize_and_tools_list() {
    let data = tempfile::tempdir().unwrap();
    seed_store(data.path());

    let mut child = Command::new(binary())
        .env("ANAMNESIS_DATA_DIR", data.path())
        .env("RUST_LOG", "warn")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn anamnesis-mcp");

    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    fn send(stdin: &mut impl Write, payload: serde_json::Value) {
        let mut line = serde_json::to_string(&payload).unwrap();
        line.push('\n');
        stdin.write_all(line.as_bytes()).unwrap();
        stdin.flush().unwrap();
    }

    fn recv(reader: &mut impl BufRead) -> serde_json::Value {
        let mut buf = String::new();
        reader.read_line(&mut buf).expect("read response line");
        assert!(!buf.is_empty(), "got EOF instead of response");
        serde_json::from_str(&buf).expect("response is valid JSON")
    }

    // 1. initialize
    send(
        &mut stdin,
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }),
    );
    let init = recv(&mut reader);
    assert_eq!(init["id"], 1);
    assert_eq!(init["result"]["serverInfo"]["name"], "anamnesis");

    // 2. tools/list — admin tools (`import_source`) are off by default,
    //    so we expect exactly 4 (the read-only catalogue).
    send(
        &mut stdin,
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }),
    );
    let tools = recv(&mut reader);
    let names: Vec<String> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["name"].as_str().map(str::to_owned))
        .collect();
    // Round 77 added `dedupe` to the non-admin catalogue (5 → 6).
    assert_eq!(
        names.len(),
        6,
        "import_source should be hidden by default; got {names:?}"
    );
    assert!(!names.contains(&"import_source".to_string()));
    assert!(names.contains(&"search_memories".to_string()));
    assert!(names.contains(&"list_sources".to_string()));
    assert!(names.contains(&"doctor".to_string()));
    assert!(names.contains(&"dedupe".to_string()));

    // 3. tools/call list_sources — should report the source we seeded.
    send(
        &mut stdin,
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_sources"}
        }),
    );
    let resp = recv(&mut reader);
    let structured = &resp["result"]["structuredContent"];
    assert!(structured["sources"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["adapter"] == "claude-code"));

    // 4. resources/list — should return 3 URI patterns.
    send(
        &mut stdin,
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/list",
            "params": {}
        }),
    );
    let resources = recv(&mut reader);
    let uris: Vec<String> = resources["result"]["resources"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| r["uri"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(uris.len(), 3);

    // Shut down cleanly by closing stdin.
    drop(stdin);
    let status = child.wait().expect("wait child");
    assert!(status.success(), "server exited non-zero: {status}");
}
