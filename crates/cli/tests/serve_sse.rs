//! End-to-end test for `anamnesis serve --sse <port>`.
//!
//! Spawns the real CLI binary as a subprocess against a temp data dir,
//! parses the bound port from stderr (since the test uses `--sse 0`
//! for an ephemeral port), and verifies the HTTP transport is reachable
//! with the expected auth contract:
//!
//!   GET /healthz                      → 200 ok
//!   POST /mcp  (no token)             → 401
//!   POST /mcp  (Bearer test-token)    → 200 with initialize result
//!
//! This is the round-14 acceptance test that unwires the
//! "use the dedicated anamnesis-mcp --sse binary instead" foot-gun.

#![cfg(feature = "sse")]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Path to the cargo-built `anamnesis` binary.
fn anamnesis_bin() -> std::path::PathBuf {
    assert_cmd::cargo::cargo_bin("anamnesis")
}

/// Spawn `anamnesis serve --sse 0 --token <token>` and block until
/// either the "listening on http://127.0.0.1:<port>" line shows up on
/// stderr (in which case we return the parsed `SocketAddr`) or the
/// child exits / we time out.
fn spawn_and_wait_for_port(
    data_dir: &std::path::Path,
    token: &str,
) -> (Child, std::net::SocketAddr) {
    let mut child = Command::new(anamnesis_bin())
        .env("ANAMNESIS_DATA_DIR", data_dir)
        // Keep noise low — only our own eprintlns + the bound-addr line.
        .env("RUST_LOG", "error")
        .args(["serve", "--sse", "0", "--token", token])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn anamnesis serve --sse 0");

    let stderr = child.stderr.take().expect("stderr pipe");
    let (tx, rx) = mpsc::channel::<String>();

    // Drain stderr on a background thread so the child doesn't block
    // on a full pipe — and forward every line back so we can parse it.
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            eprintln!("[anamnesis stderr] {line}");
            // Best effort — receiver may be gone if we already parsed.
            let _ = tx.send(line);
        }
    });

    let deadline = Instant::now() + Duration::from_secs(15);
    let prefix = "anamnesis-mcp HTTP — listening on http://";
    loop {
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("timed out waiting for bound-address line");
        }
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(line) => {
                if let Some(rest) = line.strip_prefix(prefix) {
                    let addr: std::net::SocketAddr = rest
                        .trim()
                        .parse()
                        .unwrap_or_else(|e| panic!("parse bound addr from {rest:?}: {e}"));
                    return (child, addr);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Ok(Some(status)) = child.try_wait() {
                    panic!("child exited early with status {status:?}");
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = child.kill();
                panic!("stderr drain thread closed before bound-address line");
            }
        }
    }
}

/// Best-effort cleanup so a leftover serve subprocess doesn't outlive
/// the test runner.
struct Guard(Child);
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test(flavor = "current_thread")]
async fn serve_sse_http_transport_round_trips_initialize() {
    let dir = tempfile::tempdir().expect("tempdir");
    let token = "round-14-test-token-9c1f";

    let (child, addr) = spawn_and_wait_for_port(dir.path(), token);
    let _guard = Guard(child);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();

    // 1. /healthz is open (no auth) and returns 200 "ok".
    let url = format!("http://{addr}");
    let resp = client
        .get(format!("{url}/healthz"))
        .send()
        .await
        .expect("GET /healthz");
    assert_eq!(resp.status(), 200, "/healthz should be 200");
    assert_eq!(resp.text().await.unwrap().trim(), "ok");

    // 2. /mcp without bearer → 401.
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "serve-sse-test", "version": "0" }
        }
    });
    let resp = client
        .post(format!("{url}/mcp"))
        .json(&body)
        .send()
        .await
        .expect("POST /mcp no-bearer");
    assert_eq!(
        resp.status(),
        401,
        "POST /mcp without bearer must be 401, was {}",
        resp.status()
    );

    // 3. /mcp with bearer + initialize → 200 + serverInfo.name == "anamnesis".
    let resp = client
        .post(format!("{url}/mcp"))
        .bearer_auth(token)
        .json(&body)
        .send()
        .await
        .expect("POST /mcp bearer");
    assert_eq!(
        resp.status(),
        200,
        "POST /mcp with bearer must be 200, was {}",
        resp.status()
    );
    let v: serde_json::Value = resp.json().await.expect("json initialize response");
    let server_name = v
        .get("result")
        .and_then(|r| r.get("serverInfo"))
        .and_then(|s| s.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("");
    assert_eq!(
        server_name, "anamnesis",
        "initialize must return serverInfo.name = anamnesis, got payload: {v}"
    );
}
