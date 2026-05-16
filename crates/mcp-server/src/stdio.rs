//! Stdio JSON-RPC loop — shared by the `anamnesis-mcp` binary and the
//! `anamnesis serve` CLI subcommand.

use anyhow::Result;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
use crate::server::AnamnesisServer;

/// Drive the server over stdin / stdout until stdin is closed.
///
/// Reads one JSON-RPC line per frame; writes one response line per
/// request (notifications get no reply). Errors during parse are
/// reported as JSON-RPC `-32700` parse errors.
pub async fn run(server: AnamnesisServer) -> Result<()> {
    let mut stdout = tokio::io::stdout();
    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(req) => {
                let is_note = req.is_notification();
                let resp = server.handle(req).await;
                if is_note {
                    continue;
                }
                resp
            }
            Err(e) => {
                JsonRpcResponse::err(serde_json::Value::Null, -32700, format!("parse error: {e}"))
            }
        };
        let line_out = serde_json::to_string(&response)? + "\n";
        stdout.write_all(line_out.as_bytes()).await?;
        stdout.flush().await?;
    }
    Ok(())
}
