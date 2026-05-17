//! HTTP transport for the MCP server (BLUEPRINT §6.1 "SSE" row).
//!
//! Phase 3 ships a minimal request/response HTTP transport rather than
//! pure SSE — full bidirectional Streamable-HTTP (with server-pushed
//! notifications) lands in Phase 4 alongside the watcher subsystem that
//! actually needs server-initiated events.
//!
//! Surface:
//!   POST /mcp                         — accepts JSON-RPC, returns JSON-RPC
//!   GET  /healthz                     — `200 ok` (no auth)
//!
//! Auth: every request to `/mcp` must include `Authorization: Bearer
//! <token>`. The token is generated at boot (64 random bytes →
//! url-safe-base64) and printed to stderr exactly once so the operator
//! can paste it into the client.
//!
//! Bind: defaults to `127.0.0.1:<port>` — loopback only. Operators who
//! need remote access run an SSH tunnel or wrap with a TLS-terminating
//! reverse proxy.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::State,
    http::{header::AUTHORIZATION, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::Value;
use tracing::info;

use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
use crate::server::AnamnesisServer;

/// HTTP server configuration.
#[derive(Debug, Clone)]
pub struct HttpServerConfig {
    /// TCP port to bind. Bind address is always `127.0.0.1`.
    pub port: u16,
    /// Pre-shared bearer token. `None` → generate a 64-byte random one
    /// at startup and log to stderr.
    pub token: Option<String>,
}

impl HttpServerConfig {
    /// Convenience constructor.
    pub fn new(port: u16) -> Self {
        Self { port, token: None }
    }
}

/// Build a router and serve until the process is killed or the future
/// is dropped.
///
/// Always prints the *actual* bound address to stderr (`anamnesis-mcp
/// HTTP — listening on http://127.0.0.1:<port>`) after the listener is
/// open — this is required for `--sse 0` ephemeral-port mode: tests and
/// supervisors need a way to discover which port the kernel picked.
/// Format is stable: `^anamnesis-mcp HTTP — listening on http://127\.0\.0\.1:(\d+)$`.
pub async fn run(server: AnamnesisServer, config: HttpServerConfig) -> anyhow::Result<()> {
    let app_state = AppState::new(server, config.token);
    eprintln!(
        "anamnesis-mcp HTTP — bearer token: {token}",
        token = app_state.token
    );
    eprintln!("anamnesis-mcp HTTP — clients must send `Authorization: Bearer <token>` on /mcp",);

    let app = build_router(Arc::new(app_state));
    let addr: SocketAddr = ([127, 0, 0, 1], config.port).into();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    eprintln!("anamnesis-mcp HTTP — listening on http://{bound}");
    info!(addr = %bound, "anamnesis-mcp HTTP listening");
    axum::serve(listener, app).await?;
    Ok(())
}

/// Bind without serving — returns (listener, bound_addr, app). Tests
/// use this to spawn the server on an ephemeral port.
pub async fn bind(
    server: AnamnesisServer,
    token: Option<String>,
) -> anyhow::Result<(tokio::net::TcpListener, SocketAddr, Router, String)> {
    let app_state = AppState::new(server, token);
    let token = app_state.token.clone();
    let app = build_router(Arc::new(app_state));
    let addr: SocketAddr = ([127, 0, 0, 1], 0).into();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    Ok((listener, bound, app, token))
}

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/mcp", post(handle_mcp))
        .route("/healthz", get(healthz))
        .with_state(state)
}

struct AppState {
    server: AnamnesisServer,
    token: String,
}

impl AppState {
    fn new(server: AnamnesisServer, token: Option<String>) -> Self {
        Self {
            server,
            token: token.unwrap_or_else(generate_token),
        }
    }
}

fn generate_token() -> String {
    use rand::RngCore;
    let mut buf = [0u8; 48];
    rand::thread_rng().fill_bytes(&mut buf);
    base64_url(&buf)
}

/// Tiny base64-url encoder so we don't pull in another dep just for one
/// 48-byte buffer. Standard alphabet, no padding.
fn base64_url(input: &[u8]) -> String {
    const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let mut buf = [0u8; 3];
        buf[..chunk.len()].copy_from_slice(chunk);
        let b = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);
        out.push(ALPHA[((b >> 18) & 0x3f) as usize] as char);
        out.push(ALPHA[((b >> 12) & 0x3f) as usize] as char);
        if chunk.len() >= 2 {
            out.push(ALPHA[((b >> 6) & 0x3f) as usize] as char);
        }
        if chunk.len() == 3 {
            out.push(ALPHA[(b & 0x3f) as usize] as char);
        }
    }
    out
}

async fn healthz() -> &'static str {
    "ok"
}

#[axum::debug_handler]
async fn handle_mcp(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !auth_ok(&headers, &state.token) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(JsonRpcResponse::err(
                Value::Null,
                -32001,
                "missing or invalid bearer token",
            )),
        )
            .into_response();
    }
    let req: JsonRpcRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(JsonRpcResponse::err(
                    Value::Null,
                    -32700,
                    format!("parse error: {e}"),
                )),
            )
                .into_response();
        }
    };
    let is_note = req.is_notification();
    let response = state.server.handle(req).await;
    if is_note {
        return StatusCode::NO_CONTENT.into_response();
    }
    Json(response).into_response()
}

fn auth_ok(headers: &HeaderMap, expected: &str) -> bool {
    let Some(value) = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok()) else {
        return false;
    };
    let stripped = match value.strip_prefix("Bearer ") {
        Some(s) => s,
        None => return false,
    };
    constant_time_eq(stripped.as_bytes(), expected.as_bytes())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_url_no_padding_for_aligned_input() {
        // 3 bytes → exactly 4 chars, no padding.
        assert_eq!(base64_url(b"abc"), "YWJj");
        assert_eq!(base64_url(b"hello!"), "aGVsbG8h");
    }

    #[test]
    fn base64_url_handles_unaligned_input() {
        assert_eq!(base64_url(b"ab"), "YWI"); // 2 → 3 chars
        assert_eq!(base64_url(b"a"), "YQ"); // 1 → 2 chars
    }

    #[test]
    fn generate_token_is_64_chars_url_safe() {
        let t = generate_token();
        // 48 bytes → 64 base64-url chars (no padding).
        assert_eq!(t.len(), 64);
        assert!(t
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    }

    #[test]
    fn constant_time_eq_basics() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hellos"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn auth_ok_requires_bearer_prefix() {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, "Bearer abc123".parse().unwrap());
        assert!(auth_ok(&headers, "abc123"));

        headers.insert(AUTHORIZATION, "abc123".parse().unwrap());
        assert!(!auth_ok(&headers, "abc123"));

        headers.insert(AUTHORIZATION, "Bearer wrong".parse().unwrap());
        assert!(!auth_ok(&headers, "abc123"));
    }
}
