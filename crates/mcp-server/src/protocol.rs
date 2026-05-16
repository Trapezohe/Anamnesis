//! Minimal JSON-RPC 2.0 types — just enough for the MCP subset we
//! implement. We don't pull a JSON-RPC crate because the protocol's
//! shape is tiny and we want zero surprise dependencies.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Inbound JSON-RPC request or notification.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcRequest {
    /// Always `"2.0"`. We ignore mismatches.
    pub jsonrpc: String,
    /// `id` is absent for notifications; present (string|number|null) for
    /// requests that expect a response.
    #[serde(default)]
    pub id: Option<Value>,
    /// The method name (`"tools/list"`, `"tools/call"`, …).
    pub method: String,
    /// Params object; method-specific shape.
    #[serde(default)]
    pub params: Value,
}

impl JsonRpcRequest {
    /// Notifications carry no id and don't expect a response.
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

/// Outbound JSON-RPC response.
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcResponse {
    /// Always `"2.0"`.
    pub jsonrpc: &'static str,
    /// Echo of the request id.
    pub id: Value,
    /// Result payload (mutually exclusive with `error`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error envelope.
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcError {
    /// Numeric error code; we use:
    ///   -32700 parse error
    ///   -32600 invalid request
    ///   -32601 method not found
    ///   -32602 invalid params
    ///   -32603 internal error
    pub code: i64,
    /// Human-readable message.
    pub message: String,
    /// Optional structured data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    /// Build a successful response.
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Build an error response.
    pub fn err(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_without_id_is_notification() {
        let r: JsonRpcRequest =
            serde_json::from_str(r#"{"jsonrpc":"2.0","method":"x","params":{}}"#).unwrap();
        assert!(r.is_notification());
    }

    #[test]
    fn request_with_id_is_call() {
        let r: JsonRpcRequest =
            serde_json::from_str(r#"{"jsonrpc":"2.0","id":1,"method":"x"}"#).unwrap();
        assert!(!r.is_notification());
        assert_eq!(r.id, Some(Value::from(1)));
    }

    #[test]
    fn ok_response_omits_error_field() {
        let r = JsonRpcResponse::ok(Value::from(1), serde_json::json!({"y": 2}));
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"result\""));
        assert!(!s.contains("\"error\""));
    }

    #[test]
    fn err_response_omits_result_field() {
        let r = JsonRpcResponse::err(Value::from(1), -32601, "no such method");
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"error\""));
        assert!(!s.contains("\"result\""));
        assert!(s.contains("-32601"));
    }
}
