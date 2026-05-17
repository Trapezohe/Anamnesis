//! Claude Code session JSONL parser + readable markdown renderer.
//!
//! Implements PR-H (BLUEPRINT §18.4 F3). Before this module, the session
//! normalizer shoved the entire raw JSONL text into `record.content`. With
//! 1700+ sessions per real `~/.claude/projects/`, that produced 21k+
//! chunks per record and snippets full of `{"role":"user","content":...}`
//! byte fragments — humans couldn't read them, agents couldn't reason on
//! them.
//!
//! We now:
//!   1. Parse JSONL line-by-line, tolerating bad lines.
//!   2. Keep only `type == "user" | "assistant"` (the actual turns); skip
//!      `permission-mode`, `file-history-snapshot`, `ai-title`, …
//!   3. Flatten `message.content` (string or content-blocks array) into
//!      a single text per turn, with `tool_use` / `tool_result` folded to
//!      one-line tags.
//!   4. Render the resulting `Vec<SessionMessage>` as readable markdown
//!      (`**user** — 2026-05-17T03:14Z:\n\nhi\n\n**assistant**: …`).
//!
//! Tool result content is truncated to [`TOOL_RESULT_PREVIEW_CHARS`] to
//! keep `Episode` records bounded — full tool output stays available via
//! `raw_artifacts` for provenance.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Cap tool-result preview text in the rendered transcript. Full tool
/// output is preserved verbatim in `raw_artifacts.payload_json`.
pub const TOOL_RESULT_PREVIEW_CHARS: usize = 400;

/// One parsed turn from a session JSONL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMessage {
    /// Speaker role — typically `"user"` or `"assistant"`. Falls back to
    /// the outer `type` field when `message.role` is missing.
    pub role: String,
    /// Flattened text for the turn. Empty turns (pure thinking-only /
    /// image-only) are filtered out at parse time.
    pub text: String,
    /// Outer `timestamp` in UTC if it parsed cleanly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
    /// Outer `uuid` from the JSONL row — useful for provenance / dedup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uuid: Option<String>,
}

/// Result of parsing an entire JSONL file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedSession {
    /// Turns in source order.
    pub messages: Vec<SessionMessage>,
    /// Earliest message timestamp, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_ts: Option<DateTime<Utc>>,
    /// Latest message timestamp, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_ts: Option<DateTime<Utc>>,
}

/// Parse a JSONL session file into structured turns.
///
/// Bad lines are silently skipped (logged via `tracing::trace!` for
/// debug). Non-`user`/`assistant` rows (permission mode, file snapshots,
/// system messages, …) are filtered out — they're metadata, not memory.
pub fn parse_jsonl(text: &str) -> ParsedSession {
    let mut session = ParsedSession::default();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                tracing::trace!(line = lineno + 1, error = %e, "skipping bad JSONL line");
                continue;
            }
        };
        let kind = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
        if kind != "user" && kind != "assistant" {
            continue;
        }
        let Some(msg) = v.get("message") else {
            continue;
        };
        let role = msg
            .get("role")
            .and_then(|x| x.as_str())
            .unwrap_or(kind)
            .to_string();
        let text = extract_text(msg.get("content").unwrap_or(&Value::Null));
        if text.is_empty() {
            continue; // pure thinking / image-only turn — nothing to index
        }
        let timestamp = v
            .get("timestamp")
            .and_then(|x| x.as_str())
            .and_then(parse_rfc3339);
        let uuid = v.get("uuid").and_then(|x| x.as_str()).map(str::to_owned);
        if let Some(t) = timestamp {
            session.first_ts = Some(session.first_ts.map(|f| f.min(t)).unwrap_or(t));
            session.last_ts = Some(session.last_ts.map(|l| l.max(t)).unwrap_or(t));
        }
        session.messages.push(SessionMessage {
            role,
            text,
            timestamp,
            uuid,
        });
    }
    session
}

fn parse_rfc3339(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

/// Flatten `message.content` into a single text. Handles all three shapes
/// we've seen in real Claude Code data:
///
///   - `string`                  → returned as-is
///   - `Vec<{ type: text }>`     → concatenated
///   - `Vec<{ type: tool_use }>` → folded to `[tool_use: <name>]`
///   - `Vec<{ type: tool_result }>` → folded to `[tool_result(N chars): <preview…>]`
///   - `thinking` / `image` / other → dropped (high noise, low signal)
fn extract_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut parts: Vec<String> = Vec::with_capacity(blocks.len());
            for block in blocks {
                let typ = block.get("type").and_then(|x| x.as_str()).unwrap_or("");
                match typ {
                    "text" => {
                        if let Some(t) = block.get("text").and_then(|x| x.as_str()) {
                            let t = t.trim();
                            if !t.is_empty() {
                                parts.push(t.to_string());
                            }
                        }
                    }
                    "tool_use" => {
                        let name = block.get("name").and_then(|x| x.as_str()).unwrap_or("?");
                        parts.push(format!("[tool_use: {name}]"));
                    }
                    "tool_result" => {
                        let raw = block.get("content").and_then(|x| x.as_str()).unwrap_or("");
                        let preview: String = raw.chars().take(TOOL_RESULT_PREVIEW_CHARS).collect();
                        let truncated = if raw.chars().count() > TOOL_RESULT_PREVIEW_CHARS {
                            "…"
                        } else {
                            ""
                        };
                        parts.push(format!(
                            "[tool_result({chars} chars): {preview}{truncated}]",
                            chars = raw.chars().count(),
                        ));
                    }
                    // thinking / image / unknown → drop
                    _ => {}
                }
            }
            parts.join("\n\n")
        }
        _ => String::new(),
    }
}

/// Render a parsed session as readable markdown. Each turn becomes:
///
/// ```text
/// **user** — 2026-05-17T03:14Z:
///
/// (turn text)
/// ```
///
/// Empty when there are no turns.
pub fn render_markdown(session: &ParsedSession) -> String {
    if session.messages.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(session.messages.len() * 128);
    for m in &session.messages {
        match m.timestamp {
            Some(t) => out.push_str(&format!(
                "**{}** — {}\n\n",
                m.role,
                t.format("%Y-%m-%dT%H:%MZ")
            )),
            None => out.push_str(&format!("**{}**\n\n", m.role)),
        }
        out.push_str(m.text.trim());
        out.push_str("\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(json: &str) -> String {
        json.replace('\n', "")
    }

    #[test]
    fn parse_keeps_only_user_assistant_turns() {
        let input = [
            // Non-turn rows — must all be skipped.
            line(r#"{"type":"permission-mode","permissionMode":"default"}"#),
            line(r#"{"type":"file-history-snapshot","messageId":"x"}"#),
            line(r#"{"type":"ai-title","title":"about something"}"#),
            line(r#"{"type":"system","content":"system note"}"#),
            // Real turns.
            line(r#"{"type":"user","message":{"role":"user","content":"hi"},"timestamp":"2026-05-17T03:14:00Z","uuid":"u1"}"#),
            line(r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]},"timestamp":"2026-05-17T03:14:05Z","uuid":"a1"}"#),
        ]
        .join("\n");

        let s = parse_jsonl(&input);
        assert_eq!(s.messages.len(), 2, "only the 2 real turns are kept");
        assert_eq!(s.messages[0].role, "user");
        assert_eq!(s.messages[0].text, "hi");
        assert_eq!(s.messages[0].uuid.as_deref(), Some("u1"));
        assert_eq!(s.messages[1].role, "assistant");
        assert_eq!(s.messages[1].text, "hello");
    }

    #[test]
    fn parse_extracts_first_and_last_timestamp() {
        let input = [
            r#"{"type":"user","message":{"role":"user","content":"a"},"timestamp":"2026-05-17T03:14:00Z"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"b"}]},"timestamp":"2026-05-17T03:14:30Z"}"#,
            r#"{"type":"user","message":{"role":"user","content":"c"},"timestamp":"2026-05-17T03:15:00Z"}"#,
        ]
        .join("\n");
        let s = parse_jsonl(&input);
        assert_eq!(
            s.first_ts.map(|t| t.to_rfc3339()),
            Some("2026-05-17T03:14:00+00:00".to_string())
        );
        assert_eq!(
            s.last_ts.map(|t| t.to_rfc3339()),
            Some("2026-05-17T03:15:00+00:00".to_string())
        );
    }

    #[test]
    fn parse_handles_array_content_with_mixed_block_types() {
        let input = r#"{"type":"assistant","message":{"role":"assistant","content":[
            {"type":"thinking","thinking":"internal monologue should be dropped"},
            {"type":"text","text":"Hello there"},
            {"type":"tool_use","id":"toolu_1","name":"Read","input":{"path":"/x"}},
            {"type":"text","text":"Done."}
        ]},"timestamp":"2026-05-17T03:14:00Z"}"#
            .replace('\n', "");
        let s = parse_jsonl(&input);
        assert_eq!(s.messages.len(), 1);
        let text = &s.messages[0].text;
        assert!(text.contains("Hello there"));
        assert!(text.contains("[tool_use: Read]"));
        assert!(text.contains("Done."));
        assert!(
            !text.contains("internal monologue"),
            "thinking blocks must be dropped"
        );
    }

    #[test]
    fn parse_truncates_long_tool_results_with_preview() {
        let long_output = "x".repeat(1000);
        let line = format!(
            r#"{{"type":"user","message":{{"role":"user","content":[{{"type":"tool_result","content":"{long_output}"}}]}},"timestamp":"2026-05-17T03:14:00Z"}}"#,
        );
        let s = parse_jsonl(&line);
        assert_eq!(s.messages.len(), 1);
        let text = &s.messages[0].text;
        assert!(text.starts_with("[tool_result(1000 chars):"));
        assert!(text.ends_with("…]"));
        // The text length should be bounded by the preview budget +
        // formatting overhead, not the raw 1000-char payload.
        assert!(
            text.len() < TOOL_RESULT_PREVIEW_CHARS + 100,
            "rendered tool_result must be capped"
        );
    }

    #[test]
    fn parse_drops_turns_with_no_text_content() {
        // Pure thinking-only message — no usable text after extraction.
        let input = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"...","signature":"x"}]},"timestamp":"2026-05-17T03:14:00Z"}"#;
        let s = parse_jsonl(input);
        assert!(s.messages.is_empty(), "thinking-only turns must be dropped");
    }

    #[test]
    fn parse_tolerates_bad_lines() {
        let input = [
            r#"{"type":"user","message":{"role":"user","content":"good"},"timestamp":"2026-05-17T03:14:00Z"}"#,
            "not-json-garbage",
            r#"{ "incomplete: "#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"also good"}]},"timestamp":"2026-05-17T03:14:05Z"}"#,
        ]
        .join("\n");
        let s = parse_jsonl(&input);
        assert_eq!(s.messages.len(), 2);
    }

    #[test]
    fn parse_empty_input_returns_default() {
        let s = parse_jsonl("");
        assert!(s.messages.is_empty());
        assert!(s.first_ts.is_none());
        assert!(s.last_ts.is_none());
    }

    #[test]
    fn render_markdown_uses_role_and_timestamp_headers() {
        let s = ParsedSession {
            messages: vec![
                SessionMessage {
                    role: "user".into(),
                    text: "hi there".into(),
                    timestamp: parse_rfc3339("2026-05-17T03:14:00Z"),
                    uuid: None,
                },
                SessionMessage {
                    role: "assistant".into(),
                    text: "hello back".into(),
                    timestamp: parse_rfc3339("2026-05-17T03:14:05Z"),
                    uuid: None,
                },
            ],
            first_ts: parse_rfc3339("2026-05-17T03:14:00Z"),
            last_ts: parse_rfc3339("2026-05-17T03:14:05Z"),
        };
        let md = render_markdown(&s);
        assert!(md.contains("**user**"));
        assert!(md.contains("**assistant**"));
        assert!(md.contains("2026-05-17T03:14Z"));
        assert!(md.contains("hi there"));
        assert!(md.contains("hello back"));
    }

    #[test]
    fn render_markdown_handles_missing_timestamp() {
        let s = ParsedSession {
            messages: vec![SessionMessage {
                role: "user".into(),
                text: "hi".into(),
                timestamp: None,
                uuid: None,
            }],
            first_ts: None,
            last_ts: None,
        };
        let md = render_markdown(&s);
        assert!(md.contains("**user**"));
        assert!(!md.contains("—"));
        assert!(md.contains("hi"));
    }

    #[test]
    fn render_markdown_empty_session_is_empty_string() {
        assert_eq!(render_markdown(&ParsedSession::default()), "");
    }

    #[test]
    fn chunk_count_estimate_drops_dramatically() {
        // Synthetic 200-turn session ≈ 20 KB. Today's path puts this all
        // in `content` then char-chunks it; PR-H per-turn rendering
        // produces N markdown blocks roughly bounded by turn count.
        // We just sanity-check that the rendered content is much smaller
        // than the raw JSONL.
        let mut raw = String::new();
        for i in 0..200 {
            raw.push_str(&format!(
                r#"{{"type":"user","message":{{"role":"user","content":"turn {i}"}},"timestamp":"2026-05-17T03:14:00Z"}}"#,
            ));
            raw.push('\n');
        }
        let s = parse_jsonl(&raw);
        let md = render_markdown(&s);
        assert_eq!(s.messages.len(), 200);
        assert!(
            md.len() < raw.len() / 2,
            "rendered markdown ({}) should be much smaller than raw JSONL ({})",
            md.len(),
            raw.len()
        );
    }
}
