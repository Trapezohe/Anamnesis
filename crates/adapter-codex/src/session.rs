//! Codex rollout JSONL parser + readable markdown renderer.
//!
//! Mirrors `adapter-claude-code/src/session.rs` (issue #5 / PR-H) but
//! handles the OpenAI Codex CLI rollout schema, which differs in three
//! places:
//!
//!   - turn-row type is `"response_item"`, not `"user"`/`"assistant"`
//!   - role is at `payload.role` (`"user"` / `"assistant"`)
//!   - content blocks use `"input_text"` / `"output_text"` (not `"text"`)
//!
//! Other row types are metadata we drop:
//!   - `session_meta`            — session-level info
//!   - `event_msg`               — internal lifecycle events
//!   - `turn_context`            — model/turn boundary markers
//!   - `compacted`               — LLM-generated context summaries
//!     (already-summarized; re-including them double-counts)
//!
//! Issue #69 fix: before this module the codex adapter dumped the full
//! JSONL body into `content`, producing 6,778 chunks per record on real
//! `~/.codex` data. With per-turn rendering, real `~/.codex/archived_sessions`
//! sits well inside the BLUEPRINT §16.6 "< 30 chunks/record" trigger.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Cap tool-result preview text in the rendered transcript. Kept for
/// symmetry with the claude-code parser; real codex rollouts have not
/// shown `tool_result` blocks yet (tool output goes through `event_msg`
/// in current Codex CLI versions), but the cap is here so future
/// versions don't blow up record size.
pub const TOOL_RESULT_PREVIEW_CHARS: usize = 400;

/// One parsed turn from a Codex rollout JSONL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMessage {
    /// Speaker role — `"user"` or `"assistant"` from `payload.role`.
    pub role: String,
    /// Flattened text for the turn. Empty turns (no text blocks at all)
    /// are filtered out at parse time.
    pub text: String,
    /// Outer `timestamp` in UTC if it parsed cleanly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
}

/// Result of parsing an entire Codex rollout JSONL file.
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

/// Parse a Codex rollout JSONL file into structured turns.
///
/// Skips non-message rows (`session_meta`, `event_msg`, `turn_context`,
/// `compacted`, …). Bad JSON lines are logged at `trace` level and
/// skipped — robustness over strictness, same policy as the claude-code
/// parser.
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
        if v.get("type").and_then(|x| x.as_str()) != Some("response_item") {
            continue;
        }
        let Some(payload) = v.get("payload") else {
            continue;
        };
        // Codex wraps tool calls / function results in non-`message`
        // payload types; for the Episode transcript we only keep
        // human-readable conversation turns (`type == "message"`). When
        // `payload.type` is missing entirely, default to `"message"` so
        // older Codex versions that omitted the field still parse.
        let payload_type = payload
            .get("type")
            .and_then(|x| x.as_str())
            .unwrap_or("message");
        if payload_type != "message" {
            continue;
        }
        let Some(role) = payload.get("role").and_then(|x| x.as_str()) else {
            continue;
        };
        if role != "user" && role != "assistant" {
            continue;
        }
        let text = extract_text(payload.get("content").unwrap_or(&Value::Null));
        if text.is_empty() {
            continue;
        }
        let timestamp = v
            .get("timestamp")
            .and_then(|x| x.as_str())
            .and_then(parse_rfc3339);
        if let Some(t) = timestamp {
            session.first_ts = Some(session.first_ts.map(|f| f.min(t)).unwrap_or(t));
            session.last_ts = Some(session.last_ts.map(|l| l.max(t)).unwrap_or(t));
        }
        session.messages.push(SessionMessage {
            role: role.to_string(),
            text,
            timestamp,
        });
    }
    session
}

fn parse_rfc3339(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

/// Flatten `payload.content` blocks into a single text. Handles the
/// shapes we've seen in real Codex rollouts:
///
///   - `string`                          → returned as-is
///   - `Vec<{ type: input_text, text }>` → concatenated
///   - `Vec<{ type: output_text, text }>` → concatenated
///   - `Vec<{ type: tool_result, content }>` → folded to one-line preview
///   - other block types → dropped
fn extract_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut parts: Vec<String> = Vec::with_capacity(blocks.len());
            for block in blocks {
                let typ = block.get("type").and_then(|x| x.as_str()).unwrap_or("");
                match typ {
                    "input_text" | "output_text" | "text" => {
                        if let Some(t) = block.get("text").and_then(|x| x.as_str()) {
                            let t = t.trim();
                            if !t.is_empty() {
                                parts.push(t.to_string());
                            }
                        }
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
                    // input_image / reasoning / unknown → drop
                    _ => {}
                }
            }
            parts.join("\n\n")
        }
        _ => String::new(),
    }
}

/// Render a parsed Codex rollout as readable markdown — identical to the
/// claude-code format so downstream consumers see one consistent shape.
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

    #[test]
    fn parse_keeps_only_response_item_message_rows() {
        let input = [
            // Non-message rows — all must be skipped.
            r#"{"timestamp":"2026-05-01T00:00:00Z","type":"session_meta","payload":{"id":"x"}}"#,
            r#"{"timestamp":"2026-05-01T00:00:01Z","type":"event_msg","payload":{"kind":"info"}}"#,
            r#"{"timestamp":"2026-05-01T00:00:02Z","type":"turn_context","payload":{"turn_id":"t1"}}"#,
            r#"{"timestamp":"2026-05-01T00:00:03Z","type":"compacted","payload":{"summary":"already-distilled"}}"#,
            // Real turns.
            r#"{"timestamp":"2026-05-01T00:00:10Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}"#,
            r#"{"timestamp":"2026-05-01T00:00:11Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}}"#,
        ]
        .join("\n");

        let s = parse_jsonl(&input);
        assert_eq!(s.messages.len(), 2);
        assert_eq!(s.messages[0].role, "user");
        assert_eq!(s.messages[0].text, "hi");
        assert_eq!(s.messages[1].role, "assistant");
        assert_eq!(s.messages[1].text, "hello");
    }

    #[test]
    fn parse_handles_string_content() {
        // Older Codex format: content was sometimes a bare string.
        let input = r#"{"timestamp":"2026-05-01T00:00:00Z","type":"response_item","payload":{"type":"message","role":"user","content":"raw string content"}}"#;
        let s = parse_jsonl(input);
        assert_eq!(s.messages.len(), 1);
        assert_eq!(s.messages[0].text, "raw string content");
    }

    #[test]
    fn parse_concatenates_multiple_text_blocks() {
        let input = r#"{"timestamp":"2026-05-01T00:00:00Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[
            {"type":"output_text","text":"Hello"},
            {"type":"output_text","text":"there"}
        ]}}"#
            .replace('\n', "");
        let s = parse_jsonl(&input);
        assert_eq!(s.messages.len(), 1);
        assert!(s.messages[0].text.contains("Hello"));
        assert!(s.messages[0].text.contains("there"));
    }

    #[test]
    fn parse_skips_unknown_block_types() {
        let input = r#"{"timestamp":"2026-05-01T00:00:00Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[
            {"type":"reasoning","text":"internal monologue should be dropped"},
            {"type":"output_text","text":"public answer"},
            {"type":"input_image","image_url":"https://x/y.png"}
        ]}}"#
            .replace('\n', "");
        let s = parse_jsonl(&input);
        assert_eq!(s.messages.len(), 1);
        assert_eq!(s.messages[0].text, "public answer");
    }

    #[test]
    fn parse_extracts_first_and_last_timestamp() {
        let input = [
            r#"{"timestamp":"2026-05-01T00:00:00Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"a"}]}}"#,
            r#"{"timestamp":"2026-05-01T00:00:30Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"b"}]}}"#,
            r#"{"timestamp":"2026-05-01T00:01:00Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"c"}]}}"#,
        ]
        .join("\n");
        let s = parse_jsonl(&input);
        assert_eq!(
            s.first_ts.map(|t| t.to_rfc3339()),
            Some("2026-05-01T00:00:00+00:00".to_string())
        );
        assert_eq!(
            s.last_ts.map(|t| t.to_rfc3339()),
            Some("2026-05-01T00:01:00+00:00".to_string())
        );
    }

    #[test]
    fn parse_drops_empty_turns() {
        let input = r#"{"timestamp":"2026-05-01T00:00:00Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"reasoning","text":"only thinking"}]}}"#;
        let s = parse_jsonl(input);
        assert!(s.messages.is_empty());
    }

    #[test]
    fn parse_tolerates_bad_lines() {
        let input = [
            r#"{"timestamp":"2026-05-01T00:00:00Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"good"}]}}"#,
            "not-json",
            r#"{ "incomplete: "#,
            r#"{"timestamp":"2026-05-01T00:00:05Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"also good"}]}}"#,
        ]
        .join("\n");
        let s = parse_jsonl(&input);
        assert_eq!(s.messages.len(), 2);
    }

    #[test]
    fn render_markdown_uses_role_and_timestamp_headers() {
        let s = ParsedSession {
            messages: vec![
                SessionMessage {
                    role: "user".into(),
                    text: "hi there".into(),
                    timestamp: parse_rfc3339("2026-05-01T00:00:00Z"),
                },
                SessionMessage {
                    role: "assistant".into(),
                    text: "hello back".into(),
                    timestamp: parse_rfc3339("2026-05-01T00:00:05Z"),
                },
            ],
            first_ts: parse_rfc3339("2026-05-01T00:00:00Z"),
            last_ts: parse_rfc3339("2026-05-01T00:00:05Z"),
        };
        let md = render_markdown(&s);
        assert!(md.contains("**user**"));
        assert!(md.contains("**assistant**"));
        assert!(md.contains("2026-05-01T00:00Z"));
        assert!(md.contains("hi there"));
        assert!(md.contains("hello back"));
    }

    #[test]
    fn render_markdown_empty_session_is_empty_string() {
        assert_eq!(render_markdown(&ParsedSession::default()), "");
    }

    /// Issue #69 acceptance: per-turn rendering brings content size
    /// well below the raw JSONL size, so chunks/record stays inside the
    /// BLUEPRINT §16.6 trigger instead of the 6,778× overshoot we saw
    /// on real `~/.codex/archived_sessions` data at v0.1.0.
    #[test]
    fn chunk_count_estimate_drops_dramatically() {
        let mut raw = String::new();
        for i in 0..200 {
            raw.push_str(&format!(
                r#"{{"timestamp":"2026-05-01T00:00:00Z","type":"response_item","payload":{{"type":"message","role":"user","content":[{{"type":"input_text","text":"turn {i} with some realistic context to chew on"}}]}}}}"#,
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
