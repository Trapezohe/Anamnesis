//! Minimal YAML frontmatter parser for Claude Code memory files.
//!
//! We do not depend on a full YAML library — auto-memory frontmatter only
//! ever uses a tiny grammar (top-level scalar lines + a single nested
//! `metadata:` block), and pulling in serde_yaml's dep tree for that is
//! overkill. If a memory file uses a richer YAML shape we cannot parse,
//! we just leave those fields empty; the normalizer falls back to safe
//! defaults.
//!
//! ## `type:` resolution (PR-G, BLUEPRINT §18.4 F2)
//!
//! We support **both** shapes Claude Code memory files use in the wild:
//!
//! ```yaml
//! metadata:
//!   type: reference        # Format A — current auto-generated shape (preferred)
//! ```
//!
//! ```yaml
//! type: feedback           # Format B — older / hand-written shape (fallback)
//! ```
//!
//! Resolution order: `metadata.type` first, top-level `type` second.
//! Either way, every other top-level key (e.g. `originSessionId`,
//! `node_type`, custom annotations) is preserved verbatim in
//! [`Frontmatter::extras`] so the normalizer can flow it into
//! `record.metadata` without dropping provenance.

use std::collections::BTreeMap;

// Reserved top-level keys handled by the match in `parse_yaml` directly:
// `name`, `description`, `type`, `metadata`. Anything else flows into
// `Frontmatter::extras`.

/// Parsed frontmatter fields we care about.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Frontmatter {
    /// `name:` slug — used as native id when present.
    pub name: Option<String>,
    /// `description:` one-line summary.
    pub description: Option<String>,
    /// Resolved memory type — one of {user, feedback, project, reference,
    /// preference, skill}. Falls back from `metadata.type` to top-level
    /// `type:`. See module docs.
    pub mem_type: Option<String>,
    /// All other top-level scalar keys, preserved for the normalizer to
    /// flow into `record.metadata`. Includes things like `originSessionId`,
    /// `node_type`, etc. Sorted for stable test assertions.
    pub extras: BTreeMap<String, String>,
}

/// Outcome of splitting a memory file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split<'a> {
    /// Parsed frontmatter, if any.
    pub frontmatter: Frontmatter,
    /// Markdown body (everything after the closing `---`). Returned with
    /// leading whitespace preserved trimmed of a single leading newline so
    /// callers see "natural" content.
    pub body: &'a str,
}

/// Split a memory file into frontmatter + body.
///
/// Returns `frontmatter` empty + `body = input` when no frontmatter
/// delimiters are present, so callers don't need to special-case.
pub fn split(input: &str) -> Split<'_> {
    let trimmed = input.trim_start_matches('\u{feff}'); // strip BOM if present
    if !trimmed.starts_with("---") {
        return Split {
            frontmatter: Frontmatter::default(),
            body: input,
        };
    }
    // Skip the leading `---` line.
    let after_open = match trimmed.find('\n') {
        Some(i) => &trimmed[i + 1..],
        None => {
            return Split {
                frontmatter: Frontmatter::default(),
                body: input,
            }
        }
    };
    // Find the closing `---` line.
    let close_idx = match find_close(after_open) {
        Some(i) => i,
        None => {
            return Split {
                frontmatter: Frontmatter::default(),
                body: input,
            }
        }
    };
    let yaml = &after_open[..close_idx];
    // Body starts after the line containing the closing `---`.
    let after_close = &after_open[close_idx..];
    let body_start = match after_close.find('\n') {
        Some(i) => &after_close[i + 1..],
        None => "",
    };
    Split {
        frontmatter: parse_yaml(yaml),
        body: body_start,
    }
}

fn find_close(s: &str) -> Option<usize> {
    let mut offset = 0usize;
    for line in s.split_inclusive('\n') {
        let trimmed_eol = line.trim_end_matches('\n').trim_end_matches('\r');
        if trimmed_eol == "---" {
            return Some(offset);
        }
        offset += line.len();
    }
    None
}

fn parse_yaml(yaml: &str) -> Frontmatter {
    let mut out = Frontmatter::default();
    let mut top_level_type: Option<String> = None;
    let mut metadata_type: Option<String> = None;
    let mut in_metadata = false;
    for raw_line in yaml.lines() {
        let line = raw_line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Top-level keys are not indented.
        let indent = line.bytes().take_while(|b| *b == b' ').count();
        if indent == 0 {
            in_metadata = false;
            if let Some((k, v)) = split_key_value(line) {
                match k {
                    "name" => out.name = Some(v),
                    "description" => out.description = Some(v),
                    "type" => {
                        // Format B — top-level type. Used as fallback when
                        // metadata.type is absent (see module docs).
                        top_level_type = Some(v);
                    }
                    "metadata" => {
                        // Either `metadata: {...}` inline (skipped — too rare to
                        // parse without a real YAML lib) or a nested block.
                        in_metadata = v.is_empty();
                    }
                    _ => {
                        // Preserve everything else for the normalizer to
                        // flow into record.metadata. Skip empty values to
                        // avoid noise.
                        if !v.is_empty() {
                            out.extras.insert(k.to_string(), v);
                        }
                    }
                }
            }
        } else if in_metadata {
            // Indented metadata.* lines.
            if let Some((k, v)) = split_key_value(line.trim_start()) {
                if k == "type" {
                    metadata_type = Some(v);
                }
            }
        }
    }
    // Resolution order: metadata.type wins; top-level type is the fallback.
    out.mem_type = metadata_type.or(top_level_type);
    out
}

/// Splits `key: value`, returning `(key, value-without-surrounding-whitespace-or-quotes)`.
fn split_key_value(line: &str) -> Option<(&str, String)> {
    let colon = line.find(':')?;
    let key = line[..colon].trim();
    if key.is_empty() {
        return None;
    }
    let raw_value = line[colon + 1..].trim();
    let value = raw_value
        .trim_start_matches(['"', '\''])
        .trim_end_matches(['"', '\''])
        .to_string();
    Some((key, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_typical_user_memory() {
        let input = r#"---
name: user-prefers-vim
description: User uses vim for everything
metadata:
  type: user
---

User prefers vim. Stop suggesting nano.
"#;
        let s = split(input);
        assert_eq!(s.frontmatter.name.as_deref(), Some("user-prefers-vim"));
        assert_eq!(
            s.frontmatter.description.as_deref(),
            Some("User uses vim for everything")
        );
        assert_eq!(s.frontmatter.mem_type.as_deref(), Some("user"));
        assert!(s.body.contains("User prefers vim"));
    }

    #[test]
    fn parses_feedback_with_extra_metadata() {
        let input = r#"---
name: no-mocked-db
description: Use real DB in integration tests
metadata:
  type: feedback
  added: 2026-05-16
---
Body."#;
        let fm = split(input).frontmatter;
        assert_eq!(fm.mem_type.as_deref(), Some("feedback"));
        assert_eq!(fm.name.as_deref(), Some("no-mocked-db"));
    }

    #[test]
    fn no_frontmatter_returns_whole_input_as_body() {
        let input = "just markdown, no fences\n";
        let s = split(input);
        assert!(s.frontmatter == Frontmatter::default());
        assert_eq!(s.body, input);
    }

    #[test]
    fn unterminated_frontmatter_falls_back_to_body() {
        let input = "---\nname: x\nno closing fence\n";
        let s = split(input);
        // Closing delimiter missing → treat as no frontmatter.
        assert!(s.frontmatter == Frontmatter::default());
        assert_eq!(s.body, input);
    }

    #[test]
    fn handles_bom() {
        let input = "\u{feff}---\nname: x\nmetadata:\n  type: user\n---\nbody";
        let fm = split(input).frontmatter;
        assert_eq!(fm.name.as_deref(), Some("x"));
        assert_eq!(fm.mem_type.as_deref(), Some("user"));
    }

    #[test]
    fn strips_surrounding_quotes() {
        let input = "---\nname: \"quoted-name\"\ndescription: 'apostrophed'\nmetadata:\n  type: project\n---\nx";
        let fm = split(input).frontmatter;
        assert_eq!(fm.name.as_deref(), Some("quoted-name"));
        assert_eq!(fm.description.as_deref(), Some("apostrophed"));
        assert_eq!(fm.mem_type.as_deref(), Some("project"));
    }

    #[test]
    fn body_excludes_frontmatter_block() {
        let input = "---\nname: x\nmetadata:\n  type: user\n---\nhello\nworld\n";
        let s = split(input);
        assert_eq!(s.body, "hello\nworld\n");
    }

    // ─── PR-G: top-level `type:` + extras preservation ───

    #[test]
    fn top_level_type_is_used_when_no_metadata_block() {
        // Format B in the wild: hand-written / older auto-memory.
        let input = r#"---
name: project-location
description: prefer ~/Desktop over /tmp
type: feedback
originSessionId: 76a78a2d-e2af-4a15-9be4-f970d9e26e41
---
body"#;
        let fm = split(input).frontmatter;
        assert_eq!(
            fm.mem_type.as_deref(),
            Some("feedback"),
            "top-level type: feedback must resolve when metadata.type is absent"
        );
        assert_eq!(
            fm.extras.get("originSessionId").map(String::as_str),
            Some("76a78a2d-e2af-4a15-9be4-f970d9e26e41"),
            "originSessionId must be preserved for the normalizer"
        );
    }

    #[test]
    fn metadata_type_wins_over_top_level_type() {
        // If a file has both (rare but possible), metadata.type is
        // authoritative — that's the Claude Code auto-generated convention.
        let input = r#"---
name: x
type: feedback
metadata:
  type: reference
---
body"#;
        let fm = split(input).frontmatter;
        assert_eq!(
            fm.mem_type.as_deref(),
            Some("reference"),
            "metadata.type overrides top-level type"
        );
    }

    #[test]
    fn extras_capture_unknown_top_level_keys() {
        let input = r#"---
name: x
description: y
metadata:
  type: reference
originSessionId: abc-123
node_type: memory
custom_tag: foo
---
body"#;
        let fm = split(input).frontmatter;
        assert_eq!(fm.mem_type.as_deref(), Some("reference"));
        let snapshot: Vec<_> = fm.extras.iter().collect();
        // BTreeMap iteration is sorted, so we can assert exact contents.
        assert_eq!(
            snapshot,
            vec![
                (&"custom_tag".to_string(), &"foo".to_string()),
                (&"node_type".to_string(), &"memory".to_string()),
                (&"originSessionId".to_string(), &"abc-123".to_string()),
            ]
        );
    }

    #[test]
    fn reserved_keys_do_not_leak_into_extras() {
        let input = r#"---
name: x
description: y
type: feedback
metadata:
  type: feedback
---
body"#;
        let fm = split(input).frontmatter;
        assert!(
            fm.extras.is_empty(),
            "reserved keys must not show up in extras"
        );
    }

    #[test]
    fn unrecognized_type_value_passes_through_to_normalizer() {
        // Parser doesn't validate against the enum — normalizer does.
        // We just make sure the raw string round-trips.
        let input = "---\ntype: weird-experimental\n---\nbody";
        let fm = split(input).frontmatter;
        assert_eq!(fm.mem_type.as_deref(), Some("weird-experimental"));
    }
}
