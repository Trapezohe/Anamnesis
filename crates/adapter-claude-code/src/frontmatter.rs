//! Minimal YAML frontmatter parser for Claude Code memory files.
//!
//! We do not depend on a full YAML library — auto-memory frontmatter only
//! ever uses a tiny grammar (top-level scalar lines + a single nested
//! `metadata:` block), and pulling in serde_yaml's dep tree for that is
//! overkill. If a memory file uses a richer YAML shape we cannot parse,
//! we just leave those fields empty; the normalizer falls back to safe
//! defaults.

/// Parsed frontmatter fields we care about.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Frontmatter {
    /// `name:` slug — used as native id when present.
    pub name: Option<String>,
    /// `description:` one-line summary.
    pub description: Option<String>,
    /// `metadata.type:` — one of {user, feedback, project, reference}.
    pub mem_type: Option<String>,
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
                    "metadata" => {
                        // Either `metadata: {...}` inline (skipped — too rare to
                        // parse without a real YAML lib) or a nested block.
                        in_metadata = v.is_empty();
                    }
                    _ => {}
                }
            }
        } else if in_metadata {
            // Indented metadata.* lines.
            if let Some((k, v)) = split_key_value(line.trim_start()) {
                if k == "type" {
                    out.mem_type = Some(v);
                }
            }
        }
    }
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
}
