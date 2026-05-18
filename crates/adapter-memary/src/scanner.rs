//! Filesystem scanner for Memary (kingjulio8238/Memary, MIT).
//!
//! Memary stores its local-cache memory layer as five flat files inside the
//! user-supplied data directory (the streamlit example uses `data/`):
//!
//! ```text
//! <data_dir>/
//! ├── memory_stream.json           # [{entity, date}]
//! ├── entity_knowledge_store.json  # [{entity, count, date}]
//! ├── past_chat.json               # [{role, content, ...}]   (LlamaIndex ChatMessage shape)
//! ├── system_persona.txt           # plain text system persona
//! └── user_persona.txt             # plain text user persona
//! ```
//!
//! Memary's primary knowledge graph lives in Neo4j (a server, not a file)
//! — we do **not** try to read that. Per the doc the Neo4j layer is
//! best ingested via `generic-mcp` if the user runs Memary's own MCP. The
//! files above are the local truth that's actually persistent on disk and
//! they capture the most useful provenance: entities mentioned, frequency
//! counts, raw chat, and the operator's persona.
//!
//! Per §-1.2.2 the adapter is read-only.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{TimeZone, Utc};
use serde_json::Value;

/// `memory_stream.json` — list of mentioned entities.
pub const MEMORY_STREAM_FILE: &str = "memory_stream.json";
/// `entity_knowledge_store.json` — entity-frequency tallies.
pub const ENTITY_KNOWLEDGE_STORE_FILE: &str = "entity_knowledge_store.json";
/// `past_chat.json` — LlamaIndex ChatMessage history.
pub const PAST_CHAT_FILE: &str = "past_chat.json";
/// `system_persona.txt` — operator-tuned system persona.
pub const SYSTEM_PERSONA_FILE: &str = "system_persona.txt";
/// `user_persona.txt` — end-user persona.
pub const USER_PERSONA_FILE: &str = "user_persona.txt";

/// One memory-stream entry: a mentioned entity at a point in time.
#[derive(Debug, Clone)]
pub struct MemaryStreamEntry {
    /// Source path of the JSON file (absolute).
    pub source_path: PathBuf,
    /// 0-based index of the entry within the source file.
    pub index: usize,
    /// The mentioned entity (Memary just stores a string).
    pub entity: String,
    /// ISO-8601 timestamp Memary wrote.
    pub date: Option<String>,
}

/// One entity-knowledge-store entry: how often an entity has been seen.
#[derive(Debug, Clone)]
pub struct MemaryEntityTally {
    /// Source path of the JSON file.
    pub source_path: PathBuf,
    /// 0-based index of the entry within the source file.
    pub index: usize,
    /// The entity.
    pub entity: String,
    /// Observation count.
    pub count: i64,
    /// ISO-8601 timestamp of the most recent mention.
    pub date: Option<String>,
}

/// One past-chat message.
#[derive(Debug, Clone)]
pub struct MemaryChatMessage {
    /// Source path of the JSON file.
    pub source_path: PathBuf,
    /// 0-based index in the source file.
    pub index: usize,
    /// `user` / `assistant` / `system` etc.
    pub role: String,
    /// Message content.
    pub content: String,
}

/// A persona file (system or user).
#[derive(Debug, Clone)]
pub struct MemaryPersona {
    /// Absolute path to the persona file.
    pub path: PathBuf,
    /// Persona kind: `"system"` or `"user"`.
    pub persona_kind: String,
    /// File body.
    pub content: String,
    /// File mtime (unix seconds).
    pub mtime_unix: Option<i64>,
}

/// Aggregate Memary scan.
#[derive(Debug, Default)]
pub struct MemaryScan {
    /// Per-entity rows from `memory_stream.json`.
    pub stream_entries: Vec<MemaryStreamEntry>,
    /// Per-entity tallies from `entity_knowledge_store.json`.
    pub entity_tallies: Vec<MemaryEntityTally>,
    /// Per-message rows from `past_chat.json`.
    pub chat_messages: Vec<MemaryChatMessage>,
    /// Persona files (system + user).
    pub personas: Vec<MemaryPersona>,
    /// Files that exist but failed to parse — surfaced as a diagnostic.
    pub parse_errors: Vec<String>,
}

impl MemaryScan {
    /// Total live records this scan would yield.
    pub fn total(&self) -> usize {
        self.stream_entries.len()
            + self.entity_tallies.len()
            + self.chat_messages.len()
            + self.personas.len()
    }
}

/// Scan a Memary data directory (e.g. `~/.memary/data/` or the user's chosen path).
pub fn scan_memary(data_dir: &Path) -> MemaryScan {
    let mut scan = MemaryScan::default();
    if !data_dir.is_dir() {
        return scan;
    }
    read_memory_stream(data_dir, &mut scan);
    read_entity_knowledge_store(data_dir, &mut scan);
    read_past_chat(data_dir, &mut scan);
    read_persona(data_dir, SYSTEM_PERSONA_FILE, "system", &mut scan);
    read_persona(data_dir, USER_PERSONA_FILE, "user", &mut scan);
    scan
}

fn read_memory_stream(data_dir: &Path, scan: &mut MemaryScan) {
    let path = data_dir.join(MEMORY_STREAM_FILE);
    if !path.is_file() {
        return;
    }
    let body = match fs::read_to_string(&path) {
        Ok(b) => b,
        Err(e) => {
            scan.parse_errors
                .push(format!("read {}: {}", path.display(), e));
            return;
        }
    };
    let arr: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            scan.parse_errors
                .push(format!("parse {}: {}", path.display(), e));
            return;
        }
    };
    let Some(items) = arr.as_array() else {
        scan.parse_errors
            .push(format!("{}: not a JSON array", path.display()));
        return;
    };
    for (i, item) in items.iter().enumerate() {
        let entity = item.get("entity").and_then(|v| v.as_str()).unwrap_or("");
        if entity.is_empty() {
            continue;
        }
        scan.stream_entries.push(MemaryStreamEntry {
            source_path: path.clone(),
            index: i,
            entity: entity.to_string(),
            date: item.get("date").and_then(|v| v.as_str()).map(str::to_owned),
        });
    }
}

fn read_entity_knowledge_store(data_dir: &Path, scan: &mut MemaryScan) {
    let path = data_dir.join(ENTITY_KNOWLEDGE_STORE_FILE);
    if !path.is_file() {
        return;
    }
    let body = match fs::read_to_string(&path) {
        Ok(b) => b,
        Err(e) => {
            scan.parse_errors
                .push(format!("read {}: {}", path.display(), e));
            return;
        }
    };
    let arr: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            scan.parse_errors
                .push(format!("parse {}: {}", path.display(), e));
            return;
        }
    };
    let Some(items) = arr.as_array() else {
        scan.parse_errors
            .push(format!("{}: not a JSON array", path.display()));
        return;
    };
    for (i, item) in items.iter().enumerate() {
        let entity = item.get("entity").and_then(|v| v.as_str()).unwrap_or("");
        if entity.is_empty() {
            continue;
        }
        scan.entity_tallies.push(MemaryEntityTally {
            source_path: path.clone(),
            index: i,
            entity: entity.to_string(),
            count: item.get("count").and_then(|v| v.as_i64()).unwrap_or(0),
            date: item.get("date").and_then(|v| v.as_str()).map(str::to_owned),
        });
    }
}

fn read_past_chat(data_dir: &Path, scan: &mut MemaryScan) {
    let path = data_dir.join(PAST_CHAT_FILE);
    if !path.is_file() {
        return;
    }
    let body = match fs::read_to_string(&path) {
        Ok(b) => b,
        Err(e) => {
            scan.parse_errors
                .push(format!("read {}: {}", path.display(), e));
            return;
        }
    };
    let parsed: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            scan.parse_errors
                .push(format!("parse {}: {}", path.display(), e));
            return;
        }
    };
    // past_chat.json holds a LlamaIndex ChatMessage list. It may be a flat
    // array or wrapped under `"messages"`. Tolerate both.
    let items = parsed
        .as_array()
        .or_else(|| parsed.get("messages").and_then(|v| v.as_array()));
    let Some(items) = items else {
        scan.parse_errors
            .push(format!("{}: chat layout not recognized", path.display()));
        return;
    };
    for (i, item) in items.iter().enumerate() {
        let content = extract_chat_content(item);
        if content.is_empty() {
            continue;
        }
        let role = item
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        scan.chat_messages.push(MemaryChatMessage {
            source_path: path.clone(),
            index: i,
            role,
            content,
        });
    }
}

fn extract_chat_content(item: &Value) -> String {
    // LlamaIndex ChatMessage variants: `content` may be a plain string or a
    // list of `{"type": "text"/"image", "text": ...}` blocks.
    if let Some(s) = item.get("content").and_then(|v| v.as_str()) {
        return s.to_string();
    }
    if let Some(arr) = item.get("content").and_then(|v| v.as_array()) {
        let mut out = String::new();
        for blk in arr {
            if let Some(t) = blk.get("text").and_then(|v| v.as_str()) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(t);
            }
        }
        return out;
    }
    // Some Memary fixtures use `message` instead of `content`.
    if let Some(s) = item.get("message").and_then(|v| v.as_str()) {
        return s.to_string();
    }
    String::new()
}

fn read_persona(data_dir: &Path, filename: &str, kind: &str, scan: &mut MemaryScan) {
    let path = data_dir.join(filename);
    if !path.is_file() {
        return;
    }
    let body = match fs::read_to_string(&path) {
        Ok(b) => b,
        Err(e) => {
            scan.parse_errors
                .push(format!("read {}: {}", path.display(), e));
            return;
        }
    };
    if body.trim().is_empty() {
        return;
    }
    scan.personas.push(MemaryPersona {
        mtime_unix: file_mtime_unix(&path),
        path,
        persona_kind: kind.to_string(),
        content: body,
    });
}

fn file_mtime_unix(p: &Path) -> Option<i64> {
    fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
}

/// Parse a Memary date string. Memary writes
/// `datetime.now().replace(microsecond=0).isoformat()` — typically
/// `"2026-05-01T10:00:00"` (no TZ), so treat as UTC.
pub fn parse_memary_time(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.timestamp());
    }
    let candidates = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
    ];
    for fmt in candidates {
        if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return Some(ndt.and_utc().timestamp());
        }
    }
    None
}

/// Convert a unix timestamp into a `chrono::DateTime<Utc>` if valid.
pub fn unix_to_utc(t: i64) -> Option<chrono::DateTime<Utc>> {
    Utc.timestamp_opt(t, 0).single()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static MEMARY_SCAN_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MEMARY_SCAN_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-memary-scan-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed_full(dir: &Path) {
        fs::write(
            dir.join(MEMORY_STREAM_FILE),
            r#"[{"entity":"Alice","date":"2026-05-01T10:00:00"},{"entity":"Paris","date":"2026-05-01T10:05:00"}]"#,
        )
        .unwrap();
        fs::write(
            dir.join(ENTITY_KNOWLEDGE_STORE_FILE),
            r#"[{"entity":"Alice","count":3,"date":"2026-05-01T10:05:00"}]"#,
        )
        .unwrap();
        fs::write(
            dir.join(PAST_CHAT_FILE),
            r#"[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]"#,
        )
        .unwrap();
        fs::write(dir.join(SYSTEM_PERSONA_FILE), "I am a helpful agent.").unwrap();
        fs::write(dir.join(USER_PERSONA_FILE), "user is a senior eng").unwrap();
    }

    #[test]
    fn missing_data_dir_returns_empty_scan() {
        let s = scan_memary(Path::new("/tmp/never-here-memary-xyz"));
        assert_eq!(s.total(), 0);
        assert!(s.parse_errors.is_empty());
    }

    #[test]
    fn empty_data_dir_yields_empty_scan() {
        let dir = tmp();
        let s = scan_memary(&dir);
        assert_eq!(s.total(), 0);
    }

    #[test]
    fn full_layout_reads_all_record_types() {
        let dir = tmp();
        seed_full(&dir);
        let s = scan_memary(&dir);
        assert!(s.parse_errors.is_empty(), "errs: {:?}", s.parse_errors);
        assert_eq!(s.stream_entries.len(), 2);
        assert_eq!(s.entity_tallies.len(), 1);
        assert_eq!(s.chat_messages.len(), 2);
        assert_eq!(s.personas.len(), 2);
        // Sanity-check persona kinds.
        let kinds: Vec<&str> = s.personas.iter().map(|p| p.persona_kind.as_str()).collect();
        assert!(kinds.contains(&"system"));
        assert!(kinds.contains(&"user"));
    }

    #[test]
    fn entries_without_entity_are_skipped() {
        let dir = tmp();
        fs::write(
            dir.join(MEMORY_STREAM_FILE),
            r#"[{"entity":"Alice","date":"2026-05-01T10:00:00"},{"date":"2026-05-02T10:00:00"},{"entity":""}]"#,
        )
        .unwrap();
        let s = scan_memary(&dir);
        assert_eq!(s.stream_entries.len(), 1);
        assert_eq!(s.stream_entries[0].entity, "Alice");
    }

    #[test]
    fn unparseable_json_surfaces_in_parse_errors() {
        let dir = tmp();
        fs::write(dir.join(MEMORY_STREAM_FILE), "not-json").unwrap();
        let s = scan_memary(&dir);
        assert!(!s.parse_errors.is_empty());
        assert_eq!(s.stream_entries.len(), 0);
    }

    #[test]
    fn past_chat_block_content_concatenates_text_parts() {
        let dir = tmp();
        fs::write(
            dir.join(PAST_CHAT_FILE),
            r#"[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]"#,
        )
        .unwrap();
        let s = scan_memary(&dir);
        assert_eq!(s.chat_messages.len(), 1);
        assert_eq!(s.chat_messages[0].content, "a\nb");
    }

    #[test]
    fn past_chat_wrapped_under_messages_key() {
        let dir = tmp();
        fs::write(
            dir.join(PAST_CHAT_FILE),
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        let s = scan_memary(&dir);
        assert_eq!(s.chat_messages.len(), 1);
    }

    #[test]
    fn empty_persona_file_is_skipped() {
        let dir = tmp();
        fs::write(dir.join(SYSTEM_PERSONA_FILE), "   \n\n  ").unwrap();
        let s = scan_memary(&dir);
        assert_eq!(s.personas.len(), 0);
    }

    #[test]
    fn parse_memary_time_handles_isoformat() {
        let t = parse_memary_time("2026-05-01T10:00:00").unwrap();
        assert_eq!(t, 1_777_629_600);
    }
}
