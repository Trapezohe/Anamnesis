//! Filesystem scanner for MemOS (MemTensor/MemOS, Apache-2.0).
//!
//! MemOS organizes durable memory as "MemCubes". A dumped MemCube directory
//! contains a flat layout — no subdirs — with at least these files:
//!
//! ```text
//! <mem_cube_dir>/
//! ├── config.json                  # cube config (carries memory_filename overrides)
//! ├── textual_memory.json          # textual memory items (naive/general/tree backends)
//! ├── activation_memory.pickle     # KV cache, binary — SKIPPED
//! └── parametric_memory.adapter    # LoRA weights, binary — SKIPPED
//! ```
//!
//! `textual_memory.json` is the only file we read. All textual backends
//! (naive / general / tree / simple_tree / preference) share the
//! `BaseTextMemoryConfig.memory_filename = "textual_memory.json"` default,
//! so a single probe covers every backend that this adapter cares about.
//!
//! The file contains an array of `TextualMemoryItem` objects:
//!
//! ```json
//! [
//!   {
//!     "id": "uuid",
//!     "memory": "the actual content",
//!     "metadata": {
//!       "memory_type": "LongTermMemory" | "UserMemory" | ...,
//!       "user_id": "...", "session_id": "...", "key": "...",
//!       "tags": [...], "source": "conversation" | "web" | "file" | "system",
//!       "status": "activated" | "archived" | "deleted",
//!       "created_at": "ISO8601", "updated_at": "ISO8601",
//!       ...
//!     }
//!   }
//! ]
//! ```
//!
//! Tree backends may also emit JSON wrapped under a `"nodes"` or `"items"`
//! key (with edges separately). We handle both layouts gracefully.
//!
//! Per §-1.2.2 the adapter is read-only.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Default filename MemOS writes textual memory to.
const TEXTUAL_MEMORY_FILENAME: &str = "textual_memory.json";

/// Recursion depth cap so a misconfigured root (`$HOME`, `/`) doesn't walk
/// every file on disk. A MemOS dir tree should be shallow — `.memos/<cube>/file`.
const MAX_DEPTH: usize = 6;

/// Item statuses we treat as "live" (everything else is skipped).
const LIVE_STATUSES: &[&str] = &["activated"];

/// One textual memory item we successfully extracted from a MemCube dump.
#[derive(Debug, Clone)]
pub struct MemosTextItem {
    /// MemCube directory the item came from (absolute).
    pub cube_dir: PathBuf,
    /// Item id (MemOS-supplied UUID).
    pub item_id: String,
    /// The actual memory content (`item.memory`).
    pub content: String,
    /// Lifecycle bucket (e.g. `LongTermMemory`, `UserMemory`, …).
    /// Falls back to `metadata.type` when `memory_type` is absent
    /// (naive/general backends sometimes only fill `type`).
    pub memory_type: Option<String>,
    /// `metadata.user_id`.
    pub user_id: Option<String>,
    /// `metadata.session_id`.
    pub session_id: Option<String>,
    /// `metadata.source` (one of `conversation` / `retrieved` / `web` /
    /// `file` / `system` per MemOS spec).
    pub source: Option<String>,
    /// `metadata.tags` if surfaced.
    pub tags: Vec<String>,
    /// `metadata.updated_at` (ISO-8601).
    pub updated_at: Option<String>,
    /// `metadata.created_at` (ISO-8601). MemOS only sets this on tree
    /// backends; flat backends rely on `updated_at`.
    pub created_at: Option<String>,
    /// Full raw metadata blob for callers that want more than what we
    /// promote.
    pub metadata_raw: Value,
}

/// Aggregate scan.
#[derive(Debug, Default)]
pub struct MemosScan {
    /// All textual items, across all discovered MemCubes.
    pub items: Vec<MemosTextItem>,
    /// MemCube directories we found (each contains a `textual_memory.json`).
    pub cube_dirs: Vec<PathBuf>,
    /// Diagnostic note for files that exist but fail to parse — surfaced
    /// in `health().detail` and detector notes.
    pub parse_errors: Vec<String>,
}

impl MemosScan {
    /// Total live records this scan would yield.
    pub fn total(&self) -> usize {
        self.items.len()
    }
}

/// Walk `memos_root` and pull every textual memory item from every
/// MemCube under it.
///
/// `memos_root` typically points at `~/.memos/` or a directory the user
/// dumped a MemCube into. For each subdirectory that contains a
/// `textual_memory.json`, we parse it and emit one `MemosTextItem` per
/// stored item.
pub fn scan_memos(memos_root: &Path) -> MemosScan {
    let mut scan = MemosScan::default();
    if !memos_root.is_dir() {
        return scan;
    }
    for entry in walkdir::WalkDir::new(memos_root)
        .max_depth(MAX_DEPTH)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.file_name().and_then(|s| s.to_str()) != Some(TEXTUAL_MEMORY_FILENAME) {
            continue;
        }
        let cube_dir = match path.parent() {
            Some(p) => p.to_path_buf(),
            None => continue,
        };
        scan.cube_dirs.push(cube_dir.clone());
        match read_textual_memory_file(path, &cube_dir) {
            Ok(items) => scan.items.extend(items),
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "memos scanner: skipping unreadable / unparseable file"
                );
                scan.parse_errors.push(format!("{}: {}", path.display(), e));
            }
        }
    }
    scan
}

fn read_textual_memory_file(path: &Path, cube_dir: &Path) -> Result<Vec<MemosTextItem>, String> {
    let body = fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    let root: Value = serde_json::from_str(&body).map_err(|e| format!("parse: {e}"))?;
    let items = extract_items_array(&root)
        .ok_or_else(|| "no item array found (not a recognized layout)".to_string())?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if let Some(parsed) = parse_item(item, cube_dir) {
            // Drop tombstones — they're meant to be hidden.
            let status = parsed
                .metadata_raw
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("activated");
            if LIVE_STATUSES.iter().any(|s| s.eq_ignore_ascii_case(status)) {
                out.push(parsed);
            }
        }
    }
    Ok(out)
}

fn extract_items_array(root: &Value) -> Option<&Vec<Value>> {
    // Plain array layout (naive/general backends).
    if let Some(arr) = root.as_array() {
        return Some(arr);
    }
    // Tree backends sometimes wrap items under `nodes` or `items`.
    for key in ["nodes", "items", "memories", "memory_items"] {
        if let Some(arr) = root.get(key).and_then(|v| v.as_array()) {
            return Some(arr);
        }
    }
    None
}

fn parse_item(item: &Value, cube_dir: &Path) -> Option<MemosTextItem> {
    let item_id = item.get("id").and_then(|v| v.as_str())?.to_string();
    let content = item.get("memory").and_then(|v| v.as_str())?.to_string();
    if content.trim().is_empty() {
        return None;
    }
    let metadata_raw = item.get("metadata").cloned().unwrap_or(Value::Null);
    let memory_type = metadata_raw
        .get("memory_type")
        .and_then(|v| v.as_str())
        .or_else(|| metadata_raw.get("type").and_then(|v| v.as_str()))
        .map(str::to_owned);
    let user_id = metadata_raw
        .get("user_id")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let session_id = metadata_raw
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let source = metadata_raw
        .get("source")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let tags = metadata_raw
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    let updated_at = metadata_raw
        .get("updated_at")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let created_at = metadata_raw
        .get("created_at")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    Some(MemosTextItem {
        cube_dir: cube_dir.to_path_buf(),
        item_id,
        content,
        memory_type,
        user_id,
        session_id,
        source,
        tags,
        updated_at,
        created_at,
        metadata_raw,
    })
}

/// Best-effort parse of an ISO-8601 timestamp from MemOS metadata into unix
/// seconds. Handles the `datetime.now().isoformat()` shape (no TZ) by
/// treating it as UTC — matches Memori's same heuristic.
pub fn parse_memos_time(s: &str) -> Option<i64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static MEMOS_SCAN_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MEMOS_SCAN_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-memos-scan-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn missing_root_returns_empty_scan() {
        let s = scan_memos(Path::new("/tmp/never-here-memos-xyz"));
        assert_eq!(s.total(), 0);
        assert!(s.parse_errors.is_empty());
    }

    #[test]
    fn empty_root_yields_empty_scan() {
        let dir = tmp();
        let s = scan_memos(&dir);
        assert_eq!(s.total(), 0);
        assert!(s.cube_dirs.is_empty());
    }

    #[test]
    fn flat_array_layout_picks_up_items() {
        let root = tmp();
        let cube = root.join("cube-1");
        fs::create_dir_all(&cube).unwrap();
        let payload = serde_json::json!([
            {
                "id": "item-1",
                "memory": "user prefers Rust",
                "metadata": {
                    "memory_type": "UserMemory",
                    "user_id": "u-1",
                    "session_id": "s-1",
                    "source": "conversation",
                    "tags": ["coding", "preferences"],
                    "status": "activated",
                    "updated_at": "2026-05-01T10:00:00",
                }
            },
            {
                "id": "item-2",
                "memory": "deleted item",
                "metadata": {
                    "memory_type": "UserMemory",
                    "status": "deleted",
                }
            }
        ]);
        fs::write(cube.join(TEXTUAL_MEMORY_FILENAME), payload.to_string()).unwrap();

        let s = scan_memos(&root);
        // Only the activated one is kept.
        assert_eq!(s.items.len(), 1);
        assert_eq!(s.items[0].item_id, "item-1");
        assert_eq!(s.items[0].memory_type.as_deref(), Some("UserMemory"));
        assert_eq!(s.items[0].user_id.as_deref(), Some("u-1"));
        assert_eq!(s.cube_dirs.len(), 1);
    }

    #[test]
    fn tree_wrapped_layout_under_nodes_picks_up_items() {
        let root = tmp();
        let cube = root.join("cube-tree");
        fs::create_dir_all(&cube).unwrap();
        let payload = serde_json::json!({
            "nodes": [
                {
                    "id": "node-1",
                    "memory": "Paris is the capital of France",
                    "metadata": {
                        "memory_type": "LongTermMemory",
                        "source": "web",
                        "status": "activated"
                    }
                }
            ],
            "edges": []
        });
        fs::write(cube.join(TEXTUAL_MEMORY_FILENAME), payload.to_string()).unwrap();
        let s = scan_memos(&root);
        assert_eq!(s.items.len(), 1);
        assert_eq!(s.items[0].item_id, "node-1");
    }

    #[test]
    fn unparseable_json_yields_diagnostic_not_panic() {
        let root = tmp();
        let cube = root.join("bad-cube");
        fs::create_dir_all(&cube).unwrap();
        fs::write(cube.join(TEXTUAL_MEMORY_FILENAME), "{not valid").unwrap();
        let s = scan_memos(&root);
        assert_eq!(s.items.len(), 0);
        assert!(!s.parse_errors.is_empty());
    }

    #[test]
    fn multi_cube_discovery() {
        let root = tmp();
        for name in ["cube-a", "cube-b"] {
            let cube = root.join(name);
            fs::create_dir_all(&cube).unwrap();
            let payload = serde_json::json!([
                {"id": format!("{name}-1"), "memory": "x", "metadata": {"memory_type": "LongTermMemory", "status": "activated"}}
            ]);
            fs::write(cube.join(TEXTUAL_MEMORY_FILENAME), payload.to_string()).unwrap();
        }
        let s = scan_memos(&root);
        assert_eq!(s.cube_dirs.len(), 2);
        assert_eq!(s.items.len(), 2);
    }

    #[test]
    fn item_without_memory_or_id_is_skipped() {
        let root = tmp();
        let cube = root.join("cube-x");
        fs::create_dir_all(&cube).unwrap();
        let payload = serde_json::json!([
            {"id": "ok", "memory": "valid"},
            {"id": "no-mem"},
            {"memory": "no-id"},
            {"id": "empty-mem", "memory": "   "},
        ]);
        fs::write(cube.join(TEXTUAL_MEMORY_FILENAME), payload.to_string()).unwrap();
        let s = scan_memos(&root);
        assert_eq!(s.items.len(), 1);
        assert_eq!(s.items[0].item_id, "ok");
    }

    #[test]
    fn parse_memos_time_handles_isoformat() {
        // datetime.now().isoformat() => "2026-05-01T10:00:00"
        assert_eq!(
            parse_memos_time("2026-05-01T10:00:00").unwrap(),
            1_777_629_600
        );
    }

    #[test]
    fn parse_memos_time_handles_rfc3339_z() {
        assert_eq!(
            parse_memos_time("2026-05-01T10:00:00Z").unwrap(),
            1_777_629_600
        );
    }
}
