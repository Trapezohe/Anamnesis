//! Filesystem scanner for OpenViking (volcengine/OpenViking).
//!
//! OpenViking organizes context as a virtual filesystem (VikingFS) where every
//! URI maps onto a physical AGFS path:
//!
//! ```text
//! viking://{scope}/{path}  →  {workspace}/local/{account_id}/{scope}/{path}
//! ```
//!
//! The `workspace` is defined in `ov.conf` (`storage.workspace`, default
//! `./data` — in the Docker reference setup that's `~/.openviking/data`).
//!
//! Per VikingFS spec the top-level public scopes are:
//!   - `resources`  — user-added knowledge bases (Reference / User)
//!   - `user`       — per-user memory under `<uid>/memories/...` (Preference/Fact/Episode / User)
//!   - `agent`      — per-agent memory + skills + instructions (Episode/Reference/Skill / Project)
//!   - `session`    — per-session conversation logs (Episode / Session)
//!
//! Internal scopes (`temp`, `queue`, `_system`, `tasks`) and binary mechanics
//! (vector indices, lockfiles, checkpoints) are **skipped**.
//!
//! Per-context the file layout follows the L0/L1/L2 model:
//!   - `.abstract.md`  — L0 (~100 tokens, vector-search summary)
//!   - `.overview.md`  — L1 (~2k tokens, navigation guide)
//!   - other `*.md`    — L2 (full detail)
//!   - `messages.jsonl` (in sessions only) — one JSON message per line
//!
//! Per §-1.2.2 the adapter is read-only.

use std::fs;
use std::path::{Path, PathBuf};

/// Recursion depth cap so a misconfigured root (e.g. `$HOME`) doesn't trigger
/// a full-disk walk. AGFS hierarchy goes at most ~6 levels deep
/// (`workspace/local/{account}/{scope}/.../<file>`); 10 leaves plenty of room.
const MAX_DEPTH: usize = 10;

/// Coarse classification of one classified record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenVikingScope {
    /// `resources/...`
    Resource,
    /// `user/<uid>/memories/profile.md`
    UserProfile,
    /// `user/<uid>/memories/preferences/...`
    UserPreference,
    /// `user/<uid>/memories/entities/...`
    UserEntity,
    /// `user/<uid>/memories/events/...`
    UserEvent,
    /// `agent/<aid>/memories/cases/...`
    AgentCase,
    /// `agent/<aid>/memories/patterns/...`
    AgentPattern,
    /// `agent/<aid>/memories/tools/...`
    AgentTool,
    /// `agent/<aid>/memories/skills/...` (skill *usage* memory, not the skill def)
    AgentSkillMemory,
    /// `agent/<aid>/skills/<name>/...` (the skill definition itself)
    AgentSkillDef,
    /// `agent/<aid>/instructions/...`
    AgentInstruction,
    /// `session/<sid>/.abstract.md` / `.overview.md` — session summaries.
    SessionSummary,
    /// `session/<sid>/messages.jsonl` — per-line conversation message.
    SessionMessage,
}

/// One file-level OpenViking record.
#[derive(Debug, Clone)]
pub struct OpenVikingFileRecord {
    /// Absolute file path on disk.
    pub path: PathBuf,
    /// Reconstructed `viking://...` URI (best-effort — falls back to the
    /// path if the workspace prefix can't be peeled off).
    pub viking_uri: Option<String>,
    /// Coarse classification from the path.
    pub scope: OpenVikingScope,
    /// L0/L1/L2 layer (`"L0"` for `.abstract.md`, `"L1"` for `.overview.md`,
    /// `"L2"` for everything else under the same context).
    pub layer: Option<&'static str>,
    /// File body.
    pub content: String,
    /// File mtime in unix seconds.
    pub mtime_unix: Option<i64>,
}

/// One per-line OpenViking session message (parsed from `messages.jsonl`).
#[derive(Debug, Clone)]
pub struct OpenVikingMessage {
    /// Source `messages.jsonl` path.
    pub source_path: PathBuf,
    /// 0-based line number within the source file.
    pub line_no: usize,
    /// Session id (path-derived).
    pub session_id: String,
    /// Raw JSON line (`{id, role, parts, created_at}`).
    pub raw_json: String,
    /// File mtime of the parent JSONL file.
    pub mtime_unix: Option<i64>,
}

/// Aggregate scan output.
#[derive(Debug, Default)]
pub struct OpenVikingScan {
    /// All file-shaped records (resources, memories, summaries, etc.).
    pub files: Vec<OpenVikingFileRecord>,
    /// Per-line session messages.
    pub messages: Vec<OpenVikingMessage>,
}

impl OpenVikingScan {
    /// Total record count.
    pub fn total(&self) -> usize {
        self.files.len() + self.messages.len()
    }
}

/// Walk `workspace_root` (typically `~/.openviking/data/`) and classify every
/// AGFS file under it.
pub fn scan_openviking(workspace_root: &Path) -> OpenVikingScan {
    let mut scan = OpenVikingScan::default();
    if !workspace_root.is_dir() {
        return scan;
    }
    for entry in walkdir::WalkDir::new(workspace_root)
        .max_depth(MAX_DEPTH)
        .into_iter()
        .filter_entry(|e| !is_internal_dir(e.file_name()))
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        classify_and_collect(workspace_root, path, &mut scan);
    }
    scan
}

fn is_internal_dir(name: &std::ffi::OsStr) -> bool {
    matches!(
        name.to_str().unwrap_or(""),
        "_system"
            | "tasks"
            | "history"
            | "checkpoints"
            | "summaries"
            | "scripts"
            | "queue"
            | "temp"
    )
}

fn classify_and_collect(workspace_root: &Path, path: &Path, scan: &mut OpenVikingScan) {
    // Strip the workspace prefix → relative `local/<account>/<scope>/...`.
    let rel = match path.strip_prefix(workspace_root) {
        Ok(p) => p,
        Err(_) => return,
    };
    let parts: Vec<&str> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    // Expect at least: local / <account_id> / <scope> / ... / <file>
    if parts.len() < 4 || parts[0] != "local" {
        return;
    }
    let account_id = parts[1];
    let scope = parts[2];
    let tail: &[&str] = &parts[3..];
    let file_name = match tail.last() {
        Some(n) => *n,
        None => return,
    };
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    // Skip lockfiles and obvious non-content side-files.
    if file_name == ".relations.json" || file_name == ".meta.json" || file_name == ".path.ovlock" {
        return;
    }

    let mtime_unix = file_mtime_unix(path);
    let viking_uri = Some(build_viking_uri(account_id, scope, tail));

    match scope {
        "resources" => {
            if ext != "md" {
                return;
            }
            if let Some(content) = read_text(path) {
                scan.files.push(OpenVikingFileRecord {
                    path: path.to_path_buf(),
                    viking_uri,
                    scope: OpenVikingScope::Resource,
                    layer: Some(layer_for_filename(file_name)),
                    content,
                    mtime_unix,
                });
            }
        }
        "user" => classify_user(path, &ext, tail, viking_uri, mtime_unix, scan),
        "agent" => classify_agent(path, &ext, tail, viking_uri, mtime_unix, scan),
        "session" => classify_session(path, &ext, tail, viking_uri, account_id, mtime_unix, scan),
        // `temp`, `queue`, `_system`, `tasks` and unknown scopes are skipped.
        _ => {}
    }
}

fn classify_user(
    path: &Path,
    ext: &str,
    tail: &[&str],
    viking_uri: Option<String>,
    mtime_unix: Option<i64>,
    scan: &mut OpenVikingScan,
) {
    // Expect `<uid>/memories/<bucket>/<file>` or `<uid>/memories/profile.md`.
    if ext != "md" {
        return;
    }
    if tail.len() < 2 {
        return;
    }
    let after_uid = &tail[1..]; // strip `<uid>`
    let file_name = *tail.last().unwrap();
    if after_uid.first() != Some(&"memories") {
        return;
    }
    let bucket = after_uid.get(1).copied().unwrap_or("");
    let scope = match (bucket, file_name) {
        ("profile.md", _) | (_, "profile.md") if after_uid.len() <= 3 => {
            OpenVikingScope::UserProfile
        }
        ("preferences", _) => OpenVikingScope::UserPreference,
        ("entities", _) => OpenVikingScope::UserEntity,
        ("events", _) => OpenVikingScope::UserEvent,
        // .abstract.md / .overview.md directly under memories/ — treat as preference-y context.
        ("", _) | (_, ".abstract.md") | (_, ".overview.md") => OpenVikingScope::UserProfile,
        _ => return,
    };
    if let Some(content) = read_text(path) {
        scan.files.push(OpenVikingFileRecord {
            path: path.to_path_buf(),
            viking_uri,
            scope,
            layer: Some(layer_for_filename(file_name)),
            content,
            mtime_unix,
        });
    }
}

fn classify_agent(
    path: &Path,
    ext: &str,
    tail: &[&str],
    viking_uri: Option<String>,
    mtime_unix: Option<i64>,
    scan: &mut OpenVikingScan,
) {
    // tail: `<aid>/...`
    if tail.len() < 2 {
        return;
    }
    let after_aid = &tail[1..];
    let area = match after_aid.first() {
        Some(a) => *a,
        None => return,
    };
    let file_name = *tail.last().unwrap();

    let scope = match area {
        "memories" => {
            let bucket = after_aid.get(1).copied().unwrap_or("");
            match bucket {
                "cases" => OpenVikingScope::AgentCase,
                "patterns" => OpenVikingScope::AgentPattern,
                "tools" => OpenVikingScope::AgentTool,
                "skills" => OpenVikingScope::AgentSkillMemory,
                _ => return,
            }
        }
        "skills" => OpenVikingScope::AgentSkillDef,
        "instructions" => OpenVikingScope::AgentInstruction,
        // `workspaces/` and others: skip for now.
        _ => return,
    };

    if ext != "md" {
        return;
    }
    if let Some(content) = read_text(path) {
        scan.files.push(OpenVikingFileRecord {
            path: path.to_path_buf(),
            viking_uri,
            scope,
            layer: Some(layer_for_filename(file_name)),
            content,
            mtime_unix,
        });
    }
}

fn classify_session(
    path: &Path,
    ext: &str,
    tail: &[&str],
    viking_uri: Option<String>,
    _account_id: &str,
    mtime_unix: Option<i64>,
    scan: &mut OpenVikingScan,
) {
    // tail: `<sid>/.abstract.md` | `<sid>/.overview.md` | `<sid>/messages.jsonl`
    if tail.is_empty() {
        return;
    }
    let session_id = tail[0].to_string();
    let file_name = *tail.last().unwrap();

    if ext == "jsonl" && file_name == "messages.jsonl" {
        if let Some(body) = read_text(path) {
            for (line_no, raw) in body.lines().enumerate() {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    continue;
                }
                scan.messages.push(OpenVikingMessage {
                    source_path: path.to_path_buf(),
                    line_no,
                    session_id: session_id.clone(),
                    raw_json: trimmed.to_string(),
                    mtime_unix,
                });
            }
        }
        return;
    }

    if ext == "md" && (file_name == ".abstract.md" || file_name == ".overview.md") {
        if let Some(content) = read_text(path) {
            scan.files.push(OpenVikingFileRecord {
                path: path.to_path_buf(),
                viking_uri,
                scope: OpenVikingScope::SessionSummary,
                layer: Some(layer_for_filename(file_name)),
                content,
                mtime_unix,
            });
        }
    }
}

fn layer_for_filename(name: &str) -> &'static str {
    match name {
        ".abstract.md" => "L0",
        ".overview.md" => "L1",
        _ => "L2",
    }
}

fn build_viking_uri(_account_id: &str, scope: &str, tail: &[&str]) -> String {
    // Per VikingFS spec, public-facing URIs are `viking://{scope}/{path}`
    // (the account_id is implicit in the request context). Tail already
    // excludes `local/<account_id>/<scope>`.
    let path = tail.join("/");
    if path.is_empty() {
        format!("viking://{scope}")
    } else {
        format!("viking://{scope}/{path}")
    }
}

fn read_text(p: &Path) -> Option<String> {
    match fs::read_to_string(p) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!(
                path = %p.display(),
                error = %e,
                "openviking scanner: skipping unreadable file"
            );
            None
        }
    }
}

fn file_mtime_unix(p: &Path) -> Option<i64> {
    fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static OV_SCAN_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = OV_SCAN_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p =
            std::env::temp_dir().join(format!("ov-scan-{n}-{pid}-{seq}", pid = std::process::id()));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed_minimal(workspace: &Path) {
        let acct = workspace.join("local/acct-1");
        // Resource
        fs::create_dir_all(acct.join("resources/docs/auth")).unwrap();
        fs::write(acct.join("resources/docs/auth/.abstract.md"), "abs").unwrap();
        fs::write(acct.join("resources/docs/auth/.overview.md"), "over").unwrap();
        fs::write(acct.join("resources/docs/auth/oauth.md"), "oauth body").unwrap();
        fs::write(acct.join("resources/docs/auth/.relations.json"), "{}").unwrap();

        // User memories
        fs::create_dir_all(acct.join("user/u-1/memories/preferences")).unwrap();
        fs::create_dir_all(acct.join("user/u-1/memories/entities")).unwrap();
        fs::create_dir_all(acct.join("user/u-1/memories/events")).unwrap();
        fs::write(acct.join("user/u-1/memories/profile.md"), "profile").unwrap();
        fs::write(
            acct.join("user/u-1/memories/preferences/coding.md"),
            "uses rust",
        )
        .unwrap();
        fs::write(
            acct.join("user/u-1/memories/entities/alice.md"),
            "alice is pm",
        )
        .unwrap();
        fs::write(
            acct.join("user/u-1/memories/events/2026-05-01.md"),
            "shipped v1",
        )
        .unwrap();

        // Agent memories + skills + instructions
        fs::create_dir_all(acct.join("agent/a-1/memories/cases")).unwrap();
        fs::create_dir_all(acct.join("agent/a-1/memories/patterns")).unwrap();
        fs::create_dir_all(acct.join("agent/a-1/skills/search-web")).unwrap();
        fs::create_dir_all(acct.join("agent/a-1/instructions")).unwrap();
        fs::write(
            acct.join("agent/a-1/memories/cases/c-1.md"),
            "fixed auth bug",
        )
        .unwrap();
        fs::write(
            acct.join("agent/a-1/memories/patterns/p-1.md"),
            "retry-on-5xx",
        )
        .unwrap();
        fs::write(
            acct.join("agent/a-1/skills/search-web/SKILL.md"),
            "search overview",
        )
        .unwrap();
        fs::write(
            acct.join("agent/a-1/skills/search-web/.abstract.md"),
            "search abs",
        )
        .unwrap();
        fs::write(acct.join("agent/a-1/instructions/system.md"), "be helpful").unwrap();

        // Session
        fs::create_dir_all(acct.join("session/s-1")).unwrap();
        fs::write(acct.join("session/s-1/.abstract.md"), "sess abs").unwrap();
        fs::write(acct.join("session/s-1/.overview.md"), "sess over").unwrap();
        fs::write(acct.join("session/s-1/.meta.json"), "{}").unwrap();
        fs::write(
            acct.join("session/s-1/messages.jsonl"),
            "{\"id\":\"m1\",\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"text\":\"hi\"}]}\n{\"id\":\"m2\",\"role\":\"assistant\",\"parts\":[{\"type\":\"text\",\"text\":\"hello\"}]}\n",
        )
        .unwrap();

        // Internal — should be skipped by filter_entry.
        fs::create_dir_all(acct.join("session/s-1/checkpoints")).unwrap();
        fs::write(acct.join("session/s-1/checkpoints/c1.json"), "{}").unwrap();
        fs::create_dir_all(acct.join("_system")).unwrap();
        fs::write(acct.join("_system/state.json"), "{}").unwrap();
    }

    #[test]
    fn empty_workspace_yields_empty_scan() {
        let dir = tmp();
        let s = scan_openviking(&dir);
        assert_eq!(s.total(), 0);
    }

    #[test]
    fn missing_workspace_returns_empty_not_error() {
        let s = scan_openviking(Path::new("/tmp/never-here-ov-xyz"));
        assert_eq!(s.total(), 0);
    }

    #[test]
    fn classifies_all_known_paths() {
        let dir = tmp();
        seed_minimal(&dir);
        let s = scan_openviking(&dir);

        let count = |sc: OpenVikingScope| s.files.iter().filter(|f| f.scope == sc).count();
        // Resource: abstract + overview + L2 = 3
        assert_eq!(count(OpenVikingScope::Resource), 3);
        // User profile: 1
        assert_eq!(count(OpenVikingScope::UserProfile), 1);
        // Pref/Entity/Event
        assert_eq!(count(OpenVikingScope::UserPreference), 1);
        assert_eq!(count(OpenVikingScope::UserEntity), 1);
        assert_eq!(count(OpenVikingScope::UserEvent), 1);
        // Agent
        assert_eq!(count(OpenVikingScope::AgentCase), 1);
        assert_eq!(count(OpenVikingScope::AgentPattern), 1);
        assert_eq!(count(OpenVikingScope::AgentInstruction), 1);
        // Skill def: SKILL.md + .abstract.md = 2
        assert_eq!(count(OpenVikingScope::AgentSkillDef), 2);
        // Session summary: .abstract.md + .overview.md = 2
        assert_eq!(count(OpenVikingScope::SessionSummary), 2);
        // Session messages — 2 lines.
        assert_eq!(s.messages.len(), 2);
    }

    #[test]
    fn internal_dirs_are_skipped() {
        let dir = tmp();
        seed_minimal(&dir);
        let s = scan_openviking(&dir);
        // Nothing under _system/, checkpoints/ should appear.
        assert!(!s
            .files
            .iter()
            .any(|f| f.path.to_string_lossy().contains("_system")));
        assert!(!s
            .files
            .iter()
            .any(|f| f.path.to_string_lossy().contains("checkpoints")));
    }

    #[test]
    fn layer_assignment_picks_l0_l1_l2() {
        let dir = tmp();
        seed_minimal(&dir);
        let s = scan_openviking(&dir);

        let resource_layers: Vec<_> = s
            .files
            .iter()
            .filter(|f| f.scope == OpenVikingScope::Resource)
            .map(|f| f.layer.unwrap_or(""))
            .collect();
        assert!(resource_layers.contains(&"L0"));
        assert!(resource_layers.contains(&"L1"));
        assert!(resource_layers.contains(&"L2"));
    }

    #[test]
    fn empty_jsonl_lines_are_skipped() {
        let dir = tmp();
        let acct = dir.join("local/acct/session/s-1");
        fs::create_dir_all(&acct).unwrap();
        fs::write(
            acct.join("messages.jsonl"),
            "\n{\"id\":\"a\",\"role\":\"user\"}\n  \n{\"id\":\"b\",\"role\":\"assistant\"}\n",
        )
        .unwrap();
        let s = scan_openviking(&dir);
        assert_eq!(s.messages.len(), 2);
    }

    #[test]
    fn viking_uri_is_reconstructed() {
        let dir = tmp();
        let acct = dir.join("local/acct/resources/docs");
        fs::create_dir_all(&acct).unwrap();
        fs::write(acct.join("x.md"), "x").unwrap();
        let s = scan_openviking(&dir);
        assert_eq!(s.files.len(), 1);
        assert_eq!(
            s.files[0].viking_uri.as_deref(),
            Some("viking://resources/docs/x.md")
        );
    }
}
