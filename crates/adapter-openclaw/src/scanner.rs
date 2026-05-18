//! Filesystem scanner for an OpenClaw install.
//!
//! OpenClaw stores everything under `~/.openclaw/` by default
//! (configurable via `agents.defaults.workspace` in `openclaw.json`):
//!
//! ```text
//! ~/.openclaw/
//! ├── openclaw.json                       — top-level config
//! └── workspace/
//!     ├── AGENTS.md                       — agents config preamble
//!     ├── SOUL.md                         — agent persona / system role
//!     ├── TOOLS.md                        — tool descriptions
//!     ├── skills/
//!     │   └── <skill-name>/SKILL.md       — one skill per directory
//!     └── sessions/                       — optional, format varies
//!         └── *.{json,jsonl}              — session log files
//! ```
//!
//! Per §-1.2.2 the scanner is read-only.

use std::fs;
use std::path::{Path, PathBuf};

/// Top-level injected markdown file (AGENTS / SOUL / TOOLS / openclaw.json).
#[derive(Debug, Clone)]
pub struct OpenClawConfigFile {
    /// Logical name: `"AGENTS.md"`, `"SOUL.md"`, `"TOOLS.md"`, `"openclaw.json"`.
    pub kind: String,
    /// Absolute path.
    pub path: PathBuf,
    /// File body.
    pub content: String,
    /// File mtime in unix seconds, if metadata read succeeded.
    pub mtime_unix: Option<i64>,
}

/// One skill defined under `workspace/skills/<name>/SKILL.md`.
#[derive(Debug, Clone)]
pub struct OpenClawSkill {
    /// Directory name (= skill id).
    pub name: String,
    /// Path to the SKILL.md file.
    pub path: PathBuf,
    /// File body.
    pub content: String,
    /// File mtime in unix seconds.
    pub mtime_unix: Option<i64>,
}

/// One session file blob under `workspace/sessions/`. Format is
/// unconstrained — we treat the whole file as one Episode record so
/// the importer still captures the conversation without us having to
/// pin a per-version JSONL schema.
#[derive(Debug, Clone)]
pub struct OpenClawSessionBlob {
    /// File name (relative to sessions dir).
    pub name: String,
    /// Absolute path.
    pub path: PathBuf,
    /// File body.
    pub content: String,
    /// File mtime in unix seconds.
    pub mtime_unix: Option<i64>,
}

/// Aggregate scan result.
#[derive(Debug, Default)]
pub struct OpenClawScan {
    /// AGENTS / SOUL / TOOLS / openclaw.json bundles.
    pub configs: Vec<OpenClawConfigFile>,
    /// Skill definitions.
    pub skills: Vec<OpenClawSkill>,
    /// Session log files (`.json` / `.jsonl` under `sessions/`).
    pub sessions: Vec<OpenClawSessionBlob>,
}

impl OpenClawScan {
    /// Total raw record count this scan would yield.
    pub fn total(&self) -> usize {
        self.configs.len() + self.skills.len() + self.sessions.len()
    }
}

/// Walk `data_dir` and produce an `OpenClawScan`.
///
/// Missing files / sub-directories are silently elided — an OpenClaw
/// install can have any subset of (config, skills, sessions).
pub fn scan_openclaw_dir(data_dir: &Path) -> OpenClawScan {
    let mut scan = OpenClawScan::default();

    // Top-level `openclaw.json` lives at the data dir root.
    if let Some(cf) = read_named_file(data_dir, "openclaw.json", "openclaw.json") {
        scan.configs.push(cf);
    }

    let workspace = data_dir.join("workspace");
    for (name, kind) in [
        ("AGENTS.md", "AGENTS.md"),
        ("SOUL.md", "SOUL.md"),
        ("TOOLS.md", "TOOLS.md"),
    ] {
        if let Some(cf) = read_named_file(&workspace, name, kind) {
            scan.configs.push(cf);
        }
    }

    scan.skills.extend(read_skills(&workspace.join("skills")));
    scan.sessions
        .extend(read_sessions(&workspace.join("sessions")));

    scan
}

fn read_named_file(dir: &Path, file_name: &str, kind: &str) -> Option<OpenClawConfigFile> {
    let p = dir.join(file_name);
    if !p.is_file() {
        return None;
    }
    let content = match fs::read_to_string(&p) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                path = %p.display(),
                error = %e,
                "openclaw scanner: skipping unreadable file"
            );
            return None;
        }
    };
    let mtime_unix = file_mtime_unix(&p);
    Some(OpenClawConfigFile {
        kind: kind.into(),
        path: p,
        content,
        mtime_unix,
    })
}

fn read_skills(skills_dir: &Path) -> Vec<OpenClawSkill> {
    let mut out = Vec::new();
    let Ok(read) = fs::read_dir(skills_dir) else {
        return out;
    };
    for entry in read.flatten() {
        let dir_path = entry.path();
        if !dir_path.is_dir() {
            continue;
        }
        let skill_md = dir_path.join("SKILL.md");
        if !skill_md.is_file() {
            continue;
        }
        let content = match fs::read_to_string(&skill_md) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    path = %skill_md.display(),
                    error = %e,
                    "openclaw scanner: skipping unreadable SKILL.md"
                );
                continue;
            }
        };
        let name = entry.file_name().to_str().unwrap_or("unknown").to_string();
        out.push(OpenClawSkill {
            name,
            path: skill_md.clone(),
            content,
            mtime_unix: file_mtime_unix(&skill_md),
        });
    }
    out
}

fn read_sessions(sessions_dir: &Path) -> Vec<OpenClawSessionBlob> {
    let mut out = Vec::new();
    let Ok(read) = fs::read_dir(sessions_dir) else {
        return out;
    };
    for entry in read.flatten() {
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        let Some(ext) = p.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !matches!(ext.to_lowercase().as_str(), "json" | "jsonl" | "ndjson") {
            continue;
        }
        let content = match fs::read_to_string(&p) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    path = %p.display(),
                    error = %e,
                    "openclaw scanner: skipping unreadable session file"
                );
                continue;
            }
        };
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        out.push(OpenClawSessionBlob {
            name,
            path: p.clone(),
            content,
            mtime_unix: file_mtime_unix(&p),
        });
    }
    out
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

    fn tmp() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("openclaw-scanner-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn empty_dir_yields_empty_scan() {
        let dir = tmp();
        let s = scan_openclaw_dir(&dir);
        assert_eq!(s.total(), 0);
    }

    #[test]
    fn picks_up_config_files() {
        let dir = tmp();
        let ws = dir.join("workspace");
        fs::create_dir_all(&ws).unwrap();
        fs::write(dir.join("openclaw.json"), "{}").unwrap();
        fs::write(ws.join("AGENTS.md"), "agents config").unwrap();
        fs::write(ws.join("SOUL.md"), "system persona").unwrap();
        fs::write(ws.join("TOOLS.md"), "tool list").unwrap();
        let s = scan_openclaw_dir(&dir);
        assert_eq!(s.configs.len(), 4);
        let kinds: Vec<&str> = s.configs.iter().map(|c| c.kind.as_str()).collect();
        assert!(kinds.contains(&"openclaw.json"));
        assert!(kinds.contains(&"AGENTS.md"));
        assert!(kinds.contains(&"SOUL.md"));
        assert!(kinds.contains(&"TOOLS.md"));
    }

    #[test]
    fn picks_up_skills_one_per_directory() {
        let dir = tmp();
        let skills = dir.join("workspace/skills");
        fs::create_dir_all(skills.join("write-code")).unwrap();
        fs::create_dir_all(skills.join("read-rss")).unwrap();
        fs::write(
            skills.join("write-code/SKILL.md"),
            "---\nname: write-code\n---\nbody",
        )
        .unwrap();
        fs::write(
            skills.join("read-rss/SKILL.md"),
            "---\nname: read-rss\n---\nbody",
        )
        .unwrap();
        // A dir without SKILL.md → skipped.
        fs::create_dir_all(skills.join("empty")).unwrap();
        let s = scan_openclaw_dir(&dir);
        assert_eq!(s.skills.len(), 2);
        let names: Vec<&str> = s.skills.iter().map(|sk| sk.name.as_str()).collect();
        assert!(names.contains(&"write-code"));
        assert!(names.contains(&"read-rss"));
        assert!(!names.contains(&"empty"));
    }

    #[test]
    fn picks_up_sessions_json_and_jsonl() {
        let dir = tmp();
        let sess = dir.join("workspace/sessions");
        fs::create_dir_all(&sess).unwrap();
        fs::write(sess.join("a.json"), r#"{"events": []}"#).unwrap();
        fs::write(sess.join("b.jsonl"), "{\"k\":1}\n{\"k\":2}\n").unwrap();
        // Red-herring: .txt file should be ignored.
        fs::write(sess.join("README.txt"), "ignore me").unwrap();
        let s = scan_openclaw_dir(&dir);
        assert_eq!(s.sessions.len(), 2);
    }

    #[test]
    fn unreadable_file_is_skipped_not_fatal() {
        let dir = tmp();
        let ws = dir.join("workspace");
        fs::create_dir_all(&ws).unwrap();
        fs::write(ws.join("AGENTS.md"), "ok").unwrap();
        // Manually break SOUL.md by creating a directory with that name
        // (so `read_to_string` errors but the scan continues).
        fs::create_dir_all(ws.join("SOUL.md")).unwrap();
        let s = scan_openclaw_dir(&dir);
        // AGENTS.md still picked up; SOUL.md skipped.
        let kinds: Vec<&str> = s.configs.iter().map(|c| c.kind.as_str()).collect();
        assert!(kinds.contains(&"AGENTS.md"));
        assert!(!kinds.contains(&"SOUL.md"));
    }
}
