//! Filesystem scanner for a ghast install.
//!
//! ghast is an Electron AI companion app. It has two on-disk areas
//! the user might point this adapter at:
//!
//!   1. **Source repo** (the recommended target for ghast-end users
//!      who clone the project to customize prompts/skills):
//!      `<repo>/prompts/<role>/*.md`              → Kind::Reference
//!      `<repo>/resources/bundled-skills/<skill>/SKILL.md` → Kind::Skill
//!      `<repo>/resources/bundled-skills/<skill>/REFERENCES.md` etc.
//!
//!   2. **User profile** at
//!      `~/Library/Application Support/ghast/profiles/<id>/ghast.db`
//!      — encrypted at rest (sqlite3-multiple-ciphers); ghast hasn't
//!      shipped a key-export contract yet, so the adapter detects
//!      this file and surfaces a clear `note` rather than trying to
//!      decrypt blind. When ghast adds an export path (plain JSONL
//!      or MCP `resources/list`), a future round wires that in.
//!
//! Per §-1.2.2 the adapter is read-only.

use std::fs;
use std::path::{Path, PathBuf};

/// One prompt file recovered from `prompts/<role>/*.md`.
#[derive(Debug, Clone)]
pub struct GhastPromptFile {
    /// Role directory name (`"coding"`, `"computer-use"`, …).
    pub role: String,
    /// File name minus `.md`.
    pub name: String,
    /// Absolute path on disk.
    pub path: PathBuf,
    /// File body.
    pub content: String,
    /// File mtime in unix seconds.
    pub mtime_unix: Option<i64>,
}

/// One skill recovered from `resources/bundled-skills/<name>/SKILL.md`
/// (and optional sibling files like `REFERENCES.md`).
#[derive(Debug, Clone)]
pub struct GhastSkill {
    /// Skill directory name.
    pub name: String,
    /// Logical file kind within the skill: `"SKILL.md"`,
    /// `"REFERENCES.md"`, `"NOTES.md"`, etc.
    pub file_kind: String,
    /// Absolute path.
    pub path: PathBuf,
    /// File body.
    pub content: String,
    /// File mtime in unix seconds.
    pub mtime_unix: Option<i64>,
}

/// Aggregate scan result.
#[derive(Debug, Default)]
pub struct GhastScan {
    /// Prompt files under `prompts/<role>/`.
    pub prompts: Vec<GhastPromptFile>,
    /// Bundled-skill files under `resources/bundled-skills/`.
    pub skills: Vec<GhastSkill>,
    /// Whether the user-profile `ghast.db` exists but is encrypted —
    /// surfaced through the adapter's `health()` and as a tracing
    /// warning during scan so users know there's MORE data to migrate
    /// pending a key-export path.
    pub encrypted_profile_db: Option<PathBuf>,
}

impl GhastScan {
    /// Total raw record count this scan would yield.
    pub fn total(&self) -> usize {
        self.prompts.len() + self.skills.len()
    }
}

/// Walk `root` and produce a `GhastScan`.
///
/// `root` may be either the source-repo root (containing `prompts/`
/// and `resources/bundled-skills/`) or the user-data dir
/// (`~/Library/Application Support/ghast/`); the scanner probes for
/// each path independently and silently elides what isn't there.
pub fn scan_ghast(root: &Path) -> GhastScan {
    let mut scan = GhastScan::default();
    scan.prompts.extend(read_prompts(&root.join("prompts")));
    scan.skills
        .extend(read_skills(&root.join("resources/bundled-skills")));

    // Detect encrypted profile DB. We look both in this root (in case
    // the user pointed at `~/Library/Application Support/ghast/`
    // directly) and in the standard profile location.
    if let Some(db) = find_encrypted_profile_db(root) {
        scan.encrypted_profile_db = Some(db);
    }

    scan
}

fn read_prompts(prompts_dir: &Path) -> Vec<GhastPromptFile> {
    let mut out = Vec::new();
    let Ok(roles) = fs::read_dir(prompts_dir) else {
        return out;
    };
    for role_entry in roles.flatten() {
        let role_dir = role_entry.path();
        if !role_dir.is_dir() {
            continue;
        }
        let role = role_entry
            .file_name()
            .to_str()
            .unwrap_or("unknown")
            .to_string();
        let Ok(files) = fs::read_dir(&role_dir) else {
            continue;
        };
        for f in files.flatten() {
            let p = f.path();
            if !p.is_file() {
                continue;
            }
            if p.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let content = match fs::read_to_string(&p) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        path = %p.display(),
                        error = %e,
                        "ghast scanner: skipping unreadable prompt file"
                    );
                    continue;
                }
            };
            let name = p
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            out.push(GhastPromptFile {
                role: role.clone(),
                name,
                path: p.clone(),
                content,
                mtime_unix: file_mtime_unix(&p),
            });
        }
    }
    out
}

fn read_skills(skills_dir: &Path) -> Vec<GhastSkill> {
    let mut out = Vec::new();
    let Ok(skills) = fs::read_dir(skills_dir) else {
        return out;
    };
    for skill_entry in skills.flatten() {
        let skill_dir = skill_entry.path();
        if !skill_dir.is_dir() {
            continue;
        }
        let name = skill_entry
            .file_name()
            .to_str()
            .unwrap_or("unknown")
            .to_string();
        let Ok(files) = fs::read_dir(&skill_dir) else {
            continue;
        };
        for f in files.flatten() {
            let p = f.path();
            if !p.is_file() {
                continue;
            }
            if p.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let file_kind = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            let content = match fs::read_to_string(&p) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        path = %p.display(),
                        error = %e,
                        "ghast scanner: skipping unreadable skill file"
                    );
                    continue;
                }
            };
            out.push(GhastSkill {
                name: name.clone(),
                file_kind,
                path: p.clone(),
                content,
                mtime_unix: file_mtime_unix(&p),
            });
        }
    }
    out
}

/// Find an encrypted `ghast.db` under `<root>/profiles/<id>/`.
/// Returns the first one found whose magic bytes are NOT
/// `"SQLite format 3"`. The scanner only looks under the given `root`
/// — discovery of the canonical `~/Library/Application Support/ghast`
/// location is the detector's job (so tests get hermetic results).
fn find_encrypted_profile_db(root: &Path) -> Option<PathBuf> {
    let profiles_dir = root.join("profiles");
    if profiles_dir.is_dir() {
        return scan_profiles_dir(&profiles_dir);
    }
    None
}

fn scan_profiles_dir(profiles: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(profiles).ok()?;
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let candidate = dir.join("ghast.db");
        if candidate.is_file() && is_likely_encrypted(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// SQLite plaintext files start with the magic string
/// `"SQLite format 3\0"` (16 bytes). Anything else with `.db`
/// extension we treat as encrypted / opaque.
fn is_likely_encrypted(path: &Path) -> bool {
    let Ok(mut bytes) = fs::read(path) else {
        return false;
    };
    bytes.truncate(16);
    bytes != b"SQLite format 3\0"
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

    static GHAST_SCAN_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = GHAST_SCAN_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "ghast-scanner-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn empty_root_yields_empty_scan() {
        let dir = tmp();
        let s = scan_ghast(&dir);
        assert_eq!(s.total(), 0);
        assert!(s.encrypted_profile_db.is_none());
    }

    #[test]
    fn picks_up_prompts_grouped_by_role() {
        let dir = tmp();
        fs::create_dir_all(dir.join("prompts/coding")).unwrap();
        fs::create_dir_all(dir.join("prompts/computer-use")).unwrap();
        fs::write(
            dir.join("prompts/coding/default.md"),
            "default coding prompt",
        )
        .unwrap();
        fs::write(
            dir.join("prompts/coding/anthropic.md"),
            "anthropic coding prompt",
        )
        .unwrap();
        fs::write(
            dir.join("prompts/computer-use/default.md"),
            "default cu prompt",
        )
        .unwrap();
        // Non-.md ignored
        fs::write(dir.join("prompts/coding/notes.txt"), "ignore").unwrap();
        let s = scan_ghast(&dir);
        assert_eq!(s.prompts.len(), 3);
        let roles: std::collections::HashSet<&str> =
            s.prompts.iter().map(|p| p.role.as_str()).collect();
        assert!(roles.contains("coding"));
        assert!(roles.contains("computer-use"));
    }

    #[test]
    fn picks_up_bundled_skills_files() {
        let dir = tmp();
        let skill = dir.join("resources/bundled-skills/memory-management");
        fs::create_dir_all(&skill).unwrap();
        fs::write(skill.join("SKILL.md"), "skill body").unwrap();
        fs::write(skill.join("REFERENCES.md"), "refs body").unwrap();
        let s = scan_ghast(&dir);
        assert_eq!(s.skills.len(), 2);
        let names: std::collections::HashSet<&str> =
            s.skills.iter().map(|sk| sk.name.as_str()).collect();
        assert_eq!(names.len(), 1);
        assert!(names.contains("memory-management"));
        let kinds: std::collections::HashSet<&str> =
            s.skills.iter().map(|sk| sk.file_kind.as_str()).collect();
        assert!(kinds.contains("SKILL.md"));
        assert!(kinds.contains("REFERENCES.md"));
    }

    #[test]
    fn flags_encrypted_profile_db_when_present_under_root() {
        let dir = tmp();
        let profile = dir.join("profiles/ghast-id");
        fs::create_dir_all(&profile).unwrap();
        // Random non-sqlite bytes
        fs::write(profile.join("ghast.db"), b"\xae\x56\x14\xe5encrypted body").unwrap();
        let s = scan_ghast(&dir);
        assert!(s.encrypted_profile_db.is_some());
    }

    #[test]
    fn ignores_plaintext_sqlite_as_encrypted() {
        let dir = tmp();
        let profile = dir.join("profiles/ghast-id");
        fs::create_dir_all(&profile).unwrap();
        // Plain SQLite magic — should NOT be flagged as encrypted.
        let mut buf = b"SQLite format 3\0".to_vec();
        buf.extend(std::iter::repeat_n(0u8, 100));
        fs::write(profile.join("ghast.db"), buf).unwrap();
        let s = scan_ghast(&dir);
        assert!(s.encrypted_profile_db.is_none());
    }
}
