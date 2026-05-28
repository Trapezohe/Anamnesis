//! Filesystem walker for `~/.claude/projects/`.
//!
//! Reads ONLY directory listings and file metadata during detection;
//! content reads happen during `scan` (the import phase).
//!
//! ## JSONL recursion (PR-F, BLUEPRINT §18.4 F1)
//!
//! Modern Claude Code nests session files as
//! `<project>/<session-uuid>/subagents/*.jsonl`. We recurse `*.jsonl`
//! discovery up to `MAX_JSONL_DEPTH` levels under each project dir,
//! skipping `memory/` (handled separately) and hidden entries. Memory
//! markdown remains exactly one level deep (`<project>/memory/*.md`).

use std::path::{Path, PathBuf};

/// Maximum recursion depth for `.jsonl` discovery under a project dir.
/// Real Claude Code data nests 3 levels (`<project>/<uuid>/subagents/*.jsonl`);
/// 6 gives generous headroom without inviting accidental whole-tree walks.
const MAX_JSONL_DEPTH: usize = 6;

/// Result of scanning a single project subdirectory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectScan {
    /// Project directory (`~/.claude/projects/<encoded-path>`).
    pub project_dir: PathBuf,
    /// `*.jsonl` files (typically conversation sessions).
    pub jsonl_files: Vec<PathBuf>,
    /// `memory/*.md` files except `MEMORY.md` (index file, never imported).
    pub memory_files: Vec<PathBuf>,
}

/// Walk `projects_root` and return one `ProjectScan` per project subdir.
///
/// - Skips hidden entries and non-directories under the root.
/// - Returns an empty `Vec` if `projects_root` does not exist; the caller
///   decides whether that's an error.
/// - Never reads file content.
pub fn scan_projects_root(projects_root: &Path) -> std::io::Result<Vec<ProjectScan>> {
    if !projects_root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(projects_root)? {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type()?.is_dir() {
            continue;
        }
        if is_hidden(&path) {
            continue;
        }
        out.push(scan_project_dir(&path)?);
    }
    out.sort_by(|a, b| a.project_dir.cmp(&b.project_dir));
    Ok(out)
}

fn scan_project_dir(project_dir: &Path) -> std::io::Result<ProjectScan> {
    let mut scan = ProjectScan {
        project_dir: project_dir.to_path_buf(),
        ..Default::default()
    };
    for entry in std::fs::read_dir(project_dir)? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_file() && path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            scan.jsonl_files.push(path);
        } else if ft.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name == "memory" {
                scan.memory_files = scan_memory_dir(&path)?;
            } else if !name.is_empty() && !name.starts_with('.') {
                // Recurse for nested session files
                // (`<session-uuid>/subagents/*.jsonl` is the common shape).
                collect_jsonl_recursive(&path, 1, &mut scan.jsonl_files)?;
            }
        }
    }
    scan.jsonl_files.sort();
    scan.memory_files.sort();
    Ok(scan)
}

/// Recursively collect `*.jsonl` files under `dir`, up to
/// [`MAX_JSONL_DEPTH`] levels deep. Skips hidden entries and any
/// directory called `memory` (which is owned by the markdown branch).
///
/// Errors on intermediate entries are swallowed with a `tracing::warn!` —
/// a single unreadable subdir must not kill the whole project scan.
fn collect_jsonl_recursive(
    dir: &Path,
    depth: usize,
    out: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    if depth > MAX_JSONL_DEPTH {
        return Ok(());
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(it) => it,
        Err(e) => {
            tracing::warn!(dir = %dir.display(), error = %e, "skipping unreadable subdir");
            return Ok(());
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(dir = %dir.display(), error = %e, "skipping bad dirent");
                continue;
            }
        };
        let path = entry.path();
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ft.is_file() {
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                out.push(path);
            }
        } else if ft.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.is_empty() || name.starts_with('.') || name == "memory" {
                continue;
            }
            collect_jsonl_recursive(&path, depth + 1, out)?;
        }
    }
    Ok(())
}

fn scan_memory_dir(memory_dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(memory_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type()?.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name == "MEMORY.md" {
            continue; // index file — see BLUEPRINT §6.8 rule 3
        }
        if path.extension().and_then(|e| e.to_str()) == Some("md") {
            out.push(path);
        }
    }
    Ok(out)
}

fn is_hidden(p: &Path) -> bool {
    p.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.starts_with('.'))
        .unwrap_or(false)
}

/// Sum `(memory_files, jsonl_files)` across all projects — used by the
/// detector for a cheap record-count estimate.
pub fn count_records(scans: &[ProjectScan]) -> (u64, u64) {
    let mut mem = 0u64;
    let mut jsonl = 0u64;
    for s in scans {
        mem += s.memory_files.len() as u64;
        jsonl += s.jsonl_files.len() as u64;
    }
    (mem, jsonl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    fn tmp() -> tempdir_lite::TempDir {
        tempdir_lite::TempDir::new("anamnesis-scanner")
    }

    fn touch(p: &Path, content: &str) {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = fs::File::create(p).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn missing_root_returns_empty() {
        let scans = scan_projects_root(Path::new("/nonexistent/path/anamnesis-test")).unwrap();
        assert!(scans.is_empty());
    }

    #[test]
    fn finds_jsonl_and_memory_files() {
        let tmp = tmp();
        let root = tmp.path().join("projects");
        let proj_a = root.join("project-a");
        touch(&proj_a.join("session-1.jsonl"), "{}");
        touch(&proj_a.join("session-2.jsonl"), "{}");
        touch(
            &proj_a.join("memory").join("user_role.md"),
            "---\nname: r\n---\nx",
        );
        touch(
            &proj_a.join("memory").join("feedback_x.md"),
            "---\nname: f\n---\ny",
        );
        touch(&proj_a.join("memory").join("MEMORY.md"), "index");

        let scans = scan_projects_root(&root).unwrap();
        assert_eq!(scans.len(), 1);
        let s = &scans[0];
        assert_eq!(s.project_dir, proj_a);
        assert_eq!(s.jsonl_files.len(), 2);
        assert_eq!(s.memory_files.len(), 2, "MEMORY.md must be excluded");
        assert!(s
            .memory_files
            .iter()
            .all(|p| p.file_name().unwrap() != "MEMORY.md"));
    }

    #[test]
    fn multiple_projects_are_sorted() {
        let tmp = tmp();
        let root = tmp.path().join("projects");
        touch(&root.join("proj-z").join("s.jsonl"), "");
        touch(&root.join("proj-a").join("s.jsonl"), "");
        touch(&root.join("proj-m").join("s.jsonl"), "");
        let scans = scan_projects_root(&root).unwrap();
        assert_eq!(scans.len(), 3);
        let names: Vec<_> = scans
            .iter()
            .map(|s| {
                s.project_dir
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(names, vec!["proj-a", "proj-m", "proj-z"]);
    }

    #[test]
    fn hidden_dirs_skipped() {
        let tmp = tmp();
        let root = tmp.path().join("projects");
        touch(&root.join(".hidden").join("a.jsonl"), "");
        touch(&root.join("visible").join("a.jsonl"), "");
        let scans = scan_projects_root(&root).unwrap();
        assert_eq!(scans.len(), 1);
        assert_eq!(scans[0].project_dir.file_name().unwrap(), "visible");
    }

    #[test]
    fn non_jsonl_non_memory_files_ignored() {
        let tmp = tmp();
        let root = tmp.path().join("projects");
        touch(&root.join("p").join("a.jsonl"), "");
        touch(&root.join("p").join("ignore.txt"), "");
        touch(&root.join("p").join("ignore.log"), "");
        let scans = scan_projects_root(&root).unwrap();
        assert_eq!(scans[0].jsonl_files.len(), 1);
        assert_eq!(scans[0].memory_files.len(), 0);
    }

    #[test]
    fn nested_subagent_jsonl_is_discovered() {
        // Real Claude Code layout: <project>/<session-uuid>/subagents/*.jsonl
        // Before PR-F these were silently dropped (97.8% of jsonl data lost
        // on a real ~/.claude/projects/ in the wild).
        let tmp = tmp();
        let root = tmp.path().join("projects");
        let proj = root.join("-Users-x-y");
        touch(&proj.join("top.jsonl"), "{}");
        touch(
            &proj
                .join("8c525fd3-9ed6-4e59-b27c-ba544b76a425")
                .join("subagents")
                .join("agent-a.jsonl"),
            "{}",
        );
        touch(
            &proj
                .join("8c525fd3-9ed6-4e59-b27c-ba544b76a425")
                .join("subagents")
                .join("agent-b.jsonl"),
            "{}",
        );
        // Memory directory must still NOT be entered by jsonl recursion.
        touch(
            &proj.join("memory").join("user_role.md"),
            "---\nname: r\n---\nx",
        );
        touch(
            &proj.join("memory").join("not-jsonl.jsonl"),
            "should be ignored",
        );

        let scans = scan_projects_root(&root).unwrap();
        assert_eq!(scans.len(), 1);
        let s = &scans[0];
        assert_eq!(
            s.jsonl_files.len(),
            3,
            "1 top-level + 2 nested subagents = 3; memory/ jsonl must NOT be counted"
        );
        assert!(
            s.jsonl_files.iter().any(|p| p.ends_with("agent-a.jsonl")),
            "agent-a.jsonl from nested subagents/ must be discovered"
        );
        assert!(
            !s.jsonl_files
                .iter()
                .any(|p| p.components().any(|c| c.as_os_str() == "memory")),
            "no jsonl under memory/ should be picked up"
        );
        assert_eq!(s.memory_files.len(), 1, "memory md still found");
    }

    #[test]
    fn hidden_subdirs_are_not_recursed() {
        let tmp = tmp();
        let root = tmp.path().join("projects");
        let proj = root.join("proj");
        touch(&proj.join("a.jsonl"), "");
        touch(&proj.join(".cache").join("hidden.jsonl"), "");
        let scans = scan_projects_root(&root).unwrap();
        assert_eq!(scans.len(), 1);
        assert_eq!(scans[0].jsonl_files.len(), 1);
        assert!(scans[0].jsonl_files[0].ends_with("a.jsonl"));
    }

    #[test]
    fn recursion_depth_is_bounded() {
        // Construct a chain of MAX_JSONL_DEPTH + 2 levels, place a jsonl
        // at the deepest level, and verify it's NOT discovered.
        let tmp = tmp();
        let root = tmp.path().join("projects");
        let mut leaf = root.join("proj");
        for i in 0..(MAX_JSONL_DEPTH + 2) {
            leaf = leaf.join(format!("level-{i}"));
        }
        touch(&leaf.join("too-deep.jsonl"), "");
        let scans = scan_projects_root(&root).unwrap();
        assert!(
            scans[0]
                .jsonl_files
                .iter()
                .all(|p| !p.ends_with("too-deep.jsonl")),
            "jsonl deeper than MAX_JSONL_DEPTH must not appear"
        );
    }

    #[test]
    fn count_records_sums_all_projects() {
        let tmp = tmp();
        let root = tmp.path().join("projects");
        touch(&root.join("a").join("s.jsonl"), "");
        touch(&root.join("a").join("memory").join("m1.md"), "");
        touch(&root.join("b").join("s1.jsonl"), "");
        touch(&root.join("b").join("s2.jsonl"), "");
        touch(&root.join("b").join("memory").join("m1.md"), "");
        touch(&root.join("b").join("memory").join("m2.md"), "");
        let scans = scan_projects_root(&root).unwrap();
        let (mem, jsonl) = count_records(&scans);
        assert_eq!(mem, 3);
        assert_eq!(jsonl, 3);
    }
}

#[cfg(test)]
mod tempdir_lite {
    //! Minimal RAII tempdir helper so we don't pull a tempdir crate just
    //! for scanner tests.
    use std::path::{Path, PathBuf};

    pub struct TempDir(PathBuf);

    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    impl TempDir {
        pub fn new(prefix: &str) -> Self {
            let base = std::env::temp_dir();
            let pid = std::process::id();
            // Atomic counter avoids same-nanosecond collisions on
            // platforms with coarse timer resolution (Windows).
            let n = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = base.join(format!("{prefix}-{pid}-{n}"));
            std::fs::create_dir_all(&dir).expect("create tempdir");
            Self(dir)
        }
        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
