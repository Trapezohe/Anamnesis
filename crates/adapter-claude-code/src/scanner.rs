//! Filesystem walker for `~/.claude/projects/`.
//!
//! Reads ONLY directory listings and file metadata during detection;
//! content reads happen during `scan` (the import phase).

use std::path::{Path, PathBuf};

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
        } else if ft.is_dir() && path.file_name().and_then(|n| n.to_str()) == Some("memory") {
            scan.memory_files = scan_memory_dir(&path)?;
        }
    }
    scan.jsonl_files.sort();
    scan.memory_files.sort();
    Ok(scan)
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
