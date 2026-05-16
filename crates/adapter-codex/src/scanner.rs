//! Walk a Codex root directory and find session files (any `.json` or
//! `.jsonl` regular file, recursively, but capped to avoid runaway
//! traversal on misconfigured roots).

use std::path::{Path, PathBuf};

/// Cap depth so we don't recurse into unrelated subtrees (e.g. nested
/// virtualenvs) someone might point us at by accident.
const MAX_DEPTH: usize = 4;

/// Walk `root` for session files.
pub fn scan_root(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    walk(root, 0, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if depth > MAX_DEPTH {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            if is_hidden(&path) {
                continue;
            }
            walk(&path, depth + 1, out)?;
        } else if ft.is_file() {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "json" || ext == "jsonl" {
                out.push(path);
            }
        }
    }
    Ok(())
}

fn is_hidden(p: &Path) -> bool {
    p.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.starts_with('.'))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let pid = std::process::id();
        let p = std::env::temp_dir().join(format!("anamnesis-codex-scan-{pid}-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn missing_root_returns_empty() {
        let out = scan_root(Path::new("/nonexistent/path/anamnesis-codex")).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn finds_json_and_jsonl_files_recursively() {
        let dir = tmp_dir();
        fs::create_dir_all(dir.join("sessions/sub")).unwrap();
        fs::write(dir.join("sessions/a.jsonl"), "").unwrap();
        fs::write(dir.join("sessions/sub/b.json"), "").unwrap();
        fs::write(dir.join("README.md"), "").unwrap();
        let out = scan_root(&dir).unwrap();
        assert_eq!(out.len(), 2);
        let names: Vec<&str> = out
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
            .collect();
        assert!(names.contains(&"a.jsonl"));
        assert!(names.contains(&"b.json"));
    }

    #[test]
    fn hidden_dirs_skipped() {
        let dir = tmp_dir();
        fs::create_dir_all(dir.join(".cache")).unwrap();
        fs::write(dir.join(".cache/x.jsonl"), "").unwrap();
        fs::write(dir.join("visible.json"), "").unwrap();
        let out = scan_root(&dir).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].ends_with("visible.json"));
    }

    #[test]
    fn output_is_sorted() {
        let dir = tmp_dir();
        fs::write(dir.join("z.jsonl"), "").unwrap();
        fs::write(dir.join("a.jsonl"), "").unwrap();
        fs::write(dir.join("m.jsonl"), "").unwrap();
        let out = scan_root(&dir).unwrap();
        let names: Vec<&str> = out
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
            .collect();
        assert_eq!(names, vec!["a.jsonl", "m.jsonl", "z.jsonl"]);
    }
}
