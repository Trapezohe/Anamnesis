//! Filesystem scanner for TencentDB Agent Memory (TDAI).
//!
//! TDAI persists a 4-tier hierarchical memory under
//! `~/.openclaw/memory-tdai/` (the project is an OpenClaw plugin):
//!
//!   L0 — raw conversation refs: `refs/*.md`
//!   L1 — atomic facts:          `*.jsonl` files (each line = one fact)
//!   L2 — scenario blocks:       plain `*.md` files (not in `refs/`,
//!                                not the persona file)
//!   L3 — user persona:          `persona.md`
//!
//! The repo doesn't pin the exact subdirectory layout, so the scanner
//! probes recursively (walkdir, capped) and classifies by:
//!
//!   - file name `persona.md` (case-insensitive)        → L3
//!   - parent dir name `refs` (case-insensitive)        → L0
//!   - extension `jsonl` / `ndjson`                     → L1 (per-line)
//!   - extension `md`                                   → L2
//!
//! Per §-1.2.2 the adapter is read-only.

use std::fs;
use std::path::{Path, PathBuf};

/// Recursion depth cap so a misconfigured root (e.g. `$HOME`) can't
/// walk every file on disk.
const MAX_DEPTH: usize = 8;

/// One L0 raw conversation reference.
#[derive(Debug, Clone)]
pub struct TdaiL0Ref {
    /// Absolute file path.
    pub path: PathBuf,
    /// File body.
    pub content: String,
    /// File mtime in unix seconds.
    pub mtime_unix: Option<i64>,
}

/// One L1 atomic-fact JSONL line.
#[derive(Debug, Clone)]
pub struct TdaiL1Fact {
    /// Source `.jsonl` file path.
    pub source_path: PathBuf,
    /// 0-based line number in the source file.
    pub line_no: usize,
    /// The full JSON-stringified line (one fact per line per TDAI convention).
    pub content: String,
    /// File mtime (the JSONL file as a whole). TDAI doesn't write a
    /// per-line timestamp by spec; we use file mtime as a proxy so
    /// `ScanOpts.since` works.
    pub mtime_unix: Option<i64>,
}

/// One L2 scenario block (plain markdown).
#[derive(Debug, Clone)]
pub struct TdaiL2Scenario {
    /// Absolute file path.
    pub path: PathBuf,
    /// File body.
    pub content: String,
    /// File mtime in unix seconds.
    pub mtime_unix: Option<i64>,
}

/// L3 user-persona document (only one expected per TDAI install but
/// we yield a Vec so future multi-persona variants don't break).
#[derive(Debug, Clone)]
pub struct TdaiL3Persona {
    /// Absolute file path.
    pub path: PathBuf,
    /// File body.
    pub content: String,
    /// File mtime in unix seconds.
    pub mtime_unix: Option<i64>,
}

/// Aggregate scan.
#[derive(Debug, Default)]
pub struct TdaiScan {
    /// L0 raw conversation refs.
    pub l0_refs: Vec<TdaiL0Ref>,
    /// L1 atomic facts (already line-split).
    pub l1_facts: Vec<TdaiL1Fact>,
    /// L2 scenario blocks.
    pub l2_scenarios: Vec<TdaiL2Scenario>,
    /// L3 personas.
    pub l3_personas: Vec<TdaiL3Persona>,
}

impl TdaiScan {
    /// Total raw record count this scan would yield.
    pub fn total(&self) -> usize {
        self.l0_refs.len() + self.l1_facts.len() + self.l2_scenarios.len() + self.l3_personas.len()
    }
}

/// Walk `data_dir` (default `~/.openclaw/memory-tdai/`) and classify
/// every relevant file into one of the four tiers.
pub fn scan_tdai(data_dir: &Path) -> TdaiScan {
    let mut scan = TdaiScan::default();
    if !data_dir.is_dir() {
        return scan;
    }
    for entry in walkdir::WalkDir::new(data_dir)
        .max_depth(MAX_DEPTH)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path().to_path_buf();
        classify_and_collect(&path, &mut scan);
    }
    scan
}

fn classify_and_collect(path: &Path, scan: &mut TdaiScan) {
    let mtime_unix = file_mtime_unix(path);
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let lower_name = file_name.to_lowercase();
    let parent_dir_name = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    // L3: persona.md (case-insensitive)
    if lower_name == "persona.md" {
        if let Some(content) = read_text(path) {
            scan.l3_personas.push(TdaiL3Persona {
                path: path.to_path_buf(),
                content,
                mtime_unix,
            });
            return;
        }
    }

    // L0: any *.md in a `refs/` directory
    if parent_dir_name == "refs" && ext == "md" {
        if let Some(content) = read_text(path) {
            scan.l0_refs.push(TdaiL0Ref {
                path: path.to_path_buf(),
                content,
                mtime_unix,
            });
            return;
        }
    }

    // L1: every *.jsonl / *.ndjson, line-split
    if matches!(ext.as_str(), "jsonl" | "ndjson") {
        if let Some(body) = read_text(path) {
            for (line_no, raw_line) in body.lines().enumerate() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                scan.l1_facts.push(TdaiL1Fact {
                    source_path: path.to_path_buf(),
                    line_no,
                    content: line.to_string(),
                    mtime_unix,
                });
            }
            return;
        }
    }

    // L2: any remaining *.md (not in `refs/`, not `persona.md`)
    if ext == "md" {
        if let Some(content) = read_text(path) {
            scan.l2_scenarios.push(TdaiL2Scenario {
                path: path.to_path_buf(),
                content,
                mtime_unix,
            });
        }
    }
}

fn read_text(p: &Path) -> Option<String> {
    match fs::read_to_string(p) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!(
                path = %p.display(),
                error = %e,
                "tdai scanner: skipping unreadable file"
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

    fn tmp() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("tdai-scanner-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn empty_dir_yields_empty_scan() {
        let dir = tmp();
        let s = scan_tdai(&dir);
        assert_eq!(s.total(), 0);
    }

    #[test]
    fn classifies_all_four_tiers_correctly() {
        let dir = tmp();
        // L3 persona at root.
        fs::write(dir.join("persona.md"), "user is a senior eng").unwrap();
        // L0 refs.
        fs::create_dir_all(dir.join("refs")).unwrap();
        fs::write(dir.join("refs/conv-001.md"), "raw conversation 1").unwrap();
        fs::write(dir.join("refs/conv-002.md"), "raw conversation 2").unwrap();
        // L1 atomic facts.
        fs::write(
            dir.join("facts.jsonl"),
            "{\"fact\":\"likes rust\"}\n{\"fact\":\"hates mocks\"}\n",
        )
        .unwrap();
        // L2 scenario.
        fs::write(dir.join("scenario-bug-triage.md"), "scenario block body").unwrap();

        let s = scan_tdai(&dir);
        assert_eq!(s.l3_personas.len(), 1);
        assert_eq!(s.l0_refs.len(), 2);
        assert_eq!(s.l1_facts.len(), 2);
        assert_eq!(s.l2_scenarios.len(), 1);
        assert_eq!(s.total(), 6);
    }

    #[test]
    fn empty_jsonl_lines_are_skipped() {
        let dir = tmp();
        fs::write(dir.join("a.jsonl"), "\n{\"k\":1}\n\n  \n{\"k\":2}\n").unwrap();
        let s = scan_tdai(&dir);
        assert_eq!(s.l1_facts.len(), 2);
    }

    #[test]
    fn persona_md_anywhere_routes_to_l3() {
        let dir = tmp();
        fs::create_dir_all(dir.join("nested/under")).unwrap();
        fs::write(dir.join("nested/under/persona.md"), "deep persona").unwrap();
        let s = scan_tdai(&dir);
        assert_eq!(s.l3_personas.len(), 1);
        assert_eq!(s.l2_scenarios.len(), 0);
    }

    #[test]
    fn refs_dir_classifies_md_as_l0() {
        let dir = tmp();
        fs::create_dir_all(dir.join("refs")).unwrap();
        fs::write(dir.join("refs/x.md"), "ref body").unwrap();
        fs::write(dir.join("not-refs.md"), "L2 body").unwrap();
        let s = scan_tdai(&dir);
        assert_eq!(s.l0_refs.len(), 1);
        assert_eq!(s.l2_scenarios.len(), 1);
    }

    #[test]
    fn missing_root_returns_empty_not_error() {
        let s = scan_tdai(Path::new("/tmp/never-here-tdai-xyz"));
        assert_eq!(s.total(), 0);
    }
}
