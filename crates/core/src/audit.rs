//! Append-only audit log — see `BLUEPRINT.md §7`.
//!
//! Records every privileged action (import, search, export, serve) as a
//! JSON Lines record under `$DATA_DIR/audit.log`. No rotation in Phase 1;
//! ops can `tail -f` or `jq` directly.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Round 84 (PR-78f): default `limit` for `Audit::tail` when the
/// caller leaves it unset. Matches `tail -n 10` muscle memory but
/// rounded up to surface more context per call.
pub const AUDIT_TAIL_DEFAULT_LIMIT: usize = 20;
/// Round 84: hard cap on how many entries `Audit::tail` will return
/// in one call. Guards against an operator typo (`-n 1000000`) that
/// would dump the whole log into a terminal.
pub const AUDIT_TAIL_MAX_LIMIT: usize = 1000;

/// Round 84 (PR-78f): filter + limit knobs for [`Audit::tail`].
/// All fields are optional; the default produces the last
/// [`AUDIT_TAIL_DEFAULT_LIMIT`] entries unfiltered.
#[derive(Debug, Clone, Default)]
pub struct AuditTailOptions {
    /// Cap on entries returned. `None` → [`AUDIT_TAIL_DEFAULT_LIMIT`].
    /// Caller-supplied values above [`AUDIT_TAIL_MAX_LIMIT`] are
    /// clamped down to the cap.
    pub limit: Option<usize>,
    /// Optional lower bound on `entry.timestamp`. Entries strictly
    /// older than this are dropped before the limit applies.
    pub since: Option<DateTime<Utc>>,
    /// Exact-match filter on `entry.action`. `None` returns all
    /// actions.
    pub action: Option<String>,
}

/// Round 84: one row of `Audit::tail` output. Carries the 1-based
/// file line number (so an operator can correlate with a raw
/// `head` / `sed` peek) alongside the parsed entry.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditTailRow {
    /// 1-based line number in `$DATA_DIR/audit.log`. Matches the
    /// muscle memory of `tail -n N` + `head -n LINE`.
    pub line_no: usize,
    /// The parsed audit entry — same shape `Audit::record` wrote.
    pub entry: AuditEntry,
}

/// Audit log handle. Cheap to construct.
#[derive(Debug, Clone)]
pub struct Audit {
    log_path: PathBuf,
}

impl Audit {
    /// Build an audit logger that appends to `data_dir/audit.log`.
    /// Creates the parent dir lazily on the first write.
    pub fn new(data_dir: &Path) -> Self {
        Self {
            log_path: data_dir.join("audit.log"),
        }
    }

    /// Path to the log file.
    pub fn path(&self) -> &Path {
        &self.log_path
    }

    /// Append one entry. Best-effort: failures are logged at warn level
    /// but never propagated — the user's command must succeed even if
    /// the audit log can't be written.
    pub fn record(&self, entry: AuditEntry) {
        if let Err(e) = self.try_record(&entry) {
            tracing::warn!(
                error = %e,
                path = %self.log_path.display(),
                action = %entry.action,
                "audit log write failed (ignored)"
            );
        }
    }

    fn try_record(&self, entry: &AuditEntry) -> std::io::Result<()> {
        if let Some(parent) = self.log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let line = serde_json::to_string(entry).unwrap_or_else(|_| "{}".into());
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    /// Round 84 (PR-78f): read the last N audit entries from
    /// `$DATA_DIR/audit.log`, in **file order** (oldest first
    /// within the returned window), with optional
    /// `since` / `action` filters applied before the limit. This
    /// is the read counterpart to `Audit::record`.
    ///
    /// Behaviour:
    ///   * Missing log file → empty Vec, not an error. A store
    ///     that's never had a mutation has no audit yet.
    ///   * Malformed JSON line → silently skipped + tracing::warn.
    ///     A casual operator who hand-edited `audit.log` should
    ///     still get a working `audit tail`.
    ///   * `limit` clamped to `[1, AUDIT_TAIL_MAX_LIMIT]`. `None`
    ///     uses `AUDIT_TAIL_DEFAULT_LIMIT`.
    ///   * Implementation reads the whole file. v0.1.0 acceptable;
    ///     a streaming reverse reader is a follow-up once real
    ///     deployments hit multi-MB logs.
    pub fn tail(&self, opts: &AuditTailOptions) -> std::io::Result<Vec<AuditTailRow>> {
        let limit = opts
            .limit
            .unwrap_or(AUDIT_TAIL_DEFAULT_LIMIT)
            .clamp(1, AUDIT_TAIL_MAX_LIMIT);

        let file = match std::fs::File::open(&self.log_path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let reader = std::io::BufReader::new(file);

        let mut matched: Vec<AuditTailRow> = Vec::new();
        for (idx, line) in reader.lines().enumerate() {
            let line_no = idx + 1;
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!(error = %e, "audit.log read error, stopping tail walk early");
                    break;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            let entry: AuditEntry = match serde_json::from_str(&line) {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        line_no,
                        "skipping malformed audit.log line"
                    );
                    continue;
                }
            };
            if let Some(since) = opts.since {
                if entry.timestamp < since {
                    continue;
                }
            }
            if let Some(action) = &opts.action {
                if &entry.action != action {
                    continue;
                }
            }
            matched.push(AuditTailRow { line_no, entry });
        }

        // Keep the last `limit` in file order — same window
        // semantic as `tail -n LIMIT [matching file]`.
        if matched.len() > limit {
            let drop_n = matched.len() - limit;
            matched.drain(..drop_n);
        }
        Ok(matched)
    }
}

/// One audit log entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    /// ISO-8601 UTC timestamp.
    pub timestamp: chrono::DateTime<Utc>,
    /// Action name (`"import"`, `"search"`, `"export"`, `"serve.start"`, …).
    pub action: String,
    /// Free-form detail object — keep it small and structured.
    pub detail: serde_json::Value,
}

impl AuditEntry {
    /// Convenience: build an entry stamped with `now()`.
    pub fn new(action: impl Into<String>, detail: serde_json::Value) -> Self {
        Self {
            timestamp: Utc::now(),
            action: action.into(),
            detail,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static AUDIT_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = AUDIT_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-audit-{nonce}-{pid}-{seq}",
            pid = std::process::id()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn writes_one_jsonl_per_record() {
        let dir = tmp_dir();
        let audit = Audit::new(&dir);
        audit.record(AuditEntry::new(
            "import",
            json!({"adapter": "claude-code", "records": 12}),
        ));
        audit.record(AuditEntry::new(
            "search",
            json!({"query": "vim", "hits": 3}),
        ));
        let body = std::fs::read_to_string(audit.path()).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        let first: AuditEntry = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first.action, "import");
        assert_eq!(first.detail["adapter"], "claude-code");
        let second: AuditEntry = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second.action, "search");
    }

    #[test]
    fn appends_when_log_exists() {
        let dir = tmp_dir();
        let audit = Audit::new(&dir);
        audit.record(AuditEntry::new("first", json!({})));
        audit.record(AuditEntry::new("second", json!({})));
        audit.record(AuditEntry::new("third", json!({})));
        let body = std::fs::read_to_string(audit.path()).unwrap();
        assert_eq!(body.lines().count(), 3);
    }

    #[test]
    fn creates_parent_directory_lazily() {
        let dir = tmp_dir().join("nested/sub/dir");
        // Directory does NOT exist yet.
        assert!(!dir.exists());
        let audit = Audit::new(&dir);
        audit.record(AuditEntry::new("late", json!({})));
        assert!(audit.path().exists(), "audit.log should have been created");
    }

    #[test]
    fn missing_directory_does_not_propagate_error() {
        // Even with a path we can't write to (e.g. /root on macOS), the
        // record() call must not panic or return an error.
        let audit = Audit::new(Path::new(
            "/nonexistent-anamnesis-path/that/cannot/be/created",
        ));
        // No need to assert anything except that the call completes.
        audit.record(AuditEntry::new("safe", json!({})));
    }

    // ─── Round-84 PR-78f: Audit::tail ────────────────────────────────

    /// Missing log → empty Vec, not an error. A store that's
    /// never had a mutation has no audit yet.
    #[test]
    fn tail_returns_empty_when_log_does_not_exist() {
        let dir = tmp_dir();
        let audit = Audit::new(&dir);
        let rows = audit.tail(&AuditTailOptions::default()).unwrap();
        assert!(rows.is_empty());
    }

    /// `tail -n N` semantic: the last N entries in **file order**
    /// (oldest first within the window). 1-based `line_no` lines
    /// up with `head -n LINE`.
    #[test]
    fn tail_returns_last_n_entries_in_file_order() {
        let dir = tmp_dir();
        let audit = Audit::new(&dir);
        for action in ["a", "b", "c", "d", "e"] {
            audit.record(AuditEntry::new(action, json!({})));
        }
        let rows = audit
            .tail(&AuditTailOptions {
                limit: Some(3),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].entry.action, "c");
        assert_eq!(rows[0].line_no, 3);
        assert_eq!(rows[1].entry.action, "d");
        assert_eq!(rows[1].line_no, 4);
        assert_eq!(rows[2].entry.action, "e");
        assert_eq!(rows[2].line_no, 5);
    }

    /// `--action` exact-matches the entry's action field. Filter
    /// runs *before* the limit so a giant unrelated action
    /// (e.g. 500 `search` rows) doesn't starve a small target
    /// (e.g. 3 `forget` rows).
    #[test]
    fn tail_filters_by_action_before_limit() {
        let dir = tmp_dir();
        let audit = Audit::new(&dir);
        for _ in 0..10 {
            audit.record(AuditEntry::new("search", json!({})));
        }
        audit.record(AuditEntry::new("forget", json!({"why": "test"})));
        for _ in 0..10 {
            audit.record(AuditEntry::new("search", json!({})));
        }

        let rows = audit
            .tail(&AuditTailOptions {
                limit: Some(5),
                action: Some("forget".into()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].entry.action, "forget");
        assert_eq!(rows[0].entry.detail["why"], "test");
    }

    /// `--since` drops entries strictly older than the given
    /// instant. Combine with `--action` to reproduce "show me
    /// what was forgotten in the last hour."
    #[test]
    fn tail_filters_by_since_timestamp() {
        let dir = tmp_dir();
        let audit = Audit::new(&dir);
        // Write 3 entries, then capture a cutoff, then write 2 more.
        for action in ["old1", "old2", "old3"] {
            audit.record(AuditEntry::new(action, json!({})));
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        let cutoff = Utc::now();
        std::thread::sleep(std::time::Duration::from_millis(20));
        for action in ["new1", "new2"] {
            audit.record(AuditEntry::new(action, json!({})));
        }
        let rows = audit
            .tail(&AuditTailOptions {
                limit: Some(100),
                since: Some(cutoff),
                ..Default::default()
            })
            .unwrap();
        let actions: Vec<&str> = rows.iter().map(|r| r.entry.action.as_str()).collect();
        assert_eq!(actions, vec!["new1", "new2"]);
    }

    /// Malformed JSON lines are skipped, not fatal. An operator
    /// who hand-edited `audit.log` (or whose disk corrupted one
    /// row) should still get a working tail.
    #[test]
    fn tail_skips_malformed_jsonl_lines() {
        let dir = tmp_dir();
        let audit = Audit::new(&dir);
        audit.record(AuditEntry::new("good1", json!({})));
        // Append a junk line directly.
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(audit.path())
                .unwrap();
            writeln!(file, "this is not json {{").unwrap();
        }
        audit.record(AuditEntry::new("good2", json!({})));

        let rows = audit
            .tail(&AuditTailOptions {
                limit: Some(10),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].entry.action, "good1");
        assert_eq!(rows[0].line_no, 1);
        assert_eq!(rows[1].entry.action, "good2");
        assert_eq!(
            rows[1].line_no, 3,
            "line_no must still reflect the *file* line, including the skipped row"
        );
    }

    /// Limit clamps to `[1, AUDIT_TAIL_MAX_LIMIT]` so a typo'd
    /// `-n 1000000` can't exhaust memory.
    #[test]
    fn tail_clamps_limit_to_max() {
        let dir = tmp_dir();
        let audit = Audit::new(&dir);
        for i in 0..5 {
            audit.record(AuditEntry::new(format!("a{i}"), json!({})));
        }
        // Way over the cap → returns ≤ cap, never panics.
        let rows = audit
            .tail(&AuditTailOptions {
                limit: Some(AUDIT_TAIL_MAX_LIMIT * 10),
                ..Default::default()
            })
            .unwrap();
        assert!(rows.len() <= AUDIT_TAIL_MAX_LIMIT);
        assert_eq!(rows.len(), 5, "actual data is 5 rows");
        // Limit of 0 clamps up to 1.
        let rows = audit
            .tail(&AuditTailOptions {
                limit: Some(0),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows.len(), 1);
    }
}
