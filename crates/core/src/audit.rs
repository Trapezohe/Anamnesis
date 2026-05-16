//! Append-only audit log — see `BLUEPRINT.md §7`.
//!
//! Records every privileged action (import, search, export, serve) as a
//! JSON Lines record under `$DATA_DIR/audit.log`. No rotation in Phase 1;
//! ops can `tail -f` or `jq` directly.

use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};

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

    fn tmp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("anamnesis-audit-{nonce}"));
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
}
