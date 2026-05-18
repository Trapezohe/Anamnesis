//! Filesystem + SQLite scanner for a Hermes Agent install.
//!
//! Hermes (Nous Research, MIT) keeps cross-session context in three
//! places under its data dir (default `~/.hermes/` on unix-likes,
//! `%LOCALAPPDATA%\hermes` on Windows):
//!
//!   1. `MEMORY.md` — environment info, past lessons, system state.
//!   2. `USER.md`   — user preferences, work style, custom settings.
//!   3. A SQLite database with the full session log. The exact
//!      filename and schema vary by Hermes release (v0.7+ supports 6
//!      pluggable backends), so this scanner probes for any `.db` /
//!      `.sqlite` / `.sqlite3` file in the data dir and introspects
//!      the schema via `PRAGMA table_info` instead of pinning a
//!      shape.
//!
//! All access is **read-only** — per §-1.2.2 the adapter never writes
//! back to Hermes.

use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags};

/// One row pulled from a Hermes-installed SQLite session table.
#[derive(Debug, Clone, Default)]
pub struct HermesSessionRow {
    /// Best-effort `id` (whatever PK-ish column the table exposes).
    pub id: String,
    /// Content column we recovered. Hermes hasn't shipped a stable
    /// public schema for session rows yet (the `sqlite-with-gcov`
    /// playbook docs show several variants across versions), so the
    /// scanner picks the first text column whose name matches one of
    /// the common candidates (see `CONTENT_CANDIDATES`).
    pub content: String,
    /// Source table name — preserved so the normalizer / consumer can
    /// distinguish e.g. `messages` vs `events` vs `tool_calls`.
    pub table: String,
    /// Optional `role` (when a `role` / `speaker` / `actor` column is
    /// present — helps the normalizer build a readable session log).
    pub role: Option<String>,
    /// Optional unix epoch seconds (when a recognized `*_at` /
    /// `timestamp` column is present).
    pub timestamp: Option<i64>,
    /// Anything else the row exposed that we didn't pull explicitly,
    /// keyed by column name. Captured so future schema additions
    /// don't drop information.
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Markdown blocks read from `MEMORY.md` / `USER.md`.
#[derive(Debug, Clone)]
pub struct HermesMarkdownBlock {
    /// Which file the block came from (`"MEMORY.md"` or `"USER.md"`).
    pub source_file: String,
    /// Absolute path on disk.
    pub path: PathBuf,
    /// Full file body.
    pub content: String,
    /// File mtime as unix seconds, if metadata read succeeded.
    pub mtime_unix: Option<i64>,
}

/// Walk `data_dir` and produce a `HermesScan` describing what's there.
///
/// Missing files / missing tables are NOT errors — they reduce the
/// result. A directory that contains no recognized Hermes data
/// yields `HermesScan { ..empty }` with `total() == 0` and the caller
/// can decide whether to warn.
pub fn scan_hermes_dir(data_dir: &Path) -> HermesScan {
    let mut scan = HermesScan::default();
    scan.markdown.extend(read_markdown_blocks(data_dir));

    // Probe for SQLite-shaped files; ignore anything that doesn't
    // open cleanly read-only. Treat each db file as one logical
    // session source.
    if let Ok(read_dir) = fs::read_dir(data_dir) {
        for entry in read_dir.flatten() {
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let Some(ext) = p.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            if !matches!(ext.to_lowercase().as_str(), "db" | "sqlite" | "sqlite3") {
                continue;
            }
            match read_session_rows(&p) {
                Ok(rows) => scan
                    .session_rows
                    .extend(rows.into_iter().map(|r| (p.clone(), r))),
                Err(e) => {
                    tracing::debug!(
                        path = %p.display(),
                        error = %e,
                        "hermes scanner: sqlite file skipped (open / no session-like table)"
                    );
                }
            }
        }
    }
    scan
}

/// Bag of everything `scan_hermes_dir` recovered.
#[derive(Debug, Default)]
pub struct HermesScan {
    /// Markdown blocks (MEMORY.md / USER.md).
    pub markdown: Vec<HermesMarkdownBlock>,
    /// SQLite session rows. Each entry knows which file it came from.
    pub session_rows: Vec<(PathBuf, HermesSessionRow)>,
}

impl HermesScan {
    /// Total raw record count this scan would yield.
    pub fn total(&self) -> usize {
        self.markdown.len() + self.session_rows.len()
    }
}

fn read_markdown_blocks(data_dir: &Path) -> Vec<HermesMarkdownBlock> {
    let mut out = Vec::new();
    for name in ["MEMORY.md", "USER.md"] {
        let p = data_dir.join(name);
        if !p.is_file() {
            continue;
        }
        let content = match fs::read_to_string(&p) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    path = %p.display(),
                    error = %e,
                    "hermes scanner: unreadable markdown file; skipping"
                );
                continue;
            }
        };
        let mtime_unix = fs::metadata(&p)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64);
        out.push(HermesMarkdownBlock {
            source_file: name.into(),
            path: p,
            content,
            mtime_unix,
        });
    }
    out
}

// Column-name candidate lists used by the defensive SQLite probe.
// These reflect the variants seen in Hermes-related repos (`hermes`,
// `awesome-hermes-agent`, the MemOS Cloud plugin) and adjacent
// frameworks. None of these are spec; the scanner accepts whatever
// the table actually exposes.

const CONTENT_CANDIDATES: &[&str] = &[
    "content", "message", "text", "body", "data", "response", "summary",
];
const ID_CANDIDATES: &[&str] = &["id", "rowid", "uuid", "session_id", "message_id"];
const ROLE_CANDIDATES: &[&str] = &["role", "speaker", "actor", "sender", "from_role"];
const TIME_CANDIDATES: &[&str] = &["created_at", "timestamp", "ts", "time", "at", "updated_at"];

fn read_session_rows(path: &Path) -> rusqlite::Result<Vec<HermesSessionRow>> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    let tables = list_user_tables(&conn)?;
    let mut out = Vec::new();
    for table in tables {
        let cols = pragma_columns(&conn, &table)?;
        let Some(content_col) = cols
            .iter()
            .find(|c| CONTENT_CANDIDATES.contains(&c.as_str()))
        else {
            // Skip tables that don't look session-like. We don't want
            // to pull in `alembic_version` / `migration_state` / etc.
            tracing::debug!(table = %table, "no content-shaped column; skipping");
            continue;
        };
        let id_col = cols
            .iter()
            .find(|c| ID_CANDIDATES.contains(&c.as_str()))
            .cloned();
        let role_col = cols
            .iter()
            .find(|c| ROLE_CANDIDATES.contains(&c.as_str()))
            .cloned();
        let time_col = cols
            .iter()
            .find(|c| TIME_CANDIDATES.contains(&c.as_str()))
            .cloned();

        // Build SELECT with all cols (quoted for reserved-word safety).
        let select = format!(
            "SELECT {} FROM \"{}\"",
            cols.iter()
                .map(|c| format!("\"{c}\""))
                .collect::<Vec<_>>()
                .join(", "),
            table
        );
        let mut stmt = conn.prepare(&select)?;
        let rows = stmt
            .query_map([], |r| {
                let mut row = HermesSessionRow {
                    table: table.clone(),
                    ..Default::default()
                };
                for (i, name) in cols.iter().enumerate() {
                    let value_text = read_opt_text(r, i);
                    if Some(name) == id_col.as_ref() {
                        row.id = value_text.clone().unwrap_or_default();
                    } else if name == content_col {
                        row.content = value_text.clone().unwrap_or_default();
                    } else if Some(name) == role_col.as_ref() {
                        row.role = value_text.clone();
                    } else if Some(name) == time_col.as_ref() {
                        row.timestamp = value_text.as_deref().and_then(parse_unix_seconds);
                    } else if let Some(v) = value_text {
                        row.extra.insert(name.clone(), v.into());
                    }
                }
                if row.id.is_empty() {
                    // Synthesize a stable id from (table, rowid) when
                    // no candidate id column existed.
                    row.id = format!(
                        "{}#{}",
                        table,
                        blake3::hash(row.content.as_bytes()).to_hex()
                    );
                }
                Ok(row)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        out.extend(rows);
    }
    Ok(out)
}

fn list_user_tables(conn: &Connection) -> rusqlite::Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT name FROM sqlite_master \
         WHERE type='table' AND name NOT LIKE 'sqlite_%' \
           AND name NOT LIKE 'alembic_%' \
         ORDER BY name",
    )?;
    let out = stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(out)
}

fn pragma_columns(conn: &Connection, table: &str) -> rusqlite::Result<Vec<String>> {
    let sql = format!("PRAGMA table_info(\"{table}\")");
    let mut stmt = conn.prepare(&sql)?;
    let out = stmt
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(out)
}

fn read_opt_text(row: &rusqlite::Row<'_>, idx: usize) -> Option<String> {
    match row.get_ref(idx).ok()? {
        rusqlite::types::ValueRef::Null => None,
        rusqlite::types::ValueRef::Integer(i) => Some(i.to_string()),
        rusqlite::types::ValueRef::Real(f) => Some(f.to_string()),
        rusqlite::types::ValueRef::Text(t) => Some(String::from_utf8_lossy(t).into_owned()),
        rusqlite::types::ValueRef::Blob(b) => Some(String::from_utf8_lossy(b).into_owned()),
    }
}

fn parse_unix_seconds(s: &str) -> Option<i64> {
    if let Ok(epoch) = s.parse::<i64>() {
        return Some(epoch);
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&chrono::Utc).timestamp());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static HERMES_SCAN_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = HERMES_SCAN_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "hermes-scanner-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn empty_dir_yields_empty_scan() {
        let dir = tmp();
        let s = scan_hermes_dir(&dir);
        assert_eq!(s.total(), 0);
    }

    #[test]
    fn picks_up_memory_and_user_md() {
        let dir = tmp();
        fs::write(dir.join("MEMORY.md"), "system: production").unwrap();
        fs::write(dir.join("USER.md"), "prefers Rust").unwrap();
        // A red-herring .md file we should ignore.
        fs::write(dir.join("README.md"), "should be skipped").unwrap();
        let s = scan_hermes_dir(&dir);
        assert_eq!(s.markdown.len(), 2);
        let names: Vec<&str> = s.markdown.iter().map(|m| m.source_file.as_str()).collect();
        assert!(names.contains(&"MEMORY.md"));
        assert!(names.contains(&"USER.md"));
        assert!(!names.contains(&"README.md"));
    }

    #[test]
    fn reads_sqlite_session_with_canonical_columns() {
        let dir = tmp();
        let dbp = dir.join("sessions.db");
        let conn = Connection::open(&dbp).unwrap();
        conn.execute_batch(
            r#"CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                role TEXT,
                content TEXT NOT NULL,
                created_at TEXT
            );
            INSERT INTO messages VALUES
              ('m1', 'user',      'hello',     '2026-01-01T00:00:00Z'),
              ('m2', 'assistant', 'hi back',   '2026-01-01T00:00:01Z');"#,
        )
        .unwrap();

        let s = scan_hermes_dir(&dir);
        assert_eq!(s.markdown.len(), 0);
        assert_eq!(s.session_rows.len(), 2);

        let (db_path, row0) = &s.session_rows[0];
        assert_eq!(db_path, &dbp);
        assert_eq!(row0.table, "messages");
        assert!(row0.id == "m1" || row0.id == "m2");
        assert!(!row0.content.is_empty());
        assert!(matches!(
            row0.role.as_deref(),
            Some("user") | Some("assistant")
        ));
        // Two timestamps both RFC3339 → both parse to positive epoch.
        assert!(row0.timestamp.unwrap() > 1_700_000_000);
    }

    #[test]
    fn ignores_tables_without_content_column() {
        let dir = tmp();
        let dbp = dir.join("misc.db");
        let conn = Connection::open(&dbp).unwrap();
        // A schema_migrations-style table the scanner must skip.
        conn.execute_batch(
            "CREATE TABLE schema_migrations (version TEXT, applied_at TEXT);
             INSERT INTO schema_migrations VALUES ('0001', '2024-01-01');",
        )
        .unwrap();
        let s = scan_hermes_dir(&dir);
        assert_eq!(s.session_rows.len(), 0);
    }

    #[test]
    fn synthesizes_id_when_table_lacks_one() {
        let dir = tmp();
        let dbp = dir.join("noid.db");
        let conn = Connection::open(&dbp).unwrap();
        conn.execute_batch(
            "CREATE TABLE events (content TEXT NOT NULL, role TEXT);
             INSERT INTO events VALUES ('alpha', 'user');",
        )
        .unwrap();
        let s = scan_hermes_dir(&dir);
        assert_eq!(s.session_rows.len(), 1);
        let row = &s.session_rows[0].1;
        assert!(!row.id.is_empty(), "synthesized id should be non-empty");
        assert!(row.id.starts_with("events#"));
    }
}
