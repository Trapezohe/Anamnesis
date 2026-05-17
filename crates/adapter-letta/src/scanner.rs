//! Read-only scanner for Letta's self-hosted SQLite store.
//!
//! Letta (formerly MemGPT) defaults to `~/.letta/letta.db` in
//! development mode and Postgres in production. This adapter is the
//! SQLite path; Postgres mode is out of scope until §-2.3 PR for the
//! REST/API variant lands.
//!
//! ## Schema we read
//!
//! Letta's `block` table holds the agent's **core memory** — short,
//! always-in-context chunks like `persona`, `human`, or any custom
//! label. Each row is one memory block. We pull every block as one
//! `LettaBlockRow` and let the normalizer turn it into a
//! `Kind::Fact` `AnamnesisRecord`.
//!
//! ## Defensive parsing
//!
//! Letta's schema is in active flux (ORM migrations land regularly).
//! We do **not** hardcode a column list — instead we introspect via
//! `PRAGMA table_info(block)` and pull whatever's there. Required:
//! `id`, `value`. Everything else (`label`, `description`, `limit`,
//! `template_name`, `metadata_`, `created_at`, `updated_at`,
//! `organization_id`, etc.) is best-effort.
//!
//! Future PRs add `archival_passages` (long-term memory) once we have
//! a real Letta install to validate the schema against.

use rusqlite::{Connection, OpenFlags};
use std::path::Path;

/// One row from Letta's `block` table.
///
/// Anamnesis only cares about the fields that drop cleanly into the
/// `AnamnesisRecord` schema. Anything Letta-specific we don't yet
/// understand goes into `extra` so the normalizer can stash it in
/// `record.metadata` as `letta_*` keys for traceability.
#[derive(Debug, Clone, Default)]
pub struct LettaBlockRow {
    /// `id` column. Required — used as the source-native id.
    pub id: String,
    /// `value` column. Required — becomes `AnamnesisRecord.content`.
    pub value: String,
    /// `label` column (e.g. `"persona"`, `"human"`). Optional. Used to
    /// shade the record's scope and to keep the human-meaningful name
    /// in metadata.
    pub label: Option<String>,
    /// `description` column. Optional. The block's prompt-time hint.
    pub description: Option<String>,
    /// `template_name` column. Optional. Letta's notion of a named
    /// reusable template across agents.
    pub template_name: Option<String>,
    /// `metadata_` column (note trailing underscore — Letta's actual
    /// column name) — Letta-specific JSON. Best-effort opaque preserve.
    pub metadata_json: Option<String>,
    /// `created_at` — TEXT or INTEGER. Optional.
    pub created_at: Option<String>,
    /// `updated_at` — TEXT or INTEGER. Optional.
    pub updated_at: Option<String>,
    /// Any other columns the schema exposed that we didn't explicitly
    /// pull into a typed field. Best-effort string capture so future
    /// schema additions don't drop information.
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Open the Letta SQLite file read-only and pull every `block` row.
///
/// Errors:
/// - I/O failures opening the file.
/// - The `block` table missing (returns `QueryReturnedNoRows` — the
///   caller decides whether that's "Letta installation is too new /
///   too old to match this adapter version" or "user pointed `--path`
///   at the wrong file").
pub fn read_all_blocks(path: &Path) -> rusqlite::Result<Vec<LettaBlockRow>> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    let cols = pragma_columns(&conn, "block")?;
    if !cols.iter().any(|c| c == "id") || !cols.iter().any(|c| c == "value") {
        return Err(rusqlite::Error::QueryReturnedNoRows);
    }
    // Letta's schema includes columns whose names are SQL reserved
    // keywords (e.g. `limit`). Double-quote every name in the SELECT
    // to make the query immune to reserved-word collisions.
    let select = format!(
        "SELECT {} FROM block",
        cols.iter()
            .map(|c| format!("\"{c}\""))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut stmt = conn.prepare(&select)?;
    let rows = stmt
        .query_map([], |r| {
            let mut row = LettaBlockRow::default();
            for (i, name) in cols.iter().enumerate() {
                match name.as_str() {
                    "id" => row.id = read_text(r, i)?,
                    "value" => row.value = read_text(r, i)?,
                    "label" => row.label = read_opt_text(r, i),
                    "description" => row.description = read_opt_text(r, i),
                    "template_name" => row.template_name = read_opt_text(r, i),
                    // Letta's actual column is `metadata_` with the
                    // trailing underscore (collision with `metadata`
                    // sqlalchemy builtin). Accept both for safety.
                    "metadata_" | "metadata" => row.metadata_json = read_opt_text(r, i),
                    "created_at" => row.created_at = read_opt_text(r, i),
                    "updated_at" => row.updated_at = read_opt_text(r, i),
                    other => {
                        if let Some(v) = read_opt_text(r, i) {
                            row.extra.insert(other.to_string(), v.into());
                        }
                    }
                }
            }
            Ok(row)
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn pragma_columns(conn: &Connection, table: &str) -> rusqlite::Result<Vec<String>> {
    let sql = format!("PRAGMA table_info({table})");
    let mut stmt = conn.prepare(&sql)?;
    let out = stmt
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(out)
}

fn read_text(row: &rusqlite::Row<'_>, idx: usize) -> rusqlite::Result<String> {
    let v = row.get_ref(idx)?;
    Ok(match v {
        rusqlite::types::ValueRef::Null => String::new(),
        rusqlite::types::ValueRef::Integer(i) => i.to_string(),
        rusqlite::types::ValueRef::Real(f) => f.to_string(),
        rusqlite::types::ValueRef::Text(t) => String::from_utf8_lossy(t).into_owned(),
        rusqlite::types::ValueRef::Blob(b) => String::from_utf8_lossy(b).into_owned(),
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir() -> std::path::PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("anamnesis-letta-scan-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn missing_block_table_yields_error() {
        let dir = tmp_dir();
        let p = dir.join("empty.db");
        Connection::open(&p).unwrap(); // create empty DB
        let err = read_all_blocks(&p).unwrap_err();
        // Either "no such table" (most likely) or our explicit
        // QueryReturnedNoRows guard. Both are acceptable signals.
        let msg = err.to_string();
        assert!(
            msg.contains("no such table") || matches!(err, rusqlite::Error::QueryReturnedNoRows),
            "expected missing-table error, got: {msg}"
        );
    }

    #[test]
    fn reads_minimal_two_column_schema() {
        // The minimum required schema: `id` + `value`.
        let dir = tmp_dir();
        let p = dir.join("min.db");
        let conn = Connection::open(&p).unwrap();
        conn.execute_batch(
            "CREATE TABLE block (id TEXT PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO block(id, value) VALUES
               ('b1', 'persona content'),
               ('b2', 'human content');",
        )
        .unwrap();
        let rows = read_all_blocks(&p).unwrap();
        assert_eq!(rows.len(), 2);
        let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"b1"));
        assert!(ids.contains(&"b2"));
    }

    #[test]
    fn reads_full_letta_like_schema() {
        // Closer to what an actual Letta install looks like —
        // including `metadata_` (with trailing underscore) and a
        // column the adapter doesn't know about (`limit`).
        let dir = tmp_dir();
        let p = dir.join("full.db");
        let conn = Connection::open(&p).unwrap();
        conn.execute_batch(
            r#"CREATE TABLE block (
                id TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                label TEXT,
                description TEXT,
                template_name TEXT,
                metadata_ TEXT,
                "limit" INTEGER,
                organization_id TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            INSERT INTO block VALUES
              ('b1', 'I am Sam, an AI assistant.',
               'persona', 'how the agent sees itself',
               NULL, '{"version": 2}', 2000, 'org-1',
               '2026-04-01T00:00:00Z', '2026-04-15T00:00:00Z');"#,
        )
        .unwrap();
        let rows = read_all_blocks(&p).unwrap();
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.id, "b1");
        assert_eq!(r.value, "I am Sam, an AI assistant.");
        assert_eq!(r.label.as_deref(), Some("persona"));
        assert_eq!(r.description.as_deref(), Some("how the agent sees itself"));
        assert!(r.template_name.is_none());
        assert_eq!(r.metadata_json.as_deref(), Some(r#"{"version": 2}"#));
        assert_eq!(r.created_at.as_deref(), Some("2026-04-01T00:00:00Z"));
        // Unknown column captured in extras.
        assert_eq!(r.extra.get("limit").and_then(|v| v.as_str()), Some("2000"));
        assert_eq!(
            r.extra.get("organization_id").and_then(|v| v.as_str()),
            Some("org-1")
        );
    }

    #[test]
    fn integer_id_coerces_to_string() {
        let dir = tmp_dir();
        let p = dir.join("intid.db");
        let conn = Connection::open(&p).unwrap();
        conn.execute_batch(
            "CREATE TABLE block (id INTEGER PRIMARY KEY, value TEXT);
             INSERT INTO block(id, value) VALUES (42, 'content');",
        )
        .unwrap();
        let rows = read_all_blocks(&p).unwrap();
        assert_eq!(rows[0].id, "42");
    }
}
