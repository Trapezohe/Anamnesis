//! Read raw rows from a mem0 SQLite `memories` table.
//!
//! The schema varies across mem0 versions, so we introspect column names
//! and pull whatever subset is present. Required: `id`, `memory`. Optional:
//! `user_id`, `agent_id`, `run_id`, `metadata`, `created_at`, `updated_at`,
//! `hash`. Anything else is captured into a JSON blob under `extra` so
//! downstream provenance stays useful.

use std::path::Path;

use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};

/// One mem0 row, normalized into a stable in-memory shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mem0Row {
    /// `id` column.
    pub id: String,
    /// `memory` column — the actual text.
    pub memory: String,
    /// Optional `user_id`.
    pub user_id: Option<String>,
    /// Optional `agent_id`.
    pub agent_id: Option<String>,
    /// Optional `run_id`.
    pub run_id: Option<String>,
    /// Optional JSON `metadata` blob (kept as a string to avoid forcing
    /// callers to round-trip it).
    pub metadata_json: Option<String>,
    /// Optional `created_at` (raw value from the DB; could be epoch or ISO).
    pub created_at: Option<String>,
    /// Optional `updated_at`.
    pub updated_at: Option<String>,
    /// Other columns we didn't recognise, captured verbatim.
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Open the database read-only and pull every `memories` row.
pub fn read_all(path: &Path) -> rusqlite::Result<Vec<Mem0Row>> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    let cols = pragma_columns(&conn, "memories")?;
    if !cols.iter().any(|c| c == "id") || !cols.iter().any(|c| c == "memory") {
        return Err(rusqlite::Error::QueryReturnedNoRows);
    }
    let select = format!("SELECT {} FROM memories", cols.join(", "));
    let mut stmt = conn.prepare(&select)?;
    let rows = stmt
        .query_map([], |r| {
            let mut row = Mem0Row {
                id: String::new(),
                memory: String::new(),
                user_id: None,
                agent_id: None,
                run_id: None,
                metadata_json: None,
                created_at: None,
                updated_at: None,
                extra: serde_json::Map::new(),
            };
            for (i, name) in cols.iter().enumerate() {
                match name.as_str() {
                    "id" => row.id = read_text(r, i)?,
                    "memory" => row.memory = read_text(r, i)?,
                    "user_id" => row.user_id = read_opt_text(r, i),
                    "agent_id" => row.agent_id = read_opt_text(r, i),
                    "run_id" => row.run_id = read_opt_text(r, i),
                    "metadata" => row.metadata_json = read_opt_text(r, i),
                    "created_at" => row.created_at = read_opt_text(r, i),
                    "updated_at" => row.updated_at = read_opt_text(r, i),
                    other => {
                        if let Some(val) = read_opt_text(r, i) {
                            row.extra.insert(other.to_string(), val.into());
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
    // Accept TEXT, INTEGER, REAL; coerce to string so adapter is tolerant
    // of mem0's "id as INTEGER PRIMARY KEY" variant.
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
        v => match v {
            rusqlite::types::ValueRef::Integer(i) => Some(i.to_string()),
            rusqlite::types::ValueRef::Real(f) => Some(f.to_string()),
            rusqlite::types::ValueRef::Text(t) => Some(String::from_utf8_lossy(t).into_owned()),
            rusqlite::types::ValueRef::Blob(b) => Some(String::from_utf8_lossy(b).into_owned()),
            rusqlite::types::ValueRef::Null => None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir() -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("anamnesis-mem0-scan-{nonce}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_full_schema_db(path: &std::path::Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                memory TEXT NOT NULL,
                user_id TEXT,
                agent_id TEXT,
                run_id TEXT,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT,
                hash TEXT
            );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memories VALUES('m1','user prefers vim','u1','a1','r1','{\"k\":1}','2026-05-01T00:00:00Z',NULL,'h1')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memories VALUES('m2','use real DB',NULL,NULL,NULL,NULL,NULL,NULL,NULL)",
            [],
        )
        .unwrap();
    }

    #[test]
    fn reads_full_schema_into_normalized_rows() {
        let dir = tmp_dir();
        let db = dir.join("db.sqlite");
        write_full_schema_db(&db);
        let rows = read_all(&db).unwrap();
        assert_eq!(rows.len(), 2);
        let m1 = &rows[0];
        assert_eq!(m1.id, "m1");
        assert_eq!(m1.memory, "user prefers vim");
        assert_eq!(m1.user_id.as_deref(), Some("u1"));
        assert_eq!(m1.agent_id.as_deref(), Some("a1"));
        assert_eq!(m1.run_id.as_deref(), Some("r1"));
        assert_eq!(m1.metadata_json.as_deref(), Some("{\"k\":1}"));
        assert_eq!(m1.created_at.as_deref(), Some("2026-05-01T00:00:00Z"));
        // `hash` is unrecognised → captured into extra.
        assert_eq!(m1.extra.get("hash").and_then(|v| v.as_str()), Some("h1"));

        let m2 = &rows[1];
        assert_eq!(m2.id, "m2");
        assert_eq!(m2.memory, "use real DB");
        assert!(m2.user_id.is_none());
        assert!(m2.metadata_json.is_none());
    }

    #[test]
    fn missing_optional_columns_are_handled() {
        let dir = tmp_dir();
        let db = dir.join("db.sqlite");
        let conn = Connection::open(&db).unwrap();
        // Bare-minimum schema: just id + memory.
        conn.execute_batch("CREATE TABLE memories (id TEXT PRIMARY KEY, memory TEXT NOT NULL);")
            .unwrap();
        conn.execute("INSERT INTO memories(id, memory) VALUES('x','hello')", [])
            .unwrap();
        let rows = read_all(&db).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].memory, "hello");
        assert!(rows[0].user_id.is_none());
        assert!(rows[0].extra.is_empty());
    }

    #[test]
    fn integer_id_column_is_coerced_to_string() {
        let dir = tmp_dir();
        let db = dir.join("db.sqlite");
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch("CREATE TABLE memories (id INTEGER PRIMARY KEY, memory TEXT NOT NULL);")
            .unwrap();
        conn.execute("INSERT INTO memories(id, memory) VALUES(42,'x')", [])
            .unwrap();
        let rows = read_all(&db).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "42");
    }

    #[test]
    fn missing_memories_table_errors() {
        let dir = tmp_dir();
        let db = dir.join("db.sqlite");
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch("CREATE TABLE other(x INTEGER);")
            .unwrap();
        let r = read_all(&db);
        assert!(r.is_err());
    }
}
