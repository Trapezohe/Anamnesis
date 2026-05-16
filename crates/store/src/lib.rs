//! SQLite-backed storage for Anamnesis records.
//!
//! Phase-0 scope: schema bootstrap + migration runner. Read/write APIs land
//! in Phase 1 alongside the first adapter.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::path::Path;

use rusqlite::Connection;
use thiserror::Error;

/// Embedded SQL migrations. Add new files in `migrations/` and list them here
/// in order.
const MIGRATIONS: &[(&str, &str)] = &[("0001_init", include_str!("migrations/0001_init.sql"))];

/// Store-layer errors.
#[derive(Debug, Error)]
pub enum StoreError {
    /// SQLite error.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// Schema version on disk is newer than this binary supports.
    #[error("database schema is newer than this binary supports (found {found})")]
    SchemaTooNew {
        /// Version found on disk.
        found: u32,
    },
}

/// Crate result.
pub type Result<T> = std::result::Result<T, StoreError>;

/// Anamnesis storage handle. Owns a single SQLite connection.
pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open (or create) a store at the given path and run pending migrations.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        let mut store = Self { conn };
        store.run_migrations()?;
        Ok(store)
    }

    /// Open an in-memory store (useful for tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let mut store = Self { conn };
        store.run_migrations()?;
        Ok(store)
    }

    fn run_migrations(&mut self) -> Result<()> {
        // Tiny home-grown runner: keep applied migration ids in a meta table.
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS _migrations (
                id    TEXT PRIMARY KEY,
                applied_at INTEGER NOT NULL
            );",
        )?;

        for (id, sql) in MIGRATIONS {
            let already: i64 = self.conn.query_row(
                "SELECT COUNT(1) FROM _migrations WHERE id = ?1",
                [id],
                |r| r.get(0),
            )?;
            if already == 0 {
                let tx = self.conn.transaction()?;
                tx.execute_batch(sql)?;
                tx.execute(
                    "INSERT INTO _migrations(id, applied_at) VALUES (?1, strftime('%s','now'))",
                    [id],
                )?;
                tx.commit()?;
                tracing::info!(migration = id, "applied migration");
            }
        }
        Ok(())
    }

    /// Borrow the inner connection. Internal use; expect a richer API in Phase 1.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_in_memory_runs_migrations() {
        let store = Store::open_in_memory().unwrap();
        let count: i64 = store
            .conn()
            .query_row("SELECT COUNT(1) FROM records", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);

        let version: String = store
            .conn()
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(version, "1");
    }
}
