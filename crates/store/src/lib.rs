//! SQLite-backed storage for Anamnesis records.
//!
//! The crate exposes `Store::open` / `Store::open_in_memory` plus a typed
//! API in `api` for records, chunks, embeddings, jobs, and sources. The
//! raw `Connection` is intentionally kept private to callers outside this
//! crate; only tests use `conn()` directly.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod api;
pub mod cjk;
mod vec_ext;

pub use api::{
    normalize_user_tag_name, ChunkHit, ChunkLookup, DuplicateRawHashFilter, DuplicateRawHashGroup,
    DuplicateRawHashRecord, ForgetCascadeCounts, ForgetRecordOutcome, ForgetRecordPreview,
    ForgetTombstonePreview, ForgottenRecord, LineageChain, ListForgottenFilter, McpRequestMetric,
    McpToolMetricSummary, PendingEmbeddingJob, RecordHeader, RecordSummary, SearchFilter,
    SourceRow, SourceWithCounts, StoreStats, UnforgetRecordOutcome, UserTagMutation,
    UserTagOperation, LIST_DUPLICATE_RAW_HASHES_MAX_LIMIT, LIST_FORGOTTEN_MAX_LIMIT,
    MAX_LIST_LIMIT, MCP_METRICS_CAP, TAG_RECORD_MAX_BATCH, USER_TAG_MAX_LEN,
};

use std::path::Path;

use rusqlite::functions::FunctionFlags;
use rusqlite::Connection;
use thiserror::Error;

/// Embedded SQL migrations. Add new files in `migrations/` and list them here
/// in order.
const MIGRATIONS: &[(&str, &str)] = &[
    ("0001_init", include_str!("migrations/0001_init.sql")),
    ("0002_phase1", include_str!("migrations/0002_phase1.sql")),
    ("0003_cjk_fts", include_str!("migrations/0003_cjk_fts.sql")),
    (
        "0004_provenance_derived_from",
        include_str!("migrations/0004_provenance_derived_from.sql"),
    ),
    (
        "0005_vec_index",
        include_str!("migrations/0005_vec_index.sql"),
    ),
    (
        "0006_mcp_request_metrics",
        include_str!("migrations/0006_mcp_request_metrics.sql"),
    ),
    (
        "0007_record_tombstones",
        include_str!("migrations/0007_record_tombstones.sql"),
    ),
    (
        "0008_records_raw_hash_index",
        include_str!("migrations/0008_records_raw_hash_index.sql"),
    ),
    (
        "0009_user_record_tags",
        include_str!("migrations/0009_user_record_tags.sql"),
    ),
    (
        "0010_vec_record_id_metadata",
        include_str!("migrations/0010_vec_record_id_metadata.sql"),
    ),
];

/// Register the `tokenize_cjk(text)` SQLite scalar function on `conn`.
///
/// The function is called by the `chunks_fts` triggers (`0003_cjk_fts`)
/// to turn record content into a jieba-segmented token stream before it
/// hits the FTS index. Must be installed on EVERY connection before any
/// trigger fires — the migration itself sets it up, and `Store::open`
/// re-registers because each fresh `Connection` starts without it.
fn register_cjk_function(conn: &Connection) -> rusqlite::Result<()> {
    conn.create_scalar_function(
        "tokenize_cjk",
        1,
        FunctionFlags::SQLITE_DETERMINISTIC | FunctionFlags::SQLITE_UTF8,
        |ctx| {
            let text: String = ctx.get(0).unwrap_or_default();
            Ok(crate::cjk::tokenize_indexing(&text))
        },
    )
}

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

    /// Invariant we expect SQLite + the migration set to uphold was
    /// violated — e.g. a `provenance.derived_from` chain cycle. These
    /// are loud rather than silent so corruption surfaces fast.
    #[error("store corruption: {0}")]
    Corruption(String),
}

/// Crate result.
pub type Result<T> = std::result::Result<T, StoreError>;

/// Anamnesis storage handle. The underlying SQLite connection is wrapped
/// in a `parking_lot::Mutex` so the type is `Send + Sync` and can be
/// shared across async tasks (the MCP server holds an `Arc<Store>`).
/// All methods take `&self`; the mutex enforces serialised access to the
/// connection.
pub struct Store {
    pub(crate) conn: parking_lot::Mutex<Connection>,
}

impl Store {
    /// Open (or create) a store at the given path and run pending migrations.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        // Must run BEFORE `Connection::open` so sqlite-vec's auto-extension
        // is registered when SQLite materialises the connection (and thus
        // the vec0 module is available on this and every later connection).
        vec_ext::register()?;
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        register_cjk_function(&conn)?;
        let store = Self {
            conn: parking_lot::Mutex::new(conn),
        };
        store.run_migrations()?;
        store.reindex_fts_if_pending()?;
        store.backfill_vec_if_pending()?;
        Ok(store)
    }

    /// Open an in-memory store (useful for tests).
    pub fn open_in_memory() -> Result<Self> {
        vec_ext::register()?;
        let conn = Connection::open_in_memory()?;
        register_cjk_function(&conn)?;
        let store = Self {
            conn: parking_lot::Mutex::new(conn),
        };
        store.run_migrations()?;
        store.reindex_fts_if_pending()?;
        store.backfill_vec_if_pending()?;
        Ok(store)
    }

    /// If migration 0003 set the `chunks_fts_rebuild_pending` flag,
    /// re-tokenize and re-insert every row from `record_chunks` into
    /// `chunks_fts`, then clear the flag.
    ///
    /// This is the second half of the 0003 migration: the SQL part can
    /// only drop the FTS data, because `tokenize_cjk` is per-connection
    /// and not guaranteed to be installed at migration time on a fresh
    /// DB. Doing the rebuild here keeps the work idempotent (no flag →
    /// no-op) and bounded (flagged once per DB lifetime).
    fn reindex_fts_if_pending(&self) -> Result<()> {
        let pending: Option<String> = {
            let conn = self.conn.lock();
            conn.query_row(
                "SELECT value FROM meta WHERE key = 'chunks_fts_rebuild_pending'",
                [],
                |r| r.get(0),
            )
            .ok()
        };
        if pending.as_deref() != Some("1") {
            return Ok(());
        }
        tracing::info!("0003_cjk_fts: re-tokenising existing record_chunks into chunks_fts");
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        // Wipe whatever's in chunks_fts (external-content mode means
        // there's no automatic clear when triggers re-insert).
        tx.execute(
            "INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')",
            [],
        )?;
        // Re-insert each chunk through the new tokenize_cjk trigger.
        // We do it via UPDATE-noop on record_chunks so the AFTER UPDATE
        // trigger fires consistently, which avoids encoding the
        // tokenization logic in two places.
        let n: usize = tx.execute(
            "INSERT INTO chunks_fts(rowid, content)
             SELECT rowid, tokenize_cjk(content) FROM record_chunks",
            [],
        )?;
        tx.execute(
            "DELETE FROM meta WHERE key = 'chunks_fts_rebuild_pending'",
            [],
        )?;
        tx.commit()?;
        tracing::info!(reindexed_rows = n, "0003_cjk_fts: chunks_fts rebuilt");
        Ok(())
    }

    /// One-shot: populate the per-dim vec0 tables from existing
    /// `chunk_embeddings` BLOB rows the first time a PR-67a binary
    /// opens a pre-PR-67a database. No-op on subsequent opens.
    fn backfill_vec_if_pending(&self) -> Result<()> {
        let mut conn = self.conn.lock();
        vec_ext::backfill_if_pending(&mut conn)?;
        Ok(())
    }

    fn run_migrations(&self) -> Result<()> {
        let mut conn = self.conn.lock();
        // Tiny home-grown runner: keep applied migration ids in a meta table.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS _migrations (
                id    TEXT PRIMARY KEY,
                applied_at INTEGER NOT NULL
            );",
        )?;

        for (id, sql) in MIGRATIONS {
            let already: i64 = conn.query_row(
                "SELECT COUNT(1) FROM _migrations WHERE id = ?1",
                [id],
                |r| r.get(0),
            )?;
            if already == 0 {
                let tx = conn.transaction()?;
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

    /// Borrow the inner connection. Intended for tests and ad-hoc reads;
    /// production code should call the typed methods in `api`. The
    /// returned guard holds the mutex — drop it before any `.await`.
    pub fn conn(&self) -> parking_lot::MutexGuard<'_, Connection> {
        self.conn.lock()
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
        assert_eq!(version, "9");
    }

    #[test]
    fn phase1_tables_exist() {
        let store = Store::open_in_memory().unwrap();
        for table in [
            "sources",
            "raw_artifacts",
            "record_chunks",
            "chunks_fts",
            "chunk_embeddings",
            "embedding_jobs",
            "import_errors",
        ] {
            let n: i64 = store
                .conn()
                .query_row(
                    "SELECT COUNT(1) FROM sqlite_master WHERE name = ?1",
                    [table],
                    |r| r.get(0),
                )
                .unwrap_or_else(|_| panic!("query failed for {table}"));
            assert_eq!(n, 1, "expected table/view {table} to exist");
        }
    }

    #[test]
    fn record_level_fts_was_dropped() {
        let store = Store::open_in_memory().unwrap();
        let n: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(1) FROM sqlite_master WHERE name = 'records_fts'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 0, "records_fts should not exist after 0002");
    }

    #[test]
    fn chunks_fts_is_maintained_by_triggers() {
        let store = Store::open_in_memory().unwrap();
        let conn = store.conn();

        // Insert a parent record so the FK on record_chunks is satisfied.
        conn.execute(
            "INSERT INTO records(id, adapter, instance, content, scope, kind, \
             created_at, native_id, captured_at, raw_hash) \
             VALUES('r1','claude-code',NULL,'parent','user','fact',0,'n1',0,'h')",
            [],
        )
        .unwrap();

        // Insert a chunk → AFTER INSERT trigger should populate FTS.
        conn.execute(
            "INSERT INTO record_chunks(id, record_id, seq, content, content_hash, token_estimate) \
             VALUES('r1:0','r1',0,'hello world','h0',2)",
            [],
        )
        .unwrap();

        let hits: i64 = conn
            .query_row(
                "SELECT COUNT(1) FROM chunks_fts WHERE chunks_fts MATCH 'hello'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hits, 1, "FTS should index inserted chunk content");

        // Delete the chunk → AFTER DELETE trigger should clean FTS.
        conn.execute("DELETE FROM record_chunks WHERE id = 'r1:0'", [])
            .unwrap();
        let hits: i64 = conn
            .query_row(
                "SELECT COUNT(1) FROM chunks_fts WHERE chunks_fts MATCH 'hello'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hits, 0, "FTS should drop entry on chunk delete");
    }

    #[test]
    fn embedding_jobs_unique_per_chunk_and_model() {
        let store = Store::open_in_memory().unwrap();
        let conn = store.conn();
        conn.execute(
            "INSERT INTO records(id, adapter, instance, content, scope, kind, \
             created_at, native_id, captured_at, raw_hash) \
             VALUES('r1','claude-code',NULL,'p','user','fact',0,'n1',0,'h')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO record_chunks(id, record_id, seq, content, content_hash, token_estimate) \
             VALUES('r1:0','r1',0,'x','h0',1)",
            [],
        )
        .unwrap();

        let ok = conn.execute(
            "INSERT INTO embedding_jobs(chunk_id, content_hash, model_id, status, enqueued_at) \
             VALUES('r1:0','h0','local:e5:1','pending',0)",
            [],
        );
        assert!(ok.is_ok());

        // Same (chunk_id, model_id) should violate UNIQUE.
        let dup = conn.execute(
            "INSERT INTO embedding_jobs(chunk_id, content_hash, model_id, status, enqueued_at) \
             VALUES('r1:0','h0','local:e5:1','pending',1)",
            [],
        );
        assert!(dup.is_err());

        // Different model_id → fresh job is allowed.
        let other = conn.execute(
            "INSERT INTO embedding_jobs(chunk_id, content_hash, model_id, status, enqueued_at) \
             VALUES('r1:0','h0','local:bge-m3:1','pending',2)",
            [],
        );
        assert!(other.is_ok());
    }

    #[test]
    fn cascade_delete_record_clears_chunks_and_artifacts() {
        let store = Store::open_in_memory().unwrap();
        let conn = store.conn();
        conn.execute(
            "INSERT INTO records(id, adapter, instance, content, scope, kind, \
             created_at, native_id, captured_at, raw_hash) \
             VALUES('r1','claude-code',NULL,'p','user','fact',0,'n1',0,'h')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO raw_artifacts(record_id, payload_json, captured_at) \
             VALUES('r1','{}',0)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO record_chunks(id, record_id, seq, content, content_hash, token_estimate) \
             VALUES('r1:0','r1',0,'x','h0',1)",
            [],
        )
        .unwrap();

        conn.execute("DELETE FROM records WHERE id = 'r1'", [])
            .unwrap();

        let c: i64 = conn
            .query_row("SELECT COUNT(1) FROM record_chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(c, 0, "chunks should cascade-delete with parent record");
        let a: i64 = conn
            .query_row("SELECT COUNT(1) FROM raw_artifacts", [], |r| r.get(0))
            .unwrap();
        assert_eq!(a, 0, "artifacts should cascade-delete with parent record");
    }
}
