//! sqlite-vec extension wiring + per-dim `vec0` table management.
//!
//! Round 67 PR-67a moves vector search from full-table BLOB scan +
//! Rust-side cosine into sqlite-vec's bundled C `vec0` virtual table.
//! Three responsibilities live here:
//!
//! 1. **Extension registration** (`register`) — installs sqlite-vec as a
//!    process-wide auto-extension so every subsequent `Connection::open*`
//!    has `vec0` / `vec_distance_cosine` available without per-connection
//!    setup.
//! 2. **Per-dim table naming + DDL** (`vec_table_name`, `ensure_vec_table`)
//!    — vec0 bakes the embedding dimension into the schema (`float[N]`),
//!    so we generate one virtual table per observed dim rather than one
//!    per model.
//! 3. **Backfill from BLOB → vec0** (`backfill_if_pending`) — first time
//!    a binary with PR-67a opens a pre-PR-67a database, populate the
//!    new vec0 tables from the existing `chunk_embeddings` BLOB rows.
//!
//! The BLOB column in `chunk_embeddings` remains the source of truth.
//! vec0 is a rebuildable index — wipe the registry + flag, set the
//! `chunk_vec_index_backfill_pending` meta key, and the next open
//! rebuilds.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Once;

use rusqlite::{params, Connection, OptionalExtension, Transaction};

/// Composite-key separator: ASCII unit separator (0x1F). Used inside
/// `vec_key = model_id || "\u{1f}" || chunk_id` so a single chunk can be
/// indexed by multiple models in the same per-dim table.
const KEY_SEP: char = '\u{1f}';

static INIT: Once = Once::new();
static INIT_RC: AtomicI32 = AtomicI32::new(rusqlite::ffi::SQLITE_OK);

/// Register sqlite-vec as a SQLite auto-extension for this process.
///
/// Safe to call multiple times; the actual registration runs exactly once.
/// Returns an error if the underlying SQLite C call reports a failure.
pub fn register() -> rusqlite::Result<()> {
    INIT.call_once(|| {
        // SAFETY:
        // - `sqlite3_vec_init` is the canonical entrypoint exposed by the
        //   sqlite-vec crate's bundled C source and matches SQLite's
        //   `xEntryPoint` ABI.
        // - `sqlite3_auto_extension` is documented as thread-safe and is
        //   guarded by `Once::call_once` so it runs at most once per
        //   process anyway.
        // - The transmute strips the function's typed signature to the
        //   opaque `xEntryPoint` pointer SQLite expects; this is the
        //   pattern recommended by sqlite-vec's official Rust docs at
        //   https://alexgarcia.xyz/sqlite-vec/rust.html.
        #[allow(unsafe_code)]
        let rc = unsafe {
            type EntryPoint = unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut std::os::raw::c_char,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> std::os::raw::c_int;
            rusqlite::ffi::sqlite3_auto_extension(Some(
                std::mem::transmute::<*const (), EntryPoint>(
                    sqlite_vec::sqlite3_vec_init as *const (),
                ),
            ))
        };
        INIT_RC.store(rc, Ordering::SeqCst);
    });

    match INIT_RC.load(Ordering::SeqCst) {
        rusqlite::ffi::SQLITE_OK => Ok(()),
        code => Err(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(code),
            Some("sqlite-vec auto-extension registration failed".into()),
        )),
    }
}

/// Canonical name for the per-dim vec0 table.
pub fn vec_table_name(dim: i64) -> String {
    format!("chunk_embeddings_vec_d{dim}")
}

/// Build the composite key stored as `vec_key` in vec0 tables. Allows
/// one chunk to be indexed under multiple models in the same per-dim
/// table (the `(chunk_id, model_id)` PK in `chunk_embeddings` already
/// enforces uniqueness on the BLOB side).
pub fn vec_key(model_id: &str, chunk_id: &str) -> String {
    format!("{model_id}{KEY_SEP}{chunk_id}")
}

/// Create the per-dim vec0 virtual table if it doesn't exist yet, and
/// record it in the `chunk_vec_indexes` registry. Idempotent.
pub fn ensure_vec_table(tx: &Transaction<'_>, dim: i64) -> rusqlite::Result<String> {
    let table = vec_table_name(dim);
    // vec0 columns:
    //   vec_key TEXT PRIMARY KEY           — model_id+chunk_id composite
    //   model_id TEXT PARTITION KEY        — pruning during KNN
    //   embedding float[N] distance=cosine — the vector itself
    //   adapter / instance / kind / scope / created_at — metadata,
    //   filterable inside the KNN scan (mirrors `SearchFilter`)
    //   record_id TEXT                     — metadata column added in
    //                                        Round 79 (PR-78b) so
    //                                        `--user-tag` can push
    //                                        `record_id IN (SELECT …
    //                                        FROM user_record_tags …)`
    //                                        inside the KNN MATERIALIZED
    //                                        CTE. Without this column,
    //                                        the filter would have to
    //                                        run post-RRF, which loses
    //                                        minority-tag records under
    //                                        skewed corpora.
    //   +chunk_id TEXT                     — auxiliary, returned but
    //                                        not indexed
    let ddl = format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(
             vec_key    TEXT PRIMARY KEY,
             model_id   TEXT PARTITION KEY,
             embedding  float[{dim}] distance_metric=cosine,
             adapter    TEXT,
             instance   TEXT,
             kind       TEXT,
             scope      TEXT,
             created_at INTEGER,
             record_id  TEXT,
             +chunk_id  TEXT
         );"
    );
    tx.execute_batch(&ddl)?;
    tx.execute(
        "INSERT INTO chunk_vec_indexes(dim, table_name, built_at) \
         VALUES (?1, ?2, strftime('%s','now')) \
         ON CONFLICT(dim) DO UPDATE SET \
             table_name = excluded.table_name, \
             built_at   = excluded.built_at",
        params![dim, table],
    )?;
    Ok(table)
}

/// Look up the vec0 table for `dim`. Returns `None` if no index was ever
/// built for that dim (callers fall back to the BLOB-scan path).
pub fn vec_table_for_dim(conn: &Connection, dim: i64) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT table_name FROM chunk_vec_indexes WHERE dim = ?1",
        params![dim],
        |r| r.get::<_, String>(0),
    )
    .optional()
}

/// Drive the open-time backfill if the `chunk_vec_index_backfill_pending`
/// flag is set. Reads distinct dims from `chunk_embeddings`, materialises
/// each per-dim vec0 table, copies rows over, then clears the flag.
///
/// No-op on subsequent opens because the flag is cleared on success.
/// Idempotent under crash: the registry write + DELETE FROM {table} +
/// INSERT all happen in a single transaction per dim.
pub fn backfill_if_pending(conn: &mut Connection) -> rusqlite::Result<()> {
    let pending: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'chunk_vec_index_backfill_pending'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    if pending.as_deref() != Some("1") {
        return Ok(());
    }
    tracing::info!("0005_vec_index: backfilling vec0 indexes from chunk_embeddings");

    // Distinct dims observed in the BLOB table — these are the ones we
    // need per-dim vec0 tables for. Empty result is fine (fresh DB).
    let dims: Vec<i64> = {
        let mut stmt =
            conn.prepare("SELECT DISTINCT dim FROM chunk_embeddings ORDER BY dim ASC")?;
        let rows = stmt.query_map([], |r| r.get::<_, i64>(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };

    // Enforce: any given model_id must have exactly one dim across the
    // BLOB table. vec0's per-dim partitioning makes a single-model /
    // multi-dim mixture undefined — surface it loudly rather than
    // silently picking a "winner."
    let bad: Vec<(String, i64)> = {
        let mut stmt = conn.prepare(
            "SELECT model_id, COUNT(DISTINCT dim) \
             FROM chunk_embeddings \
             GROUP BY model_id \
             HAVING COUNT(DISTINCT dim) > 1",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    if !bad.is_empty() {
        let names: Vec<String> = bad
            .into_iter()
            .map(|(m, n)| format!("{m} ({n} dims)"))
            .collect();
        return Err(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CONSTRAINT),
            Some(format!(
                "chunk_embeddings rows for these models span multiple dims, vec0 backfill cannot proceed: {}",
                names.join(", ")
            )),
        ));
    }

    for dim in dims {
        let tx = conn.transaction()?;
        let table = vec_table_name(dim);
        // Round 79 (PR-78b): vec0 schema gained a `record_id`
        // metadata column. A pre-PR-78b store will have the old
        // 8-column table on disk; `CREATE VIRTUAL TABLE IF NOT
        // EXISTS` would no-op and the new INSERT would fail
        // because `record_id` isn't part of that schema. DROP
        // first, then recreate with the current schema. Safe
        // because vec0 is a rebuildable index — the BLOB column
        // in `chunk_embeddings` is the source of truth.
        tx.execute_batch(&format!("DROP TABLE IF EXISTS {table};"))?;
        let table = ensure_vec_table(&tx, dim)?;
        let inserted = tx.execute(
            &format!(
                "INSERT INTO {table}( \
                     vec_key, model_id, embedding, \
                     adapter, instance, kind, scope, created_at, \
                     record_id, chunk_id) \
                 SELECT \
                     e.model_id || char(31) || e.chunk_id, \
                     e.model_id, \
                     e.embedding, \
                     r.adapter, \
                     COALESCE(r.instance, ''), \
                     r.kind, \
                     r.scope, \
                     r.created_at, \
                     rc.record_id, \
                     e.chunk_id \
                 FROM chunk_embeddings e \
                 JOIN record_chunks rc ON rc.id = e.chunk_id \
                 JOIN records r        ON r.id  = rc.record_id \
                 WHERE e.dim = ?1 \
                   AND length(e.embedding) = ?1 * 4"
            ),
            params![dim],
        )?;
        tx.commit()?;
        tracing::info!(
            dim,
            table = %table,
            inserted,
            "0005_vec_index: backfilled per-dim vec0"
        );
    }

    conn.execute(
        "DELETE FROM meta WHERE key = 'chunk_vec_index_backfill_pending'",
        [],
    )?;
    Ok(())
}

/// Sync helper for `complete_job` / `complete_jobs_batch`: keep the
/// per-dim vec0 row in step with the BLOB write. Called *after* the
/// `chunk_embeddings` row is in place.
///
/// Handles the dim-change case (rare — same chunk re-embedded under a
/// model whose dim has been switched out): the old vec_key is deleted
/// from whichever per-dim table held it, then the new row is inserted
/// into the dim-appropriate table.
pub fn upsert_vec_row(
    tx: &Transaction<'_>,
    chunk_id: &str,
    model_id: &str,
    dim: i64,
    embedding: &[u8],
) -> rusqlite::Result<()> {
    let new_key = vec_key(model_id, chunk_id);

    // Drop any stale rows for this (chunk_id, model_id) across *all*
    // dim tables. Cheap because chunk_vec_indexes is tiny (one row per
    // dim, typically 1–3 entries).
    let other_tables: Vec<(i64, String)> = {
        let mut stmt =
            tx.prepare("SELECT dim, table_name FROM chunk_vec_indexes WHERE dim != ?1")?;
        let rows = stmt.query_map(params![dim], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    for (_d, t) in other_tables {
        tx.execute(
            &format!("DELETE FROM {t} WHERE vec_key = ?1"),
            params![new_key],
        )?;
    }

    let table = ensure_vec_table(tx, dim)?;
    // Virtual-table UPSERT semantics aren't reliable, so do delete-
    // then-insert. The composite PK on `vec_key` makes both ops O(1).
    tx.execute(
        &format!("DELETE FROM {table} WHERE vec_key = ?1"),
        params![new_key],
    )?;
    tx.execute(
        &format!(
            "INSERT INTO {table}( \
                 vec_key, model_id, embedding, \
                 adapter, instance, kind, scope, created_at, \
                 record_id, chunk_id) \
             SELECT \
                 ?1, ?2, ?3, \
                 r.adapter, COALESCE(r.instance, ''), r.kind, r.scope, r.created_at, \
                 rc.record_id, ?4 \
             FROM record_chunks rc \
             JOIN records r ON r.id = rc.record_id \
             WHERE rc.id = ?4"
        ),
        params![new_key, model_id, embedding, chunk_id],
    )?;
    Ok(())
}

/// Sync helper for `write_chunks`: before the FK cascade deletes BLOB
/// rows, manually delete the matching vec_keys from every per-dim
/// vec0 table. Vec0 is a virtual table with no FK cascade.
pub fn delete_vec_rows_for_record(tx: &Transaction<'_>, record_id: &str) -> rusqlite::Result<()> {
    let pairs: Vec<(String, String)> = {
        let mut stmt = tx.prepare(
            "SELECT e.chunk_id, e.model_id \
             FROM chunk_embeddings e \
             JOIN record_chunks rc ON rc.id = e.chunk_id \
             WHERE rc.record_id = ?1",
        )?;
        let rows = stmt.query_map(params![record_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    if pairs.is_empty() {
        return Ok(());
    }
    let tables: Vec<String> = {
        let mut stmt = tx.prepare("SELECT table_name FROM chunk_vec_indexes")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    for (chunk_id, model_id) in &pairs {
        let key = vec_key(model_id, chunk_id);
        for t in &tables {
            tx.execute(&format!("DELETE FROM {t} WHERE vec_key = ?1"), params![key])?;
        }
    }
    Ok(())
}

/// Round 83 (PR-78e): count vec0 rows that `delete_vec_rows_for_record`
/// would actually remove for `record_id`. Uses the **same**
/// `(chunk_id, model_id) → vec_key` traversal as the delete path so
/// the dry-run preview can't drift from what the real delete will do.
///
/// Returns 0 when no chunks for the record carry embeddings (e.g.
/// the embedder job is still pending) — exactly what the delete
/// path would no-op on too.
pub fn count_vec_rows_for_record(tx: &Transaction<'_>, record_id: &str) -> rusqlite::Result<u64> {
    let pairs: Vec<(String, String)> = {
        let mut stmt = tx.prepare(
            "SELECT e.chunk_id, e.model_id \
             FROM chunk_embeddings e \
             JOIN record_chunks rc ON rc.id = e.chunk_id \
             WHERE rc.record_id = ?1",
        )?;
        let rows = stmt.query_map(params![record_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    if pairs.is_empty() {
        return Ok(0);
    }
    let tables: Vec<String> = {
        let mut stmt = tx.prepare("SELECT table_name FROM chunk_vec_indexes")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    let mut total: u64 = 0;
    for (chunk_id, model_id) in &pairs {
        let key = vec_key(model_id, chunk_id);
        for t in &tables {
            let n: i64 = tx.query_row(
                &format!("SELECT COUNT(*) FROM {t} WHERE vec_key = ?1"),
                params![key],
                |r| r.get(0),
            )?;
            total += n as u64;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec0_virtual_table_is_available_after_register() {
        register().unwrap();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE smoke USING vec0(embedding float[3]);
             INSERT INTO smoke(rowid, embedding) VALUES (1, X'0000803F0000003F00000040');",
        )
        .unwrap();
        let count: i64 = conn
            .query_row("SELECT count(*) FROM smoke", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn vec_key_is_unit_separator_composite() {
        assert_eq!(
            vec_key("openai:emb-3:1", "rec:0"),
            "openai:emb-3:1\u{1f}rec:0"
        );
    }

    #[test]
    fn vec_table_name_is_dim_suffixed() {
        assert_eq!(vec_table_name(384), "chunk_embeddings_vec_d384");
        assert_eq!(vec_table_name(3072), "chunk_embeddings_vec_d3072");
    }
}
