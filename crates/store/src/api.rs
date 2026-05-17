//! Typed read/write API over the SQLite store.
//!
//! Everything that touches the database goes through this module. `Store`
//! itself owns the `Connection`; callers must never write SQL directly.

use anamnesis_core::chunk::{Chunk, ContentHash};
use anamnesis_core::model::{AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor};
use chrono::{DateTime, TimeZone, Utc};
use rusqlite::{params, OptionalExtension, Transaction};

use crate::{Result, Store, StoreError};

// ─────────────────────────────────────────────────────────────────────────────
// Conversion helpers
// ─────────────────────────────────────────────────────────────────────────────

fn ts(dt: DateTime<Utc>) -> i64 {
    dt.timestamp()
}

fn dt(ts: i64) -> DateTime<Utc> {
    Utc.timestamp_opt(ts, 0).single().unwrap_or_else(Utc::now)
}

fn scope_str(s: Scope) -> &'static str {
    match s {
        Scope::User => "user",
        Scope::Project => "project",
        Scope::Session => "session",
        Scope::Ephemeral => "ephemeral",
    }
}

fn scope_from(s: &str) -> Scope {
    match s {
        "user" => Scope::User,
        "project" => Scope::Project,
        "session" => Scope::Session,
        "ephemeral" => Scope::Ephemeral,
        _ => Scope::Ephemeral,
    }
}

fn kind_str(k: Kind) -> &'static str {
    match k {
        Kind::Fact => "fact",
        Kind::Preference => "preference",
        Kind::Feedback => "feedback",
        Kind::Reference => "reference",
        Kind::Episode => "episode",
        Kind::Skill => "skill",
        Kind::Unknown => "unknown",
    }
}

fn kind_from(s: &str) -> Kind {
    match s {
        "fact" => Kind::Fact,
        "preference" => Kind::Preference,
        "feedback" => Kind::Feedback,
        "reference" => Kind::Reference,
        "episode" => Kind::Episode,
        "skill" => Kind::Skill,
        _ => Kind::Unknown,
    }
}

/// Serialize a `Vec<f32>` to little-endian bytes.
pub fn f32_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Parse a little-endian f32 blob back into a vector.
pub fn blob_to_f32(b: &[u8]) -> Result<Vec<f32>> {
    if b.len() % 4 != 0 {
        return Err(StoreError::Sqlite(rusqlite::Error::InvalidQuery));
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..a.len() {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

// ─────────────────────────────────────────────────────────────────────────────
// Returned shapes
// ─────────────────────────────────────────────────────────────────────────────

/// A chunk-level search hit.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkHit {
    /// Synthetic chunk id (`"{record_id}:{seq}"`).
    pub chunk_id: String,
    /// Parent record.
    pub record_id: RecordId,
    /// Per-record sequence.
    pub seq: u32,
    /// The matched chunk content.
    pub content: String,
    /// Score in the search-specific scale (FTS: bm25 rank, vector: cosine).
    pub score: f64,
}

/// A claimed embedding job.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingEmbeddingJob {
    /// Surrogate primary key from `embedding_jobs.id`.
    pub job_id: i64,
    /// Chunk to embed.
    pub chunk_id: String,
    /// `blake3` of the chunk content.
    pub content_hash: ContentHash,
    /// The model the embedding must be produced under.
    pub model_id: String,
    /// The chunk's text — included so the worker doesn't need a second query.
    pub content: String,
}

/// Coarse counters for `anamnesis status`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreStats {
    /// Records in `records` table.
    pub records: u64,
    /// Chunks in `record_chunks`.
    pub chunks: u64,
    /// Pending or in-progress embedding jobs.
    pub jobs_pending: u64,
    /// Failed embedding jobs (terminal state).
    pub jobs_failed: u64,
    /// Distinct `(adapter, instance)` source rows.
    pub sources: u64,
}

/// Full row from `sources` — what `list_sources_full` and `get_source`
/// return. The legacy `list_sources` 3-tuple shape stays for back-compat.
///
/// `instance` is the empty string `""` (NOT `None`) to represent the
/// default instance — that's the canonical key the table uses (see
/// `0002_phase1.sql`). Callers that work in `Option<String>` must convert
/// at the boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRow {
    /// Adapter id (e.g. `"claude-code"`).
    pub adapter: String,
    /// Instance discriminator — `""` for the default instance.
    pub instance: String,
    /// User-registered location (path / URL / connection string). `None`
    /// when registered without one — `import` will fall back to the
    /// adapter default and register that as the canonical location.
    pub location: Option<String>,
    /// JSON-encoded adapter-specific config, opaque to the store.
    pub config_json: Option<String>,
    /// Unix epoch seconds — when the source was first registered.
    pub added_at: i64,
    /// Unix epoch seconds — when the last successful (non-dry-run)
    /// import finished. `None` until the first import lands.
    pub last_import_at: Option<i64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Source registry
// ─────────────────────────────────────────────────────────────────────────────

impl Store {
    /// Register or update a memory source. Idempotent.
    ///
    /// `instance = None` is stored as the empty string `""` because the
    /// `sources` table uses NOT NULL DEFAULT '' on that column; matching
    /// against `NULL` would silently miss the row.
    pub fn register_source(
        &self,
        adapter: &str,
        instance: Option<&str>,
        location: Option<&str>,
        config_json: Option<&str>,
    ) -> Result<()> {
        let inst = instance.unwrap_or("");
        self.conn.lock().execute(
            "INSERT INTO sources(adapter, instance, location, config_json, added_at) \
             VALUES(?1, ?2, ?3, ?4, strftime('%s','now')) \
             ON CONFLICT(adapter, instance) DO UPDATE SET \
               location = excluded.location, \
               config_json = excluded.config_json",
            params![adapter, inst, location, config_json],
        )?;
        Ok(())
    }

    /// Look up a single source row by `(adapter, instance)`.
    ///
    /// Returns `None` if no row exists. `instance = None` is normalised to
    /// the empty string for the lookup (see `register_source` rationale).
    pub fn get_source(&self, adapter: &str, instance: Option<&str>) -> Result<Option<SourceRow>> {
        let inst = instance.unwrap_or("");
        let conn = self.conn.lock();
        let row = conn
            .query_row(
                "SELECT adapter, instance, location, config_json, added_at, last_import_at \
                 FROM sources WHERE adapter = ?1 AND instance = ?2",
                params![adapter, inst],
                |r| {
                    Ok(SourceRow {
                        adapter: r.get(0)?,
                        instance: r.get(1)?,
                        location: r.get(2)?,
                        config_json: r.get(3)?,
                        added_at: r.get(4)?,
                        last_import_at: r.get(5)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Stamp `last_import_at` for a source.
    ///
    /// Returns `Ok(true)` when the source existed and was updated, `Ok(false)`
    /// when no matching row exists (the caller should usually
    /// `register_source` first so this can never be `false` on the happy
    /// path).
    pub fn update_last_import_at(&self, adapter: &str, instance: Option<&str>) -> Result<bool> {
        let inst = instance.unwrap_or("");
        let n = self.conn.lock().execute(
            "UPDATE sources SET last_import_at = strftime('%s','now') \
             WHERE adapter = ?1 AND instance = ?2",
            params![adapter, inst],
        )?;
        Ok(n > 0)
    }

    /// Like `list_sources` but returns the full row shape including
    /// `added_at` and `last_import_at`. Newer code should prefer this; the
    /// 3-tuple `list_sources` stays for back-compat with existing callers.
    pub fn list_sources_full(&self) -> Result<Vec<SourceRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT adapter, instance, location, config_json, added_at, last_import_at \
             FROM sources ORDER BY adapter, instance",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok(SourceRow {
                    adapter: r.get(0)?,
                    instance: r.get(1)?,
                    location: r.get(2)?,
                    config_json: r.get(3)?,
                    added_at: r.get(4)?,
                    last_import_at: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Forget a source. Does NOT cascade-delete records (those keep their
    /// own provenance and can be inspected even after the source is gone).
    pub fn deregister_source(&self, adapter: &str, instance: Option<&str>) -> Result<()> {
        let inst = instance.unwrap_or("");
        self.conn.lock().execute(
            "DELETE FROM sources WHERE adapter = ?1 AND instance = ?2",
            params![adapter, inst],
        )?;
        Ok(())
    }

    /// List configured sources as `(adapter, instance, location)` triples.
    pub fn list_sources(&self) -> Result<Vec<(String, String, Option<String>)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT adapter, instance, location FROM sources ORDER BY adapter, instance",
        )?;
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Active embedding model (single-writer config knob)
// ─────────────────────────────────────────────────────────────────────────────

impl Store {
    /// Set the active embedding model. New chunks will enqueue jobs against
    /// this model. Switching models does NOT retroactively rebuild
    /// embeddings; callers (the CLI `model use` command) decide whether to
    /// also call `rebuild_embedding_jobs`.
    pub fn set_active_model(&self, model_id: &str) -> Result<()> {
        self.conn.lock().execute(
            "INSERT INTO meta(key, value) VALUES('active_embedding_model', ?1) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![model_id],
        )?;
        Ok(())
    }

    /// Returns the active model id, if any.
    pub fn active_model(&self) -> Result<Option<String>> {
        let v: Option<String> = self
            .conn
            .lock()
            .query_row(
                "SELECT value FROM meta WHERE key = 'active_embedding_model'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        Ok(v)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Records + chunks (atomic upsert)
// ─────────────────────────────────────────────────────────────────────────────

impl Store {
    /// Atomically upsert a record, its chunks, and (optionally) its raw
    /// artifact. Old chunks for this record are deleted first so re-chunking
    /// is consistent. Embedding jobs are enqueued for every chunk against
    /// the current active model (if any); duplicates are no-ops.
    ///
    /// Returns `(records_added_or_updated, chunks_written)`. Both counts
    /// are 1/N — meaningful for tests and import job summaries.
    pub fn upsert_record(
        &self,
        record: &AnamnesisRecord,
        chunks: &[Chunk],
        raw_payload_json: Option<&str>,
    ) -> Result<(u64, u64)> {
        let active = self.active_model()?;
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        let now = chrono::Utc::now().timestamp();
        write_record(&tx, record)?;
        write_raw_artifact(&tx, record, raw_payload_json, now)?;
        write_chunks(&tx, record, chunks)?;
        if let Some(model_id) = active.as_deref() {
            enqueue_jobs(&tx, chunks, model_id, now)?;
        }
        tx.commit()?;
        Ok((1, chunks.len() as u64))
    }

    /// Re-enqueue embedding jobs for every chunk under a different model.
    /// Used by `anamnesis model use <other>` to trigger a full re-embed.
    pub fn rebuild_embedding_jobs(&self, model_id: &str) -> Result<u64> {
        let now = chrono::Utc::now().timestamp();
        let n = self.conn.lock().execute(
            "INSERT INTO embedding_jobs(chunk_id, content_hash, model_id, status, enqueued_at) \
             SELECT id, content_hash, ?1, 'pending', ?2 FROM record_chunks \
             WHERE TRUE ON CONFLICT(chunk_id, model_id) DO NOTHING",
            params![model_id, now],
        )?;
        Ok(n as u64)
    }
}

fn write_record(tx: &Transaction<'_>, r: &AnamnesisRecord) -> Result<()> {
    let tags = if r.tags.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&r.tags).unwrap_or_default())
    };
    let metadata = if r.metadata.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&r.metadata).unwrap_or_default())
    };
    tx.execute(
        "INSERT INTO records(\
            id, adapter, instance, content, scope, kind, \
            created_at, updated_at, tags, metadata, \
            native_id, native_path, captured_at, raw_hash, schema_version\
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15) \
         ON CONFLICT(id) DO UPDATE SET \
            content = excluded.content, \
            scope = excluded.scope, \
            kind = excluded.kind, \
            updated_at = excluded.updated_at, \
            tags = excluded.tags, \
            metadata = excluded.metadata, \
            native_path = excluded.native_path, \
            raw_hash = excluded.raw_hash",
        params![
            r.id.0,
            r.source.adapter,
            r.source.instance.as_deref().unwrap_or(""),
            r.content,
            scope_str(r.scope),
            kind_str(r.kind),
            ts(r.created_at),
            r.updated_at.map(ts),
            tags,
            metadata,
            r.provenance.native_id,
            r.provenance.native_path,
            ts(r.provenance.captured_at),
            r.provenance.raw_hash,
            r.schema_version,
        ],
    )?;
    Ok(())
}

fn write_raw_artifact(
    tx: &Transaction<'_>,
    r: &AnamnesisRecord,
    payload_json: Option<&str>,
    now: i64,
) -> Result<()> {
    // Source vectors are kept for provenance ONLY — never queried.
    let (src_emb, src_model, src_dim) = match &r.embedding {
        Some(e) => (
            Some(f32_to_blob(&e.vector)),
            Some(e.model.clone()),
            Some(e.dim as i64),
        ),
        None => (None, None, None),
    };
    tx.execute(
        "INSERT INTO raw_artifacts(record_id, payload_json, source_embedding, \
            source_embedding_model, source_embedding_dim, captured_at) \
         VALUES(?1, ?2, ?3, ?4, ?5, ?6) \
         ON CONFLICT(record_id) DO UPDATE SET \
            payload_json = excluded.payload_json, \
            source_embedding = excluded.source_embedding, \
            source_embedding_model = excluded.source_embedding_model, \
            source_embedding_dim = excluded.source_embedding_dim, \
            captured_at = excluded.captured_at",
        params![
            r.id.0,
            payload_json,
            src_emb.as_deref(),
            src_model,
            src_dim,
            now,
        ],
    )?;
    Ok(())
}

fn write_chunks(tx: &Transaction<'_>, r: &AnamnesisRecord, chunks: &[Chunk]) -> Result<()> {
    // Re-chunking is a clean replace.
    tx.execute(
        "DELETE FROM record_chunks WHERE record_id = ?1",
        params![r.id.0],
    )?;
    for c in chunks {
        let cid = format!("{}:{}", c.record_id.0, c.seq);
        tx.execute(
            "INSERT INTO record_chunks(id, record_id, seq, content, content_hash, token_estimate) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                cid,
                c.record_id.0,
                c.seq,
                c.content,
                c.content_hash.0,
                c.token_estimate
            ],
        )?;
    }
    Ok(())
}

fn enqueue_jobs(tx: &Transaction<'_>, chunks: &[Chunk], model_id: &str, now: i64) -> Result<()> {
    for c in chunks {
        let cid = format!("{}:{}", c.record_id.0, c.seq);
        tx.execute(
            "INSERT INTO embedding_jobs(chunk_id, content_hash, model_id, status, enqueued_at) \
             VALUES(?1, ?2, ?3, 'pending', ?4) \
             ON CONFLICT(chunk_id, model_id) DO NOTHING",
            params![cid, c.content_hash.0, model_id, now],
        )?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Record reads
// ─────────────────────────────────────────────────────────────────────────────

impl Store {
    /// Fetch a record by id.
    pub fn get_record(&self, id: &RecordId) -> Result<Option<AnamnesisRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, adapter, instance, content, scope, kind, \
                    created_at, updated_at, tags, metadata, \
                    native_id, native_path, captured_at, raw_hash, schema_version \
             FROM records WHERE id = ?1",
        )?;
        let row = stmt.query_row(params![id.0], record_from_row).optional()?;
        Ok(row)
    }
}

fn record_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AnamnesisRecord> {
    let tags_json: Option<String> = row.get(8)?;
    let meta_json: Option<String> = row.get(9)?;
    let updated_at: Option<i64> = row.get(7)?;
    let instance: String = row.get(2)?;
    Ok(AnamnesisRecord {
        id: RecordId(row.get(0)?),
        source: SourceDescriptor {
            adapter: row.get(1)?,
            instance: if instance.is_empty() {
                None
            } else {
                Some(instance)
            },
            version: String::new(), // store doesn't track adapter self-version
        },
        content: row.get(3)?,
        embedding: None, // source vectors live in raw_artifacts (provenance only)
        scope: scope_from(&row.get::<_, String>(4)?),
        kind: kind_from(&row.get::<_, String>(5)?),
        created_at: dt(row.get(6)?),
        updated_at: updated_at.map(dt),
        tags: tags_json
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default(),
        metadata: meta_json
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default(),
        provenance: Provenance {
            native_id: row.get(10)?,
            native_path: row.get(11)?,
            captured_at: dt(row.get(12)?),
            raw_hash: row.get(13)?,
        },
        schema_version: row.get::<_, i64>(14)? as u32,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Search
// ─────────────────────────────────────────────────────────────────────────────

impl Store {
    /// FTS5 chunk search. Returns hits ordered by BM25 (lower rank = better);
    /// `score` is the bm25() value (negated so larger = more relevant).
    pub fn search_chunks_fts(&self, query: &str, limit: u32) -> Result<Vec<ChunkHit>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT rc.id, rc.record_id, rc.seq, rc.content, bm25(chunks_fts) AS score \
             FROM chunks_fts \
             JOIN record_chunks rc ON rc.rowid = chunks_fts.rowid \
             WHERE chunks_fts MATCH ?1 \
             ORDER BY score \
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![query, limit], |r| {
                let raw_score: f64 = r.get(4)?;
                Ok(ChunkHit {
                    chunk_id: r.get(0)?,
                    record_id: RecordId(r.get(1)?),
                    seq: r.get::<_, i64>(2)? as u32,
                    content: r.get(3)?,
                    score: -raw_score, // bm25 returns negative-ish; flip so > is better
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Linear-scan vector search over `chunk_embeddings` filtered by
    /// `model_id`. Acceptable for Phase-1 corpora (<100k chunks per
    /// BLUEPRINT §12). sqlite-vec swap-in lives behind the same API.
    pub fn search_chunks_vec(
        &self,
        query_vec: &[f32],
        model_id: &str,
        limit: u32,
    ) -> Result<Vec<ChunkHit>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT e.chunk_id, e.embedding, rc.record_id, rc.seq, rc.content \
             FROM chunk_embeddings e \
             JOIN record_chunks rc ON rc.id = e.chunk_id \
             WHERE e.model_id = ?1",
        )?;
        let mut scored: Vec<ChunkHit> = Vec::new();
        let rows = stmt.query_map(params![model_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Vec<u8>>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, String>(4)?,
            ))
        })?;
        for row in rows {
            let (chunk_id, blob, rid, seq, content) = row?;
            let v = blob_to_f32(&blob)?;
            let score = cosine(query_vec, &v);
            scored.push(ChunkHit {
                chunk_id,
                record_id: RecordId(rid),
                seq: seq as u32,
                content,
                score,
            });
        }
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit as usize);
        Ok(scored)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding job queue
// ─────────────────────────────────────────────────────────────────────────────

impl Store {
    /// Atomically claim one pending job (pending → in_progress).
    /// Returns `None` when the queue is empty.
    pub fn claim_next_job(&self, model_id: &str) -> Result<Option<PendingEmbeddingJob>> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        let now = chrono::Utc::now().timestamp();
        let row: Option<(i64, String, String)> = tx
            .query_row(
                "SELECT id, chunk_id, content_hash FROM embedding_jobs \
                 WHERE status = 'pending' AND model_id = ?1 \
                 ORDER BY enqueued_at ASC LIMIT 1",
                params![model_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let Some((job_id, chunk_id, content_hash)) = row else {
            tx.commit()?;
            return Ok(None);
        };
        tx.execute(
            "UPDATE embedding_jobs SET status = 'in_progress', claimed_at = ?1 WHERE id = ?2",
            params![now, job_id],
        )?;
        let content: String = tx.query_row(
            "SELECT content FROM record_chunks WHERE id = ?1",
            params![chunk_id],
            |r| r.get(0),
        )?;
        tx.commit()?;
        Ok(Some(PendingEmbeddingJob {
            job_id,
            chunk_id,
            content_hash: ContentHash(content_hash),
            model_id: model_id.to_string(),
            content,
        }))
    }

    /// Mark a job done and persist its embedding.
    pub fn complete_job(&self, job: &PendingEmbeddingJob, vector: &[f32]) -> Result<()> {
        let dim = vector.len() as i64;
        let blob = f32_to_blob(vector);
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        let now = chrono::Utc::now().timestamp();
        tx.execute(
            "INSERT INTO chunk_embeddings(chunk_id, model_id, content_hash, dim, embedding, created_at) \
             VALUES(?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(chunk_id, model_id) DO UPDATE SET \
                content_hash = excluded.content_hash, \
                dim = excluded.dim, \
                embedding = excluded.embedding, \
                created_at = excluded.created_at",
            params![
                job.chunk_id,
                job.model_id,
                job.content_hash.0,
                dim,
                blob,
                now,
            ],
        )?;
        tx.execute(
            "UPDATE embedding_jobs SET status = 'done', finished_at = ?1, error = NULL WHERE id = ?2",
            params![now, job.job_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Mark a job failed; the embedder may retry by re-enqueueing later.
    pub fn fail_job(&self, job_id: i64, error: &str) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        self.conn.lock().execute(
            "UPDATE embedding_jobs SET status = 'failed', finished_at = ?1, error = ?2 WHERE id = ?3",
            params![now, error, job_id],
        )?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Import errors + stats
// ─────────────────────────────────────────────────────────────────────────────

impl Store {
    /// Record a non-fatal per-record import error.
    pub fn log_import_error(
        &self,
        adapter: &str,
        instance: Option<&str>,
        native_id: Option<&str>,
        native_path: Option<&str>,
        phase: &str,
        error: &str,
    ) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        self.conn.lock().execute(
            "INSERT INTO import_errors(adapter, instance, native_id, native_path, phase, error, occurred_at) \
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![adapter, instance.unwrap_or(""), native_id, native_path, phase, error, now],
        )?;
        Ok(())
    }

    /// Coarse counters for `anamnesis status`.
    pub fn stats(&self) -> Result<StoreStats> {
        let conn = self.conn.lock();
        let records: i64 = conn.query_row("SELECT COUNT(1) FROM records", [], |r| r.get(0))?;
        let chunks: i64 = conn.query_row("SELECT COUNT(1) FROM record_chunks", [], |r| r.get(0))?;
        let pending: i64 = conn.query_row(
            "SELECT COUNT(1) FROM embedding_jobs WHERE status IN ('pending','in_progress')",
            [],
            |r| r.get(0),
        )?;
        let failed: i64 = conn.query_row(
            "SELECT COUNT(1) FROM embedding_jobs WHERE status = 'failed'",
            [],
            |r| r.get(0),
        )?;
        let sources: i64 = conn.query_row("SELECT COUNT(1) FROM sources", [], |r| r.get(0))?;
        Ok(StoreStats {
            records: records as u64,
            chunks: chunks as u64,
            jobs_pending: pending as u64,
            jobs_failed: failed as u64,
            sources: sources as u64,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{Embedding, Provenance, SourceDescriptor};
    use chrono::Utc;

    fn make_record(adapter: &str, native_id: &str, content: &str, kind: Kind) -> AnamnesisRecord {
        let id = RecordId::from_parts(adapter, None, native_id);
        AnamnesisRecord {
            id,
            source: SourceDescriptor {
                adapter: adapter.into(),
                instance: None,
                version: "0.0.1".into(),
            },
            content: content.into(),
            embedding: None,
            scope: Scope::User,
            kind,
            created_at: Utc::now(),
            updated_at: None,
            tags: vec!["t1".into(), "t2".into()],
            metadata: Default::default(),
            provenance: Provenance {
                native_id: native_id.into(),
                native_path: Some(format!("/tmp/{native_id}.md")),
                captured_at: Utc::now(),
                raw_hash: "h".into(),
            },
            schema_version: anamnesis_core::SCHEMA_VERSION,
        }
    }

    #[test]
    fn f32_blob_roundtrip() {
        let v = vec![0.1f32, -0.2, 1e10, -1e-10, 0.0];
        let back = blob_to_f32(&f32_to_blob(&v)).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn cosine_basic() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
        assert!((cosine(&[1.0, 1.0], &[1.0, 1.0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn register_and_list_sources() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("claude-code", Some("default"), Some("/home/x"), None)
            .unwrap();
        store
            .register_source(
                "mem0",
                None,
                Some("/tmp/m.db"),
                Some("{\"mode\":\"sqlite\"}"),
            )
            .unwrap();
        let mut got = store.list_sources().unwrap();
        got.sort();
        assert_eq!(
            got,
            vec![
                (
                    "claude-code".into(),
                    "default".into(),
                    Some("/home/x".into())
                ),
                ("mem0".into(), "".into(), Some("/tmp/m.db".into())),
            ]
        );
    }

    // ─── PR-B: SourceRow / get_source / update_last_import_at ───

    #[test]
    fn get_source_normalises_none_instance_to_empty_string() {
        // Codex-flagged gotcha: `sources.instance` is NOT NULL DEFAULT ''.
        // If callers pass instance=None and we lookup with SQL NULL, the
        // row will never be found → silent re-registration. Verify
        // get_source matches the same row register_source(None, ...) wrote.
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("mem0", None, Some("/path/db.sqlite"), None)
            .unwrap();
        let row = store.get_source("mem0", None).unwrap();
        let row = row.expect("instance=None must round-trip via get_source");
        assert_eq!(row.adapter, "mem0");
        assert_eq!(row.instance, "", "default instance stored as empty string");
        assert_eq!(row.location.as_deref(), Some("/path/db.sqlite"));
        assert!(row.last_import_at.is_none());
        // Also: Some("") should not be treated as a distinct instance.
        let row_via_empty = store.get_source("mem0", Some("")).unwrap();
        assert!(row_via_empty.is_some(), "Some(\"\") must hit same row");
    }

    #[test]
    fn get_source_returns_none_for_unregistered() {
        let store = Store::open_in_memory().unwrap();
        assert!(store.get_source("claude-code", None).unwrap().is_none());
        assert!(store
            .get_source("mem0", Some("nonexistent-instance"))
            .unwrap()
            .is_none());
    }

    #[test]
    fn update_last_import_at_stamps_existing_row() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("claude-code", None, Some("/p"), None)
            .unwrap();
        assert!(store
            .get_source("claude-code", None)
            .unwrap()
            .unwrap()
            .last_import_at
            .is_none());
        let updated = store.update_last_import_at("claude-code", None).unwrap();
        assert!(updated, "update returns true when a row was stamped");
        let row = store.get_source("claude-code", None).unwrap().unwrap();
        assert!(
            row.last_import_at.is_some(),
            "last_import_at must be non-null after a successful update"
        );
    }

    #[test]
    fn update_last_import_at_for_missing_row_returns_false() {
        let store = Store::open_in_memory().unwrap();
        let updated = store.update_last_import_at("claude-code", None).unwrap();
        assert!(
            !updated,
            "no matching source row → returns Ok(false) without inserting"
        );
        assert!(store.list_sources().unwrap().is_empty());
    }

    #[test]
    fn register_source_is_idempotent_keeps_added_at_stable() {
        // The trap: a second register_source must NOT insert a new row.
        // ON CONFLICT keeps added_at fixed (it's only set in INSERT).
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("mem0", None, Some("/path/A"), None)
            .unwrap();
        let row1 = store.get_source("mem0", None).unwrap().unwrap();
        std::thread::sleep(std::time::Duration::from_secs(1));
        store
            .register_source("mem0", None, Some("/path/B"), None)
            .unwrap();
        let rows = store.list_sources().unwrap();
        assert_eq!(rows.len(), 1, "no duplicate rows");
        let row2 = store.get_source("mem0", None).unwrap().unwrap();
        assert_eq!(row1.added_at, row2.added_at, "added_at stays stable");
        assert_eq!(row2.location.as_deref(), Some("/path/B"));
    }

    #[test]
    fn list_sources_full_carries_all_fields() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("claude-code", Some("work"), Some("/work"), Some("{}"))
            .unwrap();
        store
            .update_last_import_at("claude-code", Some("work"))
            .unwrap();
        store.register_source("mem0", None, None, None).unwrap(); // location=None is valid

        let rows = store.list_sources_full().unwrap();
        assert_eq!(rows.len(), 2);
        let cc = rows.iter().find(|r| r.adapter == "claude-code").unwrap();
        assert_eq!(cc.instance, "work");
        assert_eq!(cc.location.as_deref(), Some("/work"));
        assert_eq!(cc.config_json.as_deref(), Some("{}"));
        assert!(cc.last_import_at.is_some());
        let mem0 = rows.iter().find(|r| r.adapter == "mem0").unwrap();
        assert_eq!(mem0.instance, "");
        assert!(mem0.location.is_none());
        assert!(mem0.last_import_at.is_none());
    }

    #[test]
    fn upsert_round_trips_record() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("claude-code", "n1", "alpha beta gamma", Kind::Preference);
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        let (added, n_chunks) = store.upsert_record(&r, &chunks, Some("{}")).unwrap();
        assert_eq!(added, 1);
        assert_eq!(n_chunks, 1);
        let back = store.get_record(&r.id).unwrap().unwrap();
        assert_eq!(back.id, r.id);
        assert_eq!(back.content, r.content);
        assert_eq!(back.kind, Kind::Preference);
        assert_eq!(back.scope, Scope::User);
        assert_eq!(back.tags, vec!["t1".to_string(), "t2".to_string()]);
        assert_eq!(back.source.adapter, "claude-code");
        assert!(back.source.instance.is_none());
    }

    #[test]
    fn upsert_replaces_chunks_on_recall() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("a", "x", "v1", Kind::Fact);
        let c1 = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c1, None).unwrap();

        let mut r2 = r.clone();
        r2.content = "v2 different and longer ".repeat(40);
        let c2 = Chunker::default().chunk(&r2.id, &r2.content);
        store.upsert_record(&r2, &c2, None).unwrap();

        let chunk_count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(1) FROM record_chunks WHERE record_id = ?1",
                params![r2.id.0],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(chunk_count as usize, c2.len());
        // FTS index should match v2 content, not v1.
        let hits = store.search_chunks_fts("different", 5).unwrap();
        assert!(!hits.is_empty());
        let stale = store.search_chunks_fts("v1", 5).unwrap();
        assert!(stale.is_empty());
    }

    #[test]
    fn fts_search_returns_chunks() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record(
            "a",
            "x",
            "the quick brown fox jumps over the lazy dog",
            Kind::Fact,
        );
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
        let hits = store.search_chunks_fts("quick fox", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, r.id);
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn active_model_setter_reads_back() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(store.active_model().unwrap(), None);
        store.set_active_model("local:e5:1").unwrap();
        assert_eq!(store.active_model().unwrap().as_deref(), Some("local:e5:1"));
        store.set_active_model("local:bge-m3:1").unwrap();
        assert_eq!(
            store.active_model().unwrap().as_deref(),
            Some("local:bge-m3:1")
        );
    }

    #[test]
    fn upsert_enqueues_jobs_under_active_model() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("local:e5:1").unwrap();
        let r = make_record("a", "x", "hello world", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
        let n: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(1) FROM embedding_jobs WHERE status = 'pending' AND model_id = 'local:e5:1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(n, c.len() as i64);
    }

    #[test]
    fn no_active_model_means_no_jobs() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("a", "x", "hi", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
        let n: i64 = store
            .conn()
            .query_row("SELECT COUNT(1) FROM embedding_jobs", [], |row| row.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn claim_and_complete_job_cycle() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("local:fake:1").unwrap();
        let r = make_record("a", "x", "alpha", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();

        let job = store.claim_next_job("local:fake:1").unwrap().unwrap();
        assert_eq!(job.content, "alpha");
        assert_eq!(job.model_id, "local:fake:1");

        // Same call should now miss (claimed → in_progress).
        let none = store.claim_next_job("local:fake:1").unwrap();
        assert!(none.is_none());

        store.complete_job(&job, &[0.5, 0.5, 0.5, 0.5]).unwrap();

        let pending: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(1) FROM embedding_jobs WHERE status = 'pending'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(pending, 0);

        // Vector search must now find this chunk.
        let hits = store
            .search_chunks_vec(&[0.5, 0.5, 0.5, 0.5], "local:fake:1", 5)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fail_job_marks_failed_and_unblocks_next() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("local:fake:1").unwrap();
        let r1 = make_record("a", "x", "one", Kind::Fact);
        let r2 = make_record("a", "y", "two", Kind::Fact);
        let c1 = Chunker::default().chunk(&r1.id, &r1.content);
        let c2 = Chunker::default().chunk(&r2.id, &r2.content);
        store.upsert_record(&r1, &c1, None).unwrap();
        store.upsert_record(&r2, &c2, None).unwrap();

        let j1 = store.claim_next_job("local:fake:1").unwrap().unwrap();
        store.fail_job(j1.job_id, "boom").unwrap();

        let j2 = store.claim_next_job("local:fake:1").unwrap().unwrap();
        assert_ne!(j2.chunk_id, j1.chunk_id);
    }

    #[test]
    fn rebuild_jobs_targets_a_new_model() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("local:a:1").unwrap();
        let r = make_record("a", "x", "hi", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();

        let n = store.rebuild_embedding_jobs("local:b:1").unwrap();
        assert_eq!(n, c.len() as u64);

        let by_model: Vec<(String, i64)> = {
            let conn = store.conn();
            let mut stmt = conn
                .prepare(
                    "SELECT model_id, COUNT(1) FROM embedding_jobs GROUP BY model_id ORDER BY model_id",
                )
                .unwrap();
            stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap()
        };
        assert_eq!(
            by_model,
            vec![
                ("local:a:1".into(), c.len() as i64),
                ("local:b:1".into(), c.len() as i64),
            ]
        );
    }

    #[test]
    fn stats_reports_counts() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("local:fake:1").unwrap();
        store
            .register_source("claude-code", None, None, None)
            .unwrap();
        let r = make_record("a", "x", "hello", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
        let s = store.stats().unwrap();
        assert_eq!(s.records, 1);
        assert_eq!(s.chunks, c.len() as u64);
        assert_eq!(s.jobs_pending, c.len() as u64);
        assert_eq!(s.jobs_failed, 0);
        assert_eq!(s.sources, 1);
    }

    #[test]
    fn import_error_logged_and_visible() {
        let store = Store::open_in_memory().unwrap();
        store
            .log_import_error("a", None, Some("nid"), Some("/p"), "parse", "bad json")
            .unwrap();
        let count: i64 = store
            .conn()
            .query_row("SELECT COUNT(1) FROM import_errors", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn source_vector_is_persisted_to_raw_artifacts() {
        let store = Store::open_in_memory().unwrap();
        let mut r = make_record("mem0", "x", "hi", Kind::Fact);
        r.embedding = Some(Embedding {
            vector: vec![0.1, 0.2, 0.3],
            model: "openai:text-embedding-3-small".into(),
            dim: 3,
        });
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
        let (blob, model, dim): (Vec<u8>, String, i64) = store
            .conn()
            .query_row(
                "SELECT source_embedding, source_embedding_model, source_embedding_dim \
                 FROM raw_artifacts WHERE record_id = ?1",
                params![r.id.0],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(model, "openai:text-embedding-3-small");
        assert_eq!(dim, 3);
        assert_eq!(blob_to_f32(&blob).unwrap(), vec![0.1, 0.2, 0.3]);
    }
}
