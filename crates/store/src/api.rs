//! Typed read/write API over the SQLite store.
//!
//! Everything that touches the database goes through this module. `Store`
//! itself owns the `Connection`; callers must never write SQL directly.

use std::collections::HashMap;

use anamnesis_core::chunk::{Chunk, ContentHash};
use anamnesis_core::model::{AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor};
use chrono::{DateTime, TimeZone, Utc};
use rusqlite::{params, OptionalExtension, Transaction};

use crate::vec_ext;
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

/// Shared planning result for `accept_native_conflict_variant`.
struct AcceptConflictPlan {
    keep_variant: u32,
    keep_records: Vec<AcceptConflictRecord>,
    forget_records: Vec<AcceptConflictRecord>,
}

/// Forget one record inside an existing tx. Shared by single-record
/// and cascade paths. Captures `derived_from` so unforget can BFS the
/// tombstone subtree. Returns `Forgotten` / `AlreadyForgotten` / `NotFound`.
fn forget_one_in_tx(
    tx: &rusqlite::Transaction<'_>,
    id: &RecordId,
    reason: Option<&str>,
    now: i64,
) -> Result<ForgetRecordOutcome> {
    // (adapter, instance, native_id, native_path, raw_hash, derived_from).
    // Aliased to quiet clippy::type_complexity.
    type LiveCols = (
        String,
        String,
        String,
        Option<String>,
        String,
        Option<String>,
    );
    let live: Option<LiveCols> = tx
        .query_row(
            "SELECT adapter, instance, native_id, native_path, raw_hash, derived_from \
             FROM records WHERE id = ?1",
            params![id.0],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, Option<String>>(5)?,
                ))
            },
        )
        .optional()?;

    if live.is_none() {
        // Maybe already forgotten — same idempotency contract as R72.
        type TombstoneCols = (
            String,
            String,
            String,
            Option<String>,
            String,
            Option<String>,
            i64,
        );
        let existing: Option<TombstoneCols> = tx
            .query_row(
                "SELECT adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at \
                 FROM record_tombstones WHERE record_id = ?1",
                params![id.0],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                    ))
                },
            )
            .optional()?;
        return Ok(match existing {
            Some((adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at)) => {
                ForgetRecordOutcome::AlreadyForgotten(ForgottenRecord {
                    record_id: id.clone(),
                    adapter,
                    instance,
                    native_id,
                    native_path,
                    raw_hash,
                    reason,
                    forgotten_at,
                })
            }
            None => ForgetRecordOutcome::NotFound,
        });
    }

    let (adapter, instance, native_id, native_path, raw_hash, derived_from) = live.unwrap();
    crate::vec_ext::delete_vec_rows_for_record(tx, &id.0)?;
    tx.execute(
        "INSERT INTO record_tombstones( \
             record_id, adapter, instance, native_id, native_path, \
             raw_hash, reason, forgotten_at, derived_from) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            id.0,
            adapter,
            instance,
            native_id,
            native_path,
            raw_hash,
            reason,
            now,
            derived_from
        ],
    )?;
    tx.execute("DELETE FROM records WHERE id = ?1", params![id.0])?;

    Ok(ForgetRecordOutcome::Forgotten(ForgottenRecord {
        record_id: id.clone(),
        adapter,
        instance,
        native_id,
        native_path,
        raw_hash,
        reason: reason.map(str::to_owned),
        forgotten_at: now,
    }))
}

/// Char-boundary-safe truncation. Returns ≤ `max_chars` scalars + `…`.
fn truncate_preview(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out = String::with_capacity(max_chars * 2);
    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push('…');
    out
}

/// Unforget one tombstone inside an existing tx. Shared by single-record
/// and cascade paths. Returns `Unforgotten` / `NotForgotten`.
fn unforget_one_in_tx(
    tx: &rusqlite::Transaction<'_>,
    id: &RecordId,
) -> Result<UnforgetRecordOutcome> {
    type TombstoneCols = (
        String,
        String,
        String,
        Option<String>,
        String,
        Option<String>,
        i64,
    );
    let existing: Option<TombstoneCols> = tx
        .query_row(
            "SELECT adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at \
             FROM record_tombstones WHERE record_id = ?1",
            params![id.0],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                ))
            },
        )
        .optional()?;

    let Some((adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at)) =
        existing
    else {
        return Ok(UnforgetRecordOutcome::NotForgotten);
    };

    tx.execute(
        "DELETE FROM record_tombstones WHERE record_id = ?1",
        params![id.0],
    )?;

    Ok(UnforgetRecordOutcome::Unforgotten(ForgottenRecord {
        record_id: id.clone(),
        adapter,
        instance,
        native_id,
        native_path,
        raw_hash,
        reason,
        forgotten_at,
    }))
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

/// Filter pushed into the SQL candidate-retrieval stage of `search_chunks_*`.
///
/// **All fields go into the SQL `WHERE` clause before `LIMIT` is applied**,
/// so they shape the candidate pool itself — never just trim a pre-built
/// majority pool after the fact. This is the load-bearing fix from
/// BLUEPRINT §17.5 PR-C: with thousands of records from one adapter and a
/// handful from another, post-filter shrinkage can leave the minority
/// adapter's results empty even when they're the best match.
///
/// Empty filter (all fields `None`) is a no-op — the original
/// `WHERE chunks_fts MATCH ?` / `WHERE e.model_id = ?` is preserved.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchFilter {
    /// Adapter id (e.g. `"claude-code"`, `"mem0"`). When set, only chunks
    /// belonging to records from this adapter survive.
    pub source: Option<String>,
    /// Instance discriminator. Only meaningful when `source` is also set
    /// (the SQL key is `(adapter, instance)`).
    pub instance: Option<String>,
    /// `Kind` string: `"fact"` / `"preference"` / `"feedback"` / `"reference"`
    /// / `"episode"` / `"skill"` / `"unknown"`.
    pub kind: Option<String>,
    /// `Scope` string: `"user"` / `"project"` / `"session"` / `"ephemeral"`.
    pub scope: Option<String>,
    /// Inclusive lower bound on `records.created_at` (unix epoch seconds).
    pub time_from: Option<i64>,
    /// Inclusive upper bound on `records.created_at` (unix epoch seconds).
    pub time_to: Option<i64>,
    /// Round 79 (PR-78b): restrict to records that carry this
    /// user tag in the `user_record_tags` overlay (R78). The tag
    /// is normalised exactly like `tag_record` writes — call
    /// [`normalize_user_tag_name`] before stuffing into this
    /// field, or get a wire mismatch (`Keep` vs `keep`). Pushed
    /// into the SQL recall stage on all three modalities (FTS,
    /// BLOB-vec, sqlite-vec) so a tagged minority record can't
    /// be displaced by untagged majority before `LIMIT`.
    pub user_tag: Option<String>,
}

impl SearchFilter {
    /// True when every field is `None` — caller can skip the JOIN.
    pub fn is_empty(&self) -> bool {
        self.source.is_none()
            && self.instance.is_none()
            && self.kind.is_none()
            && self.scope.is_none()
            && self.time_from.is_none()
            && self.time_to.is_none()
            && self.user_tag.is_none()
    }
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
    /// Non-fatal per-record import errors logged across all runs.
    /// Each row is one record the importer skipped (scan / parse /
    /// normalize / upsert phase). Surfaces in `anamnesis status` so
    /// the user knows when a source has silently-skipped data; the
    /// rows themselves are available via `recent_import_errors`.
    pub import_errors: u64,
}

/// One row from `import_errors` for `anamnesis status` / `doctor`
/// presentation. Returned newest-first by `recent_import_errors`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportErrorRow {
    /// Adapter id (e.g. `"claude-code"`).
    pub adapter: String,
    /// Instance discriminator — `""` for the default instance.
    pub instance: String,
    /// Original record id at the source, if the error happened after
    /// the adapter produced one.
    pub native_id: Option<String>,
    /// Original record path at the source.
    pub native_path: Option<String>,
    /// Pipeline phase the error happened in. One of:
    /// `scan` | `parse` | `normalize` | `chunk` | `upsert`.
    pub phase: String,
    /// Adapter-supplied error message.
    pub error: String,
    /// Unix seconds when the row was logged.
    pub occurred_at: i64,
}

/// One row from `record_tombstones` — what `forget_record` writes
/// and what `list_forgotten` (future) would surface.
///
/// Round 72 (PR-72a): the tombstone is keyed on the same
/// `(adapter, instance, native_id)` natural tuple every adapter
/// already uses, so the importer can short-circuit a forgotten
/// record before it touches chunking / embedding.
///
/// `raw_hash` is captured so a future "allow only changed content"
/// resurrection policy can compare the live source payload against
/// what got forgotten. This PR is the conservative baseline — a
/// tombstone is permanent until an explicit `unforget` (future).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgottenRecord {
    /// Hashed `records.id` — primary key + the same id `forget_record` was called with.
    pub record_id: RecordId,
    /// Adapter id (e.g. `"claude-code"`).
    pub adapter: String,
    /// Instance discriminator — `""` for the default instance.
    pub instance: String,
    /// Native id at the source.
    pub native_id: String,
    /// Native path at the source (when the adapter has one).
    pub native_path: Option<String>,
    /// `raw_hash` captured at forget time — pinned for a future
    /// resurrection policy.
    pub raw_hash: String,
    /// Operator-supplied reason. Optional.
    pub reason: Option<String>,
    /// Unix seconds at the moment `forget_record` was committed.
    pub forgotten_at: i64,
}

/// Result of [`Store::unforget_record`]. Distinguishes the two
/// observable outcomes so the CLI / MCP surface can fail loudly
/// when the operator typoed an id from `list_forgotten` instead
/// of pretending a recovery happened.
///
/// Round 75: `unforget` deletes the tombstone but does **not**
/// recreate the live `records` row. The tombstone only carried
/// provenance, not the original normalized content — and even if
/// it did, resurrecting it would let `unforget` make data appear
/// out of nowhere, which violates Anamnesis's "read-only mirror
/// of source data" contract. The record stays absent until the
/// source is re-imported (which is now allowed because the
/// tombstone gate is gone).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnforgetRecordOutcome {
    /// A tombstone existed and is now deleted. Carries the
    /// `ForgottenRecord` snapshot the operator was responding
    /// to — useful for "you just unforgot X, here's what it was".
    Unforgotten(ForgottenRecord),
    /// No tombstone for this id. Returned as a loud error by the
    /// CLI / MCP surfaces because the operator almost certainly
    /// pasted the wrong id from `list_forgotten`.
    NotForgotten,
}

/// Result of [`Store::forget_record`]. Distinguishes the three
/// observable outcomes so the CLI / future MCP surface can render
/// them differently — and so repeated `forget` calls stay
/// idempotent without becoming silent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgetRecordOutcome {
    /// The record existed and is now forgotten. Carries the
    /// tombstone that was just written.
    Forgotten(ForgottenRecord),
    /// A tombstone was already present for this record — nothing
    /// changed, but the call is still a success from the operator's
    /// point of view (the record remains forgotten).
    AlreadyForgotten(ForgottenRecord),
    /// Neither the record nor a tombstone exists. Callers that
    /// treat this as user error (CLI) should exit non-zero; callers
    /// that treat it as benign (future MCP idempotent path) can
    /// ignore.
    NotFound,
}

/// Per-table cascade counts a `forget_record` would delete. Computed via
/// the same queries the real cascade runs (vec0 via
/// `vec_ext::count_vec_rows_for_record`; rest via FK cascade) so preview
/// and reality can't drift.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ForgetCascadeCounts {
    /// 1 for `WouldForget`, 0 otherwise.
    pub records: u64,
    /// `raw_artifacts` rows.
    pub raw_artifacts: u64,
    /// `record_chunks` rows.
    pub record_chunks: u64,
    /// `chunk_embeddings` rows.
    pub chunk_embeddings: u64,
    /// `embedding_jobs` rows.
    pub embedding_jobs: u64,
    /// `user_record_tags` rows.
    pub user_record_tags: u64,
    /// Manually counted — vec0 has no FK.
    pub vec0_rows: u64,
}

/// Three-state result of `Store::preview_forget_record`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgetRecordPreview {
    /// Live record exists — cascade-delete + tombstone-write previewed.
    WouldForget {
        /// Per-table cascade counts.
        would_delete: ForgetCascadeCounts,
        /// `ForgottenRecord` minus `forgotten_at` (stamped at commit time).
        tombstone_preview: ForgetTombstonePreview,
    },
    /// Tombstone already exists — no work; surfaces when/why.
    AlreadyForgotten(ForgottenRecord),
    /// Neither row nor tombstone.
    NotFound,
}

/// Would-be tombstone from a dry-run preview. `ForgottenRecord` minus
/// `forgotten_at` (stamped only on real commit).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgetTombstonePreview {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Adapter id.
    pub adapter: String,
    /// Instance discriminator; `""` for default.
    pub instance: String,
    /// Native id at the source.
    pub native_id: String,
    /// Native path, when the adapter has one.
    pub native_path: Option<String>,
    /// Raw hash on the live record.
    pub raw_hash: String,
    /// Operator-supplied `--reason`, echoed.
    pub reason: Option<String>,
}

/// Options for `forget_record_with_options`. Default = R72 behaviour.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ForgetCascadeOptions {
    /// `true` = also tombstone every descendant via `provenance.derived_from`.
    pub cascade_derived: bool,
}

/// One descendant's outcome inside `ForgetCascadeOutcome.derived`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedForgetRecord {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Adapter id (typically `"extractor"`).
    pub adapter: String,
    /// Instance — empty for default.
    pub instance: String,
    /// Native id at source.
    pub native_id: String,
    /// Native path, if any.
    pub native_path: Option<String>,
    /// `raw_hash`.
    pub raw_hash: String,
    /// Tombstone timestamp (freshly written or pre-existing).
    pub forgotten_at: i64,
    /// `true` = tombstone already existed before this cascade.
    pub was_already_forgotten: bool,
}

/// Outcome of `Store::forget_record_with_options`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgetCascadeOutcome {
    /// Root outcome (= R72 `forget_record`).
    pub root: ForgetRecordOutcome,
    /// Descendants visited; empty unless `cascade_derived = true`.
    pub derived: Vec<DerivedForgetRecord>,
}

/// One descendant's dry-run preview row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedForgetPreview {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Adapter id.
    pub adapter: String,
    /// Instance — empty for default.
    pub instance: String,
    /// Native id at source.
    pub native_id: String,
    /// Native path, if any.
    pub native_path: Option<String>,
    /// `raw_hash`.
    pub raw_hash: String,
    /// Per-table cascade counts for this descendant.
    pub would_delete: ForgetCascadeCounts,
    /// `Some(ts)` = tombstone already exists; cascade leaves it alone.
    pub already_forgotten_at: Option<i64>,
}

/// Preview for `Store::preview_forget_record_with_options`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgetCascadePreview {
    /// Root preview (= R83 `preview_forget_record`).
    pub root: ForgetRecordPreview,
    /// Descendants; empty unless `cascade_derived = true`.
    pub derived: Vec<DerivedForgetPreview>,
}

/// Options for `unforget_record_with_options`. Default = R75 behaviour.
/// Pre-R134 tombstones have NULL `derived_from`; cascade treats them as
/// "no descendants known" — root unforget still works.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnforgetCascadeOptions {
    /// `true` = also delete every descendant tombstone.
    pub cascade_derived: bool,
}

/// One descendant tombstone deleted by the cascade.
/// Note: does NOT recreate the live record (R75 contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedUnforgetRecord {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Adapter id from the original forget.
    pub adapter: String,
    /// Instance — empty for default.
    pub instance: String,
    /// Native id from the original forget.
    pub native_id: String,
    /// Native path, if any.
    pub native_path: Option<String>,
    /// `raw_hash`.
    pub raw_hash: String,
    /// Reason from the original forget.
    pub reason: Option<String>,
    /// Original forget timestamp.
    pub forgotten_at: i64,
}

/// Outcome of `Store::unforget_record_with_options`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnforgetCascadeOutcome {
    /// Root outcome (= R75 `unforget_record`).
    pub root: UnforgetRecordOutcome,
    /// Descendant tombstones removed; empty unless cascading.
    pub derived: Vec<DerivedUnforgetRecord>,
}

/// One descendant tombstone in a dry-run preview.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedUnforgetPreview {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Adapter id from the original forget.
    pub adapter: String,
    /// Instance — empty for default.
    pub instance: String,
    /// Native id from the original forget.
    pub native_id: String,
    /// Native path, if any.
    pub native_path: Option<String>,
    /// `raw_hash`.
    pub raw_hash: String,
    /// Reason from the original forget.
    pub reason: Option<String>,
    /// Original forget timestamp.
    pub forgotten_at: i64,
}

/// Preview for `Store::preview_unforget_record_with_options`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnforgetCascadePreview {
    /// Root preview (= R95 `preview_unforget_record`).
    pub root: UnforgetRecordOutcome,
    /// Descendant tombstones the cascade would delete.
    pub derived: Vec<DerivedUnforgetPreview>,
}

/// Lightweight projection of `records` — everything the search +
/// packer + MCP wire format actually need, *without* the heavy
/// content / tags / metadata payload that an `AnamnesisRecord` carries.
///
/// Round 68 motivation: `pack()` in the search crate used to call
/// `get_records_by_ids` for every hit, which selected `records.content`
/// (a multi-KB transcript blob for Claude Code / Codex adapters) and
/// JSON-parsed `tags` / `metadata` per row — only to discard the
/// content downstream because the MCP wire shape returns chunk snippets,
/// not full records. `RecordHeader` is the projection MCP / CLI / packer
/// actually consume, so the read path can skip the expensive columns.
///
/// If a caller does need the full record (e.g. `tool_get_record`), it
/// should still call [`Store::get_record`] / [`Store::get_records_by_ids`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordHeader {
    /// Record id (e.g. `"claude-code:/Users/.../session.jsonl:42"`).
    pub id: RecordId,
    /// Adapter + instance + (empty) version. Version is unused by the
    /// store; kept for shape parity with `AnamnesisRecord.source`.
    pub source: SourceDescriptor,
    /// `User` / `Project` / `Global`.
    pub scope: Scope,
    /// Record kind (`Fact`, `Preference`, `Skill`, …).
    pub kind: Kind,
    /// When the record was first ingested.
    pub created_at: DateTime<Utc>,
    /// Last update time if the record has been re-imported.
    pub updated_at: Option<DateTime<Utc>>,
    /// Provenance — caller (CLI / MCP) uses `native_path` to render
    /// "where this came from" and `native_id` for the trace UI.
    pub provenance: Provenance,
    /// Schema version of the row at write time. Kept so a future
    /// migration that wants to fan out by version doesn't have to
    /// add another projection.
    pub schema_version: u32,
    /// Round 78: user-applied tags from the `user_record_tags`
    /// overlay. Distinct from `AnamnesisRecord.tags`, which is
    /// adapter-derived and gets overwritten on every re-import.
    /// Sorted ASCII-ascending so the wire is stable. Empty
    /// vector when the record has no user tags (the common case).
    pub user_tags: Vec<String>,
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

/// Source row joined with its current per-source counts. Returned by
/// `list_sources_with_counts`; consumed by MCP `list_sources` and CLI
/// `source list` so agents and operators see how much data is behind
/// each registered source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceWithCounts {
    /// The source registry row itself.
    pub source: SourceRow,
    /// Number of distinct records currently in the store for this
    /// `(adapter, instance)` pair. `0` for a registered-but-never-imported
    /// source — that's a useful staleness signal, not a defect.
    pub record_count: u64,
    /// Number of chunks across all records for this source.
    pub chunk_count: u64,
    /// Round 82 (PR-78d): number of **distinct records** in this
    /// source that have ≥1 entry in `user_record_tags` — NOT the
    /// raw tag-row count, and NOT the adapter-derived `records.tags`.
    /// Lets `source list` / `list_sources` answer "where do my
    /// curated `keep-forever` records actually live?" without a
    /// second round-trip.
    pub tagged_record_count: u64,
}

/// Maximum `limit` accepted by `list_record_ids_paged` and the MCP
/// `resources/list` handler. Sized so a single page fits comfortably
/// in a JSON-RPC response (~ a few hundred KB at most). Round-21
/// (§-1.5 PR-2).
pub const MAX_LIST_LIMIT: u32 = 1000;

/// Hard cap on `Store::list_forgotten` page size. Smaller than
/// `MAX_LIST_LIMIT` because tombstone rows carry potentially
/// sensitive fields (`raw_hash`, `reason`, `native_path`) — a tight
/// page keeps any single operator request from accidentally
/// exfiltrating the whole tombstone table.
pub const LIST_FORGOTTEN_MAX_LIMIT: u32 = 100;

/// Page cap on `Store::list_duplicate_raw_hashes` — anti-exfiltration.
pub const LIST_DUPLICATE_RAW_HASHES_MAX_LIMIT: u32 = 100;

/// Page cap on `Store::list_native_content_conflicts_filtered`.
pub const LIST_NATIVE_CONFLICTS_MAX_LIMIT: u32 = 100;

/// Max chars of `content` surfaced in a conflict-record preview
/// when `include_content = true`.
pub const NATIVE_CONFLICT_PREVIEW_CHARS: usize = 240;

/// Per-tag length cap; bounded before write so pathological input
/// can't reach `user_record_tags`.
pub const USER_TAG_MAX_LEN: usize = 64;

/// Max distinct tags per `tag_record` call.
pub const TAG_RECORD_MAX_BATCH: usize = 32;

/// Direction of a `tag_record` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserTagOperation {
    /// Set-insert; existing tags are no-ops.
    Add,
    /// Set-delete; missing tags are no-ops.
    Remove,
    /// Install input as the **full** post-call set in one immediate tx.
    /// Empty input clears all tags. `Add`/`Remove` reject empty input.
    Replace,
}

/// Result of `Store::tag_record`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserTagMutation {
    /// Record the call mutated.
    pub record_id: RecordId,
    /// Direction of the call.
    pub operation: UserTagOperation,
    /// Input after `trim().to_lowercase()` + dedup + validation;
    /// input order preserved.
    pub requested: Vec<String>,
    /// `Add`/`Remove`: rows actually changed. `Replace`: set delta
    /// (re-replacing with the same set reports `0`).
    pub changed: u32,
    /// Post-call set, ASCII-ascending.
    pub user_tags: Vec<String>,
}

/// One row inside a duplicate-raw_hash group. `native_path` is here for
/// operator disambiguation; CLI/MCP wires redact it by default
/// (`include_sensitive` knob).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateRawHashRecord {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Adapter id.
    pub adapter: String,
    /// `""` for default instance.
    pub instance: String,
    /// Native id at the source.
    pub native_id: String,
    /// Native path, when the adapter has one.
    pub native_path: Option<String>,
    /// Unix seconds.
    pub created_at: i64,
    /// Unix seconds; `None` when never updated.
    pub updated_at: Option<i64>,
}

/// Records sharing one `raw_hash` (always size ≥2 via `HAVING COUNT(*) > 1`).
/// Exact byte-identical duplicates only — near-dup is R131.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateRawHashGroup {
    /// Redacted by default at CLI/MCP wires.
    pub raw_hash: String,
    /// `>= 2` records, newest-first.
    pub records: Vec<DuplicateRawHashRecord>,
}

/// Filter for `Store::list_duplicate_raw_hashes_filtered`. Groups stay whole;
/// `source`/`instance` accept single value or comma-separated OR list via
/// `anamnesis_core::parse_csv_filter`. `limit` clamped to
/// `[1, LIST_DUPLICATE_RAW_HASHES_MAX_LIMIT]`.
#[derive(Debug, Clone, Default)]
pub struct DuplicateRawHashFilter {
    /// Adapter id or comma-separated OR list; `None`/`""` = any.
    pub source: Option<String>,
    /// Instance or comma-separated OR list; `None`/`""` = any.
    pub instance: Option<String>,
    /// Max groups; clamped by the store.
    pub limit: u32,
}

/// Aggregate counts for `Store::count_duplicate_raw_hashes_by_source`.
/// `by_source[]` counts **records** (not group memberships) to avoid
/// double-counting across mixed-source groups.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DuplicateRawHashCounts {
    /// Filter-matching groups (each has ≥2 live members).
    pub total_groups: u64,
    /// Live records summed across those groups.
    pub total_records: u64,
    /// Per-`(adapter, instance)` breakdown.
    pub by_source: Vec<DuplicateRawHashSourceCount>,
}

/// One `(adapter, instance)` bucket of `DuplicateRawHashCounts.by_source`.
/// Empty `instance` serialises as JSON `null` at the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateRawHashSourceCount {
    /// Adapter id.
    pub adapter: String,
    /// Instance discriminator; `""` for default.
    pub instance: String,
    /// Live records this source contributes to the filtered duplicate set.
    pub duplicate_record_count: u64,
}

/// One record inside a [`NativeConflictGroup`]. Same privacy discipline
/// as [`DuplicateRawHashRecord`]: no `content`/`raw_hash`/`native_path`
/// by default; opt-in `content_preview` via `include_content`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeConflictRecord {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Adapter id.
    pub adapter: String,
    /// Instance discriminator; `""` for default.
    pub instance: String,
    /// Repeated per row (= group key) for flat-table rendering.
    pub native_id: String,
    /// Whether the record carries a `native_path` (path itself is redacted).
    pub has_native_path: bool,
    /// Unix seconds.
    pub created_at: i64,
    /// Unix seconds; `None` when never updated.
    pub updated_at: Option<i64>,
    /// 1-based variant index inside the group, assigned in
    /// `(adapter ASC, record_id ASC)` order. Records sharing
    /// `(native_id, content)` share a variant.
    pub content_variant: u32,
    /// Populated only when `include_content = true`; truncated to
    /// `NATIVE_CONFLICT_PREVIEW_CHARS`.
    pub content_preview: Option<String>,
}

/// Records sharing one `native_id` across ≥2 adapters with ≥2 distinct
/// (normalised) `content` values — the **identity disagreement** surface
/// (distinct from R77/R131 duplicate detection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeConflictGroup {
    /// Shared `native_id`.
    pub native_id: String,
    /// Always `>= 2`, spanning `>= 2` distinct adapters. Ordered
    /// `(adapter ASC, record_id ASC)` for stable variant numbering.
    pub records: Vec<NativeConflictRecord>,
    /// Distinct content variants, `>= 2`. Sort key for
    /// "highest-divergence groups first".
    pub content_variant_count: u32,
}

/// Filter for [`Store::list_native_content_conflicts_filtered`]. Source /
/// instance accept a single value or comma-separated OR list (parsed via
/// `anamnesis_core::parse_csv_filter`). Groups stay whole — siblings
/// outside the filter still appear in the returned group.
#[derive(Debug, Clone, Default)]
pub struct NativeConflictFilter {
    /// Adapter id or comma-separated OR list; `None`/`""` = any.
    pub source: Option<String>,
    /// Instance or comma-separated OR list; `None`/`""` = any.
    pub instance: Option<String>,
    /// Clamped to `[1, LIST_NATIVE_CONFLICTS_MAX_LIMIT]`.
    pub limit: u32,
    /// Populate `NativeConflictRecord.content_preview` (default off).
    pub include_content: bool,
}

/// Pick the winning content variant inside one `native_id` conflict.
/// Exactly one selector must be `Some`. `KeepVariant(n)` selects every
/// record whose `content_variant == n` (variant numbering matches
/// `Store::list_native_content_conflicts_filtered`). `KeepRecordId(id)`
/// keeps the record with that id and every sibling sharing its content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcceptConflictSelector {
    /// 1-based variant index from the conflict listing.
    KeepVariant(u32),
    /// Specific record id; siblings with identical content also stay live.
    KeepRecordId(RecordId),
}

/// Options for `Store::accept_native_conflict_variant` /
/// `Store::preview_accept_native_conflict_variant`.
#[derive(Debug, Clone)]
pub struct AcceptConflictOptions {
    /// `(adapter, instance, native_id)` group key — same `native_id` the
    /// conflict listing surfaces. Empty `instance` is the default.
    pub native_id: String,
    /// Which variant to keep.
    pub selector: AcceptConflictSelector,
    /// Audit/tombstone reason echoed on every loser tombstone.
    pub reason: Option<String>,
    /// Also tombstone every loser's `provenance.derived_from` descendants;
    /// kept records (and their descendants) are never touched.
    pub cascade_derived: bool,
}

/// One record inside an accept-conflict outcome / preview. Same redacted
/// projection as `NativeConflictRecord`; carries the post-call decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptConflictRecord {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Adapter id.
    pub adapter: String,
    /// Instance discriminator; `""` for default.
    pub instance: String,
    /// Shared `native_id` (the conflict key).
    pub native_id: String,
    /// 1-based variant index inside the group (matches conflict listing).
    pub content_variant: u32,
    /// `"keep"` for the chosen variant, `"forget"` for losers.
    pub decision: &'static str,
}

/// Snapshot of one descendant tombstone an apply would write (or did).
/// Mirrors `DerivedForgetRecord` minus `raw_hash` / `native_path` for
/// the same redacted privacy contract `NativeConflictRecord` honours.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptConflictDescendant {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Adapter id.
    pub adapter: String,
    /// Instance discriminator; `""` for default.
    pub instance: String,
    /// Native id at the source.
    pub native_id: String,
    /// Unix-seconds tombstone timestamp; `None` for previews.
    pub forgotten_at: Option<i64>,
    /// `true` when the descendant already carried a tombstone before this call.
    pub was_already_forgotten: bool,
}

/// Result of `accept_native_conflict_variant` (apply) or its preview.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptConflictOutcome {
    /// `native_id` the call acted on.
    pub native_id: String,
    /// 1-based variant index of the keeper (matches conflict listing).
    pub keep_variant: u32,
    /// Live records the call would keep (preview) or kept (apply).
    pub keep_records: Vec<AcceptConflictRecord>,
    /// Loser records the call would tombstone (preview) or tombstoned (apply).
    pub forget_records: Vec<AcceptConflictRecord>,
    /// Descendants tombstoned when `cascade_derived = true`; empty otherwise.
    pub cascade_derived: Vec<AcceptConflictDescendant>,
    /// `true` when no write happened (dry-run / preview path).
    pub dry_run: bool,
}

/// One side of a cross-adapter reconciliation pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconcileSourceSelector {
    /// Adapter id (e.g. `"mem0"`, `"letta"`).
    pub adapter: String,
    /// Instance discriminator; `None` = default (empty-string instance).
    pub instance: Option<String>,
}

/// Hard cap on `Store::reconcile_sources` sample-row pages.
/// Counts ignore this — only the per-bucket sample arrays are capped.
pub const RECONCILE_MAX_LIMIT: u32 = 100;

/// Options for [`Store::reconcile_sources`]. Counts are always
/// computed; sample arrays are capped at `limit`.
#[derive(Debug, Clone)]
pub struct ReconcileOptions {
    /// Left side of the comparison.
    pub left: ReconcileSourceSelector,
    /// Right side of the comparison.
    pub right: ReconcileSourceSelector,
    /// Max records returned per sample bucket. Clamped to
    /// `[1, RECONCILE_MAX_LIMIT]`. Counts are unaffected.
    pub limit: u32,
    /// When `true`, surface the per-record `identity_key`. Off by
    /// default — counts + minimal record_id/kind/scope are enough
    /// for a "what's the drift?" summary.
    pub include_identity: bool,
}

/// One sampled record inside a reconcile bucket. Redacted projection:
/// `record_id`, `kind`, `scope`, `created_at`. `identity_key` is
/// `Some(_)` only when [`ReconcileOptions::include_identity`] is set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconcileSample {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Kind (lowercase: `fact`, `preference`, …).
    pub kind: String,
    /// Scope (lowercase).
    pub scope: String,
    /// Unix-seconds created_at.
    pub created_at: i64,
    /// `metadata.anamnesis_native_id` ∨ `provenance.native_id`,
    /// only when `include_identity = true`.
    pub identity_key: Option<String>,
    /// Which field supplied the identity key: `"anamnesis_native_id"`
    /// (round-tripped, comparable) or `"native_id"` (per-adapter,
    /// only comparable when adapters share an upstream source).
    pub identity_source: &'static str,
}

/// Filter-scoped counts. Sum-of-buckets identity:
/// `only_left + only_right + both = total distinct identities`.
/// `conflicts ≤ both` — each conflict belongs to `both`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReconcileCounts {
    /// Identities present on left, absent on right.
    pub only_left: u64,
    /// Identities present on right, absent on left.
    pub only_right: u64,
    /// Identities present on both sides (regardless of content).
    pub both: u64,
    /// Subset of `both` where content differs across sides.
    pub conflicts: u64,
    /// Total distinct identities on the left side (after dedup by identity).
    pub left_total: u64,
    /// Total distinct identities on the right side.
    pub right_total: u64,
}

/// Per-bucket sample arrays. Each capped at `ReconcileOptions::limit`.
#[derive(Debug, Clone, Default)]
pub struct ReconcileSamples {
    /// Sample of left-only identities (sorted record_id ASC).
    pub only_left: Vec<ReconcileSample>,
    /// Sample of right-only identities.
    pub only_right: Vec<ReconcileSample>,
    /// Sample of conflicting identities (left side row shown).
    pub conflicts: Vec<ReconcileSample>,
}

/// Result of [`Store::reconcile_sources`].
#[derive(Debug, Clone)]
pub struct ReconcileOutcome {
    /// Left side, echoed.
    pub left: ReconcileSourceSelector,
    /// Right side, echoed.
    pub right: ReconcileSourceSelector,
    /// Filter-scoped counts.
    pub counts: ReconcileCounts,
    /// Sample rows per bucket.
    pub samples: ReconcileSamples,
}

/// Reconcile bucket selector for [`Store::reconcile_bucket_ids`].
/// (`Both` is intentionally omitted — operators reconcile drift,
/// not already-in-sync records; `Conflicts` is a separate decision
/// surfaced via [`Store::accept_native_conflict_variant`].)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconcileBucket {
    /// Identities present on left, absent on right.
    OnlyLeft,
    /// Identities present on right, absent on left.
    OnlyRight,
}

/// Filter for `Store::list_forgotten`. Mirrors
/// the `(adapter, instance)` natural key the tombstones are
/// indexed on so the operator can scope to a single source. `limit`
/// is clamped to `[1, LIST_FORGOTTEN_MAX_LIMIT]` by the store.
#[derive(Debug, Clone, Default)]
pub struct ListForgottenFilter {
    /// Adapter id (e.g. `"claude-code"`). `None` returns all sources.
    pub source: Option<String>,
    /// Instance discriminator. `None` returns all instances of
    /// the given source.
    pub instance: Option<String>,
    /// Max rows to return. Clamped to `[1, LIST_FORGOTTEN_MAX_LIMIT]`.
    pub limit: u32,
}

/// Round 94 (PR-78p): minimal projection consumed by the MCP
/// `summarize_my_preferences` prompt. Just the fields the
/// prompt renders into bullet text — no `tags`, no `metadata`,
/// no embedding readiness — so the read stays cheap even on a
/// big user-scope corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummarizePreferencesRow {
    /// Hashed `records.id`.
    pub id: String,
    /// `records.content` body.
    pub content: String,
    /// `records.kind` lowercase string (`fact`, `preference`, ...).
    pub kind: String,
    /// `records.native_path` if the adapter has one.
    pub native_path: Option<String>,
    /// `records.created_at` unix seconds — used for ordering.
    pub created_at: i64,
}

/// Round 90 (PR-78l): one `(adapter, instance)` bucket from
/// `Store::count_forgotten_by_source`. Used by
/// `list_forgotten --include-counts` to give operators a
/// total + per-source breakdown without having to page through
/// every tombstone row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgottenSourceCount {
    /// Adapter id (e.g. `"claude-code"`).
    pub adapter: String,
    /// Instance discriminator. Empty string for the default
    /// instance; the CLI / MCP wire formats serialise that as
    /// JSON `null` to match the rest of the surface.
    pub instance: String,
    /// Number of tombstones in this `(adapter, instance)` bucket.
    pub forgotten_count: u64,
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

    /// Like `list_sources_full` but also carries per-source record /
    /// chunk counts so MCP consumers can answer "is this source stale?"
    /// and "how much data lives behind it?" without a second round
    /// trip.
    ///
    /// Counts are computed via `LEFT JOIN`, so a source that's been
    /// registered but has never produced records still appears with
    /// counts of zero — which is exactly the signal an agent needs to
    /// detect a configured-but-broken adapter.
    ///
    /// Aggregation is grouped on `(adapter, instance)` because the
    /// canonical key in the `sources` table uses `instance=''` for the
    /// default instance. Grouping on `adapter` alone would silently
    /// merge multiple instances of the same adapter into one row.
    pub fn list_sources_with_counts(&self) -> Result<Vec<SourceWithCounts>> {
        // Round 82: counts are computed via **independent scalar
        // subqueries**, NOT a chain of LEFT JOINs. The R78
        // `user_record_tags` overlay would otherwise amplify
        // `chunk_count` — a record with 3 chunks and 4 tags
        // appearing as 12 chunks in the GROUP BY. Each subquery
        // counts its own table along the `(adapter, instance)`
        // axis with no cross-talk. Indexes on
        // `records(adapter, instance)`, `record_chunks(record_id)`,
        // and the `user_record_tags(record_id, tag)` PK keep this
        // O(log N) per source.
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT s.adapter, s.instance, s.location, s.config_json, \
                    s.added_at, s.last_import_at, \
                    (SELECT COUNT(*) \
                       FROM records r \
                      WHERE r.adapter = s.adapter \
                        AND r.instance = s.instance) AS record_count, \
                    (SELECT COUNT(*) \
                       FROM record_chunks rc \
                       JOIN records r ON r.id = rc.record_id \
                      WHERE r.adapter = s.adapter \
                        AND r.instance = s.instance) AS chunk_count, \
                    (SELECT COUNT(DISTINCT urt.record_id) \
                       FROM user_record_tags urt \
                       JOIN records r ON r.id = urt.record_id \
                      WHERE r.adapter = s.adapter \
                        AND r.instance = s.instance) AS tagged_record_count \
             FROM sources s \
             ORDER BY s.adapter, s.instance",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok(SourceWithCounts {
                    source: SourceRow {
                        adapter: r.get(0)?,
                        instance: r.get(1)?,
                        location: r.get(2)?,
                        config_json: r.get(3)?,
                        added_at: r.get(4)?,
                        last_import_at: r.get(5)?,
                    },
                    record_count: r.get::<_, i64>(6)? as u64,
                    chunk_count: r.get::<_, i64>(7)? as u64,
                    tagged_record_count: r.get::<_, i64>(8)? as u64,
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
    /// Returns `(records_written, chunks_written)`. Both are `0` when the
    /// record already exists with an identical `raw_hash` (= the source
    /// payload byte-for-byte unchanged), in which case **the call is a
    /// total no-op**: no `records` rewrite, no `raw_artifacts` rewrite,
    /// and crucially no `record_chunks` DELETE / INSERT — which is what
    /// keeps the jieba `chunks_ai` / `chunks_ad` triggers from firing
    /// 99,716 times on a re-import (see `docs/verification/round-6-
    /// embedding-dogfood.md` Finding 2 for the regression this fixes).
    ///
    /// The fast-path check happens **before** any DELETE so the AFTER
    /// DELETE trigger never runs on unchanged content. Putting the check
    /// after the DELETE would wipe the entire performance win — the
    /// tokenize_cjk(old.content) call inside `chunks_ad` is the
    /// expensive piece, not the INSERT.
    ///
    /// raw_hash is a pure function of the source payload (see each
    /// adapter's `normalize_*` for the blake3 input), so equal raw_hash
    /// guarantees the normalized record and its chunks are identical
    /// to what's in the store. Tags / metadata / scope / kind cannot
    /// drift independently of the source payload because every
    /// normalizer derives them deterministically from the same source
    /// bytes that produce raw_hash.
    pub fn upsert_record(
        &self,
        record: &AnamnesisRecord,
        chunks: &[Chunk],
        raw_payload_json: Option<&str>,
    ) -> Result<(u64, u64)> {
        let active = self.active_model()?;
        let mut conn = self.conn.lock();
        // Immediate transaction so the (check tombstone) -> (write
        // record) sequence is atomic against concurrent `forget` from
        // another process — otherwise a forget that lands between
        // those two statements would be silently overwritten.
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        // Round 72 (PR-72a): tombstone gate. Forgotten records are
        // suppressed at the *importer* layer too — but the store
        // owns the final write, so this is the canonical
        // enforcement point. The natural key
        // (adapter, instance, native_id) is what every adapter uses,
        // so the check covers all 13 sources uniformly.
        if record_is_tombstoned(&tx, record)? {
            tx.commit()?;
            return Ok((0, 0));
        }

        // Fast-path. The check must run before write_record / write_chunks
        // so neither the records UPSERT nor the chunks DELETE+INSERT fires
        // when nothing has changed.
        let existing_hash: Option<String> = tx
            .query_row(
                "SELECT raw_hash FROM records WHERE id = ?1",
                params![record.id.0],
                |r| r.get::<_, String>(0),
            )
            .optional()?;
        if existing_hash.as_deref() == Some(record.provenance.raw_hash.as_str()) {
            // Nothing to do — but we still want any pending embedding
            // jobs to be enqueued under the active model if they aren't
            // already (e.g. user switched models since last import).
            // `enqueue_jobs` is ON CONFLICT DO NOTHING, so this is safe
            // and cheap (no jieba calls, no chunk rewrite).
            if let Some(model_id) = active.as_deref() {
                let now = chrono::Utc::now().timestamp();
                enqueue_jobs(&tx, chunks, model_id, now)?;
            }
            tx.commit()?;
            return Ok((0, 0));
        }

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

    /// Batch variant of `upsert_record` — wraps up to `items.len()`
    /// record upserts in a single SQLite transaction so the importer
    /// pays one `fsync` per batch instead of one per record. For the
    /// claude-code import (1795 records / 50K chunks) this turns
    /// thousands of `BEGIN`/`COMMIT` round-trips into ~28 of them.
    ///
    /// Per-record semantics are identical to `upsert_record`:
    ///   - `raw_hash`-equal records are no-ops (only enqueue any
    ///     missing embedding jobs under the active model)
    ///   - mismatched-hash records get full `records` / `raw_artifacts`
    ///     / `record_chunks` rewrites + embedding jobs
    ///
    /// Returns `(records_upserted, chunks_written)` summed across the
    /// batch. If any statement inside the batch fails, the entire
    /// transaction is rolled back and the error propagates — callers
    /// that need per-record error isolation (e.g. the importer's
    /// log-and-skip behavior) should catch the error and fall back to
    /// the per-record `upsert_record` path for that batch.
    pub fn upsert_records_batch(
        &self,
        items: &[(&AnamnesisRecord, &[Chunk], Option<&str>)],
    ) -> Result<(u64, u64)> {
        if items.is_empty() {
            return Ok((0, 0));
        }
        let active = self.active_model()?;
        let mut conn = self.conn.lock();
        // Immediate transaction — see `upsert_record` for the
        // forget-race-prevention rationale.
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut total_records = 0u64;
        let mut total_chunks = 0u64;
        for (record, chunks, raw_payload_json) in items {
            // Take `now` per-item (not per-batch) so `raw_artifacts.captured_at`
            // and `embedding_jobs.enqueued_at` semantics match per-record
            // `upsert_record`. Cheap; `Utc::now()` is microseconds.
            let now = chrono::Utc::now().timestamp();
            // Round 72: skip forgotten records before raw_hash fast-path.
            // The check is per-row inside the batch tx so a forget that
            // lands mid-batch from a concurrent process still wins.
            if record_is_tombstoned(&tx, record)? {
                continue;
            }
            let existing_hash: Option<String> = tx
                .query_row(
                    "SELECT raw_hash FROM records WHERE id = ?1",
                    params![record.id.0],
                    |r| r.get::<_, String>(0),
                )
                .optional()?;
            if existing_hash.as_deref() == Some(record.provenance.raw_hash.as_str()) {
                if let Some(model_id) = active.as_deref() {
                    enqueue_jobs(&tx, chunks, model_id, now)?;
                }
                continue;
            }
            write_record(&tx, record)?;
            write_raw_artifact(&tx, record, *raw_payload_json, now)?;
            write_chunks(&tx, record, chunks)?;
            if let Some(model_id) = active.as_deref() {
                enqueue_jobs(&tx, chunks, model_id, now)?;
            }
            total_records += 1;
            total_chunks += chunks.len() as u64;
        }
        tx.commit()?;
        Ok((total_records, total_chunks))
    }

    /// Forget a record: write a `(adapter, instance, native_id)` tombstone,
    /// delete the row (cascades raw_artifacts / record_chunks /
    /// chunk_embeddings / embedding_jobs via FK), and clear vec0 rows
    /// manually (vec0 has no FK cascade). Idempotent — second call returns
    /// `AlreadyForgotten`; missing both row and tombstone returns `NotFound`.
    /// The natural-key tombstone is what blocks re-import.
    pub fn forget_record(
        &self,
        id: &RecordId,
        reason: Option<&str>,
    ) -> Result<ForgetRecordOutcome> {
        // Delegate to `forget_one_in_tx` so single-record + cascade paths
        // can't drift on `record_tombstones.derived_from` writes.
        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let now = chrono::Utc::now().timestamp();
        let outcome = forget_one_in_tx(&tx, id, reason, now)?;
        tx.commit()?;
        Ok(outcome)
    }

    /// Dry-run [`Store::forget_record`]: per-table cascade counts inside a
    /// rolled-back transaction. Read-only; callers must not record audit.
    /// vec0 counts traverse the same path as the real delete to prevent drift.
    pub fn preview_forget_record(
        &self,
        id: &RecordId,
        reason: Option<&str>,
    ) -> Result<ForgetRecordPreview> {
        let mut conn = self.conn.lock();
        // Never commits — tx gives a consistent snapshot across the 7 counts.
        let tx = conn.transaction()?;

        // Live record? If so, the preview will be WouldForget.
        let live: Option<(String, String, String, Option<String>, String)> = tx
            .query_row(
                "SELECT adapter, instance, native_id, native_path, raw_hash \
                 FROM records WHERE id = ?1",
                params![id.0],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, String>(4)?,
                    ))
                },
            )
            .optional()?;

        if let Some((adapter, instance, native_id, native_path, raw_hash)) = live {
            // Count what cascade-delete would remove. Each of these
            // tables either has a direct FK to records or cascades
            // through record_chunks → records.
            let raw_artifacts: i64 = tx.query_row(
                "SELECT COUNT(*) FROM raw_artifacts WHERE record_id = ?1",
                params![id.0],
                |r| r.get(0),
            )?;
            let record_chunks: i64 = tx.query_row(
                "SELECT COUNT(*) FROM record_chunks WHERE record_id = ?1",
                params![id.0],
                |r| r.get(0),
            )?;
            let chunk_embeddings: i64 = tx.query_row(
                "SELECT COUNT(*) \
                 FROM chunk_embeddings e \
                 JOIN record_chunks rc ON rc.id = e.chunk_id \
                 WHERE rc.record_id = ?1",
                params![id.0],
                |r| r.get(0),
            )?;
            let embedding_jobs: i64 = tx.query_row(
                "SELECT COUNT(*) \
                 FROM embedding_jobs j \
                 JOIN record_chunks rc ON rc.id = j.chunk_id \
                 WHERE rc.record_id = ?1",
                params![id.0],
                |r| r.get(0),
            )?;
            let user_record_tags: i64 = tx.query_row(
                "SELECT COUNT(*) FROM user_record_tags WHERE record_id = ?1",
                params![id.0],
                |r| r.get(0),
            )?;
            let vec0_rows = crate::vec_ext::count_vec_rows_for_record(&tx, &id.0)?;

            // No commit — the read-only snapshot is dropped here.
            drop(tx);

            return Ok(ForgetRecordPreview::WouldForget {
                would_delete: ForgetCascadeCounts {
                    records: 1,
                    raw_artifacts: raw_artifacts as u64,
                    record_chunks: record_chunks as u64,
                    chunk_embeddings: chunk_embeddings as u64,
                    embedding_jobs: embedding_jobs as u64,
                    user_record_tags: user_record_tags as u64,
                    vec0_rows,
                },
                tombstone_preview: ForgetTombstonePreview {
                    record_id: id.clone(),
                    adapter,
                    instance,
                    native_id,
                    native_path,
                    raw_hash,
                    reason: reason.map(str::to_owned),
                },
            });
        }

        // No live record. Maybe a tombstone exists.
        type TombstoneCols = (
            String,
            String,
            String,
            Option<String>,
            String,
            Option<String>,
            i64,
        );
        let existing: Option<TombstoneCols> = tx
            .query_row(
                "SELECT adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at \
                 FROM record_tombstones WHERE record_id = ?1",
                params![id.0],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                    ))
                },
            )
            .optional()?;
        drop(tx);

        Ok(match existing {
            Some((adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at)) => {
                ForgetRecordPreview::AlreadyForgotten(ForgottenRecord {
                    record_id: id.clone(),
                    adapter,
                    instance,
                    native_id,
                    native_path,
                    raw_hash,
                    reason,
                    forgotten_at,
                })
            }
            None => ForgetRecordPreview::NotFound,
        })
    }

    /// Cascade-aware forget. With `cascade_derived=true`, BFS walks
    /// `provenance.derived_from` (cycle-safe) and tombstones every
    /// descendant in one IMMEDIATE tx. Descendants returned BFS order.
    pub fn forget_record_with_options(
        &self,
        id: &RecordId,
        reason: Option<&str>,
        opts: &ForgetCascadeOptions,
    ) -> Result<ForgetCascadeOutcome> {
        if !opts.cascade_derived {
            // Back-compat path: just the single record, derived
            // vector is empty.
            let root = self.forget_record(id, reason)?;
            return Ok(ForgetCascadeOutcome {
                root,
                derived: Vec::new(),
            });
        }

        // Collect descendants *before* opening the write transaction
        // so cycle detection / read-only walks don't fight the
        // IMMEDIATE write lock. The BFS uses a fresh read transaction
        // for consistent snapshot.
        let descendants = self.collect_descendants(id)?;

        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let now = chrono::Utc::now().timestamp();

        // Root first — same shape as R72 `forget_record`, so the
        // outcome is `Forgotten | AlreadyForgotten | NotFound`.
        let root = forget_one_in_tx(&tx, id, reason, now)?;

        let mut derived_out: Vec<DerivedForgetRecord> = Vec::with_capacity(descendants.len());
        for child_id in descendants {
            let outcome = forget_one_in_tx(&tx, &child_id, None, now)?;
            match outcome {
                ForgetRecordOutcome::Forgotten(r) => derived_out.push(DerivedForgetRecord {
                    record_id: r.record_id,
                    adapter: r.adapter,
                    instance: r.instance,
                    native_id: r.native_id,
                    native_path: r.native_path,
                    raw_hash: r.raw_hash,
                    forgotten_at: r.forgotten_at,
                    was_already_forgotten: false,
                }),
                ForgetRecordOutcome::AlreadyForgotten(r) => derived_out.push(DerivedForgetRecord {
                    record_id: r.record_id,
                    adapter: r.adapter,
                    instance: r.instance,
                    native_id: r.native_id,
                    native_path: r.native_path,
                    raw_hash: r.raw_hash,
                    forgotten_at: r.forgotten_at,
                    was_already_forgotten: true,
                }),
                ForgetRecordOutcome::NotFound => {
                    // BFS read saw this id; if it's gone now without
                    // a tombstone, treat it as raced-away. Silently
                    // skip — the cascade goal (no live derived row)
                    // is satisfied.
                }
            }
        }

        tx.commit()?;
        Ok(ForgetCascadeOutcome {
            root,
            derived: derived_out,
        })
    }

    /// Dry-run preview of `forget_record_with_options`. Read-only.
    pub fn preview_forget_record_with_options(
        &self,
        id: &RecordId,
        reason: Option<&str>,
        opts: &ForgetCascadeOptions,
    ) -> Result<ForgetCascadePreview> {
        let root = self.preview_forget_record(id, reason)?;
        if !opts.cascade_derived {
            return Ok(ForgetCascadePreview {
                root,
                derived: Vec::new(),
            });
        }

        let descendants = self.collect_descendants(id)?;
        let mut derived: Vec<DerivedForgetPreview> = Vec::with_capacity(descendants.len());
        for child_id in descendants {
            let preview = self.preview_forget_record(&child_id, None)?;
            match preview {
                ForgetRecordPreview::WouldForget {
                    would_delete,
                    tombstone_preview,
                } => derived.push(DerivedForgetPreview {
                    record_id: tombstone_preview.record_id,
                    adapter: tombstone_preview.adapter,
                    instance: tombstone_preview.instance,
                    native_id: tombstone_preview.native_id,
                    native_path: tombstone_preview.native_path,
                    raw_hash: tombstone_preview.raw_hash,
                    would_delete,
                    already_forgotten_at: None,
                }),
                ForgetRecordPreview::AlreadyForgotten(r) => derived.push(DerivedForgetPreview {
                    record_id: r.record_id,
                    adapter: r.adapter,
                    instance: r.instance,
                    native_id: r.native_id,
                    native_path: r.native_path,
                    raw_hash: r.raw_hash,
                    would_delete: ForgetCascadeCounts::default(),
                    already_forgotten_at: Some(r.forgotten_at),
                }),
                ForgetRecordPreview::NotFound => {
                    // Raced-away between BFS and per-row preview;
                    // safe to skip — no work to describe.
                }
            }
        }
        Ok(ForgetCascadePreview { root, derived })
    }

    /// BFS descendants via `records.derived_from`. Excludes `start`.
    /// Cycles return `StoreError::Corruption`.
    fn collect_descendants(&self, start: &RecordId) -> Result<Vec<RecordId>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare("SELECT id FROM records WHERE derived_from = ?1 ORDER BY id ASC")?;

        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        visited.insert(start.0.clone());
        let mut frontier: Vec<RecordId> = vec![start.clone()];
        let mut out: Vec<RecordId> = Vec::new();

        while let Some(parent) = frontier.pop() {
            let children: Vec<String> = stmt
                .query_map(params![parent.0], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            for cid in children {
                if !visited.insert(cid.clone()) {
                    return Err(StoreError::Corruption(format!(
                        "derivation cycle detected at {cid}"
                    )));
                }
                out.push(RecordId(cid.clone()));
                frontier.push(RecordId(cid));
            }
        }
        Ok(out)
    }

    /// Round 75 (PR-75): remove a tombstone, so the same source can
    /// resurrect the memory on its next `import`. Does NOT recreate
    /// the live `records` row — the tombstone only stored
    /// provenance, not the original normalized content, and
    /// resurrecting from a tombstone would let `unforget` synthesise
    /// content out of nowhere. The truthful design is "remove the
    /// suppression gate, let the source's own data bring the record
    /// back."
    ///
    /// Idempotency: returns `NotForgotten` when no tombstone is
    /// present. Callers should treat that as user error (loud
    /// non-zero exit / tool error), because the operator almost
    /// certainly typoed an id from `list_forgotten`.
    pub fn unforget_record(&self, id: &RecordId) -> Result<UnforgetRecordOutcome> {
        let mut conn = self.conn.lock();
        // Immediate transaction mirrors the forget path — atomic
        // against a concurrent `forget` writing the same row.
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        // Columns: adapter, instance, native_id, native_path,
        //          raw_hash, reason, forgotten_at — aliased to
        //          quiet clippy::type_complexity.
        type TombstoneCols = (
            String,
            String,
            String,
            Option<String>,
            String,
            Option<String>,
            i64,
        );
        let existing: Option<TombstoneCols> = tx
            .query_row(
                "SELECT adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at \
                 FROM record_tombstones WHERE record_id = ?1",
                params![id.0],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                    ))
                },
            )
            .optional()?;

        let Some((adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at)) =
            existing
        else {
            tx.commit()?;
            return Ok(UnforgetRecordOutcome::NotForgotten);
        };

        tx.execute(
            "DELETE FROM record_tombstones WHERE record_id = ?1",
            params![id.0],
        )?;
        tx.commit()?;

        Ok(UnforgetRecordOutcome::Unforgotten(ForgottenRecord {
            record_id: id.clone(),
            adapter,
            instance,
            native_id,
            native_path,
            raw_hash,
            reason,
            forgotten_at,
        }))
    }

    /// Round 95 (PR-78q): dry-run preview for [`Store::unforget_record`].
    ///
    /// Returns the existing tombstone (so the operator can verify
    /// they're targeting the right row) or
    /// [`UnforgetRecordOutcome::NotForgotten`] if no tombstone
    /// exists. **Does not mutate the store** — no DELETE, no
    /// commit, no audit-log write. The CLI / MCP surfaces are
    /// responsible for not calling `Audit::record` either.
    pub fn preview_unforget_record(&self, id: &RecordId) -> Result<UnforgetRecordOutcome> {
        let conn = self.conn.lock();
        // Columns alias quiets clippy::type_complexity.
        type TombstoneCols = (
            String,
            String,
            String,
            Option<String>,
            String,
            Option<String>,
            i64,
        );
        let existing: Option<TombstoneCols> = conn
            .query_row(
                "SELECT adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at \
                 FROM record_tombstones WHERE record_id = ?1",
                params![id.0],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                    ))
                },
            )
            .optional()?;
        Ok(match existing {
            Some((adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at)) => {
                UnforgetRecordOutcome::Unforgotten(ForgottenRecord {
                    record_id: id.clone(),
                    adapter,
                    instance,
                    native_id,
                    native_path,
                    raw_hash,
                    reason,
                    forgotten_at,
                })
            }
            None => UnforgetRecordOutcome::NotForgotten,
        })
    }

    /// Cascade-aware unforget. With `cascade_derived=true`, BFS
    /// `record_tombstones.derived_from` (pre-R134 rows = NULL =
    /// invisible) and deletes every descendant tombstone in one tx.
    /// Does NOT resurrect any record — re-import is still required.
    pub fn unforget_record_with_options(
        &self,
        id: &RecordId,
        opts: &UnforgetCascadeOptions,
    ) -> Result<UnforgetCascadeOutcome> {
        if !opts.cascade_derived {
            let root = self.unforget_record(id)?;
            return Ok(UnforgetCascadeOutcome {
                root,
                derived: Vec::new(),
            });
        }

        // Collect descendant tombstones first (read-only walk; the
        // BFS uses its own scope so the upcoming IMMEDIATE write
        // doesn't have to contend with read planning).
        let descendants = self.collect_tombstone_descendants(id)?;

        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        // Root first, mirroring `forget_record_with_options`. We
        // re-read inside the tx so the unforget snapshot is
        // consistent with the same lock.
        let root = unforget_one_in_tx(&tx, id)?;

        let mut derived_out: Vec<DerivedUnforgetRecord> = Vec::with_capacity(descendants.len());
        for child_id in descendants {
            let outcome = unforget_one_in_tx(&tx, &child_id)?;
            if let UnforgetRecordOutcome::Unforgotten(r) = outcome {
                derived_out.push(DerivedUnforgetRecord {
                    record_id: r.record_id,
                    adapter: r.adapter,
                    instance: r.instance,
                    native_id: r.native_id,
                    native_path: r.native_path,
                    raw_hash: r.raw_hash,
                    reason: r.reason,
                    forgotten_at: r.forgotten_at,
                });
            }
            // NotForgotten = raced-away between BFS and per-row
            // delete (another writer beat us to the tombstone).
            // Goal — "no more tombstone here" — is satisfied.
        }

        tx.commit()?;
        Ok(UnforgetCascadeOutcome {
            root,
            derived: derived_out,
        })
    }

    /// Dry-run preview of `unforget_record_with_options`. Read-only.
    pub fn preview_unforget_record_with_options(
        &self,
        id: &RecordId,
        opts: &UnforgetCascadeOptions,
    ) -> Result<UnforgetCascadePreview> {
        let root = self.preview_unforget_record(id)?;
        if !opts.cascade_derived {
            return Ok(UnforgetCascadePreview {
                root,
                derived: Vec::new(),
            });
        }

        let descendants = self.collect_tombstone_descendants(id)?;
        let mut derived: Vec<DerivedUnforgetPreview> = Vec::with_capacity(descendants.len());
        for child_id in descendants {
            if let UnforgetRecordOutcome::Unforgotten(r) =
                self.preview_unforget_record(&child_id)?
            {
                derived.push(DerivedUnforgetPreview {
                    record_id: r.record_id,
                    adapter: r.adapter,
                    instance: r.instance,
                    native_id: r.native_id,
                    native_path: r.native_path,
                    raw_hash: r.raw_hash,
                    reason: r.reason,
                    forgotten_at: r.forgotten_at,
                });
            }
        }
        Ok(UnforgetCascadePreview { root, derived })
    }

    /// BFS through `record_tombstones.derived_from` from `start` (excluded);
    /// cycle-safe (`StoreError::Corruption` on revisit).
    fn collect_tombstone_descendants(&self, start: &RecordId) -> Result<Vec<RecordId>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT record_id FROM record_tombstones \
             WHERE derived_from = ?1 \
             ORDER BY record_id ASC",
        )?;

        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        visited.insert(start.0.clone());
        let mut frontier: Vec<RecordId> = vec![start.clone()];
        let mut out: Vec<RecordId> = Vec::new();

        while let Some(parent) = frontier.pop() {
            let children: Vec<String> = stmt
                .query_map(params![parent.0], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            for cid in children {
                if !visited.insert(cid.clone()) {
                    return Err(StoreError::Corruption(format!(
                        "tombstone derivation cycle detected at {cid}"
                    )));
                }
                out.push(RecordId(cid.clone()));
                frontier.push(RecordId(cid));
            }
        }
        Ok(out)
    }

    /// Paginated newest-first scan of `record_tombstones`, optionally scoped to
    /// `(adapter, instance)`. Read-only; `limit` is clamped to
    /// `[1, LIST_FORGOTTEN_MAX_LIMIT]`.
    pub fn list_forgotten(&self, filter: &ListForgottenFilter) -> Result<Vec<ForgottenRecord>> {
        let limit = filter.limit.clamp(1, LIST_FORGOTTEN_MAX_LIMIT);
        let conn = self.conn.lock();
        let mapper = |r: &rusqlite::Row<'_>| -> rusqlite::Result<ForgottenRecord> {
            Ok(ForgottenRecord {
                record_id: RecordId(r.get(0)?),
                adapter: r.get(1)?,
                instance: r.get(2)?,
                native_id: r.get(3)?,
                native_path: r.get(4)?,
                raw_hash: r.get(5)?,
                reason: r.get(6)?,
                forgotten_at: r.get(7)?,
            })
        };
        let rows: Vec<ForgottenRecord> = match (
            filter.source.as_deref(),
            filter.instance.as_deref(),
        ) {
            (Some(s), Some(i)) => {
                let mut stmt = conn.prepare(
                    "SELECT record_id, adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at \
                     FROM record_tombstones \
                     WHERE adapter = ?1 AND instance = ?2 \
                     ORDER BY forgotten_at DESC, record_id DESC \
                     LIMIT ?3",
                )?;
                let mapped = stmt
                    .query_map(params![s, i, limit as i64], mapper)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                mapped
            }
            (Some(s), None) => {
                let mut stmt = conn.prepare(
                    "SELECT record_id, adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at \
                     FROM record_tombstones \
                     WHERE adapter = ?1 \
                     ORDER BY forgotten_at DESC, record_id DESC \
                     LIMIT ?2",
                )?;
                let mapped = stmt
                    .query_map(params![s, limit as i64], mapper)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                mapped
            }
            (None, Some(i)) => {
                let mut stmt = conn.prepare(
                    "SELECT record_id, adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at \
                     FROM record_tombstones \
                     WHERE instance = ?1 \
                     ORDER BY forgotten_at DESC, record_id DESC \
                     LIMIT ?2",
                )?;
                let mapped = stmt
                    .query_map(params![i, limit as i64], mapper)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                mapped
            }
            (None, None) => {
                let mut stmt = conn.prepare(
                    "SELECT record_id, adapter, instance, native_id, native_path, raw_hash, reason, forgotten_at \
                     FROM record_tombstones \
                     ORDER BY forgotten_at DESC, record_id DESC \
                     LIMIT ?1",
                )?;
                let mapped = stmt
                    .query_map(params![limit as i64], mapper)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                mapped
            }
        };
        Ok(rows)
    }

    /// Round 90 (PR-78l): tombstone counts grouped by
    /// `(adapter, instance)`. Same `source` / `instance` filter as
    /// `list_forgotten` — the operator running
    /// `list-forgotten --source mem0 --include-counts` sees the
    /// total + breakdown across mem0 instances, not the entire
    /// store. `limit` is **ignored** here: counts always reflect
    /// the full matching set, not just the current page.
    ///
    /// Rows are ordered newest-tombstone first within the group
    /// via the existing `idx_record_tombstones_source` index
    /// (`(adapter, instance, forgotten_at DESC)`, migration
    /// 0007) so the helper stays O(log N + groups) even on a
    /// noisy source.
    pub fn count_forgotten_by_source(
        &self,
        filter: &ListForgottenFilter,
    ) -> Result<Vec<ForgottenSourceCount>> {
        let conn = self.conn.lock();
        let mut sql = String::from(
            "SELECT adapter, instance, COUNT(*) AS n \
             FROM record_tombstones \
             WHERE 1=1",
        );
        let mut bound: Vec<rusqlite::types::Value> = Vec::new();
        if let Some(s) = &filter.source {
            sql.push_str(" AND adapter = ?");
            bound.push(rusqlite::types::Value::Text(s.clone()));
        }
        if let Some(i) = &filter.instance {
            sql.push_str(" AND instance = ?");
            bound.push(rusqlite::types::Value::Text(i.clone()));
        }
        sql.push_str(
            " GROUP BY adapter, instance \
             ORDER BY n DESC, adapter ASC, instance ASC",
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(bound.iter()), |r| {
                Ok(ForgottenSourceCount {
                    adapter: r.get(0)?,
                    instance: r.get(1)?,
                    forgotten_count: r.get::<_, i64>(2)? as u64,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Round 94 (PR-78p): backing read for the MCP
    /// `summarize_my_preferences` prompt. Returns
    /// user-scope record summaries newest-first, optionally
    /// narrowed by a single normalised `user_tag` (filter
    /// pushes down at the SQL recall stage so a single tagged
    /// record surfaces even under a heavy untagged-majority
    /// corpus — same discipline as R79 search).
    ///
    /// Caller normalises `user_tag` via `normalize_user_tag_name`
    /// before passing it in; this method does NOT re-normalise so
    /// the error path stays with the caller (CLI / MCP layer).
    pub fn summarize_preferences_records(
        &self,
        limit: i64,
        user_tag: Option<&str>,
    ) -> Result<Vec<SummarizePreferencesRow>> {
        let conn = self.conn.lock();
        let row_map = |r: &rusqlite::Row<'_>| -> rusqlite::Result<SummarizePreferencesRow> {
            Ok(SummarizePreferencesRow {
                id: r.get(0)?,
                content: r.get(1)?,
                kind: r.get(2)?,
                native_path: r.get(3)?,
                created_at: r.get(4)?,
            })
        };
        let rows: Vec<SummarizePreferencesRow> = if let Some(tag) = user_tag {
            let mut stmt = conn.prepare(
                "SELECT id, content, kind, native_path, created_at \
                 FROM records r \
                 WHERE scope = 'user' \
                   AND EXISTS ( \
                     SELECT 1 FROM user_record_tags urt \
                      WHERE urt.record_id = r.id AND urt.tag = ?1 \
                   ) \
                 ORDER BY created_at DESC LIMIT ?2",
            )?;
            let mapped = stmt
                .query_map(params![tag, limit], row_map)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, content, kind, native_path, created_at FROM records \
                 WHERE scope = 'user' ORDER BY created_at DESC LIMIT ?1",
            )?;
            let mapped = stmt
                .query_map(params![limit], row_map)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        };
        Ok(rows)
    }

    /// Round 77 (PR-77): exact-duplicate report over `records.raw_hash`.
    ///
    /// Returns up to `limit` groups of records sharing a raw_hash,
    /// `>= 2` records each, ordered by group size (DESC), then by
    /// the group's newest `created_at` (DESC), then by raw_hash
    /// (ASC) for deterministic output. Tombstoned records were
    /// deleted from `records` by `forget_record` (R72) so they
    /// don't appear here automatically — `forget` is the operator's
    /// remediation action and this report shows what's left.
    ///
    /// **Exact** duplicates only: this matches the byte-identical
    /// source payload hash, not semantic similarity. Naming the
    /// API and the wire field around `raw_hash` is deliberate so
    /// nobody reads "dedupe" as "semantic merge."
    ///
    /// Read-only — never writes to the store.
    pub fn list_duplicate_raw_hashes(&self, limit: u32) -> Result<Vec<DuplicateRawHashGroup>> {
        self.list_duplicate_raw_hashes_filtered(&DuplicateRawHashFilter {
            source: None,
            instance: None,
            limit,
        })
    }

    /// Round 80: same as `list_duplicate_raw_hashes` but scopes
    /// the eligible groups to those whose member records include
    /// at least one matching `(adapter, instance)`. **Whole
    /// Round 131 (PR-78az): read every live record's content + minimal
    /// provenance for the near-duplicate detector
    /// (`semantic_dedupe::list_near_duplicates`). Returns the projection
    /// the algorithm needs and nothing more — the full record body is
    /// only ever held in memory during the SimHash pass.
    ///
    /// Walks live `records` only (no tombstoned rows: a forgotten
    /// memory shouldn't surface back through near-dedupe). Returns
    /// rows in arbitrary order; the detector doesn't depend on order
    /// and groups its own output newest-first inside each component.
    pub fn list_records_for_near_dedupe(
        &self,
    ) -> Result<Vec<crate::semantic_dedupe::NearDedupeScanRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, adapter, instance, native_id, native_path, \
                    content, kind, created_at, updated_at \
             FROM records",
        )?;
        let mapped = stmt
            .query_map([], |r| {
                let kind_str: String = r.get(6)?;
                let kind = match kind_str.as_str() {
                    "fact" => anamnesis_core::model::Kind::Fact,
                    "preference" => anamnesis_core::model::Kind::Preference,
                    "feedback" => anamnesis_core::model::Kind::Feedback,
                    "reference" => anamnesis_core::model::Kind::Reference,
                    "episode" => anamnesis_core::model::Kind::Episode,
                    "skill" => anamnesis_core::model::Kind::Skill,
                    _ => anamnesis_core::model::Kind::Unknown,
                };
                let native_path: Option<String> = r.get(4)?;
                Ok(crate::semantic_dedupe::NearDedupeScanRow {
                    record_id: anamnesis_core::model::RecordId(r.get::<_, String>(0)?),
                    adapter: r.get(1)?,
                    instance: r.get(2)?,
                    native_id: r.get(3)?,
                    has_native_path: native_path.is_some(),
                    content: r.get(5)?,
                    kind,
                    created_at: r.get(7)?,
                    updated_at: r.get(8)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(mapped)
    }

    /// groups** are returned — siblings outside the filter stay
    /// visible so the operator can decide which to `forget`.
    ///
    /// SQL shape (load-bearing): the `(adapter, instance)`
    /// constraint lives in the first-pass `GROUP BY raw_hash
    /// HAVING COUNT(*) > 1` *via a subquery on the eligible
    /// hashes*, NOT in the second-pass member fetch. Filtering
    /// after the outer `LIMIT` would let a huge irrelevant
    /// duplicate group consume the limit and starve the
    /// operator's actual target.
    pub fn list_duplicate_raw_hashes_filtered(
        &self,
        filter: &DuplicateRawHashFilter,
    ) -> Result<Vec<DuplicateRawHashGroup>> {
        let limit = filter.limit.clamp(1, LIST_DUPLICATE_RAW_HASHES_MAX_LIMIT);
        let conn = self.conn.lock();

        // First pass: pick the duplicate hashes, optionally
        // narrowed to groups that contain >=1 record matching
        // the source/instance filter. We use an inner subquery
        // to find the eligible hashes; the outer GROUP BY uses
        // them to enforce HAVING COUNT(*) > 1 across the *full*
        // record set (so a group with 1 mem0 + 5 claude-code is
        // still a duplicate group, not collapsed to a singleton
        // by the filter).
        // Round 104 (PR-78z): `filter.source` is parsed through
        // core's shared `parse_csv_filter` so a single adapter
        // (`"mem0"`) and a comma-separated OR list
        // (`"mem0,claude-code"`) both work. Empty parse = no
        // source filter (back-compat with R80).
        // Round 115: `filter.instance` now uses the same parser
        // and emits `instance IN (?, ?, ...)` when non-empty.
        let sources = anamnesis_core::parse_csv_filter(filter.source.as_deref());
        let instances = anamnesis_core::parse_csv_filter(filter.instance.as_deref());
        let mut sql = String::from(
            "SELECT raw_hash, COUNT(*) AS n, MAX(created_at) AS newest \
             FROM records \
             WHERE raw_hash IN (",
        );
        let mut eligible_params: Vec<rusqlite::types::Value> = Vec::new();
        if !sources.is_empty() || !instances.is_empty() {
            sql.push_str("SELECT raw_hash FROM records WHERE 1=1");
            if !sources.is_empty() {
                // `adapter IN (?, ?, ...)` — N placeholders
                // matching the parsed token count.
                let placeholders = std::iter::repeat_n("?", sources.len())
                    .collect::<Vec<_>>()
                    .join(",");
                sql.push_str(&format!(" AND adapter IN ({placeholders})"));
                for s in &sources {
                    eligible_params.push(rusqlite::types::Value::Text(s.clone()));
                }
            }
            if !instances.is_empty() {
                let placeholders = std::iter::repeat_n("?", instances.len())
                    .collect::<Vec<_>>()
                    .join(",");
                sql.push_str(&format!(" AND instance IN ({placeholders})"));
                for i in &instances {
                    eligible_params.push(rusqlite::types::Value::Text(i.clone()));
                }
            }
        } else {
            // No filter: trivially "all hashes are eligible."
            sql.push_str("SELECT raw_hash FROM records");
        }
        sql.push_str(
            ") \
             GROUP BY raw_hash \
             HAVING COUNT(*) > 1 \
             ORDER BY COUNT(*) DESC, MAX(created_at) DESC, raw_hash ASC \
             LIMIT ?",
        );
        eligible_params.push(rusqlite::types::Value::Integer(limit as i64));

        let hashes: Vec<String> = {
            let mut stmt = conn.prepare(&sql)?;
            let mapped = stmt
                .query_map(rusqlite::params_from_iter(eligible_params.iter()), |r| {
                    r.get::<_, String>(0)
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        };
        if hashes.is_empty() {
            return Ok(Vec::new());
        }

        // Second pass: pull every record for those hashes in one
        // IN() query — order so siblings are grouped contiguously
        // and the operator sees newest-first inside each group.
        // **Groups are not filtered at this stage**: the operator
        // needs the full sibling set to decide which to forget.
        let placeholders = std::iter::repeat_n("?", hashes.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT raw_hash, id, adapter, instance, native_id, native_path, \
                    created_at, updated_at \
             FROM records \
             WHERE raw_hash IN ({}) \
             ORDER BY raw_hash ASC, created_at DESC, id DESC",
            placeholders
        );
        let mut stmt = conn.prepare(&sql)?;
        let params_iter: Vec<&dyn rusqlite::ToSql> =
            hashes.iter().map(|h| h as &dyn rusqlite::ToSql).collect();
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params_iter), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    DuplicateRawHashRecord {
                        record_id: RecordId(r.get(1)?),
                        adapter: r.get(2)?,
                        instance: r.get(3)?,
                        native_id: r.get(4)?,
                        native_path: r.get(5)?,
                        created_at: r.get(6)?,
                        updated_at: r.get(7)?,
                    },
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut by_hash: std::collections::HashMap<String, Vec<DuplicateRawHashRecord>> =
            std::collections::HashMap::new();
        for (h, rec) in rows {
            by_hash.entry(h).or_default().push(rec);
        }
        let mut out = Vec::with_capacity(hashes.len());
        for h in hashes {
            if let Some(records) = by_hash.remove(&h) {
                out.push(DuplicateRawHashGroup {
                    raw_hash: h,
                    records,
                });
            }
        }
        Ok(out)
    }

    /// Round 97 (PR-78s): filter-scoped aggregate counts for
    /// `dedupe --include-counts`.
    ///
    /// Returns the **full** filter-scoped duplicate set, not
    /// just the current page. `filter.limit` is ignored.
    ///
    /// Semantic decisions (load-bearing):
    ///   * `total_groups` is the number of duplicate groups
    ///     that match the filter (same eligibility rule as
    ///     `list_duplicate_raw_hashes_filtered`: a group is
    ///     "matched" if ≥1 member matches `source`/`instance`).
    ///   * `total_records` is the sum of live records across
    ///     those whole groups — including non-matching siblings.
    ///   * `by_source[]` counts **records**, not group
    ///     memberships, because a mixed-source group belongs
    ///     to multiple sources and group-arithmetic would
    ///     double-count.
    pub fn count_duplicate_raw_hashes_by_source(
        &self,
        filter: &DuplicateRawHashFilter,
    ) -> Result<DuplicateRawHashCounts> {
        let conn = self.conn.lock();

        // First pass: pick the eligible duplicate hashes,
        // optionally narrowed by source/instance. Same shape as
        // `list_duplicate_raw_hashes_filtered` but without the
        // LIMIT — counts must reflect the full matching set.
        // Round 104 (PR-78z): same multi-source parser shared
        // with `list_duplicate_raw_hashes_filtered`. Counts must
        // honour the same eligibility rule as the list, so we
        // emit `adapter IN (?, ?, ...)` when the parsed token
        // list is non-empty.
        // Round 115: same rule for multi-instance OR.
        let sources = anamnesis_core::parse_csv_filter(filter.source.as_deref());
        let instances = anamnesis_core::parse_csv_filter(filter.instance.as_deref());
        let mut sql = String::from(
            "SELECT raw_hash, COUNT(*) AS n \
             FROM records \
             WHERE raw_hash IN (",
        );
        let mut bound: Vec<rusqlite::types::Value> = Vec::new();
        if !sources.is_empty() || !instances.is_empty() {
            sql.push_str("SELECT raw_hash FROM records WHERE 1=1");
            if !sources.is_empty() {
                let placeholders = std::iter::repeat_n("?", sources.len())
                    .collect::<Vec<_>>()
                    .join(",");
                sql.push_str(&format!(" AND adapter IN ({placeholders})"));
                for s in &sources {
                    bound.push(rusqlite::types::Value::Text(s.clone()));
                }
            }
            if !instances.is_empty() {
                let placeholders = std::iter::repeat_n("?", instances.len())
                    .collect::<Vec<_>>()
                    .join(",");
                sql.push_str(&format!(" AND instance IN ({placeholders})"));
                for i in &instances {
                    bound.push(rusqlite::types::Value::Text(i.clone()));
                }
            }
        } else {
            sql.push_str("SELECT raw_hash FROM records");
        }
        sql.push_str(
            ") \
             GROUP BY raw_hash \
             HAVING COUNT(*) > 1",
        );
        let group_rows: Vec<(String, i64)> = {
            let mut stmt = conn.prepare(&sql)?;
            let mapped = stmt
                .query_map(rusqlite::params_from_iter(bound.iter()), |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        };
        if group_rows.is_empty() {
            return Ok(DuplicateRawHashCounts::default());
        }
        let total_groups = group_rows.len() as u64;
        let total_records: u64 = group_rows.iter().map(|(_, n)| *n as u64).sum();

        // Second pass: per-`(adapter, instance)` record count
        // across all eligible groups (whole sibling set
        // included — that's how filtered membership works in
        // R80's API).
        let placeholders = std::iter::repeat_n("?", group_rows.len())
            .collect::<Vec<_>>()
            .join(",");
        let by_source_sql = format!(
            "SELECT adapter, instance, COUNT(*) AS n \
             FROM records \
             WHERE raw_hash IN ({placeholders}) \
             GROUP BY adapter, instance \
             ORDER BY n DESC, adapter ASC, instance ASC"
        );
        let hash_params: Vec<rusqlite::types::Value> = group_rows
            .iter()
            .map(|(h, _)| rusqlite::types::Value::Text(h.clone()))
            .collect();
        let by_source: Vec<DuplicateRawHashSourceCount> = {
            let mut stmt = conn.prepare(&by_source_sql)?;
            let mapped = stmt
                .query_map(rusqlite::params_from_iter(hash_params.iter()), |r| {
                    Ok(DuplicateRawHashSourceCount {
                        adapter: r.get(0)?,
                        instance: r.get(1)?,
                        duplicate_record_count: r.get::<_, i64>(2)? as u64,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        };

        Ok(DuplicateRawHashCounts {
            total_groups,
            total_records,
            by_source,
        })
    }

    /// Cross-adapter identity-disagreement detector: groups of live records
    /// sharing one `native_id` across ≥2 adapters with ≥2 distinct content
    /// values. Filter (R104 CSV grammar) scopes group eligibility but never
    /// trims siblings — operator needs the full set to act. Ordering:
    /// `content_variant_count DESC, native_id ASC`; within a group,
    /// `(adapter ASC, record_id ASC)` for stable variant numbering.
    /// `content` is preview-only when `include_content = true`.
    pub fn list_native_content_conflicts_filtered(
        &self,
        filter: &NativeConflictFilter,
    ) -> Result<Vec<NativeConflictGroup>> {
        let limit = filter.limit.clamp(1, LIST_NATIVE_CONFLICTS_MAX_LIMIT);
        let sources = anamnesis_core::parse_csv_filter(filter.source.as_deref());
        let instances = anamnesis_core::parse_csv_filter(filter.instance.as_deref());

        let conn = self.conn.lock();

        // Pick eligible native_ids (≥2 rows, ≥2 adapters, ≥2 content variants,
        // ≥1 filter match). Outer GROUP BY runs against the *full* row set so
        // siblings outside the filter still belong to the group.
        let mut sql = String::from(
            "SELECT native_id, COUNT(DISTINCT content) AS variant_count \
             FROM records \
             WHERE native_id IN (",
        );
        let mut eligible_params: Vec<rusqlite::types::Value> = Vec::new();
        if !sources.is_empty() || !instances.is_empty() {
            sql.push_str("SELECT DISTINCT native_id FROM records WHERE 1=1");
            if !sources.is_empty() {
                let placeholders = vec!["?"; sources.len()].join(", ");
                sql.push_str(&format!(" AND adapter IN ({placeholders})"));
                for s in &sources {
                    eligible_params.push(rusqlite::types::Value::Text(s.clone()));
                }
            }
            if !instances.is_empty() {
                let placeholders = vec!["?"; instances.len()].join(", ");
                sql.push_str(&format!(" AND instance IN ({placeholders})"));
                for s in &instances {
                    eligible_params.push(rusqlite::types::Value::Text(s.clone()));
                }
            }
        } else {
            // No filter — every distinct native_id is eligible.
            sql.push_str("SELECT DISTINCT native_id FROM records");
        }
        sql.push_str(
            ") \
             GROUP BY native_id \
             HAVING COUNT(*) > 1 \
                AND COUNT(DISTINCT adapter) > 1 \
                AND COUNT(DISTINCT content) > 1 \
             ORDER BY variant_count DESC, native_id ASC \
             LIMIT ?",
        );
        eligible_params.push(rusqlite::types::Value::Integer(limit as i64));

        let group_rows: Vec<(String, u32)> = {
            let mut stmt = conn.prepare(&sql)?;
            let mapped = stmt
                .query_map(rusqlite::params_from_iter(eligible_params.iter()), |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as u32))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        };

        if group_rows.is_empty() {
            return Ok(Vec::new());
        }

        // Second pass: fetch every member of every qualifying group.
        // One query for the whole batch is cheaper than N round-trips.
        let placeholders = vec!["?"; group_rows.len()].join(", ");
        let member_sql = format!(
            "SELECT id, adapter, instance, native_id, native_path, content, created_at, updated_at \
             FROM records \
             WHERE native_id IN ({placeholders}) \
             ORDER BY native_id ASC, adapter ASC, id ASC"
        );
        let member_params: Vec<rusqlite::types::Value> = group_rows
            .iter()
            .map(|(nid, _)| rusqlite::types::Value::Text(nid.clone()))
            .collect();

        // Carry raw content along so we can compute the per-group
        // variant index without re-fetching. Content stays in-memory
        // only; the public `NativeConflictRecord.content_preview`
        // gates exposure on the caller's `include_content` flag.
        struct RawMember {
            record_id: RecordId,
            adapter: String,
            instance: String,
            native_id: String,
            has_native_path: bool,
            content: String,
            created_at: i64,
            updated_at: Option<i64>,
        }
        let mut by_group: HashMap<String, Vec<RawMember>> = HashMap::new();
        {
            let mut stmt = conn.prepare(&member_sql)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(member_params.iter()), |r| {
                Ok(RawMember {
                    record_id: RecordId(r.get::<_, String>(0)?),
                    adapter: r.get(1)?,
                    instance: r.get(2)?,
                    native_id: r.get(3)?,
                    has_native_path: r.get::<_, Option<String>>(4)?.is_some(),
                    content: r.get(5)?,
                    created_at: r.get(6)?,
                    updated_at: r.get(7)?,
                })
            })?;
            for row in rows {
                let row = row?;
                by_group.entry(row.native_id.clone()).or_default().push(row);
            }
        }

        // Build groups in the same order the eligibility query
        // returned. Per-group: assign 1-based `content_variant`
        // indices by first-occurrence of each unique content
        // string (stable adapter-asc, id-asc traversal).
        let mut out: Vec<NativeConflictGroup> = Vec::with_capacity(group_rows.len());
        for (native_id, variant_count) in &group_rows {
            let Some(members) = by_group.remove(native_id) else {
                continue;
            };
            let mut content_to_variant: HashMap<String, u32> = HashMap::new();
            let mut next_variant: u32 = 1;
            let records: Vec<NativeConflictRecord> = members
                .into_iter()
                .map(|m| {
                    let variant =
                        *content_to_variant
                            .entry(m.content.clone())
                            .or_insert_with(|| {
                                let v = next_variant;
                                next_variant += 1;
                                v
                            });
                    let content_preview = if filter.include_content {
                        Some(truncate_preview(&m.content, NATIVE_CONFLICT_PREVIEW_CHARS))
                    } else {
                        None
                    };
                    NativeConflictRecord {
                        record_id: m.record_id,
                        adapter: m.adapter,
                        instance: m.instance,
                        native_id: m.native_id,
                        has_native_path: m.has_native_path,
                        created_at: m.created_at,
                        updated_at: m.updated_at,
                        content_variant: variant,
                        content_preview,
                    }
                })
                .collect();
            out.push(NativeConflictGroup {
                native_id: native_id.clone(),
                records,
                content_variant_count: *variant_count,
            });
        }
        Ok(out)
    }

    /// Dry-run [`Store::accept_native_conflict_variant`]: partitions the conflict
    /// group into keep/forget sets without mutating the store.
    pub fn preview_accept_native_conflict_variant(
        &self,
        opts: &AcceptConflictOptions,
    ) -> Result<AcceptConflictOutcome> {
        let plan = self.plan_accept_conflict(opts)?;
        let mut cascade: Vec<AcceptConflictDescendant> = Vec::new();
        if opts.cascade_derived {
            for loser in &plan.forget_records {
                for child in self.collect_descendants(&loser.record_id)? {
                    let snap = self.snapshot_descendant_for_preview(&child)?;
                    if let Some(d) = snap {
                        cascade.push(d);
                    }
                }
            }
        }
        Ok(AcceptConflictOutcome {
            native_id: opts.native_id.clone(),
            keep_variant: plan.keep_variant,
            keep_records: plan.keep_records,
            forget_records: plan.forget_records,
            cascade_derived: cascade,
            dry_run: true,
        })
    }

    /// Resolve one `native_id` content conflict by keeping the chosen
    /// variant and tombstoning every loser. Writes happen inside one
    /// IMMEDIATE transaction; partial-apply is impossible.
    /// `cascade_derived=true` also tombstones loser descendants
    /// (`provenance.derived_from`); kept records never lose descendants.
    pub fn accept_native_conflict_variant(
        &self,
        opts: &AcceptConflictOptions,
    ) -> Result<AcceptConflictOutcome> {
        let plan = self.plan_accept_conflict(opts)?;
        let mut loser_descendants: Vec<RecordId> = Vec::new();
        if opts.cascade_derived {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            for loser in &plan.forget_records {
                for child in self.collect_descendants(&loser.record_id)? {
                    if seen.insert(child.0.clone()) {
                        loser_descendants.push(child);
                    }
                }
            }
        }

        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let now = chrono::Utc::now().timestamp();
        let reason = opts.reason.as_deref();

        for loser in &plan.forget_records {
            forget_one_in_tx(&tx, &loser.record_id, reason, now)?;
        }

        let mut cascade: Vec<AcceptConflictDescendant> =
            Vec::with_capacity(loser_descendants.len());
        for child_id in &loser_descendants {
            let outcome = forget_one_in_tx(&tx, child_id, None, now)?;
            match outcome {
                ForgetRecordOutcome::Forgotten(r) => cascade.push(AcceptConflictDescendant {
                    record_id: r.record_id,
                    adapter: r.adapter,
                    instance: r.instance,
                    native_id: r.native_id,
                    forgotten_at: Some(r.forgotten_at),
                    was_already_forgotten: false,
                }),
                ForgetRecordOutcome::AlreadyForgotten(r) => {
                    cascade.push(AcceptConflictDescendant {
                        record_id: r.record_id,
                        adapter: r.adapter,
                        instance: r.instance,
                        native_id: r.native_id,
                        forgotten_at: Some(r.forgotten_at),
                        was_already_forgotten: true,
                    });
                }
                ForgetRecordOutcome::NotFound => {
                    // Raced away between snapshot and apply — cascade
                    // goal (no live derived row) is already satisfied.
                }
            }
        }
        tx.commit()?;

        Ok(AcceptConflictOutcome {
            native_id: opts.native_id.clone(),
            keep_variant: plan.keep_variant,
            keep_records: plan.keep_records,
            forget_records: plan.forget_records,
            cascade_derived: cascade,
            dry_run: false,
        })
    }

    /// Shared planning step: read the conflict group, validate the
    /// selector, partition records into keep/forget. Read-only.
    fn plan_accept_conflict(&self, opts: &AcceptConflictOptions) -> Result<AcceptConflictPlan> {
        // Read every live record sharing this native_id.
        type Row = (RecordId, String, String, String, String);
        let rows: Vec<Row> = {
            let conn = self.conn.lock();
            let mut stmt = conn.prepare(
                "SELECT id, adapter, instance, native_id, content \
                 FROM records WHERE native_id = ?1 \
                 ORDER BY adapter ASC, id ASC",
            )?;
            let mapped = stmt
                .query_map(params![opts.native_id], |r| {
                    Ok((
                        RecordId(r.get::<_, String>(0)?),
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, String>(4)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        };
        if rows.is_empty() {
            return Err(StoreError::Corruption(format!(
                "accept_conflict: no live records for native_id {:?}",
                opts.native_id
            )));
        }

        // Same variant numbering as `list_native_content_conflicts_filtered`:
        // first-occurrence in `(adapter ASC, id ASC)` order.
        let mut variants: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut next_variant: u32 = 1;
        let mut decorated: Vec<(RecordId, String, String, String, u32)> =
            Vec::with_capacity(rows.len());
        for (rid, adapter, instance, native_id, content) in rows {
            let variant = *variants.entry(content).or_insert_with(|| {
                let v = next_variant;
                next_variant += 1;
                v
            });
            decorated.push((rid, adapter, instance, native_id, variant));
        }
        let variant_count = next_variant - 1;
        let adapter_count = decorated
            .iter()
            .map(|r| r.1.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();
        if decorated.len() < 2 || adapter_count < 2 || variant_count < 2 {
            return Err(StoreError::Corruption(format!(
                "accept_conflict: native_id {:?} is not a cross-adapter content conflict \
                 (records={}, adapters={}, variants={})",
                opts.native_id,
                decorated.len(),
                adapter_count,
                variant_count
            )));
        }

        let keep_variant = match &opts.selector {
            AcceptConflictSelector::KeepVariant(v) => {
                if *v < 1 || *v > variant_count {
                    return Err(StoreError::Corruption(format!(
                        "accept_conflict: keep_variant={v} out of range [1, {variant_count}] \
                         for native_id {:?}",
                        opts.native_id
                    )));
                }
                *v
            }
            AcceptConflictSelector::KeepRecordId(id) => {
                let found = decorated.iter().find(|r| &r.0 == id).ok_or_else(|| {
                    StoreError::Corruption(format!(
                        "accept_conflict: keep_record_id {:?} is not a member of \
                         native_id {:?}'s conflict group",
                        id.0, opts.native_id
                    ))
                })?;
                found.4
            }
        };

        let mut keep_records: Vec<AcceptConflictRecord> = Vec::new();
        let mut forget_records: Vec<AcceptConflictRecord> = Vec::new();
        for (rid, adapter, instance, native_id, variant) in decorated {
            let decision = if variant == keep_variant {
                "keep"
            } else {
                "forget"
            };
            let row = AcceptConflictRecord {
                record_id: rid,
                adapter,
                instance,
                native_id,
                content_variant: variant,
                decision,
            };
            if variant == keep_variant {
                keep_records.push(row);
            } else {
                forget_records.push(row);
            }
        }
        if keep_records.is_empty() || forget_records.is_empty() {
            return Err(StoreError::Corruption(format!(
                "accept_conflict: selector for native_id {:?} would keep \
                 every (or no) record — nothing to resolve",
                opts.native_id
            )));
        }
        Ok(AcceptConflictPlan {
            keep_variant,
            keep_records,
            forget_records,
        })
    }

    /// Read-only descendant snapshot for the dry-run path.
    fn snapshot_descendant_for_preview(
        &self,
        id: &RecordId,
    ) -> Result<Option<AcceptConflictDescendant>> {
        let conn = self.conn.lock();
        let live: Option<(String, String, String)> = conn
            .query_row(
                "SELECT adapter, instance, native_id FROM records WHERE id = ?1",
                params![id.0],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        if let Some((adapter, instance, native_id)) = live {
            return Ok(Some(AcceptConflictDescendant {
                record_id: id.clone(),
                adapter,
                instance,
                native_id,
                forgotten_at: None,
                was_already_forgotten: false,
            }));
        }
        let tomb: Option<(String, String, String, i64)> = conn
            .query_row(
                "SELECT adapter, instance, native_id, forgotten_at \
                 FROM record_tombstones WHERE record_id = ?1",
                params![id.0],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()?;
        Ok(tomb.map(
            |(adapter, instance, native_id, ts)| AcceptConflictDescendant {
                record_id: id.clone(),
                adapter,
                instance,
                native_id,
                forgotten_at: Some(ts),
                was_already_forgotten: true,
            },
        ))
    }

    /// Cross-adapter drift reconciliation. For each side, identity key =
    /// `metadata.anamnesis_native_id` (round-tripped, safe to compare)
    /// when present, else `provenance.native_id` (per-adapter, only
    /// meaningful when adapters share an upstream source).
    ///
    /// Buckets: `only_left` / `only_right` / `both` / `conflicts`
    /// (subset of `both` where `content` differs). Sample arrays are
    /// capped at `opts.limit`; counts ignore the cap.
    pub fn reconcile_sources(&self, opts: &ReconcileOptions) -> Result<ReconcileOutcome> {
        let limit = opts.limit.clamp(1, RECONCILE_MAX_LIMIT) as usize;

        // (identity_key, identity_source, record_id, kind, scope, created_at, content)
        type SideRow = (String, &'static str, RecordId, String, String, i64, String);
        let read_side = |sel: &ReconcileSourceSelector| -> Result<Vec<SideRow>> {
            let inst = sel.instance.clone().unwrap_or_default();
            let conn = self.conn.lock();
            let mut stmt = conn.prepare(
                "SELECT id, kind, scope, created_at, content, metadata \
                 FROM records WHERE adapter = ?1 AND instance = ?2 \
                 ORDER BY id ASC",
            )?;
            let rows = stmt
                .query_map(params![sel.adapter, inst], |r| {
                    let id: String = r.get(0)?;
                    let kind: String = r.get(1)?;
                    let scope: String = r.get(2)?;
                    let created_at: i64 = r.get(3)?;
                    let content: String = r.get(4)?;
                    let metadata_json: Option<String> = r.get(5)?;
                    Ok((id, kind, scope, created_at, content, metadata_json))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            let mut out: Vec<SideRow> = Vec::with_capacity(rows.len());
            for (rid, kind, scope, created_at, content, metadata_json) in rows {
                // Decide identity_key. Prefer the round-tripped
                // `anamnesis_native_id` (cross-adapter stable) over
                // `provenance.native_id` (per-adapter). Native id read
                // from a per-row query keeps the read pipelined.
                let metadata: serde_json::Value = metadata_json
                    .as_deref()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(serde_json::Value::Null);
                let anamnesis_native_id = metadata
                    .get("anamnesis_native_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned);
                let native_id: Option<String> = conn
                    .query_row(
                        "SELECT native_id FROM records WHERE id = ?1",
                        params![rid],
                        |r| r.get::<_, String>(0),
                    )
                    .optional()?;
                let (identity_key, identity_source): (String, &'static str) =
                    match (anamnesis_native_id, native_id) {
                        (Some(a), _) => (a, "anamnesis_native_id"),
                        (None, Some(n)) => (n, "native_id"),
                        (None, None) => continue, // skip pathological rows
                    };
                out.push((
                    identity_key,
                    identity_source,
                    RecordId(rid),
                    kind,
                    scope,
                    created_at,
                    content,
                ));
            }
            Ok(out)
        };

        let left_rows = read_side(&opts.left)?;
        let right_rows = read_side(&opts.right)?;

        let mut left_by_id: std::collections::HashMap<String, &SideRow> =
            std::collections::HashMap::with_capacity(left_rows.len());
        for row in &left_rows {
            left_by_id.entry(row.0.clone()).or_insert(row);
        }
        let mut right_by_id: std::collections::HashMap<String, &SideRow> =
            std::collections::HashMap::with_capacity(right_rows.len());
        for row in &right_rows {
            right_by_id.entry(row.0.clone()).or_insert(row);
        }

        let make_sample = |row: &SideRow, include_id: bool| ReconcileSample {
            record_id: row.2.clone(),
            kind: row.3.clone(),
            scope: row.4.clone(),
            created_at: row.5,
            identity_key: if include_id {
                Some(row.0.clone())
            } else {
                None
            },
            identity_source: row.1,
        };

        let mut only_left: Vec<ReconcileSample> = Vec::new();
        let mut only_left_total: u64 = 0;
        for row in &left_rows {
            if !right_by_id.contains_key(&row.0) {
                only_left_total += 1;
                if only_left.len() < limit {
                    only_left.push(make_sample(row, opts.include_identity));
                }
            }
        }
        let mut only_right: Vec<ReconcileSample> = Vec::new();
        let mut only_right_total: u64 = 0;
        for row in &right_rows {
            if !left_by_id.contains_key(&row.0) {
                only_right_total += 1;
                if only_right.len() < limit {
                    only_right.push(make_sample(row, opts.include_identity));
                }
            }
        }
        let mut both_total: u64 = 0;
        let mut conflicts: Vec<ReconcileSample> = Vec::new();
        let mut conflict_total: u64 = 0;
        for row in &left_rows {
            if let Some(rr) = right_by_id.get(&row.0) {
                both_total += 1;
                if row.6 != rr.6 {
                    conflict_total += 1;
                    if conflicts.len() < limit {
                        conflicts.push(make_sample(row, opts.include_identity));
                    }
                }
            }
        }

        Ok(ReconcileOutcome {
            left: opts.left.clone(),
            right: opts.right.clone(),
            counts: ReconcileCounts {
                only_left: only_left_total,
                only_right: only_right_total,
                both: both_total,
                conflicts: conflict_total,
                left_total: left_rows.len() as u64,
                right_total: right_rows.len() as u64,
            },
            samples: ReconcileSamples {
                only_left,
                only_right,
                conflicts,
            },
        })
    }

    /// Return the **full** record-id list for a single reconcile bucket
    /// (uncapped, sorted record_id ASC). Built for R147 reconcile-export
    /// — `run_export_with_ids` consumes the result and writes the diff
    /// into a fresh round-trip file. Read-only.
    pub fn reconcile_bucket_ids(
        &self,
        left: &ReconcileSourceSelector,
        right: &ReconcileSourceSelector,
        bucket: ReconcileBucket,
    ) -> Result<Vec<RecordId>> {
        // Re-uses the same identity-key resolution as `reconcile_sources`
        // so a follow-up export aligns 1:1 with the diff the operator saw.
        type SideRow = (String, RecordId);
        let read_side = |sel: &ReconcileSourceSelector| -> Result<Vec<SideRow>> {
            let inst = sel.instance.clone().unwrap_or_default();
            let conn = self.conn.lock();
            let mut stmt = conn.prepare(
                "SELECT id, metadata, native_id \
                 FROM records WHERE adapter = ?1 AND instance = ?2 \
                 ORDER BY id ASC",
            )?;
            let rows = stmt
                .query_map(params![sel.adapter, inst], |r| {
                    let id: String = r.get(0)?;
                    let metadata_json: Option<String> = r.get(1)?;
                    let native_id: Option<String> = r.get(2)?;
                    Ok((id, metadata_json, native_id))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            let mut out: Vec<SideRow> = Vec::with_capacity(rows.len());
            for (rid, metadata_json, native_id) in rows {
                let metadata: serde_json::Value = metadata_json
                    .as_deref()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(serde_json::Value::Null);
                let identity = metadata
                    .get("anamnesis_native_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
                    .or(native_id);
                if let Some(key) = identity {
                    out.push((key, RecordId(rid)));
                }
            }
            Ok(out)
        };
        let left_rows = read_side(left)?;
        let right_rows = read_side(right)?;
        let left_keys: std::collections::HashSet<String> =
            left_rows.iter().map(|r| r.0.clone()).collect();
        let right_keys: std::collections::HashSet<String> =
            right_rows.iter().map(|r| r.0.clone()).collect();
        let bucket_rows = match bucket {
            ReconcileBucket::OnlyLeft => left_rows
                .into_iter()
                .filter(|(k, _)| !right_keys.contains(k))
                .map(|(_, id)| id)
                .collect(),
            ReconcileBucket::OnlyRight => right_rows
                .into_iter()
                .filter(|(k, _)| !left_keys.contains(k))
                .map(|(_, id)| id)
                .collect(),
        };
        Ok(bucket_rows)
    }

    /// Round 78 (PR-78): apply or remove user-tags on a record.
    ///
    /// **Set semantics.** Re-adding an existing tag is a no-op
    /// (the `changed` count tells the caller how many rows
    /// actually moved). Removing a missing tag is also a no-op.
    /// This lets the CLI / MCP surface stay idempotent without
    /// the caller having to first read the tags to figure out
    /// what's actually new.
    ///
    /// Tag normalisation runs *before* the FK check on
    /// `records.id`, so callers get a clear "your tag input is
    /// bad" error instead of a SQL constraint failure. Rules:
    ///   * `trim().to_lowercase()`
    ///   * dedup (preserve input order)
    ///   * reject empty
    ///   * reject any control character (no `\n`, `\t`, …)
    ///   * each tag ≤ `USER_TAG_MAX_LEN` bytes after trim
    ///   * at most `TAG_RECORD_MAX_BATCH` tags per call
    ///
    /// If the record id doesn't exist, returns
    /// `StoreError::Corruption` (the operator typoed the id,
    /// likely from `list-forgotten` or `search`).
    pub fn tag_record(
        &self,
        id: &RecordId,
        tags: &[String],
        operation: UserTagOperation,
    ) -> Result<UserTagMutation> {
        // Round 81: Replace is the only operation that accepts an
        // empty tag list (= "clear all tags"). Add/Remove are
        // no-ops on empty input so we keep rejecting them — the
        // caller probably forgot a positional arg.
        let requested = if matches!(operation, UserTagOperation::Replace) && tags.is_empty() {
            Vec::new()
        } else {
            normalize_user_tags(tags)?
        };
        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        // Refuse early when the record doesn't exist. The FK on
        // `user_record_tags.record_id` would catch this too, but
        // the error message is friendlier here.
        let exists: i64 = tx.query_row(
            "SELECT COUNT(1) FROM records WHERE id = ?1",
            params![id.0],
            |r| r.get(0),
        )?;
        if exists == 0 {
            return Err(StoreError::Corruption(format!(
                "tag_record: no live record with id {:?}; \
                 forget/unforget cycle erases tags via FK cascade",
                id.0
            )));
        }

        let mut changed = 0u32;
        match operation {
            UserTagOperation::Add => {
                let now = chrono::Utc::now().timestamp();
                for t in &requested {
                    let n = tx.execute(
                        "INSERT INTO user_record_tags(record_id, tag, created_at) \
                         VALUES (?1, ?2, ?3) \
                         ON CONFLICT(record_id, tag) DO NOTHING",
                        params![id.0, t, now],
                    )?;
                    changed += n as u32;
                }
            }
            UserTagOperation::Remove => {
                for t in &requested {
                    let n = tx.execute(
                        "DELETE FROM user_record_tags WHERE record_id = ?1 AND tag = ?2",
                        params![id.0, t],
                    )?;
                    changed += n as u32;
                }
            }
            UserTagOperation::Replace => {
                // Read the current set inside the same Immediate
                // transaction so a concurrent writer can't slip
                // between the read and the delete/insert.
                let current: std::collections::BTreeSet<String> = {
                    let mut stmt =
                        tx.prepare("SELECT tag FROM user_record_tags WHERE record_id = ?1")?;
                    let mapped = stmt
                        .query_map(params![id.0], |r| r.get::<_, String>(0))?
                        .collect::<rusqlite::Result<std::collections::BTreeSet<_>>>()?;
                    mapped
                };
                let target: std::collections::BTreeSet<String> =
                    requested.iter().cloned().collect();
                // Set delta drives `changed` — re-replacing with
                // the same set reports 0, matching Add/Remove's
                // idempotent semantic. We don't count the raw
                // DELETE+INSERT row totals.
                let to_remove: Vec<&String> = current.difference(&target).collect();
                let to_add: Vec<&String> = target.difference(&current).collect();
                for t in &to_remove {
                    tx.execute(
                        "DELETE FROM user_record_tags WHERE record_id = ?1 AND tag = ?2",
                        params![id.0, t],
                    )?;
                }
                let now = chrono::Utc::now().timestamp();
                for t in &to_add {
                    tx.execute(
                        "INSERT INTO user_record_tags(record_id, tag, created_at) \
                         VALUES (?1, ?2, ?3) \
                         ON CONFLICT(record_id, tag) DO NOTHING",
                        params![id.0, t, now],
                    )?;
                }
                changed = (to_add.len() + to_remove.len()) as u32;
            }
        }

        // Post-call set so the caller can render the new state.
        let user_tags: Vec<String> = {
            let mut stmt = tx.prepare(
                "SELECT tag FROM user_record_tags WHERE record_id = ?1 ORDER BY tag ASC",
            )?;
            let mapped = stmt
                .query_map(params![id.0], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        };

        tx.commit()?;
        Ok(UserTagMutation {
            record_id: id.clone(),
            operation,
            requested,
            changed,
            user_tags,
        })
    }

    /// Round 78: list the user tags on one record. Sorted
    /// ASCII-ascending. Empty vector for records that have
    /// never been tagged (the common case).
    pub fn user_tags(&self, id: &RecordId) -> Result<Vec<String>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare("SELECT tag FROM user_record_tags WHERE record_id = ?1 ORDER BY tag ASC")?;
        let rows = stmt
            .query_map(params![id.0], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Round 78: batched variant of `user_tags`. Used internally
    /// by `get_record_headers_by_ids` so the search packer pays
    /// one round-trip for the overlay, not N. Returned map omits
    /// ids with zero tags (so the caller can default to empty
    /// without paying for absent rows).
    pub fn user_tags_by_ids(
        &self,
        ids: &[RecordId],
    ) -> Result<std::collections::HashMap<RecordId, Vec<String>>> {
        if ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT record_id, tag \
             FROM user_record_tags \
             WHERE record_id IN ({}) \
             ORDER BY record_id ASC, tag ASC",
            placeholders
        );
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(&sql)?;
        let params_iter: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|id| &id.0 as &dyn rusqlite::ToSql).collect();
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params_iter), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut out: std::collections::HashMap<RecordId, Vec<String>> =
            std::collections::HashMap::new();
        for (rid, tag) in rows {
            out.entry(RecordId(rid)).or_default().push(tag);
        }
        Ok(out)
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

/// Round 72 (PR-72a): is this `(adapter, instance, native_id)`
/// triple in `record_tombstones`? Used by the upsert paths to
/// short-circuit a forgotten record before any chunking work.
///
/// Lives as a free fn (not a method) so it can be called from
/// inside an already-open `Transaction`. The store-public
/// [`Store::forget_record`] guarantees the tombstone is keyed on
/// the same triple every adapter's `RecordId::from_parts` builds.
/// Round 79 (PR-78b): per-tag normalisation. Pure function —
/// trim + lowercase + bound + reject empty/control. Shared
/// between `tag_record` writes and `SearchFilter.user_tag`
/// reads so a tag written as `Keep` and queried as `Keep`
/// both normalise to `keep` and the read hits.
pub fn normalize_user_tag_name(raw: &str) -> Result<String> {
    let normalised = raw.trim().to_lowercase();
    if normalised.is_empty() {
        return Err(StoreError::Corruption(
            "user tag: empty (after trim) is not allowed".into(),
        ));
    }
    if normalised.len() > USER_TAG_MAX_LEN {
        return Err(StoreError::Corruption(format!(
            "user tag: {normalised:?} exceeds {USER_TAG_MAX_LEN}-byte limit"
        )));
    }
    if normalised.chars().any(|c| c.is_control()) {
        return Err(StoreError::Corruption(format!(
            "user tag: {normalised:?} contains a control character"
        )));
    }
    Ok(normalised)
}

/// Round 78: normalise the caller's tag list before the SQL
/// write. Returns deduped normalised tags ready to insert/delete.
/// Round 79: delegates per-tag work to `normalize_user_tag_name`
/// so write + filter paths share a single source of truth.
fn normalize_user_tags(raw: &[String]) -> Result<Vec<String>> {
    if raw.is_empty() {
        return Err(StoreError::Corruption(
            "tag_record: at least one tag is required".into(),
        ));
    }
    if raw.len() > TAG_RECORD_MAX_BATCH {
        return Err(StoreError::Corruption(format!(
            "tag_record: too many tags in one call ({}); max is {}",
            raw.len(),
            TAG_RECORD_MAX_BATCH
        )));
    }
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out: Vec<String> = Vec::with_capacity(raw.len());
    for t in raw {
        let normalised = normalize_user_tag_name(t)?;
        if seen.insert(normalised.clone()) {
            out.push(normalised);
        }
    }
    Ok(out)
}

fn record_is_tombstoned(tx: &Transaction<'_>, r: &AnamnesisRecord) -> Result<bool> {
    let instance = r.source.instance.as_deref().unwrap_or("");
    let n: i64 = tx.query_row(
        "SELECT COUNT(1) FROM record_tombstones \
         WHERE adapter = ?1 AND instance = ?2 AND native_id = ?3",
        params![&r.source.adapter, instance, &r.provenance.native_id],
        |row| row.get(0),
    )?;
    Ok(n > 0)
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
            native_id, native_path, captured_at, raw_hash, schema_version, \
            derived_from\
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16) \
         ON CONFLICT(id) DO UPDATE SET \
            content = excluded.content, \
            scope = excluded.scope, \
            kind = excluded.kind, \
            updated_at = excluded.updated_at, \
            tags = excluded.tags, \
            metadata = excluded.metadata, \
            native_path = excluded.native_path, \
            raw_hash = excluded.raw_hash, \
            derived_from = excluded.derived_from",
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
            r.provenance.derived_from.as_ref().map(|rid| rid.0.clone()),
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
    // Re-chunking is a clean replace. The BLOB table has FK ON DELETE
    // CASCADE, but vec0 virtual tables don't honor FKs — manually drop
    // any vec0 rows for this record's chunks before the BLOBs go away.
    vec_ext::delete_vec_rows_for_record(tx, &r.id.0)?;
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
    /// Return the most recently created record ids, newest first.
    ///
    /// Used by MCP `resources/list` to enumerate concrete record URIs
    /// — generic-mcp loopback (Anamnesis → Anamnesis) needs real URIs
    /// to consume, not just `anamnesis://record/{id}` templates that
    /// the adapter (correctly) filters out.
    ///
    /// `limit` is bounded — the resource catalogue is meant to be a
    /// window into the store, not a full dump. 100 is a reasonable
    /// default for "what's recent enough to be worth surfacing".
    pub fn list_recent_record_ids(&self, limit: u32) -> Result<Vec<String>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT id FROM records ORDER BY created_at DESC LIMIT ?1")?;
        let rows = stmt
            .query_map(params![limit], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Paged listing of record ids for **complete migration**.
    ///
    /// Round-21 (§-1.5 PR-2): the original `list_recent_record_ids` is
    /// a "what's recent" window; this is "give me everything, page by
    /// page" so a downstream generic-mcp client can pull the entire
    /// catalogue without dropping records past the 100-row cap.
    ///
    /// Ordering: lexicographic ascending by id. Record ids are
    /// content-derived (blake3 of provenance triple), so the order is
    /// stable across calls and across hosts — making cursor-based
    /// pagination an opaque string the client just round-trips.
    ///
    /// Contract:
    ///   * `cursor = None` → return the first `limit` ids.
    ///   * `cursor = Some(last_id)` → return the next `limit` ids
    ///     STRICTLY AFTER `last_id` in ascending order.
    ///   * `limit` is clamped to `[1, MAX_LIST_LIMIT]`.
    ///   * Returns `(ids, next_cursor)`. `next_cursor` is `Some(last)`
    ///     when the page hit the limit (i.e. another page may exist),
    ///     `None` when the page returned fewer than `limit` rows (= end
    ///     of catalogue).
    pub fn list_record_ids_paged(
        &self,
        cursor: Option<&str>,
        limit: u32,
    ) -> Result<(Vec<String>, Option<String>)> {
        let limit = limit.clamp(1, MAX_LIST_LIMIT);
        let conn = self.conn.lock();
        // `stmt` must outlive the iterator from `query_map`; bind it
        // explicitly in each branch to give the iterator a stable
        // borrow for the duration of the `collect`.
        let rows: Vec<String> = match cursor {
            Some(c) => {
                let mut stmt =
                    conn.prepare("SELECT id FROM records WHERE id > ?1 ORDER BY id ASC LIMIT ?2")?;
                let out = stmt
                    .query_map(params![c, limit], |r| r.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                out
            }
            None => {
                let mut stmt = conn.prepare("SELECT id FROM records ORDER BY id ASC LIMIT ?1")?;
                let out = stmt
                    .query_map(params![limit], |r| r.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                out
            }
        };
        // `next_cursor` is the last row's id IFF we hit the limit —
        // otherwise we're at the end and signal that to the caller.
        let next = if rows.len() as u32 == limit {
            rows.last().cloned()
        } else {
            None
        };
        Ok((rows, next))
    }

    /// Fetch a record by id.
    pub fn get_record(&self, id: &RecordId) -> Result<Option<AnamnesisRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, adapter, instance, content, scope, kind, \
                    created_at, updated_at, tags, metadata, \
                    native_id, native_path, captured_at, raw_hash, schema_version, \
                    derived_from \
             FROM records WHERE id = ?1",
        )?;
        let row = stmt.query_row(params![id.0], record_from_row).optional()?;
        Ok(row)
    }

    /// Batch variant of `get_record` — fetches many records in a single
    /// `WHERE id IN (?, ?, …)` query, returning a `HashMap<RecordId,
    /// AnamnesisRecord>` indexed by id. Missing ids are simply absent
    /// from the map (callers like the search packer want "skip vanished
    /// records" semantics, not an error).
    ///
    /// Used by the search packer to retire its per-id `get_record` loop.
    pub fn get_records_by_ids(
        &self,
        ids: &[RecordId],
    ) -> Result<std::collections::HashMap<RecordId, AnamnesisRecord>> {
        if ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT id, adapter, instance, content, scope, kind, \
                    created_at, updated_at, tags, metadata, \
                    native_id, native_path, captured_at, raw_hash, schema_version, \
                    derived_from \
             FROM records WHERE id IN ({})",
            placeholders
        );
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(&sql)?;
        let params_iter: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|id| &id.0 as &dyn rusqlite::ToSql).collect();
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params_iter), record_from_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut out = std::collections::HashMap::with_capacity(rows.len());
        for r in rows {
            out.insert(r.id.clone(), r);
        }
        Ok(out)
    }

    /// Round-68: the metadata-only variant of `get_records_by_ids`.
    ///
    /// Returns a `HashMap` of `RecordHeader` keyed by id — every field
    /// the search packer / MCP wire / CLI rendering actually consume,
    /// without selecting or deserialising `content`, `tags`, or
    /// `metadata`. For long-transcript records (Claude Code / Codex
    /// adapter records can carry 64KiB+ of rendered session text) this
    /// is dramatically cheaper on the search hot path, where the
    /// downstream caller would have thrown the content away anyway.
    ///
    /// Missing ids are absent from the map — same semantics as
    /// `get_records_by_ids`. Callers that still need the full record
    /// (e.g. the MCP `get_record` tool) should keep calling
    /// `get_records_by_ids` / `get_record`.
    pub fn get_record_headers_by_ids(
        &self,
        ids: &[RecordId],
    ) -> Result<std::collections::HashMap<RecordId, RecordHeader>> {
        if ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT id, adapter, instance, scope, kind, \
                    created_at, updated_at, \
                    native_id, native_path, captured_at, raw_hash, \
                    schema_version, derived_from \
             FROM records WHERE id IN ({})",
            placeholders
        );
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(&sql)?;
        let params_iter: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|id| &id.0 as &dyn rusqlite::ToSql).collect();
        let rows = stmt
            .query_map(
                rusqlite::params_from_iter(params_iter),
                record_header_from_row,
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut out: std::collections::HashMap<RecordId, RecordHeader> =
            std::collections::HashMap::with_capacity(rows.len());
        for r in rows {
            out.insert(r.id.clone(), r);
        }

        // Round 78: batched second query to fill `user_tags` so
        // the search packer pays one round-trip for the overlay
        // instead of N. Tags arrive sorted ASCII-ascending so the
        // wire is deterministic.
        if !out.is_empty() {
            let tag_sql = format!(
                "SELECT record_id, tag \
                 FROM user_record_tags \
                 WHERE record_id IN ({}) \
                 ORDER BY record_id ASC, tag ASC",
                placeholders
            );
            let mut tag_stmt = conn.prepare(&tag_sql)?;
            let tag_params: Vec<&dyn rusqlite::ToSql> =
                ids.iter().map(|id| &id.0 as &dyn rusqlite::ToSql).collect();
            let tag_rows = tag_stmt
                .query_map(rusqlite::params_from_iter(tag_params), |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            for (rid, tag) in tag_rows {
                if let Some(h) = out.get_mut(&RecordId(rid)) {
                    h.user_tags.push(tag);
                }
            }
        }

        Ok(out)
    }

    /// Direct children of `parent` — records whose
    /// `provenance.derived_from == parent`. Hits the
    /// `idx_records_derived_from` partial index (see migration 0004).
    ///
    /// Limit is clamped to `[1, MAX_LIST_LIMIT]` to match the rest of
    /// the listing API. Pass a high limit if you genuinely want every
    /// child — the partial index keeps the query cheap.
    ///
    /// Used by `anamnesis lineage` to show the §-1.5 PR-6 audit trail
    /// (which Facts/Preferences/Skills got distilled out of a given
    /// Episode).
    pub fn list_derivations(&self, parent: &RecordId, limit: u32) -> Result<Vec<AnamnesisRecord>> {
        let limit = limit.clamp(1, MAX_LIST_LIMIT);
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, adapter, instance, content, scope, kind, \
                    created_at, updated_at, tags, metadata, \
                    native_id, native_path, captured_at, raw_hash, schema_version, \
                    derived_from \
             FROM records \
             WHERE derived_from = ?1 \
             ORDER BY created_at ASC, id ASC \
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![parent.0, limit], record_from_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Walk `start` → `start.derived_from` → `…` up to the root of the
    /// lineage chain. The returned `Vec` is ordered child-first: index 0
    /// is `start` itself, the last element is the root (a record whose
    /// `derived_from` is `None`, or the deepest record still in the store
    /// — broken parents are tolerated but reported via the second tuple
    /// element).
    ///
    /// Cycle-safe: if a malformed write ever creates `A → B → A`, the
    /// walk stops at the second encounter and the cycle is signaled as
    /// `Err(StoreError::Corruption)` so callers can surface the
    /// corruption instead of silently truncating.
    ///
    /// Returns `Ok(None)` when `start` itself doesn't exist.
    pub fn lineage_chain(&self, start: &RecordId) -> Result<Option<LineageChain>> {
        let mut chain: Vec<AnamnesisRecord> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut cursor = Some(start.clone());
        let mut missing_parent: Option<RecordId> = None;

        while let Some(cur) = cursor {
            if !seen.insert(cur.0.clone()) {
                return Err(StoreError::Corruption(format!(
                    "lineage cycle detected at {}",
                    cur.0
                )));
            }
            match self.get_record(&cur)? {
                Some(record) => {
                    let next = record.provenance.derived_from.clone();
                    chain.push(record);
                    cursor = next;
                }
                None => {
                    // Parent record is missing. If this is the first hop,
                    // the caller's `start` doesn't exist — return None.
                    if chain.is_empty() {
                        return Ok(None);
                    }
                    missing_parent = Some(cur);
                    break;
                }
            }
        }

        Ok(Some(LineageChain {
            records: chain,
            missing_parent,
        }))
    }

    /// Per-record summary an MCP consumer needs to decide what to do
    /// with a hit (or with `get_record` output) without a second
    /// round trip: how many chunks live behind this record, how many
    /// are embedded under the *active* model, and whether the source
    /// adapter included its own pre-existing embedding for provenance.
    ///
    /// Returns `None` when no record with `id` exists. The active-model
    /// chunk count is deliberately scoped: an embedding produced under
    /// a previous model (e.g. before `anamnesis model use`) does NOT
    /// count toward "ready for vector search right now". This matches
    /// the contract `search_chunks_vec` enforces (it filters on the
    /// caller's `model_id`).
    pub fn record_summary(&self, id: &RecordId) -> Result<Option<RecordSummary>> {
        let conn = self.conn.lock();

        // Cheap probe — does the record exist?
        let exists: bool = conn
            .query_row("SELECT 1 FROM records WHERE id = ?1", params![id.0], |_| {
                Ok(true)
            })
            .optional()?
            .unwrap_or(false);
        if !exists {
            return Ok(None);
        }

        let chunk_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM record_chunks WHERE record_id = ?1",
            params![id.0],
            |r| r.get(0),
        )?;

        // Active model — None when the user has never set one.
        let active_model: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'active_embedding_model'",
                [],
                |r| r.get(0),
            )
            .optional()?;

        // Chunks that have a fresh embedding under the active model.
        // Returns 0 when active_model is None or no embeddings exist.
        let embedded_chunk_count: i64 = match active_model.as_deref() {
            Some(model) => conn.query_row(
                "SELECT COUNT(*) FROM chunk_embeddings e \
                 JOIN record_chunks rc ON rc.id = e.chunk_id \
                 WHERE rc.record_id = ?1 AND e.model_id = ?2",
                params![id.0, model],
                |r| r.get(0),
            )?,
            None => 0,
        };

        // Source-vector presence — never the vector itself; just a
        // tiny breadcrumb so the agent knows mem0's OpenAI embeddings
        // (etc.) are on file as provenance.
        let (source_model, source_dim): (Option<String>, Option<i64>) = conn
            .query_row(
                "SELECT source_embedding_model, source_embedding_dim \
                 FROM raw_artifacts WHERE record_id = ?1",
                params![id.0],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?
            .unwrap_or((None, None));

        Ok(Some(RecordSummary {
            chunk_count: chunk_count as u64,
            embedded_chunk_count: embedded_chunk_count as u64,
            active_model,
            source_embedding_model: source_model,
            source_embedding_dim: source_dim.map(|d| d as u32),
        }))
    }

    /// Fetch one chunk by its id.
    ///
    /// `chunk_id` is the synthetic `"{record_id}:{seq}"` string written
    /// by `write_chunks`. We don't parse it here — instead we JOIN
    /// `record_chunks` against `records` so the returned parent
    /// `record_id` survives any future change to the chunk-id format
    /// without callers having to update.
    pub fn get_chunk(&self, chunk_id: &str) -> Result<Option<ChunkLookup>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT rc.id, rc.record_id, rc.seq, rc.content, \
                    rc.content_hash, rc.token_estimate \
             FROM record_chunks rc \
             WHERE rc.id = ?1",
            params![chunk_id],
            |r| {
                Ok(ChunkLookup {
                    chunk_id: r.get(0)?,
                    record_id: RecordId(r.get(1)?),
                    seq: r.get::<_, i64>(2)? as u32,
                    content: r.get(3)?,
                    content_hash: ContentHash(r.get(4)?),
                    token_estimate: r.get::<_, i64>(5)? as u32,
                })
            },
        )
        .optional()
        .map_err(Into::into)
    }
}

/// Lightweight per-record summary an MCP / CLI consumer needs to decide
/// what to do with a `get_record` result without a second round trip.
/// Computed by `Store::record_summary`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordSummary {
    /// Number of chunks behind this record.
    pub chunk_count: u64,
    /// Number of chunks that have a fresh embedding under the *active*
    /// embedding model. Equal to `chunk_count` when the record is
    /// fully ready for vector search; less when the embedder hasn't
    /// caught up; `0` when no active model is configured.
    pub embedded_chunk_count: u64,
    /// The currently-active embedding model id (e.g.
    /// `"local:default:1"`). `None` when no model is set.
    pub active_model: Option<String>,
    /// If the source adapter shipped a pre-existing embedding for this
    /// record's raw payload, this is its model id (informational only —
    /// source vectors NEVER reach retrieval per BLUEPRINT §6.6.1).
    pub source_embedding_model: Option<String>,
    /// Dimensionality of the source embedding, when present.
    pub source_embedding_dim: Option<u32>,
}

/// Result of `Store::lineage_chain` — an ordered walk from a starting
/// record up to the root of its `provenance.derived_from` chain.
///
/// `records[0]` is the record the caller asked about (the leaf). The
/// last element is whichever ancestor terminated the walk:
///
/// - if it has `provenance.derived_from == None`, it's the true root;
/// - if `missing_parent` is `Some`, the walk stopped because that
///   parent id wasn't in the store (e.g. it was deleted, or the
///   derived record was created with a dangling lineage reference).
///   The chain is still usable; callers can surface the dangling id.
///
/// Cycles cause `Store::lineage_chain` to return `Err`, not a truncated
/// `LineageChain`.
#[derive(Debug, Clone, PartialEq)]
pub struct LineageChain {
    /// Records from leaf to root (or as far up as the chain is intact).
    pub records: Vec<AnamnesisRecord>,
    /// If the walk stopped because a parent `RecordId` wasn't in the
    /// store, this is that missing id. `None` when the walk reached a
    /// real root (a record with `derived_from = None`).
    pub missing_parent: Option<RecordId>,
}

/// One chunk row, joined with enough provenance for downstream tools
/// (currently `trace_provenance`) to surface chunk-level debug info
/// without a second round trip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkLookup {
    /// The synthetic chunk id (`"{record_id}:{seq}"`).
    pub chunk_id: String,
    /// Parent record id.
    pub record_id: RecordId,
    /// Per-record chunk index.
    pub seq: u32,
    /// Chunk text content (original, NOT jieba-tokenized).
    pub content: String,
    /// `blake3` of the content — match key for embedding-job dedup.
    pub content_hash: ContentHash,
    /// Heuristic token count used by the chunker.
    pub token_estimate: u32,
}

/// Light-weight row mapper for `get_record_headers_by_ids`. Mirrors
/// the same source-of-truth ordering as `record_from_row` but skips
/// `content`, `tags`, and `metadata` — the three columns that
/// dominate row-materialization cost for Claude Code / Codex records.
///
/// Projection (must match the SQL in `get_record_headers_by_ids`):
///   0 id, 1 adapter, 2 instance, 3 scope, 4 kind,
///   5 created_at, 6 updated_at,
///   7 native_id, 8 native_path, 9 captured_at, 10 raw_hash,
///   11 schema_version, 12 derived_from.
fn record_header_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RecordHeader> {
    let instance: String = row.get(2)?;
    let updated_at: Option<i64> = row.get(6)?;
    Ok(RecordHeader {
        id: RecordId(row.get(0)?),
        source: SourceDescriptor {
            adapter: row.get(1)?,
            instance: if instance.is_empty() {
                None
            } else {
                Some(instance)
            },
            version: String::new(),
        },
        scope: scope_from(&row.get::<_, String>(3)?),
        kind: kind_from(&row.get::<_, String>(4)?),
        created_at: dt(row.get(5)?),
        updated_at: updated_at.map(dt),
        provenance: Provenance {
            native_id: row.get(7)?,
            native_path: row.get(8)?,
            captured_at: dt(row.get(9)?),
            raw_hash: row.get(10)?,
            derived_from: row.get::<_, Option<String>>(12)?.map(RecordId),
        },
        schema_version: row.get::<_, i64>(11)? as u32,
        // Filled in by the batched second query in
        // `get_record_headers_by_ids`. This per-row mapper can't
        // do the join itself without forcing N+1.
        user_tags: Vec::new(),
    })
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
            derived_from: row.get::<_, Option<String>>(15)?.map(RecordId),
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
    pub fn search_chunks_fts(
        &self,
        query: &str,
        filter: &SearchFilter,
        limit: u32,
    ) -> Result<Vec<ChunkHit>> {
        // PR-Jieba (round-5 consult, see `cjk` module): we MUST tokenize
        // the query through the same pipeline that indexed the chunks.
        // Otherwise FTS5 MATCH compares raw codepoints against the
        // jieba-segmented index, and Chinese queries return zero hits.
        // The Codex consult flagged this asymmetry as the load-bearing
        // trap of the whole feature.
        let match_query = crate::cjk::tokenize_query(query);
        if match_query.is_empty() {
            // FTS5 errors on empty MATCH; an empty user query has no
            // searchable tokens, so zero hits is the right answer.
            return Ok(Vec::new());
        }

        // Build the SQL + bound parameters together — the candidate pool
        // is filtered BEFORE the `LIMIT` truncates it.
        // The first two bound params are always (query, limit); filter
        // params start at index 3 in declaration order below.
        // All placeholders are anonymous `?`. SQLite forbids mixing
        // numbered (`?1`) and unnumbered placeholders within one
        // statement, which is exactly what would happen if we kept the
        // pre-PR-C `?1` MATCH placeholder and appended `?` filter
        // predicates after it.
        let mut sql = String::from(
            "SELECT rc.id, rc.record_id, rc.seq, rc.content, bm25(chunks_fts) AS score \
             FROM chunks_fts \
             JOIN record_chunks rc ON rc.rowid = chunks_fts.rowid",
        );
        let need_records_join = !filter.is_empty();
        if need_records_join {
            sql.push_str(" JOIN records r ON r.id = rc.record_id");
        }
        sql.push_str(" WHERE chunks_fts MATCH ?");
        let filter_params = append_filter_predicates(&mut sql, filter);
        sql.push_str(" ORDER BY score LIMIT ?");

        let conn = self.conn.lock();
        let mut stmt = conn.prepare(&sql)?;
        let mut bound: Vec<rusqlite::types::Value> = Vec::with_capacity(2 + filter_params.len());
        bound.push(rusqlite::types::Value::Text(match_query));
        bound.extend(filter_params);
        bound.push(rusqlite::types::Value::Integer(limit as i64));
        let rows = stmt
            .query_map(rusqlite::params_from_iter(bound.iter()), |r| {
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

    /// Vector top-k over `chunk_embeddings` filtered by `model_id`.
    ///
    /// PR-67a path: route through the per-dim sqlite-vec `vec0` table,
    /// which evaluates cosine in C and applies `SearchFilter` predicates
    /// (`adapter` / `instance` / `kind` / `scope`) *inside* the KNN scan
    /// via vec0 metadata + `model_id` PARTITION KEY pruning. This avoids
    /// the post-filter regression where minority adapters in a heavily-
    /// skewed corpus (1700+:7 distributions) would be evicted before any
    /// of their hits surfaced.
    ///
    /// Fallback: if no vec0 table exists for this dim yet — e.g. fresh
    /// DB with no embeddings, or a new model whose embeddings haven't
    /// been completed since the backfill ran — use the original BLOB
    /// full-scan path so behaviour matches pre-PR-67a exactly.
    pub fn search_chunks_vec(
        &self,
        query_vec: &[f32],
        model_id: &str,
        filter: &SearchFilter,
        limit: u32,
    ) -> Result<Vec<ChunkHit>> {
        if limit == 0 || query_vec.is_empty() {
            return Ok(Vec::new());
        }
        let dim = query_vec.len() as i64;

        let table = {
            let conn = self.conn.lock();
            vec_ext::vec_table_for_dim(&conn, dim)?
        };
        let Some(table) = table else {
            return self.search_chunks_vec_blob_scan(query_vec, model_id, filter, limit);
        };

        // vec0 KNN: `embedding MATCH ?` + `k = ?` are sqlite-vec syntax;
        // partition + metadata predicates are folded into the same WHERE
        // so vec0 narrows the candidate pool *before* distance is scored.
        //
        // `AS MATERIALIZED` is load-bearing: without it SQLite's CTE
        // inliner pushes the outer `JOIN record_chunks ON rc.id =
        // knn.chunk_id` predicate back into the vec0 scan, and vec0
        // rejects any WHERE constraint on auxiliary columns (chunk_id
        // is stored as `+chunk_id` for return-only access).
        let mut sql = format!(
            "WITH knn AS MATERIALIZED ( \
                 SELECT chunk_id, distance \
                 FROM {table} \
                 WHERE embedding MATCH ?1 \
                   AND k = ?2 \
                   AND model_id = ?3"
        );
        let filter_params = append_vec0_filter_predicates(&mut sql, filter);
        sql.push_str(
            " ) \
             SELECT knn.chunk_id, rc.record_id, rc.seq, rc.content, knn.distance \
             FROM knn \
             JOIN record_chunks rc ON rc.id = knn.chunk_id \
             ORDER BY knn.distance ASC",
        );

        let query_blob = f32_to_blob(query_vec);
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(&sql)?;
        let mut bound: Vec<rusqlite::types::Value> = Vec::with_capacity(3 + filter_params.len());
        bound.push(rusqlite::types::Value::Blob(query_blob));
        bound.push(rusqlite::types::Value::Integer(limit as i64));
        bound.push(rusqlite::types::Value::Text(model_id.to_string()));
        bound.extend(filter_params);

        let rows = stmt
            .query_map(rusqlite::params_from_iter(bound.iter()), |r| {
                let distance: f64 = r.get(4)?;
                Ok(ChunkHit {
                    chunk_id: r.get(0)?,
                    record_id: RecordId(r.get(1)?),
                    seq: r.get::<_, i64>(2)? as u32,
                    content: r.get(3)?,
                    // vec0 reports cosine *distance* (1 - cos). Existing
                    // call sites expect "higher = better" similarity.
                    score: 1.0 - distance,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// BLOB-scan fallback. Kept private and only used by
    /// `search_chunks_vec` when no per-dim vec0 table exists for the
    /// query's dim. Behaviour mirrors the pre-PR-67a implementation
    /// exactly so existing tests + corpora keep their semantics.
    fn search_chunks_vec_blob_scan(
        &self,
        query_vec: &[f32],
        model_id: &str,
        filter: &SearchFilter,
        limit: u32,
    ) -> Result<Vec<ChunkHit>> {
        let mut sql = String::from(
            "SELECT e.chunk_id, e.embedding, rc.record_id, rc.seq, rc.content \
             FROM chunk_embeddings e \
             JOIN record_chunks rc ON rc.id = e.chunk_id",
        );
        let need_records_join = !filter.is_empty();
        if need_records_join {
            sql.push_str(" JOIN records r ON r.id = rc.record_id");
        }
        sql.push_str(" WHERE e.model_id = ?");
        let filter_params = append_filter_predicates(&mut sql, filter);

        let conn = self.conn.lock();
        let mut stmt = conn.prepare(&sql)?;
        let mut bound: Vec<rusqlite::types::Value> = Vec::with_capacity(1 + filter_params.len());
        bound.push(rusqlite::types::Value::Text(model_id.to_string()));
        bound.extend(filter_params);
        let mut scored: Vec<ChunkHit> = Vec::new();
        let rows = stmt.query_map(rusqlite::params_from_iter(bound.iter()), |r| {
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

/// Append filter predicates to `sql` and return the bound parameters in
/// declaration order. Caller decides where in their param stream these
/// land — they're given as positional values via `params_from_iter`.
///
/// Predicates use `r.<col>`, requiring the caller to have already added
/// `JOIN records r ON r.id = rc.record_id` (we don't add it here so the
/// SQL builder owns join shape).
fn append_filter_predicates(
    sql: &mut String,
    filter: &SearchFilter,
) -> Vec<rusqlite::types::Value> {
    use rusqlite::types::Value as V;
    let mut params: Vec<V> = Vec::new();
    if let Some(s) = &filter.source {
        sql.push_str(" AND r.adapter = ?");
        params.push(V::Text(s.clone()));
    }
    if let Some(i) = &filter.instance {
        // BLUEPRINT §18 trap: `records.instance` is NOT NULL DEFAULT ''.
        // We normalise the *empty / None* case to `''` so SQL key lookup
        // never misses, mirroring the sources-registry handling in PR-B.
        sql.push_str(" AND r.instance = ?");
        params.push(V::Text(i.clone()));
    }
    if let Some(k) = &filter.kind {
        sql.push_str(" AND r.kind = ?");
        params.push(V::Text(k.clone()));
    }
    if let Some(sc) = &filter.scope {
        sql.push_str(" AND r.scope = ?");
        params.push(V::Text(sc.clone()));
    }
    if let Some(from) = filter.time_from {
        sql.push_str(" AND r.created_at >= ?");
        params.push(V::Integer(from));
    }
    if let Some(to) = filter.time_to {
        sql.push_str(" AND r.created_at <= ?");
        params.push(V::Integer(to));
    }
    // Round 79 (PR-78b): `--user-tag` push-down. Subquery against
    // `user_record_tags` keyed on the indexed `(tag, record_id)`
    // covers FTS + BLOB-vec paths. Sits in the same WHERE so the
    // candidate pool shrinks *before* `LIMIT`.
    if let Some(tag) = &filter.user_tag {
        sql.push_str(
            " AND EXISTS ( \
                 SELECT 1 FROM user_record_tags urt \
                 WHERE urt.record_id = rc.record_id AND urt.tag = ?)",
        );
        params.push(V::Text(tag.clone()));
    }
    params
}

/// vec0 flavor of `append_filter_predicates`. The vec0 KNN scan reads
/// metadata columns *directly* from the virtual table (we mirror
/// `adapter / instance / kind / scope / created_at` into the per-dim
/// table at backfill + upsert time), so predicates are unqualified
/// (no `r.` prefix) and never need a `records` join. This is what makes
/// the filter pushdown happen *inside* the KNN scan rather than after.
fn append_vec0_filter_predicates(
    sql: &mut String,
    filter: &SearchFilter,
) -> Vec<rusqlite::types::Value> {
    use rusqlite::types::Value as V;
    let mut params: Vec<V> = Vec::new();
    if let Some(s) = &filter.source {
        sql.push_str(" AND adapter = ?");
        params.push(V::Text(s.clone()));
    }
    if let Some(i) = &filter.instance {
        sql.push_str(" AND instance = ?");
        params.push(V::Text(i.clone()));
    }
    if let Some(k) = &filter.kind {
        sql.push_str(" AND kind = ?");
        params.push(V::Text(k.clone()));
    }
    if let Some(sc) = &filter.scope {
        sql.push_str(" AND scope = ?");
        params.push(V::Text(sc.clone()));
    }
    if let Some(from) = filter.time_from {
        sql.push_str(" AND created_at >= ?");
        params.push(V::Integer(from));
    }
    if let Some(to) = filter.time_to {
        sql.push_str(" AND created_at <= ?");
        params.push(V::Integer(to));
    }
    // Round 79 (PR-78b): `--user-tag` push-down inside the KNN
    // MATERIALIZED CTE. vec0 can't JOIN an external table, so
    // we constrain the `record_id` metadata column (added in
    // R79 alongside this filter) against the overlay. Stays
    // inside the KNN scan, so a tagged minority record can't be
    // displaced by an untagged majority before `LIMIT`.
    if let Some(tag) = &filter.user_tag {
        sql.push_str(
            " AND record_id IN ( \
                 SELECT record_id FROM user_record_tags WHERE tag = ?)",
        );
        params.push(V::Text(tag.clone()));
    }
    params
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
    /// Batch variant of `claim_next_job` — atomically claims up to `limit`
    /// pending jobs in FIFO order in a single transaction. Used by the
    /// embedding worker's batched drain path so it can hand a whole
    /// `embed_batch` worth of texts to the provider in one call.
    ///
    /// Empty queue → returns `Vec::new()` (not an error).
    pub fn claim_next_jobs(
        &self,
        model_id: &str,
        limit: usize,
    ) -> Result<Vec<PendingEmbeddingJob>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        let now = chrono::Utc::now().timestamp();
        let rows: Vec<(i64, String, String)> = {
            let mut stmt = tx.prepare(
                "SELECT id, chunk_id, content_hash FROM embedding_jobs \
                 WHERE status = 'pending' AND model_id = ?1 \
                 ORDER BY enqueued_at ASC LIMIT ?2",
            )?;
            let mapped = stmt
                .query_map(params![model_id, limit as i64], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        };
        if rows.is_empty() {
            tx.commit()?;
            return Ok(Vec::new());
        }
        let mut jobs = Vec::with_capacity(rows.len());
        for (job_id, chunk_id, content_hash) in rows {
            tx.execute(
                "UPDATE embedding_jobs SET status = 'in_progress', claimed_at = ?1 \
                 WHERE id = ?2",
                params![now, job_id],
            )?;
            let content: String = tx.query_row(
                "SELECT content FROM record_chunks WHERE id = ?1",
                params![chunk_id],
                |r| r.get(0),
            )?;
            jobs.push(PendingEmbeddingJob {
                job_id,
                chunk_id,
                content_hash: ContentHash(content_hash),
                model_id: model_id.to_string(),
                content,
            });
        }
        tx.commit()?;
        Ok(jobs)
    }

    /// Batch variant of `complete_job` — persists embeddings for an entire
    /// batch of jobs in one transaction, paired with their `complete` state
    /// transitions. Vector slice length must equal `jobs.len()`.
    ///
    /// Either the whole batch commits or the whole batch rolls back; callers
    /// that need per-job error isolation should fall back to `complete_job`.
    pub fn complete_jobs_batch(
        &self,
        jobs: &[PendingEmbeddingJob],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        if jobs.len() != vectors.len() {
            return Err(StoreError::Corruption(format!(
                "complete_jobs_batch: jobs.len()={} != vectors.len()={}",
                jobs.len(),
                vectors.len()
            )));
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        let now = chrono::Utc::now().timestamp();
        for (job, vector) in jobs.iter().zip(vectors.iter()) {
            let dim = vector.len() as i64;
            let blob = f32_to_blob(vector);
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
            // Mirror into the per-dim vec0 table so the live write path
            // keeps the search index in step with the BLOB row.
            vec_ext::upsert_vec_row(&tx, &job.chunk_id, &job.model_id, dim, &blob)?;
            tx.execute(
                "UPDATE embedding_jobs SET status = 'done', finished_at = ?1, error = NULL \
                 WHERE id = ?2",
                params![now, job.job_id],
            )?;
        }
        tx.commit()?;
        Ok(())
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
        vec_ext::upsert_vec_row(&tx, &job.chunk_id, &job.model_id, dim, &blob)?;
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
        let import_errors: i64 =
            conn.query_row("SELECT COUNT(1) FROM import_errors", [], |r| r.get(0))?;
        Ok(StoreStats {
            records: records as u64,
            chunks: chunks as u64,
            jobs_pending: pending as u64,
            jobs_failed: failed as u64,
            sources: sources as u64,
            import_errors: import_errors as u64,
        })
    }

    /// One-shot per-adapter count of `import_errors` rows. Returned as
    /// a `HashMap<String, u64>` keyed by adapter id. Adapters with no
    /// errors are simply absent from the map (callers should default to 0).
    ///
    /// Used by `anamnesis doctor` to avoid an N+1 query against
    /// `recent_import_errors(Some(adapter), …)` once per registered
    /// source: one `GROUP BY` instead of N row-materializing scans.
    pub fn count_import_errors_by_adapter(&self) -> Result<std::collections::HashMap<String, u64>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare("SELECT adapter, COUNT(1) FROM import_errors GROUP BY adapter")?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows.into_iter().map(|(a, n)| (a, n as u64)).collect())
    }

    /// Most-recent rows from `import_errors`, newest first. Used by
    /// `anamnesis status` and `anamnesis doctor` to surface what
    /// silently failed during recent imports without making the user
    /// dig into the SQLite database directly.
    ///
    /// Pass `adapter = Some(...)` to scope to one source (matches
    /// `doctor`'s per-source path); `adapter = None` returns all
    /// sources combined.
    pub fn recent_import_errors(
        &self,
        adapter: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ImportErrorRow>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock();
        let mapper = |r: &rusqlite::Row<'_>| -> rusqlite::Result<ImportErrorRow> {
            Ok(ImportErrorRow {
                adapter: r.get(0)?,
                instance: r.get(1)?,
                native_id: r.get(2)?,
                native_path: r.get(3)?,
                phase: r.get(4)?,
                error: r.get(5)?,
                occurred_at: r.get(6)?,
            })
        };
        let rows: Vec<ImportErrorRow> = if let Some(a) = adapter {
            let mut stmt = conn.prepare(
                "SELECT adapter, instance, native_id, native_path, phase, error, occurred_at \
                 FROM import_errors \
                 WHERE adapter = ?1 \
                 ORDER BY occurred_at DESC, id DESC \
                 LIMIT ?2",
            )?;
            let mapped = stmt
                .query_map(params![a, limit as i64], mapper)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        } else {
            let mut stmt = conn.prepare(
                "SELECT adapter, instance, native_id, native_path, phase, error, occurred_at \
                 FROM import_errors \
                 ORDER BY occurred_at DESC, id DESC \
                 LIMIT ?1",
            )?;
            let mapped = stmt
                .query_map(params![limit as i64], mapper)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        };
        Ok(rows)
    }

    /// Round 86 (PR-78h): per-`(adapter, instance)` import-error
    /// view for `anamnesis source show` / MCP `source_show`.
    /// **Distinct from `recent_import_errors(Some(adapter), ..)`**:
    /// that helper is adapter-scoped, this one also filters by
    /// instance, so a `(mem0, self-hosted)` view can't leak rows
    /// from `(mem0, cloud)`. The `idx_errors_source` index on
    /// `(adapter, instance, occurred_at DESC)` (migration 0002)
    /// keeps this O(log N + limit).
    ///
    /// `instance = None` matches rows stored as `""` (the default
    /// instance), same convention every other read path uses.
    pub fn recent_import_errors_for_source(
        &self,
        adapter: &str,
        instance: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ImportErrorRow>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let inst = instance.unwrap_or("");
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT adapter, instance, native_id, native_path, phase, error, occurred_at \
             FROM import_errors \
             WHERE adapter = ?1 AND instance = ?2 \
             ORDER BY occurred_at DESC, id DESC \
             LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![adapter, inst, limit as i64], |r| {
                Ok(ImportErrorRow {
                    adapter: r.get(0)?,
                    instance: r.get(1)?,
                    native_id: r.get(2)?,
                    native_path: r.get(3)?,
                    phase: r.get(4)?,
                    error: r.get(5)?,
                    occurred_at: r.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Round 86 (PR-78h): single-source variant of
    /// [`Store::list_sources_with_counts`]. Returns `None` when
    /// the `(adapter, instance)` pair isn't in the registry —
    /// CLI / MCP turn that into a loud "source not found" error.
    ///
    /// `instance = None` matches the `""` default-instance row.
    /// Counts come from the same scalar-subquery shape
    /// `list_sources_with_counts` uses, so the JOIN-amplification
    /// guard is identical.
    pub fn get_source_with_counts(
        &self,
        adapter: &str,
        instance: Option<&str>,
    ) -> Result<Option<SourceWithCounts>> {
        let inst = instance.unwrap_or("");
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT s.adapter, s.instance, s.location, s.config_json, \
                    s.added_at, s.last_import_at, \
                    (SELECT COUNT(*) \
                       FROM records r \
                      WHERE r.adapter = s.adapter \
                        AND r.instance = s.instance) AS record_count, \
                    (SELECT COUNT(*) \
                       FROM record_chunks rc \
                       JOIN records r ON r.id = rc.record_id \
                      WHERE r.adapter = s.adapter \
                        AND r.instance = s.instance) AS chunk_count, \
                    (SELECT COUNT(DISTINCT urt.record_id) \
                       FROM user_record_tags urt \
                       JOIN records r ON r.id = urt.record_id \
                      WHERE r.adapter = s.adapter \
                        AND r.instance = s.instance) AS tagged_record_count \
             FROM sources s \
             WHERE s.adapter = ?1 AND s.instance = ?2",
        )?;
        let row = stmt
            .query_row(params![adapter, inst], |r| {
                Ok(SourceWithCounts {
                    source: SourceRow {
                        adapter: r.get(0)?,
                        instance: r.get(1)?,
                        location: r.get(2)?,
                        config_json: r.get(3)?,
                        added_at: r.get(4)?,
                        last_import_at: r.get(5)?,
                    },
                    record_count: r.get::<_, i64>(6)? as u64,
                    chunk_count: r.get::<_, i64>(7)? as u64,
                    tagged_record_count: r.get::<_, i64>(8)? as u64,
                })
            })
            .optional()?;
        Ok(row)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Round-69: MCP request metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Hard cap on `mcp_request_metrics` rows. The writer trims after
/// each insert, so the table never grows beyond this — bounded
/// memory + bounded backup size, regardless of how chatty an MCP
/// client gets. See `0006_mcp_request_metrics.sql` for the rationale.
pub const MCP_METRICS_CAP: i64 = 5_000;

/// One MCP `tools/call` request. Created by the MCP server around
/// the dispatcher and handed to [`Store::record_mcp_request_metric`]
/// after the response has been built.
///
/// **Privacy contract**: every field is either a tool name, a
/// success bit, a duration, a result count, or a pre-existing
/// structured argument (`mode`, `source`, `instance`, `limit`) the
/// caller already chose to disclose by passing them. Query text,
/// raw arguments, snippets, and result payloads are NEVER stored.
/// Adding a field here that could carry user content is a bug.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpRequestMetric {
    /// Unix seconds at request entry.
    pub started_at: i64,
    /// `tools/call.name`.
    pub tool: String,
    /// Whether the tool returned successfully.
    pub ok: bool,
    /// Wall time of the dispatch, in milliseconds.
    pub duration_ms: i64,
    /// `search_memories`: number of hits returned. `None` for tools
    /// whose result shape isn't list-like.
    pub result_count: Option<i64>,
    /// Short stable token (`"missing_arg"`, `"unknown_tool"`, …) on
    /// error. `None` on success.
    pub error_kind: Option<String>,
    /// `search_memories`: `hybrid` / `fulltext` / `vector`.
    pub mode: Option<String>,
    /// `search_memories`: adapter filter, if the caller supplied one.
    pub source: Option<String>,
    /// `search_memories`: instance filter.
    pub instance: Option<String>,
    /// `search_memories`: requested `limit`.
    pub limit_value: Option<i64>,
}

/// Per-tool aggregate over a recent window. Returned by
/// [`Store::summarize_mcp_request_metrics`] and surfaced by `doctor`.
///
/// Percentiles use nearest-rank — the smallest value `v` such that
/// at least `p%` of samples are `<= v`. Computed in Rust over the
/// in-memory durations vector after the SQL pull; nothing fancy
/// is needed at our row cap (5000).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpToolMetricSummary {
    /// `tools/call.name`.
    pub tool: String,
    /// Number of requests in the window.
    pub count: u64,
    /// Number that returned an error.
    pub errors: u64,
    /// p50 / p95 / p99 duration in milliseconds.
    pub p50_ms: u64,
    /// p95 duration in milliseconds.
    pub p95_ms: u64,
    /// p99 duration in milliseconds.
    pub p99_ms: u64,
    /// Last request's duration in milliseconds.
    pub last_ms: u64,
    /// Last request's `result_count`, when applicable.
    pub last_result_count: Option<i64>,
    /// Unix seconds of the most recent request in the window.
    pub last_started_at: i64,
}

impl Store {
    /// Persist one MCP request metric. Trims the table to
    /// [`MCP_METRICS_CAP`] rows on each insert so the table is
    /// self-bounded — the user does not need to schedule cleanup.
    ///
    /// All writes are tiny (one INSERT + at most one DELETE) and
    /// happen *after* the MCP response is built, so this cannot
    /// affect tool latency observed by the client.
    pub fn record_mcp_request_metric(&self, m: &McpRequestMetric) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO mcp_request_metrics( \
                 started_at, tool, ok, duration_ms, \
                 result_count, error_kind, \
                 mode, source, instance, limit_value) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                m.started_at,
                m.tool,
                if m.ok { 1_i64 } else { 0_i64 },
                m.duration_ms,
                m.result_count,
                m.error_kind,
                m.mode,
                m.source,
                m.instance,
                m.limit_value,
            ],
        )?;
        // Self-trim: keep the most recent CAP rows. Cheap because
        // `id` is the primary key and the row-count cap is tiny.
        conn.execute(
            "DELETE FROM mcp_request_metrics \
             WHERE id <= (SELECT MAX(id) FROM mcp_request_metrics) - ?1",
            params![MCP_METRICS_CAP],
        )?;
        Ok(())
    }

    /// Per-tool summary over the last `since_ts` (None = all-time).
    /// Tools with zero requests in the window are absent from the
    /// returned vec — callers default to "no data" semantics.
    pub fn summarize_mcp_request_metrics(
        &self,
        since_ts: Option<i64>,
    ) -> Result<Vec<McpToolMetricSummary>> {
        let conn = self.conn.lock();
        let rows: Vec<(String, i64, i64, i64, Option<i64>, i64)> = if let Some(t) = since_ts {
            let mut stmt = conn.prepare(
                "SELECT tool, ok, duration_ms, started_at, result_count, id \
                 FROM mcp_request_metrics \
                 WHERE started_at >= ?1 \
                 ORDER BY tool ASC, started_at ASC, id ASC",
            )?;
            let mapped = stmt
                .query_map(params![t], |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        } else {
            let mut stmt = conn.prepare(
                "SELECT tool, ok, duration_ms, started_at, result_count, id \
                 FROM mcp_request_metrics \
                 ORDER BY tool ASC, started_at ASC, id ASC",
            )?;
            let mapped = stmt
                .query_map([], |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            mapped
        };

        // Group by tool. Per-tool: collect durations for percentile
        // (nearest-rank), keep last started_at + last duration + last
        // result_count (highest id wins because we ORDER BY started_at
        // ASC, id ASC).
        //
        // Tuple = (ok_flag, duration_ms, started_at, result_count, id).
        // Aliased so clippy::type_complexity stays quiet.
        type Sample = (i64, i64, i64, Option<i64>, i64);
        let mut by_tool: std::collections::BTreeMap<String, Vec<Sample>> =
            std::collections::BTreeMap::new();
        for (tool, ok, duration_ms, started_at, result_count, id) in rows {
            by_tool
                .entry(tool)
                .or_default()
                .push((ok, duration_ms, started_at, result_count, id));
        }

        let mut out = Vec::with_capacity(by_tool.len());
        for (tool, mut samples) in by_tool {
            // Already in (started_at, id) ASC order from the SQL.
            let count = samples.len() as u64;
            let errors = samples.iter().filter(|(ok, ..)| *ok == 0).count() as u64;
            let last = samples.last().copied().expect("group has >=1 sample");
            let last_started_at = last.2;
            let last_ms = last.1.max(0) as u64;
            let last_result_count = last.3;

            samples.sort_by_key(|(_ok, d, _ts, _rc, _id)| *d);
            let durations: Vec<i64> = samples.iter().map(|(_o, d, ..)| *d).collect();
            let p = |q: f64| -> u64 {
                if durations.is_empty() {
                    return 0;
                }
                // Nearest-rank: ceil(q * N) - 1, clamped to [0, N-1].
                let n = durations.len();
                let rank = ((q * n as f64).ceil() as usize)
                    .saturating_sub(1)
                    .min(n - 1);
                durations[rank].max(0) as u64
            };
            out.push(McpToolMetricSummary {
                tool,
                count,
                errors,
                p50_ms: p(0.50),
                p95_ms: p(0.95),
                p99_ms: p(0.99),
                last_ms,
                last_result_count,
                last_started_at,
            });
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::chunker::{Chunker, ChunkerConfig};
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
                derived_from: None,
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

    // ─── Round-9: list_sources_with_counts (per-source aggregation) ───

    #[test]
    fn list_sources_with_counts_includes_zero_for_never_imported_source() {
        // Codex acceptance: a source that's been registered but has no
        // records yet must STILL appear with record_count/chunk_count = 0.
        // This is the "registered but stale / never imported" signal an
        // agent needs to detect a misconfigured adapter.
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("mem0", None, Some("/tmp/missing.db"), None)
            .unwrap();
        let rows = store.list_sources_with_counts().unwrap();
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.source.adapter, "mem0");
        assert_eq!(r.record_count, 0);
        assert_eq!(r.chunk_count, 0);
        assert!(r.source.last_import_at.is_none());
    }

    #[test]
    fn list_sources_with_counts_aggregates_records_and_chunks_per_source() {
        // Two sources, different shape:
        //   claude-code  (default instance): 3 records, 3 chunks
        //   mem0         (instance="prod"):  1 record,  1 chunk
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("claude-code", None, Some("/c"), None)
            .unwrap();
        store
            .register_source("mem0", Some("prod"), Some("/m"), None)
            .unwrap();

        for native in ["a", "b", "c"] {
            let r = make_record("claude-code", native, "x", Kind::Fact);
            let c = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &c, None).unwrap();
        }
        // Note: make_record sets instance=None, which is stored as "".
        // We need a "claude-code"/"" row to match the records above —
        // the register_source(None, ...) call already did that.

        // For mem0 we need a record under instance="prod" so the JOIN
        // hits the right source row. Build it manually.
        let mut mem_r = make_record("mem0", "m1", "y", Kind::Fact);
        mem_r.source.instance = Some("prod".into());
        mem_r.id = RecordId::from_parts("mem0", Some("prod"), "m1");
        let mem_c = Chunker::default().chunk(&mem_r.id, &mem_r.content);
        store.upsert_record(&mem_r, &mem_c, None).unwrap();

        let rows = store.list_sources_with_counts().unwrap();
        assert_eq!(rows.len(), 2);
        let cc = rows
            .iter()
            .find(|r| r.source.adapter == "claude-code")
            .unwrap();
        assert_eq!(
            cc.source.instance, "",
            "default instance kept as empty string"
        );
        assert_eq!(cc.record_count, 3);
        assert_eq!(cc.chunk_count, 3);
        let mem = rows.iter().find(|r| r.source.adapter == "mem0").unwrap();
        assert_eq!(
            mem.source.instance, "prod",
            "instance must round-trip through the JOIN"
        );
        assert_eq!(mem.record_count, 1);
        assert_eq!(mem.chunk_count, 1);
    }

    #[test]
    fn list_sources_with_counts_groups_by_adapter_and_instance_not_just_adapter() {
        // Trap Codex flagged: grouping by adapter alone would collapse
        // (mem0, "self-hosted") and (mem0, "cloud") into one row even
        // when they have different counts. Pin the right behavior here.
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("mem0", Some("self-hosted"), Some("/local"), None)
            .unwrap();
        store
            .register_source("mem0", Some("cloud"), Some("https://x"), None)
            .unwrap();

        // 2 records under "self-hosted", 0 under "cloud".
        for native in ["x", "y"] {
            let mut r = make_record("mem0", native, "z", Kind::Fact);
            r.source.instance = Some("self-hosted".into());
            r.id = RecordId::from_parts("mem0", Some("self-hosted"), native);
            let c = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &c, None).unwrap();
        }

        let rows = store.list_sources_with_counts().unwrap();
        assert_eq!(rows.len(), 2, "two distinct (adapter, instance) rows");
        let local = rows
            .iter()
            .find(|r| r.source.instance == "self-hosted")
            .unwrap();
        assert_eq!(local.record_count, 2);
        let cloud = rows.iter().find(|r| r.source.instance == "cloud").unwrap();
        assert_eq!(cloud.record_count, 0);
    }

    // ─── Round-82 PR-78d: tagged_record_count on list_sources_with_counts ─

    /// Newly-registered, never-imported source must report
    /// `tagged_record_count = 0`. Matches the existing
    /// "never-imported source still appears" contract from R9.
    #[test]
    fn list_sources_with_counts_tagged_record_count_zero_for_untagged_source() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("mem0", None, Some("/tmp/missing.db"), None)
            .unwrap();
        // Seed a record but tag nothing on it.
        let r = make_record("mem0", "n1", "x", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();

        let rows = store.list_sources_with_counts().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].record_count, 1);
        assert_eq!(
            rows[0].tagged_record_count, 0,
            "record exists but has no user tags"
        );
    }

    /// **Load-bearing**: this is the JOIN-amplification regression
    /// test. One record with 3 chunks and 4 distinct user tags
    /// must report `chunk_count = 3` and `tagged_record_count = 1`
    /// — NOT `chunk_count = 12` (chunks × tags) and NOT
    /// `tagged_record_count = 4` (tag rows, not records).
    #[test]
    fn list_sources_with_counts_tag_count_is_records_not_tag_rows() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("claude-code", None, Some("/c"), None)
            .unwrap();
        // Three chunks on a single record.
        let r = make_record(
            "claude-code",
            "multi-chunk",
            // Force the chunker to emit ≥3 chunks by passing a
            // chunky body. The default chunker splits on size,
            // so a ~3KB body comfortably yields multiple chunks.
            &"alpha beta gamma delta epsilon zeta eta theta. ".repeat(120),
            Kind::Fact,
        );
        let c = Chunker::default().chunk(&r.id, &r.content);
        assert!(
            c.len() >= 3,
            "fixture must produce ≥3 chunks for the test to be meaningful"
        );
        store.upsert_record(&r, &c, None).unwrap();
        let chunks_before = c.len() as u64;
        store
            .tag_record(
                &r.id,
                &["a".into(), "b".into(), "c".into(), "d".into()],
                UserTagOperation::Add,
            )
            .unwrap();

        let rows = store.list_sources_with_counts().unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.record_count, 1);
        assert_eq!(
            row.chunk_count, chunks_before,
            "chunks must NOT be multiplied by user-tag count — \
             JOIN amplification regression guard"
        );
        assert_eq!(
            row.tagged_record_count, 1,
            "field counts distinct *records* with ≥1 tag, not tag rows"
        );
    }

    /// `tagged_record_count` is partitioned by `(adapter, instance)`
    /// the same way `record_count` is. Tags on records under one
    /// instance must not leak into another instance's row.
    #[test]
    fn list_sources_with_counts_tagged_record_count_partitioned_by_instance() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("mem0", Some("self-hosted"), Some("/local"), None)
            .unwrap();
        store
            .register_source("mem0", Some("cloud"), Some("https://x"), None)
            .unwrap();

        // self-hosted: 1 tagged record.
        let mut r = make_record("mem0", "h1", "x", Kind::Fact);
        r.source.instance = Some("self-hosted".into());
        r.id = RecordId::from_parts("mem0", Some("self-hosted"), "h1");
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
        store
            .tag_record(&r.id, &["keep".into()], UserTagOperation::Add)
            .unwrap();

        // cloud: 1 untagged record.
        let mut r2 = make_record("mem0", "c1", "y", Kind::Fact);
        r2.source.instance = Some("cloud".into());
        r2.id = RecordId::from_parts("mem0", Some("cloud"), "c1");
        let c2 = Chunker::default().chunk(&r2.id, &r2.content);
        store.upsert_record(&r2, &c2, None).unwrap();

        let rows = store.list_sources_with_counts().unwrap();
        assert_eq!(rows.len(), 2);
        let local = rows
            .iter()
            .find(|r| r.source.instance == "self-hosted")
            .unwrap();
        let cloud = rows.iter().find(|r| r.source.instance == "cloud").unwrap();
        assert_eq!(local.tagged_record_count, 1);
        assert_eq!(
            cloud.tagged_record_count, 0,
            "tag on the self-hosted record must NOT show up under cloud"
        );
    }

    /// Forgetting a tagged record removes both the live row and
    /// (via FK cascade) its `user_record_tags` entries — so the
    /// per-source `tagged_record_count` drops by 1.
    #[test]
    fn list_sources_with_counts_tagged_record_count_drops_after_forget() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("claude-code", None, Some("/c"), None)
            .unwrap();
        let r = make_record("claude-code", "k1", "x", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
        store
            .tag_record(&r.id, &["k".into()], UserTagOperation::Add)
            .unwrap();
        assert_eq!(
            store.list_sources_with_counts().unwrap()[0].tagged_record_count,
            1
        );

        store.forget_record(&r.id, None).unwrap();
        assert_eq!(
            store.list_sources_with_counts().unwrap()[0].tagged_record_count,
            0,
            "FK cascade on forget must clear the user_record_tags entry"
        );
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

    // ─── Round-7: write_chunks dedup (BLUEPRINT round-6 Finding 2 fix) ───
    //
    // Codex's acceptance: re-upserting a record whose raw_hash is
    // unchanged must NOT touch record_chunks at all. The win is that
    // the AFTER DELETE / AFTER INSERT triggers (which call
    // tokenize_cjk(content)) don't fire on no-op re-imports.
    //
    // The store-level test asserts the invariant by counting trigger
    // side effects: chunks_fts row content stays byte-identical across
    // the re-upsert, which is only possible if no DELETE+INSERT cycle
    // happened.

    #[test]
    fn reupsert_with_unchanged_raw_hash_returns_zero_zero() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("a", "x", "stable content", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        let (n1, k1) = store.upsert_record(&r, &c, Some("{\"v\":1}")).unwrap();
        assert_eq!((n1, k1), (1, c.len() as u64));

        // Second call with the same record (same raw_hash) → no-op.
        let (n2, k2) = store.upsert_record(&r, &c, Some("{\"v\":1}")).unwrap();
        assert_eq!(
            (n2, k2),
            (0, 0),
            "re-upsert with unchanged raw_hash must report zero work"
        );
    }

    #[test]
    fn reupsert_with_unchanged_raw_hash_does_not_touch_chunks() {
        // Pin Codex's load-bearing assertion: the row in `chunks_fts`
        // must be the SAME row (same rowid, same content) across a no-op
        // re-upsert. If write_chunks fired its DELETE+INSERT, the chunk
        // would get a fresh rowid (record_chunks.id stays the same but
        // SQLite rowid is reassigned on INSERT after DELETE).
        let store = Store::open_in_memory().unwrap();
        let r = make_record("a", "x", "the quick brown fox", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
        let rowid_before: i64 = store
            .conn()
            .query_row(
                "SELECT rowid FROM record_chunks WHERE record_id = ?1",
                params![r.id.0],
                |row| row.get(0),
            )
            .unwrap();

        store.upsert_record(&r, &c, None).unwrap();
        let rowid_after: i64 = store
            .conn()
            .query_row(
                "SELECT rowid FROM record_chunks WHERE record_id = ?1",
                params![r.id.0],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            rowid_before, rowid_after,
            "rowid changed → DELETE+INSERT happened → jieba triggers fired"
        );
        // FTS still finds the content (because chunks_fts wasn't touched).
        let hits = store
            .search_chunks_fts("quick fox", &SearchFilter::default(), 5)
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn reupsert_with_changed_raw_hash_still_rewrites_chunks() {
        // Negative case: when raw_hash genuinely changes the fast-path
        // must NOT swallow the update. Content rewrite + FTS reindex
        // must still happen.
        let store = Store::open_in_memory().unwrap();
        let mut r = make_record("a", "x", "old content", Kind::Fact);
        r.provenance.raw_hash = "hash-v1".into();
        let c1 = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c1, None).unwrap();

        let mut r2 = r.clone();
        r2.content = "new completely different content".into();
        r2.provenance.raw_hash = "hash-v2".into();
        let c2 = Chunker::default().chunk(&r2.id, &r2.content);
        let (n, k) = store.upsert_record(&r2, &c2, None).unwrap();
        assert_eq!(n, 1, "raw_hash changed → record written");
        assert_eq!(k, c2.len() as u64, "chunks rewritten");
        let hits = store
            .search_chunks_fts("different", &SearchFilter::default(), 5)
            .unwrap();
        assert!(!hits.is_empty(), "new content searchable");
        let stale = store
            .search_chunks_fts("old", &SearchFilter::default(), 5)
            .unwrap();
        assert!(stale.is_empty(), "old content evicted");
    }

    #[test]
    fn reupsert_no_op_still_enqueues_jobs_for_active_model() {
        // If the user switched embedding models between two imports,
        // the no-op fast-path must still enqueue jobs for the NEW model
        // (otherwise chunks would be invisible to vector search under
        // the new model). enqueue_jobs is ON CONFLICT DO NOTHING so
        // this is safe + cheap.
        let store = Store::open_in_memory().unwrap();
        let r = make_record("a", "x", "hello world", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        // First import with model A.
        store.set_active_model("local:model-a:1").unwrap();
        store.upsert_record(&r, &c, None).unwrap();

        // Switch model, re-import the same record. raw_hash is identical
        // so write path skips, but jobs should be enqueued under model-b.
        store.set_active_model("local:model-b:1").unwrap();
        let (n, k) = store.upsert_record(&r, &c, None).unwrap();
        assert_eq!((n, k), (0, 0));

        let pending_for_b: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM embedding_jobs \
                 WHERE status = 'pending' AND model_id = 'local:model-b:1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            pending_for_b as usize,
            c.len(),
            "fast-path must still enqueue jobs under the active model"
        );
    }

    #[test]
    fn upsert_replaces_chunks_on_recall() {
        let store = Store::open_in_memory().unwrap();
        let mut r = make_record("a", "x", "v1", Kind::Fact);
        r.provenance.raw_hash = "v1-hash".into();
        let c1 = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c1, None).unwrap();

        let mut r2 = r.clone();
        r2.content = "v2 different and longer ".repeat(40);
        // Round-7: a content change must come with a raw_hash bump, or
        // the fast-path will (correctly) treat the upsert as a no-op.
        // Real adapters always recompute raw_hash from the source bytes
        // so this is automatic in practice; the test must mirror that
        // by bumping the hash here.
        r2.provenance.raw_hash = "v2-hash".into();
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
        let hits = store
            .search_chunks_fts("different", &SearchFilter::default(), 5)
            .unwrap();
        assert!(!hits.is_empty());
        let stale = store
            .search_chunks_fts("v1", &SearchFilter::default(), 5)
            .unwrap();
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
        let hits = store
            .search_chunks_fts("quick fox", &SearchFilter::default(), 5)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, r.id);
        assert!(hits[0].score > 0.0);
    }

    // ─── PR-Jieba (round-5): CJK FTS round-trip ───
    //
    // The point of jieba-based pre-tokenization is that a multi-char
    // Chinese phrase the user typed maps to the same word boundaries
    // jieba picked when we indexed the document. unicode61 alone
    // (the pre-PR-Jieba behaviour) would still match — because every
    // Han codepoint becomes its own token — but BM25 scoring would be
    // dominated by character frequency, not phrase frequency. The
    // semantics that matter to users only emerge once we agree that
    // "记忆" is one token, not two.

    #[test]
    fn cjk_phrase_search_finds_indexed_document() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record(
            "claude-code",
            "cjk-1",
            "Anamnesis 是跨 agent 的记忆基础设施，本地优先，无 telemetry",
            Kind::Fact,
        );
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();

        // The exact phrase "记忆基础" (or any 2-char Chinese substring
        // of the content) must surface the indexed record.
        for query in &["记忆", "基础设施", "本地优先"] {
            let hits = store
                .search_chunks_fts(query, &SearchFilter::default(), 5)
                .unwrap();
            assert!(
                !hits.is_empty(),
                "CJK query {query:?} must find the indexed record"
            );
            assert_eq!(hits[0].record_id, r.id, "wrong record for query {query:?}");
        }
    }

    #[test]
    fn cjk_search_distinguishes_distinct_words() {
        // Two documents that share characters but not jieba-segmented
        // words. With unicode61 they'd both match a single-char query;
        // with jieba they're correctly separated.
        let store = Store::open_in_memory().unwrap();
        let a = make_record("a", "a1", "我的偏好是 vim", Kind::Preference);
        let b = make_record("a", "b1", "项目里有很多代码", Kind::Fact);
        let ca = Chunker::default().chunk(&a.id, &a.content);
        let cb = Chunker::default().chunk(&b.id, &b.content);
        store.upsert_record(&a, &ca, None).unwrap();
        store.upsert_record(&b, &cb, None).unwrap();

        let hits_pref = store
            .search_chunks_fts("偏好", &SearchFilter::default(), 5)
            .unwrap();
        assert_eq!(hits_pref.len(), 1);
        assert_eq!(hits_pref[0].record_id, a.id);

        let hits_proj = store
            .search_chunks_fts("项目", &SearchFilter::default(), 5)
            .unwrap();
        assert_eq!(hits_proj.len(), 1);
        assert_eq!(hits_proj[0].record_id, b.id);
    }

    #[test]
    fn empty_or_punctuation_only_query_returns_no_hits() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("a", "x", "alpha beta gamma", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();

        // FTS5 errors on empty MATCH — we must short-circuit instead.
        let empty = store
            .search_chunks_fts("", &SearchFilter::default(), 5)
            .unwrap();
        assert!(empty.is_empty());
        let punct = store
            .search_chunks_fts("!!!  ???", &SearchFilter::default(), 5)
            .unwrap();
        assert!(punct.is_empty());
    }

    #[test]
    fn cjk_reindex_picks_up_existing_chunks() {
        // Migration 0003 sets `chunks_fts_rebuild_pending`; verify that
        // `Store::open` running over an existing DB with rows in
        // `record_chunks` reconstructs the FTS index. We can't easily
        // simulate the pre-0003 DB state from in-memory tests, so we
        // assert the simpler invariant: after `upsert_record + open`
        // the FTS row count equals the chunks row count. This catches
        // regression in the reindex path even if the migration shape
        // changes.
        let store = Store::open_in_memory().unwrap();
        let r = make_record("a", "x", "重新索引 测试", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
        let conn = store.conn.lock();
        let chunks_n: i64 = conn
            .query_row("SELECT COUNT(*) FROM record_chunks", [], |r| r.get(0))
            .unwrap();
        let fts_n: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks_fts", [], |r| r.get(0))
            .unwrap();
        assert_eq!(chunks_n, fts_n, "every chunk has an FTS row");
        assert!(chunks_n > 0);
    }

    // ─── PR-C: candidate-side filter pushdown ───
    //
    // Codex's acceptance assertion (BLUEPRINT §17.5 PR-C consult):
    //
    //   "Construct 1744 claude-code records + 7 mem0 records sharing
    //    one query term; `source=mem0` must return non-empty results,
    //    all from mem0, even with a candidate-pool limit smaller than
    //    the claude-code majority."
    //
    // If filter pushdown is wrong, FTS picks the top-pool by BM25
    // unfiltered → the pool fills with claude-code chunks → post-filter
    // shrinks to zero. The whole point of pushdown is that the SQL
    // recall stage drops claude-code BEFORE the limit applies.

    #[test]
    fn filter_pushdown_returns_minority_source_under_majority_dominance() {
        let store = Store::open_in_memory().unwrap();
        // 1744 claude-code records (every one matches "sharedterm").
        for i in 0..1744u32 {
            let r = make_record(
                "claude-code",
                &format!("cc-{i:04}"),
                "sharedterm claude noise",
                Kind::Episode,
            );
            let c = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &c, None).unwrap();
        }
        // 7 mem0 records, all matching the same term.
        for i in 0..7u32 {
            let r = make_record(
                "mem0",
                &format!("m0-{i}"),
                "sharedterm mem0 fact",
                Kind::Fact,
            );
            let c = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &c, None).unwrap();
        }

        // With NO filter, the pool of 50 is dominated by claude-code.
        let none = store
            .search_chunks_fts("sharedterm", &SearchFilter::default(), 50)
            .unwrap();
        assert_eq!(none.len(), 50, "unfiltered hits fill the pool");
        let mem0_in_unfiltered = none
            .iter()
            .filter(|h| h.content.contains("mem0 fact"))
            .count();
        assert!(
            mem0_in_unfiltered <= 7,
            "without pushdown, the 7 mem0 records are squeezed by the 1744 claude-code majority"
        );

        // WITH source=mem0 pushed into SQL, the pool is drawn from mem0
        // chunks only — even at the same pool size of 50.
        let filter = SearchFilter {
            source: Some("mem0".into()),
            ..SearchFilter::default()
        };
        let mem0_hits = store.search_chunks_fts("sharedterm", &filter, 50).unwrap();
        assert!(
            !mem0_hits.is_empty(),
            "source=mem0 must return non-empty results from the minority adapter"
        );
        assert_eq!(
            mem0_hits.len(),
            7,
            "filter pushdown must surface all 7 mem0 chunks, not zero"
        );
        for h in &mem0_hits {
            assert!(
                h.content.contains("mem0 fact"),
                "every hit must come from the mem0 adapter, not the claude-code majority"
            );
            assert!(
                !h.content.contains("claude noise"),
                "no claude-code chunk should leak through the SQL filter"
            );
        }
    }

    #[test]
    fn filter_pushdown_supports_kind_and_scope_independently() {
        let store = Store::open_in_memory().unwrap();
        for (na, content, kind) in &[
            ("a", "shared topic alpha", Kind::Fact),
            ("b", "shared topic beta", Kind::Preference),
            ("c", "shared topic gamma", Kind::Feedback),
        ] {
            let r = make_record("claude-code", na, content, *kind);
            let c = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &c, None).unwrap();
        }
        let kind_filter = SearchFilter {
            kind: Some("preference".into()),
            ..SearchFilter::default()
        };
        let hits = store
            .search_chunks_fts("shared topic", &kind_filter, 10)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].content.contains("beta"));
    }

    #[test]
    fn filter_pushdown_respects_time_range() {
        let store = Store::open_in_memory().unwrap();
        // Manually crafted records at known timestamps.
        for (na, content, ts) in &[
            ("old", "shared topic", 1700000000_i64), // 2023-11
            ("mid", "shared topic", 1750000000_i64), // 2025-06
            ("new", "shared topic", 1800000000_i64), // 2027-01
        ] {
            let mut r = make_record("claude-code", na, content, Kind::Episode);
            r.created_at = Utc.timestamp_opt(*ts, 0).unwrap();
            let c = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &c, None).unwrap();
        }
        let filter = SearchFilter {
            time_from: Some(1720000000),
            time_to: Some(1780000000),
            ..SearchFilter::default()
        };
        let hits = store
            .search_chunks_fts("shared topic", &filter, 10)
            .unwrap();
        assert_eq!(hits.len(), 1, "only the mid record falls in the window");
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
            .search_chunks_vec(
                &[0.5, 0.5, 0.5, 0.5],
                "local:fake:1",
                &SearchFilter::default(),
                5,
            )
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

    // ─────────────────────────────────────────────────────────────────────
    // Round-21 (§-1.5 PR-2): list_record_ids_paged cursor contract.
    // ─────────────────────────────────────────────────────────────────────

    fn seed_n_records(store: &Store, n: usize) {
        for i in 0..n {
            let r = make_record(
                "claude-code",
                &format!("seed-{i:04}"),
                &format!("content {i}"),
                Kind::Fact,
            );
            let c = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &c, None).unwrap();
        }
    }

    #[test]
    fn paged_listing_walks_through_full_catalogue_via_cursor() {
        let store = Store::open_in_memory().unwrap();
        seed_n_records(&store, 25);

        let mut collected: Vec<String> = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..100 {
            // Outer cap; we expect to terminate in ≤ 3 iterations.
            let (page, next) = store.list_record_ids_paged(cursor.as_deref(), 10).unwrap();
            // Pages are non-empty until we exit.
            assert!(!page.is_empty(), "non-final page must have rows");
            collected.extend(page);
            if next.is_none() {
                break;
            }
            cursor = next;
        }
        assert_eq!(
            collected.len(),
            25,
            "pagination must yield every record exactly once"
        );

        // Lexicographic ascending order is the documented contract.
        let mut sorted = collected.clone();
        sorted.sort();
        assert_eq!(collected, sorted);

        // No duplicates.
        let unique: std::collections::HashSet<&String> = collected.iter().collect();
        assert_eq!(unique.len(), collected.len());
    }

    #[test]
    fn paged_listing_signals_end_with_none_cursor() {
        // When the page returns fewer than `limit` rows, `next_cursor`
        // must be `None` — that's the "end of catalogue" signal.
        let store = Store::open_in_memory().unwrap();
        seed_n_records(&store, 3);
        let (page, next) = store.list_record_ids_paged(None, 10).unwrap();
        assert_eq!(page.len(), 3);
        assert!(next.is_none(), "page < limit must clear nextCursor");
    }

    #[test]
    fn paged_listing_clamps_limit() {
        // limit=0 must clamp to 1; limit>MAX must clamp to MAX. The
        // store should never refuse a malformed limit — it should be
        // permissive at the edge and let the caller see useful data.
        let store = Store::open_in_memory().unwrap();
        seed_n_records(&store, 5);
        let (page, _) = store.list_record_ids_paged(None, 0).unwrap();
        assert_eq!(page.len(), 1, "limit=0 must clamp to 1");
        let (page, _) = store.list_record_ids_paged(None, u32::MAX).unwrap();
        assert!(page.len() <= MAX_LIST_LIMIT as usize);
        assert_eq!(page.len(), 5);
    }

    #[test]
    fn derived_from_roundtrips_through_store() {
        // §-1.5 PR-6 regression: a record carrying `provenance.derived_from`
        // must survive upsert + get_record without losing the lineage link.
        // This is the only audit hook §-1.5 #6 promises.
        let store = Store::open_in_memory().unwrap();
        let parent = make_record("claude-code", "ep-1", "raw conversation", Kind::Episode);
        let parent_id = parent.id.clone();
        let chunks = Chunker::default().chunk(&parent.id, &parent.content);
        store.upsert_record(&parent, &chunks, None).unwrap();

        let mut derived = make_record("extractor", "fact-1", "user lives in Paris", Kind::Fact);
        derived.provenance.derived_from = Some(parent_id.clone());
        let derived_chunks = Chunker::default().chunk(&derived.id, &derived.content);
        let derived_id = derived.id.clone();
        store
            .upsert_record(&derived, &derived_chunks, None)
            .unwrap();

        let got_parent = store.get_record(&parent_id).unwrap().unwrap();
        assert!(
            got_parent.provenance.derived_from.is_none(),
            "non-derived records keep derived_from = None on the way back"
        );

        let got_derived = store.get_record(&derived_id).unwrap().unwrap();
        assert_eq!(
            got_derived.provenance.derived_from.as_ref().map(|r| &r.0),
            Some(&parent_id.0),
            "derived record's lineage must point at the source Episode after round-trip"
        );
    }

    #[test]
    fn list_derivations_returns_only_direct_children() {
        let store = Store::open_in_memory().unwrap();
        let parent = make_record("claude-code", "ep-1", "raw conversation", Kind::Episode);
        let pid = parent.id.clone();
        let pc = Chunker::default().chunk(&parent.id, &parent.content);
        store.upsert_record(&parent, &pc, None).unwrap();

        let mut child_a = make_record("extractor", "fact-a", "user lives in Paris", Kind::Fact);
        child_a.provenance.derived_from = Some(pid.clone());
        let c_a = Chunker::default().chunk(&child_a.id, &child_a.content);
        store.upsert_record(&child_a, &c_a, None).unwrap();

        let mut child_b = make_record("extractor", "pref-a", "prefers Rust", Kind::Preference);
        child_b.provenance.derived_from = Some(pid.clone());
        let c_b = Chunker::default().chunk(&child_b.id, &child_b.content);
        store.upsert_record(&child_b, &c_b, None).unwrap();

        // Sibling that is NOT derived from parent — must not appear.
        let unrelated = make_record("claude-code", "ep-2", "different episode", Kind::Episode);
        let cu = Chunker::default().chunk(&unrelated.id, &unrelated.content);
        store.upsert_record(&unrelated, &cu, None).unwrap();

        let children = store.list_derivations(&pid, 50).unwrap();
        assert_eq!(children.len(), 2);
        let kinds: std::collections::HashSet<_> = children.iter().map(|r| r.kind).collect();
        assert!(kinds.contains(&Kind::Fact));
        assert!(kinds.contains(&Kind::Preference));
    }

    #[test]
    fn lineage_chain_walks_to_root() {
        let store = Store::open_in_memory().unwrap();
        // Episode (root) → Fact (mid) → Skill (leaf).
        let root = make_record("claude-code", "ep-1", "raw conv", Kind::Episode);
        let root_id = root.id.clone();
        let rc = Chunker::default().chunk(&root.id, &root.content);
        store.upsert_record(&root, &rc, None).unwrap();

        let mut mid = make_record("extractor", "fact-a", "Paris is capital", Kind::Fact);
        mid.provenance.derived_from = Some(root_id.clone());
        let mid_id = mid.id.clone();
        let mc = Chunker::default().chunk(&mid.id, &mid.content);
        store.upsert_record(&mid, &mc, None).unwrap();

        let mut leaf = make_record("extractor", "skill-a", "how to check capital", Kind::Skill);
        leaf.provenance.derived_from = Some(mid_id.clone());
        let leaf_id = leaf.id.clone();
        let lc = Chunker::default().chunk(&leaf.id, &leaf.content);
        store.upsert_record(&leaf, &lc, None).unwrap();

        let chain = store.lineage_chain(&leaf_id).unwrap().unwrap();
        assert_eq!(chain.records.len(), 3);
        assert_eq!(chain.records[0].id.0, leaf_id.0);
        assert_eq!(chain.records[1].id.0, mid_id.0);
        assert_eq!(chain.records[2].id.0, root_id.0);
        assert!(chain.missing_parent.is_none());
    }

    #[test]
    fn lineage_chain_missing_parent_is_signaled() {
        let store = Store::open_in_memory().unwrap();
        let phantom = RecordId("never-stored-record".into());
        let mut orphan = make_record("extractor", "orphan", "dangling fact", Kind::Fact);
        orphan.provenance.derived_from = Some(phantom.clone());
        let oid = orphan.id.clone();
        let oc = Chunker::default().chunk(&orphan.id, &orphan.content);
        store.upsert_record(&orphan, &oc, None).unwrap();

        let chain = store.lineage_chain(&oid).unwrap().unwrap();
        assert_eq!(chain.records.len(), 1);
        assert_eq!(chain.records[0].id.0, oid.0);
        assert_eq!(chain.missing_parent.unwrap().0, phantom.0);
    }

    #[test]
    fn lineage_chain_returns_none_for_unknown_start() {
        let store = Store::open_in_memory().unwrap();
        let chain = store
            .lineage_chain(&RecordId("does-not-exist".into()))
            .unwrap();
        assert!(chain.is_none());
    }

    #[test]
    fn lineage_chain_detects_cycle_and_errors() {
        // Build A → B → A via direct DB writes. The high-level API
        // can't construct this (insertion order forbids it) but a
        // corrupted file or future bug could — make sure the walk
        // bails loudly instead of looping forever.
        let store = Store::open_in_memory().unwrap();
        let a = make_record("extractor", "a", "node a", Kind::Fact);
        let b = make_record("extractor", "b", "node b", Kind::Fact);
        let aid = a.id.clone();
        let bid = b.id.clone();
        let ac = Chunker::default().chunk(&a.id, &a.content);
        let bc = Chunker::default().chunk(&b.id, &b.content);
        store.upsert_record(&a, &ac, None).unwrap();
        store.upsert_record(&b, &bc, None).unwrap();
        // Hand-write the cycle.
        store
            .conn()
            .execute(
                "UPDATE records SET derived_from = ?1 WHERE id = ?2",
                params![bid.0, aid.0],
            )
            .unwrap();
        store
            .conn()
            .execute(
                "UPDATE records SET derived_from = ?1 WHERE id = ?2",
                params![aid.0, bid.0],
            )
            .unwrap();
        let err = store.lineage_chain(&aid).unwrap_err();
        match err {
            StoreError::Corruption(msg) => assert!(msg.contains("cycle")),
            other => panic!("expected Corruption, got {other:?}"),
        }
    }

    #[test]
    fn derived_from_index_is_present_after_migration() {
        // The migration explicitly creates `idx_records_derived_from`. If
        // a future change drops it, the `anamnesis lineage` query path
        // would regress to a full table scan — fail loudly here so the
        // perf characteristic is part of the contract.
        let store = Store::open_in_memory().unwrap();
        let count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master \
                 WHERE type = 'index' AND name = 'idx_records_derived_from'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            count, 1,
            "derived_from index must exist after 0004 migration"
        );
    }

    // ─── Round-62: upsert_records_batch parity tests ─────────────────────

    /// `upsert_records_batch` with one item must produce the same
    /// records / chunks / counts as `upsert_record` does for that
    /// same record. Guards against drift between the two paths.
    #[test]
    fn upsert_records_batch_size_one_matches_upsert_record() {
        let single = Store::open_in_memory().unwrap();
        let batched = Store::open_in_memory().unwrap();

        let mut r = make_record("claude-code", "alpha", "alpha content", Kind::Fact);
        r.provenance.raw_hash = "alpha-hash".into();
        let chunks = Chunker::default().chunk(&r.id, &r.content);

        let (single_recs, single_chunks) = single.upsert_record(&r, &chunks, None).unwrap();
        let chunks_slice = chunks.as_slice();
        let (batched_recs, batched_chunks) = batched
            .upsert_records_batch(&[(&r, chunks_slice, None)])
            .unwrap();

        assert_eq!((single_recs, single_chunks), (batched_recs, batched_chunks));

        let single_records: i64 = single
            .conn()
            .query_row("SELECT COUNT(*) FROM records", [], |row| row.get(0))
            .unwrap();
        let batched_records: i64 = batched
            .conn()
            .query_row("SELECT COUNT(*) FROM records", [], |row| row.get(0))
            .unwrap();
        assert_eq!(single_records, batched_records);

        let single_chunks_n: i64 = single
            .conn()
            .query_row("SELECT COUNT(*) FROM record_chunks", [], |row| row.get(0))
            .unwrap();
        let batched_chunks_n: i64 = batched
            .conn()
            .query_row("SELECT COUNT(*) FROM record_chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(single_chunks_n, batched_chunks_n);
    }

    // ─── Round-63: claim/complete batch parity + get_records_by_ids ──

    /// `claim_next_jobs(model, n)` must return the same jobs in the same
    /// order as `n` consecutive `claim_next_job` calls would.
    #[test]
    fn claim_next_jobs_parity_with_claim_next_job() {
        let single = Store::open_in_memory().unwrap();
        let batched = Store::open_in_memory().unwrap();
        single.set_active_model("test:model:1").unwrap();
        batched.set_active_model("test:model:1").unwrap();

        // Seed both stores with the same 5 chunks (one record, fan-out by
        // tiny chunker budget so we get N distinct embedding_jobs).
        let r = make_record("claude-code", "p", &"x ".repeat(40), Kind::Fact);
        let chunker = Chunker::new(ChunkerConfig {
            max_tokens: 10,
            min_tokens: 1,
        });
        let chunks = chunker.chunk(&r.id, &r.content);
        assert!(
            chunks.len() >= 5,
            "test needs >=5 chunks; got {}",
            chunks.len()
        );
        single.upsert_record(&r, &chunks, None).unwrap();
        batched.upsert_record(&r, &chunks, None).unwrap();

        let mut single_drained = Vec::new();
        while let Some(job) = single.claim_next_job("test:model:1").unwrap() {
            single_drained.push((job.chunk_id, job.content_hash.0));
        }
        let batched_drained: Vec<(String, String)> = batched
            .claim_next_jobs("test:model:1", chunks.len() + 4)
            .unwrap()
            .into_iter()
            .map(|j| (j.chunk_id, j.content_hash.0))
            .collect();

        assert_eq!(single_drained, batched_drained);
    }

    /// `get_records_by_ids` must match what N independent `get_record`
    /// calls would return, with missing ids simply absent from the map.
    #[test]
    fn get_records_by_ids_parity_with_get_record() {
        let store = Store::open_in_memory().unwrap();
        let mut r1 = make_record("claude-code", "one", "first", Kind::Fact);
        r1.provenance.raw_hash = "h1".into();
        let mut r2 = make_record("claude-code", "two", "second", Kind::Fact);
        r2.provenance.raw_hash = "h2".into();
        let mut r3 = make_record("claude-code", "three", "third", Kind::Fact);
        r3.provenance.raw_hash = "h3".into();
        store
            .upsert_record(&r1, &Chunker::default().chunk(&r1.id, &r1.content), None)
            .unwrap();
        store
            .upsert_record(&r2, &Chunker::default().chunk(&r2.id, &r2.content), None)
            .unwrap();
        store
            .upsert_record(&r3, &Chunker::default().chunk(&r3.id, &r3.content), None)
            .unwrap();

        let phantom = RecordId::from_parts("claude-code", None, "missing");
        let single_results: Vec<Option<AnamnesisRecord>> = [&r1.id, &phantom, &r2.id, &r3.id]
            .iter()
            .map(|id| store.get_record(id).unwrap())
            .collect();
        let batched_map = store
            .get_records_by_ids(&[r1.id.clone(), phantom.clone(), r2.id.clone(), r3.id.clone()])
            .unwrap();

        // The map must hold exactly the 3 existing ids (no entry for `phantom`).
        assert_eq!(batched_map.len(), 3);
        assert!(!batched_map.contains_key(&phantom));
        for (idx, id) in [&r1.id, &phantom, &r2.id, &r3.id].iter().enumerate() {
            let single = &single_results[idx];
            let batched = batched_map.get(id);
            assert_eq!(single.as_ref(), batched);
        }
    }

    /// Round-68: `get_record_headers_by_ids` must agree with
    /// `get_records_by_ids` on every header-projection field. The point
    /// of the lighter method is to drop `content / tags / metadata`
    /// from the SQL projection — *not* to drift on the columns that
    /// remain (anything wire-visible to MCP / CLI would silently
    /// regress otherwise).
    #[test]
    fn get_record_headers_by_ids_parity_with_get_records_by_ids() {
        let store = Store::open_in_memory().unwrap();
        let mut r1 = make_record("claude-code", "one", "first content body", Kind::Fact);
        r1.provenance.raw_hash = "h1".into();
        let mut r2 = make_record("codex", "two", "second content body", Kind::Preference);
        r2.provenance.raw_hash = "h2".into();
        let mut r3 = make_record(
            "mem0",
            "three",
            &"long body ".repeat(2_000), /* ~22 KB body */
            Kind::Fact,
        );
        r3.provenance.raw_hash = "h3".into();
        for r in [&r1, &r2, &r3] {
            store
                .upsert_record(r, &Chunker::default().chunk(&r.id, &r.content), None)
                .unwrap();
        }

        let phantom = RecordId::from_parts("claude-code", None, "missing");
        let ids = [r1.id.clone(), phantom.clone(), r2.id.clone(), r3.id.clone()];

        let full = store.get_records_by_ids(&ids).unwrap();
        let heads = store.get_record_headers_by_ids(&ids).unwrap();

        assert_eq!(full.len(), heads.len(), "vanished id stays absent");
        assert!(!heads.contains_key(&phantom));

        for id in [&r1.id, &r2.id, &r3.id] {
            let f = full.get(id).expect("full present");
            let h = heads.get(id).expect("head present");
            assert_eq!(h.id, f.id);
            assert_eq!(h.source.adapter, f.source.adapter);
            assert_eq!(h.source.instance, f.source.instance);
            assert_eq!(h.scope, f.scope);
            assert_eq!(h.kind, f.kind);
            assert_eq!(h.created_at, f.created_at);
            assert_eq!(h.updated_at, f.updated_at);
            assert_eq!(h.provenance, f.provenance);
            assert_eq!(h.schema_version, f.schema_version);
        }
    }

    /// `complete_jobs_batch` must leave the store in the same end-state
    /// as N independent `complete_job` calls would.
    #[test]
    fn complete_jobs_batch_parity_with_complete_job() {
        let single = Store::open_in_memory().unwrap();
        let batched = Store::open_in_memory().unwrap();
        single.set_active_model("test:model:1").unwrap();
        batched.set_active_model("test:model:1").unwrap();

        // Seed with 3 chunks via per-record path.
        let r = make_record("claude-code", "p", &"x ".repeat(40), Kind::Fact);
        let chunker = Chunker::new(ChunkerConfig {
            max_tokens: 15,
            min_tokens: 1,
        });
        let chunks = chunker.chunk(&r.id, &r.content);
        assert!(chunks.len() >= 3);
        single.upsert_record(&r, &chunks, None).unwrap();
        batched.upsert_record(&r, &chunks, None).unwrap();

        // single: drain jobs one at a time via complete_job
        let mut single_jobs = Vec::new();
        while let Some(job) = single.claim_next_job("test:model:1").unwrap() {
            single_jobs.push(job);
        }
        for job in &single_jobs {
            single.complete_job(job, &[0.5; 4]).unwrap();
        }

        // batched: drain all jobs at once via complete_jobs_batch
        let batch_jobs = batched
            .claim_next_jobs("test:model:1", single_jobs.len() + 10)
            .unwrap();
        let vectors: Vec<Vec<f32>> = batch_jobs.iter().map(|_| vec![0.5; 4]).collect();
        batched.complete_jobs_batch(&batch_jobs, &vectors).unwrap();

        // Both stores should now have the same status mix.
        for store in [&single, &batched] {
            let done: i64 = store
                .conn()
                .query_row(
                    "SELECT COUNT(*) FROM embedding_jobs WHERE status = 'done'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(done as usize, single_jobs.len());
        }
        // …and the same chunk_embeddings rows.
        let single_n: i64 = single
            .conn()
            .query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |row| {
                row.get(0)
            })
            .unwrap();
        let batched_n: i64 = batched
            .conn()
            .query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(single_n, batched_n);
    }

    // ─── Round-64: import_errors exposure ────────────────────────────

    /// `stats().import_errors` must reflect every row written by
    /// `log_import_error` and must surface as 0 on a fresh store.
    #[test]
    fn stats_counts_import_errors() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(store.stats().unwrap().import_errors, 0);
        store
            .log_import_error("claude-code", None, Some("r1"), None, "parse", "bad json")
            .unwrap();
        store
            .log_import_error("mem0", Some("self"), Some("r2"), None, "upsert", "boom")
            .unwrap();
        assert_eq!(store.stats().unwrap().import_errors, 2);
    }

    /// `recent_import_errors(None, limit)` returns rows newest-first
    /// across every adapter; `recent_import_errors(Some(adapter), ...)`
    /// scopes to that adapter; `limit = 0` returns an empty Vec without
    /// touching the database.
    #[test]
    fn recent_import_errors_orders_newest_first_and_scopes_by_adapter() {
        let store = Store::open_in_memory().unwrap();
        // Insert in order so the natural id sequence also matches the
        // newest-first ORDER BY tiebreaker.
        store
            .log_import_error("claude-code", None, Some("a"), None, "parse", "first")
            .unwrap();
        store
            .log_import_error("mem0", Some("self"), Some("b"), None, "upsert", "second")
            .unwrap();
        store
            .log_import_error("claude-code", None, Some("c"), None, "scan", "third")
            .unwrap();

        let all = store.recent_import_errors(None, 10).unwrap();
        assert_eq!(all.len(), 3);
        // Newest first: "third", "second", "first".
        assert_eq!(all[0].error, "third");
        assert_eq!(all[1].error, "second");
        assert_eq!(all[2].error, "first");

        let claude = store.recent_import_errors(Some("claude-code"), 10).unwrap();
        assert_eq!(claude.len(), 2);
        assert_eq!(claude[0].error, "third");
        assert_eq!(claude[1].error, "first");

        let limited = store.recent_import_errors(None, 1).unwrap();
        assert_eq!(limited.len(), 1);
        assert_eq!(limited[0].error, "third");

        let zero = store.recent_import_errors(None, 0).unwrap();
        assert!(zero.is_empty());
    }

    // ─── Round-86 PR-78h: source_show backing helpers ────────────────

    /// Per-source error helper must filter by **both** adapter
    /// and instance — pinning the no-leakage invariant codex
    /// flagged as risk #2. Two mem0 instances each get a
    /// distinct error; the `(mem0, self-hosted)` query returns
    /// only the self-hosted one.
    #[test]
    fn source_show_recent_import_errors_for_source_does_not_leak_across_instances() {
        let store = Store::open_in_memory().unwrap();
        store
            .log_import_error(
                "mem0",
                Some("self-hosted"),
                Some("h1"),
                None,
                "parse",
                "self-error",
            )
            .unwrap();
        store
            .log_import_error(
                "mem0",
                Some("cloud"),
                Some("c1"),
                None,
                "parse",
                "cloud-error",
            )
            .unwrap();
        // A third row in a different adapter, must also stay out.
        store
            .log_import_error("claude-code", None, Some("cc1"), None, "parse", "cc-error")
            .unwrap();

        let scoped = store
            .recent_import_errors_for_source("mem0", Some("self-hosted"), 10)
            .unwrap();
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].error, "self-error");
        let cloud = store
            .recent_import_errors_for_source("mem0", Some("cloud"), 10)
            .unwrap();
        assert_eq!(cloud.len(), 1);
        assert_eq!(cloud[0].error, "cloud-error");
        let cc = store
            .recent_import_errors_for_source("claude-code", None, 10)
            .unwrap();
        assert_eq!(cc.len(), 1);
        assert_eq!(cc[0].error, "cc-error");
        // limit=0 short-circuits like the adapter-scoped variant.
        let zero = store
            .recent_import_errors_for_source("mem0", Some("self-hosted"), 0)
            .unwrap();
        assert!(zero.is_empty());
    }

    /// `get_source_with_counts` returns `Some` with the same
    /// JOIN-amplification-safe counts as `list_sources_with_counts`,
    /// and `None` for `(adapter, instance)` pairs that aren't in
    /// the registry (CLI/MCP map that to a loud "source not found").
    #[test]
    fn source_show_get_source_with_counts_returns_counts_or_none() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("claude-code", None, Some("/cc"), None)
            .unwrap();
        // Seed two records + tag one — the same shape R82's tests use.
        for n in ["a", "b"] {
            let r = make_record("claude-code", n, &format!("body {n}"), Kind::Fact);
            let c = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &c, None).unwrap();
        }
        let id_a = RecordId::from_parts("claude-code", None, "a");
        store
            .tag_record(
                &id_a,
                &["keep".into(), "todo".into()],
                UserTagOperation::Add,
            )
            .unwrap();

        // Hit: counts match.
        let hit = store
            .get_source_with_counts("claude-code", None)
            .unwrap()
            .expect("registered source");
        assert_eq!(hit.source.adapter, "claude-code");
        assert_eq!(hit.source.instance, "");
        assert_eq!(hit.record_count, 2);
        assert!(hit.chunk_count >= 2);
        assert_eq!(
            hit.tagged_record_count, 1,
            "two tags on one record = 1 tagged record"
        );

        // Miss: unknown adapter → None.
        assert!(store
            .get_source_with_counts("does-not-exist", None)
            .unwrap()
            .is_none());

        // Miss: wrong instance on a registered adapter → None.
        assert!(store
            .get_source_with_counts("claude-code", Some("not-registered"))
            .unwrap()
            .is_none());
    }

    /// A batch that mixes already-imported (raw_hash-equal → no-op) records
    /// with new records must report only the new records as upserted, and
    /// must leave the store's row count equal to total-distinct-records.
    #[test]
    fn upsert_records_batch_mixed_dedup_and_new_counts_only_new_rows() {
        let store = Store::open_in_memory().unwrap();

        // Seed two records via per-record path.
        let mut r1 = make_record("claude-code", "one", "first content", Kind::Fact);
        r1.provenance.raw_hash = "h1".into();
        let c1 = Chunker::default().chunk(&r1.id, &r1.content);
        store.upsert_record(&r1, &c1, None).unwrap();

        let mut r2 = make_record("claude-code", "two", "second content", Kind::Fact);
        r2.provenance.raw_hash = "h2".into();
        let c2 = Chunker::default().chunk(&r2.id, &r2.content);
        store.upsert_record(&r2, &c2, None).unwrap();

        // Now build a batch: r1 unchanged, r2 unchanged, r3 brand new.
        let mut r3 = make_record("claude-code", "three", "third content", Kind::Fact);
        r3.provenance.raw_hash = "h3".into();
        let c3 = Chunker::default().chunk(&r3.id, &r3.content);

        let batch: Vec<(&AnamnesisRecord, &[Chunk], Option<&str>)> = vec![
            (&r1, c1.as_slice(), None),
            (&r2, c2.as_slice(), None),
            (&r3, c3.as_slice(), None),
        ];
        let (recs, chunks_written) = store.upsert_records_batch(&batch).unwrap();

        assert_eq!(
            recs, 1,
            "only r3 was new, so the batch should report 1 upsert"
        );
        assert_eq!(
            chunks_written as usize,
            c3.len(),
            "batch must only write chunks for the new record"
        );

        let total_records: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM records", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            total_records, 3,
            "store should now hold all 3 distinct records"
        );
    }

    /// Two models with the same dim must coexist in the same per-dim
    /// vec0 table without collision. `model_id` is a PARTITION KEY so
    /// a search constrained to one model never sees the other's rows
    /// — proves the partition pruning is hooked up correctly.
    #[test]
    fn vec0_partition_isolates_models_at_same_dim() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("model-a").unwrap();
        let r = make_record("a", "x", "shared content", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();

        // Manually enqueue + complete for both models so the same chunk
        // ends up with two embeddings (same dim, different model).
        store.rebuild_embedding_jobs("model-b").unwrap();

        let jobs_a = store.claim_next_jobs("model-a", 16).unwrap();
        let jobs_b = store.claim_next_jobs("model-b", 16).unwrap();
        assert!(!jobs_a.is_empty() && !jobs_b.is_empty());

        let vec_a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let vec_b: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0];
        let vecs_a: Vec<Vec<f32>> = jobs_a.iter().map(|_| vec_a.clone()).collect();
        let vecs_b: Vec<Vec<f32>> = jobs_b.iter().map(|_| vec_b.clone()).collect();
        store.complete_jobs_batch(&jobs_a, &vecs_a).unwrap();
        store.complete_jobs_batch(&jobs_b, &vecs_b).unwrap();

        // A search for model-a's vector must hit model-a's row only.
        let hits = store
            .search_chunks_vec(&vec_a, "model-a", &SearchFilter::default(), 10)
            .unwrap();
        assert!(!hits.is_empty());
        assert!(
            hits[0].score > 0.99,
            "model-a query should match model-a embedding closely; got {}",
            hits[0].score
        );

        // Same query under a different model id must not return that
        // row — partition pruning kicks in.
        let cross = store
            .search_chunks_vec(&vec_a, "model-b", &SearchFilter::default(), 10)
            .unwrap();
        // Either: empty (no model-b chunk matches `vec_a` well) or hits
        // belong only to model-b. The contract is "no cross-model
        // leakage", which we verify by checking that the top model-b
        // hit's score is not the artificial 1.0 we'd see if we mixed.
        if let Some(h) = cross.first() {
            assert!(
                h.score < 0.99,
                "model-b query must not see model-a's perfect-match row; got {}",
                h.score
            );
        }
    }

    /// vec0 rows must be cleaned up when a record is re-chunked.
    /// `write_chunks` does `DELETE FROM record_chunks` which cascades
    /// the BLOB rows via FK, but vec0 is a virtual table without FK
    /// support — `delete_vec_rows_for_record` is the manual sync.
    #[test]
    fn vec0_rows_are_dropped_when_chunks_replaced() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("model-x").unwrap();
        let r = make_record("a", "x", "first content for chunking", Kind::Fact);
        let c1 = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c1, None).unwrap();

        let jobs = store.claim_next_jobs("model-x", 16).unwrap();
        let vecs: Vec<Vec<f32>> = jobs.iter().map(|_| vec![1.0, 2.0, 3.0, 4.0]).collect();
        store.complete_jobs_batch(&jobs, &vecs).unwrap();

        let count_before: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM chunk_embeddings_vec_d4", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count_before as usize, c1.len());

        // Re-chunk under brand-new content — write_chunks should clear
        // the vec0 rows for the old chunk_ids before the BLOB rows go.
        let mut r2 = r.clone();
        r2.content = "completely different second content".into();
        r2.provenance.raw_hash = "h2".into();
        let c2 = Chunker::default().chunk(&r2.id, &r2.content);
        store.upsert_record(&r2, &c2, None).unwrap();

        let count_after: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM chunk_embeddings_vec_d4", [], |r| {
                r.get(0)
            })
            .unwrap();
        // The old chunks' vec rows must be gone; only whatever embeddings
        // exist for the new content remain (jobs are pending; no new vec
        // rows yet).
        assert_eq!(
            count_after, 0,
            "stale vec0 rows from the old chunks should be deleted; found {count_after}"
        );
    }

    // ─── Round-69: MCP request metrics ──────────────────────────────

    fn mk_metric(tool: &str, ok: bool, duration_ms: i64, started_at: i64) -> McpRequestMetric {
        McpRequestMetric {
            started_at,
            tool: tool.into(),
            ok,
            duration_ms,
            result_count: if ok && tool == "search_memories" {
                Some(3)
            } else {
                None
            },
            error_kind: if ok { None } else { Some("missing_arg".into()) },
            mode: if tool == "search_memories" {
                Some("hybrid".into())
            } else {
                None
            },
            source: None,
            instance: None,
            limit_value: None,
        }
    }

    #[test]
    fn record_mcp_metric_round_trips() {
        let store = Store::open_in_memory().unwrap();
        store
            .record_mcp_request_metric(&mk_metric("search_memories", true, 12, 1_000))
            .unwrap();
        let summaries = store.summarize_mcp_request_metrics(None).unwrap();
        assert_eq!(summaries.len(), 1);
        let s = &summaries[0];
        assert_eq!(s.tool, "search_memories");
        assert_eq!(s.count, 1);
        assert_eq!(s.errors, 0);
        assert_eq!(s.p50_ms, 12);
        assert_eq!(s.last_ms, 12);
        assert_eq!(s.last_result_count, Some(3));
    }

    /// Percentiles must be ordered (p50 ≤ p95 ≤ p99) for any sample
    /// distribution. Easiest invariant to catch off-by-one bugs in
    /// the nearest-rank computation.
    #[test]
    fn summarize_mcp_metrics_percentiles_are_ordered() {
        let store = Store::open_in_memory().unwrap();
        for (i, d) in [3_i64, 8, 12, 15, 22, 41, 80, 110, 250, 1_400]
            .iter()
            .enumerate()
        {
            store
                .record_mcp_request_metric(&mk_metric(
                    "search_memories",
                    true,
                    *d,
                    1_000 + i as i64,
                ))
                .unwrap();
        }
        let s = &store.summarize_mcp_request_metrics(None).unwrap()[0];
        assert_eq!(s.count, 10);
        assert!(
            s.p50_ms <= s.p95_ms && s.p95_ms <= s.p99_ms,
            "p50={} p95={} p99={}",
            s.p50_ms,
            s.p95_ms,
            s.p99_ms,
        );
        // Tail of the synthetic distribution.
        assert_eq!(s.p99_ms, 1_400);
    }

    /// `ok = false` entries must count toward `errors`, not `count -
    /// errors`. Guards against a flag flip in the SQL or summary path.
    #[test]
    fn summarize_mcp_metrics_counts_errors_separately() {
        let store = Store::open_in_memory().unwrap();
        store
            .record_mcp_request_metric(&mk_metric("search_memories", true, 10, 1_000))
            .unwrap();
        store
            .record_mcp_request_metric(&mk_metric("search_memories", false, 7, 1_001))
            .unwrap();
        let s = &store.summarize_mcp_request_metrics(None).unwrap()[0];
        assert_eq!(s.count, 2);
        assert_eq!(s.errors, 1);
    }

    /// The 5000-row cap is the privacy + size guarantee. Writer must
    /// trim on every insert so the table never grows past it,
    /// regardless of how many requests show up.
    #[test]
    fn record_mcp_metric_self_caps_table() {
        let store = Store::open_in_memory().unwrap();
        // Punch past the cap by a small margin so the test stays fast.
        let extra = 25;
        for i in 0..(MCP_METRICS_CAP + extra) {
            store
                .record_mcp_request_metric(&mk_metric("search_memories", true, 10, 1_000 + i))
                .unwrap();
        }
        let n: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM mcp_request_metrics", [], |r| r.get(0))
            .unwrap();
        assert!(
            n <= MCP_METRICS_CAP,
            "row count {n} must be <= cap {MCP_METRICS_CAP}",
        );
        // The trimmed rows must be the *oldest*, so the most recent
        // started_at is still present.
        let most_recent: i64 = store
            .conn()
            .query_row("SELECT MAX(started_at) FROM mcp_request_metrics", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(most_recent, 1_000 + MCP_METRICS_CAP + extra - 1);
    }

    /// Privacy contract: the table must not contain *any* column
    /// capable of carrying user-typed content. If a future migration
    /// adds something like `query_text`, this guard fires and the
    /// reviewer has to think before approving.
    #[test]
    fn mcp_metrics_table_has_no_user_content_columns() {
        let store = Store::open_in_memory().unwrap();
        let cols: Vec<String> = {
            let conn = store.conn();
            let mut stmt = conn
                .prepare("SELECT name FROM pragma_table_info('mcp_request_metrics')")
                .unwrap();
            stmt.query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap()
        };
        let allowed = [
            "id",
            "started_at",
            "tool",
            "ok",
            "duration_ms",
            "result_count",
            "error_kind",
            "mode",
            "source",
            "instance",
            "limit_value",
        ];
        for c in &cols {
            assert!(
                allowed.contains(&c.as_str()),
                "unexpected column {c}: would need a privacy review before landing"
            );
        }
    }

    // ─── Round-72 PR-72a: forget_record + tombstone suppression ─────

    /// Forget removes the record + cascades chunk_embeddings + vec0 +
    /// raw_artifacts (via FK), and writes the tombstone the importer
    /// can later consult.
    #[test]
    fn forget_record_deletes_indexes_and_writes_tombstone() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("local:fake:1").unwrap();
        let mut r = make_record("claude-code", "rec-1", "secret content", Kind::Fact);
        r.provenance.raw_hash = "h-1".into();
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        // Drive an embedding so vec0 has something to drop.
        let jobs = store.claim_next_jobs("local:fake:1", 16).unwrap();
        let vecs: Vec<Vec<f32>> = jobs.iter().map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect();
        store.complete_jobs_batch(&jobs, &vecs).unwrap();
        // Sanity: record + chunks + embeddings + tombstone all in expected state.
        let n_records: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM records WHERE id = ?1",
                params![r.id.0],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(n_records, 1);
        let n_tombstones_before: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM record_tombstones", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(n_tombstones_before, 0);

        let outcome = store.forget_record(&r.id, Some("user requested")).unwrap();
        match outcome {
            ForgetRecordOutcome::Forgotten(rec) => {
                assert_eq!(rec.adapter, "claude-code");
                assert_eq!(rec.native_id, "rec-1");
                assert_eq!(rec.raw_hash, "h-1");
                assert_eq!(rec.reason.as_deref(), Some("user requested"));
            }
            other => panic!("expected Forgotten, got {other:?}"),
        }

        let n_records_after: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM records WHERE id = ?1",
                params![r.id.0],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(n_records_after, 0, "record row should be gone");
        // FK cascade should have cleared the chunk rows too.
        let n_chunks: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM record_chunks WHERE record_id = ?1",
                params![r.id.0],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(n_chunks, 0);
        let n_embeddings: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM chunk_embeddings_vec_d4", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(
            n_embeddings, 0,
            "vec0 rows for forgotten chunks should be cleared"
        );
        let n_tombstones_after: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM record_tombstones", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(n_tombstones_after, 1);
    }

    /// A second `forget` on the same id must return `AlreadyForgotten`
    /// with the original tombstone — never silently double-write.
    #[test]
    fn forget_record_second_call_returns_already_forgotten() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("mem0", "m1", "x", Kind::Fact);
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        let first = store.forget_record(&r.id, Some("once")).unwrap();
        let second = store.forget_record(&r.id, Some("twice")).unwrap();
        assert!(matches!(first, ForgetRecordOutcome::Forgotten(_)));
        match second {
            ForgetRecordOutcome::AlreadyForgotten(rec) => {
                assert_eq!(
                    rec.reason.as_deref(),
                    Some("once"),
                    "must preserve original reason"
                );
            }
            other => panic!("expected AlreadyForgotten, got {other:?}"),
        }
    }

    #[test]
    fn forget_record_returns_not_found_when_id_never_existed() {
        let store = Store::open_in_memory().unwrap();
        let phantom = RecordId::from_parts("claude-code", None, "never-existed");
        let outcome = store.forget_record(&phantom, None).unwrap();
        assert!(matches!(outcome, ForgetRecordOutcome::NotFound));
    }

    // ─── Round-83 PR-78e: preview_forget_record dry-run ─────────────

    /// Live-record preview reports cascade counts that match what
    /// the real `forget_record` would touch, and **does not mutate
    /// the store** — the record is still searchable + tombstone
    /// table empty + tags untouched after the preview returns.
    #[test]
    fn preview_forget_record_counts_live_cascade_without_mutating() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("claude-code", "rec-prev", "preview body", Kind::Fact);
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        store
            .tag_record(
                &r.id,
                &["keep".into(), "todo".into()],
                UserTagOperation::Add,
            )
            .unwrap();

        let preview = store.preview_forget_record(&r.id, Some("test")).unwrap();
        match preview {
            ForgetRecordPreview::WouldForget {
                would_delete,
                tombstone_preview,
            } => {
                assert_eq!(would_delete.records, 1);
                assert_eq!(would_delete.record_chunks, chunks.len() as u64);
                assert_eq!(would_delete.user_record_tags, 2);
                // No active model in this test → embeddings/jobs/vec0 are 0.
                assert_eq!(would_delete.chunk_embeddings, 0);
                assert_eq!(would_delete.embedding_jobs, 0);
                assert_eq!(would_delete.vec0_rows, 0);
                assert_eq!(tombstone_preview.adapter, "claude-code");
                assert_eq!(tombstone_preview.native_id, "rec-prev");
                assert_eq!(tombstone_preview.reason.as_deref(), Some("test"));
            }
            other => panic!("expected WouldForget, got {other:?}"),
        }

        // Mutation guard: the record is still live + tags intact +
        // tombstone table empty.
        assert!(store.get_record(&r.id).unwrap().is_some());
        assert_eq!(store.user_tags(&r.id).unwrap().len(), 2);
        assert_eq!(
            store
                .list_forgotten(&ListForgottenFilter {
                    source: None,
                    instance: None,
                    limit: 100,
                })
                .unwrap()
                .len(),
            0,
            "dry-run must not write a tombstone"
        );
    }

    /// Already-forgotten records report `AlreadyForgotten` with the
    /// existing tombstone shape, no counts, no writes.
    #[test]
    fn preview_forget_record_already_forgotten_reports_tombstone() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("mem0", "rec-gone", "body", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
        store.forget_record(&r.id, Some("init")).unwrap();

        let preview = store.preview_forget_record(&r.id, Some("retry")).unwrap();
        match preview {
            ForgetRecordPreview::AlreadyForgotten(t) => {
                assert_eq!(t.record_id, r.id);
                assert_eq!(t.adapter, "mem0");
                assert_eq!(
                    t.reason.as_deref(),
                    Some("init"),
                    "must echo the *existing* tombstone reason, not the preview's reason"
                );
            }
            other => panic!("expected AlreadyForgotten, got {other:?}"),
        }
    }

    /// `NotFound` when neither a record nor a tombstone exists —
    /// matches `forget_record` precedent so the CLI dry-run can
    /// exit non-zero on typo'd ids without a special case.
    #[test]
    fn preview_forget_record_not_found_when_id_never_existed() {
        let store = Store::open_in_memory().unwrap();
        let phantom = RecordId::from_parts("claude-code", None, "never-existed");
        let preview = store.preview_forget_record(&phantom, None).unwrap();
        assert!(matches!(preview, ForgetRecordPreview::NotFound));
    }

    /// Preview-then-real-forget invariant: the counts the preview
    /// emits match what the real forget actually removes. Pinned
    /// because the preview and the delete walk separate code paths
    /// and would silently drift if anyone changed one without the
    /// other.
    #[test]
    fn preview_forget_record_counts_match_real_forget_cascade() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("claude-code", "rec-match", "match body", Kind::Fact);
        let c = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c, None).unwrap();
        store
            .tag_record(
                &r.id,
                &["a".into(), "b".into(), "c".into()],
                UserTagOperation::Add,
            )
            .unwrap();

        // Take the preview snapshot first.
        let counts = match store.preview_forget_record(&r.id, None).unwrap() {
            ForgetRecordPreview::WouldForget { would_delete, .. } => would_delete,
            other => panic!("expected WouldForget, got {other:?}"),
        };

        // Pre-forget row counts.
        let (chunks_before, tags_before): (i64, i64) = {
            let c = store.conn();
            let chunks: i64 = c
                .query_row(
                    "SELECT COUNT(*) FROM record_chunks WHERE record_id = ?1",
                    params![r.id.0],
                    |row| row.get(0),
                )
                .unwrap();
            let tags: i64 = c
                .query_row(
                    "SELECT COUNT(*) FROM user_record_tags WHERE record_id = ?1",
                    params![r.id.0],
                    |row| row.get(0),
                )
                .unwrap();
            (chunks, tags)
        };

        // Real forget runs and tombstones get written.
        store.forget_record(&r.id, None).unwrap();

        assert_eq!(counts.record_chunks, chunks_before as u64);
        assert_eq!(counts.user_record_tags, tags_before as u64);
        assert_eq!(counts.records, 1);
        // Post-forget cleanup.
        let chunks_after: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM record_chunks WHERE record_id = ?1",
                params![r.id.0],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(chunks_after, 0, "real forget cascaded chunks");
    }

    /// After a record is forgotten, re-running upsert with the same
    /// natural key (i.e. what a re-import would do) must be a no-op
    /// — no `records` row resurrected, no chunks written, no
    /// embedding jobs enqueued.
    #[test]
    fn tombstoned_record_upsert_is_suppressed() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("local:fake:1").unwrap();
        let mut r = make_record("claude-code", "rec-x", "first body", Kind::Fact);
        r.provenance.raw_hash = "h-first".into();
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        store.forget_record(&r.id, None).unwrap();

        // Simulate a re-import — same natural key, but the source
        // content may have drifted (raw_hash changes too).
        let mut reimport = make_record("claude-code", "rec-x", "second body", Kind::Fact);
        reimport.provenance.raw_hash = "h-second".into();
        let new_chunks = Chunker::default().chunk(&reimport.id, &reimport.content);
        let (recs, chunks_n) = store.upsert_record(&reimport, &new_chunks, None).unwrap();
        assert_eq!(
            (recs, chunks_n),
            (0, 0),
            "tombstone must suppress re-upsert"
        );

        let n: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM records WHERE id = ?1",
                params![reimport.id.0],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(n, 0);
        let n_jobs: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM embedding_jobs", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            n_jobs, 0,
            "no embedding jobs should be enqueued for a forgotten record"
        );
    }

    /// Same suppression, batched. Importer-side path goes through
    /// `upsert_records_batch` for the 1795-record corpus, so the
    /// tombstone gate has to be inside the batch loop too.
    #[test]
    fn tombstoned_record_batch_upsert_is_suppressed() {
        let store = Store::open_in_memory().unwrap();
        let mut r1 = make_record("claude-code", "a", "alpha", Kind::Fact);
        r1.provenance.raw_hash = "h-a".into();
        let mut r2 = make_record("claude-code", "b", "beta", Kind::Fact);
        r2.provenance.raw_hash = "h-b".into();
        let c1 = Chunker::default().chunk(&r1.id, &r1.content);
        let c2 = Chunker::default().chunk(&r2.id, &r2.content);
        let batch = vec![(&r1, c1.as_slice(), None), (&r2, c2.as_slice(), None)];
        let (recs, _) = store.upsert_records_batch(&batch).unwrap();
        assert_eq!(recs, 2);

        store.forget_record(&r1.id, None).unwrap();

        // Re-import the same batch — r1 is tombstoned and must be
        // suppressed, r2 is unchanged so it's a raw_hash no-op.
        let (re_recs, re_chunks) = store.upsert_records_batch(&batch).unwrap();
        assert_eq!(
            (re_recs, re_chunks),
            (0, 0),
            "tombstoned r1 must not resurrect; unchanged r2 is raw_hash no-op",
        );
    }

    // ─── Round-74: list_forgotten ───────────────────────────────────

    /// Helper: seed `n` forgettable records for `(adapter, instance)`
    /// and forget all of them. `instance = None` uses the default ("").
    fn seed_and_forget(store: &Store, adapter: &str, n: usize) -> Vec<RecordId> {
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let nid = format!("rec-{i}");
            let mut r = make_record(adapter, &nid, &format!("content {i}"), Kind::Fact);
            r.provenance.raw_hash = format!("h-{adapter}-{i}");
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &chunks, None).unwrap();
            store
                .forget_record(&r.id, Some(&format!("test forget #{i}")))
                .unwrap();
            ids.push(r.id);
        }
        ids
    }

    /// `list_forgotten` returns rows newest-first, honours the
    /// `limit` cap, and ignores `MAX_LIST_LIMIT` (its own
    /// `LIST_FORGOTTEN_MAX_LIMIT = 100` is the binding cap).
    #[test]
    fn list_forgotten_returns_newest_first_respecting_limit() {
        let store = Store::open_in_memory().unwrap();
        seed_and_forget(&store, "claude-code", 5);
        let filter = ListForgottenFilter {
            source: None,
            instance: None,
            limit: 3,
        };
        let rows = store.list_forgotten(&filter).unwrap();
        assert_eq!(rows.len(), 3, "limit must be respected");
        for w in rows.windows(2) {
            assert!(
                w[0].forgotten_at >= w[1].forgotten_at,
                "must be sorted newest-first"
            );
        }
    }

    /// Source filter narrows to one adapter even when other
    /// adapters' tombstones exist.
    #[test]
    fn list_forgotten_filters_by_source() {
        let store = Store::open_in_memory().unwrap();
        seed_and_forget(&store, "claude-code", 2);
        seed_and_forget(&store, "mem0", 3);
        let filter = ListForgottenFilter {
            source: Some("mem0".into()),
            instance: None,
            limit: 100,
        };
        let rows = store.list_forgotten(&filter).unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|r| r.adapter == "mem0"));
    }

    // ─── Round-90 PR-78l: count_forgotten_by_source ─────────────────

    /// Tombstone counts group by `(adapter, instance)` and
    /// respect the same source/instance filter as `list_forgotten`.
    /// Counts reflect the **full** matching set, not the page —
    /// that's the whole point: an operator on page 1 of 100
    /// can still see "there are 137 tombstones total."
    #[test]
    fn count_forgotten_by_source_aggregates_per_adapter_instance() {
        let store = Store::open_in_memory().unwrap();
        seed_and_forget(&store, "claude-code", 5);
        seed_and_forget(&store, "mem0", 2);

        let all = store
            .count_forgotten_by_source(&ListForgottenFilter::default())
            .unwrap();
        let cc = all.iter().find(|b| b.adapter == "claude-code").unwrap();
        let mem = all.iter().find(|b| b.adapter == "mem0").unwrap();
        assert_eq!(cc.forgotten_count, 5);
        assert_eq!(mem.forgotten_count, 2);
        // Order: count DESC, then adapter ASC — claude-code (5) before mem0 (2).
        assert_eq!(all[0].adapter, "claude-code");
        assert_eq!(all[1].adapter, "mem0");

        // source filter respected.
        let scoped = store
            .count_forgotten_by_source(&ListForgottenFilter {
                source: Some("mem0".into()),
                instance: None,
                limit: 0,
            })
            .unwrap();
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].adapter, "mem0");
        assert_eq!(scoped[0].forgotten_count, 2);
    }

    /// Empty tombstone table returns an empty Vec — not an
    /// error, not a `Some(zero bucket)`. CLI/MCP wire shapes
    /// turn this into `{"total": 0, "by_source": []}`.
    #[test]
    fn count_forgotten_by_source_returns_empty_when_no_tombstones() {
        let store = Store::open_in_memory().unwrap();
        let out = store
            .count_forgotten_by_source(&ListForgottenFilter::default())
            .unwrap();
        assert!(out.is_empty());
    }

    /// `limit = 0` is a guard, not an empty-result short-circuit —
    /// clamp to 1 so the caller always sees at least the most-
    /// recent entry. Caps above `LIST_FORGOTTEN_MAX_LIMIT` clamp
    /// down to `LIST_FORGOTTEN_MAX_LIMIT`.
    #[test]
    fn list_forgotten_clamps_limit_into_window() {
        let store = Store::open_in_memory().unwrap();
        seed_and_forget(&store, "claude-code", 4);
        // 0 → clamped to 1
        let rows = store
            .list_forgotten(&ListForgottenFilter {
                source: None,
                instance: None,
                limit: 0,
            })
            .unwrap();
        assert_eq!(rows.len(), 1);
        // > cap → clamped to cap, but only 4 rows exist
        let rows = store
            .list_forgotten(&ListForgottenFilter {
                source: None,
                instance: None,
                limit: LIST_FORGOTTEN_MAX_LIMIT * 10,
            })
            .unwrap();
        assert_eq!(rows.len(), 4);
    }

    // ─── Round-75: unforget_record ──────────────────────────────────

    /// `unforget` removes the tombstone and returns the snapshot
    /// that was just deleted, but does NOT recreate the live
    /// `records` row (the tombstone has provenance only, not the
    /// original content).
    #[test]
    fn unforget_removes_tombstone_but_does_not_resurrect_record() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("claude-code", "rec-a", "alpha content", Kind::Fact);
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        store.forget_record(&r.id, Some("oops")).unwrap();

        let outcome = store.unforget_record(&r.id).unwrap();
        match outcome {
            UnforgetRecordOutcome::Unforgotten(rec) => {
                assert_eq!(rec.record_id, r.id);
                assert_eq!(rec.adapter, "claude-code");
                assert_eq!(rec.reason.as_deref(), Some("oops"));
            }
            UnforgetRecordOutcome::NotForgotten => panic!("expected Unforgotten"),
        }
        let n_tombstones: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM record_tombstones", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(n_tombstones, 0, "tombstone row should be gone");
        let n_records: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM records WHERE id = ?1",
                params![r.id.0],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            n_records, 0,
            "unforget must NOT resurrect the record itself — re-import is required",
        );
    }

    /// After `unforget`, a re-upsert of the same record is no
    /// longer suppressed — the tombstone gate is gone. This is
    /// what makes `unforget` actually useful (otherwise it would
    /// just remove an entry from `list_forgotten` with no
    /// behavioural effect).
    #[test]
    fn upsert_after_unforget_is_accepted_again() {
        let store = Store::open_in_memory().unwrap();
        let mut r = make_record("mem0", "m1", "first content", Kind::Fact);
        r.provenance.raw_hash = "h-first".into();
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        store.forget_record(&r.id, None).unwrap();

        // Suppression still holds before unforget.
        let mut re = make_record("mem0", "m1", "second content", Kind::Fact);
        re.provenance.raw_hash = "h-second".into();
        let new_chunks = Chunker::default().chunk(&re.id, &re.content);
        let (recs, _) = store.upsert_record(&re, &new_chunks, None).unwrap();
        assert_eq!(recs, 0, "tombstone must suppress upsert");

        store.unforget_record(&r.id).unwrap();

        let (recs, chunks_n) = store.upsert_record(&re, &new_chunks, None).unwrap();
        assert_eq!(
            (recs, chunks_n as usize),
            (1, new_chunks.len()),
            "after unforget the same (adapter,instance,native_id) must be importable again"
        );
    }

    #[test]
    fn unforget_record_with_no_tombstone_returns_not_forgotten() {
        let store = Store::open_in_memory().unwrap();
        let phantom = RecordId::from_parts("claude-code", None, "never-tombstoned");
        let outcome = store.unforget_record(&phantom).unwrap();
        assert!(matches!(outcome, UnforgetRecordOutcome::NotForgotten));
    }

    /// Repeated `unforget` calls must NOT silently succeed — second
    /// call should return `NotForgotten` because there's nothing
    /// left to remove. (Distinct from `forget`'s `AlreadyForgotten`,
    /// which carries the original tombstone; here there's no
    /// payload to return.)
    #[test]
    fn unforget_record_second_call_returns_not_forgotten() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("claude-code", "rec-twice", "x", Kind::Fact);
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        store.forget_record(&r.id, None).unwrap();
        let first = store.unforget_record(&r.id).unwrap();
        let second = store.unforget_record(&r.id).unwrap();
        assert!(matches!(first, UnforgetRecordOutcome::Unforgotten(_)));
        assert!(matches!(second, UnforgetRecordOutcome::NotForgotten));
    }

    // ─── Round-95 PR-78q: preview_unforget_record ─────────────────

    /// Preview returns the existing tombstone without touching
    /// the store. List-forgotten still sees the row afterward.
    #[test]
    fn preview_unforget_record_returns_tombstone_without_deleting() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("claude-code", "rec-prev-un", "x", Kind::Fact);
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        store.forget_record(&r.id, Some("test-reason")).unwrap();

        let preview = store.preview_unforget_record(&r.id).unwrap();
        match preview {
            UnforgetRecordOutcome::Unforgotten(rec) => {
                assert_eq!(rec.record_id, r.id);
                assert_eq!(rec.reason.as_deref(), Some("test-reason"));
            }
            UnforgetRecordOutcome::NotForgotten => {
                panic!("preview must return the existing tombstone")
            }
        }

        // Mutation guard: tombstone still in the table.
        let still_there = store
            .list_forgotten(&ListForgottenFilter {
                source: None,
                instance: None,
                limit: 10,
            })
            .unwrap();
        assert_eq!(
            still_there.len(),
            1,
            "preview must NOT delete the tombstone"
        );
        assert_eq!(still_there[0].record_id, r.id);
    }

    /// Missing tombstone → `NotForgotten`, matches the real
    /// `unforget_record` shape so CLI/MCP can branch uniformly.
    #[test]
    fn preview_unforget_record_returns_not_forgotten_for_unknown_id() {
        let store = Store::open_in_memory().unwrap();
        let phantom = RecordId::from_parts("claude-code", None, "phantom");
        let preview = store.preview_unforget_record(&phantom).unwrap();
        assert!(matches!(preview, UnforgetRecordOutcome::NotForgotten));
    }

    // ─── Round-77: list_duplicate_raw_hashes ────────────────────────

    /// Helper: insert a record with a forced raw_hash so the
    /// grouping behaviour is deterministic.
    fn seed_with_raw_hash(store: &Store, adapter: &str, native: &str, hash: &str) -> RecordId {
        let mut r = make_record(
            adapter,
            native,
            &format!("{adapter}|{native} content"),
            Kind::Fact,
        );
        r.provenance.raw_hash = hash.into();
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        r.id
    }

    fn seed_with_raw_hash_instance(
        store: &Store,
        adapter: &str,
        native: &str,
        instance: &str,
        hash: &str,
    ) -> RecordId {
        let mut r = make_record(
            adapter,
            native,
            &format!("{adapter}:{instance}|{native} content"),
            Kind::Fact,
        );
        r.source.instance = Some(instance.into());
        r.id = RecordId::from_parts(adapter, Some(instance), native);
        r.provenance.raw_hash = hash.into();
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        r.id
    }

    /// The report must skip raw_hashes with only one record
    /// (singletons are not duplicates) and include only live
    /// records (`forget` removes the row from `records`).
    #[test]
    fn list_duplicate_raw_hashes_returns_only_hashes_with_multiple_live_records() {
        let store = Store::open_in_memory().unwrap();
        // Two duplicates on h-shared, one singleton on h-solo.
        let _a = seed_with_raw_hash(&store, "claude-code", "a", "h-shared");
        let _b = seed_with_raw_hash(&store, "mem0", "b", "h-shared");
        let _c = seed_with_raw_hash(&store, "claude-code", "c", "h-solo");

        let groups = store.list_duplicate_raw_hashes(20).unwrap();
        assert_eq!(groups.len(), 1, "only the >1-row hash is a duplicate group");
        let g = &groups[0];
        assert_eq!(g.raw_hash, "h-shared");
        assert_eq!(g.records.len(), 2);
        // Both adapters present.
        let mut adapters: Vec<_> = g.records.iter().map(|r| r.adapter.clone()).collect();
        adapters.sort();
        assert_eq!(adapters, vec!["claude-code", "mem0"]);
    }

    /// Group ordering: larger groups before smaller; within a tie,
    /// the group whose newest record is more recent comes first.
    #[test]
    fn list_duplicate_raw_hashes_orders_by_group_size_then_newest() {
        let store = Store::open_in_memory().unwrap();
        // Group A: size 3 (oldest).
        for n in ["a1", "a2", "a3"] {
            seed_with_raw_hash(&store, "claude-code", n, "h-A");
        }
        // Group B: size 2 (newer than A, but smaller).
        std::thread::sleep(std::time::Duration::from_millis(10));
        for n in ["b1", "b2"] {
            seed_with_raw_hash(&store, "mem0", n, "h-B");
        }
        let groups = store.list_duplicate_raw_hashes(20).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].raw_hash, "h-A", "size 3 must outrank size 2");
        assert_eq!(groups[1].raw_hash, "h-B");
    }

    /// Limit is clamped: 0 → 1, anything past the cap → cap.
    /// Verifies the report can't be used to dump unbounded
    /// provenance in a single call.
    #[test]
    fn list_duplicate_raw_hashes_clamps_limit() {
        let store = Store::open_in_memory().unwrap();
        for i in 0..5 {
            seed_with_raw_hash(&store, "claude-code", &format!("g{i}-a"), &format!("h-{i}"));
            seed_with_raw_hash(&store, "claude-code", &format!("g{i}-b"), &format!("h-{i}"));
        }
        // limit = 0 clamps to 1.
        let groups = store.list_duplicate_raw_hashes(0).unwrap();
        assert_eq!(groups.len(), 1);
        // limit > CAP clamps but the actual data is 5 groups.
        let groups = store
            .list_duplicate_raw_hashes(LIST_DUPLICATE_RAW_HASHES_MAX_LIMIT * 10)
            .unwrap();
        assert_eq!(groups.len(), 5);
    }

    /// A forgotten record disappears from the report because
    /// `forget_record` deletes the live `records` row.
    /// This is what makes dedupe + forget compose without
    /// needing an "exclude tombstoned" SQL filter.
    #[test]
    fn forgotten_record_disappears_from_duplicate_report() {
        let store = Store::open_in_memory().unwrap();
        let a = seed_with_raw_hash(&store, "claude-code", "a", "h-shared");
        let _b = seed_with_raw_hash(&store, "mem0", "b", "h-shared");
        // Pre-forget: one group of 2.
        assert_eq!(store.list_duplicate_raw_hashes(20).unwrap().len(), 1);
        // After forget: now only 1 live record → no duplicate group.
        store.forget_record(&a, None).unwrap();
        assert_eq!(
            store.list_duplicate_raw_hashes(20).unwrap().len(),
            0,
            "group should drop out once forget left only 1 live sibling",
        );
    }

    // ─── Round-80: list_duplicate_raw_hashes_filtered ───────────────

    /// `--source` keeps a group if ≥1 member matches, AND returns
    /// the whole sibling set (the non-matching siblings stay
    /// visible because the operator may want to `forget` either
    /// side).
    #[test]
    fn list_duplicate_raw_hashes_filtered_keeps_group_whole_when_source_partial_match() {
        let store = Store::open_in_memory().unwrap();
        // h-mixed: one mem0 + one claude-code, both sharing hash.
        let _m = seed_with_raw_hash(&store, "mem0", "m1", "h-mixed");
        let _c = seed_with_raw_hash(&store, "claude-code", "c1", "h-mixed");

        let groups = store
            .list_duplicate_raw_hashes_filtered(&DuplicateRawHashFilter {
                source: Some("mem0".into()),
                instance: None,
                limit: 20,
            })
            .unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].records.len(),
            2,
            "filter scopes eligibility, NOT membership — both siblings must surface"
        );
        let adapters: std::collections::BTreeSet<_> = groups[0]
            .records
            .iter()
            .map(|r| r.adapter.as_str())
            .collect();
        assert!(adapters.contains("mem0"));
        assert!(adapters.contains("claude-code"));
    }

    /// A duplicate group with no matching member must be excluded.
    #[test]
    fn list_duplicate_raw_hashes_filtered_excludes_groups_with_no_source_match() {
        let store = Store::open_in_memory().unwrap();
        // h-A: two claude-code records (no mem0).
        let _a1 = seed_with_raw_hash(&store, "claude-code", "a1", "h-A");
        let _a2 = seed_with_raw_hash(&store, "claude-code", "a2", "h-A");
        // h-B: two mem0 records.
        let _b1 = seed_with_raw_hash(&store, "mem0", "b1", "h-B");
        let _b2 = seed_with_raw_hash(&store, "mem0", "b2", "h-B");

        let groups = store
            .list_duplicate_raw_hashes_filtered(&DuplicateRawHashFilter {
                source: Some("mem0".into()),
                instance: None,
                limit: 20,
            })
            .unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].raw_hash, "h-B");
    }

    /// **Filter-before-LIMIT** discipline: a huge irrelevant group
    /// must not eat the limit. With `limit=1` and a 3-row
    /// claude-code group ranked first, the mem0 group still wins
    /// because the filter narrows eligibility first.
    #[test]
    fn list_duplicate_raw_hashes_filtered_limit_not_starved_by_irrelevant_group() {
        let store = Store::open_in_memory().unwrap();
        // Irrelevant: 3-row claude-code group (highest rank
        // without the filter).
        for n in ["x1", "x2", "x3"] {
            seed_with_raw_hash(&store, "claude-code", n, "h-X");
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        // Relevant: 2-row mem0 group.
        for n in ["y1", "y2"] {
            seed_with_raw_hash(&store, "mem0", n, "h-Y");
        }

        let groups = store
            .list_duplicate_raw_hashes_filtered(&DuplicateRawHashFilter {
                source: Some("mem0".into()),
                instance: None,
                limit: 1,
            })
            .unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].raw_hash, "h-Y",
            "filter must run before LIMIT — h-X would otherwise outrank h-Y on size"
        );
    }

    /// `instance` alone narrows eligibility without `source`,
    /// matching the CLI shape where the user might want
    /// "everything from instance=primary."
    #[test]
    fn list_duplicate_raw_hashes_filtered_by_instance_alone() {
        let store = Store::open_in_memory().unwrap();

        // Helper to set adapter+instance on a forced raw_hash.
        // `instance` lives on `source`, not `provenance`.
        let seed_inst = |adapter: &str, native: &str, instance: &str, hash: &str| {
            let mut r = make_record(
                adapter,
                native,
                &format!("{adapter}|{native} content"),
                Kind::Fact,
            );
            r.source.instance = Some(instance.into());
            // RecordId is derived from (adapter, instance, native_id),
            // so re-derive it after stamping the instance — otherwise
            // both p1 and p2 collide on the no-instance id and the
            // second upsert overwrites the first.
            r.id = RecordId::from_parts(adapter, Some(instance), native);
            r.provenance.raw_hash = hash.into();
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &chunks, None).unwrap();
        };

        // h-P: contains a record on instance=primary (eligible).
        seed_inst("mem0", "p1", "primary", "h-P");
        seed_inst("mem0", "p2", "secondary", "h-P");
        // h-S: only secondary (NOT eligible under instance=primary).
        seed_inst("mem0", "s1", "secondary", "h-S");
        seed_inst("mem0", "s2", "secondary", "h-S");

        let groups = store
            .list_duplicate_raw_hashes_filtered(&DuplicateRawHashFilter {
                source: None,
                instance: Some("primary".into()),
                limit: 20,
            })
            .unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].raw_hash, "h-P");
        // Full sibling set still returned, secondary included.
        assert_eq!(groups[0].records.len(), 2);
    }

    // ─── Round-104 PR-78z: dedupe source multi-value OR ──────────────

    /// `source = "mem0,claude-code"` is the OR filter: groups
    /// whose members include at least one record from any
    /// listed adapter survive. Groups whose members live
    /// entirely outside the OR-set drop. Whitespace and empty
    /// tokens normalise via core's `parse_csv_filter`.
    #[test]
    fn list_duplicate_raw_hashes_filtered_multi_source_or_keeps_eligible_groups() {
        let store = Store::open_in_memory().unwrap();
        // 3 duplicate groups, each restricted to a single adapter.
        for n in ["m1", "m2"] {
            seed_with_raw_hash(&store, "mem0", n, "h-mem");
        }
        for n in ["c1", "c2"] {
            seed_with_raw_hash(&store, "claude-code", n, "h-cc");
        }
        for n in ["x1", "x2"] {
            seed_with_raw_hash(&store, "codex", n, "h-cx");
        }

        // mem0 + claude-code wins both; codex must drop.
        let groups = store
            .list_duplicate_raw_hashes_filtered(&DuplicateRawHashFilter {
                source: Some("mem0, , claude-code".into()),
                instance: None,
                limit: 20,
            })
            .unwrap();
        let hashes: std::collections::BTreeSet<_> =
            groups.iter().map(|g| g.raw_hash.as_str()).collect();
        assert_eq!(
            hashes,
            ["h-mem", "h-cc"].into_iter().collect(),
            "expected only the two eligible groups; got {hashes:?}"
        );
    }

    /// Multi-source OR also preserves mixed-group whole-sibling
    /// semantics: if a group has 1 mem0 + 1 codex and we filter
    /// to `mem0,claude-code`, the group survives (mem0
    /// qualifies) AND the codex sibling is still returned in
    /// the records[] list so the operator can decide which to
    /// forget.
    #[test]
    fn list_duplicate_raw_hashes_filtered_multi_source_or_keeps_whole_siblings() {
        let store = Store::open_in_memory().unwrap();
        seed_with_raw_hash(&store, "mem0", "m1", "h-mixed");
        seed_with_raw_hash(&store, "codex", "x1", "h-mixed");

        let groups = store
            .list_duplicate_raw_hashes_filtered(&DuplicateRawHashFilter {
                source: Some("mem0,claude-code".into()),
                instance: None,
                limit: 20,
            })
            .unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].records.len(), 2);
        let adapters: std::collections::BTreeSet<_> = groups[0]
            .records
            .iter()
            .map(|r| r.adapter.as_str())
            .collect();
        assert!(adapters.contains("mem0"));
        assert!(adapters.contains("codex"), "non-matching sibling stays");
    }

    /// Aggregate counts honour the same multi-source eligibility
    /// as the list helper — `by_source[]` reports every
    /// surviving group's adapters, and `total_groups` /
    /// `total_records` reflect the full multi-source-eligible
    /// set (filter ignores `limit`).
    #[test]
    fn count_duplicate_raw_hashes_by_source_multi_source_or_aggregates_full_set() {
        let store = Store::open_in_memory().unwrap();
        for n in ["m1", "m2"] {
            seed_with_raw_hash(&store, "mem0", n, "h-mem");
        }
        for n in ["c1", "c2"] {
            seed_with_raw_hash(&store, "claude-code", n, "h-cc");
        }
        for n in ["x1", "x2"] {
            seed_with_raw_hash(&store, "codex", n, "h-cx");
        }

        let counts = store
            .count_duplicate_raw_hashes_by_source(&DuplicateRawHashFilter {
                source: Some("mem0,claude-code".into()),
                instance: None,
                limit: 1,
            })
            .unwrap();
        assert_eq!(counts.total_groups, 2, "h-mem + h-cc are eligible");
        assert_eq!(counts.total_records, 4);
        // codex group must not contribute to by_source.
        assert!(
            counts.by_source.iter().all(|b| b.adapter != "codex"),
            "codex must be excluded from by_source: {:?}",
            counts.by_source
        );
    }

    // ─── Round-115: dedupe instance multi-value OR ───────────────────

    /// `instance = "prod,dev"` is an OR filter, and mixed groups
    /// still return their full sibling set. `qa` is only visible
    /// when it shares a hash with a matching instance.
    #[test]
    fn list_duplicate_raw_hashes_filtered_multi_instance_or_keeps_eligible_groups() {
        let store = Store::open_in_memory().unwrap();
        seed_with_raw_hash_instance(&store, "mem0", "p1", "prod", "h-prod");
        seed_with_raw_hash_instance(&store, "mem0", "p2", "prod", "h-prod");
        seed_with_raw_hash_instance(&store, "mem0", "d1", "dev", "h-dev");
        seed_with_raw_hash_instance(&store, "mem0", "q1", "qa", "h-dev");
        seed_with_raw_hash_instance(&store, "mem0", "q2", "qa", "h-qa");
        seed_with_raw_hash_instance(&store, "mem0", "q3", "qa", "h-qa");

        let groups = store
            .list_duplicate_raw_hashes_filtered(&DuplicateRawHashFilter {
                source: Some("mem0".into()),
                instance: Some("prod, dev".into()),
                limit: 20,
            })
            .unwrap();
        let hashes: std::collections::BTreeSet<_> =
            groups.iter().map(|g| g.raw_hash.as_str()).collect();
        assert_eq!(hashes, ["h-dev", "h-prod"].into_iter().collect());
        let h_dev = groups.iter().find(|g| g.raw_hash == "h-dev").unwrap();
        assert!(
            h_dev.records.iter().any(|r| r.instance == "qa"),
            "non-matching sibling must stay visible"
        );
    }

    #[test]
    fn count_duplicate_raw_hashes_by_source_multi_instance_or_respects_same_filter() {
        let store = Store::open_in_memory().unwrap();
        seed_with_raw_hash_instance(&store, "mem0", "p1", "prod", "h-prod");
        seed_with_raw_hash_instance(&store, "mem0", "p2", "prod", "h-prod");
        seed_with_raw_hash_instance(&store, "mem0", "d1", "dev", "h-dev");
        seed_with_raw_hash_instance(&store, "mem0", "q1", "qa", "h-dev");
        seed_with_raw_hash_instance(&store, "mem0", "q2", "qa", "h-qa");
        seed_with_raw_hash_instance(&store, "mem0", "q3", "qa", "h-qa");

        let counts = store
            .count_duplicate_raw_hashes_by_source(&DuplicateRawHashFilter {
                source: Some("mem0".into()),
                instance: Some("prod,dev".into()),
                limit: 1,
            })
            .unwrap();
        assert_eq!(counts.total_groups, 2);
        assert_eq!(counts.total_records, 4);
        assert!(
            counts
                .by_source
                .iter()
                .any(|b| b.instance == "qa" && b.duplicate_record_count == 1),
            "qa sibling from h-dev remains in whole-group counts: {:?}",
            counts.by_source
        );
    }

    /// Empty filter must behave identically to the legacy
    /// unfiltered API — backward compatibility guarantee.
    #[test]
    fn list_duplicate_raw_hashes_filtered_empty_filter_matches_legacy() {
        let store = Store::open_in_memory().unwrap();
        let _a = seed_with_raw_hash(&store, "claude-code", "a", "h-shared");
        let _b = seed_with_raw_hash(&store, "mem0", "b", "h-shared");

        let legacy = store.list_duplicate_raw_hashes(20).unwrap();
        let filtered = store
            .list_duplicate_raw_hashes_filtered(&DuplicateRawHashFilter::default())
            .unwrap();
        assert_eq!(legacy.len(), filtered.len());
        assert_eq!(legacy[0].raw_hash, filtered[0].raw_hash);
        assert_eq!(legacy[0].records.len(), filtered[0].records.len());
    }

    // ─── Round-97 PR-78s: count_duplicate_raw_hashes_by_source ─────

    /// Aggregate counts reflect the **full** filter-scoped
    /// duplicate set, not the current page. `limit=1` doesn't
    /// truncate `total_groups`/`total_records` — that's the whole
    /// point of `--include-counts`.
    #[test]
    fn count_duplicate_raw_hashes_aggregates_full_set_ignoring_limit() {
        let store = Store::open_in_memory().unwrap();
        // 3 duplicate groups, 2 records each.
        for name in ["g1-a", "g1-b"] {
            seed_with_raw_hash(&store, "claude-code", name, "h-g1");
        }
        for name in ["g2-a", "g2-b"] {
            seed_with_raw_hash(&store, "claude-code", name, "h-g2");
        }
        for name in ["g3-a", "g3-b"] {
            seed_with_raw_hash(&store, "mem0", name, "h-g3");
        }

        // limit=1 truncates the *rows*, but counts must see all
        // 3 groups.
        let counts = store
            .count_duplicate_raw_hashes_by_source(&DuplicateRawHashFilter {
                source: None,
                instance: None,
                limit: 1,
            })
            .unwrap();
        assert_eq!(counts.total_groups, 3);
        assert_eq!(counts.total_records, 6);
        // by_source counts records, not groups: claude-code = 4
        // (two groups of 2), mem0 = 2.
        let cc = counts
            .by_source
            .iter()
            .find(|b| b.adapter == "claude-code")
            .unwrap();
        assert_eq!(cc.duplicate_record_count, 4);
        let mem = counts
            .by_source
            .iter()
            .find(|b| b.adapter == "mem0")
            .unwrap();
        assert_eq!(mem.duplicate_record_count, 2);
    }

    /// **Load-bearing**: mixed-source groups contribute records
    /// to *every* source they touch — `by_source` cannot
    /// double-count by treating a mixed group as belonging to
    /// each member adapter. The count must be record-level.
    #[test]
    fn count_duplicate_raw_hashes_by_source_counts_records_not_group_memberships() {
        let store = Store::open_in_memory().unwrap();
        // Mixed-source group: 1 mem0 + 1 claude-code on the
        // same hash.
        seed_with_raw_hash(&store, "mem0", "m", "h-mixed");
        seed_with_raw_hash(&store, "claude-code", "c", "h-mixed");

        let counts = store
            .count_duplicate_raw_hashes_by_source(&DuplicateRawHashFilter::default())
            .unwrap();
        assert_eq!(counts.total_groups, 1);
        assert_eq!(counts.total_records, 2);
        let mem = counts
            .by_source
            .iter()
            .find(|b| b.adapter == "mem0")
            .unwrap();
        let cc = counts
            .by_source
            .iter()
            .find(|b| b.adapter == "claude-code")
            .unwrap();
        assert_eq!(mem.duplicate_record_count, 1);
        assert_eq!(cc.duplicate_record_count, 1);
        // Records, not groups: sum across by_source equals
        // total_records (2), NOT total_records × adapter_count.
        let by_source_sum: u64 = counts
            .by_source
            .iter()
            .map(|b| b.duplicate_record_count)
            .sum();
        assert_eq!(by_source_sum, counts.total_records);
    }

    /// `source` filter narrows eligibility before counting: a
    /// group has to contain ≥1 record from the named adapter to
    /// be counted. But once eligible, the whole sibling set
    /// contributes to `total_records` and `by_source[]` — same
    /// semantic as `list_duplicate_raw_hashes_filtered`.
    #[test]
    fn count_duplicate_raw_hashes_by_source_respects_filter() {
        let store = Store::open_in_memory().unwrap();
        // h-mixed: 1 mem0 + 1 claude-code (filter-matched on `mem0`).
        seed_with_raw_hash(&store, "mem0", "m", "h-mixed");
        seed_with_raw_hash(&store, "claude-code", "c", "h-mixed");
        // h-cc-only: 2 claude-code records (NOT matched on `mem0`).
        seed_with_raw_hash(&store, "claude-code", "cc1", "h-cc");
        seed_with_raw_hash(&store, "claude-code", "cc2", "h-cc");

        let counts = store
            .count_duplicate_raw_hashes_by_source(&DuplicateRawHashFilter {
                source: Some("mem0".into()),
                instance: None,
                limit: 0,
            })
            .unwrap();
        assert_eq!(counts.total_groups, 1, "only h-mixed has a mem0 member");
        assert_eq!(
            counts.total_records, 2,
            "the whole sibling set counts toward total_records"
        );
        // by_source surfaces both adapters in the matched
        // group, including the non-mem0 sibling — that's the
        // operator-visible "where does this duplicate set live."
        assert_eq!(counts.by_source.len(), 2);
    }

    /// No duplicates → empty counts, not an error.
    #[test]
    fn count_duplicate_raw_hashes_by_source_returns_empty_when_no_duplicates() {
        let store = Store::open_in_memory().unwrap();
        seed_with_raw_hash(&store, "claude-code", "solo", "h-singleton");
        let counts = store
            .count_duplicate_raw_hashes_by_source(&DuplicateRawHashFilter::default())
            .unwrap();
        assert_eq!(counts.total_groups, 0);
        assert_eq!(counts.total_records, 0);
        assert!(counts.by_source.is_empty());
    }

    // ─── Round-78: user_record_tags ────────────────────────────────

    fn seed_one(store: &Store, adapter: &str, native: &str) -> RecordId {
        let r = make_record(adapter, native, "content", Kind::Fact);
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        r.id
    }

    #[test]
    fn tag_record_add_then_remove_is_set_semantic() {
        let store = Store::open_in_memory().unwrap();
        let id = seed_one(&store, "claude-code", "rec-a");

        let m = store
            .tag_record(&id, &["todo".into(), "keep".into()], UserTagOperation::Add)
            .unwrap();
        assert_eq!(m.changed, 2);
        assert_eq!(m.user_tags, vec!["keep".to_string(), "todo".to_string()]);

        // Add again — same set, no-op.
        let m = store
            .tag_record(&id, &["todo".into()], UserTagOperation::Add)
            .unwrap();
        assert_eq!(m.changed, 0, "re-add must be a no-op");
        assert_eq!(m.user_tags, vec!["keep".to_string(), "todo".to_string()]);

        // Remove one — set shrinks.
        let m = store
            .tag_record(&id, &["todo".into()], UserTagOperation::Remove)
            .unwrap();
        assert_eq!(m.changed, 1);
        assert_eq!(m.user_tags, vec!["keep".to_string()]);

        // Remove missing — no-op.
        let m = store
            .tag_record(&id, &["nonexistent".into()], UserTagOperation::Remove)
            .unwrap();
        assert_eq!(m.changed, 0);
        assert_eq!(m.user_tags, vec!["keep".to_string()]);
    }

    #[test]
    fn tag_record_normalises_input() {
        let store = Store::open_in_memory().unwrap();
        let id = seed_one(&store, "claude-code", "rec-norm");
        let m = store
            .tag_record(
                &id,
                &[
                    "  TODO  ".into(),
                    "todo".into(),
                    "Keep".into(),
                    "keep".into(),
                ],
                UserTagOperation::Add,
            )
            .unwrap();
        // Trimmed + lowercased + deduped → exactly 2 unique tags.
        assert_eq!(m.requested, vec!["todo".to_string(), "keep".to_string()]);
        assert_eq!(m.changed, 2);
        assert_eq!(m.user_tags, vec!["keep".to_string(), "todo".to_string()]);
    }

    #[test]
    fn tag_record_rejects_empty_and_oversized() {
        let store = Store::open_in_memory().unwrap();
        let id = seed_one(&store, "claude-code", "rec-bad");

        assert!(store
            .tag_record(&id, &["   ".into()], UserTagOperation::Add)
            .is_err());
        assert!(store
            .tag_record(
                &id,
                &["x".repeat(USER_TAG_MAX_LEN + 1)],
                UserTagOperation::Add
            )
            .is_err());
        assert!(store
            .tag_record(&id, &["bad\nnewline".into()], UserTagOperation::Add)
            .is_err());
        // Over-batch cap.
        let many: Vec<String> = (0..(TAG_RECORD_MAX_BATCH + 1))
            .map(|i| format!("t{i}"))
            .collect();
        assert!(store.tag_record(&id, &many, UserTagOperation::Add).is_err());
    }

    // ─── Round-81 PR-78c: tag_record Replace ────────────────────────

    /// `Replace` overwrites the prior set: anything not in the
    /// input disappears, anything new is inserted. `changed` is
    /// the set delta (1 added + 1 removed = 2, not the 3-row
    /// physical delete+insert count).
    #[test]
    fn tag_record_replace_overwrites_prior_set_with_delta_count() {
        let store = Store::open_in_memory().unwrap();
        let id = seed_one(&store, "claude-code", "rec-rep");

        // Prime with {keep, todo}.
        store
            .tag_record(&id, &["keep".into(), "todo".into()], UserTagOperation::Add)
            .unwrap();

        // Replace with {keep, final} — `todo` goes, `final` added.
        let m = store
            .tag_record(
                &id,
                &["keep".into(), "final".into()],
                UserTagOperation::Replace,
            )
            .unwrap();
        assert_eq!(m.operation, UserTagOperation::Replace);
        // Set delta: 1 added (final) + 1 removed (todo) = 2.
        assert_eq!(
            m.changed, 2,
            "changed must be set delta, not raw delete+insert rows"
        );
        assert_eq!(
            m.user_tags,
            vec!["final".to_string(), "keep".to_string()],
            "post-call set must equal the requested set"
        );
    }

    /// Replacing with the same set is a no-op — `changed = 0`
    /// even though every row was physically rewritten internally.
    /// This is the property that makes Replace safely re-runnable
    /// (e.g. from an idempotent migration script).
    #[test]
    fn tag_record_replace_idempotent_when_set_unchanged() {
        let store = Store::open_in_memory().unwrap();
        let id = seed_one(&store, "claude-code", "rec-idem");

        store
            .tag_record(&id, &["a".into(), "b".into()], UserTagOperation::Add)
            .unwrap();

        // Same set, in different order — set delta is empty.
        let m = store
            .tag_record(&id, &["b".into(), "a".into()], UserTagOperation::Replace)
            .unwrap();
        assert_eq!(m.changed, 0, "same-set replace must report 0 changes");
        assert_eq!(m.user_tags, vec!["a".to_string(), "b".to_string()]);
    }

    /// Empty `Replace` clears the overlay. This is the only
    /// path through `tag_record` that accepts an empty tag list —
    /// `Add`/`Remove` still reject it because they would be
    /// no-ops without input.
    #[test]
    fn tag_record_replace_empty_clears_all_user_tags() {
        let store = Store::open_in_memory().unwrap();
        let id = seed_one(&store, "claude-code", "rec-clear");

        store
            .tag_record(
                &id,
                &["one".into(), "two".into(), "three".into()],
                UserTagOperation::Add,
            )
            .unwrap();
        assert_eq!(store.user_tags(&id).unwrap().len(), 3);

        let m = store
            .tag_record(&id, &[], UserTagOperation::Replace)
            .unwrap();
        assert_eq!(m.changed, 3, "clearing 3 tags = 3 deletions");
        assert!(m.requested.is_empty());
        assert!(m.user_tags.is_empty());
        // And the table really is empty for this record.
        assert!(store.user_tags(&id).unwrap().is_empty());
    }

    /// Empty `Add`/`Remove` still error — only `Replace` carries
    /// the "explicit clear" intent. Guards against a CLI/MCP
    /// regression where someone wires an empty list to Add and
    /// silently no-ops.
    #[test]
    fn tag_record_add_or_remove_empty_still_errors() {
        let store = Store::open_in_memory().unwrap();
        let id = seed_one(&store, "claude-code", "rec-empty");
        assert!(store.tag_record(&id, &[], UserTagOperation::Add).is_err());
        assert!(store
            .tag_record(&id, &[], UserTagOperation::Remove)
            .is_err());
    }

    /// Replace still respects the per-call cap and per-tag
    /// validation — sets >32 tags or contains malformed tags
    /// must fail before any write happens.
    #[test]
    fn tag_record_replace_respects_cap_and_validation() {
        let store = Store::open_in_memory().unwrap();
        let id = seed_one(&store, "claude-code", "rec-cap");

        // Seed something so we can prove the failed Replace
        // didn't touch it.
        store
            .tag_record(&id, &["pre".into()], UserTagOperation::Add)
            .unwrap();

        let many: Vec<String> = (0..(TAG_RECORD_MAX_BATCH + 1))
            .map(|i| format!("t{i}"))
            .collect();
        assert!(store
            .tag_record(&id, &many, UserTagOperation::Replace)
            .is_err());
        assert!(store
            .tag_record(&id, &["bad\nnewline".into()], UserTagOperation::Replace)
            .is_err());
        // Pre-existing tag survives the failed Replace.
        assert_eq!(store.user_tags(&id).unwrap(), vec!["pre".to_string()]);
    }

    /// **Load-bearing**: this is the test that justifies a
    /// separate overlay table. Re-importing the same source
    /// (raw_hash unchanged) must NOT erase user tags.
    #[test]
    fn user_tags_survive_raw_hash_equal_reimport() {
        let store = Store::open_in_memory().unwrap();
        let mut r = make_record("claude-code", "rec-x", "stable content", Kind::Fact);
        r.provenance.raw_hash = "h-stable".into();
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        store
            .tag_record(&r.id, &["keep-me".into()], UserTagOperation::Add)
            .unwrap();

        // Re-import identical content (raw_hash unchanged → fast path).
        let (recs, _) = store.upsert_record(&r, &chunks, None).unwrap();
        assert_eq!(recs, 0, "raw_hash-equal re-upsert must be a no-op");
        let tags = store.user_tags(&r.id).unwrap();
        assert_eq!(
            tags,
            vec!["keep-me".to_string()],
            "user tags must survive re-import",
        );
    }

    /// Also load-bearing: when raw_hash changes (real content
    /// drift), the records row is rewritten — but user_tags
    /// hangs off `record_id`, which is stable across re-import
    /// because it's derived from `(adapter, instance, native_id)`.
    /// So tags survive that path too.
    #[test]
    fn user_tags_survive_raw_hash_changed_reimport() {
        let store = Store::open_in_memory().unwrap();
        let mut r = make_record("claude-code", "rec-y", "first content", Kind::Fact);
        r.provenance.raw_hash = "h-first".into();
        let c1 = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &c1, None).unwrap();
        store
            .tag_record(&r.id, &["keep-me".into()], UserTagOperation::Add)
            .unwrap();

        // Source content drifted; same natural key, new raw_hash.
        let mut r2 = make_record("claude-code", "rec-y", "second content", Kind::Fact);
        r2.provenance.raw_hash = "h-second".into();
        let c2 = Chunker::default().chunk(&r2.id, &r2.content);
        store.upsert_record(&r2, &c2, None).unwrap();
        assert_eq!(
            store.user_tags(&r2.id).unwrap(),
            vec!["keep-me".to_string()],
            "user tags must survive content drift; record_id is stable",
        );
    }

    /// `forget_record` deletes the live `records` row; FK
    /// cascade removes user_tags rows tied to that record.
    /// Documented behaviour — the user can't tag a memory that
    /// doesn't exist anymore.
    #[test]
    fn forget_record_cascades_user_tags() {
        let store = Store::open_in_memory().unwrap();
        let id = seed_one(&store, "claude-code", "rec-cas");
        store
            .tag_record(&id, &["doomed".into()], UserTagOperation::Add)
            .unwrap();
        assert_eq!(store.user_tags(&id).unwrap().len(), 1);
        store.forget_record(&id, None).unwrap();
        let n: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM user_record_tags", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0, "FK cascade must clear user tags");
    }

    #[test]
    fn tag_record_errors_on_unknown_record_id() {
        let store = Store::open_in_memory().unwrap();
        let phantom = RecordId::from_parts("claude-code", None, "never-existed");
        let r = store.tag_record(&phantom, &["x".into()], UserTagOperation::Add);
        assert!(r.is_err());
    }

    #[test]
    fn user_tags_by_ids_batches_overlay_lookup() {
        let store = Store::open_in_memory().unwrap();
        let a = seed_one(&store, "claude-code", "rec-batch-a");
        let b = seed_one(&store, "claude-code", "rec-batch-b");
        let c = seed_one(&store, "claude-code", "rec-batch-c");
        store
            .tag_record(
                &a,
                &["alpha".into(), "shared".into()],
                UserTagOperation::Add,
            )
            .unwrap();
        store
            .tag_record(&b, &["shared".into()], UserTagOperation::Add)
            .unwrap();
        // c stays untagged.

        let map = store
            .user_tags_by_ids(&[a.clone(), b.clone(), c.clone()])
            .unwrap();
        assert_eq!(map.len(), 2, "untagged records are absent from the map");
        assert_eq!(
            map.get(&a).unwrap(),
            &vec!["alpha".to_string(), "shared".to_string()]
        );
        assert_eq!(map.get(&b).unwrap(), &vec!["shared".to_string()]);
    }

    /// `get_record_headers_by_ids` (the hot path search packer uses)
    /// must surface user_tags in a single batched query — no per-id
    /// follow-up.
    #[test]
    fn get_record_headers_by_ids_includes_user_tags() {
        let store = Store::open_in_memory().unwrap();
        let id = seed_one(&store, "claude-code", "rec-hdr");
        store
            .tag_record(&id, &["one".into(), "two".into()], UserTagOperation::Add)
            .unwrap();
        let heads = store
            .get_record_headers_by_ids(std::slice::from_ref(&id))
            .unwrap();
        let h = heads.get(&id).expect("present");
        assert_eq!(h.user_tags, vec!["one".to_string(), "two".to_string()]);
    }

    // ─── Round-79 PR-78b: --user-tag search filter pushdown ─────────

    /// FTS path: a minority-tagged record under a majority of
    /// untagged records still surfaces — the user_tag JOIN
    /// shrinks the candidate pool *before* `LIMIT`, not after.
    /// Mirrors R65's PR-C minority-dominance contract.
    #[test]
    fn user_tag_filter_fts_returns_tagged_minority_under_dominance() {
        let store = Store::open_in_memory().unwrap();
        // 12 untagged claude-code records with the shared marker.
        for i in 0..12 {
            let r = make_record(
                "claude-code",
                &format!("cc-{i}"),
                "alpha shared marker content",
                Kind::Fact,
            );
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &chunks, None).unwrap();
        }
        // 1 tagged mem0 record carrying the same marker.
        let m = make_record("mem0", "m-0", "alpha shared marker content", Kind::Fact);
        let chunks = Chunker::default().chunk(&m.id, &m.content);
        store.upsert_record(&m, &chunks, None).unwrap();
        store
            .tag_record(&m.id, &["keep-forever".into()], UserTagOperation::Add)
            .unwrap();

        let filter = SearchFilter {
            user_tag: Some("keep-forever".into()),
            ..Default::default()
        };
        // limit=1: a post-filter implementation would return 0 hits
        // (the BM25 top-1 is an untagged claude-code chunk, then
        // filtered out). Pushdown returns the tagged mem0 record.
        let hits = store.search_chunks_fts("alpha", &filter, 1).unwrap();
        assert_eq!(hits.len(), 1, "tagged minority must survive limit=1");
        assert_eq!(hits[0].record_id.0, m.id.0);
    }

    /// BLOB vector fallback: same minority-dominance guarantee,
    /// this time through the no-vec0 path (`search_chunks_vec`
    /// falls back to `search_chunks_vec_blob_scan` when no per-
    /// dim vec0 table exists for the query's dim). The vec
    /// fallback uses `append_filter_predicates`, same helper FTS
    /// uses, so this proves the user_tag JOIN is wired into
    /// that path too.
    #[test]
    fn user_tag_filter_blob_vec_fallback_returns_tagged_minority() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("model-a").unwrap();
        // 6 untagged + 1 tagged, all sharing a query-favourable vec.
        let mut tagged: Option<RecordId> = None;
        for i in 0..7 {
            let r = make_record(
                "claude-code",
                &format!("cc-{i}"),
                &format!("content {i}"),
                Kind::Fact,
            );
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &chunks, None).unwrap();
            if i == 6 {
                tagged = Some(r.id.clone());
            }
        }
        // Drive embeddings under a brand-new dim (777) that no
        // vec0 table exists for — forces blob fallback.
        let jobs = store.claim_next_jobs("model-a", 32).unwrap();
        // Tagged record gets the strongest cos vs query; everyone
        // else gets the same weaker vector, so under a normal
        // (no-filter) search the tagged record would win anyway.
        // The filter test value is in `limit=1` with --user-tag
        // when several records ALSO tag-match — we want the
        // pushdown, not just luck.
        let vecs: Vec<Vec<f32>> = jobs
            .iter()
            .map(|j| {
                if j.chunk_id.starts_with(&tagged.as_ref().unwrap().0) {
                    vec![1.0; 777]
                } else {
                    vec![0.5; 777]
                }
            })
            .collect();
        store.complete_jobs_batch(&jobs, &vecs).unwrap();

        // Manually drop the vec0 table that R67 creates so we
        // force the BLOB fallback path. The blob scan still
        // honours filters via append_filter_predicates.
        store
            .conn()
            .execute("DROP TABLE IF EXISTS chunk_embeddings_vec_d777", [])
            .unwrap();
        store
            .conn()
            .execute("DELETE FROM chunk_vec_indexes WHERE dim = 777", [])
            .unwrap();

        // Tag both the strong-vec record and one weak-vec record.
        store
            .tag_record(
                tagged.as_ref().unwrap(),
                &["keep-forever".into()],
                UserTagOperation::Add,
            )
            .unwrap();
        let weak_id = RecordId::from_parts("claude-code", None, "cc-2");
        store
            .tag_record(&weak_id, &["keep-forever".into()], UserTagOperation::Add)
            .unwrap();

        let filter = SearchFilter {
            user_tag: Some("keep-forever".into()),
            ..Default::default()
        };
        let query = vec![1.0_f32; 777];
        let hits = store
            .search_chunks_vec(&query, "model-a", &filter, 1)
            .unwrap();
        assert_eq!(hits.len(), 1);
        // Either tagged record is acceptable; the contract is
        // "only tagged records survive."
        let allowed: std::collections::HashSet<&str> =
            [tagged.as_ref().unwrap().0.as_str(), weak_id.0.as_str()]
                .into_iter()
                .collect();
        assert!(
            allowed.contains(hits[0].record_id.0.as_str()),
            "result must be one of the tagged records; got {}",
            hits[0].record_id.0
        );
    }

    /// sqlite-vec path: the filter pushes down inside the
    /// MATERIALIZED knn CTE via the new `record_id` metadata
    /// column. Test forces the untagged record to be the
    /// strongest vector match so that without the filter
    /// `limit=1` would never return the tagged one.
    #[test]
    fn user_tag_filter_sqlite_vec_path_returns_tagged_minority() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("model-a").unwrap();
        let strong_untagged = make_record("claude-code", "strong", "alpha", Kind::Fact);
        let weak_tagged = make_record("claude-code", "weak", "alpha", Kind::Fact);
        for r in [&strong_untagged, &weak_tagged] {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }
        // Strong-untagged gets [1,1,0,0]; weak-tagged gets
        // [0.5,0.5,0,0]. Query [1,1,0,0] makes strong-untagged
        // the natural top-1.
        let jobs = store.claim_next_jobs("model-a", 16).unwrap();
        let vecs: Vec<Vec<f32>> = jobs
            .iter()
            .map(|j| {
                if j.chunk_id.starts_with(&strong_untagged.id.0) {
                    vec![1.0, 1.0, 0.0, 0.0]
                } else {
                    vec![0.5, 0.5, 0.0, 0.0]
                }
            })
            .collect();
        store.complete_jobs_batch(&jobs, &vecs).unwrap();
        store
            .tag_record(&weak_tagged.id, &["keep".into()], UserTagOperation::Add)
            .unwrap();

        let filter = SearchFilter {
            user_tag: Some("keep".into()),
            ..Default::default()
        };
        let hits = store
            .search_chunks_vec(&[1.0, 1.0, 0.0, 0.0], "model-a", &filter, 1)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].record_id.0, weak_tagged.id.0,
            "tagged record must win under filter even when untagged vec is stronger",
        );
    }

    /// Normalisation parity: a tag written as `Keep-Forever`
    /// gets stored as `keep-forever` and a filter for
    /// `Keep-Forever` must hit. Without shared normalisation,
    /// the filter would miss.
    #[test]
    fn user_tag_filter_normalises_match_to_write_path() {
        let store = Store::open_in_memory().unwrap();
        let r = make_record("claude-code", "alpha", "alpha content", Kind::Fact);
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        store
            .tag_record(&r.id, &["Keep-Forever".into()], UserTagOperation::Add)
            .unwrap();

        let normalised = normalize_user_tag_name("  KEEP-FOREVER  ").unwrap();
        assert_eq!(normalised, "keep-forever");
        let filter = SearchFilter {
            user_tag: Some(normalised),
            ..Default::default()
        };
        let hits = store.search_chunks_fts("alpha", &filter, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id.0, r.id.0);
    }

    // ─── Round-131 PR-78az: near-duplicate detection ─────────────

    /// End-to-end on the store. mem0 and claude-code both capture
    /// the same operator preference in slightly different wording
    /// (different `raw_hash`); near-dedupe should group them.
    /// An unrelated third record stays out of any group.
    #[test]
    fn near_dedupe_groups_cross_adapter_paraphrases() {
        let store = Store::open_in_memory().unwrap();

        // Make a near-duplicate pair across adapters. Same key
        // tokens (user prefers thorough error handling +
        // integration tests + real fixtures + no mocks), different
        // surface form so raw_hash differs.
        let a = make_record(
            "mem0",
            "rec-a",
            "The user prefers thorough error handling in Rust code and \
             writes comprehensive integration tests with real fixtures \
             and never uses mocks for critical paths.",
            Kind::Fact,
        );
        let b = make_record(
            "claude-code",
            "rec-b",
            "User prefers thorough error handling in Rust code; \
             comprehensive integration tests with real fixtures; \
             no mocks on critical paths.",
            Kind::Fact,
        );
        // Unrelated long record to ensure it doesn't join.
        let c = make_record(
            "codex",
            "rec-c",
            "Configure the database connection pool to recycle stale \
             connections after sixty seconds and enable TLS for all \
             outbound traffic through the corporate proxy.",
            Kind::Fact,
        );
        for r in [&a, &b, &c] {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }

        let groups = crate::semantic_dedupe::list_near_duplicates(
            &store,
            &crate::semantic_dedupe::NearDuplicateFilter::default(),
        )
        .unwrap();
        assert_eq!(
            groups.len(),
            1,
            "must find the mem0+claude-code near-dup group: {groups:?}"
        );
        let group = &groups[0];
        let adapters: std::collections::BTreeSet<&str> =
            group.records.iter().map(|r| r.adapter.as_str()).collect();
        assert_eq!(
            adapters,
            ["claude-code", "mem0"].into_iter().collect(),
            "expected cross-adapter group: {group:?}"
        );
        assert!(
            group.min_similarity >= 0.6,
            "min_similarity must be ≥ Jaccard threshold: {}",
            group.min_similarity
        );

        // Privacy: the group records carry no content / raw_hash.
        // The `NearDuplicateRecord` struct doesn't even have those
        // fields, so this is compile-time enforced; this assertion
        // pins the wire contract going forward.
        let serialised = format!("{:?}", group.records);
        assert!(
            !serialised.contains("comprehensive integration"),
            "content must not leak into group debug repr"
        );
    }

    /// Single-adapter near-dups are filtered by default
    /// (require_cross_source=true) — Anamnesis is about the
    /// cross-adapter interop story.
    #[test]
    fn near_dedupe_default_filters_single_adapter_groups() {
        let store = Store::open_in_memory().unwrap();
        // Two mem0 records with the same content (different
        // native_id so raw_hash differs even before our
        // tokenizer normalisation).
        let a = make_record(
            "mem0",
            "rec-a",
            "User prefers thorough error handling in Rust code with \
             comprehensive integration tests and real fixtures.",
            Kind::Fact,
        );
        let b = make_record(
            "mem0",
            "rec-b",
            "User prefers thorough error handling in Rust code with \
             comprehensive integration tests and real fixtures.",
            Kind::Fact,
        );
        for r in [&a, &b] {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }

        let groups = crate::semantic_dedupe::list_near_duplicates(
            &store,
            &crate::semantic_dedupe::NearDuplicateFilter::default(),
        )
        .unwrap();
        assert!(
            groups.is_empty(),
            "default filter must drop single-adapter near-dup groups: {groups:?}"
        );

        // With cross-source disabled, the same fixture surfaces.
        let groups2 = crate::semantic_dedupe::list_near_duplicates(
            &store,
            &crate::semantic_dedupe::NearDuplicateFilter {
                require_cross_source: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            groups2.len(),
            1,
            "with require_cross_source=false the single-adapter group must surface"
        );
    }

    /// Short records (fewer than MIN_TOKENS) are skipped so
    /// random vocabulary co-occurrence on 2-token records can't
    /// produce false-positive groups.
    #[test]
    fn near_dedupe_skips_short_records() {
        let store = Store::open_in_memory().unwrap();
        let a = make_record("mem0", "short-a", "hello world", Kind::Fact);
        let b = make_record("claude-code", "short-b", "hello world", Kind::Fact);
        for r in [&a, &b] {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }
        let groups = crate::semantic_dedupe::list_near_duplicates(
            &store,
            &crate::semantic_dedupe::NearDuplicateFilter::default(),
        )
        .unwrap();
        assert!(
            groups.is_empty(),
            "short records (< MIN_TOKENS) must be skipped: {groups:?}"
        );
    }

    // ─── Round-133 PR-78bb: forget cascade derived ───────────────

    /// Build a `parent → child → grandchild` derivation chain so
    /// cascade tests have something realistic to walk.
    fn seed_derivation_chain(store: &Store) -> (RecordId, RecordId, RecordId) {
        let parent = make_record(
            "claude-code",
            "ep-parent",
            "Episode root content for derivation chain",
            Kind::Episode,
        );
        let mut child = make_record(
            "extractor",
            "fact-child",
            "Distilled fact derived from the episode",
            Kind::Fact,
        );
        child.provenance.derived_from = Some(parent.id.clone());
        let mut grand = make_record(
            "extractor",
            "fact-grandchild",
            "Second-stage distillation chained off the first fact",
            Kind::Fact,
        );
        grand.provenance.derived_from = Some(child.id.clone());

        for r in [&parent, &child, &grand] {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }
        (parent.id, child.id, grand.id)
    }

    /// Back-compat: default `forget_record_with_options` (cascade
    /// off) leaves derived descendants untouched. Same outcome the
    /// R72 path produced.
    #[test]
    fn forget_with_options_default_leaves_derived_alive() {
        let store = Store::open_in_memory().unwrap();
        let (parent, child, grand) = seed_derivation_chain(&store);

        let outcome = store
            .forget_record_with_options(&parent, None, &ForgetCascadeOptions::default())
            .unwrap();
        assert!(matches!(outcome.root, ForgetRecordOutcome::Forgotten(_)));
        assert!(outcome.derived.is_empty());

        // Parent gone, derived still live.
        assert!(store.get_record(&parent).unwrap().is_none());
        assert!(store.get_record(&child).unwrap().is_some());
        assert!(store.get_record(&grand).unwrap().is_some());
    }

    /// Cascade forget tombstones every descendant. After it returns,
    /// none of the records can be fetched and each derived row has
    /// a tombstone.
    #[test]
    fn forget_with_options_cascade_tombstones_full_chain() {
        let store = Store::open_in_memory().unwrap();
        let (parent, child, grand) = seed_derivation_chain(&store);

        let outcome = store
            .forget_record_with_options(
                &parent,
                Some("operator decided"),
                &ForgetCascadeOptions {
                    cascade_derived: true,
                },
            )
            .unwrap();
        assert!(matches!(outcome.root, ForgetRecordOutcome::Forgotten(_)));
        assert_eq!(outcome.derived.len(), 2, "child + grandchild: {outcome:?}");
        for d in &outcome.derived {
            assert!(!d.was_already_forgotten);
            assert!(d.forgotten_at > 0);
        }
        let derived_ids: std::collections::BTreeSet<&str> = outcome
            .derived
            .iter()
            .map(|d| d.record_id.0.as_str())
            .collect();
        assert!(derived_ids.contains(child.0.as_str()));
        assert!(derived_ids.contains(grand.0.as_str()));

        // All three live records are gone; all three have tombstones.
        assert!(store.get_record(&parent).unwrap().is_none());
        assert!(store.get_record(&child).unwrap().is_none());
        assert!(store.get_record(&grand).unwrap().is_none());
        for id in [&parent, &child, &grand] {
            let n: i64 = store
                .conn()
                .query_row(
                    "SELECT COUNT(*) FROM record_tombstones WHERE record_id = ?1",
                    params![id.0],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(n, 1, "tombstone must exist for {}", id.0);
        }
    }

    /// Cascade with the root already tombstoned still cleans the
    /// derived children. Matters because an operator may have
    /// already run a non-cascade forget on the parent and then
    /// realise the children also need to go.
    #[test]
    fn forget_with_options_cascade_when_root_already_forgotten() {
        let store = Store::open_in_memory().unwrap();
        let (parent, child, _grand) = seed_derivation_chain(&store);

        // Pre-tombstone parent only.
        store.forget_record(&parent, Some("first round")).unwrap();
        // Now cascade — parent should report AlreadyForgotten but
        // derived must still be cleaned (they were live).
        let outcome = store
            .forget_record_with_options(
                &parent,
                None,
                &ForgetCascadeOptions {
                    cascade_derived: true,
                },
            )
            .unwrap();
        assert!(matches!(
            outcome.root,
            ForgetRecordOutcome::AlreadyForgotten(_)
        ));
        // Derived rows were never tombstoned before this call, so
        // they're freshly written here. Number depends on whether
        // BFS still finds the chain through the existing records.
        assert!(
            !outcome.derived.is_empty(),
            "must clean derived children even when root is already forgotten: {outcome:?}"
        );
        // The previously-live child must now be gone.
        assert!(store.get_record(&child).unwrap().is_none());
    }

    /// Dry-run preview reports the same descendants the real cascade
    /// would touch, with `would_delete.records == 1` for live
    /// descendants and `already_forgotten_at = Some(_)` for the
    /// pre-tombstoned ones. Must NOT mutate the store.
    #[test]
    fn preview_forget_with_options_cascade_does_not_mutate() {
        let store = Store::open_in_memory().unwrap();
        let (parent, child, grand) = seed_derivation_chain(&store);

        let preview = store
            .preview_forget_record_with_options(
                &parent,
                Some("dry"),
                &ForgetCascadeOptions {
                    cascade_derived: true,
                },
            )
            .unwrap();
        assert!(matches!(
            preview.root,
            ForgetRecordPreview::WouldForget { .. }
        ));
        assert_eq!(preview.derived.len(), 2);
        for d in &preview.derived {
            assert!(d.already_forgotten_at.is_none());
            assert_eq!(d.would_delete.records, 1);
        }
        // Live records unchanged.
        assert!(store.get_record(&parent).unwrap().is_some());
        assert!(store.get_record(&child).unwrap().is_some());
        assert!(store.get_record(&grand).unwrap().is_some());
        // No tombstones written.
        let n: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM record_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0, "preview must not write tombstones");
    }

    /// Root not found + cascade flag still returns NotFound without
    /// panicking. Edge case but matters for scripted callers.
    #[test]
    fn forget_with_options_cascade_not_found_for_phantom_id() {
        let store = Store::open_in_memory().unwrap();
        let phantom = RecordId::from_parts("claude-code", None, "never");
        let outcome = store
            .forget_record_with_options(
                &phantom,
                None,
                &ForgetCascadeOptions {
                    cascade_derived: true,
                },
            )
            .unwrap();
        assert!(matches!(outcome.root, ForgetRecordOutcome::NotFound));
        assert!(outcome.derived.is_empty());
    }

    // ─── Round-134 PR-78bc: unforget cascade derived ─────────────

    /// Forget cascade now persists `derived_from` on the tombstone
    /// (migration 0011). Cross-check: after cascade forget, every
    /// derived tombstone's `derived_from` matches its live parent.
    #[test]
    fn cascade_forget_persists_derived_from_on_tombstones() {
        let store = Store::open_in_memory().unwrap();
        let (parent, child, grand) = seed_derivation_chain(&store);

        store
            .forget_record_with_options(
                &parent,
                None,
                &ForgetCascadeOptions {
                    cascade_derived: true,
                },
            )
            .unwrap();

        // child.derived_from should point at parent.
        let child_df: Option<String> = store
            .conn()
            .query_row(
                "SELECT derived_from FROM record_tombstones WHERE record_id = ?1",
                params![child.0],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(child_df.as_deref(), Some(parent.0.as_str()));

        // grand.derived_from should point at child.
        let grand_df: Option<String> = store
            .conn()
            .query_row(
                "SELECT derived_from FROM record_tombstones WHERE record_id = ?1",
                params![grand.0],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(grand_df.as_deref(), Some(child.0.as_str()));
    }

    /// Default `unforget_record_with_options` (cascade off) deletes
    /// only the root tombstone — same outcome as the R75 path.
    #[test]
    fn unforget_with_options_default_only_removes_root_tombstone() {
        let store = Store::open_in_memory().unwrap();
        let (parent, child, grand) = seed_derivation_chain(&store);
        store
            .forget_record_with_options(
                &parent,
                None,
                &ForgetCascadeOptions {
                    cascade_derived: true,
                },
            )
            .unwrap();

        let outcome = store
            .unforget_record_with_options(&parent, &UnforgetCascadeOptions::default())
            .unwrap();
        assert!(matches!(
            outcome.root,
            UnforgetRecordOutcome::Unforgotten(_)
        ));
        assert!(outcome.derived.is_empty());

        // Root tombstone gone, child + grand tombstones remain.
        for (id, expected) in [(&parent, 0i64), (&child, 1i64), (&grand, 1i64)] {
            let n: i64 = store
                .conn()
                .query_row(
                    "SELECT COUNT(*) FROM record_tombstones WHERE record_id = ?1",
                    params![id.0],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(n, expected, "tombstone count for {} mismatch", id.0);
        }
    }

    /// Cascade unforget deletes the root and every descendant
    /// tombstone in one shot. Mirrors the R133 cascade-forget proof.
    #[test]
    fn unforget_with_options_cascade_removes_descendant_tombstones() {
        let store = Store::open_in_memory().unwrap();
        let (parent, child, grand) = seed_derivation_chain(&store);
        store
            .forget_record_with_options(
                &parent,
                None,
                &ForgetCascadeOptions {
                    cascade_derived: true,
                },
            )
            .unwrap();

        let outcome = store
            .unforget_record_with_options(
                &parent,
                &UnforgetCascadeOptions {
                    cascade_derived: true,
                },
            )
            .unwrap();
        assert!(matches!(
            outcome.root,
            UnforgetRecordOutcome::Unforgotten(_)
        ));
        assert_eq!(outcome.derived.len(), 2, "child + grandchild");
        let derived_ids: std::collections::BTreeSet<&str> = outcome
            .derived
            .iter()
            .map(|d| d.record_id.0.as_str())
            .collect();
        assert!(derived_ids.contains(child.0.as_str()));
        assert!(derived_ids.contains(grand.0.as_str()));

        // No tombstones remain for any of the three.
        let n: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM record_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0, "all three tombstones must be gone");
    }

    /// Dry-run cascade preview reports descendants without mutating
    /// the tombstone table.
    #[test]
    fn preview_unforget_with_options_cascade_does_not_mutate() {
        let store = Store::open_in_memory().unwrap();
        let (parent, child, grand) = seed_derivation_chain(&store);
        store
            .forget_record_with_options(
                &parent,
                None,
                &ForgetCascadeOptions {
                    cascade_derived: true,
                },
            )
            .unwrap();

        let before: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM record_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(before, 3);

        let preview = store
            .preview_unforget_record_with_options(
                &parent,
                &UnforgetCascadeOptions {
                    cascade_derived: true,
                },
            )
            .unwrap();
        assert!(matches!(
            preview.root,
            UnforgetRecordOutcome::Unforgotten(_)
        ));
        assert_eq!(preview.derived.len(), 2);

        // No deletes happened.
        let after: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM record_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(after, 3, "preview must not delete tombstones");

        // Verify the derived previews include the two descendants.
        let ids: std::collections::BTreeSet<&str> = preview
            .derived
            .iter()
            .map(|d| d.record_id.0.as_str())
            .collect();
        assert!(ids.contains(child.0.as_str()));
        assert!(ids.contains(grand.0.as_str()));
    }

    // ─── Round-135 PR-78bd: list_native_content_conflicts ────────

    /// Plant two adapters each emitting a record at the same
    /// `native_id` but with different content. Plus a control:
    /// same `native_id` but identical content across adapters
    /// (must NOT group).
    fn seed_native_conflicts(store: &Store) {
        // Cross-adapter conflict on `shared-1`: mem0 says A,
        // claude-code says B → must group.
        let mut a = make_record("mem0", "shared-1", "Memory body variant A", Kind::Fact);
        a.provenance.native_id = "shared-1".into();
        let mut b = make_record(
            "claude-code",
            "shared-1",
            "Memory body variant B",
            Kind::Fact,
        );
        b.provenance.native_id = "shared-1".into();
        // Control: `shared-2` has identical content across
        // adapters → must NOT group (no content disagreement).
        let mut c = make_record("mem0", "shared-2", "Same content both sides", Kind::Fact);
        c.provenance.native_id = "shared-2".into();
        let mut d = make_record(
            "claude-code",
            "shared-2",
            "Same content both sides",
            Kind::Fact,
        );
        d.provenance.native_id = "shared-2".into();
        // Control: `unique-3` only one adapter → singleton, drops.
        let mut e = make_record("codex", "unique-3", "Solo record", Kind::Fact);
        e.provenance.native_id = "unique-3".into();

        for r in [&a, &b, &c, &d, &e] {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }
    }

    #[test]
    fn native_conflicts_only_returns_cross_adapter_content_disagreements() {
        let store = Store::open_in_memory().unwrap();
        seed_native_conflicts(&store);

        let groups = store
            .list_native_content_conflicts_filtered(&NativeConflictFilter {
                limit: 20,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            groups.len(),
            1,
            "must drop same-content and singleton groups: {groups:?}"
        );
        let g = &groups[0];
        assert_eq!(g.native_id, "shared-1");
        assert_eq!(g.records.len(), 2);
        assert_eq!(g.content_variant_count, 2);
        let adapters: std::collections::BTreeSet<&str> =
            g.records.iter().map(|r| r.adapter.as_str()).collect();
        assert!(adapters.contains("claude-code"));
        assert!(adapters.contains("mem0"));
        // Records have distinct content variants (1, 2).
        let variants: std::collections::BTreeSet<u32> =
            g.records.iter().map(|r| r.content_variant).collect();
        assert_eq!(variants, [1u32, 2].into_iter().collect());
    }

    #[test]
    fn native_conflicts_default_filter_redacts_content_preview() {
        let store = Store::open_in_memory().unwrap();
        seed_native_conflicts(&store);

        let groups = store
            .list_native_content_conflicts_filtered(&NativeConflictFilter {
                limit: 20,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(groups.len(), 1);
        for r in &groups[0].records {
            assert!(
                r.content_preview.is_none(),
                "default filter must NOT populate content_preview: {r:?}"
            );
        }
    }

    #[test]
    fn native_conflicts_include_content_returns_truncated_preview() {
        let store = Store::open_in_memory().unwrap();
        // Plant a long content so we exercise the truncation path.
        let mut a = make_record(
            "mem0",
            "long-1",
            &"A".repeat(NATIVE_CONFLICT_PREVIEW_CHARS + 64),
            Kind::Fact,
        );
        a.provenance.native_id = "long-1".into();
        let mut b = make_record(
            "claude-code",
            "long-1",
            "completely different body",
            Kind::Fact,
        );
        b.provenance.native_id = "long-1".into();
        for r in [&a, &b] {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }

        let groups = store
            .list_native_content_conflicts_filtered(&NativeConflictFilter {
                limit: 20,
                include_content: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(groups.len(), 1);
        let long_row = groups[0]
            .records
            .iter()
            .find(|r| r.adapter == "mem0")
            .unwrap();
        let preview = long_row.content_preview.as_ref().unwrap();
        // Truncation contract: ≤ MAX chars + 1 ellipsis.
        assert!(preview.chars().count() <= NATIVE_CONFLICT_PREVIEW_CHARS + 1);
        assert!(preview.ends_with('…'), "must end with truncation marker");
        let short_row = groups[0]
            .records
            .iter()
            .find(|r| r.adapter == "claude-code")
            .unwrap();
        let short_preview = short_row.content_preview.as_ref().unwrap();
        // Short content passes through unmodified (no ellipsis).
        assert_eq!(short_preview, "completely different body");
    }

    #[test]
    fn native_conflicts_source_filter_keeps_siblings_whole() {
        let store = Store::open_in_memory().unwrap();
        seed_native_conflicts(&store);

        // Filter on `mem0` — but the group spans mem0 + claude-code
        // and we return the whole sibling set so the operator sees
        // what they'd be choosing between.
        let groups = store
            .list_native_content_conflicts_filtered(&NativeConflictFilter {
                source: Some("mem0".into()),
                limit: 20,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(groups.len(), 1);
        let adapters: std::collections::BTreeSet<&str> = groups[0]
            .records
            .iter()
            .map(|r| r.adapter.as_str())
            .collect();
        assert!(adapters.contains("mem0"));
        assert!(
            adapters.contains("claude-code"),
            "siblings must stay visible under source filter"
        );

        // Filter on a non-matching adapter — group drops entirely.
        let groups = store
            .list_native_content_conflicts_filtered(&NativeConflictFilter {
                source: Some("hermes".into()),
                limit: 20,
                ..Default::default()
            })
            .unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn native_conflicts_empty_store_returns_empty_vec() {
        let store = Store::open_in_memory().unwrap();
        let groups = store
            .list_native_content_conflicts_filtered(&NativeConflictFilter {
                limit: 20,
                ..Default::default()
            })
            .unwrap();
        assert!(groups.is_empty());
    }

    /// Pre-R134 tombstones carry NULL `derived_from`. Synthesise
    /// that case directly via SQL and confirm the cascade still
    /// unforgets the root but reports zero descendants.
    #[test]
    fn unforget_cascade_treats_null_derived_from_as_no_descendants() {
        let store = Store::open_in_memory().unwrap();
        // Plant a tombstone manually with NULL derived_from (legacy
        // shape).
        let id = RecordId::from_parts("claude-code", None, "legacy");
        let now = chrono::Utc::now().timestamp();
        store
            .conn()
            .execute(
                "INSERT INTO record_tombstones( \
                    record_id, adapter, instance, native_id, native_path, \
                    raw_hash, reason, forgotten_at, derived_from) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL)",
                params![
                    id.0,
                    "claude-code",
                    "",
                    "legacy",
                    Option::<String>::None,
                    "raw-legacy",
                    Option::<String>::None,
                    now
                ],
            )
            .unwrap();

        let outcome = store
            .unforget_record_with_options(
                &id,
                &UnforgetCascadeOptions {
                    cascade_derived: true,
                },
            )
            .unwrap();
        assert!(matches!(
            outcome.root,
            UnforgetRecordOutcome::Unforgotten(_)
        ));
        assert!(outcome.derived.is_empty());
        let n: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM record_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    // ─── R144: accept_native_conflict_variant ────────────────────────

    /// Three records share `shared-1`: mem0 + claude-code with content A
    /// (variant 1), and mem0-prod with content B (variant 2). Accepting
    /// variant 1 must keep two records and tombstone one.
    fn seed_three_record_conflict(store: &Store) {
        let a = make_record("claude-code", "shared-1", "Body variant A", Kind::Fact);
        let mut b = make_record("mem0", "shared-1", "Body variant A", Kind::Fact);
        b.id = anamnesis_core::model::RecordId::from_parts("mem0", None, "shared-1");
        let mut c = make_record("mem0", "shared-1", "Body variant B", Kind::Fact);
        c.id = anamnesis_core::model::RecordId::from_parts("mem0", Some("prod"), "shared-1");
        c.source.instance = Some("prod".into());
        for r in [&a, &b, &c] {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }
    }

    #[test]
    fn preview_accept_conflict_partitions_without_mutating() {
        let store = Store::open_in_memory().unwrap();
        seed_three_record_conflict(&store);

        let outcome = store
            .preview_accept_native_conflict_variant(&AcceptConflictOptions {
                native_id: "shared-1".into(),
                selector: AcceptConflictSelector::KeepVariant(1),
                reason: None,
                cascade_derived: false,
            })
            .unwrap();

        assert!(outcome.dry_run);
        assert_eq!(outcome.keep_variant, 1);
        assert_eq!(outcome.keep_records.len(), 2);
        assert_eq!(outcome.forget_records.len(), 1);
        assert_eq!(outcome.forget_records[0].content_variant, 2);

        // Store is unchanged: still 3 live records, 0 tombstones.
        let live: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM records", [], |r| r.get(0))
            .unwrap();
        assert_eq!(live, 3);
        let tomb: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM record_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(tomb, 0);
    }

    #[test]
    fn accept_conflict_tombstones_losers_and_blocks_reimport() {
        let store = Store::open_in_memory().unwrap();
        seed_three_record_conflict(&store);

        let outcome = store
            .accept_native_conflict_variant(&AcceptConflictOptions {
                native_id: "shared-1".into(),
                selector: AcceptConflictSelector::KeepVariant(1),
                reason: Some("operator picked variant 1".into()),
                cascade_derived: false,
            })
            .unwrap();
        assert!(!outcome.dry_run);
        assert_eq!(outcome.keep_records.len(), 2);
        assert_eq!(outcome.forget_records.len(), 1);

        let live: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM records", [], |r| r.get(0))
            .unwrap();
        assert_eq!(live, 2, "loser must be tombstoned");
        let tomb: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM record_tombstones", [], |r| r.get(0))
            .unwrap();
        assert_eq!(tomb, 1);
        // Conflict is resolved.
        let groups = store
            .list_native_content_conflicts_filtered(&NativeConflictFilter {
                limit: 20,
                ..Default::default()
            })
            .unwrap();
        assert!(groups.is_empty(), "conflict gone after accept: {groups:?}");
    }

    #[test]
    fn accept_conflict_keep_record_id_selector_matches_variant() {
        let store = Store::open_in_memory().unwrap();
        seed_three_record_conflict(&store);
        let target = anamnesis_core::model::RecordId::from_parts("mem0", Some("prod"), "shared-1");
        let outcome = store
            .accept_native_conflict_variant(&AcceptConflictOptions {
                native_id: "shared-1".into(),
                selector: AcceptConflictSelector::KeepRecordId(target.clone()),
                reason: None,
                cascade_derived: false,
            })
            .unwrap();
        assert_eq!(outcome.keep_variant, 2);
        let keep_ids: std::collections::BTreeSet<&str> = outcome
            .keep_records
            .iter()
            .map(|r| r.record_id.0.as_str())
            .collect();
        assert!(keep_ids.contains(target.0.as_str()));
    }

    #[test]
    fn accept_conflict_rejects_unknown_native_id() {
        let store = Store::open_in_memory().unwrap();
        let err = store
            .preview_accept_native_conflict_variant(&AcceptConflictOptions {
                native_id: "nope".into(),
                selector: AcceptConflictSelector::KeepVariant(1),
                reason: None,
                cascade_derived: false,
            })
            .unwrap_err();
        assert!(matches!(err, StoreError::Corruption(_)));
    }

    #[test]
    fn accept_conflict_rejects_non_conflict_native_id() {
        // Same content across adapters → not a conflict.
        let store = Store::open_in_memory().unwrap();
        let a = make_record("mem0", "shared-2", "Same body", Kind::Fact);
        let mut b = make_record("claude-code", "shared-2", "Same body", Kind::Fact);
        b.id = anamnesis_core::model::RecordId::from_parts("claude-code", None, "shared-2");
        for r in [&a, &b] {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }
        let err = store
            .preview_accept_native_conflict_variant(&AcceptConflictOptions {
                native_id: "shared-2".into(),
                selector: AcceptConflictSelector::KeepVariant(1),
                reason: None,
                cascade_derived: false,
            })
            .unwrap_err();
        assert!(matches!(err, StoreError::Corruption(_)));
    }

    #[test]
    fn accept_conflict_rejects_out_of_range_variant() {
        let store = Store::open_in_memory().unwrap();
        seed_three_record_conflict(&store);
        let err = store
            .preview_accept_native_conflict_variant(&AcceptConflictOptions {
                native_id: "shared-1".into(),
                selector: AcceptConflictSelector::KeepVariant(99),
                reason: None,
                cascade_derived: false,
            })
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("out of range"), "{msg}");
    }

    #[test]
    fn accept_conflict_rejects_keep_record_id_not_in_group() {
        let store = Store::open_in_memory().unwrap();
        seed_three_record_conflict(&store);
        let stranger = anamnesis_core::model::RecordId::from_parts("codex", None, "stranger");
        let err = store
            .preview_accept_native_conflict_variant(&AcceptConflictOptions {
                native_id: "shared-1".into(),
                selector: AcceptConflictSelector::KeepRecordId(stranger),
                reason: None,
                cascade_derived: false,
            })
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not a member"), "{msg}");
    }

    // ─── R146: reconcile_sources ──────────────────────────────────────

    /// Plant a deterministic 4-bucket fixture:
    ///   * `shared-only-mem0` lives only on mem0
    ///   * `shared-only-letta` lives only on letta
    ///   * `shared-agree` lives on both with identical content
    ///   * `shared-conflict` lives on both with different content
    ///
    /// Records on letta carry `anamnesis_native_id` (round-tripped from mem0).
    fn seed_reconcile_fixture(store: &Store) {
        let mut a = make_record("mem0", "shared-only-mem0", "Only on mem0", Kind::Fact);
        a.provenance.native_id = "shared-only-mem0".into();

        let mut b = make_record("letta", "shared-only-letta", "Only on letta", Kind::Fact);
        b.provenance.native_id = "shared-only-letta".into();

        // Agree: present on both sides via round-trip (letta side carries
        // `anamnesis_native_id` pointing back at mem0's native id).
        let mut agree_mem = make_record("mem0", "shared-agree", "Agreed body", Kind::Fact);
        agree_mem.provenance.native_id = "shared-agree".into();
        let mut agree_letta = make_record("letta", "letta-block-agree", "Agreed body", Kind::Fact);
        agree_letta.provenance.native_id = "letta-block-agree".into();
        agree_letta.metadata.insert(
            "anamnesis_native_id".into(),
            serde_json::json!("shared-agree"),
        );

        // Conflict: same identity on both sides via round-trip, content differs.
        let mut conf_mem = make_record("mem0", "shared-conflict", "Conflict body A", Kind::Fact);
        conf_mem.provenance.native_id = "shared-conflict".into();
        let mut conf_letta = make_record(
            "letta",
            "letta-block-conflict",
            "Conflict body B",
            Kind::Fact,
        );
        conf_letta.provenance.native_id = "letta-block-conflict".into();
        conf_letta.metadata.insert(
            "anamnesis_native_id".into(),
            serde_json::json!("shared-conflict"),
        );

        for r in [&a, &b, &agree_mem, &agree_letta, &conf_mem, &conf_letta] {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }
    }

    #[test]
    fn reconcile_sources_partitions_into_only_left_only_right_both_and_conflicts() {
        let store = Store::open_in_memory().unwrap();
        seed_reconcile_fixture(&store);

        let outcome = store
            .reconcile_sources(&ReconcileOptions {
                left: ReconcileSourceSelector {
                    adapter: "mem0".into(),
                    instance: None,
                },
                right: ReconcileSourceSelector {
                    adapter: "letta".into(),
                    instance: None,
                },
                limit: 10,
                include_identity: false,
            })
            .unwrap();
        assert_eq!(outcome.counts.only_left, 1);
        assert_eq!(outcome.counts.only_right, 1);
        assert_eq!(
            outcome.counts.both, 2,
            "agree + conflict identities present"
        );
        assert_eq!(outcome.counts.conflicts, 1);
        assert_eq!(outcome.counts.left_total, 3);
        assert_eq!(outcome.counts.right_total, 3);
    }

    #[test]
    fn reconcile_sources_include_identity_surfaces_keys_with_source_provenance() {
        let store = Store::open_in_memory().unwrap();
        seed_reconcile_fixture(&store);

        let outcome = store
            .reconcile_sources(&ReconcileOptions {
                left: ReconcileSourceSelector {
                    adapter: "mem0".into(),
                    instance: None,
                },
                right: ReconcileSourceSelector {
                    adapter: "letta".into(),
                    instance: None,
                },
                limit: 10,
                include_identity: true,
            })
            .unwrap();
        // Conflict sample carries the round-tripped key.
        let conf = outcome
            .samples
            .conflicts
            .iter()
            .find(|s| s.identity_key.as_deref() == Some("shared-conflict"))
            .expect("conflict sample present");
        // mem0 side originally typed the native_id — no anamnesis_native_id
        // on the *left* row of a conflict, so identity_source is "native_id".
        assert_eq!(conf.identity_source, "native_id");
        // only_right sample carries `native_id` (no round-trip metadata).
        let only_right = &outcome.samples.only_right[0];
        assert!(only_right.identity_key.is_some());
    }

    #[test]
    fn reconcile_sources_redacts_identity_by_default() {
        let store = Store::open_in_memory().unwrap();
        seed_reconcile_fixture(&store);
        let outcome = store
            .reconcile_sources(&ReconcileOptions {
                left: ReconcileSourceSelector {
                    adapter: "mem0".into(),
                    instance: None,
                },
                right: ReconcileSourceSelector {
                    adapter: "letta".into(),
                    instance: None,
                },
                limit: 10,
                include_identity: false,
            })
            .unwrap();
        for s in outcome
            .samples
            .only_left
            .iter()
            .chain(outcome.samples.only_right.iter())
            .chain(outcome.samples.conflicts.iter())
        {
            assert!(
                s.identity_key.is_none(),
                "include_identity=false must hide identity_key: {s:?}"
            );
        }
    }

    #[test]
    fn reconcile_sources_caps_samples_but_not_counts() {
        let store = Store::open_in_memory().unwrap();
        // Seed many only-left rows so `limit` matters.
        for i in 0..25 {
            let mut r = make_record(
                "mem0",
                &format!("left-{i}"),
                &format!("body {i}"),
                Kind::Fact,
            );
            r.provenance.native_id = format!("left-{i}");
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &chunks, None).unwrap();
        }
        let outcome = store
            .reconcile_sources(&ReconcileOptions {
                left: ReconcileSourceSelector {
                    adapter: "mem0".into(),
                    instance: None,
                },
                right: ReconcileSourceSelector {
                    adapter: "letta".into(),
                    instance: None,
                },
                limit: 5,
                include_identity: false,
            })
            .unwrap();
        assert_eq!(outcome.counts.only_left, 25, "counts ignore the cap");
        assert_eq!(
            outcome.samples.only_left.len(),
            5,
            "samples respect the cap"
        );
    }

    #[test]
    fn reconcile_bucket_ids_returns_only_left_full_ids() {
        let store = Store::open_in_memory().unwrap();
        seed_reconcile_fixture(&store);
        let ids = store
            .reconcile_bucket_ids(
                &ReconcileSourceSelector {
                    adapter: "mem0".into(),
                    instance: None,
                },
                &ReconcileSourceSelector {
                    adapter: "letta".into(),
                    instance: None,
                },
                ReconcileBucket::OnlyLeft,
            )
            .unwrap();
        assert_eq!(ids.len(), 1, "shared-only-mem0");
    }

    #[test]
    fn reconcile_bucket_ids_returns_only_right_full_ids() {
        let store = Store::open_in_memory().unwrap();
        seed_reconcile_fixture(&store);
        let ids = store
            .reconcile_bucket_ids(
                &ReconcileSourceSelector {
                    adapter: "mem0".into(),
                    instance: None,
                },
                &ReconcileSourceSelector {
                    adapter: "letta".into(),
                    instance: None,
                },
                ReconcileBucket::OnlyRight,
            )
            .unwrap();
        assert_eq!(ids.len(), 1, "shared-only-letta");
    }

    #[test]
    fn reconcile_sources_returns_empty_when_both_sides_empty() {
        let store = Store::open_in_memory().unwrap();
        let outcome = store
            .reconcile_sources(&ReconcileOptions {
                left: ReconcileSourceSelector {
                    adapter: "nothing".into(),
                    instance: None,
                },
                right: ReconcileSourceSelector {
                    adapter: "also-nothing".into(),
                    instance: None,
                },
                limit: 10,
                include_identity: false,
            })
            .unwrap();
        assert_eq!(outcome.counts.only_left, 0);
        assert_eq!(outcome.counts.only_right, 0);
        assert_eq!(outcome.counts.both, 0);
        assert_eq!(outcome.counts.conflicts, 0);
        assert!(outcome.samples.only_left.is_empty());
    }
}
