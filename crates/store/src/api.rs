//! Typed read/write API over the SQLite store.
//!
//! Everything that touches the database goes through this module. `Store`
//! itself owns the `Connection`; callers must never write SQL directly.

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

/// Round 77: hard cap on `Store::list_duplicate_raw_hashes` page size.
/// Same reason as `LIST_FORGOTTEN_MAX_LIMIT` (provenance-bearing
/// diagnostic surface — no single operator request should
/// exfiltrate thousands of records at once).
pub const LIST_DUPLICATE_RAW_HASHES_MAX_LIMIT: u32 = 100;

/// Round 78: per-tag length limit. Tags are operator-supplied
/// short labels (`"todo"`, `"keep-forever"`, …); 64 bytes is
/// already generous. Bounds enforced before the SQL write so
/// pathological input never reaches `user_record_tags`.
pub const USER_TAG_MAX_LEN: usize = 64;

/// Round 78: max distinct tags applied in a single `tag_record`
/// call. A user with thousands of memories that genuinely wants
/// to bulk-apply tags should script per-record calls; this cap
/// exists to keep one shell typo from blasting thousands of
/// rows into the overlay table.
pub const TAG_RECORD_MAX_BATCH: usize = 32;

/// Which direction a `tag_record` call is moving the overlay.
/// Distinguished as an enum (not a `bool add: true`) so the
/// CLI / MCP wire stays readable and a future `Toggle` /
/// `Replace` semantic doesn't need a third boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserTagOperation {
    /// Insert each tag; existing rows are a no-op (set semantics).
    Add,
    /// Delete each tag; missing rows are a no-op (set semantics).
    Remove,
}

/// Result of `Store::tag_record`. Carries the resolved tags the
/// call acted on, the count of rows actually changed (so the
/// caller can render "added 2 of 3 — `dog` was already there"),
/// and the post-call set on the record so the CLI / MCP surface
/// doesn't need a second round-trip to show the new state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserTagMutation {
    /// Record the call mutated.
    pub record_id: RecordId,
    /// Which direction the call moved (mirrors the input).
    pub operation: UserTagOperation,
    /// The caller's tag list after normalisation
    /// (`trim().to_lowercase()` + dedup + validation). Order
    /// preserved from input minus collapsed duplicates.
    pub requested: Vec<String>,
    /// How many rows in `user_record_tags` actually changed.
    /// `Add`: new (record_id, tag) pairs. `Remove`: deletions.
    pub changed: u32,
    /// Full post-call set of user tags on this record, sorted
    /// ASCII-ascending. Stable so callers can render it
    /// deterministically.
    pub user_tags: Vec<String>,
}

/// One row inside a duplicate-raw_hash group. Carries the minimum
/// the operator needs to decide which sibling to `forget`:
/// `(adapter, instance, native_id)` for "where did this come
/// from" and `record_id` for the actual forget call.
///
/// `native_path` is kept here because the operator often
/// disambiguates duplicates by file path, but the CLI / MCP wire
/// layers redact it by default — see the `include_sensitive` knob
/// at those surfaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateRawHashRecord {
    /// Hashed `records.id`.
    pub record_id: RecordId,
    /// Adapter id (e.g. `"claude-code"`).
    pub adapter: String,
    /// Instance discriminator — `""` for the default instance.
    pub instance: String,
    /// Native id at the source.
    pub native_id: String,
    /// Native path at the source, when the adapter has one.
    pub native_path: Option<String>,
    /// Record created_at, unix seconds.
    pub created_at: i64,
    /// Record updated_at, unix seconds. `None` when never updated.
    pub updated_at: Option<i64>,
}

/// One group of records sharing a single `raw_hash`. Always size 2+
/// (the SQL `HAVING COUNT(*) > 1` filter strips singletons).
///
/// Round 77 design choice: this is **exact** duplicate detection —
/// only catches records whose source payload hashes byte-for-byte
/// identically. Semantic / near-duplicate detection is a much
/// bigger product decision and explicitly out of scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateRawHashGroup {
    /// The shared raw_hash. CLI / MCP wire surfaces redact this
    /// by default since it can carry signal about the source
    /// payload's structure.
    pub raw_hash: String,
    /// All live records sharing this raw_hash. Length is always
    /// `>= 2`. Ordered newest-first so the operator reads the
    /// most-recent sibling first.
    pub records: Vec<DuplicateRawHashRecord>,
}

/// Round 80 (PR-77 follow-up): filter for
/// `Store::list_duplicate_raw_hashes_filtered`. Scopes the
/// duplicate report to groups whose member records include at
/// least one matching `(adapter, instance)`. Groups are
/// returned **whole** — all siblings stay visible because the
/// operator needs the full set to decide which to `forget`.
///
/// `limit` is clamped to
/// `[1, LIST_DUPLICATE_RAW_HASHES_MAX_LIMIT]` by the store,
/// same as the unfiltered API.
#[derive(Debug, Clone, Default)]
pub struct DuplicateRawHashFilter {
    /// Adapter id (e.g. `"mem0"`). `None` matches any adapter.
    pub source: Option<String>,
    /// Instance discriminator. Only meaningful when `source` is
    /// also set. `None` matches any instance.
    pub instance: Option<String>,
    /// Max groups to return. Clamped by the store.
    pub limit: u32,
}

/// Round 74 (PR-74): filter for `Store::list_forgotten`. Mirrors
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
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT s.adapter, s.instance, s.location, s.config_json, \
                    s.added_at, s.last_import_at, \
                    COUNT(DISTINCT r.id) AS record_count, \
                    COUNT(rc.id)         AS chunk_count \
             FROM sources s \
             LEFT JOIN records r \
                    ON r.adapter = s.adapter AND r.instance = s.instance \
             LEFT JOIN record_chunks rc \
                    ON rc.record_id = r.id \
             GROUP BY s.adapter, s.instance \
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

    /// Forget a record permanently: write a tombstone, delete the
    /// record row (cascading raw_artifacts / record_chunks /
    /// chunk_embeddings / embedding_jobs via FK), and remove any
    /// vec0 rows for the record's chunks (vec0 is a virtual table
    /// without FK cascade — same gotcha [`vec_ext::delete_vec_rows_for_record`]
    /// solves in `write_chunks`).
    ///
    /// Idempotent: a second `forget_record` on the same id returns
    /// [`ForgetRecordOutcome::AlreadyForgotten`] without touching the
    /// store. If neither the record nor a tombstone exists,
    /// returns [`ForgetRecordOutcome::NotFound`] — the caller decides
    /// whether that's user error.
    ///
    /// The tombstone is keyed on `(adapter, instance, native_id)` so
    /// `upsert_record` / `upsert_records_batch` can short-circuit a
    /// re-import before any chunking / embedding work. This is what
    /// makes "forget" mean "stay forgotten" instead of "delete until
    /// the next `anamnesis import`."
    pub fn forget_record(
        &self,
        id: &RecordId,
        reason: Option<&str>,
    ) -> Result<ForgetRecordOutcome> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        // Look up the live record's natural key + raw_hash. If the
        // record is gone, fall back to checking the tombstone table
        // so a repeat-forget remains idempotent.
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

        if live.is_none() {
            // Maybe already forgotten — return the existing tombstone
            // so the caller can render "already forgotten at <when>".
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
            let existing: Option<TombstoneCols> = tx.query_row(
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
            tx.commit()?;
            return Ok(match existing {
                Some((
                    adapter,
                    instance,
                    native_id,
                    native_path,
                    raw_hash,
                    reason,
                    forgotten_at,
                )) => ForgetRecordOutcome::AlreadyForgotten(ForgottenRecord {
                    record_id: id.clone(),
                    adapter,
                    instance,
                    native_id,
                    native_path,
                    raw_hash,
                    reason,
                    forgotten_at,
                }),
                None => ForgetRecordOutcome::NotFound,
            });
        }

        let (adapter, instance, native_id, native_path, raw_hash) = live.unwrap();
        let now = chrono::Utc::now().timestamp();

        // Vec0 has no FK cascade — must drop manually *before* the
        // records row goes (the helper joins through record_chunks).
        crate::vec_ext::delete_vec_rows_for_record(&tx, &id.0)?;

        tx.execute(
            "INSERT INTO record_tombstones( \
                 record_id, adapter, instance, native_id, native_path, \
                 raw_hash, reason, forgotten_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                id.0,
                adapter,
                instance,
                native_id,
                native_path,
                raw_hash,
                reason,
                now
            ],
        )?;

        // The cascade clears raw_artifacts / record_chunks /
        // chunk_embeddings / embedding_jobs — all of those have
        // FK ON DELETE CASCADE to records (or to record_chunks, which
        // itself cascades). Verified at `0002_phase1.sql`.
        tx.execute("DELETE FROM records WHERE id = ?1", params![id.0])?;

        tx.commit()?;

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

    /// Round 74 (PR-74): paginated read of `record_tombstones`,
    /// newest-first. Used by `anamnesis list-forgotten` and the
    /// admin-gated MCP `list_forgotten` tool to surface what's
    /// been tombstoned so an operator can audit + (eventually)
    /// `unforget` mistakes.
    ///
    /// Filters mirror what the importer keys on so the operator can
    /// scope to one source: `(adapter, instance)`. `limit` is
    /// clamped to `[1, LIST_FORGOTTEN_MAX_LIMIT]`. Empty filter
    /// returns the most recent N across all sources.
    ///
    /// This method is read-only — callers are responsible for any
    /// field-level redaction they want at the wire boundary.
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
        let mut sql = String::from(
            "SELECT raw_hash, COUNT(*) AS n, MAX(created_at) AS newest \
             FROM records \
             WHERE raw_hash IN (",
        );
        let mut eligible_params: Vec<rusqlite::types::Value> = Vec::new();
        if filter.source.is_some() || filter.instance.is_some() {
            sql.push_str("SELECT raw_hash FROM records WHERE 1=1");
            if let Some(s) = &filter.source {
                sql.push_str(" AND adapter = ?");
                eligible_params.push(rusqlite::types::Value::Text(s.clone()));
            }
            if let Some(i) = &filter.instance {
                sql.push_str(" AND instance = ?");
                eligible_params.push(rusqlite::types::Value::Text(i.clone()));
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
        let requested = normalize_user_tags(tags)?;
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
}
