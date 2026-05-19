//! Importer pipeline: glue between any `MemoryAdapter` and `Store`.
//!
//! ## Responsibilities
//!
//!   1. Run `adapter.scan()` and consume the resulting `RawRecord` stream.
//!   2. Call `adapter.normalize(raw)` for each item.
//!   3. Run the chunker over each `AnamnesisRecord`.
//!   4. `store.upsert_record(record, chunks, raw_payload_json)` atomically.
//!   5. On per-record errors, write a row to `import_errors` and continue;
//!      a single bad record never aborts a run.
//!
//! The importer is intentionally minimal — it does not own the queue, the
//! embedding workers, or the CLI's UX. Those live elsewhere and consume
//! the `ImportSummary` returned here.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use anamnesis_core::adapter::{MemoryAdapter, ScanOpts};
use anamnesis_core::chunker::Chunker;
use anamnesis_core::{Audit, AuditEntry, RawRecord};
use anamnesis_store::Store;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that abort the whole import (vs per-record errors, which are
/// logged to `import_errors` and counted in the summary).
#[derive(Debug, Error)]
pub enum ImportError {
    /// Underlying store failure.
    #[error("store: {0}")]
    Store(#[from] anamnesis_store::StoreError),
}

/// Per-run summary. Returned by `ImportRunner::run`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportSummary {
    /// Adapter id (matches `SourceDescriptor::adapter`).
    pub adapter: String,
    /// Optional adapter instance.
    pub instance: Option<String>,
    /// Raw records produced by `scan`.
    pub raw_seen: u64,
    /// Records successfully upserted to the store.
    pub records_upserted: u64,
    /// Total chunks written across all upserts.
    pub chunks_written: u64,
    /// Per-record errors (each also logged to `import_errors`).
    pub errors: u64,
}

impl ImportSummary {
    fn empty(adapter: &str, instance: Option<&str>) -> Self {
        Self {
            adapter: adapter.into(),
            instance: instance.map(str::to_owned),
            raw_seen: 0,
            records_upserted: 0,
            chunks_written: 0,
            errors: 0,
        }
    }
}

/// The runner. Owns no state of its own; takes a mutable borrow of the
/// store so the upserts can transact.
pub struct ImportRunner<'a, A: MemoryAdapter> {
    adapter: &'a A,
    chunker: Chunker,
}

impl<'a, A: MemoryAdapter> ImportRunner<'a, A> {
    /// Build a runner with the default chunker config (max=512 tokens).
    pub fn new(adapter: &'a A) -> Self {
        Self {
            adapter,
            chunker: Chunker::default(),
        }
    }

    /// Override the chunker (e.g. tests with tiny budgets).
    pub fn with_chunker(mut self, chunker: Chunker) -> Self {
        self.chunker = chunker;
        self
    }

    /// Drain `adapter.scan(Default::default())` into `store`. Equivalent
    /// to `run_with_opts(store, ScanOpts::default())`; kept as a stable
    /// shortcut for callers that don't care about increments.
    pub async fn run(&self, store: &Store) -> std::result::Result<ImportSummary, ImportError> {
        self.run_with_opts(store, ScanOpts::default()).await
    }

    /// Drain `adapter.scan(opts)` into `store`. Returns the run summary.
    ///
    /// Round-19 (§-1.5 PR-4a): `ScanOpts` is finally honored. `opts.since`
    /// filters records the adapter considers "older than the threshold";
    /// `opts.full` tells the adapter to ignore `opts.since` (it does NOT
    /// disable the store's `raw_hash` fast-path — that's a separate
    /// `--reembed` concern, not an import concern).
    pub async fn run_with_opts(
        &self,
        store: &Store,
        opts: ScanOpts,
    ) -> std::result::Result<ImportSummary, ImportError> {
        let descriptor = self.adapter.descriptor();
        let mut summary = ImportSummary::empty(&descriptor.adapter, descriptor.instance.as_deref());
        let mut stream = self.adapter.scan(opts);

        // Round-62 perf: batch normalized records and hand them to
        // `Store::upsert_records_batch` so the importer pays one
        // `fsync` per batch instead of one per record. Default flush
        // threshold balances throughput against memory pressure for
        // the largest realistic chunk fan-out (codex transcripts have
        // hundreds of chunks per record after PR-H).
        let mut batch: Vec<UpsertItem> = Vec::with_capacity(UPSERT_BATCH);

        while let Some(item) = stream.next().await {
            match item {
                Ok(raw) => {
                    summary.raw_seen += 1;
                    self.process_one(raw, store, &descriptor, &mut summary, &mut batch);
                    if batch.len() >= UPSERT_BATCH {
                        Self::flush_batch(&mut batch, store, &descriptor, &mut summary);
                    }
                }
                Err(e) => {
                    summary.errors += 1;
                    let _ = store.log_import_error(
                        &descriptor.adapter,
                        descriptor.instance.as_deref(),
                        None,
                        None,
                        "scan",
                        &format!("{e}"),
                    );
                    tracing::warn!(
                        adapter = %descriptor.adapter,
                        error = %e,
                        "scan stream yielded error"
                    );
                }
            }
        }
        Self::flush_batch(&mut batch, store, &descriptor, &mut summary);
        Ok(summary)
    }

    fn process_one(
        &self,
        raw: RawRecord,
        store: &Store,
        descriptor: &anamnesis_core::SourceDescriptor,
        summary: &mut ImportSummary,
        batch: &mut Vec<UpsertItem>,
    ) {
        // Preserve the raw payload as JSON for raw_artifacts provenance.
        let raw_payload_json = serde_json::to_string(&raw.payload).ok();
        let native_id = raw.native_id.clone();
        let native_path = raw.native_path.clone();

        let records = match self.adapter.normalize(raw) {
            Ok(rs) => rs,
            Err(e) => {
                summary.errors += 1;
                let _ = store.log_import_error(
                    &descriptor.adapter,
                    descriptor.instance.as_deref(),
                    Some(&native_id),
                    native_path.as_deref(),
                    "normalize",
                    &format!("{e}"),
                );
                tracing::warn!(
                    adapter = %descriptor.adapter,
                    native_id = %native_id,
                    error = %e,
                    "normalize failed; skipping record"
                );
                return;
            }
        };

        for record in records {
            let chunks = self.chunker.chunk(&record.id, &record.content);
            batch.push((record, chunks, raw_payload_json.clone()));
        }
    }

    /// Drain `batch` into `Store::upsert_records_batch`. If the batch
    /// fails (a single bad record poisons the SQLite transaction),
    /// retry each record individually through the per-record path so
    /// one corrupt record doesn't drop the rest of the batch on the
    /// floor. Per-record `upsert_record` keeps the original log-and-
    /// continue behavior.
    fn flush_batch(
        batch: &mut Vec<UpsertItem>,
        store: &Store,
        descriptor: &anamnesis_core::SourceDescriptor,
        summary: &mut ImportSummary,
    ) {
        if batch.is_empty() {
            return;
        }
        match store.upsert_records_batch(batch) {
            Ok((n_records, n_chunks)) => {
                summary.records_upserted += n_records;
                summary.chunks_written += n_chunks;
            }
            Err(batch_err) => {
                tracing::warn!(
                    adapter = %descriptor.adapter,
                    error = %batch_err,
                    batch_size = batch.len(),
                    "batch upsert failed; falling back to per-record path",
                );
                for (record, chunks, raw_payload_json) in batch.iter() {
                    match store.upsert_record(record, chunks, raw_payload_json.as_deref()) {
                        Ok((n_records, n_chunks)) => {
                            summary.records_upserted += n_records;
                            summary.chunks_written += n_chunks;
                        }
                        Err(e) => {
                            summary.errors += 1;
                            let _ = store.log_import_error(
                                &descriptor.adapter,
                                descriptor.instance.as_deref(),
                                Some(&record.provenance.native_id),
                                record.provenance.native_path.as_deref(),
                                "upsert",
                                &format!("{e}"),
                            );
                            tracing::warn!(
                                adapter = %descriptor.adapter,
                                record = %record.id,
                                error = %e,
                                "upsert failed"
                            );
                        }
                    }
                }
            }
        }
        batch.clear();
    }
}

/// One (record, chunks, optional raw-JSON) tuple buffered by the importer
/// for batched upsert. Matches the signature `Store::upsert_records_batch`
/// expects.
type UpsertItem = (
    anamnesis_core::AnamnesisRecord,
    Vec<anamnesis_core::Chunk>,
    Option<String>,
);

/// Records buffered between `Store::upsert_records_batch` flushes. 64 is
/// large enough to amortize SQLite `fsync` cost without holding pathological
/// amounts of normalized content in memory for adapters that emit many
/// chunks per record (codex sessions can be hundreds of chunks each).
const UPSERT_BATCH: usize = 64;

/// Options accepted by `ImportService::import`.
///
/// Round-18 (§-1.5 PR-3) — moved out of CLI / MCP. CLI fills these from
/// argv; MCP fills them from the JSON-RPC `tools/call` arguments. Either
/// caller sees the same source-registry + audit + `last_import_at` side
/// effects.
///
/// Round-19 (§-1.5 PR-4a) — `scan_opts` is now first-class. CLI passes
/// `--since` / `--full` through here so adapters can do incremental
/// imports. `Default::default()` means "full scan" — MCP callers and
/// older test harnesses that don't set it keep their original behavior.
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// Scan-only mode. Counts `raw_seen` but writes nothing to the store,
    /// the source registry, or the audit log.
    pub dry_run: bool,
    /// Canonical location to write back into `sources.location` on a
    /// successful run. `None` means "leave whatever's there" — used by
    /// URL-based adapters where the URL was already written by `source
    /// add` and re-deriving it here would just be a round-trip.
    pub canonical_location: Option<String>,
    /// Whether the caller had an explicit location (CLI `--path`, MCP
    /// registry entry). Surfaces in the audit entry for diagnostics; the
    /// store doesn't care.
    pub source_was_explicit: bool,
    /// Scan-level filters (`since`, `full`). Passed straight through to
    /// `adapter.scan(opts)` so adapters that honor it can skip records
    /// older than `since`. Default is "no filter / full scan".
    pub scan_opts: ScanOpts,
}

/// Errors that `ImportService` can fail with, on top of plain
/// `ImportError`. Distinguished so callers can render the friendlier
/// "import succeeded but audit failed" type messages.
#[derive(Debug, Error)]
pub enum ImportServiceError {
    /// Underlying `ImportRunner` failure (store-level abort).
    #[error("runner: {0}")]
    Runner(#[from] ImportError),
    /// Store-level failure outside the runner (registry / last_import_at).
    #[error("store: {0}")]
    Store(#[from] anamnesis_store::StoreError),
}

/// Service object that wraps `ImportRunner` with the side effects
/// every caller agreed to share (§-1.6.9): write through `source`
/// registry (without clobbering `config_json`), stamp `last_import_at`
/// only on a real run, and append a single `import` entry to the
/// data-dir `audit.log`.
///
/// This is the load-bearing seam for §-1.5 PR-3 — before round-18 the
/// CLI `run_import` did all of this inline while the MCP
/// `tool_import_source` did almost none of it. Both now flow through
/// here; see `crates/cli/src/main.rs::run_import` and
/// `crates/mcp-server/src/server.rs::tool_import_source`.
pub struct ImportService<'a> {
    store: &'a Store,
    audit: Audit,
}

impl<'a> ImportService<'a> {
    /// Build a service bound to `store` and an `Audit` log writer.
    pub fn new(store: &'a Store, audit: Audit) -> Self {
        Self { store, audit }
    }

    /// Run an import.
    ///
    /// Order of operations on a non-dry-run:
    ///   1. Look up the existing `(adapter, instance)` row so we can
    ///      preserve `config_json` (and `location` when the caller
    ///      passes `None`). This is the §-1.4 round-17 contract:
    ///      registry side-channel data set by `source add` must
    ///      survive re-import.
    ///   2. `register_source` — idempotent upsert with preserved
    ///      `config_json`. This happens BEFORE the runner so even a
    ///      partially-failing run leaves the source visible in
    ///      `source list`.
    ///   3. `ImportRunner::run` — the actual scan/normalize/upsert.
    ///   4. `update_last_import_at` — only on success.
    ///   5. `audit.record("import", ...)` — single line per call.
    ///
    /// Dry-run mode short-circuits to `adapter.scan().count()` without
    /// any of (1–5). `summary.raw_seen` reflects the count;
    /// `records_upserted == 0`, `chunks_written == 0`, `errors == 0`.
    pub async fn import<A: MemoryAdapter>(
        &self,
        adapter: &A,
        opts: ImportOptions,
    ) -> std::result::Result<ImportSummary, ImportServiceError> {
        let descriptor = adapter.descriptor();

        if opts.dry_run {
            // dry-run NEVER touches the registry, last_import_at, or
            // audit (`source list` should reflect only persisted state).
            //
            // Round-19: dry-run honors `scan_opts` too — `--dry-run
            // --since X` should report what an incremental run WOULD
            // pull in, not the full corpus.
            let mut stream = adapter.scan(opts.scan_opts.clone());
            let mut seen = 0u64;
            while let Some(item) = stream.next().await {
                if item.is_ok() {
                    seen += 1;
                }
            }
            let mut summary =
                ImportSummary::empty(&descriptor.adapter, descriptor.instance.as_deref());
            summary.raw_seen = seen;
            return Ok(summary);
        }

        // 1. Preserve registry side-channel data.
        let existing = self
            .store
            .get_source(&descriptor.adapter, descriptor.instance.as_deref())?;
        let new_location: Option<String> = opts
            .canonical_location
            .clone()
            .or_else(|| existing.as_ref().and_then(|r| r.location.clone()));
        let existing_config = existing.and_then(|r| r.config_json);

        // 2. Pre-register so `source list` shows the source even after a
        //    partial run.
        self.store.register_source(
            &descriptor.adapter,
            descriptor.instance.as_deref(),
            new_location.as_deref(),
            existing_config.as_deref(),
        )?;

        // 3. The real work — pass scan opts through to the adapter.
        let summary = ImportRunner::new(adapter)
            .run_with_opts(self.store, opts.scan_opts.clone())
            .await?;

        // 4. Mark success.
        self.store
            .update_last_import_at(&descriptor.adapter, descriptor.instance.as_deref())?;

        // 5. Audit.
        self.audit.record(AuditEntry::new(
            "import",
            serde_json::json!({
                "adapter": descriptor.adapter,
                "instance": descriptor.instance,
                "raw_seen": summary.raw_seen,
                "records_upserted": summary.records_upserted,
                "chunks_written": summary.chunks_written,
                "errors": summary.errors,
                "location": new_location,
                "source_was_explicit": opts.source_was_explicit,
            }),
        ));

        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::adapter::{HealthStatus, RawRecord, ScanOpts};
    use anamnesis_core::error::{Error, Result};
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use async_trait::async_trait;
    use chrono::Utc;
    use futures::stream::{self, BoxStream};

    /// In-memory adapter: emits the records you constructed it with, and
    /// optionally fails normalize() on `bad_id` so we can test error paths.
    struct FakeAdapter {
        records: Vec<(RawRecord, AnamnesisRecord)>,
        fail_on: Option<String>,
    }

    impl FakeAdapter {
        fn new(records: Vec<(RawRecord, AnamnesisRecord)>) -> Self {
            Self {
                records,
                fail_on: None,
            }
        }
        fn with_normalize_failure(mut self, native_id: &str) -> Self {
            self.fail_on = Some(native_id.to_string());
            self
        }
    }

    #[async_trait]
    impl MemoryAdapter for FakeAdapter {
        fn descriptor(&self) -> SourceDescriptor {
            SourceDescriptor {
                adapter: "fake".into(),
                instance: Some("default".into()),
                version: "0".into(),
            }
        }
        fn scan<'a>(&'a self, opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
            // Round-19 (§-1.5 PR-4a): mock honors `opts.since` so the
            // importer-level test asserting "ScanOpts flows through
            // ImportService.import" can be exercised without dragging
            // in a real file/sqlite/HTTP adapter.
            let raws: Vec<Result<RawRecord>> = self
                .records
                .iter()
                .filter(|(r, _)| match opts.since {
                    Some(t) if !opts.full => r.captured_at > t,
                    _ => true,
                })
                .map(|(r, _)| Ok(r.clone()))
                .collect();
            Box::pin(stream::iter(raws))
        }
        fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
            if let Some(bad) = &self.fail_on {
                if &raw.native_id == bad {
                    return Err(Error::InvalidRecord("fake failure".into()));
                }
            }
            let rec = self
                .records
                .iter()
                .find(|(r, _)| r.native_id == raw.native_id)
                .map(|(_, rec)| rec.clone())
                .ok_or_else(|| Error::InvalidRecord("not found".into()))?;
            Ok(vec![rec])
        }
        async fn health(&self) -> HealthStatus {
            HealthStatus {
                ok: true,
                detail: "fake".into(),
            }
        }
    }

    fn pair(native_id: &str, content: &str) -> (RawRecord, AnamnesisRecord) {
        let raw = RawRecord {
            native_id: native_id.into(),
            native_path: Some(format!("/fake/{native_id}")),
            payload: serde_json::json!({"v": 1}),
            captured_at: Utc::now(),
        };
        let record = AnamnesisRecord {
            id: RecordId::from_parts("fake", Some("default"), native_id),
            source: SourceDescriptor {
                adapter: "fake".into(),
                instance: Some("default".into()),
                version: "0".into(),
            },
            content: content.into(),
            embedding: None,
            scope: Scope::User,
            kind: Kind::Fact,
            created_at: Utc::now(),
            updated_at: None,
            tags: vec![],
            metadata: Default::default(),
            provenance: Provenance {
                native_id: native_id.into(),
                native_path: Some(format!("/fake/{native_id}")),
                captured_at: Utc::now(),
                raw_hash: "h".into(),
                derived_from: None,
            },
            schema_version: SCHEMA_VERSION,
        };
        (raw, record)
    }

    #[tokio::test]
    async fn happy_path_upserts_all_records() {
        let store = Store::open_in_memory().unwrap();
        let adapter = FakeAdapter::new(vec![
            pair("a", "alpha"),
            pair("b", "beta gamma"),
            pair("c", "delta epsilon zeta"),
        ]);
        let summary = ImportRunner::new(&adapter).run(&store).await.unwrap();
        assert_eq!(summary.adapter, "fake");
        assert_eq!(summary.instance.as_deref(), Some("default"));
        assert_eq!(summary.raw_seen, 3);
        assert_eq!(summary.records_upserted, 3);
        assert_eq!(summary.chunks_written, 3);
        assert_eq!(summary.errors, 0);

        let stats = store.stats().unwrap();
        assert_eq!(stats.records, 3);
        assert_eq!(stats.chunks, 3);
    }

    #[tokio::test]
    async fn normalize_failure_logs_error_and_continues() {
        let store = Store::open_in_memory().unwrap();
        let adapter = FakeAdapter::new(vec![
            pair("good1", "x"),
            pair("bad", "y"),
            pair("good2", "z"),
        ])
        .with_normalize_failure("bad");
        let summary = ImportRunner::new(&adapter).run(&store).await.unwrap();
        assert_eq!(summary.raw_seen, 3);
        assert_eq!(summary.records_upserted, 2);
        assert_eq!(summary.errors, 1);

        // import_errors should have one row.
        let n: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(1) FROM import_errors WHERE phase = 'normalize'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
    }

    #[tokio::test]
    async fn import_is_idempotent() {
        // Round-7 contract change: `upsert_record` now returns (0, 0) when
        // raw_hash is unchanged, so the second run honestly reports
        // "saw the same 2 raw records, wrote zero". The row count remains
        // 2 (idempotency holds), but the summary stops claiming work it
        // didn't do — which was the round-6 dogfood finding.
        let store = Store::open_in_memory().unwrap();
        let adapter = FakeAdapter::new(vec![pair("a", "alpha"), pair("b", "beta")]);
        let s1 = ImportRunner::new(&adapter).run(&store).await.unwrap();
        assert_eq!(s1.records_upserted, 2);
        assert_eq!(s1.chunks_written, 2);
        let s2 = ImportRunner::new(&adapter).run(&store).await.unwrap();
        assert_eq!(s2.raw_seen, 2, "second run still scans both raws");
        assert_eq!(
            s2.records_upserted, 0,
            "no record rewrite when raw_hash unchanged"
        );
        assert_eq!(
            s2.chunks_written, 0,
            "no chunk DELETE/INSERT when raw_hash unchanged"
        );
        assert_eq!(s2.errors, 0);
        // Two runs do not double the row count (idempotency on disk).
        let stats = store.stats().unwrap();
        assert_eq!(stats.records, 2);
        assert_eq!(stats.chunks, 2);
    }

    #[tokio::test]
    async fn raw_payload_persisted_for_provenance() {
        let store = Store::open_in_memory().unwrap();
        let adapter = FakeAdapter::new(vec![pair("a", "alpha")]);
        ImportRunner::new(&adapter).run(&store).await.unwrap();
        let payload: String = store
            .conn()
            .query_row("SELECT payload_json FROM raw_artifacts LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(payload.contains("\"v\":1"));
    }

    #[tokio::test]
    async fn embedding_jobs_enqueued_when_active_model_set() {
        let store = Store::open_in_memory().unwrap();
        store.set_active_model("local:fake:1").unwrap();
        let adapter = FakeAdapter::new(vec![pair("a", "alpha"), pair("b", "beta")]);
        ImportRunner::new(&adapter).run(&store).await.unwrap();
        let pending: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(1) FROM embedding_jobs WHERE status = 'pending'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(pending, 2, "one job per chunk × 2 records");
    }

    #[tokio::test]
    async fn no_jobs_when_no_active_model() {
        let store = Store::open_in_memory().unwrap();
        let adapter = FakeAdapter::new(vec![pair("a", "alpha")]);
        ImportRunner::new(&adapter).run(&store).await.unwrap();
        let n: i64 = store
            .conn()
            .query_row("SELECT COUNT(1) FROM embedding_jobs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[tokio::test]
    async fn custom_chunker_takes_effect() {
        use anamnesis_core::chunker::{Chunker as Ck, ChunkerConfig};
        let store = Store::open_in_memory().unwrap();
        let long = "para one ".repeat(80) + "\n\n" + &"para two ".repeat(80);
        let adapter = FakeAdapter::new(vec![pair("a", &long)]);
        let tiny = Ck::new(ChunkerConfig {
            max_tokens: 50,
            min_tokens: 5,
        });
        let summary = ImportRunner::new(&adapter)
            .with_chunker(tiny)
            .run(&store)
            .await
            .unwrap();
        assert_eq!(summary.records_upserted, 1);
        assert!(
            summary.chunks_written > 1,
            "tiny budget should produce >1 chunk"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Round-18 (§-1.5 PR-3): ImportService side-effect contract tests
    // ─────────────────────────────────────────────────────────────────────

    /// Open a real on-disk store + audit log writer in `tempdir`. The
    /// in-memory store would work for store side effects, but `Audit`
    /// needs a real `data_dir` to write `audit.log`.
    fn store_and_audit() -> (Store, anamnesis_core::Audit, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Store::open(dir.path().join("anamnesis.sqlite")).expect("open store");
        let audit = anamnesis_core::Audit::new(dir.path());
        (store, audit, dir)
    }

    /// Read the audit.log file produced by `Audit::new(dir).record(...)`
    /// and return one JSON value per line.
    fn read_audit_lines(dir: &std::path::Path) -> Vec<serde_json::Value> {
        let raw = std::fs::read_to_string(dir.join("audit.log")).unwrap_or_default();
        raw.lines()
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("audit line is json"))
            .collect()
    }

    #[tokio::test]
    async fn import_service_dry_run_writes_nothing_but_counts_raw() {
        let (store, audit, dir) = store_and_audit();
        let adapter = FakeAdapter::new(vec![pair("a", "alpha"), pair("b", "beta")]);

        let summary = ImportService::new(&store, audit)
            .import(
                &adapter,
                ImportOptions {
                    dry_run: true,
                    canonical_location: Some("/somewhere".into()),
                    source_was_explicit: true,
                    ..Default::default()
                },
            )
            .await
            .expect("dry-run must succeed");

        // raw_seen reflects the scan; nothing else moves.
        assert_eq!(summary.raw_seen, 2);
        assert_eq!(summary.records_upserted, 0);
        assert_eq!(summary.chunks_written, 0);
        assert_eq!(summary.errors, 0);

        // Source registry untouched: dry-run NEVER pre-registers.
        let row = store.get_source("fake", Some("default")).unwrap();
        assert!(
            row.is_none(),
            "dry-run must not register the source (would mislead `source list`)"
        );

        // No audit entry.
        let lines = read_audit_lines(dir.path());
        assert!(
            lines.is_empty(),
            "dry-run must not append to audit.log; got: {lines:?}"
        );
    }

    #[tokio::test]
    async fn import_service_writes_registry_last_import_at_and_audit_on_success() {
        let (store, audit, dir) = store_and_audit();
        let adapter = FakeAdapter::new(vec![pair("a", "alpha")]);

        let summary = ImportService::new(&store, audit)
            .import(
                &adapter,
                ImportOptions {
                    dry_run: false,
                    canonical_location: Some("/tmp/round18".into()),
                    source_was_explicit: true,
                    ..Default::default()
                },
            )
            .await
            .expect("import must succeed");

        assert_eq!(summary.raw_seen, 1);
        assert_eq!(summary.records_upserted, 1);
        assert!(summary.chunks_written >= 1);

        // Source row exists with the canonical location AND last_import_at.
        let row = store
            .get_source("fake", Some("default"))
            .unwrap()
            .expect("source row must exist after import");
        assert_eq!(row.location.as_deref(), Some("/tmp/round18"));
        assert!(
            row.last_import_at.is_some(),
            "last_import_at must be stamped on a successful run"
        );

        // Exactly one audit line, of action "import".
        let lines = read_audit_lines(dir.path());
        assert_eq!(lines.len(), 1, "expected one audit line, got {lines:?}");
        assert_eq!(lines[0]["action"], "import");
        assert_eq!(lines[0]["detail"]["adapter"], "fake");
        assert_eq!(lines[0]["detail"]["records_upserted"], 1);
        assert_eq!(lines[0]["detail"]["location"], "/tmp/round18");
        assert_eq!(lines[0]["detail"]["source_was_explicit"], true);
    }

    #[tokio::test]
    async fn import_service_preserves_config_json_across_reimport() {
        // Round-17 PR-#25's fix in CLI's run_import migrated here: a
        // generic-mcp source registered with config_json={"token_env":
        // "..."} must keep that config_json after re-import (no None
        // clobber).
        let (store, audit, dir) = store_and_audit();

        // 1. Operator did `source add fake --token-env FOO` first.
        store
            .register_source(
                "fake",
                Some("default"),
                Some("https://upstream/api"),
                Some(r#"{"token_env":"ANAMNESIS_FAKE_TOKEN"}"#),
            )
            .unwrap();

        // 2. ImportService runs with canonical_location=None (the URL is
        //    already in registry — file adapters pass Some(...), URL
        //    adapters pass None).
        let adapter = FakeAdapter::new(vec![pair("a", "alpha")]);
        ImportService::new(&store, audit)
            .import(
                &adapter,
                ImportOptions {
                    dry_run: false,
                    canonical_location: None,
                    source_was_explicit: true,
                    ..Default::default()
                },
            )
            .await
            .expect("import must succeed");

        // 3. The URL AND the token_env config must survive.
        let row = store
            .get_source("fake", Some("default"))
            .unwrap()
            .expect("source row must exist");
        assert_eq!(
            row.location.as_deref(),
            Some("https://upstream/api"),
            "URL must survive re-registration"
        );
        let cfg = row.config_json.as_deref().unwrap_or("");
        assert!(
            cfg.contains("ANAMNESIS_FAKE_TOKEN"),
            "token_env must survive; got config_json={cfg:?}"
        );

        // Audit still recorded.
        assert_eq!(read_audit_lines(dir.path()).len(), 1);
    }

    #[tokio::test]
    async fn import_service_does_not_stamp_last_import_on_normalize_only_failures() {
        // A normalize() failure is per-record (counted in `errors`), not a
        // store-level abort — ImportRunner returns Ok with errors > 0.
        // We still consider the run "successful" in the system-state
        // sense (it produced an `import` audit entry, `last_import_at`
        // got stamped, sources were registered). This documents that
        // contract so a future "fail run if any per-record errors" change
        // doesn't silently break the audit/registry contract.
        let (store, audit, dir) = store_and_audit();
        let adapter = FakeAdapter::new(vec![pair("good", "ok"), pair("bad", "x")])
            .with_normalize_failure("bad");

        let summary = ImportService::new(&store, audit)
            .import(
                &adapter,
                ImportOptions {
                    dry_run: false,
                    canonical_location: Some("/p".into()),
                    source_was_explicit: false,
                    ..Default::default()
                },
            )
            .await
            .expect("partial-failure run still returns Ok at runner level");

        assert_eq!(summary.raw_seen, 2);
        assert_eq!(summary.records_upserted, 1);
        assert_eq!(summary.errors, 1);

        let row = store.get_source("fake", Some("default")).unwrap().unwrap();
        assert!(row.last_import_at.is_some());
        let lines = read_audit_lines(dir.path());
        assert_eq!(lines[0]["detail"]["errors"], 1);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Round-19 (§-1.5 PR-4a): ScanOpts flows through ImportService.import
    // ─────────────────────────────────────────────────────────────────────

    /// Like `pair`, but stamps `RawRecord.captured_at` at the given UTC
    /// time so tests can compose deterministic before/after windows.
    fn pair_at(
        native_id: &str,
        content: &str,
        captured_at: chrono::DateTime<chrono::Utc>,
    ) -> (RawRecord, AnamnesisRecord) {
        let (mut raw, mut record) = pair(native_id, content);
        raw.captured_at = captured_at;
        record.provenance.captured_at = captured_at;
        (raw, record)
    }

    #[tokio::test]
    async fn import_service_scan_opts_since_drops_records_older_than_threshold() {
        // Three records at t0, t0+5min, t0+10min. ImportService called
        // with scan_opts.since = t0+3min must only ingest the t0+5min
        // and t0+10min ones.
        let (store, audit, _dir) = store_and_audit();
        let t0 = chrono::Utc::now() - chrono::Duration::hours(1);
        let adapter = FakeAdapter::new(vec![
            pair_at("old", "alpha", t0),
            pair_at("mid", "bravo", t0 + chrono::Duration::minutes(5)),
            pair_at("new", "charlie", t0 + chrono::Duration::minutes(10)),
        ]);

        let summary = ImportService::new(&store, audit)
            .import(
                &adapter,
                ImportOptions {
                    dry_run: false,
                    canonical_location: Some("/p".into()),
                    source_was_explicit: true,
                    scan_opts: ScanOpts {
                        since: Some(t0 + chrono::Duration::minutes(3)),
                        full: false,
                    },
                },
            )
            .await
            .expect("import");

        assert_eq!(
            summary.raw_seen, 2,
            "since-window should have dropped the t0 record before scan"
        );
        assert_eq!(summary.records_upserted, 2);
    }

    #[tokio::test]
    async fn import_service_scan_opts_full_overrides_since_filter() {
        // `full = true` must IGNORE `since` — the §-1.5 PR-4a contract.
        // Three records at t0/+5/+10, since=t0+3min, full=true → all 3
        // come through.
        let (store, audit, _dir) = store_and_audit();
        let t0 = chrono::Utc::now() - chrono::Duration::hours(1);
        let adapter = FakeAdapter::new(vec![
            pair_at("old", "alpha", t0),
            pair_at("mid", "bravo", t0 + chrono::Duration::minutes(5)),
            pair_at("new", "charlie", t0 + chrono::Duration::minutes(10)),
        ]);

        let summary = ImportService::new(&store, audit)
            .import(
                &adapter,
                ImportOptions {
                    dry_run: false,
                    canonical_location: Some("/p".into()),
                    source_was_explicit: true,
                    scan_opts: ScanOpts {
                        since: Some(t0 + chrono::Duration::minutes(3)),
                        full: true,
                    },
                },
            )
            .await
            .expect("import");

        assert_eq!(
            summary.raw_seen, 3,
            "--full must override --since; expected all 3 records through"
        );
        assert_eq!(summary.records_upserted, 3);
    }

    #[tokio::test]
    async fn import_service_dry_run_honors_since_filter() {
        // Dry-run + since: should COUNT only post-threshold records
        // without touching the store. This guards the §-1.5 PR-4a
        // promise that `--dry-run --since X` reports incremental
        // size, not full-corpus size.
        let (store, audit, dir) = store_and_audit();
        let t0 = chrono::Utc::now() - chrono::Duration::hours(1);
        let adapter = FakeAdapter::new(vec![
            pair_at("old", "alpha", t0),
            pair_at("new", "bravo", t0 + chrono::Duration::minutes(10)),
        ]);

        let summary = ImportService::new(&store, audit)
            .import(
                &adapter,
                ImportOptions {
                    dry_run: true,
                    canonical_location: Some("/p".into()),
                    source_was_explicit: true,
                    scan_opts: ScanOpts {
                        since: Some(t0 + chrono::Duration::minutes(5)),
                        full: false,
                    },
                },
            )
            .await
            .expect("dry-run");

        assert_eq!(
            summary.raw_seen, 1,
            "dry-run must count only post-since raw records"
        );
        assert_eq!(summary.records_upserted, 0);
        // No registry, no audit on dry-run.
        assert!(store.get_source("fake", Some("default")).unwrap().is_none());
        assert!(read_audit_lines(dir.path()).is_empty());
    }
}
