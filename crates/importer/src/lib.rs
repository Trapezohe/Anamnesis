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

use anamnesis_core::adapter::MemoryAdapter;
use anamnesis_core::chunker::Chunker;
use anamnesis_core::RawRecord;
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

    /// Drain `adapter.scan()` into `store`. Returns the run summary.
    pub async fn run(&self, store: &Store) -> std::result::Result<ImportSummary, ImportError> {
        let descriptor = self.adapter.descriptor();
        let mut summary = ImportSummary::empty(&descriptor.adapter, descriptor.instance.as_deref());
        let mut stream = self.adapter.scan(Default::default());

        while let Some(item) = stream.next().await {
            match item {
                Ok(raw) => {
                    summary.raw_seen += 1;
                    self.process_one(raw, store, &descriptor, &mut summary);
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
        Ok(summary)
    }

    fn process_one(
        &self,
        raw: RawRecord,
        store: &Store,
        descriptor: &anamnesis_core::SourceDescriptor,
        summary: &mut ImportSummary,
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
            match store.upsert_record(&record, &chunks, raw_payload_json.as_deref()) {
                Ok((_, n_chunks)) => {
                    summary.records_upserted += 1;
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
        fn scan<'a>(&'a self, _opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
            let raws: Vec<Result<RawRecord>> =
                self.records.iter().map(|(r, _)| Ok(r.clone())).collect();
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
        let store = Store::open_in_memory().unwrap();
        let adapter = FakeAdapter::new(vec![pair("a", "alpha"), pair("b", "beta")]);
        let s1 = ImportRunner::new(&adapter).run(&store).await.unwrap();
        let s2 = ImportRunner::new(&adapter).run(&store).await.unwrap();
        assert_eq!(s1, s2);
        // Two runs should not double the row count.
        let stats = store.stats().unwrap();
        assert_eq!(stats.records, 2);
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
}
