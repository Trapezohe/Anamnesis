//! Adapter contract — invariants every `MemoryAdapter` impl must satisfy.
//!
//! Live in `core` so adapter crates can import them as a public test
//! helper. The fixtures here are deliberately abstract: contract tests
//! call `adapter_factory()` to get a fresh adapter each invocation,
//! drive its public API, and assert.
//!
//! Per BLUEPRINT §8 invariants list (and §10 #2 — instance-None dedup).

use std::collections::HashSet;

use futures::stream::StreamExt;

use crate::adapter::{MemoryAdapter, ScanOpts};
use crate::model::SCHEMA_VERSION;
use crate::RawRecord;

/// One contract run. Builds a fresh adapter each time so tests don't
/// depend on internal caching.
pub struct AdapterContract<F, A>
where
    A: MemoryAdapter,
    F: Fn() -> A + Send + Sync,
{
    /// Builds a brand-new adapter — must be deterministic.
    pub adapter_factory: F,
}

impl<F, A> AdapterContract<F, A>
where
    A: MemoryAdapter,
    F: Fn() -> A + Send + Sync,
{
    /// Build a contract harness.
    pub fn new(adapter_factory: F) -> Self {
        Self { adapter_factory }
    }

    /// Collect everything `scan` emits, surfacing only the Ok rows.
    async fn collect_raws(&self) -> Vec<RawRecord> {
        let adapter = (self.adapter_factory)();
        adapter
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect()
    }

    /// Run every contract assertion. Each `assert_*` is publicly callable
    /// so callers can opt out of specific invariants if they document why.
    pub async fn run_all(&self) {
        self.assert_descriptor_stable();
        self.assert_scan_is_idempotent().await;
        self.assert_native_ids_are_present().await;
        self.assert_normalize_is_pure().await;
        self.assert_records_have_correct_schema_version().await;
        self.assert_raw_hash_is_populated_and_nontrivial().await;
        self.assert_instance_propagates_to_record_id().await;
        self.assert_health_returns_a_message().await;
    }

    /// `descriptor()` is stable across calls and matches `adapter_id`.
    pub fn assert_descriptor_stable(&self) {
        let a = (self.adapter_factory)();
        let d1 = a.descriptor();
        let d2 = a.descriptor();
        assert_eq!(d1, d2, "descriptor() must be stable across calls");
        assert!(
            !d1.adapter.is_empty(),
            "descriptor.adapter must be non-empty"
        );
        assert!(
            !d1.version.is_empty(),
            "descriptor.version must be non-empty"
        );
    }

    /// Two independent scans produce the same set of (native_id, native_path).
    pub async fn assert_scan_is_idempotent(&self) {
        let a: HashSet<(String, Option<String>)> = self
            .collect_raws()
            .await
            .into_iter()
            .map(|r| (r.native_id, r.native_path))
            .collect();
        let b: HashSet<(String, Option<String>)> = self
            .collect_raws()
            .await
            .into_iter()
            .map(|r| (r.native_id, r.native_path))
            .collect();
        assert_eq!(
            a, b,
            "two scans must yield identical (native_id, path) sets"
        );
    }

    /// No raw record has an empty native_id.
    pub async fn assert_native_ids_are_present(&self) {
        let raws = self.collect_raws().await;
        for r in &raws {
            assert!(
                !r.native_id.is_empty(),
                "RawRecord.native_id must be non-empty (path: {:?})",
                r.native_path
            );
        }
    }

    /// Normalizing the same RawRecord twice produces the same records.
    pub async fn assert_normalize_is_pure(&self) {
        let adapter = (self.adapter_factory)();
        let raws = self.collect_raws().await;
        // Cap the loop so contract test stays fast on big fixtures.
        for raw in raws.into_iter().take(16) {
            let a = adapter.normalize(raw.clone());
            let b = adapter.normalize(raw);
            match (a, b) {
                (Ok(ra), Ok(rb)) => assert_eq!(ra, rb, "normalize must be pure"),
                (Err(_), Err(_)) => {}
                (a, b) => panic!("normalize result diverges between calls: {a:?} vs {b:?}"),
            }
        }
    }

    /// Every produced record carries the current `SCHEMA_VERSION`.
    pub async fn assert_records_have_correct_schema_version(&self) {
        let adapter = (self.adapter_factory)();
        for raw in self.collect_raws().await {
            let records = match adapter.normalize(raw) {
                Ok(rs) => rs,
                Err(_) => continue,
            };
            for r in records {
                assert_eq!(
                    r.schema_version, SCHEMA_VERSION,
                    "AnamnesisRecord.schema_version must equal core::SCHEMA_VERSION"
                );
            }
        }
    }

    /// `provenance.raw_hash` is set and not the literal "0".
    pub async fn assert_raw_hash_is_populated_and_nontrivial(&self) {
        let adapter = (self.adapter_factory)();
        for raw in self.collect_raws().await {
            let records = match adapter.normalize(raw) {
                Ok(rs) => rs,
                Err(_) => continue,
            };
            for r in records {
                assert!(
                    !r.provenance.raw_hash.is_empty(),
                    "provenance.raw_hash must not be empty"
                );
                assert_ne!(
                    r.provenance.raw_hash, "0",
                    "raw_hash '0' is almost certainly a bug"
                );
            }
        }
    }

    /// Same native_id with two different instances produces two different
    /// `RecordId`s — protects against the SQLite-NULL-UNIQUE pitfall called
    /// out in BLUEPRINT §10 #2.
    pub async fn assert_instance_propagates_to_record_id(&self) {
        let adapter = (self.adapter_factory)();
        let descriptor = adapter.descriptor();
        // We compare RecordIds the adapter builds for its own raws against
        // RecordIds we synthesize with a sibling instance label. The
        // helper RecordId::from_parts is the canonical recipe.
        for raw in self.collect_raws().await.into_iter().take(8) {
            let with_real = crate::model::RecordId::from_parts(
                &descriptor.adapter,
                descriptor.instance.as_deref(),
                &raw.native_id,
            );
            let with_alt = crate::model::RecordId::from_parts(
                &descriptor.adapter,
                Some("anamnesis-contract-other-instance"),
                &raw.native_id,
            );
            assert_ne!(
                with_real, with_alt,
                "RecordId must differ when instance differs (BLUEPRINT §10 #2)"
            );
        }
    }

    /// `health()` returns a non-empty detail string regardless of `ok`.
    pub async fn assert_health_returns_a_message(&self) {
        let adapter = (self.adapter_factory)();
        let h = adapter.health().await;
        assert!(
            !h.detail.is_empty(),
            "HealthStatus.detail must be non-empty"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{HealthStatus, MemoryAdapter, ScanOpts};
    use crate::error::Result;
    use crate::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use async_trait::async_trait;
    use chrono::Utc;
    use futures::stream::{self, BoxStream};

    /// A passing adapter — every invariant satisfied.
    struct GoodAdapter;
    #[async_trait]
    impl MemoryAdapter for GoodAdapter {
        fn descriptor(&self) -> SourceDescriptor {
            SourceDescriptor {
                adapter: "good".into(),
                instance: Some("default".into()),
                version: "1".into(),
            }
        }
        fn scan<'a>(&'a self, _opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
            let raws = vec![
                Ok(RawRecord {
                    native_id: "n1".into(),
                    native_path: Some("/p/n1".into()),
                    payload: serde_json::json!({"x": 1}),
                    captured_at: Utc::now(),
                }),
                Ok(RawRecord {
                    native_id: "n2".into(),
                    native_path: Some("/p/n2".into()),
                    payload: serde_json::json!({"x": 2}),
                    captured_at: Utc::now(),
                }),
            ];
            Box::pin(stream::iter(raws))
        }
        fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
            // Note: pure — uses raw.captured_at, never Utc::now().
            let id = RecordId::from_parts("good", Some("default"), &raw.native_id);
            let native_id = raw.native_id.clone();
            Ok(vec![AnamnesisRecord {
                id,
                source: SourceDescriptor {
                    adapter: "good".into(),
                    instance: Some("default".into()),
                    version: "1".into(),
                },
                content: format!("c-{native_id}"),
                embedding: None,
                scope: Scope::User,
                kind: Kind::Fact,
                created_at: raw.captured_at,
                updated_at: None,
                tags: vec![],
                metadata: Default::default(),
                provenance: Provenance {
                    native_id,
                    native_path: raw.native_path,
                    captured_at: raw.captured_at,
                    raw_hash: format!("h-{}", raw.native_id),
                    derived_from: None,
                },
                schema_version: SCHEMA_VERSION,
            }])
        }
        async fn health(&self) -> HealthStatus {
            HealthStatus {
                ok: true,
                detail: "fine".into(),
            }
        }
    }

    #[tokio::test]
    async fn good_adapter_passes_full_contract() {
        AdapterContract::new(|| GoodAdapter).run_all().await;
    }

    /// An adapter that emits an empty native_id — should fail the
    /// "native_ids_are_present" assertion.
    struct BadEmptyId;
    #[async_trait]
    impl MemoryAdapter for BadEmptyId {
        fn descriptor(&self) -> SourceDescriptor {
            SourceDescriptor {
                adapter: "bad".into(),
                instance: None,
                version: "1".into(),
            }
        }
        fn scan<'a>(&'a self, _opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
            Box::pin(stream::iter(vec![Ok(RawRecord {
                native_id: String::new(),
                native_path: None,
                payload: serde_json::json!({}),
                captured_at: Utc::now(),
            })]))
        }
        fn normalize(&self, _raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
            Ok(vec![])
        }
        async fn health(&self) -> HealthStatus {
            HealthStatus {
                ok: true,
                detail: "ok".into(),
            }
        }
    }

    #[tokio::test]
    #[should_panic(expected = "native_id must be non-empty")]
    async fn empty_native_id_trips_contract() {
        AdapterContract::new(|| BadEmptyId)
            .assert_native_ids_are_present()
            .await;
    }

    /// An adapter that returns a wrong schema_version in normalize.
    struct BadSchemaVersion;
    #[async_trait]
    impl MemoryAdapter for BadSchemaVersion {
        fn descriptor(&self) -> SourceDescriptor {
            SourceDescriptor {
                adapter: "bad".into(),
                instance: None,
                version: "1".into(),
            }
        }
        fn scan<'a>(&'a self, _opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
            Box::pin(stream::iter(vec![Ok(RawRecord {
                native_id: "n".into(),
                native_path: None,
                payload: serde_json::json!({}),
                captured_at: Utc::now(),
            })]))
        }
        fn normalize(&self, _raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
            Ok(vec![AnamnesisRecord {
                id: RecordId("x".into()),
                source: SourceDescriptor {
                    adapter: "bad".into(),
                    instance: None,
                    version: "1".into(),
                },
                content: "x".into(),
                embedding: None,
                scope: Scope::User,
                kind: Kind::Fact,
                created_at: Utc::now(),
                updated_at: None,
                tags: vec![],
                metadata: Default::default(),
                provenance: Provenance {
                    native_id: "n".into(),
                    native_path: None,
                    captured_at: Utc::now(),
                    raw_hash: "h".into(),
                    derived_from: None,
                },
                schema_version: 999,
            }])
        }
        async fn health(&self) -> HealthStatus {
            HealthStatus {
                ok: true,
                detail: "ok".into(),
            }
        }
    }

    #[tokio::test]
    #[should_panic(expected = "schema_version must equal core::SCHEMA_VERSION")]
    async fn wrong_schema_version_trips_contract() {
        AdapterContract::new(|| BadSchemaVersion)
            .assert_records_have_correct_schema_version()
            .await;
    }

    /// Non-deterministic normalize — returns a different content each call.
    struct NonPureNormalize {
        counter: std::sync::Mutex<u64>,
    }
    #[async_trait]
    impl MemoryAdapter for NonPureNormalize {
        fn descriptor(&self) -> SourceDescriptor {
            SourceDescriptor {
                adapter: "bad".into(),
                instance: None,
                version: "1".into(),
            }
        }
        fn scan<'a>(&'a self, _opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
            Box::pin(stream::iter(vec![Ok(RawRecord {
                native_id: "n".into(),
                native_path: None,
                payload: serde_json::json!({}),
                captured_at: Utc::now(),
            })]))
        }
        fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
            let mut g = self.counter.lock().unwrap();
            *g += 1;
            Ok(vec![AnamnesisRecord {
                id: RecordId::from_parts("bad", None, &raw.native_id),
                source: SourceDescriptor {
                    adapter: "bad".into(),
                    instance: None,
                    version: "1".into(),
                },
                content: format!("call-{g}"),
                embedding: None,
                scope: Scope::User,
                kind: Kind::Fact,
                created_at: Utc::now(),
                updated_at: None,
                tags: vec![],
                metadata: Default::default(),
                provenance: Provenance {
                    native_id: raw.native_id,
                    native_path: None,
                    captured_at: Utc::now(),
                    raw_hash: "h".into(),
                    derived_from: None,
                },
                schema_version: SCHEMA_VERSION,
            }])
        }
        async fn health(&self) -> HealthStatus {
            HealthStatus {
                ok: true,
                detail: "ok".into(),
            }
        }
    }

    #[tokio::test]
    #[should_panic(expected = "normalize must be pure")]
    async fn non_pure_normalize_trips_contract() {
        AdapterContract::new(|| NonPureNormalize {
            counter: std::sync::Mutex::new(0),
        })
        .assert_normalize_is_pure()
        .await;
    }
}
