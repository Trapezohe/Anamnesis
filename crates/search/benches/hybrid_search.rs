//! Round-66 baseline: end-to-end hybrid search through `HybridSearcher`.
//!
//! Measures the fulltext-only path (no embedding provider) so the bench
//! doesn't depend on fastembed model files. The FTS path still exercises
//! the SQL prepare + jieba tokenizer + bm25 scoring, which is the second
//! hot path after vector search.
//!
//! Round 67's sqlite-vec swap won't touch the FTS path, so this bench
//! locks in a "vector-side improvement must not regress FTS" guarantee.

use anamnesis_core::chunker::Chunker;
use anamnesis_core::embedding::{EmbeddingProvider, EmbeddingTask, ModelId};
use anamnesis_core::error::{Error as CoreError, Result as CoreResult};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_search::{HybridOpts, HybridSearcher, SearchMode};
use anamnesis_store::Store;
use async_trait::async_trait;
use chrono::Utc;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use tokio::runtime::Runtime;

const QUERIES: &[&str] = &[
    "alpha",
    "bench",
    "memory record",
    "interoperability layer",
    "concurrent reads",
];

/// A provider that satisfies `HybridSearcher`'s type bound without ever
/// being called — `SearchMode::Fulltext` never asks for embeddings.
struct UnusedProvider;

#[async_trait]
impl EmbeddingProvider for UnusedProvider {
    fn model_id(&self) -> ModelId {
        ModelId::new("bench", "unused", 1)
    }
    fn dim(&self) -> u16 {
        1
    }
    async fn embed_batch(
        &self,
        _texts: &[&str],
        _task: EmbeddingTask,
    ) -> CoreResult<Vec<Vec<f32>>> {
        Err(CoreError::Other(
            "UnusedProvider must not be invoked in fulltext-only bench".into(),
        ))
    }
}

fn make_record(i: usize) -> AnamnesisRecord {
    let id = RecordId::from_parts("bench", None, &format!("r{i}"));
    AnamnesisRecord {
        id,
        source: SourceDescriptor {
            adapter: "bench".into(),
            instance: None,
            version: "0".into(),
        },
        content: format!(
            "bench record {i} — alpha bench memory record about an interoperability layer with concurrent reads"
        ),
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at: Utc::now(),
        updated_at: None,
        tags: vec![],
        metadata: Default::default(),
        provenance: Provenance {
            native_id: format!("r{i}"),
            native_path: None,
            captured_at: Utc::now(),
            raw_hash: format!("h{i}"),
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    }
}

fn seed(store: &Store, n: usize) {
    let chunker = Chunker::default();
    let mut buf: Vec<(AnamnesisRecord, Vec<_>, Option<String>)> = Vec::with_capacity(n);
    for i in 0..n {
        let r = make_record(i);
        let chunks = chunker.chunk(&r.id, &r.content);
        buf.push((r, chunks, None));
    }
    let borrowed: Vec<(&AnamnesisRecord, &[_], Option<&str>)> = buf
        .iter()
        .map(|(r, c, j)| (r, c.as_slice(), j.as_deref()))
        .collect();
    store.upsert_records_batch(&borrowed).unwrap();
}

fn bench_hybrid_fulltext(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("hybrid_fulltext_search");
    for n in [100usize, 1_000, 5_000] {
        let store = Store::open_in_memory().unwrap();
        seed(&store, n);
        let opts = HybridOpts {
            limit: 10,
            candidate_pool: 80,
            mode: SearchMode::Fulltext,
        };
        let mut q = 0usize;
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let query = QUERIES[q % QUERIES.len()];
                q += 1;
                let hits = rt.block_on(async {
                    HybridSearcher::<UnusedProvider>::fulltext_only()
                        .search(&store, black_box(query), black_box(&opts))
                        .await
                        .unwrap()
                });
                black_box(hits);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hybrid_fulltext);
criterion_main!(benches);
