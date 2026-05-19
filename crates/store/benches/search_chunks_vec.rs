//! Round-66 baseline: full-table vector search via `Store::search_chunks_vec`.
//!
//! `search_chunks_vec` runs a full-table scan + Rust-side cosine; there's no
//! ANN index (the schema comment calls out sqlite-vec as a "later migration").
//! This bench freezes a baseline so Round 67's sqlite-vec swap has concrete
//! before/after numbers instead of "feels faster."
//!
//! The bench seeds N synthetic records (one chunk per record, one 384-dim
//! embedding per chunk) into an in-memory store, then measures one
//! `search_chunks_vec` call per iteration with `limit = 10`.

use anamnesis_core::chunker::Chunker;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_store::{SearchFilter, Store};
use chrono::Utc;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

const DIM: usize = 384;
const MODEL: &str = "bench:e5-small:1";

fn make_record(i: usize) -> AnamnesisRecord {
    let id = RecordId::from_parts("bench", None, &format!("r{i}"));
    AnamnesisRecord {
        id: id.clone(),
        source: SourceDescriptor {
            adapter: "bench".into(),
            instance: None,
            version: "0".into(),
        },
        content: format!("bench record {i} content"),
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

/// Build a deterministic 384-dim vector that differs across `i` so cosine
/// scores spread (otherwise everything ties and the bench is meaningless).
fn make_vec(i: usize) -> Vec<f32> {
    let base = (i as f32 + 1.0) * 0.001;
    (0..DIM).map(|d| base + (d as f32 * 0.0001)).collect()
}

fn seed(store: &Store, n: usize) {
    store.set_active_model(MODEL).unwrap();
    let chunker = Chunker::default();
    let mut buffered: Vec<(AnamnesisRecord, Vec<_>, Option<String>)> = Vec::with_capacity(n);
    for i in 0..n {
        let r = make_record(i);
        let chunks = chunker.chunk(&r.id, &r.content);
        buffered.push((r, chunks, None));
    }
    let borrowed: Vec<(&AnamnesisRecord, &[_], Option<&str>)> = buffered
        .iter()
        .map(|(r, c, j)| (r, c.as_slice(), j.as_deref()))
        .collect();
    store.upsert_records_batch(&borrowed).unwrap();

    // Drain the embedding_jobs queue with deterministic synthetic vectors.
    let jobs = store.claim_next_jobs(MODEL, n + 16).unwrap();
    let vectors: Vec<Vec<f32>> = (0..jobs.len()).map(make_vec).collect();
    store.complete_jobs_batch(&jobs, &vectors).unwrap();
}

fn bench_search_chunks_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_chunks_vec");
    for n in [100usize, 1_000, 5_000] {
        let store = Store::open_in_memory().unwrap();
        seed(&store, n);
        let query = make_vec(0); // matches the seed distribution.
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let hits = store
                    .search_chunks_vec(
                        black_box(&query),
                        black_box(MODEL),
                        black_box(&SearchFilter::default()),
                        black_box(10),
                    )
                    .unwrap();
                black_box(hits)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_search_chunks_vec);
criterion_main!(benches);
