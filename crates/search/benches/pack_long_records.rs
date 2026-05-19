//! Round-68 baseline / target: `pack()` over long-content records.
//!
//! `pack()` aggregates `RankedChunk` hits into `PackedRecord`s by
//! looking up record metadata in one batched store call. Pre-Round-68
//! this batched call selected `records.content` even though MCP /
//! CLI never returned it in the wire format — the path materialised
//! and threw away multi-KB transcripts for every hit. Round 68
//! switches `pack()` to `get_record_headers_by_ids`, which omits
//! `content / tags / metadata` from the projection.
//!
//! This bench seeds `N` records, each with a ~64 KiB synthetic
//! `content` (representative of Claude Code / Codex
//! adapter-rendered session transcripts), then measures one
//! `pack()` call over `40` `RankedChunk` hits sampled across those
//! records. The wall time *should* fall ≥60% once Round 68 lands
//! relative to the previous `get_records_by_ids` path.

use std::hint::black_box;

use anamnesis_core::chunker::Chunker;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_search::{pack, ContextBudget, RankedChunk};
use anamnesis_store::Store;
use chrono::Utc;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

const RECORDS: usize = 1_000;
const HITS: usize = 40;
/// ~64 KiB per record `content`. Built by repeating a fixed token so
/// the chunker actually produces multiple chunks per record (otherwise
/// the bench would degenerate into a single-chunk-per-record case
/// that doesn't stress `pack()`'s grouping path).
const CONTENT_TOKEN: &str = "lorem ipsum dolor sit amet consectetur adipiscing elit ";
const CONTENT_REPEATS: usize = 1_200; // ≈64 KiB.

fn make_long_record(i: usize) -> AnamnesisRecord {
    let id = RecordId::from_parts("bench", None, &format!("r{i}"));
    let content = CONTENT_TOKEN.repeat(CONTENT_REPEATS);
    AnamnesisRecord {
        id,
        source: SourceDescriptor {
            adapter: "bench".into(),
            instance: None,
            version: "0".into(),
        },
        content,
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

fn seed(store: &Store) {
    let chunker = Chunker::default();
    let mut buf: Vec<(AnamnesisRecord, Vec<_>, Option<String>)> = Vec::with_capacity(RECORDS);
    for i in 0..RECORDS {
        let r = make_long_record(i);
        let chunks = chunker.chunk(&r.id, &r.content);
        buf.push((r, chunks, None));
    }
    let borrowed: Vec<(&AnamnesisRecord, &[_], Option<&str>)> = buf
        .iter()
        .map(|(r, c, j)| (r, c.as_slice(), j.as_deref()))
        .collect();
    store.upsert_records_batch(&borrowed).unwrap();
}

/// Synthesize `HITS` `RankedChunk`s spread across the seeded records.
/// Snippet length stays small (one `CONTENT_TOKEN`) so the chunk-side
/// payload doesn't dominate — the cost we want to surface is the
/// record-side projection, not the chunk text we always have to
/// carry.
fn make_hits() -> Vec<RankedChunk> {
    (0..HITS)
        .map(|i| {
            // Step through the corpus rather than concentrating on the
            // first records — pack()'s grouping + diversity caps want
            // a believable spread.
            let r_idx = (i * (RECORDS / HITS.max(1))) % RECORDS;
            let record_id = RecordId::from_parts("bench", None, &format!("r{r_idx}"));
            let chunk_id = format!("{}:0", record_id.0);
            RankedChunk {
                chunk_id,
                record_id,
                seq: 0,
                content: CONTENT_TOKEN.to_string(),
                score: 1.0 / (1.0 + i as f64),
                fts_score: Some(0.5),
                vector_score: None,
                from_fts: true,
                from_vec: false,
            }
        })
        .collect()
}

fn bench_pack_long_records(c: &mut Criterion) {
    let store = Store::open_in_memory().unwrap();
    seed(&store);
    let hits = make_hits();
    let budget = ContextBudget {
        max_records: HITS,
        max_total_tokens: None,
        max_per_source: None,
        max_per_project: None,
    };
    let mut group = c.benchmark_group("pack_long_records");
    group.bench_with_input(
        BenchmarkId::from_parameter(format!("{HITS}_hits_{RECORDS}_records")),
        &(),
        |b, _| {
            b.iter(|| {
                let out = pack(&store, black_box(&hits), black_box(&budget)).unwrap();
                black_box(out);
            });
        },
    );
    group.finish();
}

criterion_group!(benches, bench_pack_long_records);
criterion_main!(benches);
