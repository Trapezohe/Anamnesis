//! ContextPacker — aggregate `RankedChunk` hits into record-level results
//! ready for MCP / agent consumption.
//!
//! Per BLUEPRINT §6.6.1:
//!   - One PackedRecord per unique parent record.
//!   - All matched chunks for that record kept, ordered by chunk score.
//!   - Provenance preserved (adapter, instance, native_path).
//!   - Diversity cap so one adapter can't dominate results.
//!   - Approximate token budget (chunker's heuristic) to bound the
//!     payload returned to LLMs.
//!   - **No LLM summarization** — Phase 1 keeps the data raw.

use anamnesis_core::chunker::estimate_tokens;
use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::AnamnesisRecord;
use anamnesis_store::Store;
use std::collections::HashMap;

use crate::hybrid::RankedChunk;

/// Budget knobs for `pack`.
#[derive(Debug, Clone)]
pub struct ContextBudget {
    /// Hard cap on the number of returned records.
    pub max_records: usize,
    /// Approximate cumulative chunk-token budget. `None` means unbounded.
    /// Counted using the chunker's heuristic, so it's a guide rather than
    /// a hard model-tokenizer limit.
    pub max_total_tokens: Option<u32>,
    /// Per-(adapter, instance) cap. `None` means no diversity restriction.
    pub max_per_source: Option<usize>,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_records: 20,
            max_total_tokens: Some(4_000),
            max_per_source: Some(5),
        }
    }
}

/// One record returned by the packer, with the chunks that matched.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedRecord {
    /// Full record metadata (provenance, scope, kind, tags, …).
    pub record: AnamnesisRecord,
    /// Chunks that hit the query, score-descending.
    pub matched_chunks: Vec<RankedChunk>,
    /// Best chunk score in this record — the record-level rank key.
    pub score: f64,
}

/// Aggregate hits into records, apply provenance + diversity + budget.
///
/// Hits with a `record_id` that no longer exists in the store (e.g. a
/// race with deletion) are skipped silently.
pub fn pack(
    store: &Store,
    hits: &[RankedChunk],
    budget: &ContextBudget,
) -> Result<Vec<PackedRecord>> {
    // 1. Group by record_id, capture the max score.
    let mut groups: HashMap<String, Vec<RankedChunk>> = HashMap::new();
    let mut order_seen: Vec<String> = Vec::new();
    for h in hits {
        let key = h.record_id.0.clone();
        if !groups.contains_key(&key) {
            order_seen.push(key.clone());
        }
        groups.entry(key).or_default().push(h.clone());
    }
    // 2. Materialize: look up record metadata. Skip vanished records.
    let mut materialized: Vec<PackedRecord> = Vec::new();
    for rid in &order_seen {
        let chunks = groups.remove(rid).expect("group exists by construction");
        let record_id = chunks[0].record_id.clone();
        let record = match store
            .get_record(&record_id)
            .map_err(|e| Error::Other(format!("store get_record: {e}")))?
        {
            Some(r) => r,
            None => continue,
        };
        let mut chunks = chunks;
        chunks.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let score = chunks.first().map(|c| c.score).unwrap_or(0.0);
        materialized.push(PackedRecord {
            record,
            matched_chunks: chunks,
            score,
        });
    }
    // 3. Sort by record-level score descending.
    materialized.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 4. Apply per-source diversity cap (preserves overall ordering).
    if let Some(max_per) = budget.max_per_source {
        let mut counts: HashMap<(String, String), usize> = HashMap::new();
        materialized.retain(|p| {
            let key = (
                p.record.source.adapter.clone(),
                p.record.source.instance.clone().unwrap_or_default(),
            );
            let entry = counts.entry(key).or_insert(0);
            if *entry >= max_per {
                false
            } else {
                *entry += 1;
                true
            }
        });
    }

    // 5. Apply max_records cap.
    materialized.truncate(budget.max_records);

    // 6. Apply token budget by trimming late records / late chunks.
    if let Some(max_tokens) = budget.max_total_tokens {
        let mut used = 0u32;
        let mut out: Vec<PackedRecord> = Vec::new();
        for mut p in materialized {
            if used >= max_tokens {
                break;
            }
            // Keep adding chunks until we'd exceed; one chunk minimum so
            // every retained record always has at least one snippet.
            let mut kept: Vec<RankedChunk> = Vec::new();
            for chunk in p.matched_chunks.drain(..) {
                let cost = estimate_tokens(&chunk.content);
                let first_chunk = kept.is_empty();
                let fits = used.saturating_add(cost) <= max_tokens;
                if first_chunk || fits {
                    kept.push(chunk);
                    used = used.saturating_add(cost);
                } else {
                    break;
                }
            }
            p.matched_chunks = kept;
            out.push(p);
        }
        materialized = out;
    }

    Ok(materialized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use chrono::Utc;

    fn rec(
        adapter: &str,
        instance: Option<&str>,
        native_id: &str,
        content: &str,
    ) -> AnamnesisRecord {
        AnamnesisRecord {
            id: RecordId::from_parts(adapter, instance, native_id),
            source: SourceDescriptor {
                adapter: adapter.into(),
                instance: instance.map(str::to_owned),
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
                native_path: Some(format!("/p/{native_id}")),
                captured_at: Utc::now(),
                raw_hash: "h".into(),
            },
            schema_version: SCHEMA_VERSION,
        }
    }

    fn hit(record_id: &RecordId, seq: u32, content: &str, score: f64) -> RankedChunk {
        RankedChunk {
            chunk_id: format!("{}:{seq}", record_id.0),
            record_id: record_id.clone(),
            seq,
            content: content.into(),
            score,
            from_fts: true,
            from_vec: false,
        }
    }

    fn seed_store_with(records: &[AnamnesisRecord]) -> Store {
        let mut store = Store::open_in_memory().unwrap();
        for r in records {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }
        store
    }

    #[test]
    fn empty_hits_yield_empty_output() {
        let store = Store::open_in_memory().unwrap();
        let out = pack(&store, &[], &ContextBudget::default()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn single_record_aggregates_multiple_chunks() {
        let r = rec("a", None, "x", "alpha alpha alpha");
        let store = seed_store_with(std::slice::from_ref(&r));
        let hits = vec![
            hit(&r.id, 0, "alpha part one", 0.9),
            hit(&r.id, 1, "alpha part two", 0.5),
        ];
        let out = pack(&store, &hits, &ContextBudget::default()).unwrap();
        assert_eq!(out.len(), 1);
        let p = &out[0];
        assert_eq!(p.record.id, r.id);
        assert_eq!(p.matched_chunks.len(), 2);
        // Should be sorted by chunk score, best first.
        assert!(p.matched_chunks[0].score >= p.matched_chunks[1].score);
        assert_eq!(p.score, 0.9);
    }

    #[test]
    fn record_score_is_max_of_chunk_scores() {
        let r = rec("a", None, "x", "x");
        let store = seed_store_with(std::slice::from_ref(&r));
        let hits = vec![hit(&r.id, 0, "x", 0.3), hit(&r.id, 1, "y", 0.8)];
        let out = pack(&store, &hits, &ContextBudget::default()).unwrap();
        assert_eq!(out[0].score, 0.8);
    }

    #[test]
    fn output_sorted_by_record_score_descending() {
        let r1 = rec("a", None, "1", "one");
        let r2 = rec("a", None, "2", "two");
        let r3 = rec("a", None, "3", "three");
        let store = seed_store_with(&[r1.clone(), r2.clone(), r3.clone()]);
        let hits = vec![
            hit(&r3.id, 0, "three", 0.1),
            hit(&r1.id, 0, "one", 0.7),
            hit(&r2.id, 0, "two", 0.4),
        ];
        let out = pack(&store, &hits, &ContextBudget::default()).unwrap();
        let ids: Vec<_> = out.iter().map(|p| p.record.id.clone()).collect();
        assert_eq!(ids, vec![r1.id, r2.id, r3.id]);
    }

    #[test]
    fn vanished_records_silently_skipped() {
        let r1 = rec("a", None, "1", "alpha");
        let store = seed_store_with(std::slice::from_ref(&r1));
        let missing_id = RecordId::from_parts("a", None, "gone");
        let hits = vec![
            hit(&r1.id, 0, "alpha", 0.9),
            hit(&missing_id, 0, "ghost", 0.95),
        ];
        let out = pack(&store, &hits, &ContextBudget::default()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].record.id, r1.id);
    }

    #[test]
    fn max_records_truncates() {
        let r1 = rec("a", None, "1", "one");
        let r2 = rec("a", None, "2", "two");
        let r3 = rec("a", None, "3", "three");
        let store = seed_store_with(&[r1.clone(), r2.clone(), r3.clone()]);
        let hits = vec![
            hit(&r1.id, 0, "one", 0.9),
            hit(&r2.id, 0, "two", 0.8),
            hit(&r3.id, 0, "three", 0.7),
        ];
        let budget = ContextBudget {
            max_records: 2,
            max_total_tokens: None,
            max_per_source: None,
        };
        let out = pack(&store, &hits, &budget).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].record.id, r1.id);
        assert_eq!(out[1].record.id, r2.id);
    }

    #[test]
    fn diversity_cap_limits_per_source() {
        // 4 hits from adapter "a", 1 from adapter "b". max_per_source=2.
        let a1 = rec("a", None, "1", "one");
        let a2 = rec("a", None, "2", "two");
        let a3 = rec("a", None, "3", "three");
        let a4 = rec("a", None, "4", "four");
        let b1 = rec("b", None, "1", "bb");
        let store = seed_store_with(&[a1.clone(), a2.clone(), a3.clone(), a4.clone(), b1.clone()]);
        let hits = vec![
            hit(&a1.id, 0, "one", 0.9),
            hit(&a2.id, 0, "two", 0.85),
            hit(&a3.id, 0, "three", 0.8),
            hit(&a4.id, 0, "four", 0.75),
            hit(&b1.id, 0, "bb", 0.7),
        ];
        let budget = ContextBudget {
            max_records: 10,
            max_total_tokens: None,
            max_per_source: Some(2),
        };
        let out = pack(&store, &hits, &budget).unwrap();
        let by_adapter: std::collections::HashMap<String, usize> = {
            let mut m = std::collections::HashMap::new();
            for p in &out {
                *m.entry(p.record.source.adapter.clone()).or_insert(0) += 1;
            }
            m
        };
        assert_eq!(by_adapter.get("a").copied().unwrap_or(0), 2);
        assert_eq!(by_adapter.get("b").copied().unwrap_or(0), 1);
    }

    #[test]
    fn token_budget_keeps_at_least_one_chunk_per_record() {
        // A single very long chunk in r1, plus a small r2. Budget < r1
        // size: pack must still keep one chunk from r1 to surface the hit.
        let big = "word ".repeat(2000); // ~600 tokens at 4 chars/token
        let r1 = rec("a", None, "1", &big);
        let r2 = rec("a", None, "2", "tiny");
        let store = seed_store_with(&[r1.clone(), r2.clone()]);
        let hits = vec![hit(&r1.id, 0, &big, 0.9), hit(&r2.id, 0, "tiny", 0.8)];
        let budget = ContextBudget {
            max_records: 10,
            max_total_tokens: Some(100),
            max_per_source: None,
        };
        let out = pack(&store, &hits, &budget).unwrap();
        assert!(!out.is_empty());
        assert!(
            !out[0].matched_chunks.is_empty(),
            "first record must keep ≥1 chunk"
        );
    }

    #[test]
    fn provenance_preserved() {
        let r = rec("claude-code", Some("default"), "fbid", "fb");
        let store = seed_store_with(std::slice::from_ref(&r));
        let hits = vec![hit(&r.id, 0, "fb", 0.9)];
        let out = pack(&store, &hits, &ContextBudget::default()).unwrap();
        let p = &out[0];
        assert_eq!(p.record.source.adapter, "claude-code");
        assert_eq!(p.record.source.instance.as_deref(), Some("default"));
        assert_eq!(p.record.provenance.native_id, "fbid");
        assert_eq!(p.record.provenance.native_path.as_deref(), Some("/p/fbid"));
    }
}
