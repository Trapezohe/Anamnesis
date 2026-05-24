//! Cross-source near-duplicate detection (read-only, never mutates).
//!
//! Pipeline: tokenize → SimHash 64-bit → LSH banding → re-rank by
//! Hamming ≤ `HAMMING_THRESHOLD` AND Jaccard ≥ `JACCARD_THRESHOLD` →
//! union-find → cross-source filter (default on).
//!
//! Privacy: content, tokens, and SimHash are NEVER returned. Per-record
//! shape mirrors R77 `DuplicateRawHashGroup`; per-group reports
//! `min_similarity` / `max_distance` so callers can rank groups
//! without re-reading bodies.

use std::collections::{HashMap, HashSet};

use anamnesis_core::model::{Kind, RecordId};

use crate::cjk::tokenize_indexing;
use crate::{Result, Store};

/// LSH bands. 16×4 chosen over 4×16: recall ≈100% vs ≈40% at Hamming≤8
/// per `P = 1 − (1 − (1−d/n)^r)^b`. Costs 16 hash inserts/record/scan.
const SIMHASH_BANDS: usize = 16;
/// Bits per band; `SIMHASH_BANDS * SIMHASH_BAND_BITS == 64`.
const SIMHASH_BAND_BITS: u32 = 4;
/// Hamming cutoff (8/64 ≈ 87% similarity); calibrated on paraphrase fixtures.
const HAMMING_THRESHOLD: u32 = 8;
/// Jaccard cutoff. Paraphrases score 0.6-0.75; unrelated < 0.3.
const JACCARD_THRESHOLD: f64 = 0.6;
/// Skip records below this token count to avoid false positives on tiny text.
const MIN_TOKENS: usize = 4;

/// One record inside a near-duplicate group. Privacy-safe: no content / raw_hash / tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NearDuplicateRecord {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Adapter id.
    pub adapter: String,
    /// Instance — empty string for default.
    pub instance: String,
    /// Native id at source.
    pub native_id: String,
    /// `native_path` exists (path itself is never returned here).
    pub has_native_path: bool,
    /// Created at, unix seconds.
    pub created_at: i64,
    /// Updated at, unix seconds.
    pub updated_at: Option<i64>,
}

/// Connected component of near-duplicate records.
/// `min_similarity` is the smallest
/// pairwise Jaccard score that survived re-ranking — useful for
/// sorting groups from "definitely duplicate" to "borderline".
#[derive(Debug, Clone, PartialEq)]
pub struct NearDuplicateGroup {
    /// Members of this near-duplicate component, ordered newest-first.
    pub records: Vec<NearDuplicateRecord>,
    /// Smallest Jaccard similarity observed between any pair in the
    /// group. Range `[JACCARD_THRESHOLD, 1.0]`. Returned to surfaces
    /// so an agent / operator can rank groups by confidence.
    pub min_similarity: f64,
    /// Largest pairwise Hamming distance on the SimHash. Range
    /// `[0, HAMMING_THRESHOLD]`. Exposed for the same reason as
    /// `min_similarity`.
    pub max_distance: u32,
}

/// Filter knobs for the near-duplicate scan.
#[derive(Debug, Clone)]
pub struct NearDuplicateFilter {
    /// Source filter; CSV OR grammar via `parse_csv_filter`. `None` = all.
    pub source: Option<String>,
    /// Instance filter; same grammar, AND-combined with `source`.
    pub instance: Option<String>,
    /// `true` (default) = only groups spanning ≥2 distinct adapters.
    pub require_cross_source: bool,
    /// Group cap, clamped to `[1, MAX_LIMIT]`.
    pub limit: u32,
}

impl Default for NearDuplicateFilter {
    fn default() -> Self {
        Self {
            source: None,
            instance: None,
            require_cross_source: true,
            limit: 20,
        }
    }
}

/// Hard cap on groups per call.
pub const MAX_LIMIT: u32 = 100;

/// In-memory row carried through the pipeline.
struct ScannedRecord {
    record_id: RecordId,
    adapter: String,
    instance: String,
    native_id: String,
    has_native_path: bool,
    created_at: i64,
    updated_at: Option<i64>,
    kind: Kind,
    tokens: HashSet<String>,
    simhash: u64,
}

/// Test-only: SimHash a content string in isolation.
#[cfg(test)]
pub fn debug_simhash(content: &str) -> (u64, HashSet<String>) {
    let tokens = unique_tokens(content);
    (simhash_64(&tokens), tokens)
}

/// Entry point: tokenize → SimHash → LSH → re-rank → union-find → filter.
pub fn list_near_duplicates(
    store: &Store,
    filter: &NearDuplicateFilter,
) -> Result<Vec<NearDuplicateGroup>> {
    let limit = filter.limit.clamp(1, MAX_LIMIT) as usize;

    // 1. Load + tokenize + SimHash every eligible record.
    let scanned = scan_records(store, filter)?;
    if scanned.len() < 2 {
        return Ok(Vec::new());
    }

    // 2. LSH banding → candidate pair set (deduped).
    let mut candidates: HashSet<(usize, usize)> = HashSet::new();
    for band in 0..SIMHASH_BANDS {
        let shift = (band as u32) * SIMHASH_BAND_BITS;
        let mask: u64 = ((1u64 << SIMHASH_BAND_BITS) - 1) << shift;
        let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, r) in scanned.iter().enumerate() {
            let key = r.simhash & mask;
            buckets.entry(key).or_default().push(i);
        }
        for indices in buckets.values() {
            if indices.len() < 2 {
                continue;
            }
            for a_pos in 0..indices.len() {
                for b_pos in (a_pos + 1)..indices.len() {
                    let (i, j) = (indices[a_pos], indices[b_pos]);
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    candidates.insert((lo, hi));
                }
            }
        }
    }

    // 3. Re-rank candidates with Hamming + Jaccard. Survivors fuel
    //    union-find.
    let mut uf = UnionFind::new(scanned.len());
    let mut pair_stats: HashMap<(usize, usize), (f64, u32)> = HashMap::new();
    for (i, j) in &candidates {
        let a = &scanned[*i];
        let b = &scanned[*j];
        let hamming = (a.simhash ^ b.simhash).count_ones();
        if hamming > HAMMING_THRESHOLD {
            continue;
        }
        let jaccard = jaccard_similarity(&a.tokens, &b.tokens);
        if jaccard < JACCARD_THRESHOLD {
            continue;
        }
        uf.union(*i, *j);
        pair_stats.insert((*i, *j), (jaccard, hamming));
    }

    // 4. Collect components.
    let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..scanned.len() {
        let root = uf.find(i);
        buckets.entry(root).or_default().push(i);
    }

    // 5. Build groups, apply cross-source filter, sort.
    let mut groups: Vec<NearDuplicateGroup> = Vec::new();
    for (_, members) in buckets {
        if members.len() < 2 {
            continue;
        }

        // Cross-source check.
        if filter.require_cross_source {
            let adapters: HashSet<&str> = members
                .iter()
                .map(|i| scanned[*i].adapter.as_str())
                .collect();
            if adapters.len() < 2 {
                continue;
            }
        }

        // Aggregate per-group stats from the pair table. A pair only
        // exists in `pair_stats` when both members are in the same
        // component, so we can iterate the cartesian product safely.
        let mut min_sim = 1.0f64;
        let mut max_dist = 0u32;
        for a_pos in 0..members.len() {
            for b_pos in (a_pos + 1)..members.len() {
                let key = if members[a_pos] < members[b_pos] {
                    (members[a_pos], members[b_pos])
                } else {
                    (members[b_pos], members[a_pos])
                };
                if let Some((sim, dist)) = pair_stats.get(&key) {
                    if *sim < min_sim {
                        min_sim = *sim;
                    }
                    if *dist > max_dist {
                        max_dist = *dist;
                    }
                }
            }
        }

        // Map back to the privacy-safe public shape, ordered
        // newest-first inside each group (same convention as R77
        // `DuplicateRawHashGroup`).
        let mut records: Vec<NearDuplicateRecord> = members
            .iter()
            .map(|i| {
                let r = &scanned[*i];
                NearDuplicateRecord {
                    record_id: r.record_id.clone(),
                    adapter: r.adapter.clone(),
                    instance: r.instance.clone(),
                    native_id: r.native_id.clone(),
                    has_native_path: r.has_native_path,
                    created_at: r.created_at,
                    updated_at: r.updated_at,
                }
            })
            .collect();
        records.sort_by_key(|r| std::cmp::Reverse(r.created_at));

        groups.push(NearDuplicateGroup {
            records,
            min_similarity: min_sim,
            max_distance: max_dist,
        });
    }

    // Stable order: largest groups first; ties broken by lower
    // similarity (more "interesting" — borderline matches surface
    // first so an operator's eye lands on them).
    groups.sort_by(|a, b| {
        b.records.len().cmp(&a.records.len()).then_with(|| {
            a.min_similarity
                .partial_cmp(&b.min_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    groups.truncate(limit);
    Ok(groups)
}

/// Load every live record matching the filter, tokenize it, compute
/// its SimHash, and drop the in-memory copies of records that are
/// too short to dedupe meaningfully.
fn scan_records(store: &Store, filter: &NearDuplicateFilter) -> Result<Vec<ScannedRecord>> {
    let sources = anamnesis_core::parse_csv_filter(filter.source.as_deref());
    let instances = anamnesis_core::parse_csv_filter(filter.instance.as_deref());

    let rows = store.list_records_for_near_dedupe()?;
    let mut out: Vec<ScannedRecord> = Vec::with_capacity(rows.len());
    for r in rows {
        if !sources.is_empty() && !sources.iter().any(|s| s == &r.adapter) {
            continue;
        }
        if !instances.is_empty() && !instances.iter().any(|i| i == &r.instance) {
            continue;
        }
        let tokens = unique_tokens(&r.content);
        if tokens.len() < MIN_TOKENS {
            continue;
        }
        let simhash = simhash_64(&tokens);
        out.push(ScannedRecord {
            record_id: r.record_id,
            adapter: r.adapter,
            instance: r.instance,
            native_id: r.native_id,
            has_native_path: r.has_native_path,
            created_at: r.created_at,
            updated_at: r.updated_at,
            kind: r.kind,
            tokens,
            simhash,
        });
    }
    // `kind` is captured but not yet used by v1; kept for future
    // boost / penalty rules (e.g. don't merge `episode` into `fact`).
    let _ = out.iter().any(|r| matches!(r.kind, Kind::Fact));
    Ok(out)
}

/// Reuse the FTS-side tokenizer so near-dedupe sees the same word
/// boundaries the search index does (jieba for CJK, ASCII word-break
/// elsewhere). Lowercase normalisation collapses cross-adapter case
/// drift ("The user" vs "user") which would otherwise inflate the
/// token-set union and depress Jaccard below threshold. Result is
/// the unique lowercased token set; order is irrelevant for SimHash
/// and Jaccard.
fn unique_tokens(content: &str) -> HashSet<String> {
    tokenize_indexing(content)
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .collect()
}

/// 64-bit Charikar SimHash over a token set. Each token contributes
/// its blake3-derived 64-bit hash with ±1 weight per bit. Empty input
/// returns 0 (won't collide with any non-empty record because the
/// MIN_TOKENS filter guards against it upstream).
fn simhash_64(tokens: &HashSet<String>) -> u64 {
    if tokens.is_empty() {
        return 0;
    }
    let mut counts: [i32; 64] = [0; 64];
    for tok in tokens {
        let h = blake3_u64(tok);
        for (i, c) in counts.iter_mut().enumerate() {
            if (h >> i) & 1 == 1 {
                *c = c.saturating_add(1);
            } else {
                *c = c.saturating_sub(1);
            }
        }
    }
    let mut out: u64 = 0;
    for (i, c) in counts.iter().enumerate() {
        if *c > 0 {
            out |= 1u64 << i;
        }
    }
    out
}

/// First 8 bytes of blake3 as a u64 (little-endian). Reusing blake3
/// keeps us in the same hash family the rest of the store uses for
/// `raw_hash` / `content_hash`.
fn blake3_u64(s: &str) -> u64 {
    let h = blake3::hash(s.as_bytes());
    let bytes = h.as_bytes();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Jaccard similarity `|A ∩ B| / |A ∪ B|`. Both empty → 1.0 (a
/// pathological case we never hit thanks to MIN_TOKENS).
fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Tiny union-find with path compression — enough for the component
/// merge step. ~1k records at most in normal operator workflows.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut cur = x;
        while self.parent[cur] != cur {
            self.parent[cur] = self.parent[self.parent[cur]];
            cur = self.parent[cur];
        }
        cur
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
    }
}

/// Minimal projection of `records` columns the near-dedupe scan
/// needs. Carried back from the store via
/// [`Store::list_records_for_near_dedupe`].
pub struct NearDedupeScanRow {
    /// Hashed record id.
    pub record_id: RecordId,
    /// Adapter id.
    pub adapter: String,
    /// Instance discriminator — empty string for default.
    pub instance: String,
    /// Native id at source.
    pub native_id: String,
    /// Whether the record has a native_path (the path itself is
    /// never returned).
    pub has_native_path: bool,
    /// Full record content for tokenization (kept in-memory only;
    /// never crosses a wire boundary).
    pub content: String,
    /// Record kind — captured for future kind-aware boost/penalty.
    pub kind: Kind,
    /// Created at, unix seconds.
    pub created_at: i64,
    /// Updated at, unix seconds.
    pub updated_at: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(words: &[&str]) -> HashSet<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn jaccard_known_values() {
        let a = toks(&["a", "b", "c", "d"]);
        let b = toks(&["a", "b", "c", "d"]);
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < 1e-9);

        let c = toks(&["a", "b", "x", "y"]);
        // |∩| = 2, |∪| = 6 → 1/3
        assert!((jaccard_similarity(&a, &c) - 2.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn simhash_stable_for_same_token_set() {
        let a = toks(&["alpha", "beta", "gamma", "delta"]);
        let b = toks(&["delta", "gamma", "beta", "alpha"]);
        assert_eq!(simhash_64(&a), simhash_64(&b));
    }

    #[test]
    fn simhash_distance_grows_with_token_diff_count() {
        // SimHash is a probabilistic estimator; over a small token
        // set, individual bit shifts dominate. The contract we
        // actually depend on at the algorithm level is monotonic:
        // *more* token differences produce a *larger or equal*
        // expected Hamming distance. Sanity-check that here so a
        // future refactor of the hashing primitive can't silently
        // invert the relationship.
        let a = toks(&[
            "user",
            "prefers",
            "thorough",
            "error",
            "handling",
            "in",
            "rust",
            "code",
            "and",
            "comprehensive",
            "tests",
            "with",
            "real",
            "fixtures",
            "no",
            "mocks",
            "for",
            "the",
            "critical",
            "paths",
        ]);
        let mut b = a.clone();
        b.insert("strongly".into());
        let dist_1 = (simhash_64(&a) ^ simhash_64(&b)).count_ones();

        let mut c = a.clone();
        for w in ["strongly", "very", "much", "indeed", "really", "completely"] {
            c.insert(w.into());
        }
        let dist_many = (simhash_64(&a) ^ simhash_64(&c)).count_ones();

        // Reasonable bound: a single-token diff doesn't dominate
        // a 20-token corpus; should produce small distance.
        assert!(
            dist_1 < 32,
            "1-token diff in a 20-token set should not flip half the hash; got {dist_1}"
        );
        // Many-token diff must produce ≥ single-token diff.
        assert!(
            dist_many >= dist_1,
            "more token differences should not reduce SimHash distance: 1-diff={dist_1}, many-diff={dist_many}"
        );
    }

    #[test]
    fn unique_tokens_dedupes_repeats() {
        let toks = unique_tokens("the user the user the");
        assert_eq!(
            toks,
            ["the", "user"].iter().map(|s| s.to_string()).collect()
        );
    }
}
