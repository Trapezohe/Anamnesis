//! Round 141 (PR-78bj): operator-decision tooling for
//! `dedupe --mode near --merge-preview`.
//!
//! R131 built the near-duplicate detector. R132 surfaced it on
//! CLI and MCP. R141 adds the question that comes next: "which
//! record should I keep?" Returns a deterministic ranking per
//! group with a winner and the `provenance.derived_from` edges a
//! future merge mutation would write. Preview-only — operator
//! action stays `anamnesis forget <record_id>`.
//!
//! ## Ranking heuristic (fixed, explainable)
//!
//! For each pair of records inside a near-duplicate group, compare:
//!
//! 1. **user_tag_count DESC** — operator-curated tags signal
//!    "I care about this one", so a tagged record wins over an
//!    untagged sibling.
//! 2. **effective_at DESC** — `updated_at.unwrap_or(created_at)`.
//!    More recent wins, because an `update` typically reflects
//!    the operator's latest understanding.
//! 3. **has_native_path DESC** — a record with a real upstream
//!    path is easier to refresh on re-import than one without.
//! 4. **adapter ASC** — alphabetical, just to make the ranking
//!    stable when everything above ties.
//! 5. **record_id ASC** — final tiebreaker; never tied.
//!
//! Steps 1-3 are operator-meaningful; steps 4-5 exist purely so
//! two runs of the same fixture always produce the same ranking
//! (no `HashMap` order leakage).
//!
//! ## Privacy
//!
//! The ranker never reads `content`, `raw_hash`, or `native_path`
//! itself — only the count of user tags and the boolean
//! `has_native_path`. The output carries no tag NAMES either
//! (just counts), so an operator running merge-preview won't
//! accidentally leak the operator's tag vocabulary in shared
//! diagnostic dumps.

use std::collections::HashMap;

use anamnesis_core::model::RecordId;
use anamnesis_store::{NearDuplicateGroup, NearDuplicateRecord};

/// One row of the per-group ranking. The full record set is
/// returned (not just the winner) so an operator can audit the
/// proposed decision before running `forget` on the losers.
#[derive(Debug, Clone)]
pub struct RankedRecord<'a> {
    /// 1-based rank within the group. Rank 1 = keep.
    pub rank: u32,
    /// The R131 record being ranked. Borrowed from the group
    /// input so callers can pluck the same fields they already
    /// render.
    pub record: &'a NearDuplicateRecord,
    /// Number of user tags on this record. Counts only, never
    /// the tag names themselves — preserves operator privacy.
    pub user_tag_count: u32,
    /// Unix-seconds canonical "freshness" stamp: `updated_at`
    /// when set, else `created_at`. Same convention search /
    /// list surfaces use for ordering "most recent".
    pub effective_at: i64,
    /// Mirror of `NearDuplicateRecord.has_native_path` for
    /// flat-table-friendly rendering.
    pub has_native_path: bool,
    /// `keep` for the winner (rank 1), `forget` for everyone else.
    pub decision: &'static str,
}

/// Merge-preview view of one near-duplicate group. Composed by
/// [`build_merge_preview`] from the R131 group + a per-id
/// user-tag-count map.
pub struct GroupMergePreview<'a> {
    /// The record id chosen as the winner (rank 1).
    pub keep_record_id: RecordId,
    /// Losers — the records `forget_record` should remove.
    pub forget_record_ids: Vec<RecordId>,
    /// Full ranking, ordered rank-asc. Includes the winner.
    pub ranking: Vec<RankedRecord<'a>>,
}

/// Build the per-group merge preview from R131's `NearDuplicateGroup`
/// and a user-tag-count map (one lookup, batched ahead of time so
/// we don't run N tag queries here). Returns `None` for groups
/// with fewer than 2 records — those don't need a decision.
pub fn build_merge_preview<'a>(
    group: &'a NearDuplicateGroup,
    user_tag_counts: &HashMap<String, u32>,
) -> Option<GroupMergePreview<'a>> {
    if group.records.len() < 2 {
        return None;
    }
    // Decorate-sort-undecorate: build a ranking key per record
    // exactly once. The 5-key tuple is the comparison signature
    // (DESC for the numeric keys → negate so natural ASC sort
    // works). The borrowed `record` and `(tag_count, effective_at,
    // has_native_path)` ride along so the `RankedRecord` rows we
    // emit downstream get them for free.
    //
    // Aliased to quiet clippy::type_complexity. Tuple columns:
    //   0: -user_tag_count                  (DESC)
    //   1: -effective_at                    (DESC)
    //   2: 0 if has_native_path else 1      (DESC)
    //   3: adapter                          (ASC)
    //   4: record_id                        (ASC)
    //   5: borrowed record                  (carry)
    //   6: tag_count                        (carry)
    //   7: effective_at                     (carry)
    //   8: has_native_path                  (carry)
    type DecoratedRow<'a> = (
        i64,
        i64,
        i64,
        String,
        String,
        &'a NearDuplicateRecord,
        u32,
        i64,
        bool,
    );
    let mut decorated: Vec<DecoratedRow<'a>> = group
        .records
        .iter()
        .map(|r| {
            let tag_count = user_tag_counts.get(&r.record_id.0).copied().unwrap_or(0);
            let effective_at = r.updated_at.unwrap_or(r.created_at);
            (
                -(tag_count as i64),
                -effective_at,
                if r.has_native_path { 0 } else { 1 },
                r.adapter.clone(),
                r.record_id.0.clone(),
                r,
                tag_count,
                effective_at,
                r.has_native_path,
            )
        })
        .collect();
    decorated.sort_by(|a, b| (a.0, a.1, a.2, &a.3, &a.4).cmp(&(b.0, b.1, b.2, &b.3, &b.4)));

    let mut ranking: Vec<RankedRecord<'a>> = Vec::with_capacity(decorated.len());
    for (i, row) in decorated.iter().enumerate() {
        let rank = (i as u32) + 1;
        let decision = if rank == 1 { "keep" } else { "forget" };
        ranking.push(RankedRecord {
            rank,
            record: row.5,
            user_tag_count: row.6,
            effective_at: row.7,
            has_native_path: row.8,
            decision,
        });
    }
    let keep_record_id = ranking[0].record.record_id.clone();
    let forget_record_ids: Vec<RecordId> = ranking[1..]
        .iter()
        .map(|r| r.record.record_id.clone())
        .collect();
    Some(GroupMergePreview {
        keep_record_id,
        forget_record_ids,
        ranking,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(record_id: &str, adapter: &str, created_at: i64, has_path: bool) -> NearDuplicateRecord {
        NearDuplicateRecord {
            record_id: RecordId(record_id.into()),
            adapter: adapter.into(),
            instance: String::new(),
            native_id: format!("{adapter}-native"),
            has_native_path: has_path,
            created_at,
            updated_at: None,
        }
    }

    fn grp(records: Vec<NearDuplicateRecord>) -> NearDuplicateGroup {
        NearDuplicateGroup {
            records,
            min_similarity: 0.85,
            max_distance: 4,
        }
    }

    #[test]
    fn build_merge_preview_skips_singleton_groups() {
        let g = grp(vec![mk("a", "mem0", 100, true)]);
        assert!(build_merge_preview(&g, &HashMap::new()).is_none());
    }

    #[test]
    fn user_tag_count_wins_over_recency() {
        // Older record `a` has 2 user tags; newer `b` has 0.
        // Tag count must dominate → `a` keeps.
        let g = grp(vec![
            mk("a", "mem0", 100, false),
            mk("b", "claude-code", 200, false),
        ]);
        let mut tags = HashMap::new();
        tags.insert("a".into(), 2);
        let preview = build_merge_preview(&g, &tags).unwrap();
        assert_eq!(preview.keep_record_id.0, "a");
        assert_eq!(preview.forget_record_ids.len(), 1);
        assert_eq!(preview.forget_record_ids[0].0, "b");
        assert_eq!(preview.ranking[0].decision, "keep");
        assert_eq!(preview.ranking[1].decision, "forget");
    }

    #[test]
    fn recency_wins_when_tag_counts_tie() {
        // Both 0 tags → newer (b, 200) wins.
        let g = grp(vec![
            mk("a", "mem0", 100, false),
            mk("b", "claude-code", 200, false),
        ]);
        let preview = build_merge_preview(&g, &HashMap::new()).unwrap();
        assert_eq!(preview.keep_record_id.0, "b");
    }

    #[test]
    fn has_native_path_wins_when_recency_ties() {
        // Same effective_at; b has a path, a doesn't.
        let g = grp(vec![
            mk("a", "mem0", 100, false),
            mk("b", "claude-code", 100, true),
        ]);
        let preview = build_merge_preview(&g, &HashMap::new()).unwrap();
        assert_eq!(preview.keep_record_id.0, "b");
    }

    #[test]
    fn adapter_then_record_id_break_remaining_ties() {
        // Identical metadata; adapter ASC then id ASC.
        let g = grp(vec![
            mk("z", "mem0", 100, true),
            mk("a", "claude-code", 100, true),
        ]);
        let preview = build_merge_preview(&g, &HashMap::new()).unwrap();
        // claude-code < mem0 alphabetically → claude-code wins.
        assert_eq!(preview.keep_record_id.0, "a");
    }

    #[test]
    fn ranking_returns_full_set_for_audit() {
        let g = grp(vec![
            mk("a", "mem0", 100, false),
            mk("b", "claude-code", 200, false),
            mk("c", "codex", 300, false),
        ]);
        let preview = build_merge_preview(&g, &HashMap::new()).unwrap();
        assert_eq!(preview.ranking.len(), 3);
        assert_eq!(preview.ranking[0].rank, 1);
        assert_eq!(preview.ranking[1].rank, 2);
        assert_eq!(preview.ranking[2].rank, 3);
    }
}
