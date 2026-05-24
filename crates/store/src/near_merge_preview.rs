//! Deterministic keep/forget proposal for one near-duplicate group.
//!
//! Ranking keys (DESC, first non-tie wins):
//!   1. `user_tag_count` — operator-curated signal.
//!   2. `effective_at` (`updated_at` ∨ `created_at`) — recency.
//!   3. `has_native_path` — re-import-friendly survives.
//!   4. `adapter` ASC — alphabetical tiebreaker.
//!   5. `record_id` ASC — final tiebreaker.
//!
//! Privacy: never reads `content` / `raw_hash` / `native_path` (just
//! the boolean) / tag names. Output carries tag *counts* only.
//!
//! Preview-only — operator action stays `forget_record`.

use std::collections::HashMap;

use anamnesis_core::model::RecordId;

use crate::semantic_dedupe::{NearDuplicateGroup, NearDuplicateRecord};

/// One row of a per-group ranking.
#[derive(Debug, Clone)]
pub struct RankedRecord<'a> {
    /// 1-based rank. Rank 1 = `keep`.
    pub rank: u32,
    /// Borrowed from the input group so callers reuse fields they already render.
    pub record: &'a NearDuplicateRecord,
    /// User-tag count (no tag names — privacy).
    pub user_tag_count: u32,
    /// `updated_at` when set, else `created_at` (unix seconds).
    pub effective_at: i64,
    /// Mirror of `NearDuplicateRecord.has_native_path`.
    pub has_native_path: bool,
    /// `"keep"` for rank 1, `"forget"` otherwise.
    pub decision: &'static str,
}

/// Merge-preview view of one group.
pub struct GroupMergePreview<'a> {
    /// Rank-1 winner.
    pub keep_record_id: RecordId,
    /// Losers — what `forget_record` would remove.
    pub forget_record_ids: Vec<RecordId>,
    /// Full ranking, rank-asc.
    pub ranking: Vec<RankedRecord<'a>>,
}

/// Build the per-group merge preview. `user_tag_counts` should come from
/// `Store::user_tags_by_ids` (one batched lookup, not N). Returns `None`
/// for groups with fewer than 2 records.
pub fn build_merge_preview<'a>(
    group: &'a NearDuplicateGroup,
    user_tag_counts: &HashMap<String, u32>,
) -> Option<GroupMergePreview<'a>> {
    if group.records.len() < 2 {
        return None;
    }
    // Decorate-sort-undecorate. Sort key columns (ASC sort, so DESC keys
    // are negated): -tag_count, -effective_at, !has_native_path, adapter,
    // record_id. Carried columns: borrowed record + the three render values.
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
        let g = grp(vec![
            mk("a", "mem0", 100, false),
            mk("b", "claude-code", 200, false),
        ]);
        let preview = build_merge_preview(&g, &HashMap::new()).unwrap();
        assert_eq!(preview.keep_record_id.0, "b");
    }

    #[test]
    fn has_native_path_wins_when_recency_ties() {
        let g = grp(vec![
            mk("a", "mem0", 100, false),
            mk("b", "claude-code", 100, true),
        ]);
        let preview = build_merge_preview(&g, &HashMap::new()).unwrap();
        assert_eq!(preview.keep_record_id.0, "b");
    }

    #[test]
    fn adapter_then_record_id_break_remaining_ties() {
        let g = grp(vec![
            mk("z", "mem0", 100, true),
            mk("a", "claude-code", 100, true),
        ]);
        let preview = build_merge_preview(&g, &HashMap::new()).unwrap();
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
