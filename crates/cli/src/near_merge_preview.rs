//! Deterministic per-group ranking for `dedupe --mode near --merge-preview`.
//!
//! Ranking keys (DESC; first non-tie wins): user_tag_count, effective_at
//! (`updated_at`∨`created_at`), has_native_path, adapter ASC, record_id ASC.
//! The first three are operator-meaningful; the last two are stable tiebreakers.
//!
//! Privacy: reads tag COUNTS only (never tag names), never content / raw_hash /
//! native_path. Output is safe to dump.

use std::collections::HashMap;

use anamnesis_core::model::RecordId;
use anamnesis_store::{NearDuplicateGroup, NearDuplicateRecord};

/// One ranked record. Full set returned so the operator can audit.
#[derive(Debug, Clone)]
pub struct RankedRecord<'a> {
    /// 1-based rank. Rank 1 = keep.
    pub rank: u32,
    /// Source record.
    pub record: &'a NearDuplicateRecord,
    /// Tag count (never tag names).
    pub user_tag_count: u32,
    /// `updated_at` ?? `created_at`, unix seconds.
    pub effective_at: i64,
    /// Mirror for flat rendering.
    pub has_native_path: bool,
    /// `keep` for rank 1, else `forget`.
    pub decision: &'static str,
}

/// Per-group merge preview.
pub struct GroupMergePreview<'a> {
    /// Winner (rank 1).
    pub keep_record_id: RecordId,
    /// Losers, the targets for `forget_record`.
    pub forget_record_ids: Vec<RecordId>,
    /// Full ranking, ordered rank-asc.
    pub ranking: Vec<RankedRecord<'a>>,
}

/// Rank one group. Returns `None` for singletons.
pub fn build_merge_preview<'a>(
    group: &'a NearDuplicateGroup,
    user_tag_counts: &HashMap<String, u32>,
) -> Option<GroupMergePreview<'a>> {
    if group.records.len() < 2 {
        return None;
    }
    // Decorate-sort-undecorate. Negate numeric keys for DESC via natural ASC.
    // Tuple cols: 0..4 are sort keys; 5..8 carry through for the output rows.
    // Aliased to quiet clippy::type_complexity.
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
