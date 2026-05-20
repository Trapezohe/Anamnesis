-- 0010_vec_record_id_metadata: Round-79 PR-78b vec0 rebuild flag.
--
-- Round 78 (PR-78) added the `user_record_tags` overlay and made
-- it visible to read paths. Round 79 (PR-78b) makes it
-- *queryable* — `search --user-tag <name>` / `search_memories({
-- user_tag })`. The filter must push down through all three
-- retrieval paths at the SQL recall stage:
--
--   * FTS5: `JOIN user_record_tags urt ON urt.record_id = rc.record_id`
--   * BLOB vec fallback: same JOIN
--   * sqlite-vec `vec0`: external joins aren't supported inside
--     KNN; we add `record_id TEXT` as a vec0 *metadata column*
--     and use `record_id IN (SELECT … FROM user_record_tags …)`
--     inside the MATERIALIZED knn CTE.
--
-- The vec0 schema change happens in `vec_ext::ensure_vec_table` —
-- this migration just sets the backfill flag so any existing
-- pre-PR-78b store rebuilds its per-dim `chunk_embeddings_vec_d{N}`
-- tables with the new `record_id` metadata column on next open.
-- Fresh stores don't have a vec0 table yet so the rebuild is a
-- no-op for them.
--
-- Old `chunk_embeddings_vec_d*` tables get dropped before rebuild
-- by `vec_ext::backfill_if_pending`. The BLOB column in
-- `chunk_embeddings` remains the source of truth; vec0 is a
-- rebuildable index.

INSERT OR REPLACE INTO meta(key, value)
VALUES ('chunk_vec_index_backfill_pending', '1');

UPDATE meta SET value = '9' WHERE key = 'schema_version';
