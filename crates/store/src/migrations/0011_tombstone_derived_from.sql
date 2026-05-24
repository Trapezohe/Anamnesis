-- 0011_tombstone_derived_from: Round-134 PR-78bc — make tombstones lineage-aware.
--
-- R133 added `forget --cascade-derived`, which walks
-- `records.derived_from` to tombstone+delete a whole subtree in
-- one transaction. The inverse operation — `unforget
-- --cascade-derived` — needs the same walk, but over tombstoned
-- records, not live ones. Until now `record_tombstones` did NOT
-- persist the parent pointer, so an unforget cascade had no way
-- to find the children it should bring back.
--
-- This migration:
--   * adds nullable `derived_from TEXT` to record_tombstones,
--     mirroring the `records.derived_from` column (R4 PR-4 added
--     the live counterpart).
--   * adds a partial index on it so an `unforget_record_with_options`
--     BFS can fan out cheaply.
--
-- Schema rationale:
--   * Nullable: pre-R134 tombstones don't carry the pointer, and
--     a forget on a true root record has no parent anyway. NULL is
--     the honest signal "no known parent."
--   * No FK to records(id): the parent record may itself be deleted
--     (it's the whole point of forget). Mirrors the live column
--     (R4 `records.derived_from` is also plain TEXT, no FK).
--   * Partial index excludes NULLs so the BFS query
--     `WHERE derived_from = ?1` hits a small, dense index — same
--     pattern as `idx_records_derived_from`.
--
-- Migration safety:
--   * `ALTER TABLE ... ADD COLUMN` with a NULL default is an
--     O(1) metadata change in SQLite, no row rewrites.
--   * Pre-existing tombstones get `derived_from = NULL`. The R134
--     unforget cascade treats those as "no descendants known" and
--     surfaces only the root — operators can still unforget the
--     root individually, just without the cascade speedup.

ALTER TABLE record_tombstones ADD COLUMN derived_from TEXT;

CREATE INDEX IF NOT EXISTS idx_record_tombstones_derived_from
    ON record_tombstones(derived_from)
    WHERE derived_from IS NOT NULL;

UPDATE meta SET value = '10' WHERE key = 'schema_version';
