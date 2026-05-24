-- Add `derived_from` to record_tombstones so unforget --cascade-derived
-- can BFS the tombstone subtree (R133 forget cascade walks live records;
-- R134 unforget cascade walks tombstones). Nullable (pre-R134 rows + true
-- roots), no FK (parent may be deleted). Partial index excludes NULLs.

ALTER TABLE record_tombstones ADD COLUMN derived_from TEXT;

CREATE INDEX IF NOT EXISTS idx_record_tombstones_derived_from
    ON record_tombstones(derived_from)
    WHERE derived_from IS NOT NULL;

UPDATE meta SET value = '10' WHERE key = 'schema_version';
