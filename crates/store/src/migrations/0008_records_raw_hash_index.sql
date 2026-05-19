-- 0008_records_raw_hash_index: Round-77 PR-77 dedupe detector index.
--
-- Round 77 adds a read-only "show me duplicate raw_hash groups"
-- report (`anamnesis dedupe` / MCP `dedupe`). The query is
-- `SELECT raw_hash, … FROM records GROUP BY raw_hash HAVING COUNT(*) > 1`,
-- which without an index would full-scan `records` every time —
-- on the 1795-record corpus from earlier dogfooding that's
-- already noticeable; on a 50k+ corpus it's user-visible slow.
--
-- This migration adds the single-column index. Diagnostic-only:
-- no new tables, no behavioural change beyond a faster GROUP BY.

CREATE INDEX IF NOT EXISTS idx_records_raw_hash ON records(raw_hash);

UPDATE meta SET value = '7' WHERE key = 'schema_version';
