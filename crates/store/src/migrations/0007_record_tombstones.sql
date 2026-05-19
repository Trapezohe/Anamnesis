-- 0007_record_tombstones: Round-72 PR-72a `forget record` foundation.
--
-- The store had nothing to say to a user holding a poisoned memory.
-- They could `anamnesis search` to find it, and they could read raw
-- SQL to delete it, but the next `anamnesis import <source>` would
-- happily resurrect it because Anamnesis derives every record from
-- the source's own files / DB row / API response. The system had no
-- record of "this *used* to exist and the user said never again."
--
-- This migration adds the table that turns "delete" into "forget":
-- a small per-record tombstone, keyed by the same
-- `(adapter, instance, native_id)` natural key the source uses, so
-- subsequent imports can short-circuit before any chunking / embedding
-- work.
--
-- Schema rationale:
--   - `record_id` is the hashed primary key; mirrors `records.id`.
--   - `(adapter, instance, native_id)` is what the importer can see
--     *before* it materialises a record — UNIQUE so a single source
--     entry can only be forgotten once.
--   - `raw_hash` is captured for future "allow only changed content"
--     semantics (a follow-up PR can let the source resurrect the
--     record if the raw payload genuinely changed; this PR is the
--     conservative "never resurrect" baseline).
--   - `native_path`, `reason` are optional context for the operator
--     reviewing what's been forgotten.
--   - `forgotten_at` is unix seconds so `list_forgotten` (future) can
--     sort by recency without a JOIN.
--
-- This PR does NOT touch source data — the upstream Claude Code memory
-- file / mem0 SQLite row / etc. stays where it is. The tombstone only
-- prevents Anamnesis's local store from serving it.

CREATE TABLE IF NOT EXISTS record_tombstones (
    record_id     TEXT    NOT NULL PRIMARY KEY,
    adapter       TEXT    NOT NULL,
    instance      TEXT    NOT NULL DEFAULT '',
    native_id     TEXT    NOT NULL,
    native_path   TEXT,
    raw_hash      TEXT    NOT NULL,
    reason        TEXT,
    forgotten_at  INTEGER NOT NULL,
    UNIQUE(adapter, instance, native_id)
);

CREATE INDEX IF NOT EXISTS idx_record_tombstones_source
    ON record_tombstones(adapter, instance, forgotten_at DESC);

UPDATE meta SET value = '6' WHERE key = 'schema_version';
