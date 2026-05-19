-- 0009_user_record_tags: Round-78 PR-78 user-tags overlay.
--
-- `records.tags` is **adapter-derived** — every normalizer fills
-- it from source payload, and `write_record` overwrites it on
-- every upsert. That means a user tag added through CLI or MCP
-- would be silently erased on the next `import` of the same
-- source. The overlay lives in its own table so it survives
-- re-import + re-chunking.
--
-- Design choices:
--   * Composite PK `(record_id, tag)` — a tag on a record is a
--     set member, not a multi-row event. Re-adding the same tag
--     is a deliberate no-op (ON CONFLICT DO NOTHING at the
--     write site).
--   * FK ON DELETE CASCADE — when `forget_record` deletes the
--     live `records` row, the user-tags go with it. Reasonable:
--     the user can't tag something that doesn't exist anymore,
--     and `unforget` is "remove suppression," not "resurrect
--     state."
--   * `(tag, record_id)` index for the future PR-78b
--     `--user-tag <name>` search filter — we read by tag there,
--     not by record. Adding the index now keeps the migration
--     count to one PR.
--   * `created_at` only — no `updated_at` because tags don't
--     mutate in place. Remove+re-add is the way to "re-date" a
--     tag if anyone ever needs it.
--
-- Wire boundary:
--   * `RecordHeader.user_tags` (added in the same PR) carries
--     these into search / get_record output.
--   * `AnamnesisRecord.tags` is unchanged and still adapter-
--     derived — never merged with user_tags at the wire to keep
--     provenance unambiguous.

CREATE TABLE IF NOT EXISTS user_record_tags (
    record_id  TEXT    NOT NULL,
    tag        TEXT    NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (record_id, tag),
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_record_tags_tag_record
    ON user_record_tags(tag, record_id);

UPDATE meta SET value = '8' WHERE key = 'schema_version';
