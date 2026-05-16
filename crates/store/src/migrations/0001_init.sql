-- Anamnesis schema v1 — initial.
-- See docs/BLUEPRINT.md §6.6 for the field-level rationale.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1');

CREATE TABLE IF NOT EXISTS records (
    id              TEXT    PRIMARY KEY,
    adapter         TEXT    NOT NULL,
    instance        TEXT,
    content         TEXT    NOT NULL,
    scope           TEXT    NOT NULL,
    kind            TEXT    NOT NULL,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER,
    tags            TEXT,                    -- JSON array
    metadata        TEXT,                    -- JSON object
    native_id       TEXT    NOT NULL,
    native_path     TEXT,
    captured_at     INTEGER NOT NULL,
    raw_hash        TEXT    NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    UNIQUE(adapter, instance, native_id)
);

CREATE INDEX IF NOT EXISTS idx_records_adapter ON records(adapter, instance);
CREATE INDEX IF NOT EXISTS idx_records_created ON records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_records_kind    ON records(kind);

-- FTS5 over content + tags. Maintained via triggers below.
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
    content,
    tags,
    content='records',
    content_rowid='rowid',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN
    INSERT INTO records_fts(rowid, content, tags)
    VALUES (new.rowid, new.content, COALESCE(new.tags, ''));
END;

CREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON records BEGIN
    INSERT INTO records_fts(records_fts, rowid, content, tags)
    VALUES('delete', old.rowid, old.content, COALESCE(old.tags, ''));
END;

CREATE TRIGGER IF NOT EXISTS records_au AFTER UPDATE ON records BEGIN
    INSERT INTO records_fts(records_fts, rowid, content, tags)
    VALUES('delete', old.rowid, old.content, COALESCE(old.tags, ''));
    INSERT INTO records_fts(rowid, content, tags)
    VALUES (new.rowid, new.content, COALESCE(new.tags, ''));
END;

-- Job log for import runs.
CREATE TABLE IF NOT EXISTS import_jobs (
    id              TEXT    PRIMARY KEY,
    adapter         TEXT    NOT NULL,
    instance        TEXT,
    started_at      INTEGER NOT NULL,
    finished_at     INTEGER,
    status          TEXT    NOT NULL,        -- 'running' | 'done' | 'failed'
    records_seen    INTEGER NOT NULL DEFAULT 0,
    records_added   INTEGER NOT NULL DEFAULT 0,
    records_updated INTEGER NOT NULL DEFAULT 0,
    error           TEXT
);

-- Vector column lives in a separate virtual table (sqlite-vec). Schema
-- creation is deferred to runtime in `store::vec` when the extension
-- is loaded, so binaries built without the extension still work.
