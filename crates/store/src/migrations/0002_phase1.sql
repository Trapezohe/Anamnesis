-- Anamnesis schema v2 — Phase 1: chunk-level RAG.
-- See docs/BLUEPRINT.md §6.6 and §16 for the rationale.
--
-- Changes vs v1:
--   - DROP records_fts + its triggers (chunk-level FTS replaces record-level)
--   - ADD record_chunks (with chunks_fts + sync triggers)
--   - ADD chunk_embeddings (BLOB-backed; sqlite-vec swap-in is a later migration)
--   - ADD embedding_jobs queue
--   - ADD sources registry
--   - ADD raw_artifacts (provenance only — keeps source vectors out of retrieval)
--   - ADD import_errors (per-record failures, so a bad row doesn't abort imports)
--   - BUMP meta('schema_version') 1 → 2

DROP TRIGGER IF EXISTS records_ai;
DROP TRIGGER IF EXISTS records_au;
DROP TRIGGER IF EXISTS records_ad;
DROP TABLE   IF EXISTS records_fts;

-- ──────────────────────────────────────────────────────────────────────────
-- sources: user-registered import sources.
-- (adapter, instance) is the canonical key; instance defaults to '' so the
-- UNIQUE constraint is not bypassed by NULL semantics.
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sources (
    adapter         TEXT    NOT NULL,
    instance        TEXT    NOT NULL DEFAULT '',
    location        TEXT,
    config_json     TEXT,
    added_at        INTEGER NOT NULL,
    last_import_at  INTEGER,
    PRIMARY KEY (adapter, instance)
);

-- ──────────────────────────────────────────────────────────────────────────
-- raw_artifacts: the original payload + (optionally) the source's own vector.
-- The vector lives here PURELY as provenance and is NEVER read by the
-- retrieval path. See BLUEPRINT §6.6.1.
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw_artifacts (
    record_id              TEXT PRIMARY KEY,
    payload_json           TEXT,
    source_embedding       BLOB,
    source_embedding_model TEXT,
    source_embedding_dim   INTEGER,
    captured_at            INTEGER NOT NULL,
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);

-- ──────────────────────────────────────────────────────────────────────────
-- record_chunks: the unit of retrieval indexing.
-- id format: "{record_id}:{seq}" (kept opaque to callers).
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS record_chunks (
    id             TEXT    PRIMARY KEY,
    record_id      TEXT    NOT NULL,
    seq            INTEGER NOT NULL,
    content        TEXT    NOT NULL,
    content_hash   TEXT    NOT NULL,
    token_estimate INTEGER NOT NULL,
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE,
    UNIQUE (record_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_chunks_record       ON record_chunks(record_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON record_chunks(content_hash);

-- Chunk-level FTS5 (external content mode keyed by rowid).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='record_chunks',
    content_rowid='rowid',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON record_chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON record_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON record_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- ──────────────────────────────────────────────────────────────────────────
-- chunk_embeddings: BLOB-backed for Phase 1 (sqlite-vec can be added later
-- without a schema break — the BLOB column becomes the fallback path).
-- (chunk_id, model_id) is the unique key: every chunk × every model.
-- The store layer enforces that all rows for a given chunk share the same
-- dim/model in any one query.
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id     TEXT    NOT NULL,
    model_id     TEXT    NOT NULL,
    content_hash TEXT    NOT NULL,
    dim          INTEGER NOT NULL,
    embedding    BLOB    NOT NULL,    -- f32 little-endian, length = dim * 4
    created_at   INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, model_id),
    FOREIGN KEY (chunk_id) REFERENCES record_chunks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_emb_model_hash ON chunk_embeddings(model_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_emb_model      ON chunk_embeddings(model_id);

-- ──────────────────────────────────────────────────────────────────────────
-- embedding_jobs: queue worked by the background embedder.
-- Status state machine: pending → in_progress → (done | failed)
-- (chunk_id, model_id) UNIQUE so duplicate enqueues are idempotent.
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS embedding_jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id     TEXT    NOT NULL,
    content_hash TEXT    NOT NULL,
    model_id     TEXT    NOT NULL,
    status       TEXT    NOT NULL,    -- 'pending' | 'in_progress' | 'done' | 'failed'
    enqueued_at  INTEGER NOT NULL,
    claimed_at   INTEGER,
    finished_at  INTEGER,
    error        TEXT,
    UNIQUE (chunk_id, model_id),
    FOREIGN KEY (chunk_id) REFERENCES record_chunks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_enqueued ON embedding_jobs(status, enqueued_at);

-- ──────────────────────────────────────────────────────────────────────────
-- import_errors: per-record failures so a single bad row doesn't abort the
-- whole import. `phase` is one of: scan | parse | normalize | chunk | upsert.
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS import_errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    adapter     TEXT    NOT NULL,
    instance    TEXT    NOT NULL DEFAULT '',
    native_id   TEXT,
    native_path TEXT,
    phase       TEXT    NOT NULL,
    error       TEXT    NOT NULL,
    occurred_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_errors_source ON import_errors(adapter, instance, occurred_at DESC);

-- ──────────────────────────────────────────────────────────────────────────
-- Bump schema_version.
-- ──────────────────────────────────────────────────────────────────────────
UPDATE meta SET value = '2' WHERE key = 'schema_version';
