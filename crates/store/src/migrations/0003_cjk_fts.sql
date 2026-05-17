-- 0003_cjk_fts: CJK-aware FTS5 indexing via application-side tokenization.
--
-- BLUEPRINT §17.5 round-5 / user-flagged: FTS5's `unicode61` tokenizer
-- splits Chinese by codepoint, so every Han character becomes its own
-- token. BM25 on Chinese queries is meaningless. The Anamnesis user
-- runs Claude Code in Mandarin; the agent ecosystem's "memory recall"
-- promise is broken for them until this is fixed.
--
-- Fix: pre-tokenize via jieba in Rust (`crate::cjk::tokenize_indexing`),
-- exposed to SQLite as the scalar function `tokenize_cjk`. The FTS5
-- table itself keeps `tokenize='unicode61'` — the tokens we feed it are
-- already split on spaces, so unicode61 is the right "split on
-- whitespace" tokenizer for the *post-jieba* stream.
--
-- The triggers replace the verbatim-content writes that 0002 set up.
-- Existing rows are reindexed by `Store::reindex_fts_if_pending` on the
-- next `Store::open`, gated by the `chunks_fts_rebuild_pending` flag we
-- set at the bottom of this migration.

DROP TRIGGER IF EXISTS chunks_ai;
DROP TRIGGER IF EXISTS chunks_au;
DROP TRIGGER IF EXISTS chunks_ad;

-- AFTER INSERT: tokenize new content before pushing into the FTS table.
CREATE TRIGGER chunks_ai AFTER INSERT ON record_chunks BEGIN
    INSERT INTO chunks_fts(rowid, content)
    VALUES (new.rowid, tokenize_cjk(new.content));
END;

-- AFTER DELETE: external-content FTS demands the "delete" command to
-- remove the row. We pass the OLD tokenized content so FTS5 can locate
-- the right entry — passing original content would mismatch.
CREATE TRIGGER chunks_ad AFTER DELETE ON record_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES('delete', old.rowid, tokenize_cjk(old.content));
END;

CREATE TRIGGER chunks_au AFTER UPDATE ON record_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES('delete', old.rowid, tokenize_cjk(old.content));
    INSERT INTO chunks_fts(rowid, content)
    VALUES (new.rowid, tokenize_cjk(new.content));
END;

-- Flag: existing rows in record_chunks still have their original (codepoint-
-- tokenised) content sitting in chunks_fts. `Store::open` will detect this
-- flag, re-tokenize everything, then clear it. We don't do the rebuild
-- here because the SQL migration runs before `tokenize_cjk` is guaranteed
-- to be installed on the connection.
INSERT OR REPLACE INTO meta(key, value) VALUES ('chunks_fts_rebuild_pending', '1');

UPDATE meta SET value = '3' WHERE key = 'schema_version';
