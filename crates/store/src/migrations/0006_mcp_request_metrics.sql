-- 0006_mcp_request_metrics: Round-69 MCP request observability.
--
-- Round 67 / 68 cut MCP search latency by 60-73% in micro-benches, but
-- the user has no way to see whether those wins translate at their
-- actual scale — a user running `doctor` from Claude Desktop sees
-- source health and stale flags, not "search_memories p50 = 12 ms,
-- p99 = 41 ms, 0 errors in the last 24h." This migration adds the
-- table that turns request-side latency from "you trust the benchmark"
-- into "you see your own numbers."
--
-- Privacy: this table never stores query text, full args, snippets,
-- result payloads, or anything user-typed. The columns are a strict
-- whitelist of non-PII operational metadata — tool name, ok/error,
-- duration, result_count, and a handful of pre-existing structured
-- arg shapes (mode / source / instance / limit) that the user has
-- already chosen to disclose by passing them. If a future column
-- looks like it might carry user content, don't add it here — add a
-- separate opt-in table.
--
-- Growth: capped at 5000 rows by the writer (DELETE on insert when
-- the row count exceeds the cap). This is roughly 24 hours of
-- traffic at "search every 17 seconds, all day," which matches the
-- 24h default window. Bigger windows are not supported; if you want
-- longer history, run a separate metrics sink.

CREATE TABLE IF NOT EXISTS mcp_request_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    INTEGER NOT NULL,       -- unix seconds at request entry
    tool          TEXT    NOT NULL,       -- tools/call.name
    ok            INTEGER NOT NULL,       -- 1 = success, 0 = error
    duration_ms   INTEGER NOT NULL,       -- wall time, ms
    result_count  INTEGER,                -- search_memories: hits.len, else NULL
    error_kind    TEXT,                   -- short stable token, NULL on success
    -- Non-PII structured args. Whitelisted only.
    mode          TEXT,                   -- search_memories: hybrid / fulltext / vector
    source        TEXT,                   -- search_memories: adapter filter
    instance      TEXT,                   -- search_memories: instance filter
    limit_value   INTEGER                 -- search_memories: limit
);

CREATE INDEX IF NOT EXISTS idx_mcp_metrics_tool_time
    ON mcp_request_metrics(tool, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_mcp_metrics_time
    ON mcp_request_metrics(started_at DESC);

UPDATE meta SET value = '5' WHERE key = 'schema_version';
