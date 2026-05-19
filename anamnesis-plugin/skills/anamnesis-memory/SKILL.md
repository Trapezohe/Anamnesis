---
name: anamnesis-memory
description: >
  Use Anamnesis MCP tools to retrieve existing cross-agent memory. Search
  deliberately when prior decisions, user preferences, project conventions, or
  source provenance would improve the answer. Treat Anamnesis as a read-first
  bridge over memory produced elsewhere, not as a new memory owner.
---

# Anamnesis Memory Protocol

You have access to Anamnesis through MCP. Anamnesis imports and normalizes memory from existing agents, frameworks, local stores, and registered MCP resources.

Use it to recover context that may already exist outside this session.

## Decide: search or skip

Search Anamnesis when the user:

- asks about previous decisions, conventions, preferences, or project history
- says "we decided", "last time", "remember", "same as before", or similar
- starts non-trivial work in a known project
- asks for debugging, refactoring, architecture, migration, or tool setup help
- needs cross-agent continuity between Codex, Claude Code, mem0, Letta, Hermes, OpenClaw, or another registered source

Skip Anamnesis when:

- the prompt is a trivial acknowledgement or continuation
- the answer is purely syntactic or general knowledge
- you already searched the relevant scope in this turn
- the user is only providing new information and not asking for recall

Empty results are normal. Continue without memory instead of inventing context.

## Search well

Prefer 2-4 targeted `search_memories` calls over one sentence-length query.

Good query shapes:

- `project architecture decisions`
- `test conventions`
- `user coding preferences`
- `auth refactor failures`

Use filters when the scope is obvious:

- `source`: one adapter such as `codex`, `claude-code`, `mem0`, `letta`, `hermes`, `openclaw`
- `kind`: `fact`, `preference`, `feedback`, `reference`, `episode`, `skill`, `unknown`
- `scope`: `user`, `project`, `session`, `ephemeral`
- `since` / `until`: RFC3339 timestamps when recency matters

## Verify provenance

When a memory affects a high-impact decision, call `trace_provenance` with the `record_id` or `chunk_id` returned by `search_memories`. Prefer memories with clear source, native path, timestamps, and content.

Use `get_record` when you need the full normalized record.

Use `list_sources` or `doctor` when:

- search returns nothing but the user expected memories to exist
- you need to know which sources are registered
- you suspect a source is stale or unhealthy

## Safety boundary

Do not call `import_source` unless the user explicitly asks to import memory and the server exposes admin tools. Import is intentionally admin-gated.

Do not claim ghast AI is a supported source. ghast AI can consume Anamnesis through MCP, but Anamnesis does not currently import ghast AI's encrypted user memory database.

Do not write to upstream memory stores. Anamnesis preserves source ownership and serves normalized recall.
