---
description: Search Anamnesis for relevant cross-agent memory before work.
argument-hint: <query>
---

Search Anamnesis for memories relevant to this request:

1. Turn the user's request into 2-4 concise noun-phrase queries.
2. Call `search_memories` with `mode: "hybrid"` and a small limit first.
3. Use filters such as `source`, `kind`, `scope`, `since`, or `until` when the user gives enough context.
4. For any memory that will materially affect the answer, call `trace_provenance` with the returned `record_id` or `chunk_id`.
5. Report only memories that are relevant and cite their source/provenance in plain language.

If search returns nothing useful, say that no relevant Anamnesis memory was found and continue from the current context.
