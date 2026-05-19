---
description: Check Anamnesis source health and summarize available memory stores.
---

Use the Anamnesis MCP tools to inspect the current memory setup:

1. Call `doctor` to check registered source health.
2. Call `list_sources` to list source counters, freshness, and active model details.
3. Summarize which sources are healthy, stale, empty, or missing.
4. If no sources are registered, suggest the exact CLI shape:

```bash
anamnesis discover
anamnesis source add <adapter> --path <path>
anamnesis import <adapter>
```

Do not call `import_source` unless the user explicitly asks for an import and admin tools are enabled.
