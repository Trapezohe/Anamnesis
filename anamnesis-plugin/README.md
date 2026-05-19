# Anamnesis Plugin for Claude Code and Codex

Use Anamnesis from your coding agent as a local-first bridge over memory that already exists in other agents and memory frameworks.

Anamnesis does not create a new remote memory account. It reads local or registered sources, normalizes them into one local store, and serves the results through MCP.

## Quick path for agents

If you are an AI agent setting up Anamnesis for the user, do the smallest safe setup:

```bash
anamnesis init
anamnesis discover
anamnesis status
```

Then register only sources that exist on the user's machine. Do not guess paths.

```bash
anamnesis source add codex --path ~/.codex
anamnesis import codex
```

## Step 1: Install Anamnesis

The plugin expects the `anamnesis` binary to be available. From a local source checkout:

```bash
cargo install --path crates/cli
cargo install --path crates/mcp-server
anamnesis init
```

Confirm it works:

```bash
anamnesis status
anamnesis mcp config
```

If `anamnesis` is installed in a non-standard location, export `ANAMNESIS_BIN` before starting Claude Code:

```bash
export ANAMNESIS_BIN="/absolute/path/to/anamnesis"
```

## Step 2: Add memory sources

Discover local sources:

```bash
anamnesis discover
```

Register explicit sources:

```bash
anamnesis source add claude-code --path ~/.claude/projects
anamnesis source add codex --path ~/.codex
anamnesis source add mem0 --path ~/.mem0/db.sqlite
```

Import:

```bash
anamnesis import claude-code
anamnesis import codex
anamnesis import mem0
```

Check health:

```bash
anamnesis doctor
```

## Step 3: Install the plugin

### Claude Code

Add the marketplace:

```text
/plugin marketplace add Trapezohe/Anamnesis
```

Install the plugin:

```text
/plugin install anamnesis@anamnesis-plugins
```

For a large checkout, use sparse installation from the CLI:

```bash
claude plugin marketplace add Trapezohe/Anamnesis --sparse .claude-plugin anamnesis-plugin
claude plugin install anamnesis@anamnesis-plugins
```

Restart Claude Code after installation so the MCP server is spawned.

### Codex

Option A: direct MCP, fastest:

```bash
codex mcp add anamnesis -- anamnesis serve
```

Option B: marketplace plugin:

```bash
codex plugin marketplace add Trapezohe/Anamnesis --sparse .agents/plugins anamnesis-plugin
```

Restart Codex, open the plugin UI, and install Anamnesis from the Anamnesis Plugins marketplace.

Do not combine Option A and Option B in the same Codex profile unless you intentionally want duplicate MCP registrations.

## Verify it works

After restarting the client, ask:

- "List my Anamnesis memory sources"
- "Search Anamnesis for prior decisions about this project"
- "Use Anamnesis to find relevant user preferences before editing"

If no results appear, run:

```bash
anamnesis doctor
anamnesis source list
```

## What's included

| Component | Claude Code marketplace | Codex direct MCP | Codex marketplace |
| --- | :---: | :---: | :---: |
| MCP server | Yes | Yes | Yes |
| Anamnesis memory skill | Yes | No | Yes |
| Slash commands | Yes | No | No |
| Lifecycle hooks | No | No | No |
| Remote account/API key | No | No | No |

- **MCP server**: exposes Anamnesis search, record read, source listing, provenance, and doctor tools.
- **Skill**: teaches the agent when and how to search existing memory without treating Anamnesis as the source of truth.
- **Slash commands**: quick Claude Code prompts for source health and targeted memory search.
- **No hooks by default**: Anamnesis is currently a read-first bridge over existing memory. It should not silently capture or write memories from Claude/Codex sessions.

## Important boundary

ghast AI is a consumer of Anamnesis, not a supported memory source today. Anamnesis can serve imported memory to ghast AI, Claude Code, Codex, and other MCP clients. It does not currently import ghast AI's own encrypted user memory database.

## Updating

When the plugin updates, restart the client so the MCP server handle is recreated:

- Claude Code: run `/restart` or close and reopen the CLI.
- Codex: restart the session.

If you installed Anamnesis from source, update the binary separately:

```bash
git pull
cargo install --path crates/cli
cargo install --path crates/mcp-server
```

## License

Apache-2.0.
