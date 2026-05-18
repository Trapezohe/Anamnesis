# Connecting MCP clients to Anamnesis

Every common MCP-aware client (Claude Desktop, Cursor, ghast, Continue,
Windsurf, Zed-mcp, etc.) reads a JSON file with a top-level
`mcpServers` object. Anamnesis ships a one-shot generator so you don't
have to hand-write the snippet:

```bash
# Default: stdio transport — works with every client below.
anamnesis mcp config
```

This prints the minimal pasteable block:

```json
{
  "mcpServers": {
    "anamnesis": {
      "command": "/Users/you/.local/bin/anamnesis",
      "args": ["serve"]
    }
  }
}
```

The `command` path is the **absolute path of the binary that's running
the command** — so the snippet works even when the host client (a GUI
app) doesn't inherit your shell `$PATH`. If you packaged Anamnesis
under a non-standard path (Nix store, Homebrew Cellar, etc.) override
it with `--binary <path>`.

To name the server something other than `anamnesis` in the host
config, pass `--name <name>`. To rename your server in Claude Desktop
to `my-memory`, for instance:

```bash
anamnesis mcp config --name my-memory
```

---

## Claude Desktop

Config file:

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

```bash
# 1. Generate the snippet
anamnesis mcp config > /tmp/anamnesis-mcp.json

# 2a. (clean install) — drop it in place
mv /tmp/anamnesis-mcp.json \
   "$HOME/Library/Application Support/Claude/claude_desktop_config.json"

# 2b. (existing config) — merge with jq
jq -s '.[0] * .[1]' \
   "$HOME/Library/Application Support/Claude/claude_desktop_config.json" \
   /tmp/anamnesis-mcp.json \
   > /tmp/merged.json && \
mv /tmp/merged.json \
   "$HOME/Library/Application Support/Claude/claude_desktop_config.json"

# 3. Quit and re-launch Claude Desktop. The "MCP" gear icon should
#    list 5 tools (search_memories, get_record, list_sources,
#    trace_provenance, doctor) under the `anamnesis` server.
```

---

## Cursor

Config file: `~/.cursor/mcp.json` (per-user) or `<project>/.cursor/mcp.json` (per-project).

The format is identical to Claude Desktop's, so the same snippet works:

```bash
mkdir -p ~/.cursor
anamnesis mcp config > ~/.cursor/mcp.json
# Cursor picks up changes on next launch.
```

To make Anamnesis project-specific, drop the file at `.cursor/mcp.json`
in the project root and commit it (the snippet contains no secrets —
the binary path is the only machine-specific bit).

---

## ghast

Config file: `~/.ghast/mcp.json` (or whatever your local ghast install
points at — check the app's "MCP servers" preferences pane).

ghast follows the same `mcpServers` schema:

```bash
anamnesis mcp config > ~/.ghast/mcp.json
```

After restart, `search_memories` from inside ghast should return hits
across every Anamnesis-registered source (claude-code, codex, mem0,
letta, hermes, openclaw, ghast itself, tdai, openviking, mempalace,
memori, memos, memary, generic-mcp).

---

## Continue.dev

Config file: `~/.continue/config.json`.

Continue supports both the `mcpServers` shape directly and an older
`experimental.modelContextProtocolServer` block. The generator emits
the standard shape; merge it into your `~/.continue/config.json` the
same way as Claude Desktop.

---

## Windsurf, Zed-mcp, and other clients

Anything that speaks the public MCP spec consumes the same `mcpServers`
JSON. If your client doesn't, file an issue — the only client-specific
divergence Anamnesis has seen so far is path-of-config-file.

---

## HTTP / SSE transport

For long-running daemonised servers (and for clients that don't speak
stdio), use SSE mode:

```bash
# 1. Start the server somewhere reachable:
ANAMNESIS_MCP_TOKEN=$(openssl rand -hex 32)
export ANAMNESIS_MCP_TOKEN
anamnesis serve --sse 7878 &

# 2. Generate the matching client config:
anamnesis mcp config --transport sse --sse-port 7878 \
                     --token-env ANAMNESIS_MCP_TOKEN
```

Output:

```json
{
  "mcpServers": {
    "anamnesis": {
      "url": "http://127.0.0.1:7878",
      "headers": {
        "Authorization": "Bearer ${env:ANAMNESIS_MCP_TOKEN}"
      }
    }
  }
}
```

The `${env:NAME}` placeholder is resolved at request time by every
host client that supports SSE transport. The actual token value never
lands in any config file — `anamnesis mcp config` only emits the
env-var name you tell it to.

---

## Verifying the connection

End-to-end smoke test (no GUI needed):

```bash
docs/demo/quickstart.sh
```

The script bootstraps a fresh data dir, seeds a handful of demo
records, starts the MCP server on stdio, and exercises every tool the
host clients above expose. Successful run = green tick at the bottom.
