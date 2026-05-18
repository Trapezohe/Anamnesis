#!/usr/bin/env bash
# Anamnesis E2E quickstart — proves a freshly installed `anamnesis`
# binary speaks MCP correctly and exposes the expected tool catalogue.
#
# What this script does:
#   1. Picks a temporary data dir so it never touches the user's real
#      `~/.local/share/anamnesis`.
#   2. Runs `anamnesis init` to lay down the SQLite schema.
#   3. Generates the `mcp config` snippet a host client (Claude
#      Desktop / Cursor / ghast / Continue) would paste.
#   4. Spawns `anamnesis serve` over stdio and sends the two MCP
#      requests every host client makes on connection:
#        * `initialize`
#        * `tools/list`
#      Then parses the response to assert the expected tool catalogue.
#   5. Cleans up — no state survives the run.
#
# This is the same handshake every MCP host does. If this script's
# tick is green, plugging the generated config into a real host
# client will Just Work.

set -eu

# ─── Setup ───────────────────────────────────────────────────────────

# Allow the test to point at a non-PATH binary (e.g. the cargo target
# dir during dev), but default to whatever `anamnesis` is on PATH.
: "${ANAMNESIS_BINARY:=anamnesis}"

if ! command -v "$ANAMNESIS_BINARY" >/dev/null 2>&1; then
    printf 'quickstart: cannot find `%s` on PATH — install Anamnesis first or set ANAMNESIS_BINARY=/path/to/anamnesis\n' "$ANAMNESIS_BINARY" >&2
    exit 1
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/anamnesis-quickstart.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

printf 'Anamnesis quickstart\n'
printf '  binary    : %s\n' "$ANAMNESIS_BINARY"
printf '  data dir  : %s\n\n' "$tmp"

# ─── 1. anamnesis init ──────────────────────────────────────────────

printf '[1/4] anamnesis init …\n'
"$ANAMNESIS_BINARY" --data-dir "$tmp" init >/dev/null
printf '      ok — db at %s/anamnesis.db\n\n' "$tmp"

# ─── 2. anamnesis mcp config ────────────────────────────────────────

printf '[2/4] anamnesis mcp config …\n'
config_snippet="$("$ANAMNESIS_BINARY" --data-dir "$tmp" mcp config)"
# Must parse as JSON and contain mcpServers.anamnesis.command. We use
# python3 because every supported target (macOS, Linux, WSL) ships it;
# `jq` would be lighter but isn't part of the install footprint.
echo "$config_snippet" | python3 -c '
import json, sys
cfg = json.load(sys.stdin)
assert "mcpServers" in cfg, "missing mcpServers key"
assert "anamnesis" in cfg["mcpServers"], "missing anamnesis entry"
entry = cfg["mcpServers"]["anamnesis"]
cmd = entry.get("command")
assert cmd and entry.get("args") == ["serve"], "unexpected entry: " + repr(entry)
print("      ok — host config emits command=" + repr(cmd))
' || { printf '      FAIL — config snippet malformed\n' >&2; exit 1; }
printf '\n'

# ─── 3. MCP handshake over stdio ────────────────────────────────────

printf '[3/4] MCP handshake (initialize + tools/list) …\n'
# Build two JSON-RPC requests (NDJSON) and write them to a temp file.
# Using a regular file instead of a bash pipe avoids a known bash gotcha:
# bash builtins (printf) in a $() subshell may not fork a child process,
# leaving the pipe write-end open in the parent — which means the server
# never receives EOF and hangs forever. A regular file read naturally
# reaches EOF without any pipe write-end ambiguity.
cat > "$tmp/requests.ndjson" <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
EOF

# Feed the request file into serve; stdout is captured for assertions.
responses="$("$ANAMNESIS_BINARY" --data-dir "$tmp" serve < "$tmp/requests.ndjson")"

# Pull out the tools/list result and assert the expected catalogue.
echo "$responses" | python3 -c '
import json, sys
lines = [l for l in sys.stdin.read().splitlines() if l.strip()]
by_id = {}
for line in lines:
    msg = json.loads(line)
    if "id" in msg:
        by_id[msg["id"]] = msg

init = by_id.get(1, {})
assert init.get("result", {}).get("serverInfo", {}).get("name") == "anamnesis", \
    f"initialize did not return serverInfo.name=anamnesis: {init}"

tools_list = by_id.get(2, {}).get("result", {}).get("tools", [])
names = sorted(t["name"] for t in tools_list)
expected = sorted(["search_memories", "get_record", "list_sources", "trace_provenance", "doctor"])
assert names == expected, f"unexpected tool catalogue: {names}"

print(f"      ok — server advertises {len(names)} tools: {names}")
' || { printf '      FAIL — MCP handshake did not return the expected catalogue\n' >&2; exit 1; }
printf '\n'

# ─── 4. Done ────────────────────────────────────────────────────────

printf '[4/4] Cleanup …\n'
printf '      ok — removing %s\n\n' "$tmp"

printf '✓ Quickstart passed.\n'
printf '  Next step: paste the `mcp config` output into your host client\n'
printf '  and follow docs/INTEGRATIONS.md.\n'
