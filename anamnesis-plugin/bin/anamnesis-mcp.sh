#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${ANAMNESIS_BIN:-}" ]]; then
  exec "$ANAMNESIS_BIN" serve
fi

if command -v anamnesis >/dev/null 2>&1; then
  exec "$(command -v anamnesis)" serve
fi

for candidate in \
  "$HOME/.cargo/bin/anamnesis" \
  "/opt/homebrew/bin/anamnesis" \
  "/usr/local/bin/anamnesis" \
  "$HOME/.local/bin/anamnesis"
do
  if [[ -x "$candidate" ]]; then
    exec "$candidate" serve
  fi
done

cat >&2 <<'EOF'
Anamnesis binary not found.

Install Anamnesis first, then restart Claude Code:

  cargo install --path crates/cli

If Anamnesis is installed in a custom location, set ANAMNESIS_BIN to the full
path of the anamnesis binary before starting Claude Code.
EOF
exit 127
