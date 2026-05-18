#!/usr/bin/env sh
# Anamnesis installer. POSIX-shell, no external deps beyond `curl`,
# `tar`, `mktemp`, `uname`. Detects platform, downloads the matching
# release archive from GitHub, verifies its SHA-256 checksum, and
# extracts the two binaries (`anamnesis` + `anamnesis-mcp`) into a
# destination directory.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Trapezohe/Anamnesis/main/install.sh | sh
#   # or, pinning a version:
#   curl -fsSL https://raw.githubusercontent.com/Trapezohe/Anamnesis/main/install.sh | ANAMNESIS_VERSION=v0.0.2 sh
#
# Environment overrides:
#   ANAMNESIS_VERSION   — e.g. `v0.0.2`. Default: latest release tag.
#   ANAMNESIS_PREFIX    — install dir. Default: `$HOME/.local/bin`.
#   ANAMNESIS_REPO      — `owner/repo`. Default: `Trapezohe/Anamnesis`.
#
# Local-first contract: this script ONLY talks to GitHub release URLs
# and verifies checksums. No telemetry, no third-party CDN.

set -eu

repo="${ANAMNESIS_REPO:-Trapezohe/Anamnesis}"
prefix="${ANAMNESIS_PREFIX:-$HOME/.local/bin}"
requested_version="${ANAMNESIS_VERSION:-}"

err() {
    printf 'install.sh: %s\n' "$*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || err "missing required command: $1"
}

require_cmd curl
require_cmd tar
require_cmd mktemp
require_cmd uname

# Resolve the latest tag via GitHub API if the caller didn't pin one.
if [ -z "$requested_version" ]; then
    requested_version="$(
        curl -fsSL "https://api.github.com/repos/${repo}/releases/latest" \
            | grep -E '"tag_name":' \
            | head -n 1 \
            | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/'
    )"
    [ -n "$requested_version" ] || err "could not resolve latest release tag"
fi

# Map `uname -sm` to a Rust target triple. The release workflow builds
# exactly four targets; reject everything else with a clear error.
detect_target() {
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os/$arch" in
        Linux/x86_64)    printf 'x86_64-unknown-linux-gnu' ;;
        Linux/aarch64|Linux/arm64)
            # Currently parked in the release workflow; surface a hint.
            err "Linux aarch64 builds are not yet published. \
Track https://github.com/${repo}/issues for status, or build from source: \
\`cargo install --locked anamnesis-cli anamnesis-mcp-server\`."
            ;;
        Darwin/x86_64)   printf 'x86_64-apple-darwin' ;;
        Darwin/arm64)    printf 'aarch64-apple-darwin' ;;
        # Windows: this installer is POSIX-only. PowerShell users
        # should download the .zip from the release page directly.
        *)
            err "unsupported platform: $os/$arch. \
Supported: Linux x86_64, macOS x86_64, macOS aarch64. \
Windows users: grab the .zip from https://github.com/${repo}/releases"
            ;;
    esac
}

target="$(detect_target)"
version_nopfx="${requested_version#v}"
archive_name="anamnesis-${version_nopfx}-${target}.tar.gz"
archive_url="https://github.com/${repo}/releases/download/${requested_version}/${archive_name}"
sha_url="${archive_url}.sha256"

printf '\nAnamnesis installer\n'
printf '  repo     : %s\n' "$repo"
printf '  version  : %s\n' "$requested_version"
printf '  target   : %s\n' "$target"
printf '  prefix   : %s\n' "$prefix"
printf '  archive  : %s\n\n' "$archive_url"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/anamnesis-install.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

printf 'Downloading archive…\n'
curl -fSL "$archive_url" -o "$tmp_dir/$archive_name" || err "archive download failed"

printf 'Downloading checksum…\n'
if curl -fSL "$sha_url" -o "$tmp_dir/$archive_name.sha256"; then
    printf 'Verifying SHA-256…\n'
    expected="$(awk '{print $1}' "$tmp_dir/$archive_name.sha256")"
    actual=""
    if command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "$tmp_dir/$archive_name" | awk '{print $1}')"
    elif command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$tmp_dir/$archive_name" | awk '{print $1}')"
    else
        printf 'WARNING: no shasum/sha256sum available; skipping verification\n' >&2
    fi
    if [ -n "$actual" ] && [ "$actual" != "$expected" ]; then
        err "checksum mismatch: expected $expected, got $actual"
    fi
else
    printf 'WARNING: no .sha256 published for %s; skipping verification\n' "$archive_name" >&2
fi

printf 'Extracting…\n'
tar -xzf "$tmp_dir/$archive_name" -C "$tmp_dir"
staged="$tmp_dir/anamnesis-${version_nopfx}-${target}"
[ -d "$staged" ] || err "archive did not unpack into the expected directory"

mkdir -p "$prefix"
install -m 0755 "$staged/anamnesis"     "$prefix/anamnesis"
install -m 0755 "$staged/anamnesis-mcp" "$prefix/anamnesis-mcp"

printf '\n✓ Installed anamnesis and anamnesis-mcp into %s\n' "$prefix"

# Hint about PATH if the prefix isn't already on it.
case ":$PATH:" in
    *":$prefix:"*) ;;
    *)
        printf '\nNote: %s is not on your PATH. Add this to your shell rc:\n' "$prefix"
        printf '  export PATH="%s:$PATH"\n' "$prefix"
        ;;
esac

printf '\nNext steps:\n'
printf '  anamnesis init                 # first-time setup\n'
printf '  anamnesis discover             # detect installed memory sources\n'
printf '  anamnesis --help               # full CLI reference\n'
