# Anamnesis packaging

Distribution recipes for `anamnesis` and `anamnesis-mcp`. Each lives
in its own subdirectory; the root [`../install.sh`](../install.sh) is
the canonical "curl | sh" installer and works on every supported
POSIX target.

## What ships today

| Channel | File | Status |
|---|---|---|
| GitHub Releases (binaries) | `../.github/workflows/release.yml` | Working — every `v*` tag triggers a 4-target build, signs each archive with a SHA-256 sidecar, and uploads to a draft release that auto-promotes when all jobs pass |
| POSIX installer | [`../install.sh`](../install.sh) | Working — `uname -sm` → matching tarball, SHA-256 verified before extract |
| Homebrew tap (template) | [`./homebrew/anamnesis.rb`](./homebrew/anamnesis.rb) | Template — drop into a separate `homebrew-anamnesis` repo, refresh sha256 per release |
| crates.io | (run `cargo publish` from each crate dir) | Manual — see "crates.io" below |

## Recommended install paths for users

```bash
# Path 1: POSIX one-liner — picks the right tarball for your machine
curl -fsSL https://raw.githubusercontent.com/Trapezohe/Anamnesis/main/install.sh | sh

# Path 2: Homebrew (once the tap is published)
brew tap Trapezohe/anamnesis
brew install anamnesis

# Path 3: from crates.io (any platform with a Rust toolchain)
cargo install --locked anamnesis-cli anamnesis-mcp-server
```

## crates.io

Anamnesis is split into 17+ workspace crates. Publishing order matters
because internal dependencies must already be on crates.io before the
crates that depend on them resolve.

Dependency layers (publish bottom-up):

1. **Leaf crates** — `anamnesis-core`, `anamnesis-embedder`
2. **Mid-layer** — `anamnesis-store`, `anamnesis-search`, `anamnesis-importer`,
   `anamnesis-extractor`
3. **Adapters** — every `anamnesis-adapter-*` (14 of them, ordered any
   way you like within this layer)
4. **Top binaries** — `anamnesis-cli`, `anamnesis-mcp-server`

Publish one crate, wait ~1 minute for crates.io to index, publish the
next. `cargo publish --dry-run -p <crate>` first to catch errors.

## Homebrew tap maintenance

Every Git tag should bump the formula. After release `v0.0.X`:

```bash
# Inside the homebrew-anamnesis tap repo:
ver=0.0.X
for target in x86_64-unknown-linux-gnu x86_64-apple-darwin aarch64-apple-darwin; do
    echo "$target:"
    curl -sL "https://github.com/Trapezohe/Anamnesis/releases/download/v${ver}/anamnesis-${ver}-${target}.tar.gz.sha256"
done
```

Paste each sha256 into the matching `sha256` line in
`Formula/anamnesis.rb`, bump `version`, commit, push.

## Adding a new platform

The release workflow's matrix is in `.github/workflows/release.yml`
under the `build` job. To add (say) `aarch64-unknown-linux-gnu`:

1. Add the target to the matrix.
2. Add the matching `cross` invocation if cross-compilation is needed.
3. Mirror the target into `detect_target()` in `install.sh`.
4. If Homebrew should pick it up, extend the `on_linux do … on_arm do`
   block in `homebrew/anamnesis.rb`.

The existing four targets cover the high-trust default; everything
else can fall back to `cargo install --locked` from crates.io.
