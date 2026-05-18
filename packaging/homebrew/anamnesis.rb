# Homebrew formula for Anamnesis.
#
# This is a TEMPLATE — to make `brew install anamnesis` work, the
# operator who maintains the tap needs to:
#
#   1. Create a separate Github repo `homebrew-anamnesis` (any
#      organization that owns the tap).
#   2. Copy this file in as `Formula/anamnesis.rb`.
#   3. After every release, bump `version`, refresh the four `url` +
#      `sha256` pairs, and push.
#   4. Users then run:
#         brew tap Trapezohe/anamnesis
#         brew install anamnesis
#
# The formula intentionally builds nothing locally — it downloads the
# pre-built binaries that `.github/workflows/release.yml` already
# attaches to every Git tag, and verifies them against the `.sha256`
# files the workflow also publishes. Compiling from source would
# require `rust` as a dep and a full ~10-minute cold cargo build on
# the user's machine.
#
# To make a release tap-able:
#
#   tag=v0.0.2; ver=${tag#v}
#   # Linux x86_64
#   curl -sL https://github.com/Trapezohe/Anamnesis/releases/download/$tag/anamnesis-$ver-x86_64-unknown-linux-gnu.tar.gz.sha256
#   # macOS x86_64
#   curl -sL https://github.com/Trapezohe/Anamnesis/releases/download/$tag/anamnesis-$ver-x86_64-apple-darwin.tar.gz.sha256
#   # macOS aarch64
#   curl -sL https://github.com/Trapezohe/Anamnesis/releases/download/$tag/anamnesis-$ver-aarch64-apple-darwin.tar.gz.sha256
#
# Plug each sha256 into the matching `sha256` line below.

class Anamnesis < Formula
  desc "Cross-agent local-first memory infrastructure (CLI + MCP server)"
  homepage "https://github.com/Trapezohe/Anamnesis"
  version "0.0.2"
  license "Apache-2.0"

  # The release workflow builds for these three POSIX targets. Windows
  # users grab the .zip directly; Linux aarch64 is parked.
  on_macos do
    on_arm do
      url "https://github.com/Trapezohe/Anamnesis/releases/download/v#{version}/anamnesis-#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
    on_intel do
      url "https://github.com/Trapezohe/Anamnesis/releases/download/v#{version}/anamnesis-#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/Trapezohe/Anamnesis/releases/download/v#{version}/anamnesis-#{version}-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  def install
    bin.install "anamnesis"
    bin.install "anamnesis-mcp"
    doc.install "README.md"
    pkgshare.install "LICENSE"
  end

  test do
    # `--version` is a stable surface and exercises both binaries
    # without touching the user's data dir.
    assert_match(/anamnesis #{version}/, shell_output("#{bin}/anamnesis --version"))
    assert_match(/anamnesis-mcp #{version}/, shell_output("#{bin}/anamnesis-mcp --version"))
  end
end
