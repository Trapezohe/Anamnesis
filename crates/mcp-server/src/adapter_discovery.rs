//! Round 137 (PR-78bf): MCP `discover_adapters` helper.
//!
//! Builds the static capability roster (every adapter compiled into
//! this MCP binary) plus the dynamic detection pass (every
//! `SourceDetector` that found a candidate on disk). The CLI's
//! `cmd_discover` runs the same `Discovery` orchestrator, but the
//! CLI is text-output and shell-only — an MCP agent that wants to
//! reason about "what adapters does this Anamnesis support, and
//! what's already on this machine?" had no programmatic surface
//! until now.
//!
//! Design constraints:
//!
//! - **Read-only metadata only.** The `Discovery` contract already
//!   forbids reading memory content during detection (paths, glob
//!   counts, DB schema names — never user memory). This module
//!   inherits that contract.
//! - **No new ACL.** This is a *capability discovery* surface and
//!   stays non-admin alongside `dedupe`, `list_conflicts`,
//!   `search_memories`. The action half (registering a source,
//!   importing) is admin-gated through the existing
//!   `import_source` workflow.
//! - **Honour `home_override`.** Test fixtures must be able to
//!   point detectors at a tempdir without touching `$HOME`. We
//!   thread the override into `DetectOpts.home_override` AND into
//!   the per-detector `.with_home()` builders (only the
//!   `home_override`-aware ones — the others ignore the flag and
//!   read `DetectOpts`).

use std::path::Path;

use anamnesis_adapter_claude_code::ClaudeCodeDetector;
use anamnesis_adapter_codex::CodexDetector;
use anamnesis_adapter_hermes::HermesDetector;
use anamnesis_adapter_letta::LettaSqliteDetector;
use anamnesis_adapter_mem0::Mem0SqliteDetector;
use anamnesis_adapter_memary::MemaryDetector;
use anamnesis_adapter_memori::MemoriDetector;
use anamnesis_adapter_memos::MemosDetector;
use anamnesis_adapter_mempalace::MempalaceDetector;
use anamnesis_adapter_openclaw::OpenClawDetector;
use anamnesis_adapter_openviking::OpenVikingDetector;
use anamnesis_adapter_tdai::TdaiDetector;
use anamnesis_core::discovery::{DetectOpts, DetectedSource, Discovery, SourceDetector};
use serde_json::{json, Value};

/// One row in the static capability roster returned alongside the
/// dynamic detection results. Tells an MCP agent "this adapter is
/// compiled in; here's how to point it at data."
///
/// Kept as a private struct (we render directly to `serde_json`) so
/// the wire shape stays in one place — the MCP layer is the source
/// of truth, not a derived `Serialize`.
struct AdapterCapability {
    /// Stable adapter id (matches `SourceDescriptor.adapter`).
    id: &'static str,
    /// Whether this adapter has a `SourceDetector` registered.
    /// `false` means the adapter is usable but requires the
    /// operator to register a source manually (e.g. `generic-mcp`
    /// needs a URL).
    detectable: bool,
    /// Human-readable hint about where this adapter usually finds
    /// its data — e.g. `"~/.claude/projects"`, `"~/.codex"`.
    /// `None` for adapters with no canonical default location.
    default_location_hint: Option<&'static str>,
    /// On-disk / wire format hint (e.g. `"sqlite"`, `"jsonl"`,
    /// `"markdown"`, `"mcp-http"`). Free-form; meant for human
    /// rendering only.
    format: &'static str,
    /// One-line hint for how an agent / operator would register
    /// this adapter. Example: `"anamnesis source add claude-code
    /// --path ~/.claude/projects"`.
    registration_hint: &'static str,
}

/// The full compile-time roster. **Single source of truth** for the
/// adapter catalogue surfaced by `discover_adapters`. Adding a new
/// adapter requires editing this list AND
/// [`registered_detectors`] below.
fn adapter_roster() -> Vec<AdapterCapability> {
    vec![
        AdapterCapability {
            id: "claude-code",
            detectable: true,
            default_location_hint: Some("~/.claude/projects/*/memory/*.md"),
            format: "markdown",
            registration_hint: "anamnesis source add claude-code --path ~/.claude/projects",
        },
        AdapterCapability {
            id: "codex",
            detectable: true,
            default_location_hint: Some("~/.codex/sessions"),
            format: "jsonl",
            registration_hint: "anamnesis source add codex --path ~/.codex/sessions",
        },
        AdapterCapability {
            id: "mem0",
            detectable: true,
            default_location_hint: Some("~/.mem0/history.db"),
            format: "sqlite",
            registration_hint: "anamnesis source add mem0 --path ~/.mem0/history.db",
        },
        AdapterCapability {
            id: "letta",
            detectable: true,
            default_location_hint: Some("~/.letta/letta.db"),
            format: "sqlite",
            registration_hint: "anamnesis source add letta --path ~/.letta/letta.db",
        },
        AdapterCapability {
            id: "hermes",
            detectable: true,
            default_location_hint: Some("~/.hermes/memory"),
            format: "markdown",
            registration_hint: "anamnesis source add hermes --path <hermes-home>",
        },
        AdapterCapability {
            id: "openclaw",
            detectable: true,
            default_location_hint: Some("~/.openclaw"),
            format: "json",
            registration_hint: "anamnesis source add openclaw --path <openclaw-home>",
        },
        AdapterCapability {
            id: "tdai",
            detectable: true,
            default_location_hint: Some("~/.tdai"),
            format: "json",
            registration_hint: "anamnesis source add tdai --path <tdai-home>",
        },
        AdapterCapability {
            id: "openviking",
            detectable: true,
            default_location_hint: Some("~/.openviking"),
            format: "json",
            registration_hint: "anamnesis source add openviking --path <openviking-home>",
        },
        AdapterCapability {
            id: "mempalace",
            detectable: true,
            default_location_hint: Some("~/.mempalace"),
            format: "json",
            registration_hint: "anamnesis source add mempalace --path <mempalace-home>",
        },
        AdapterCapability {
            id: "memori",
            detectable: true,
            default_location_hint: Some("~/.memori"),
            format: "json",
            registration_hint: "anamnesis source add memori --path <memori-home>",
        },
        AdapterCapability {
            id: "memos",
            detectable: true,
            default_location_hint: Some("~/.memos"),
            format: "json",
            registration_hint: "anamnesis source add memos --path <memos-home>",
        },
        AdapterCapability {
            id: "memary",
            detectable: true,
            default_location_hint: Some("~/.memary"),
            format: "json",
            registration_hint: "anamnesis source add memary --path <memary-home>",
        },
        AdapterCapability {
            id: "generic-mcp",
            detectable: false,
            default_location_hint: None,
            format: "mcp-http",
            registration_hint: "anamnesis source add generic-mcp --url <mcp-endpoint>",
        },
    ]
}

/// Build the live detector orchestrator. Mirrors the CLI
/// [`run_all_detectors`] roster — keeping these in sync is the same
/// maintenance burden as the adapter catalogue, but cheaper than a
/// trait-objects-of-detector-factories indirection.
///
/// `home_override`, when supplied, is threaded into the
/// per-detector builder for the adapters that ship `.with_home()`.
/// Adapters that don't expose `.with_home()` already read
/// `DetectOpts.home_override` directly, so passing the same path
/// through `detect_all` covers them.
fn registered_detectors(home_override: Option<&Path>) -> Discovery {
    let mut d = Discovery::new()
        .register(Box::new(ClaudeCodeDetector::new()))
        .register(Box::new(Mem0SqliteDetector::new()))
        .register(Box::new(CodexDetector::new()));
    let with_home = |det: Box<dyn SourceDetector>| -> Box<dyn SourceDetector> { det };
    // The adapters below expose a `.with_home(PathBuf)` builder.
    // We thread the server-supplied override so tests can scope a
    // fresh tempdir without touching the real `$HOME`. (Adapters
    // without a builder read `DetectOpts.home_override`.)
    let h = home_override.map(|p| p.to_path_buf());
    let letta = LettaSqliteDetector::new();
    let letta: Box<dyn SourceDetector> = match h.clone() {
        Some(p) => Box::new(letta.with_home(p)),
        None => Box::new(letta),
    };
    d = d.register(with_home(letta));
    let hermes = HermesDetector::new();
    let hermes: Box<dyn SourceDetector> = match h.clone() {
        Some(p) => Box::new(hermes.with_home(p)),
        None => Box::new(hermes),
    };
    d = d.register(with_home(hermes));
    let openclaw = OpenClawDetector::new();
    let openclaw: Box<dyn SourceDetector> = match h.clone() {
        Some(p) => Box::new(openclaw.with_home(p)),
        None => Box::new(openclaw),
    };
    d = d.register(with_home(openclaw));
    let tdai = TdaiDetector::new();
    let tdai: Box<dyn SourceDetector> = match h.clone() {
        Some(p) => Box::new(tdai.with_home(p)),
        None => Box::new(tdai),
    };
    d = d.register(with_home(tdai));
    let openviking = OpenVikingDetector::new();
    let openviking: Box<dyn SourceDetector> = match h.clone() {
        Some(p) => Box::new(openviking.with_home(p)),
        None => Box::new(openviking),
    };
    d = d.register(with_home(openviking));
    let mempalace = MempalaceDetector::new();
    let mempalace: Box<dyn SourceDetector> = match h.clone() {
        Some(p) => Box::new(mempalace.with_home(p)),
        None => Box::new(mempalace),
    };
    d = d.register(with_home(mempalace));
    let memori = MemoriDetector::new();
    let memori: Box<dyn SourceDetector> = match h.clone() {
        Some(p) => Box::new(memori.with_home(p)),
        None => Box::new(memori),
    };
    d = d.register(with_home(memori));
    let memos = MemosDetector::new();
    let memos: Box<dyn SourceDetector> = match h.clone() {
        Some(p) => Box::new(memos.with_home(p)),
        None => Box::new(memos),
    };
    d = d.register(with_home(memos));
    let memary = MemaryDetector::new();
    let memary: Box<dyn SourceDetector> = match h.clone() {
        Some(p) => Box::new(memary.with_home(p)),
        None => Box::new(memary),
    };
    d.register(with_home(memary))
}

/// Run the full discovery pass and return the structured payload
/// the MCP `discover_adapters` tool emits. `home_override` is
/// supplied by `AnamnesisServer` and threaded through detectors
/// per the contract above.
pub async fn build_discover_adapters_payload(home_override: Option<&Path>) -> Value {
    let roster = adapter_roster();
    let discovery = registered_detectors(home_override);
    let opts = DetectOpts {
        home_override: home_override.map(|p| p.to_path_buf()),
        extra_paths: Vec::new(),
    };
    let detected: Vec<DetectedSource> = discovery.detect_all(&opts).await;
    let detector_count = roster.iter().filter(|a| a.detectable).count();

    let adapter_count = roster.len();
    let detected_count = detected.len();
    let summary = format!(
        "{} adapters compiled in ({} auto-detectable); {} candidate source(s) detected on this machine.",
        adapter_count, detector_count, detected_count,
    );

    let adapters: Vec<Value> = roster
        .iter()
        .map(|a| {
            json!({
                "adapter":               a.id,
                "detectable":            a.detectable,
                "default_location_hint": a.default_location_hint,
                "format":                a.format,
                "registration_hint":     a.registration_hint,
            })
        })
        .collect();

    let detected_json: Vec<Value> = detected
        .iter()
        .map(|s| {
            json!({
                "adapter":           s.adapter,
                "instance":          s.instance,
                "location":          s.location,
                "confidence":        match s.confidence {
                    anamnesis_core::Confidence::High   => "high",
                    anamnesis_core::Confidence::Medium => "medium",
                    anamnesis_core::Confidence::Low    => "low",
                },
                "estimated_records": s.estimated_records,
                "note":              s.note,
            })
        })
        .collect();

    json!({
        "summary":     summary,
        "adapters":    adapters,
        "detected":    detected_json,
        "stats": {
            "adapter_count":  adapter_count,
            "detector_count": detector_count,
            "detected_count": detected_count,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The catalogue must be in sync with the adapter crates listed
    /// in `Cargo.toml`. If you add a new adapter crate, you must
    /// also add it to [`adapter_roster`] (and, if it has a
    /// detector, to [`registered_detectors`]).
    #[test]
    fn adapter_roster_lists_thirteen_adapters() {
        let roster = adapter_roster();
        assert_eq!(roster.len(), 13);
        let ids: std::collections::BTreeSet<&str> = roster.iter().map(|a| a.id).collect();
        for expected in [
            "claude-code",
            "codex",
            "mem0",
            "letta",
            "hermes",
            "openclaw",
            "tdai",
            "openviking",
            "mempalace",
            "memori",
            "memos",
            "memary",
            "generic-mcp",
        ] {
            assert!(
                ids.contains(expected),
                "adapter `{expected}` must be in the roster"
            );
        }
    }

    /// `generic-mcp` is the only non-detectable adapter (it needs a
    /// URL the operator types). Pinning this invariant catches a
    /// future bug where someone marks it `detectable: true` without
    /// also writing a `SourceDetector` for it.
    #[test]
    fn generic_mcp_is_the_only_non_detectable_adapter() {
        let roster = adapter_roster();
        let non_detectable: Vec<&str> = roster
            .iter()
            .filter(|a| !a.detectable)
            .map(|a| a.id)
            .collect();
        assert_eq!(non_detectable, vec!["generic-mcp"]);
    }

    /// Empty `home_override` (a fresh tempdir) returns zero
    /// `detected[]` but the roster + stats are still populated.
    /// The detector_count matches the roster's `detectable` count.
    #[tokio::test]
    async fn discover_adapters_empty_home_returns_capability_roster_with_zero_detections() {
        let tempdir = tempfile::tempdir().unwrap();
        let payload = build_discover_adapters_payload(Some(tempdir.path())).await;
        assert_eq!(payload["stats"]["adapter_count"], 13);
        assert_eq!(payload["stats"]["detector_count"], 12);
        assert_eq!(payload["stats"]["detected_count"], 0);
        let adapters = payload["adapters"].as_array().unwrap();
        assert_eq!(adapters.len(), 13);
        let detected = payload["detected"].as_array().unwrap();
        assert!(
            detected.is_empty(),
            "fresh tempdir has nothing to detect: {detected:?}"
        );
    }

    /// When the `home_override` actually contains a detectable
    /// layout, the detection pass picks it up and the `detected[]`
    /// row mirrors the `DetectedSource` shape.
    #[tokio::test]
    async fn discover_adapters_picks_up_letta_sqlite_under_home_override() {
        let tempdir = tempfile::tempdir().unwrap();
        // Seed a `~/.letta/letta.db` file shape so the Letta
        // detector's path probe fires. We don't need a real schema
        // — the detector reports a finding off path existence; the
        // file just has to exist.
        let letta_dir = tempdir.path().join(".letta");
        std::fs::create_dir_all(&letta_dir).unwrap();
        std::fs::write(letta_dir.join("letta.db"), b"").unwrap();

        let payload = build_discover_adapters_payload(Some(tempdir.path())).await;
        let detected = payload["detected"].as_array().unwrap();
        let adapters_hit: Vec<&str> = detected
            .iter()
            .map(|d| d["adapter"].as_str().unwrap())
            .collect();
        assert!(
            adapters_hit.contains(&"letta"),
            "letta detector must fire under home_override: detected={detected:?}"
        );
    }

    /// The summary string is human-readable and references the
    /// three numbers the MCP client cares about — keeps the
    /// summary line stable for an agent that grepf's the string.
    #[tokio::test]
    async fn discover_adapters_summary_mentions_counts() {
        let tempdir = tempfile::tempdir().unwrap();
        let payload = build_discover_adapters_payload(Some(tempdir.path())).await;
        let summary = payload["summary"].as_str().unwrap();
        assert!(summary.contains("13 adapters"));
        assert!(summary.contains("12 auto-detectable"));
        assert!(summary.contains("0 candidate source(s) detected"));
    }
}
