//! `discover_adapters` helper: static capability roster + dynamic detection pass.
//! Inherits the `Discovery` contract — metadata only, never user memory content.
//! Non-admin; the action half (register/import) stays admin-gated elsewhere.
//! `home_override` threads through both `DetectOpts` AND per-detector `.with_home()`.

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

/// One capability row. Private; rendered directly to JSON.
struct AdapterCapability {
    /// Adapter id (`SourceDescriptor.adapter`).
    id: &'static str,
    /// `false` = no detector; operator registers manually (e.g. `generic-mcp`).
    detectable: bool,
    /// Canonical default location hint, or `None`.
    default_location_hint: Option<&'static str>,
    /// On-disk / wire format (`sqlite`, `jsonl`, `markdown`, `mcp-http`).
    format: &'static str,
    /// Suggested registration command.
    registration_hint: &'static str,
}

/// Compile-time adapter roster — single source of truth.
/// New adapter: edit this list AND [`registered_detectors`].
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

/// Build the detector orchestrator. Mirror of CLI `run_all_detectors`.
/// Adapters with `.with_home()` get `home_override` threaded; others
/// read `DetectOpts.home_override` in `build_discover_adapters_payload`.
fn registered_detectors(home_override: Option<&Path>) -> Discovery {
    let mut d = Discovery::new()
        .register(Box::new(ClaudeCodeDetector::new()))
        .register(Box::new(Mem0SqliteDetector::new()))
        .register(Box::new(CodexDetector::new()));
    let with_home = |det: Box<dyn SourceDetector>| -> Box<dyn SourceDetector> { det };
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

/// Run the discovery pass and assemble the `discover_adapters` MCP payload.
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

    /// Catalogue must match adapter crates in Cargo.toml.
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

    /// `generic-mcp` is the only non-detectable adapter (needs explicit URL).
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

    /// Empty home: zero detections but roster + stats still populated.
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

    /// `home_override` with a planted Letta layout fires the detector.
    #[tokio::test]
    async fn discover_adapters_picks_up_letta_sqlite_under_home_override() {
        let tempdir = tempfile::tempdir().unwrap();
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

    /// Summary string is stable for grepping by MCP clients.
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
