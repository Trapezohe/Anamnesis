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
    /// R149: round-trip export format token an MCP peer can pass to
    /// `export_memories` / `reconcile_export_bucket` / `import_source
    /// --reconcile-export` to write a file the adapter's importer
    /// reads natively. `None` = no round-trip target yet.
    round_trip_export_format: Option<&'static str>,
}

/// Round-trip export token for an adapter, when one exists. Delegates to
/// `anamnesis_export::round_trip_format_for_adapter` — the single source of
/// truth shared with `reconcile-export`.
fn round_trip_format_for(adapter: &str) -> Option<&'static str> {
    anamnesis_export::round_trip_format_for_adapter(adapter).map(|f| f.as_token())
}

/// Compile-time adapter roster — single source of truth.
/// New adapter: edit this list AND [`registered_detectors`].
fn adapter_roster() -> Vec<AdapterCapability> {
    let row = |id: &'static str,
               detectable: bool,
               default_location_hint: Option<&'static str>,
               format: &'static str,
               registration_hint: &'static str| AdapterCapability {
        id,
        detectable,
        default_location_hint,
        format,
        registration_hint,
        round_trip_export_format: round_trip_format_for(id),
    };
    vec![
        row(
            "claude-code",
            true,
            Some("~/.claude/projects/*/memory/*.md"),
            "markdown",
            "anamnesis source add claude-code --path ~/.claude/projects",
        ),
        row(
            "codex",
            true,
            Some("~/.codex/sessions"),
            "jsonl",
            "anamnesis source add codex --path ~/.codex/sessions",
        ),
        row(
            "mem0",
            true,
            Some("~/.mem0/history.db"),
            "sqlite",
            "anamnesis source add mem0 --path ~/.mem0/history.db",
        ),
        row(
            "letta",
            true,
            Some("~/.letta/letta.db"),
            "sqlite",
            "anamnesis source add letta --path ~/.letta/letta.db",
        ),
        row(
            "hermes",
            true,
            Some("~/.hermes/memory"),
            "markdown",
            "anamnesis source add hermes --path <hermes-home>",
        ),
        row(
            "openclaw",
            true,
            Some("~/.openclaw"),
            "json",
            "anamnesis source add openclaw --path <openclaw-home>",
        ),
        row(
            "tdai",
            true,
            Some("~/.tdai"),
            "json",
            "anamnesis source add tdai --path <tdai-home>",
        ),
        row(
            "openviking",
            true,
            Some("~/.openviking"),
            "json",
            "anamnesis source add openviking --path <openviking-home>",
        ),
        row(
            "mempalace",
            true,
            Some("~/.mempalace"),
            "json",
            "anamnesis source add mempalace --path <mempalace-home>",
        ),
        row(
            "memori",
            true,
            Some("~/.memori"),
            "json",
            "anamnesis source add memori --path <memori-home>",
        ),
        row(
            "memos",
            true,
            Some("~/.memos"),
            "json",
            "anamnesis source add memos --path <memos-home>",
        ),
        row(
            "memary",
            true,
            Some("~/.memary"),
            "json",
            "anamnesis source add memary --path <memary-home>",
        ),
        row(
            "generic-mcp",
            false,
            None,
            "mcp-http",
            "anamnesis source add generic-mcp --url <mcp-endpoint>",
        ),
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
    let round_trip_count = roster
        .iter()
        .filter(|a| a.round_trip_export_format.is_some())
        .count();

    let adapter_count = roster.len();
    let detected_count = detected.len();
    let summary = format!(
        "{} adapters compiled in ({} auto-detectable, {} round-trip export targets); \
         {} candidate source(s) detected on this machine.",
        adapter_count, detector_count, round_trip_count, detected_count,
    );

    let adapters: Vec<Value> = roster
        .iter()
        .map(|a| {
            json!({
                "adapter":                  a.id,
                "detectable":               a.detectable,
                "default_location_hint":    a.default_location_hint,
                "format":                   a.format,
                "registration_hint":        a.registration_hint,
                "round_trip_export_format": a.round_trip_export_format,
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
            "adapter_count":    adapter_count,
            "detector_count":   detector_count,
            "detected_count":   detected_count,
            "round_trip_count": round_trip_count,
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
        assert!(summary.contains("4 round-trip export targets"));
        assert!(summary.contains("0 candidate source(s) detected"));
    }

    /// R153: exactly the four round-trip-capable adapters surface a
    /// non-null `round_trip_export_format`; everyone else is `null`.
    #[test]
    fn round_trip_export_format_is_set_only_for_the_round_trip_targets() {
        let roster = adapter_roster();
        let with_target: std::collections::BTreeMap<&str, &str> = roster
            .iter()
            .filter_map(|a| a.round_trip_export_format.map(|f| (a.id, f)))
            .collect();
        assert_eq!(with_target.get("mem0"), Some(&"mem0-sqlite"));
        assert_eq!(with_target.get("letta"), Some(&"letta-sqlite"));
        assert_eq!(with_target.get("memos"), Some(&"memos-dir"));
        assert_eq!(with_target.get("memori"), Some(&"memori-sqlite"));
        assert_eq!(
            with_target.len(),
            4,
            "exactly four round-trip targets today"
        );
    }

    /// Every advertised `round_trip_export_format` must be a token the
    /// shared `anamnesis_export::ExportFormat::parse` accepts — that's
    /// the contract MCP peers rely on when they pipe the value into
    /// `export_memories` / `reconcile_export_bucket`.
    #[test]
    fn round_trip_export_format_tokens_parse_via_anamnesis_export() {
        let roster = adapter_roster();
        for a in roster.iter() {
            if let Some(token) = a.round_trip_export_format {
                anamnesis_export::ExportFormat::parse(token).unwrap_or_else(|e| {
                    panic!(
                        "{}.round_trip_export_format={:?} must parse: {e}",
                        a.id, token
                    )
                });
            }
        }
    }

    /// JSON payload surfaces `round_trip_export_format` per adapter and
    /// `stats.round_trip_count` for quick agent-side discovery.
    #[tokio::test]
    async fn discover_adapters_payload_surfaces_round_trip_capability() {
        let tempdir = tempfile::tempdir().unwrap();
        let payload = build_discover_adapters_payload(Some(tempdir.path())).await;
        assert_eq!(payload["stats"]["round_trip_count"], 4);
        let by_id: std::collections::BTreeMap<String, Value> = payload["adapters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| (a["adapter"].as_str().unwrap().to_owned(), a.clone()))
            .collect();
        assert_eq!(by_id["mem0"]["round_trip_export_format"], "mem0-sqlite");
        assert_eq!(by_id["letta"]["round_trip_export_format"], "letta-sqlite");
        assert_eq!(by_id["memos"]["round_trip_export_format"], "memos-dir");
        assert_eq!(by_id["memori"]["round_trip_export_format"], "memori-sqlite");
        assert!(by_id["claude-code"]["round_trip_export_format"].is_null());
        assert!(by_id["generic-mcp"]["round_trip_export_format"].is_null());
    }
}
