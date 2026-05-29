//! Single source of truth for `adapter id → concrete fs adapter`.
//!
//! `import` and `watch` both need to turn a registered source into a live
//! [`MemoryAdapter`]. Each used to carry its own `match` on the adapter id
//! (the comment in `watch.rs` literally said "kept in sync by hand") —
//! adding an adapter meant editing both and they drifted. This factory is
//! now the one place to touch for those two (R158). `doctor`'s health probe
//! still has its own match (build-then-health + a `generic-mcp` `/healthz`
//! special case); migrating it onto this factory is a follow-up.
//!
//! Returns `None` for URL adapters (`generic-mcp`, built from a URL + token
//! by the caller) and for unknown ids — so callers keep their own handling
//! for those two cases.

use std::path::PathBuf;

use anamnesis_core::adapter::MemoryAdapter;

/// Build the boxed fs adapter for `adapter_id` rooted at `location`.
/// `None` ⇒ not an fs adapter (URL/`generic-mcp`) or unknown id.
pub fn build_fs_adapter(
    adapter_id: &str,
    location: PathBuf,
    instance: Option<&str>,
) -> Option<Box<dyn MemoryAdapter>> {
    let adapter: Box<dyn MemoryAdapter> = match adapter_id {
        anamnesis_adapter_claude_code::ADAPTER_ID => {
            Box::new(anamnesis_adapter_claude_code::ClaudeCodeAdapter::new(
                anamnesis_adapter_claude_code::ClaudeCodeConfig {
                    projects_root: location,
                    instance: instance.map(str::to_owned),
                },
            ))
        }
        anamnesis_adapter_mem0::ADAPTER_ID => {
            Box::new(anamnesis_adapter_mem0::sqlite_adapter(location, instance))
        }
        anamnesis_adapter_codex::ADAPTER_ID => {
            Box::new(anamnesis_adapter_codex::codex_adapter(location, instance))
        }
        anamnesis_adapter_letta::ADAPTER_ID => {
            Box::new(anamnesis_adapter_letta::letta_adapter(location, instance))
        }
        anamnesis_adapter_hermes::ADAPTER_ID => {
            Box::new(anamnesis_adapter_hermes::hermes_adapter(location, instance))
        }
        anamnesis_adapter_openclaw::ADAPTER_ID => Box::new(
            anamnesis_adapter_openclaw::openclaw_adapter(location, instance),
        ),
        anamnesis_adapter_tdai::ADAPTER_ID => {
            Box::new(anamnesis_adapter_tdai::tdai_adapter(location, instance))
        }
        anamnesis_adapter_openviking::ADAPTER_ID => Box::new(
            anamnesis_adapter_openviking::openviking_adapter(location, instance),
        ),
        anamnesis_adapter_mempalace::ADAPTER_ID => Box::new(
            anamnesis_adapter_mempalace::mempalace_adapter(location, instance),
        ),
        anamnesis_adapter_memori::ADAPTER_ID => {
            Box::new(anamnesis_adapter_memori::memori_adapter(location, instance))
        }
        anamnesis_adapter_memos::ADAPTER_ID => {
            Box::new(anamnesis_adapter_memos::memos_adapter(location, instance))
        }
        anamnesis_adapter_memary::ADAPTER_ID => {
            Box::new(anamnesis_adapter_memary::memary_adapter(location, instance))
        }
        // generic-mcp is URL-based (built by the caller); unknown ids fall
        // through too. Both are the caller's responsibility.
        _ => return None,
    };
    Some(adapter)
}
