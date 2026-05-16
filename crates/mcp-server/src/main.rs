//! MCP server entry point.
//!
//! Phase 0: stub binary that prints the planned tool/resource surface and
//! exits. Phase 1 wires in `rmcp` and the real handlers.
//!
//! Planned MCP surface — see `docs/BLUEPRINT.md §6.3`:
//!
//! Tools:
//!   - `search_memories`     cross-source query
//!   - `get_record`          fetch one record by id
//!   - `list_sources`        registered adapters + health
//!   - `import_source`       trigger an import job
//!   - `trace_provenance`    return native path / surrounding records
//!
//! Resources:
//!   - `anamnesis://record/{id}`
//!   - `anamnesis://source/{adapter}`
//!   - `anamnesis://timeline/{date}`

#![forbid(unsafe_code)]

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        schema = anamnesis_core::SCHEMA_VERSION,
        "anamnesis-mcp Phase 0 stub — handlers land in Phase 1",
    );

    eprintln!(
        "anamnesis-mcp {} — Phase 0 stub.\n\
         Planned tools: search_memories, get_record, list_sources, import_source, trace_provenance.\n\
         See docs/BLUEPRINT.md for the full surface.",
        env!("CARGO_PKG_VERSION"),
    );

    Ok(())
}
