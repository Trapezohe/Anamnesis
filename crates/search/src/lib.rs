//! Retrieval composition layer: Hybrid (FTS5 + vector + RRF) search and
//! ContextPacker (record aggregation + provenance).
//!
//! This crate is the only place where FTS hits, vector hits, and provider
//! query embeddings are mixed. Everything that runs queries — CLI,
//! MCP server, ghast — should call into here instead of touching
//! `Store::search_chunks_fts` / `search_chunks_vec` directly.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod eval;
pub mod hybrid;
pub mod packer;

pub use eval::{
    evaluate_query_at, summarize_quality, JudgedQuery, JudgedRecordRef, QualitySummary, QueryEval,
    RankedRecordRef,
};
pub use hybrid::{
    HybridOpts, HybridSearcher, RankedChunk, SearchMode, SearchStageCounts, SearchStageTimings,
    SearchTrace, TracedSearchResult,
};
pub use packer::{pack, ContextBudget, PackedRecord};
