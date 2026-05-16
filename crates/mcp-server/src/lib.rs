//! Anamnesis MCP server — JSON-RPC over stdio.
//!
//! Per BLUEPRINT §6.3 we expose 5 tools and 3 resource URI patterns.
//! The protocol layer is a minimal hand-rolled subset of MCP (initialize,
//! tools/list, tools/call, resources/list, resources/read) rather than a
//! full SDK dependency — keeps the binary tiny and the contract obvious.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod protocol;
pub mod server;

pub use server::AnamnesisServer;
