//! Chunks — the unit of retrieval indexing.
//!
//! A `Chunk` is what the chunker produces from an `AnamnesisRecord`. It is
//! what the FTS5 and vector indexes actually contain. Records are the
//! semantic unit returned to callers; chunks are the search unit underneath.
//!
//! Invariants:
//! - `content_hash` is `blake3(content + model-agnostic normalization)`. It
//!   is the cache key paired with `ModelId` in `chunk_embeddings`.
//! - `record_id` ties a chunk back to its parent record.
//! - `seq` is a stable per-record ordering (0-indexed). Re-chunking a
//!   record may produce a different number of chunks with different `seq`
//!   values; consumers should treat `(record_id, seq)` as ephemeral and
//!   `content_hash` as the durable identity for cache purposes.

use serde::{Deserialize, Serialize};

use crate::model::RecordId;

/// Deterministic content fingerprint, used as the embedding cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentHash(pub String);

impl ContentHash {
    /// Compute `blake3(text)` and store as hex. Pre-normalize text upstream
    /// (e.g. trim, collapse whitespace) if you want hashes to be stable
    /// across cosmetic edits.
    pub fn of(text: &str) -> Self {
        Self(blake3::hash(text.as_bytes()).to_hex().to_string())
    }

    /// Borrow as `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A single chunk produced by the chunker.
///
/// `Chunk` is *not* persisted directly — the store decomposes it into
/// `record_chunks` rows (FTS5) and `chunk_embeddings` rows (vec0).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    /// Parent record this chunk belongs to.
    pub record_id: RecordId,
    /// Stable per-record ordering. `0` for single-chunk records.
    pub seq: u32,
    /// The text actually indexed and (optionally) embedded.
    pub content: String,
    /// `blake3(content)` — the embedding cache key.
    pub content_hash: ContentHash,
    /// Rough token estimate (used by the chunker to enforce budgets; not
    /// load-bearing once the chunk is built).
    pub token_estimate: u32,
}

impl Chunk {
    /// Build a chunk, computing `content_hash` from `content`.
    pub fn new(record_id: RecordId, seq: u32, content: String, token_estimate: u32) -> Self {
        let content_hash = ContentHash::of(&content);
        Self {
            record_id,
            seq,
            content,
            content_hash,
            token_estimate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_is_deterministic_and_hex() {
        let a = ContentHash::of("hello");
        let b = ContentHash::of("hello");
        let c = ContentHash::of("world");
        assert_eq!(a, b);
        assert_ne!(a, c);
        // 32-byte blake3 → 64 hex chars
        assert_eq!(a.as_str().len(), 64);
        assert!(a.as_str().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn chunk_new_computes_hash() {
        let rid = RecordId::from_parts("claude-code", None, "x");
        let c = Chunk::new(rid.clone(), 0, "abc".into(), 1);
        assert_eq!(c.record_id, rid);
        assert_eq!(c.seq, 0);
        assert_eq!(c.content, "abc");
        assert_eq!(c.content_hash, ContentHash::of("abc"));
        assert_eq!(c.token_estimate, 1);
    }

    #[test]
    fn chunk_roundtrips_through_json() {
        let rid = RecordId::from_parts("claude-code", None, "x");
        let c = Chunk::new(rid, 2, "payload".into(), 3);
        let s = serde_json::to_string(&c).unwrap();
        let back: Chunk = serde_json::from_str(&s).unwrap();
        assert_eq!(c, back);
    }
}
