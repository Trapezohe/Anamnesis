//! Curated embedding model registry — see `docs/BLUEPRINT.md §16.7 / §16.8`.
//!
//! Five hand-picked models that cover the size × language × quality
//! matrix users actually face. The registry itself is pure data — no
//! fastembed dep — so non-embedder crates can render `anamnesis model
//! list` without paying ONNX's compile cost.

use serde::{Deserialize, Serialize};

/// One curated model entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CuratedModel {
    /// Stable short key — what users type in CLI / config. Never rename.
    pub key: &'static str,
    /// HuggingFace model id (informational; the local provider maps it
    /// to a `fastembed::EmbeddingModel` enum value internally).
    pub hf_id: &'static str,
    /// Vector dimensionality this model produces.
    pub dim: u16,
    /// Approximate quantized download size in MB. Shown in `model list`.
    pub approx_size_mb: u32,
    /// Free-form language coverage label (e.g. `"100+"`, `"english"`).
    pub languages: &'static str,
    /// One-line description for `model list`.
    pub description: &'static str,
    /// Optional prefix to prepend to query strings (e.g. e5: `"query: "`).
    pub query_prefix: Option<&'static str>,
    /// Optional prefix to prepend to document strings (e.g. e5: `"passage: "`).
    pub doc_prefix: Option<&'static str>,
    /// `true` when this entry runs locally (the only kind in Phase 1).
    pub is_local: bool,
    /// `true` for the out-of-the-box default.
    pub is_default: bool,
}

/// Curated registry. Order = display order in `model list`.
pub const REGISTRY: &[CuratedModel] = &[
    CuratedModel {
        key: "default",
        hf_id: "intfloat/multilingual-e5-small",
        dim: 384,
        approx_size_mb: 120,
        languages: "100+",
        description: "Multilingual e5-small. Out-of-the-box default; balanced size and coverage.",
        query_prefix: Some("query: "),
        doc_prefix: Some("passage: "),
        is_local: true,
        is_default: true,
    },
    CuratedModel {
        key: "tiny",
        hf_id: "sentence-transformers/all-MiniLM-L6-v2",
        dim: 384,
        approx_size_mb: 90,
        languages: "english-focused",
        description: "MiniLM-L6-v2 quantized. Smallest local model; old machines or thin clients.",
        query_prefix: None,
        doc_prefix: None,
        is_local: true,
        is_default: false,
    },
    CuratedModel {
        key: "en",
        hf_id: "BAAI/bge-small-en-v1.5",
        dim: 384,
        approx_size_mb: 130,
        languages: "english",
        description: "English-only. Highest English quality at this size class.",
        query_prefix: Some("Represent this sentence for searching relevant passages: "),
        doc_prefix: None,
        is_local: true,
        is_default: false,
    },
    CuratedModel {
        key: "multi-strong",
        hf_id: "intfloat/multilingual-e5-base",
        dim: 768,
        approx_size_mb: 280,
        languages: "100+",
        description: "Multilingual e5-base. Larger model, stronger CN/EN mixed quality.",
        query_prefix: Some("query: "),
        doc_prefix: Some("passage: "),
        is_local: true,
        is_default: false,
    },
    CuratedModel {
        key: "cloud-voyage",
        hf_id: "voyage-3",
        dim: 1024,
        approx_size_mb: 0,
        languages: "multilingual",
        description:
            "Voyage cloud API. Requires VOYAGE_API_KEY; never called without explicit opt-in.",
        query_prefix: None,
        doc_prefix: None,
        is_local: false,
        is_default: false,
    },
];

/// Look up a curated model by key. Returns `None` if the user supplied an
/// unknown name — callers should render `available` to nudge them.
pub fn by_key(key: &str) -> Option<&'static CuratedModel> {
    REGISTRY.iter().find(|m| m.key == key)
}

/// The default model — what `anamnesis init` picks when no `--model` is
/// supplied. Panics at compile-time discoverability if the registry
/// loses its default entry (it shouldn't).
pub fn default_model() -> &'static CuratedModel {
    REGISTRY
        .iter()
        .find(|m| m.is_default)
        .expect("registry must contain exactly one default")
}

/// All keys, in display order.
pub fn available() -> Vec<&'static str> {
    REGISTRY.iter().map(|m| m.key).collect()
}

/// Local-only subset (excludes cloud providers).
pub fn local_only() -> Vec<&'static CuratedModel> {
    REGISTRY.iter().filter(|m| m.is_local).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_at_least_one_default() {
        let defaults: Vec<_> = REGISTRY.iter().filter(|m| m.is_default).collect();
        assert_eq!(
            defaults.len(),
            1,
            "exactly one curated model must be marked is_default"
        );
    }

    #[test]
    fn default_is_multilingual() {
        let d = default_model();
        assert_eq!(d.key, "default");
        assert_eq!(d.hf_id, "intfloat/multilingual-e5-small");
        assert!(d.is_local, "default must be a local model");
        assert!(d.languages.contains('+') || d.languages.contains("multi"));
    }

    #[test]
    fn by_key_finds_every_entry() {
        for m in REGISTRY {
            let lookup = by_key(m.key).unwrap();
            assert_eq!(lookup.hf_id, m.hf_id);
        }
        assert!(by_key("nonsense").is_none());
    }

    #[test]
    fn keys_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for m in REGISTRY {
            assert!(seen.insert(m.key), "duplicate key: {}", m.key);
        }
    }

    #[test]
    fn available_includes_all_keys() {
        let keys = available();
        assert_eq!(keys.len(), REGISTRY.len());
        for m in REGISTRY {
            assert!(keys.contains(&m.key));
        }
    }

    #[test]
    fn local_only_excludes_cloud() {
        let locals = local_only();
        assert!(locals.iter().all(|m| m.is_local));
        assert!(!locals.iter().any(|m| m.key == "cloud-voyage"));
    }

    #[test]
    fn cloud_voyage_present_but_marked_remote() {
        let v = by_key("cloud-voyage").unwrap();
        assert!(!v.is_local);
        assert_eq!(v.approx_size_mb, 0, "cloud model has no local download");
    }

    #[test]
    fn five_curated_entries() {
        // §16.8: exactly five.
        assert_eq!(REGISTRY.len(), 5);
    }

    #[test]
    fn dims_are_positive() {
        for m in REGISTRY {
            assert!(m.dim > 0, "{} has zero dim", m.key);
        }
    }

    #[test]
    fn registry_serializes_to_json() {
        // Round-trip via owned strings since CuratedModel uses &'static str.
        for m in REGISTRY {
            let s = serde_json::to_string(m).unwrap();
            // Verify the JSON contains the model's key as a property value.
            assert!(s.contains(m.key), "json should embed key {}", m.key);
            assert!(s.contains(m.hf_id), "json should embed hf_id {}", m.hf_id);
        }
    }
}
