//! Anamnesis adapter for mem0.
//!
//! Two modes (planned, see `docs/BLUEPRINT.md §6.9`):
//!   - `Sqlite { path }` — read the self-hosted mem0 SQLite database directly.
//!   - `Api { base_url, api_key_env }` — call the mem0 REST API.
//!
//! Phase 0: stub only.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::path::PathBuf;

use anamnesis_core::adapter::{HealthStatus, MemoryAdapter, RawRecord, ScanOpts};
use anamnesis_core::error::Result;
use anamnesis_core::model::{AnamnesisRecord, SourceDescriptor};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};

/// Adapter configuration.
#[derive(Debug, Clone)]
pub enum Mem0Config {
    /// Read mem0's self-hosted SQLite store.
    Sqlite {
        /// Path to the SQLite file.
        path: PathBuf,
        /// Instance discriminator.
        instance: Option<String>,
    },
    /// Call the mem0 cloud REST API.
    Api {
        /// API base URL.
        base_url: String,
        /// Environment variable name holding the API key.
        api_key_env: String,
        /// Instance discriminator.
        instance: Option<String>,
    },
}

impl Mem0Config {
    fn instance(&self) -> Option<&str> {
        match self {
            Self::Sqlite { instance, .. } | Self::Api { instance, .. } => instance.as_deref(),
        }
    }
}

/// The adapter.
pub struct Mem0Adapter {
    config: Mem0Config,
}

impl Mem0Adapter {
    /// Build a new adapter from config.
    pub fn new(config: Mem0Config) -> Self {
        Self { config }
    }
}

#[async_trait]
impl MemoryAdapter for Mem0Adapter {
    fn descriptor(&self) -> SourceDescriptor {
        SourceDescriptor {
            adapter: "mem0".into(),
            instance: self.config.instance().map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    fn scan<'a>(&'a self, _opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
        Box::pin(stream::empty())
    }

    fn normalize(&self, _raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
        Ok(Vec::new())
    }

    async fn health(&self) -> HealthStatus {
        match &self.config {
            Mem0Config::Sqlite { path, .. } => HealthStatus {
                ok: path.exists(),
                detail: format!("sqlite path: {}", path.display()),
            },
            Mem0Config::Api {
                base_url,
                api_key_env,
                ..
            } => HealthStatus {
                ok: std::env::var(api_key_env).is_ok(),
                detail: format!("api base: {base_url} (key env: {api_key_env})"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sqlite_mode_descriptor() {
        let a = Mem0Adapter::new(Mem0Config::Sqlite {
            path: "/tmp/x.sqlite".into(),
            instance: Some("self-hosted".into()),
        });
        let d = a.descriptor();
        assert_eq!(d.adapter, "mem0");
        assert_eq!(d.instance.as_deref(), Some("self-hosted"));
    }
}
