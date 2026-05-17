//! Anamnesis config file (`$XDG_CONFIG_HOME/anamnesis/config.toml`).
//!
//! Per BLUEPRINT §6.5. Layering: env > CLI flags > config file > builtin
//! defaults. The struct is intentionally small in Phase 1 — `[embedding]`
//! and `[server]` blocks plus a list of pre-registered `[[sources]]`.
//! New fields can be added with serde `default` so old configs keep
//! parsing.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Top-level config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// Embedding settings.
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    /// Server settings (MCP).
    #[serde(default)]
    pub server: ServerConfig,
    /// Pre-registered sources. CLI `source add` writes here when
    /// `--persist` is set (future flag); for now this is read-only and
    /// just describes what the user wants imported.
    #[serde(default)]
    pub sources: Vec<SourceEntry>,
}

/// Embedding section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Curated model key (default: `"default"`).
    #[serde(default = "default_model_key")]
    pub model: String,
    /// Provider — `"local"` (default) or `"voyage"` once Phase 2.x lands.
    #[serde(default = "default_provider")]
    pub provider: String,
    /// Hint to the embedder for batch size.
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    /// Cache directory override (defaults to `$DATA_DIR/models`).
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: default_model_key(),
            provider: default_provider(),
            batch_size: default_batch_size(),
            cache_dir: None,
        }
    }
}

fn default_model_key() -> String {
    "default".into()
}
fn default_provider() -> String {
    "local".into()
}
fn default_batch_size() -> u32 {
    32
}

/// MCP server section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Allowed MCP clients (Phase 4+ wildcarding); defaults to `["*"]`.
    #[serde(default = "default_allowed_clients")]
    pub allowed_clients: Vec<String>,
    /// Whether the SSE transport requires a bearer token (default true
    /// when SSE is enabled).
    #[serde(default = "default_require_token")]
    pub require_token: bool,
    /// Whether admin tools (`import_source`, etc.) are exposed over MCP.
    ///
    /// Disabled by default — an MCP server is a *memory provider*, not a
    /// remote shell. Tools that mutate state, read arbitrary filesystem
    /// paths, or otherwise step outside read-only memory access live
    /// behind this flag. Set to `true` only if you trust every connected
    /// MCP client. See BLUEPRINT §17.5 PR-A.
    #[serde(default = "default_allow_admin_tools")]
    pub allow_admin_tools: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            allowed_clients: default_allowed_clients(),
            require_token: default_require_token(),
            allow_admin_tools: default_allow_admin_tools(),
        }
    }
}

fn default_allowed_clients() -> Vec<String> {
    vec!["*".into()]
}
fn default_require_token() -> bool {
    true
}
fn default_allow_admin_tools() -> bool {
    false
}

/// One pre-registered source entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceEntry {
    /// Adapter id (`"claude-code"`, `"mem0"`, …).
    pub adapter: String,
    /// Instance discriminator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    /// Filesystem path override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Whether to watch this source for changes (Phase 4 watcher).
    #[serde(default)]
    pub watch: bool,
}

impl Config {
    /// Load from `path`. Missing file → `Default::default()`. Malformed
    /// file → `Err`.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text =
            std::fs::read_to_string(path).map_err(|e| ConfigError::Io(path.to_path_buf(), e))?;
        toml::from_str(&text).map_err(|e| ConfigError::Parse(path.to_path_buf(), e.to_string()))
    }

    /// Default location: `$XDG_CONFIG_HOME/anamnesis/config.toml` or
    /// `~/.config/anamnesis/config.toml` (macOS uses
    /// `~/Library/Application Support/anamnesis/config.toml` to mirror
    /// the data dir).
    pub fn default_path(home: &Path) -> PathBuf {
        if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
            return PathBuf::from(xdg).join("anamnesis").join("config.toml");
        }
        if cfg!(target_os = "macos") {
            home.join("Library/Application Support")
                .join("anamnesis/config.toml")
        } else {
            home.join(".config/anamnesis/config.toml")
        }
    }

    /// Serialize to a TOML string. Used by `anamnesis config init`
    /// (future).
    pub fn to_toml(&self) -> Result<String, ConfigError> {
        toml::to_string_pretty(self).map_err(|e| ConfigError::Serialize(e.to_string()))
    }
}

/// Config file errors.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// IO error reading the config file.
    #[error("read {}: {1}", .0.display())]
    Io(PathBuf, std::io::Error),
    /// TOML parse error.
    #[error("parse {}: {1}", .0.display())]
    Parse(PathBuf, String),
    /// TOML serialize error (round-trip path).
    #[error("serialize: {0}")]
    Serialize(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("anamnesis-config-{nonce}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn missing_file_returns_defaults() {
        let dir = tmp();
        let path = dir.join("nope.toml");
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg, Config::default());
        assert_eq!(cfg.embedding.model, "default");
        assert_eq!(cfg.embedding.batch_size, 32);
    }

    #[test]
    fn partial_file_is_merged_with_defaults() {
        let dir = tmp();
        let path = dir.join("config.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"[embedding]\nmodel = \"en\"\n").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.embedding.model, "en");
        assert_eq!(cfg.embedding.provider, "local"); // default
        assert_eq!(cfg.embedding.batch_size, 32); // default
    }

    #[test]
    fn sources_block_parses() {
        let dir = tmp();
        let path = dir.join("config.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(
            br#"
[[sources]]
adapter = "claude-code"
instance = "default"
path = "/Users/x/.claude/projects"
watch = true

[[sources]]
adapter = "mem0"
path = "/Users/x/.mem0/db.sqlite"
"#,
        )
        .unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.sources.len(), 2);
        assert_eq!(cfg.sources[0].adapter, "claude-code");
        assert_eq!(cfg.sources[0].instance.as_deref(), Some("default"));
        assert!(cfg.sources[0].watch);
        assert!(!cfg.sources[1].watch); // default
    }

    #[test]
    fn malformed_toml_errors() {
        let dir = tmp();
        let path = dir.join("config.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"this is not toml = ===").unwrap();
        let err = Config::load(&path).unwrap_err();
        assert!(format!("{err}").contains("parse"));
    }

    #[test]
    fn default_path_honors_xdg_config_home() {
        // Save + restore env so we don't leak across tests.
        let prev = std::env::var_os("XDG_CONFIG_HOME");
        std::env::set_var("XDG_CONFIG_HOME", "/tmp/xdg-config-test");
        let p = Config::default_path(Path::new("/home/x"));
        assert_eq!(
            p,
            PathBuf::from("/tmp/xdg-config-test/anamnesis/config.toml")
        );
        match prev {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }

    #[test]
    fn roundtrip_default_through_toml() {
        let cfg = Config::default();
        let s = cfg.to_toml().unwrap();
        // Just ensure key blocks render.
        assert!(s.contains("[embedding]"));
        assert!(s.contains("model = \"default\""));
    }
}
