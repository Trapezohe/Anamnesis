//! Filesystem + SQLite scanner for a MemPalace install.
//!
//! MemPalace (`mempalace/mempalace`, AGPLv3 upstream) persists state under
//! `~/.mempalace/`:
//!
//!   * `identity.txt`         — L0 user identity (plain text)
//!   * `palace/chroma.sqlite3` — ChromaDB persistent client root.
//!     Inside it:
//!       - collection `mempalace_drawers`  — mined memory chunks
//!       - collection `mempalace_closets`  — searchable index summaries
//!
//! Each "drawer" is one mined chunk of content stored in Chroma. Chroma's
//! persistent schema (since 0.4.x):
//!
//! ```text
//! collections(id, name, dimension, ...)
//! segments(id, collection, scope, ...)
//! embeddings(id, segment_id, embedding_id, seq_id, created_at)
//! embedding_metadata(id, key, string_value, int_value, float_value, bool_value)
//!   ↳ the document text lives under key='chroma:document'
//!   ↳ user metadata (wing, room, source_file, …) lives under each key
//! ```
//!
//! Per §-1.2.2 the adapter is read-only.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::Utc;
use rusqlite::{params, Connection, OpenFlags};

/// MemPalace registry-sentinel rooms — drawers with `room == _registry` are
/// internal bookkeeping (one per file telling MemPalace "already mined this
/// file"). Skip them; they're not user-facing memory.
const REGISTRY_ROOM: &str = "_registry";

/// The Chroma metadata key under which the document body is stored on every
/// embedding row.
const CHROMA_DOC_KEY: &str = "chroma:document";

/// Default collection names MemPalace creates inside Chroma.
const COLLECTION_DRAWERS: &str = "mempalace_drawers";
const COLLECTION_CLOSETS: &str = "mempalace_closets";

/// Identity file (Layer 0).
#[derive(Debug, Clone)]
pub struct MempalaceIdentity {
    /// Absolute path (`~/.mempalace/identity.txt`).
    pub path: PathBuf,
    /// File body.
    pub content: String,
    /// File mtime in unix seconds.
    pub mtime_unix: Option<i64>,
}

/// One drawer (or closet) row pulled from ChromaDB.
#[derive(Debug, Clone)]
pub struct MempalaceDrawer {
    /// Chroma collection this drawer belongs to (`mempalace_drawers` or
    /// `mempalace_closets`).
    pub collection_name: String,
    /// User-facing drawer id (e.g. `drawer_default_general_<hash>`).
    pub embedding_id: String,
    /// Document body (the chunk content).
    pub content: String,
    /// User-supplied metadata (`wing`, `room`, `source_file`, etc.) — already
    /// JSON-encoded as an object so the normalizer can pluck what it needs.
    pub metadata: serde_json::Value,
    /// Drawer creation time (unix seconds) if Chroma surfaces it; otherwise
    /// `metadata.filed_at` is the next-best source.
    pub created_unix: Option<i64>,
}

/// Aggregate scan result.
#[derive(Debug, Default)]
pub struct MempalaceScan {
    /// L0 identity record (zero or one).
    pub identities: Vec<MempalaceIdentity>,
    /// All drawers/closets (registry sentinels filtered out).
    pub drawers: Vec<MempalaceDrawer>,
    /// Diagnostic note: set when the Chroma DB exists but couldn't be read
    /// (locked, schema mismatch, etc.). `health()` surfaces this.
    pub chroma_error: Option<String>,
}

impl MempalaceScan {
    /// Total record count.
    pub fn total(&self) -> usize {
        self.identities.len() + self.drawers.len()
    }
}

/// Walk a MemPalace home dir (default `~/.mempalace/`).
pub fn scan_mempalace(home_dir: &Path) -> MempalaceScan {
    let mut scan = MempalaceScan::default();
    if !home_dir.is_dir() {
        return scan;
    }

    // L0 identity.
    let identity_path = home_dir.join("identity.txt");
    if identity_path.is_file() {
        if let Some(content) = read_text(&identity_path) {
            scan.identities.push(MempalaceIdentity {
                mtime_unix: file_mtime_unix(&identity_path),
                path: identity_path,
                content,
            });
        }
    }

    // Palace (ChromaDB).
    let chroma_db = home_dir.join("palace").join("chroma.sqlite3");
    if chroma_db.is_file() {
        match read_chroma(&chroma_db) {
            Ok(drawers) => scan.drawers = drawers,
            Err(e) => {
                tracing::warn!(
                    path = %chroma_db.display(),
                    error = %e,
                    "mempalace scanner: chroma DB unreadable; skipping"
                );
                scan.chroma_error = Some(e);
            }
        }
    }

    scan
}

fn read_chroma(db_path: &Path) -> Result<Vec<MempalaceDrawer>, String> {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("open: {e}"))?;

    // Schema probe: bail with a clear diagnostic if the DB isn't a Chroma
    // persistent client we recognize.
    if !has_table(&conn, "collections") || !has_table(&conn, "embeddings") {
        return Err("chroma schema missing required tables (collections, embeddings)".into());
    }
    if !has_table(&conn, "embedding_metadata") {
        return Err("chroma schema missing embedding_metadata table".into());
    }

    // 1. Find the (id, name) of our two target collections (if present).
    let collections = list_target_collections(&conn)?;
    if collections.is_empty() {
        // Not a MemPalace palace — that's a Confidence::Low for the detector
        // but it isn't an error.
        return Ok(vec![]);
    }

    // 2. Pull per-collection rows.
    let mut out = Vec::new();
    let has_segments = has_table(&conn, "segments");
    for (collection_id, collection_name) in collections {
        let rows = if has_segments {
            // Chroma 0.4+: embeddings → segments(collection) → collections(id).
            fetch_drawers_via_segments(&conn, &collection_id, &collection_name)?
        } else {
            // Older / unusual layouts: embeddings has direct collection_id.
            fetch_drawers_direct(&conn, &collection_id, &collection_name)?
        };
        out.extend(rows);
    }
    Ok(out)
}

fn list_target_collections(conn: &Connection) -> Result<Vec<(String, String)>, String> {
    let mut stmt = conn
        .prepare("SELECT id, name FROM collections WHERE name IN (?, ?)")
        .map_err(|e| format!("collections prepare: {e}"))?;
    let mut rows = stmt
        .query(params![COLLECTION_DRAWERS, COLLECTION_CLOSETS])
        .map_err(|e| format!("collections query: {e}"))?;
    let mut out = Vec::new();
    while let Some(r) = rows.next().map_err(|e| format!("collections next: {e}"))? {
        let id: String = r.get(0).map_err(|e| format!("collections.id: {e}"))?;
        let name: String = r.get(1).map_err(|e| format!("collections.name: {e}"))?;
        out.push((id, name));
    }
    Ok(out)
}

fn fetch_drawers_via_segments(
    conn: &Connection,
    collection_id: &str,
    collection_name: &str,
) -> Result<Vec<MempalaceDrawer>, String> {
    // Chroma writes per-collection segments; an embedding's segment_id points
    // into `segments.id`. The METADATA-scope segment is what carries the
    // document/metadata rows we want.
    let mut stmt = conn
        .prepare(
            "SELECT e.id, e.embedding_id, e.created_at \
             FROM embeddings e \
             JOIN segments s ON s.id = e.segment_id \
             WHERE s.collection = ?",
        )
        .map_err(|e| format!("embeddings prepare: {e}"))?;
    let mut rows = stmt
        .query(params![collection_id])
        .map_err(|e| format!("embeddings query: {e}"))?;

    let mut drawers = Vec::new();
    while let Some(r) = rows.next().map_err(|e| format!("embeddings next: {e}"))? {
        let internal_id: i64 = r.get(0).map_err(|e| format!("embeddings.id: {e}"))?;
        let embedding_id: String = r
            .get(1)
            .map_err(|e| format!("embeddings.embedding_id: {e}"))?;
        let created_at: Option<i64> = r.get(2).ok();

        if let Some(drawer) =
            hydrate_drawer(conn, internal_id, embedding_id, created_at, collection_name)?
        {
            drawers.push(drawer);
        }
    }
    Ok(drawers)
}

fn fetch_drawers_direct(
    conn: &Connection,
    collection_id: &str,
    collection_name: &str,
) -> Result<Vec<MempalaceDrawer>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, embedding_id, created_at \
             FROM embeddings \
             WHERE collection_id = ?",
        )
        .map_err(|e| format!("embeddings(direct) prepare: {e}"))?;
    let mut rows = stmt
        .query(params![collection_id])
        .map_err(|e| format!("embeddings(direct) query: {e}"))?;

    let mut drawers = Vec::new();
    while let Some(r) = rows
        .next()
        .map_err(|e| format!("embeddings(direct) next: {e}"))?
    {
        let internal_id: i64 = r.get(0).map_err(|e| format!("embeddings.id: {e}"))?;
        let embedding_id: String = r
            .get(1)
            .map_err(|e| format!("embeddings.embedding_id: {e}"))?;
        let created_at: Option<i64> = r.get(2).ok();

        if let Some(drawer) =
            hydrate_drawer(conn, internal_id, embedding_id, created_at, collection_name)?
        {
            drawers.push(drawer);
        }
    }
    Ok(drawers)
}

fn hydrate_drawer(
    conn: &Connection,
    internal_id: i64,
    embedding_id: String,
    created_at: Option<i64>,
    collection_name: &str,
) -> Result<Option<MempalaceDrawer>, String> {
    // Pull every metadata k/v for this embedding.
    let mut stmt = conn
        .prepare(
            "SELECT key, string_value, int_value, float_value, bool_value \
             FROM embedding_metadata WHERE id = ?",
        )
        .map_err(|e| format!("embedding_metadata prepare: {e}"))?;
    let mut rows = stmt
        .query(params![internal_id])
        .map_err(|e| format!("embedding_metadata query: {e}"))?;

    let mut metadata = serde_json::Map::new();
    let mut content: Option<String> = None;
    while let Some(r) = rows
        .next()
        .map_err(|e| format!("embedding_metadata next: {e}"))?
    {
        let key: String = r.get(0).map_err(|e| format!("metadata.key: {e}"))?;
        let s_val: Option<String> = r.get(1).ok();
        let i_val: Option<i64> = r.get(2).ok();
        let f_val: Option<f64> = r.get(3).ok();
        let b_val: Option<bool> = r.get(4).ok();

        if key == CHROMA_DOC_KEY {
            content = s_val.clone();
            continue;
        }
        let value: serde_json::Value = match (s_val, i_val, f_val, b_val) {
            (Some(s), _, _, _) => serde_json::Value::String(s),
            (_, Some(i), _, _) => serde_json::json!(i),
            (_, _, Some(f), _) => serde_json::json!(f),
            (_, _, _, Some(b)) => serde_json::Value::Bool(b),
            _ => serde_json::Value::Null,
        };
        metadata.insert(key, value);
    }

    // Skip registry sentinels — bookkeeping rows MemPalace writes per source
    // file to mark "already mined". They have no user-facing content.
    let room = metadata.get("room").and_then(|v| v.as_str()).unwrap_or("");
    let ingest_mode = metadata
        .get("ingest_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if room == REGISTRY_ROOM || ingest_mode == "registry" {
        return Ok(None);
    }

    let body = match content {
        Some(s) if !s.is_empty() => s,
        // No document body and not a sentinel — surface as empty so the
        // normalizer can reject it (rather than silently dropping).
        _ => return Ok(None),
    };

    Ok(Some(MempalaceDrawer {
        collection_name: collection_name.to_string(),
        embedding_id,
        content: body,
        metadata: serde_json::Value::Object(metadata),
        created_unix: created_at,
    }))
}

fn has_table(conn: &Connection, name: &str) -> bool {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        params![name],
        |_| Ok(()),
    )
    .is_ok()
}

fn read_text(p: &Path) -> Option<String> {
    match fs::read_to_string(p) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!(
                path = %p.display(),
                error = %e,
                "mempalace scanner: unreadable file"
            );
            None
        }
    }
}

fn file_mtime_unix(p: &Path) -> Option<i64> {
    fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
}

/// Best-effort: parse a `metadata.filed_at` ISO-8601 timestamp into unix
/// seconds. Used by the normalizer to fall back when `created_at` isn't
/// surfaced by the Chroma schema.
pub fn filed_at_unix(metadata: &serde_json::Value) -> Option<i64> {
    let s = metadata.get("filed_at").and_then(|v| v.as_str())?;
    let dt = chrono::DateTime::parse_from_rfc3339(s).ok()?;
    Some(dt.with_timezone(&Utc).timestamp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static MP_SCAN_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MP_SCAN_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-mempalace-scan-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// Seed a chroma.sqlite3 with the minimal schema our reader expects.
    /// This isn't a full Chroma DB — just enough rows to exercise the
    /// classification + metadata-extraction logic.
    fn seed_chroma(db_path: &Path) {
        let conn = Connection::open(db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, dimension INTEGER);
             CREATE TABLE segments (id TEXT PRIMARY KEY, collection TEXT, scope TEXT);
             CREATE TABLE embeddings (
                 id INTEGER PRIMARY KEY,
                 segment_id TEXT,
                 embedding_id TEXT,
                 seq_id BLOB,
                 created_at INTEGER
             );
             CREATE TABLE embedding_metadata (
                 id INTEGER,
                 key TEXT,
                 string_value TEXT,
                 int_value INTEGER,
                 float_value REAL,
                 bool_value INTEGER
             );",
        )
        .unwrap();
        // mempalace_drawers collection.
        conn.execute(
            "INSERT INTO collections (id, name, dimension) VALUES (?, ?, ?)",
            params!["coll-drawers", COLLECTION_DRAWERS, 384],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO segments (id, collection, scope) VALUES (?, ?, ?)",
            params!["seg-drawers", "coll-drawers", "METADATA"],
        )
        .unwrap();
        // Drawer 1: real content.
        conn.execute(
            "INSERT INTO embeddings (id, segment_id, embedding_id, created_at) \
             VALUES (?, ?, ?, ?)",
            params![
                1,
                "seg-drawers",
                "drawer_default_general_aaa",
                1_730_000_000_i64
            ],
        )
        .unwrap();
        for (k, sv) in [
            (
                "chroma:document",
                "user prefers dark mode and tabs over spaces",
            ),
            ("wing", "default"),
            ("room", "general"),
            ("source_file", "/repo/CLAUDE.md"),
            ("filed_at", "2026-05-01T10:00:00Z"),
        ] {
            conn.execute(
                "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
                params![1, k, sv],
            )
            .unwrap();
        }
        // Drawer 2: registry sentinel — should be skipped.
        conn.execute(
            "INSERT INTO embeddings (id, segment_id, embedding_id, created_at) \
             VALUES (?, ?, ?, ?)",
            params![2, "seg-drawers", "_reg_foo", 1_730_000_001_i64],
        )
        .unwrap();
        for (k, sv) in [
            ("chroma:document", "[registry] /repo/CLAUDE.md"),
            ("wing", "default"),
            ("room", "_registry"),
            ("ingest_mode", "registry"),
            ("source_file", "/repo/CLAUDE.md"),
        ] {
            conn.execute(
                "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
                params![2, k, sv],
            )
            .unwrap();
        }
        // mempalace_closets collection.
        conn.execute(
            "INSERT INTO collections (id, name, dimension) VALUES (?, ?, ?)",
            params!["coll-closets", COLLECTION_CLOSETS, 384],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO segments (id, collection, scope) VALUES (?, ?, ?)",
            params!["seg-closets", "coll-closets", "METADATA"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO embeddings (id, segment_id, embedding_id, created_at) \
             VALUES (?, ?, ?, ?)",
            params![
                3,
                "seg-closets",
                "closet_default_general_zzz",
                1_730_000_002_i64
            ],
        )
        .unwrap();
        for (k, sv) in [
            ("chroma:document", "rooms in wing default: general, work"),
            ("wing", "default"),
        ] {
            conn.execute(
                "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
                params![3, k, sv],
            )
            .unwrap();
        }
    }

    #[test]
    fn empty_home_yields_empty_scan() {
        let dir = tmp();
        let s = scan_mempalace(&dir);
        assert_eq!(s.total(), 0);
        assert!(s.chroma_error.is_none());
    }

    #[test]
    fn picks_up_identity_txt() {
        let dir = tmp();
        fs::write(dir.join("identity.txt"), "I am Atlas, an AI for Alice.").unwrap();
        let s = scan_mempalace(&dir);
        assert_eq!(s.identities.len(), 1);
        assert!(s.identities[0].content.starts_with("I am Atlas"));
        assert_eq!(s.drawers.len(), 0);
    }

    #[test]
    fn reads_chroma_drawers_and_closets_and_skips_registry() {
        let dir = tmp();
        fs::create_dir_all(dir.join("palace")).unwrap();
        let db = dir.join("palace/chroma.sqlite3");
        seed_chroma(&db);

        let s = scan_mempalace(&dir);
        assert!(s.chroma_error.is_none(), "scan err: {:?}", s.chroma_error);
        // 1 real drawer + 1 closet; the registry sentinel is filtered.
        assert_eq!(s.drawers.len(), 2);
        let drawer_names: Vec<_> = s
            .drawers
            .iter()
            .map(|d| d.collection_name.as_str())
            .collect();
        assert!(drawer_names.contains(&COLLECTION_DRAWERS));
        assert!(drawer_names.contains(&COLLECTION_CLOSETS));
        // The mined drawer has wing/room/source_file in metadata.
        let mined = s
            .drawers
            .iter()
            .find(|d| d.collection_name == COLLECTION_DRAWERS)
            .unwrap();
        assert_eq!(
            mined.metadata.get("wing").and_then(|v| v.as_str()),
            Some("default")
        );
        assert_eq!(
            mined.metadata.get("room").and_then(|v| v.as_str()),
            Some("general")
        );
        assert!(mined.content.contains("dark mode"));
        assert_eq!(mined.created_unix, Some(1_730_000_000));
    }

    #[test]
    fn unrecognized_chroma_schema_produces_error_note_not_panic() {
        let dir = tmp();
        fs::create_dir_all(dir.join("palace")).unwrap();
        let db = dir.join("palace/chroma.sqlite3");
        // An empty SQLite file (no tables) — unrecognized as Chroma.
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch("CREATE TABLE unrelated (x INTEGER);")
            .unwrap();
        drop(conn);

        let s = scan_mempalace(&dir);
        assert!(s.chroma_error.is_some());
        assert_eq!(s.drawers.len(), 0);
    }

    #[test]
    fn empty_chroma_with_known_schema_returns_no_drawers() {
        let dir = tmp();
        fs::create_dir_all(dir.join("palace")).unwrap();
        let db = dir.join("palace/chroma.sqlite3");
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE collections (id TEXT, name TEXT, dimension INTEGER);
             CREATE TABLE segments (id TEXT, collection TEXT, scope TEXT);
             CREATE TABLE embeddings (id INTEGER, segment_id TEXT, embedding_id TEXT, created_at INTEGER);
             CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT, int_value INTEGER, float_value REAL, bool_value INTEGER);",
        )
        .unwrap();
        drop(conn);

        let s = scan_mempalace(&dir);
        assert!(s.chroma_error.is_none());
        assert_eq!(s.drawers.len(), 0);
    }

    #[test]
    fn filed_at_unix_parses_rfc3339() {
        let m = serde_json::json!({"filed_at": "2026-05-01T10:00:00Z"});
        let unix = filed_at_unix(&m).unwrap();
        // 2026-05-01T10:00:00Z → 1777629600 (per `date -ujf` cross-check).
        assert_eq!(unix, 1_777_629_600);
    }
}
