//! SQLite scanner for a Memori install.
//!
//! Memori (MemoriLabs/Memori, Apache-2.0) writes its SQLite schema with a
//! fixed `memori_` table-name prefix. The tables we care about:
//!
//!   - `memori_entity_fact`          — extracted facts about a user (Fact / User)
//!   - `memori_process_attribute`    — extracted facts about an agent/app (Reference / Project)
//!   - `memori_conversation_message` — raw conversation turns (Episode / Session)
//!   - `memori_conversation`         — per-conversation summary (Episode / Session)
//!   - `memori_knowledge_graph` + `memori_subject` + `memori_predicate`
//!     + `memori_object`             — extracted s-p-o triples (Fact / User)
//!
//! All probes are schema-tolerant: a missing table just yields zero rows
//! for that record kind; only a missing `entity_fact` AND `conversation`
//! AND `kg` is treated as "this isn't a Memori DB".
//!
//! Per §-1.2.2 the adapter is read-only — we open with
//! `SQLITE_OPEN_READ_ONLY`.

use std::path::Path;

use rusqlite::{params, Connection, OpenFlags, Row};

/// One `memori_entity_fact` row.
#[derive(Debug, Clone)]
pub struct MemoriEntityFact {
    /// Row UUID (stable across reruns).
    pub uuid: String,
    /// External id of the entity this fact is attached to (a user id).
    pub entity_external_id: Option<String>,
    /// Fact text.
    pub content: String,
    /// Times this fact has been observed (Memori's frequency counter).
    pub num_times: i64,
    /// Last time this fact was observed (ISO-8601 string per the
    /// `datetime('now')` SQLite default).
    pub date_last_time: Option<String>,
    /// Initial creation timestamp.
    pub date_created: Option<String>,
    /// Optional `metadata` JSON column. Native Memori has no such column;
    /// Anamnesis round-trip exports write the provenance block here so
    /// re-import restores the original `anamnesis_native_id` / raw_hash.
    pub metadata: Option<String>,
}

/// One `memori_process_attribute` row.
#[derive(Debug, Clone)]
pub struct MemoriProcessAttribute {
    /// Row UUID.
    pub uuid: String,
    /// External id of the process (an agent/app id).
    pub process_external_id: Option<String>,
    /// Attribute text.
    pub content: String,
    /// Observation counter.
    pub num_times: i64,
    /// Last-observed time.
    pub date_last_time: Option<String>,
    /// Initial creation timestamp.
    pub date_created: Option<String>,
}

/// One `memori_conversation_message` row.
#[derive(Debug, Clone)]
pub struct MemoriConversationMessage {
    /// Row UUID.
    pub uuid: String,
    /// `user` / `assistant` / `system` etc.
    pub role: String,
    /// Optional type field (e.g. `text`, `tool`).
    pub type_: Option<String>,
    /// Message body.
    pub content: String,
    /// Session UUID (joined via conversation → session).
    pub session_uuid: Option<String>,
    /// Initial creation timestamp.
    pub date_created: Option<String>,
}

/// One `memori_conversation` summary row.
#[derive(Debug, Clone)]
pub struct MemoriConversationSummary {
    /// Conversation row UUID.
    pub uuid: String,
    /// Session UUID this conversation belongs to.
    pub session_uuid: Option<String>,
    /// The LLM-generated summary text.
    pub summary: String,
    /// Initial creation timestamp.
    pub date_created: Option<String>,
}

/// One s-p-o triple from `memori_knowledge_graph` joined with its
/// subject/predicate/object tables.
#[derive(Debug, Clone)]
pub struct MemoriKgTriple {
    /// Triple row UUID.
    pub uuid: String,
    /// External id of the entity this triple is attached to.
    pub entity_external_id: Option<String>,
    /// `(subject_name :: subject_type)`.
    pub subject: String,
    /// Predicate text.
    pub predicate: String,
    /// `(object_name :: object_type)`.
    pub object: String,
    /// Observation counter.
    pub num_times: i64,
    /// Last-observed time.
    pub date_last_time: Option<String>,
    /// Initial creation timestamp.
    pub date_created: Option<String>,
}

/// Aggregate Memori scan.
#[derive(Debug, Default)]
pub struct MemoriScan {
    /// All entity facts.
    pub entity_facts: Vec<MemoriEntityFact>,
    /// All process attributes.
    pub process_attrs: Vec<MemoriProcessAttribute>,
    /// All conversation messages.
    pub messages: Vec<MemoriConversationMessage>,
    /// All conversation summaries (rows where `summary IS NOT NULL`).
    pub summaries: Vec<MemoriConversationSummary>,
    /// All KG triples.
    pub kg_triples: Vec<MemoriKgTriple>,
    /// Diagnostic note set when the file exists but isn't a Memori DB.
    pub schema_error: Option<String>,
}

impl MemoriScan {
    /// Total row count.
    pub fn total(&self) -> usize {
        self.entity_facts.len()
            + self.process_attrs.len()
            + self.messages.len()
            + self.summaries.len()
            + self.kg_triples.len()
    }
}

/// Scan a Memori SQLite database file.
pub fn scan_memori(db_path: &Path) -> MemoriScan {
    let mut scan = MemoriScan::default();
    if !db_path.is_file() {
        return scan;
    }

    let conn = match Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                path = %db_path.display(),
                error = %e,
                "memori scanner: cannot open DB"
            );
            scan.schema_error = Some(format!("open: {e}"));
            return scan;
        }
    };

    // Quick schema sniff — at least one of the Memori-specific tables must
    // exist; otherwise this is somebody else's SQLite.
    let has_any_memori_table = [
        "memori_entity_fact",
        "memori_conversation",
        "memori_knowledge_graph",
    ]
    .iter()
    .any(|t| has_table(&conn, t));
    if !has_any_memori_table {
        scan.schema_error = Some("not a Memori schema (no memori_* tables found)".into());
        return scan;
    }

    if has_table(&conn, "memori_entity_fact") {
        scan.entity_facts = read_entity_facts(&conn).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "memori scanner: entity_fact read failed");
            vec![]
        });
    }
    if has_table(&conn, "memori_process_attribute") {
        scan.process_attrs = read_process_attrs(&conn).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "memori scanner: process_attribute read failed");
            vec![]
        });
    }
    if has_table(&conn, "memori_conversation_message") {
        scan.messages = read_messages(&conn).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "memori scanner: conversation_message read failed");
            vec![]
        });
    }
    if has_table(&conn, "memori_conversation") {
        scan.summaries = read_summaries(&conn).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "memori scanner: conversation summary read failed");
            vec![]
        });
    }
    if has_table(&conn, "memori_knowledge_graph") {
        scan.kg_triples = read_kg_triples(&conn).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "memori scanner: knowledge_graph read failed");
            vec![]
        });
    }

    scan
}

fn read_entity_facts(conn: &Connection) -> Result<Vec<MemoriEntityFact>, String> {
    // Join to `memori_entity` so we have the external user id for provenance.
    // `f.metadata` exists only on Anamnesis round-trip exports; select NULL
    // when the column is absent so native Memori DBs still scan.
    let meta_col = if has_column(conn, "memori_entity_fact", "metadata") {
        "f.metadata"
    } else {
        "NULL"
    };
    let mut stmt = conn
        .prepare(&format!(
            "SELECT f.uuid, e.external_id, f.content, f.num_times, \
                    f.date_last_time, f.date_created, {meta_col} \
             FROM memori_entity_fact f \
             LEFT JOIN memori_entity e ON e.id = f.entity_id"
        ))
        .map_err(|e| format!("entity_fact prepare: {e}"))?;
    let rows = stmt
        .query_map([], |r: &Row<'_>| {
            Ok(MemoriEntityFact {
                uuid: r.get(0)?,
                entity_external_id: r.get(1).ok(),
                content: r.get(2)?,
                num_times: r.get::<_, i64>(3).unwrap_or(0),
                date_last_time: r.get(4).ok(),
                date_created: r.get(5).ok(),
                metadata: r.get(6).ok(),
            })
        })
        .map_err(|e| format!("entity_fact query: {e}"))?;
    let mut out = Vec::new();
    for row in rows.flatten() {
        out.push(row);
    }
    Ok(out)
}

fn read_process_attrs(conn: &Connection) -> Result<Vec<MemoriProcessAttribute>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT a.uuid, p.external_id, a.content, a.num_times, \
                    a.date_last_time, a.date_created \
             FROM memori_process_attribute a \
             LEFT JOIN memori_process p ON p.id = a.process_id",
        )
        .map_err(|e| format!("process_attribute prepare: {e}"))?;
    let rows = stmt
        .query_map([], |r: &Row<'_>| {
            Ok(MemoriProcessAttribute {
                uuid: r.get(0)?,
                process_external_id: r.get(1).ok(),
                content: r.get(2)?,
                num_times: r.get::<_, i64>(3).unwrap_or(0),
                date_last_time: r.get(4).ok(),
                date_created: r.get(5).ok(),
            })
        })
        .map_err(|e| format!("process_attribute query: {e}"))?;
    let mut out = Vec::new();
    for row in rows.flatten() {
        out.push(row);
    }
    Ok(out)
}

fn read_messages(conn: &Connection) -> Result<Vec<MemoriConversationMessage>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT m.uuid, m.role, m.type, m.content, s.uuid, m.date_created \
             FROM memori_conversation_message m \
             LEFT JOIN memori_conversation c ON c.id = m.conversation_id \
             LEFT JOIN memori_session s ON s.id = c.session_id",
        )
        .map_err(|e| format!("conversation_message prepare: {e}"))?;
    let rows = stmt
        .query_map([], |r: &Row<'_>| {
            Ok(MemoriConversationMessage {
                uuid: r.get(0)?,
                role: r.get::<_, String>(1).unwrap_or_else(|_| "unknown".into()),
                type_: r.get(2).ok(),
                content: r.get(3)?,
                session_uuid: r.get(4).ok(),
                date_created: r.get(5).ok(),
            })
        })
        .map_err(|e| format!("conversation_message query: {e}"))?;
    let mut out = Vec::new();
    for row in rows.flatten() {
        out.push(row);
    }
    Ok(out)
}

fn read_summaries(conn: &Connection) -> Result<Vec<MemoriConversationSummary>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT c.uuid, s.uuid, c.summary, c.date_created \
             FROM memori_conversation c \
             LEFT JOIN memori_session s ON s.id = c.session_id \
             WHERE c.summary IS NOT NULL AND TRIM(c.summary) != ''",
        )
        .map_err(|e| format!("conversation summary prepare: {e}"))?;
    let rows = stmt
        .query_map([], |r: &Row<'_>| {
            Ok(MemoriConversationSummary {
                uuid: r.get(0)?,
                session_uuid: r.get(1).ok(),
                summary: r.get(2)?,
                date_created: r.get(3).ok(),
            })
        })
        .map_err(|e| format!("conversation summary query: {e}"))?;
    let mut out = Vec::new();
    for row in rows.flatten() {
        out.push(row);
    }
    Ok(out)
}

fn read_kg_triples(conn: &Connection) -> Result<Vec<MemoriKgTriple>, String> {
    // The KG joins entity + subject + predicate + object. We tolerate missing
    // sidecar tables (subject/predicate/object) by checking for them first.
    if !has_table(conn, "memori_subject")
        || !has_table(conn, "memori_predicate")
        || !has_table(conn, "memori_object")
    {
        return Ok(vec![]);
    }
    let mut stmt = conn
        .prepare(
            "SELECT kg.uuid, e.external_id, \
                    s.name || ' :: ' || s.type, \
                    p.content, \
                    o.name || ' :: ' || o.type, \
                    kg.num_times, kg.date_last_time, kg.date_created \
             FROM memori_knowledge_graph kg \
             LEFT JOIN memori_entity e    ON e.id = kg.entity_id \
             LEFT JOIN memori_subject s   ON s.id = kg.subject_id \
             LEFT JOIN memori_predicate p ON p.id = kg.predicate_id \
             LEFT JOIN memori_object o    ON o.id = kg.object_id",
        )
        .map_err(|e| format!("knowledge_graph prepare: {e}"))?;
    let rows = stmt
        .query_map([], |r: &Row<'_>| {
            Ok(MemoriKgTriple {
                uuid: r.get(0)?,
                entity_external_id: r.get(1).ok(),
                subject: r.get::<_, String>(2).unwrap_or_default(),
                predicate: r.get::<_, String>(3).unwrap_or_default(),
                object: r.get::<_, String>(4).unwrap_or_default(),
                num_times: r.get::<_, i64>(5).unwrap_or(0),
                date_last_time: r.get(6).ok(),
                date_created: r.get(7).ok(),
            })
        })
        .map_err(|e| format!("knowledge_graph query: {e}"))?;
    let mut out = Vec::new();
    for row in rows.flatten() {
        out.push(row);
    }
    Ok(out)
}

fn has_table(conn: &Connection, name: &str) -> bool {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        params![name],
        |_| Ok(()),
    )
    .is_ok()
}

fn has_column(conn: &Connection, table: &str, column: &str) -> bool {
    conn.prepare(&format!("PRAGMA table_info({table})"))
        .and_then(|mut stmt| {
            let names = stmt
                .query_map([], |r| r.get::<_, String>(1))?
                .filter_map(Result::ok)
                .any(|c| c == column);
            Ok(names)
        })
        .unwrap_or(false)
}

/// Parse Memori's `datetime('now')`-style ISO-8601 timestamps into unix
/// seconds (UTC). Handles both `"2026-05-01T10:00:00"` and
/// `"2026-05-01 10:00:00"` and `"...Z"` forms.
pub fn parse_memori_time(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // Try strict RFC3339 first.
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.timestamp());
    }
    // SQLite's datetime('now') returns "YYYY-MM-DD HH:MM:SS" without TZ.
    // Treat as UTC since that's what SQLite emits.
    let candidates = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
    ];
    for fmt in candidates {
        if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return Some(ndt.and_utc().timestamp());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static MEMORI_SCAN_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_db_path() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MEMORI_SCAN_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "anamnesis-memori-scan-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir.join("memori.db")
    }

    fn seed_full_schema(db_path: &Path) {
        let conn = Connection::open(db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE memori_entity (id INTEGER PRIMARY KEY, uuid TEXT, external_id TEXT);
             CREATE TABLE memori_process (id INTEGER PRIMARY KEY, uuid TEXT, external_id TEXT);
             CREATE TABLE memori_session (id INTEGER PRIMARY KEY, uuid TEXT, entity_id INTEGER, process_id INTEGER);
             CREATE TABLE memori_conversation (id INTEGER PRIMARY KEY, uuid TEXT, session_id INTEGER, summary TEXT, date_created TEXT);
             CREATE TABLE memori_conversation_message (
                 id INTEGER PRIMARY KEY, uuid TEXT, conversation_id INTEGER,
                 role TEXT, type TEXT, content TEXT, date_created TEXT
             );
             CREATE TABLE memori_entity_fact (
                 id INTEGER PRIMARY KEY, uuid TEXT, entity_id INTEGER,
                 content TEXT, num_times INTEGER, date_last_time TEXT, date_created TEXT
             );
             CREATE TABLE memori_process_attribute (
                 id INTEGER PRIMARY KEY, uuid TEXT, process_id INTEGER,
                 content TEXT, num_times INTEGER, date_last_time TEXT, date_created TEXT
             );
             CREATE TABLE memori_subject (id INTEGER PRIMARY KEY, uuid TEXT, name TEXT, type TEXT);
             CREATE TABLE memori_predicate (id INTEGER PRIMARY KEY, uuid TEXT, content TEXT);
             CREATE TABLE memori_object (id INTEGER PRIMARY KEY, uuid TEXT, name TEXT, type TEXT);
             CREATE TABLE memori_knowledge_graph (
                 id INTEGER PRIMARY KEY, uuid TEXT, entity_id INTEGER,
                 subject_id INTEGER, predicate_id INTEGER, object_id INTEGER,
                 num_times INTEGER, date_last_time TEXT, date_created TEXT
             );",
        )
        .unwrap();

        conn.execute(
            "INSERT INTO memori_entity (id, uuid, external_id) VALUES (?, ?, ?)",
            params![1, "ent-uuid", "user-123"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_process (id, uuid, external_id) VALUES (?, ?, ?)",
            params![10, "proc-uuid", "my-app"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_session (id, uuid, entity_id, process_id) VALUES (?, ?, ?, ?)",
            params![100, "sess-uuid", 1, 10],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_conversation (id, uuid, session_id, summary, date_created) \
             VALUES (?, ?, ?, ?, ?)",
            params![
                1000,
                "conv-uuid",
                100,
                "User asked about colors and cities.",
                "2026-05-01 10:00:00",
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_conversation_message \
             (uuid, conversation_id, role, type, content, date_created) \
             VALUES (?, ?, ?, ?, ?, ?)",
            params![
                "msg1",
                1000,
                "user",
                "text",
                "My favorite color is blue",
                "2026-05-01 10:00:00",
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_conversation_message \
             (uuid, conversation_id, role, type, content, date_created) \
             VALUES (?, ?, ?, ?, ?, ?)",
            params![
                "msg2",
                1000,
                "assistant",
                "text",
                "Got it.",
                "2026-05-01 10:00:01",
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_entity_fact \
             (uuid, entity_id, content, num_times, date_last_time, date_created) \
             VALUES (?, ?, ?, ?, ?, ?)",
            params![
                "fact-uuid",
                1,
                "user lives in Paris",
                3,
                "2026-05-01 10:00:00",
                "2026-04-01 10:00:00",
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_process_attribute \
             (uuid, process_id, content, num_times, date_last_time, date_created) \
             VALUES (?, ?, ?, ?, ?, ?)",
            params![
                "attr-uuid",
                10,
                "app prefers JSON responses",
                5,
                "2026-05-01 10:00:00",
                "2026-04-01 10:00:00",
            ],
        )
        .unwrap();
        // KG triple.
        conn.execute(
            "INSERT INTO memori_subject (id, uuid, name, type) VALUES (?, ?, ?, ?)",
            params![1, "subj-uuid", "user", "Person"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_predicate (id, uuid, content) VALUES (?, ?, ?)",
            params![1, "pred-uuid", "lives_in"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_object (id, uuid, name, type) VALUES (?, ?, ?, ?)",
            params![1, "obj-uuid", "Paris", "City"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_knowledge_graph \
             (uuid, entity_id, subject_id, predicate_id, object_id, \
              num_times, date_last_time, date_created) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                "kg-uuid",
                1,
                1,
                1,
                1,
                2,
                "2026-05-01 10:00:00",
                "2026-04-01 10:00:00",
            ],
        )
        .unwrap();
    }

    #[test]
    fn missing_db_returns_empty_scan() {
        let s = scan_memori(Path::new("/tmp/never-here-memori-xyz.db"));
        assert_eq!(s.total(), 0);
        assert!(s.schema_error.is_none());
    }

    #[test]
    fn non_memori_sqlite_produces_schema_error() {
        let db = tmp_db_path();
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch("CREATE TABLE unrelated (x INTEGER);")
            .unwrap();
        drop(conn);
        let s = scan_memori(&db);
        assert!(s.schema_error.is_some());
        assert_eq!(s.total(), 0);
    }

    #[test]
    fn reads_all_five_record_kinds() {
        let db = tmp_db_path();
        seed_full_schema(&db);
        let s = scan_memori(&db);
        assert!(s.schema_error.is_none(), "schema_err: {:?}", s.schema_error);
        assert_eq!(s.entity_facts.len(), 1);
        assert_eq!(s.process_attrs.len(), 1);
        assert_eq!(s.messages.len(), 2);
        assert_eq!(s.summaries.len(), 1);
        assert_eq!(s.kg_triples.len(), 1);
        assert_eq!(s.total(), 6);

        let fact = &s.entity_facts[0];
        assert_eq!(fact.entity_external_id.as_deref(), Some("user-123"));
        assert!(fact.content.contains("Paris"));

        let msg = &s.messages[0];
        assert_eq!(msg.session_uuid.as_deref(), Some("sess-uuid"));

        let triple = &s.kg_triples[0];
        assert_eq!(triple.subject, "user :: Person");
        assert_eq!(triple.predicate, "lives_in");
        assert_eq!(triple.object, "Paris :: City");
    }

    #[test]
    fn conversation_without_summary_yields_no_summary_record() {
        let db = tmp_db_path();
        seed_full_schema(&db);
        let conn = Connection::open(&db).unwrap();
        conn.execute(
            "UPDATE memori_conversation SET summary = NULL WHERE id = 1000",
            [],
        )
        .unwrap();
        drop(conn);
        let s = scan_memori(&db);
        assert_eq!(s.summaries.len(), 0);
        // Other rows still picked up.
        assert!(!s.entity_facts.is_empty());
    }

    #[test]
    fn kg_without_sidecar_tables_skipped_safely() {
        let db = tmp_db_path();
        let conn = Connection::open(&db).unwrap();
        // Only the KG table, no subject/predicate/object.
        conn.execute_batch(
            "CREATE TABLE memori_knowledge_graph (
                 id INTEGER PRIMARY KEY, uuid TEXT, entity_id INTEGER,
                 subject_id INTEGER, predicate_id INTEGER, object_id INTEGER,
                 num_times INTEGER, date_last_time TEXT, date_created TEXT
             );",
        )
        .unwrap();
        drop(conn);
        let s = scan_memori(&db);
        assert!(s.schema_error.is_none()); // KG table present → recognized as Memori
        assert_eq!(s.kg_triples.len(), 0); // but no triples without sidecars
    }

    #[test]
    fn parse_memori_time_handles_sqlite_default_format() {
        // SQLite's `datetime('now')`: no TZ marker, treat as UTC.
        let t = parse_memori_time("2026-05-01 10:00:00").unwrap();
        assert_eq!(t, 1_777_629_600);
    }

    #[test]
    fn parse_memori_time_handles_iso8601_with_z() {
        let t = parse_memori_time("2026-05-01T10:00:00Z").unwrap();
        assert_eq!(t, 1_777_629_600);
    }
}
