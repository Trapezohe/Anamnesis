//! SQLite-format exporters (`mem0-sqlite`, `letta-sqlite`).
//!
//! Every row's `metadata` carries an `anamnesis_*` provenance block so
//! a re-import preserves lineage. Callers MUST validate `out` via
//! `crate::validate_sqlite_output` first — these fns won't refuse overwrite.

use std::path::Path;

use anamnesis_core::RecordId;
use anamnesis_store::Store;
use serde_json::Value;

use crate::ExportError;

/// `anamnesis_*` provenance keys layered onto the record's metadata.
/// Shared by both SQLite exporters so the round-trip contract stays uniform.
fn anamnesis_provenance_block(
    rec: &anamnesis_core::AnamnesisRecord,
) -> serde_json::Map<String, Value> {
    let mut meta: serde_json::Map<String, Value> = rec.metadata.clone();
    meta.insert(
        "anamnesis_source_adapter".into(),
        Value::String(rec.source.adapter.clone()),
    );
    if let Some(inst) = &rec.source.instance {
        meta.insert(
            "anamnesis_source_instance".into(),
            Value::String(inst.clone()),
        );
    }
    meta.insert(
        "anamnesis_kind".into(),
        Value::String(format!("{:?}", rec.kind).to_lowercase()),
    );
    meta.insert(
        "anamnesis_scope".into(),
        Value::String(format!("{:?}", rec.scope).to_lowercase()),
    );
    meta.insert("anamnesis_tags".into(), serde_json::json!(rec.tags));
    meta.insert(
        "anamnesis_native_id".into(),
        Value::String(rec.provenance.native_id.clone()),
    );
    meta.insert(
        "anamnesis_raw_hash".into(),
        Value::String(rec.provenance.raw_hash.clone()),
    );
    if let Some(parent) = &rec.provenance.derived_from {
        meta.insert(
            "anamnesis_derived_from".into(),
            Value::String(parent.0.clone()),
        );
    }
    meta
}

/// Write `memories` table matching mem0's scanner probe.
pub fn export_mem0_sqlite(store: &Store, ids: &[String], out: &Path) -> Result<(), ExportError> {
    let conn = rusqlite::Connection::open(out)?;
    conn.execute_batch(
        "CREATE TABLE memories ( \
            id         TEXT PRIMARY KEY, \
            memory     TEXT NOT NULL, \
            user_id    TEXT, \
            agent_id   TEXT, \
            run_id     TEXT, \
            metadata   TEXT, \
            created_at TEXT, \
            updated_at TEXT, \
            hash       TEXT \
        );",
    )?;
    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO memories \
                (id, memory, user_id, agent_id, run_id, metadata, created_at, updated_at, hash) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )?;
        for id in ids {
            let Some(rec) = store.get_record(&RecordId(id.clone()))? else {
                continue;
            };
            let metadata_json =
                serde_json::to_string(&Value::Object(anamnesis_provenance_block(&rec)))?;
            let created_iso = rec.created_at.to_rfc3339();
            let updated_iso = rec.updated_at.map(|t| t.to_rfc3339());
            stmt.execute(rusqlite::params![
                rec.id.0,
                rec.content,
                Option::<String>::None,
                Option::<String>::None,
                Option::<String>::None,
                metadata_json,
                created_iso,
                updated_iso,
                rec.provenance.raw_hash,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Write `block` table matching Letta's scanner probe. Letta-origin rows
/// restore native label/description/template_name from metadata; foreign
/// rows get `anamnesis/<adapter>` as label.
pub fn export_letta_sqlite(store: &Store, ids: &[String], out: &Path) -> Result<(), ExportError> {
    let conn = rusqlite::Connection::open(out)?;
    conn.execute_batch(
        "CREATE TABLE block ( \
            id            TEXT PRIMARY KEY, \
            value         TEXT NOT NULL, \
            label         TEXT, \
            description   TEXT, \
            template_name TEXT, \
            metadata_     TEXT, \
            created_at    TEXT, \
            updated_at    TEXT \
        );",
    )?;
    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO block \
                (id, value, label, description, template_name, metadata_, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )?;
        for id in ids {
            let Some(rec) = store.get_record(&RecordId(id.clone()))? else {
                continue;
            };
            let letta_label = rec
                .metadata
                .get("letta_label")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
            let letta_description = rec
                .metadata
                .get("letta_description")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
            let letta_template = rec
                .metadata
                .get("letta_template")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
            let label = letta_label.unwrap_or_else(|| format!("anamnesis/{}", rec.source.adapter));

            let metadata_json =
                serde_json::to_string(&Value::Object(anamnesis_provenance_block(&rec)))?;
            let created_iso = rec.created_at.to_rfc3339();
            let updated_iso = rec.updated_at.map(|t| t.to_rfc3339());
            stmt.execute(rusqlite::params![
                rec.id.0,
                rec.content,
                label,
                letta_description,
                letta_template,
                metadata_json,
                created_iso,
                updated_iso,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Write `memori_entity_fact` (+ a single synthetic `memori_entity`)
/// matching Memori's scanner probe. The `metadata` JSON column carries the
/// Anamnesis provenance block so re-import restores the original
/// `anamnesis_native_id` / raw_hash. Memori-origin rows reuse their native
/// `memori_num_times`; foreign rows default to 1.
pub fn export_memori_sqlite(store: &Store, ids: &[String], out: &Path) -> Result<(), ExportError> {
    let conn = rusqlite::Connection::open(out)?;
    conn.execute_batch(
        "CREATE TABLE memori_entity ( \
            id          INTEGER PRIMARY KEY, \
            uuid        TEXT, \
            external_id TEXT \
        ); \
        CREATE TABLE memori_entity_fact ( \
            id             INTEGER PRIMARY KEY, \
            uuid           TEXT NOT NULL, \
            entity_id      INTEGER, \
            content        TEXT NOT NULL, \
            num_times      INTEGER, \
            date_last_time TEXT, \
            date_created   TEXT, \
            metadata       TEXT \
        ); \
        INSERT INTO memori_entity (id, uuid, external_id) \
            VALUES (1, 'anamnesis-export-entity', 'anamnesis-export');",
    )?;
    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO memori_entity_fact \
                (uuid, entity_id, content, num_times, date_last_time, date_created, metadata) \
             VALUES (?1, 1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for id in ids {
            let Some(rec) = store.get_record(&RecordId(id.clone()))? else {
                continue;
            };
            let num_times = rec
                .metadata
                .get("memori_num_times")
                .and_then(|v| v.as_i64())
                .unwrap_or(1);
            let metadata_json =
                serde_json::to_string(&Value::Object(anamnesis_provenance_block(&rec)))?;
            let created_iso = rec.created_at.to_rfc3339();
            let last_iso = rec.updated_at.unwrap_or(rec.created_at).to_rfc3339();
            stmt.execute(rusqlite::params![
                rec.provenance.native_id,
                rec.content,
                num_times,
                last_iso,
                created_iso,
                metadata_json,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}
