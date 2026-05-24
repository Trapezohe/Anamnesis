//! SQLite-format exporters (`mem0-sqlite`, `letta-sqlite`).
//!
//! R138 / R139 introduced these inline in the CLI; R140 lifts them
//! into the shared crate so MCP `export_memories` reuses the exact
//! same write paths, the same provenance metadata convention, and
//! the same fresh-file safety contract.
//!
//! ## Provenance metadata convention
//!
//! Every SQLite-output row carries an `anamnesis_*` block inside
//! its `metadata` / `metadata_` column so a downstream re-import
//! preserves which source originally produced each row. Shared
//! across both exporters via [`anamnesis_provenance_block`].
//!
//! ## Safety
//!
//! These functions trust the caller has already validated `out`
//! via [`crate::validate_sqlite_output`] — they will NOT refuse
//! to overwrite an existing file on their own. The contract is
//! enforced at the entry point so the failure mode surfaces
//! before any work happens.

use std::path::Path;

use anamnesis_core::RecordId;
use anamnesis_store::Store;
use serde_json::Value;

use crate::ExportError;

/// Build the `anamnesis_*` provenance JSON block stored in every
/// SQLite-output row's `metadata` column. Shared by both
/// mem0-sqlite and letta-sqlite exporters so the round-trip
/// contract stays uniform.
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

/// Export to a fresh SQLite DB matching mem0's canonical
/// `memories` table (the schema `adapter-mem0/src/scanner.rs`
/// probes). See [`crate::ExportFormat::Mem0Sqlite`].
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

/// Export to a fresh SQLite DB matching Letta's canonical `block`
/// table. For Letta-origin rows, reconstructs the native
/// `label` / `description` / `template_name` from metadata; for
/// foreign-origin rows, falls back to a stable
/// `anamnesis/<adapter>` label. See
/// [`crate::ExportFormat::LettaSqlite`].
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
