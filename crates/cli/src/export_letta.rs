//! Round 139 (PR-78bh): `anamnesis export --format letta-sqlite`.
//!
//! Extends R138's bidirectional round-trip story (mem0) to Letta.
//! Writes a fresh SQLite database with the same `block` table
//! shape Letta's own SQLite store uses — the schema
//! `adapter-letta/src/scanner.rs` probes via PRAGMA.
//!
//! ## Round-trip semantics
//!
//! For Letta-origin records, the normalizer already stashes
//! `letta_label`, `letta_description`, `letta_template` keys in
//! `record.metadata` (see `adapter-letta/src/normalizer.rs`).
//! This exporter unpacks those back into native `block` columns
//! so a Letta re-import gets faithful values, not Anamnesis
//! defaults.
//!
//! For non-Letta-origin records, we default to a stable
//! `anamnesis/<adapter>` label so the row is still a well-formed
//! Letta `block` row. The provenance backlink lives in
//! `metadata_` under `anamnesis_*` keys, matching the convention
//! R138 established.
//!
//! ## What this is NOT
//!
//! - **Not a Postgres migrator.** Letta in production runs
//!   Postgres; this exporter targets the SQLite path only (same
//!   scope as the existing adapter-letta scanner).
//! - **Not `archival_passages` / `messages`.** Out of scope per
//!   the Letta adapter's own §-2.3 PR-1 design.
//! - **Not a live overwrite.** Refuses to clobber an existing
//!   file — the operator's real `~/.letta/letta.db` is
//!   one-typo-away.

use std::path::Path;

use anamnesis_store::Store;
use anyhow::Result;

/// Default value for the `block.label` column when the source
/// record was NOT produced by the Letta adapter. Includes the
/// origin adapter id so a downstream operator can tell at a
/// glance "this block came from Anamnesis re-exporting a mem0
/// record" vs a native Letta block.
fn default_label_for_foreign_record(adapter: &str) -> String {
    format!("anamnesis/{adapter}")
}

/// Export the given records as a fresh Letta-compatible SQLite
/// `block` table. Mirrors the safety contract of R138 — the caller
/// is responsible for refusing an existing-file overwrite at the
/// CLI layer; this function trusts `out` is a path that doesn't
/// yet exist.
///
/// Wire shape (matches `adapter-letta/src/scanner.rs` PRAGMA probe):
///
/// ```sql
/// CREATE TABLE block (
///     id            TEXT PRIMARY KEY,
///     value         TEXT NOT NULL,
///     label         TEXT,
///     description   TEXT,
///     template_name TEXT,
///     metadata_     TEXT,   -- JSON; carries Anamnesis provenance
///     created_at    TEXT,   -- RFC3339
///     updated_at    TEXT    -- RFC3339
/// );
/// ```
///
/// Note: `metadata_` (trailing underscore) matches Letta's own
/// column name, which avoids the SQL reserved word `metadata` on
/// older SQLite builds.
pub fn export_letta_sqlite(store: &Store, ids: &[String], out: &Path) -> Result<()> {
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
            let Some(rec) = store.get_record(&anamnesis_core::RecordId(id.clone()))? else {
                continue;
            };

            // Reconstruct native Letta `block` columns from the
            // `letta_*` keys the Letta normalizer left in metadata
            // for Letta-origin rows. For foreign-origin rows
            // (mem0, claude-code, etc.) these keys are absent and
            // we fall back to safe defaults.
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
            let label = letta_label
                .unwrap_or_else(|| default_label_for_foreign_record(&rec.source.adapter));

            // Build the `metadata_` JSON object: original
            // metadata + `anamnesis_*` provenance backlink, same
            // shape R138 mem0-sqlite uses (and same purpose: a
            // re-import preserves which source originally produced
            // each row).
            let mut meta: serde_json::Map<String, serde_json::Value> = rec.metadata.clone();
            meta.insert(
                "anamnesis_source_adapter".into(),
                serde_json::json!(rec.source.adapter),
            );
            if let Some(inst) = &rec.source.instance {
                meta.insert("anamnesis_source_instance".into(), serde_json::json!(inst));
            }
            meta.insert(
                "anamnesis_kind".into(),
                serde_json::json!(format!("{:?}", rec.kind).to_lowercase()),
            );
            meta.insert(
                "anamnesis_scope".into(),
                serde_json::json!(format!("{:?}", rec.scope).to_lowercase()),
            );
            meta.insert("anamnesis_tags".into(), serde_json::json!(rec.tags));
            meta.insert(
                "anamnesis_native_id".into(),
                serde_json::json!(rec.provenance.native_id),
            );
            meta.insert(
                "anamnesis_raw_hash".into(),
                serde_json::json!(rec.provenance.raw_hash),
            );
            if let Some(parent) = &rec.provenance.derived_from {
                meta.insert("anamnesis_derived_from".into(), serde_json::json!(parent.0));
            }
            let metadata_json = serde_json::to_string(&serde_json::Value::Object(meta))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_label_includes_origin_adapter_for_traceability() {
        assert_eq!(default_label_for_foreign_record("mem0"), "anamnesis/mem0");
        assert_eq!(
            default_label_for_foreign_record("claude-code"),
            "anamnesis/claude-code"
        );
    }
}
