-- 0004_provenance_derived_from.sql
--
-- Add the `derived_from` provenance column so the §-1.5 PR-6 session
-- extractor can record lineage from each derived Fact/Preference/Skill/
-- Feedback back to the source Episode record.
--
-- Per the schema-evolution convention in BLUEPRINT §4: this is an
-- additive nullable column, so existing rows stay valid without backfill.
-- New extractor-produced records populate it; adapter-produced records
-- leave it NULL.
--
-- We also index it because future `anamnesis lineage <record-id>` queries
-- will need to walk derivations efficiently.

ALTER TABLE records ADD COLUMN derived_from TEXT;

CREATE INDEX IF NOT EXISTS idx_records_derived_from
    ON records(derived_from)
    WHERE derived_from IS NOT NULL;
