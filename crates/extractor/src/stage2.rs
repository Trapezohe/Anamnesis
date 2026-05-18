//! Stage 2 — run the configured `LlmProvider` over every Stage 1
//! candidate and produce derived `AnamnesisRecord`s.
//!
//! Stage 2 stays small and explicit by design:
//!
//! 1. Build a kind-specific prompt for each candidate ([`crate::prompt`])
//! 2. Hand it to the provider's `complete()`
//! 3. Parse the response loosely (JSON expected, plain text tolerated)
//! 4. Build derived `AnamnesisRecord`s with `provenance.derived_from`
//!    pointing at the source Episode (§-1.5 #6 audit trail)
//! 5. Return the records — persistence is the caller's job (the CLI
//!    upserts them via `Store`)
//!
//! No record is mutated in-place; Stage 2 only emits new ones.

use anamnesis_core::error::Result;
use anamnesis_core::model::{
    AnamnesisRecord, Provenance, RecordId, SourceDescriptor, SCHEMA_VERSION,
};
use chrono::Utc;
use serde_json::Value;

use crate::provider::LlmProvider;
use crate::{prompt, Candidate, ExtractKind};

/// Adapter id used for records that came out of the Stage 2 extractor.
/// Not a real upstream — we ship as the "extractor" pseudo-adapter so
/// `anamnesis search --source extractor` returns just the derived
/// memory.
pub const EXTRACTOR_ADAPTER_ID: &str = "extractor";

/// One item the LLM returned for one candidate. The driver builds
/// `AnamnesisRecord`s from these.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedItem {
    /// The distilled content (text that becomes the new record's
    /// `content` field).
    pub content: String,
    /// Optional model-reported confidence in `[0.0, 1.0]`. Surfaced in
    /// `metadata.extractor_confidence`.
    pub confidence: Option<f32>,
}

/// Outcome of one Stage 2 run.
#[derive(Debug, Clone, Default)]
pub struct Stage2Report {
    /// New records ready to be `Store::upsert_record`'d by the caller.
    pub records: Vec<AnamnesisRecord>,
    /// Candidates whose LLM call returned no usable items.
    pub skipped: usize,
    /// Candidates whose LLM call failed; one error string per failure
    /// for surfacing in audit logs.
    pub errors: Vec<String>,
    /// Sum of `provider.estimate_tokens(prompt)` across every call —
    /// the same number shown to the user in the cost preview, kept
    /// here so audit logs can reconcile.
    pub estimated_input_tokens: usize,
}

/// Run Stage 2 over a slice of Stage 1 candidates.
///
/// The driver is async and intentionally **sequential**: even a small
/// concurrency level here pushes you into rate-limit territory on
/// hosted providers without giving the user a chance to inspect
/// progress. Concurrency can come in a follow-up PR behind a flag.
pub async fn run_stage2<P: LlmProvider>(
    provider: &P,
    candidates: &[Candidate],
    target_kind: ExtractKind,
    instance: Option<&str>,
) -> Result<Stage2Report> {
    let mut report = Stage2Report::default();
    for candidate in candidates {
        let prompt = prompt::build_prompt(target_kind, &candidate.record.content);
        report.estimated_input_tokens += provider.estimate_tokens(&prompt);
        let raw = match provider.complete(&prompt).await {
            Ok(s) => s,
            Err(e) => {
                report
                    .errors
                    .push(format!("{}: {e}", candidate.record.id.0));
                continue;
            }
        };
        let items = parse_extracted_items(&raw);
        if items.is_empty() {
            report.skipped += 1;
            continue;
        }
        for (i, item) in items.iter().enumerate() {
            let rec = build_derived_record(candidate, target_kind, item, i, instance);
            report.records.push(rec);
        }
    }
    Ok(report)
}

/// Parse the model's raw response into a list of [`ExtractedItem`]s.
///
/// Tolerant: accepts strict JSON arrays of `{content, confidence}`,
/// JSON arrays of bare strings, and (as a last resort) treats the
/// entire response as one item if it's not empty.
pub fn parse_extracted_items(raw: &str) -> Vec<ExtractedItem> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return vec![];
    }
    // Try strict JSON parse first.
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        if let Some(arr) = value.as_array() {
            return arr
                .iter()
                .filter_map(|v| {
                    if let Some(content) = v.as_str() {
                        let trimmed_str = content.trim();
                        if trimmed_str.is_empty() {
                            return None;
                        }
                        return Some(ExtractedItem {
                            content: trimmed_str.to_string(),
                            confidence: None,
                        });
                    }
                    let obj = v.as_object()?;
                    let content = obj.get("content").and_then(|v| v.as_str())?;
                    let trimmed_content = content.trim();
                    if trimmed_content.is_empty() {
                        return None;
                    }
                    let confidence = obj
                        .get("confidence")
                        .and_then(|v| v.as_f64())
                        .map(|f| f as f32);
                    Some(ExtractedItem {
                        content: trimmed_content.to_string(),
                        confidence,
                    })
                })
                .collect();
        }
        if let Some(content) = value.as_str() {
            let trimmed_str = content.trim();
            if !trimmed_str.is_empty() {
                return vec![ExtractedItem {
                    content: trimmed_str.to_string(),
                    confidence: None,
                }];
            }
        }
    }
    // Fallback: treat the whole response as one item.
    vec![ExtractedItem {
        content: trimmed.to_string(),
        confidence: None,
    }]
}

fn build_derived_record(
    candidate: &Candidate,
    target_kind: ExtractKind,
    item: &ExtractedItem,
    item_index: usize,
    instance: Option<&str>,
) -> AnamnesisRecord {
    let source_record = &candidate.record;
    // Each derived record's `native_id` encodes the source + item index
    // so re-running Stage 2 on the same candidate doesn't duplicate.
    let native_id = format!(
        "{}#stage2:{}:{item_index}",
        source_record.id.0,
        target_kind.as_str()
    );
    let id = RecordId::from_parts(EXTRACTOR_ADAPTER_ID, instance, &native_id);
    let raw_hash = blake3::hash(item.content.as_bytes()).to_hex().to_string();

    let mut metadata = serde_json::Map::new();
    metadata.insert("extractor_stage".into(), Value::String("stage2".into()));
    metadata.insert(
        "extractor_target_kind".into(),
        Value::String(target_kind.as_str().into()),
    );
    if let Some(c) = item.confidence {
        metadata.insert(
            "extractor_confidence".into(),
            Value::Number(serde_json::Number::from_f64(c as f64).unwrap_or_else(|| 0.into())),
        );
    }
    metadata.insert(
        "extractor_source_adapter".into(),
        Value::String(source_record.source.adapter.clone()),
    );
    if let Some(inst) = source_record.source.instance.as_deref() {
        metadata.insert(
            "extractor_source_instance".into(),
            Value::String(inst.into()),
        );
    }

    // Scope: extracted records adopt the scope of the source episode —
    // an episode about a project produces project-scoped facts, a
    // user-scoped persona produces user-scoped facts, etc.
    let scope = source_record.scope;

    AnamnesisRecord {
        id,
        source: SourceDescriptor {
            adapter: EXTRACTOR_ADAPTER_ID.into(),
            instance: instance.map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        content: item.content.clone(),
        embedding: None,
        scope,
        kind: target_kind.target_kind(),
        created_at: Utc::now(),
        updated_at: None,
        tags: vec!["stage2-extracted".into(), target_kind.as_str().into()],
        metadata,
        provenance: Provenance {
            native_id,
            native_path: Some(format!(
                "stage2:{}:{}#{}",
                target_kind.as_str(),
                source_record.id.0,
                item_index
            )),
            captured_at: Utc::now(),
            raw_hash,
            derived_from: Some(source_record.id.clone()),
        },
        schema_version: SCHEMA_VERSION,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::MockProvider;
    use anamnesis_core::model::{
        Kind as CoreKind, Provenance, Scope as CoreScope, SourceDescriptor,
    };

    fn fake_candidate(content: &str) -> Candidate {
        let id = RecordId::from_parts("claude-code", None, content);
        Candidate {
            record: AnamnesisRecord {
                id,
                source: SourceDescriptor {
                    adapter: "claude-code".into(),
                    instance: None,
                    version: "0".into(),
                },
                content: content.into(),
                embedding: None,
                scope: CoreScope::Session,
                kind: CoreKind::Episode,
                created_at: Utc::now(),
                updated_at: None,
                tags: vec![],
                metadata: serde_json::Map::new(),
                provenance: Provenance {
                    native_id: "n".into(),
                    native_path: None,
                    captured_at: Utc::now(),
                    raw_hash: "h".into(),
                    derived_from: None,
                },
                schema_version: SCHEMA_VERSION,
            },
            score: 0.8,
            rationale: vec![],
        }
    }

    #[test]
    fn parse_extracted_items_handles_strict_array_of_objects() {
        let raw = r#"[{"content":"a","confidence":0.9},{"content":"b","confidence":0.4}]"#;
        let items = parse_extracted_items(raw);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].content, "a");
        assert!((items[0].confidence.unwrap() - 0.9).abs() < 1e-3);
        assert_eq!(items[1].content, "b");
    }

    #[test]
    fn parse_extracted_items_handles_array_of_strings() {
        let raw = r#"["fact one","fact two"]"#;
        let items = parse_extracted_items(raw);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].content, "fact one");
        assert!(items[0].confidence.is_none());
    }

    #[test]
    fn parse_extracted_items_handles_plain_text_fallback() {
        let raw = "Not JSON at all";
        let items = parse_extracted_items(raw);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "Not JSON at all");
    }

    #[test]
    fn parse_extracted_items_drops_empty_strings_inside_array() {
        let raw = r#"["", "real", "   "]"#;
        let items = parse_extracted_items(raw);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "real");
    }

    #[test]
    fn parse_extracted_items_drops_empty_objects() {
        let raw = r#"[{"content":""}, {"content":"keep"}]"#;
        let items = parse_extracted_items(raw);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "keep");
    }

    #[test]
    fn parse_extracted_items_empty_array_yields_zero_items() {
        assert!(parse_extracted_items("[]").is_empty());
    }

    #[test]
    fn parse_extracted_items_whitespace_only_yields_zero() {
        assert!(parse_extracted_items("   \n\n  ").is_empty());
    }

    #[test]
    fn build_derived_record_carries_lineage() {
        let candidate = fake_candidate("source episode body");
        let source_id = candidate.record.id.clone();
        let item = ExtractedItem {
            content: "extracted fact".into(),
            confidence: Some(0.75),
        };
        let rec = build_derived_record(&candidate, ExtractKind::Fact, &item, 0, Some("inst"));
        assert_eq!(rec.kind, CoreKind::Fact);
        assert_eq!(rec.scope, CoreScope::Session); // adopts source scope
        assert_eq!(rec.source.adapter, EXTRACTOR_ADAPTER_ID);
        assert_eq!(rec.content, "extracted fact");
        // Lineage link must point at the source.
        assert_eq!(
            rec.provenance.derived_from.as_ref().map(|r| &r.0),
            Some(&source_id.0)
        );
        // Tags + metadata flag this as Stage 2 output.
        assert!(rec.tags.contains(&"stage2-extracted".into()));
        assert!(rec.tags.contains(&"fact".into()));
        assert_eq!(
            rec.metadata.get("extractor_stage").and_then(|v| v.as_str()),
            Some("stage2")
        );
        assert_eq!(
            rec.metadata
                .get("extractor_source_adapter")
                .and_then(|v| v.as_str()),
            Some("claude-code")
        );
    }

    #[tokio::test]
    async fn run_stage2_against_mock_emits_one_record_per_candidate() {
        let provider = MockProvider::default_instance();
        let candidates = vec![
            fake_candidate("episode A about Paris"),
            fake_candidate("episode B about Tokyo"),
        ];
        let report = run_stage2(&provider, &candidates, ExtractKind::Fact, None)
            .await
            .unwrap();
        // MockProvider returns one item per prompt → 2 records.
        assert_eq!(report.records.len(), 2);
        assert_eq!(report.skipped, 0);
        assert!(report.errors.is_empty());
        assert!(report.estimated_input_tokens > 0);
        // Each derived record links back to its source.
        for (rec, candidate) in report.records.iter().zip(candidates.iter()) {
            assert_eq!(
                rec.provenance.derived_from.as_ref().map(|r| &r.0),
                Some(&candidate.record.id.0)
            );
        }
    }

    #[tokio::test]
    async fn run_stage2_empty_candidates_returns_empty_report() {
        let provider = MockProvider::default_instance();
        let report = run_stage2(&provider, &[], ExtractKind::Skill, None)
            .await
            .unwrap();
        assert!(report.records.is_empty());
        assert_eq!(report.skipped, 0);
        assert_eq!(report.estimated_input_tokens, 0);
    }
}
