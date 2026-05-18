//! Per-`ExtractKind` prompt templates.
//!
//! Stage 2 wraps each candidate's Episode content in a kind-specific
//! prompt and asks the LLM to return JSON. The shape we expect back is:
//!
//! ```json
//! [
//!   {"content": "...", "confidence": 0.0–1.0}
//! ]
//! ```
//!
//! The driver tolerates degraded output (e.g. plain text instead of
//! JSON) — see [`crate::stage2::parse_extracted_items`].

use crate::ExtractKind;

/// Build the full Stage 2 prompt for one candidate Episode.
///
/// The prompt is intentionally short and prescriptive — Stage 2 is
/// **not** the place for clever chain-of-thought, multi-turn refinement,
/// or self-critique. Anything more elaborate belongs in a follow-up
/// PR and a feature flag.
pub fn build_prompt(kind: ExtractKind, episode_content: &str) -> String {
    let header = kind_header(kind);
    format!(
        "{header}\n\n\
         ## Episode\n\n\
         {episode_content}\n\n\
         ## Output\n\n\
         Return a JSON array. Each item must have `content` (string) \
         and `confidence` (0.0–1.0). If no items can be confidently \
         extracted, return an empty array `[]`. Do not include \
         commentary outside the JSON."
    )
}

fn kind_header(kind: ExtractKind) -> &'static str {
    match kind {
        ExtractKind::Fact => {
            "You are a memory extractor. Read the Episode below and extract \
             stable, third-person facts about the user or the world. Skip \
             anything ephemeral or already obvious from context."
        }
        ExtractKind::Preference => {
            "You are a memory extractor. Read the Episode below and extract \
             first-person preferences — taste, style, workflow choices the \
             user expressed explicitly. Skip momentary reactions; we want \
             durable preferences."
        }
        ExtractKind::Feedback => {
            "You are a memory extractor. Read the Episode below and extract \
             corrections or feedback the user gave to the agent — things \
             that should change how the agent behaves next time. Skip \
             agent-side errors that the user didn't actually flag."
        }
        ExtractKind::Skill => {
            "You are a memory extractor. Read the Episode below and extract \
             repeatable how-to / procedure / workflow patterns. Each \
             extracted skill should be re-executable by the agent later. \
             Skip one-off discussions that aren't procedural."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_prompt_includes_episode_content() {
        let p = build_prompt(ExtractKind::Fact, "user said the sky is blue");
        assert!(p.contains("user said the sky is blue"));
    }

    #[test]
    fn build_prompt_is_kind_specific() {
        let fact_prompt = build_prompt(ExtractKind::Fact, "x");
        let pref_prompt = build_prompt(ExtractKind::Preference, "x");
        let feedback_prompt = build_prompt(ExtractKind::Feedback, "x");
        let skill_prompt = build_prompt(ExtractKind::Skill, "x");
        // Each kind's header word should appear in its own prompt and
        // not in any other.
        assert!(fact_prompt.contains("facts"));
        assert!(pref_prompt.contains("preferences"));
        assert!(feedback_prompt.contains("feedback"));
        assert!(skill_prompt.contains("how-to"));
        assert!(!fact_prompt.contains("how-to"));
        assert!(!pref_prompt.contains("third-person facts"));
    }

    #[test]
    fn build_prompt_asks_for_strict_json() {
        let p = build_prompt(ExtractKind::Fact, "x");
        assert!(p.contains("JSON array"));
        assert!(p.contains("confidence"));
        assert!(p.contains("`[]`"));
        assert!(p.contains("Do not include"));
    }
}
