//! CJK-aware FTS tokenization for Anamnesis.
//!
//! ## Why this exists
//!
//! SQLite FTS5 ships with `unicode61`, which segments by codepoint. For
//! CJK that means **every Han character becomes its own token** — so
//! BM25 can no longer distinguish "项目偏好" from any other 2-char Han
//! sequence containing 项 and 偏. Cross-agent memory recall is broken
//! for the Chinese user the moment they type in their native language.
//!
//! Anamnesis is *own-RAG infrastructure*; we cannot delegate this to a
//! third party. The fix is application-layer pre-tokenization with
//! jieba — the same strategy the ghast client uses (see
//! `src/main/db/database-runtime-helpers.ts::ftsTokenize`).
//!
//! ## Architecture
//!
//! - `tokenize_indexing(text)`: jieba `cut_for_search`, drop punctuation /
//!   whitespace, dedupe, return a single space-joined token stream. Stored
//!   verbatim in `chunks_fts.content` via a SQLite trigger.
//! - `tokenize_query(text)`: same tokenization, then wrap each token in
//!   `"..."` and join with spaces to form a valid FTS5 MATCH query
//!   (implicit AND between phrases). Used at search time.
//!
//! The pair is symmetric: the indexed stream and the query stream agree
//! on what a "token" is, so a Chinese phrase typed at the prompt finds
//! the chunks where jieba split out the same words.
//!
//! Both functions are pure / cheap; jieba's `Jieba::new()` is allocated
//! once behind a `OnceLock` to amortise startup. ASCII-only input falls
//! through quickly because `cut_for_search` recognises whitespace word
//! boundaries directly.

use std::sync::OnceLock;

use jieba_rs::Jieba;

/// Lazy-initialised global jieba instance. `Jieba::new()` builds the
/// default dictionary in roughly tens of milliseconds — too slow to do
/// on every chunk insert; harmless to share across threads (Jieba is
/// `Sync`).
fn jieba() -> &'static Jieba {
    static INSTANCE: OnceLock<Jieba> = OnceLock::new();
    INSTANCE.get_or_init(Jieba::new)
}

/// Is this character useful as part of a search token?
///
/// Matches the same shape as ghast's `isSearchToken`:
///   `[\p{L}\p{N}_]` — Unicode letter / number / underscore.
fn is_search_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Returns `true` when *every* character in the token is searchable.
/// Mixed punctuation tokens (e.g. `","` from a jieba split) are dropped.
fn is_search_token(token: &str) -> bool {
    !token.is_empty() && token.chars().all(is_search_char)
}

/// Tokenize text for **storage** in the FTS index.
///
/// Returns a single space-joined string of unique search tokens. Order
/// preserves first-seen position (so BM25 still has positional signal
/// for natural-language input).
///
/// Empty / whitespace input returns the empty string; the FTS row will
/// simply have no terms.
pub fn tokenize_indexing(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut seen = std::collections::HashSet::<String>::new();
    let mut out: Vec<String> = Vec::new();
    for tok in jieba().cut_for_search(trimmed, true) {
        let t = tok.trim();
        if !is_search_token(t) {
            continue;
        }
        if seen.insert(t.to_owned()) {
            out.push(t.to_owned());
        }
    }
    out.join(" ")
}

/// Question / filler words that carry no retrieval signal. An agent
/// rarely types keywords — it asks "what does the user prefer about X".
/// AND-ing those words against the index makes natural-language queries
/// recall zero, so [`plan_query`] drops them from the salient set. Kept
/// deliberately small + conservative: only words that are almost never
/// the thing you're searching *for*. Compared lowercased.
const STOPWORDS_EN: &[&str] = &[
    "a", "about", "an", "and", "any", "are", "as", "did", "do", "does", "find", "for", "get",
    "give", "how", "i", "in", "is", "it", "know", "me", "my", "of", "on", "or", "please", "show",
    "tell", "that", "the", "their", "them", "they", "this", "to", "user", "users", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
];

/// CJK question / filler words, the Chinese counterpart to [`STOPWORDS_EN`].
/// jieba segments these as standalone tokens, so they're easy to drop.
const STOPWORDS_ZH: &[&str] = &[
    "什么", "哪些", "哪个", "如何", "怎么", "怎样", "请", "帮我", "是", "的", "了", "吗", "呢",
    "我", "我的", "你", "有", "关于", "这个", "那个", "找", "找出", "显示", "告诉", "用户",
];

/// Is `token` a question / filler word with no retrieval value?
fn is_stopword(token: &str) -> bool {
    let lower = token.to_lowercase();
    STOPWORDS_EN.contains(&lower.as_str()) || STOPWORDS_ZH.contains(&token)
}

/// Tokenize + quote into FTS5 phrase tokens (`"foo"`), first-seen order,
/// deduped. When `drop_stopwords` is set, question / filler words are
/// removed (used for the salient query plan); otherwise every search
/// token is kept (used by [`tokenize_query`]).
fn quoted_phrase_tokens(text: &str, drop_stopwords: bool) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let mut seen = std::collections::HashSet::<String>::new();
    let mut out: Vec<String> = Vec::new();
    for tok in jieba().cut_for_search(trimmed, true) {
        let t = tok.trim();
        if !is_search_token(t) {
            continue;
        }
        if drop_stopwords && is_stopword(t) {
            continue;
        }
        if seen.insert(t.to_owned()) {
            // Double-quote escape rule: replace `"` with `""` and wrap.
            let escaped = t.replace('"', "\"\"");
            out.push(format!("\"{escaped}\""));
        }
    }
    out
}

/// Tokenize text for an FTS5 **MATCH query**.
///
/// Wraps each token in `"..."` (escaping embedded `"` as `""`) and joins
/// with spaces. FTS5 treats space-separated quoted phrases as an
/// implicit AND, which is what users expect when they type multiple
/// words. Keeps every search token (no stopword removal) — for the
/// natural-language strict/relaxed plan use [`plan_query`].
///
/// Empty input returns the empty string; callers must check and skip
/// the MATCH (FTS5 errors on empty queries).
pub fn tokenize_query(text: &str) -> String {
    quoted_phrase_tokens(text, false).join(" ")
}

/// A two-tier FTS5 MATCH plan for a natural-language query (R160).
///
/// `search_chunks_fts` runs [`strict`](Self::strict) first (precise), then
/// — only if it under-fills the limit — tops up with [`relaxed`](Self::relaxed)
/// (recall). The change is purely **additive**: `strict` keeps the exact
/// AND-of-all-tokens semantics keyword search always had (so precise
/// queries don't regress), and stopword stripping only shapes the recall
/// tail. Splitting the decision out keeps it pure and testable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FtsQueryPlan {
    /// AND of **every** search token (`"the" "user" "shell"`), exactly like
    /// the original keyword tokenizer. Empty only when the query has no
    /// searchable tokens at all.
    pub strict: String,
    /// OR of the **salient** tokens — question/filler words dropped
    /// (`"prefer" OR "shell"`). The recall fallback for natural-language
    /// queries the strict AND can't satisfy. Empty when it would add
    /// nothing over `strict` (a single-token query).
    pub relaxed: String,
}

/// Build a [`FtsQueryPlan`] from raw user text.
///
/// `strict` = AND of all tokens (unchanged keyword behavior). `relaxed` =
/// OR of the salient tokens (stopwords removed); if every token is a
/// stopword (e.g. "what is it") it falls back to the full set so recall is
/// never silently empty. `relaxed` is omitted only for a single-token
/// query, where OR == AND and the second pass would be wasted work.
pub fn plan_query(text: &str) -> FtsQueryPlan {
    let all = quoted_phrase_tokens(text, false);
    let salient = quoted_phrase_tokens(text, true);
    let strict = all.join(" ");

    let or_tokens = if salient.is_empty() { &all } else { &salient };
    let relaxed = if or_tokens.is_empty() || (all.len() == 1 && or_tokens.len() == 1) {
        // No tokens, or a single token where OR and AND are identical.
        String::new()
    } else {
        or_tokens.join(" OR ")
    };
    FtsQueryPlan { strict, relaxed }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_passes_through_word_boundaries() {
        let tokens = tokenize_indexing("hello world");
        assert!(tokens.contains("hello"));
        assert!(tokens.contains("world"));
    }

    #[test]
    fn chinese_phrase_segments_into_words() {
        // "项目偏好" should split into something containing the word
        // "项目" and "偏好" rather than 4 separate Han characters.
        let tokens = tokenize_indexing("我的项目偏好");
        // We don't assert exact split (jieba dict may vary) — only that
        // a multi-char Chinese word survived as a single token.
        let any_multi_char_token = tokens
            .split_whitespace()
            .any(|t| t.chars().filter(|c| !c.is_ascii()).count() >= 2);
        assert!(
            any_multi_char_token,
            "expected at least one multi-char Chinese token in {tokens:?}"
        );
    }

    #[test]
    fn punctuation_is_dropped() {
        let tokens = tokenize_indexing("hello, world!");
        let toks: Vec<_> = tokens.split_whitespace().collect();
        assert!(!toks.iter().any(|t| t.contains(',')));
        assert!(!toks.iter().any(|t| t.contains('!')));
    }

    #[test]
    fn dedup_preserves_first_position() {
        let tokens = tokenize_indexing("alpha beta alpha gamma alpha");
        assert_eq!(tokens, "alpha beta gamma");
    }

    #[test]
    fn query_form_quotes_each_token() {
        let q = tokenize_query("项目 偏好");
        // Each token wrapped, joined with spaces.
        let parts: Vec<_> = q.split_whitespace().collect();
        assert!(!parts.is_empty());
        for p in &parts {
            assert!(p.starts_with('"') && p.ends_with('"'), "bad quote: {p}");
        }
    }

    #[test]
    fn query_form_escapes_embedded_quote() {
        // A literal `"` inside a token must become `""` per FTS5 quoting.
        // We synthesise this by tokenizing a string that includes one.
        let q = tokenize_query(r#"say "hi""#);
        // After jieba + filter, `"` itself is dropped (not is_search_char).
        // So we only check that the surviving tokens are well-quoted and
        // no raw unescaped `"` leaks out of token boundaries.
        for tok in q.split_whitespace() {
            // Tokens always start and end with a quote.
            assert!(tok.starts_with('"') && tok.ends_with('"'));
            // No empty quoted tokens.
            assert!(tok.len() >= 2);
        }
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(tokenize_indexing("").is_empty());
        assert!(tokenize_indexing("   ").is_empty());
        assert!(tokenize_query("").is_empty());
        assert!(tokenize_query("   ").is_empty());
    }

    #[test]
    fn mixed_chinese_english_round_trip_via_query() {
        // ghast pattern: index a doc, then a substring of its Chinese
        // tokens should produce a query that overlaps the indexed terms.
        let indexed = tokenize_indexing("Anamnesis 是跨 agent 记忆基础设施");
        let query = tokenize_query("记忆");
        // The query has at least one quoted token; that token (without
        // quotes) must appear in the indexed stream as a word boundary.
        let q_inner: String = query
            .trim_matches('"')
            .chars()
            .take_while(|c| *c != '"')
            .collect();
        assert!(
            indexed.split_whitespace().any(|w| w == q_inner),
            "indexed stream {indexed:?} should contain query token {q_inner:?}"
        );
    }

    // ─── R160: natural-language query planning ────────────────────────

    #[test]
    fn strict_keeps_all_tokens_no_regression() {
        // strict must stay AND-of-all (incl. stopwords) so precise keyword
        // queries don't silently broaden. Only `relaxed` drops fillers.
        let plan = plan_query("user shell");
        assert_eq!(plan.strict, "\"user\" \"shell\"");
    }

    #[test]
    fn relaxed_drops_english_question_words() {
        let plan = plan_query("what does the user prefer about shell");
        // Salient terms survive in the recall tail; filler words are gone.
        assert!(plan.relaxed.contains("\"prefer\""));
        assert!(plan.relaxed.contains("\"shell\""));
        assert!(!plan.relaxed.contains("\"what\""));
        assert!(!plan.relaxed.contains("\"does\""));
        assert!(!plan.relaxed.contains("\"the\""));
        assert!(!plan.relaxed.contains("\"about\""));
        // "user" is a filler word here too (the agent, not the content).
        assert!(!plan.relaxed.contains("\"user\""));
        // strict still carries every token verbatim.
        assert!(plan.strict.contains("\"what\"") && plan.strict.contains("\"user\""));
    }

    #[test]
    fn relaxed_drops_chinese_question_words() {
        let plan = plan_query("用户偏好是什么");
        // jieba keeps 偏好; 是 / 什么 / 用户 are dropped from the recall tail.
        assert!(plan.relaxed.contains("\"偏好\""), "plan was {plan:?}");
        assert!(!plan.relaxed.contains("\"是\""));
        assert!(!plan.relaxed.contains("\"什么\""));
    }

    #[test]
    fn plan_strict_is_and_relaxed_is_or() {
        let plan = plan_query("prefer shell");
        // No stopwords: strict AND-joins, relaxed OR-joins the same tokens.
        assert_eq!(plan.strict, "\"prefer\" \"shell\"");
        assert_eq!(plan.relaxed, "\"prefer\" OR \"shell\"");
    }

    #[test]
    fn plan_single_token_has_no_relaxed_tail() {
        // One token total: OR == AND, so relaxed is omitted as wasted work.
        let one = plan_query("prefer");
        assert_eq!(one.strict, "\"prefer\"");
        assert_eq!(one.relaxed, "");
        // But a single SALIENT token with dropped fillers still gets a
        // relaxed tail, because it's broader than the AND-of-all strict.
        let nl = plan_query("what is the shell");
        assert_eq!(nl.relaxed, "\"shell\"");
    }

    #[test]
    fn plan_all_stopwords_falls_back_to_literal() {
        // Every token is filler — must NOT yield an empty query (that would
        // silently return zero hits); strict keeps the literal words and
        // relaxed falls back to their OR.
        let plan = plan_query("what is the");
        assert!(!plan.strict.is_empty(), "plan was {plan:?}");
        assert!(plan.relaxed.contains(" OR "), "plan was {plan:?}");
    }
}
