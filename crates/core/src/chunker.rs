//! Chunker — turn an `AnamnesisRecord`'s content into one or more `Chunk`s
//! suitable for FTS5 and vector indexing.
//!
//! Phase-1 strategy (BLUEPRINT §16.3, decision B3):
//!
//!   - Short records (≤ `max_tokens`) → exactly one chunk, no work.
//!   - Long records → split greedily on the strongest available boundary:
//!     paragraph (`\n\n`) → line (`\n`) → sentence (`. ! ? 。 ！ ？`) →
//!     word (whitespace) → character. Each chunk is built to fit
//!     `max_tokens` and is at least `min_tokens` (to avoid degenerate tails).
//!   - Token estimate is a script-aware heuristic — exact tokenizers are
//!     model-specific and we'd rather not pin one in `core`. The store uses
//!     this only for budget enforcement and observability, not for cache
//!     keys (those use `ContentHash`).

use crate::chunk::Chunk;
use crate::model::RecordId;

/// Chunker configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkerConfig {
    /// Maximum tokens per chunk. Default: 512 — a comfortable budget for
    /// every model in the curated registry (e.g. multilingual-e5-small
    /// accepts 512).
    pub max_tokens: u32,
    /// Smallest acceptable chunk in tokens. Used to avoid tiny trailing
    /// chunks when a record is just over `max_tokens`. Default: 32.
    pub min_tokens: u32,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            min_tokens: 32,
        }
    }
}

/// The chunker.
#[derive(Debug, Clone, Copy)]
pub struct Chunker {
    config: ChunkerConfig,
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new(ChunkerConfig::default())
    }
}

impl Chunker {
    /// Build a chunker with explicit config.
    pub fn new(config: ChunkerConfig) -> Self {
        Self { config }
    }

    /// Split `content` for `record_id`, returning at least one chunk.
    ///
    /// An empty `content` still produces one chunk (with empty text). This
    /// keeps the invariant "every record has ≥ 1 chunk" — callers do not
    /// have to special-case empties.
    pub fn chunk(&self, record_id: &RecordId, content: &str) -> Vec<Chunk> {
        let total = estimate_tokens(content);
        if total <= self.config.max_tokens {
            return vec![Chunk::new(record_id.clone(), 0, content.to_string(), total)];
        }
        let segments = split_within_budget(content, self.config.max_tokens, self.config.min_tokens);
        segments
            .into_iter()
            .enumerate()
            .map(|(i, text)| {
                let tokens = estimate_tokens(&text);
                Chunk::new(record_id.clone(), i as u32, text, tokens)
            })
            .collect()
    }
}

/// Script-aware token heuristic.
///
/// - Every CJK / Japanese / Korean character counts as ~1 token (these
///   scripts pack a lot of information per character).
/// - Non-CJK runs contribute `chars / 4 + 1` (the classic ~4-chars-per-
///   token English approximation).
///
/// This is intentionally cheap and stable; we don't pull in a real
/// tokenizer at the `core` layer.
pub fn estimate_tokens(s: &str) -> u32 {
    if s.is_empty() {
        return 0;
    }
    let mut cjk = 0usize;
    let mut non_cjk_chars = 0usize;
    for c in s.chars() {
        if is_dense_script(c) {
            cjk += 1;
        } else {
            non_cjk_chars += 1;
        }
    }
    let non_cjk_tokens = if non_cjk_chars == 0 {
        0
    } else {
        non_cjk_chars.div_ceil(4)
    };
    (cjk + non_cjk_tokens) as u32
}

fn is_dense_script(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x309F |        // Hiragana
        0x30A0..=0x30FF |        // Katakana
        0x3400..=0x4DBF |        // CJK Ext A
        0x4E00..=0x9FFF |        // CJK Unified
        0xAC00..=0xD7AF |        // Hangul Syllables
        0xF900..=0xFAFF          // CJK Compatibility
    )
}

/// Greedy splitter that respects boundary strength and the token budget.
fn split_within_budget(content: &str, max_tokens: u32, min_tokens: u32) -> Vec<String> {
    // Try boundaries from strongest to weakest. As soon as every segment
    // fits the budget, stop. If we exhaust boundaries, character-level
    // emergency split below guarantees termination.
    for boundary in BOUNDARIES {
        let pieces = split_by(content, boundary.sep);
        if pieces.iter().all(|p| estimate_tokens(p) <= max_tokens) {
            return coalesce(pieces, max_tokens, min_tokens, boundary.glue);
        }
    }
    // Last resort: hard char-window split.
    hard_split_by_chars(content, max_tokens)
}

struct Boundary {
    sep: &'static [char],
    /// Glue used when recombining adjacent pieces. Reflects what `split_by`
    /// removed.
    glue: &'static str,
}

const BOUNDARIES: &[Boundary] = &[
    // Paragraph break.
    Boundary {
        sep: &['\u{2029}'], // we manually detect "\n\n" below; this is a sentinel
        glue: "\n\n",
    },
    // Line break.
    Boundary {
        sep: &['\n'],
        glue: "\n",
    },
    // Sentence terminators (ASCII + CJK).
    Boundary {
        sep: &['.', '!', '?', '。', '！', '？'],
        glue: " ",
    },
    // Whitespace.
    Boundary {
        sep: &[' ', '\t'],
        glue: " ",
    },
];

/// Splits text by any of `sep`. Special case: the first boundary's sep is
/// the paragraph sentinel; we treat "\n\n" as one break.
fn split_by(text: &str, sep: &[char]) -> Vec<String> {
    if sep == ['\u{2029}'].as_slice() {
        // Paragraph mode: split on double-newline.
        return text
            .split("\n\n")
            .map(str::to_string)
            .filter(|p| !p.is_empty())
            .collect();
    }
    text.split(|c: char| sep.contains(&c))
        .map(str::to_string)
        .filter(|p| !p.is_empty())
        .collect()
}

/// Recombine adjacent pieces using `glue` until each combined chunk is
/// either at the budget or about to exceed it.
fn coalesce(pieces: Vec<String>, max_tokens: u32, min_tokens: u32, glue: &str) -> Vec<String> {
    if pieces.is_empty() {
        return vec![String::new()];
    }
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut buf_tokens: u32 = 0;
    for piece in pieces {
        let piece_tokens = estimate_tokens(&piece);
        if buf.is_empty() {
            buf = piece;
            buf_tokens = piece_tokens;
            continue;
        }
        let glued_tokens = buf_tokens + piece_tokens + estimate_tokens(glue);
        if glued_tokens <= max_tokens {
            buf.push_str(glue);
            buf.push_str(&piece);
            buf_tokens = glued_tokens;
        } else {
            // Avoid stranded micro-chunks: if appending would still leave
            // buf below min_tokens, pack it in even if we overflow slightly.
            if buf_tokens < min_tokens {
                buf.push_str(glue);
                buf.push_str(&piece);
                out.push(std::mem::take(&mut buf));
                buf_tokens = 0;
            } else {
                out.push(std::mem::take(&mut buf));
                buf = piece;
                buf_tokens = piece_tokens;
            }
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

fn hard_split_by_chars(content: &str, max_tokens: u32) -> Vec<String> {
    // Conservative inverse of estimate_tokens: assume worst case = 1 char
    // per token (all CJK). One chunk holds max_tokens chars.
    let window = max_tokens.max(1) as usize;
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut count = 0usize;
    for c in content.chars() {
        buf.push(c);
        count += 1;
        if count >= window {
            out.push(std::mem::take(&mut buf));
            count = 0;
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    if out.is_empty() {
        vec![String::new()]
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rid() -> RecordId {
        RecordId::from_parts("claude-code", None, "test")
    }

    #[test]
    fn empty_content_yields_single_empty_chunk() {
        let c = Chunker::default().chunk(&rid(), "");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].seq, 0);
        assert_eq!(c[0].content, "");
        assert_eq!(c[0].token_estimate, 0);
    }

    #[test]
    fn short_content_is_single_chunk() {
        let c = Chunker::default().chunk(&rid(), "hello world, this is short");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].seq, 0);
        assert!(c[0].token_estimate <= 32);
    }

    #[test]
    fn estimate_tokens_handles_empty_and_short() {
        assert_eq!(estimate_tokens(""), 0);
        assert!(estimate_tokens("a") >= 1);
        assert!(estimate_tokens("hello world") >= 2);
    }

    #[test]
    fn estimate_tokens_counts_cjk_per_char() {
        // 6 CJK chars → ≥ 6 tokens; same character count in English is much fewer.
        let cjk = estimate_tokens("用户偏好简洁");
        assert!(cjk >= 6, "expected ≥6 tokens for 6 CJK chars, got {cjk}");
        let eng = estimate_tokens("brief");
        assert!(eng < cjk);
    }

    #[test]
    fn long_paragraphs_split_on_double_newline() {
        // Two ~300-token paragraphs joined by \n\n; total >512 → must split.
        let para = "word ".repeat(300); // ~300 words → ~375+ tokens
        let content = format!("{para}\n\n{para}");
        let chunks = Chunker::default().chunk(&rid(), &content);
        assert!(
            chunks.len() >= 2,
            "expected ≥2 chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert!(
                c.token_estimate <= 512,
                "chunk over budget: {} tokens",
                c.token_estimate
            );
        }
    }

    #[test]
    fn extremely_long_single_word_falls_back_to_hard_split() {
        // No paragraph / line / sentence / space boundaries available.
        let content = "a".repeat(5000); // ~1250 tokens at 4 chars/token
        let chunks = Chunker::default().chunk(&rid(), &content);
        assert!(chunks.len() >= 2);
        // Every chunk fits the budget (allowing for the conservative
        // 1-char-per-token inverse used by hard_split_by_chars).
        for c in &chunks {
            assert!(c.token_estimate <= 512);
        }
        // No content lost.
        let rejoined: String = chunks.iter().map(|c| c.content.clone()).collect();
        assert_eq!(rejoined.len(), 5000);
    }

    #[test]
    fn seq_is_zero_indexed_and_monotonic() {
        let content = "x. ".repeat(1000); // many sentences, > budget
        let chunks = Chunker::default().chunk(&rid(), &content);
        assert!(chunks.len() > 1);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.seq, i as u32);
        }
    }

    #[test]
    fn content_hash_is_set_per_chunk() {
        let chunks = Chunker::default().chunk(&rid(), "hello");
        assert_eq!(
            chunks[0].content_hash,
            crate::chunk::ContentHash::of("hello")
        );
    }

    #[test]
    fn chunker_is_deterministic() {
        let content = "alpha beta gamma. delta epsilon.\n\nzeta.\n\neta theta iota.";
        let a = Chunker::default().chunk(&rid(), content);
        let b = Chunker::default().chunk(&rid(), content);
        assert_eq!(a, b);
    }

    #[test]
    fn custom_config_respected() {
        let small = Chunker::new(ChunkerConfig {
            max_tokens: 10,
            min_tokens: 2,
        });
        let chunks = small.chunk(
            &rid(),
            "one two three\n\nfour five six\n\nseven eight nine ten",
        );
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.token_estimate <= 10);
        }
    }
}
