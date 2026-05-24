//! `anamnesis` CLI entry point.

#![forbid(unsafe_code)]
// Aligned table headers are clearer than building format strings with
// inlined idents — clippy's literal-format-arg lint isn't load-bearing here.
#![allow(clippy::print_literal)]

// Round 140 (PR-78bi): R138/R139 introduced `mem0-sqlite` and
// `letta-sqlite` exporters inline in the CLI. R140 lifted the
// whole format catalogue into `anamnesis-export` so the CLI's
// `anamnesis export` and the new MCP `export_memories` tool share
// one implementation. The CLI is now just clap-parsing + audit
// glue around `anamnesis_export::run_export`.

// Round 141 (PR-78bj): `dedupe --mode near --merge-preview` —
// operator-decision tooling that closes the loop on R131/R132's
// detector. Picks a winner per group with a deterministic
// ranking heuristic; preview-only.
mod near_merge_preview;

use std::path::PathBuf;

use anamnesis_adapter_claude_code::{ClaudeCodeAdapter, ClaudeCodeConfig, ClaudeCodeDetector};
use anamnesis_adapter_codex::{codex_adapter, CodexDetector};
use anamnesis_adapter_hermes::{hermes_adapter, HermesDetector};
use anamnesis_adapter_letta::{letta_adapter, LettaSqliteDetector};
use anamnesis_adapter_mem0::{sqlite_adapter as mem0_sqlite_adapter, Mem0SqliteDetector};
use anamnesis_adapter_memary::{memary_adapter, MemaryDetector};
use anamnesis_adapter_memori::{memori_adapter, MemoriDetector};
use anamnesis_adapter_memos::{memos_adapter, MemosDetector};
use anamnesis_adapter_mempalace::{mempalace_adapter, MempalaceDetector};
use anamnesis_adapter_openclaw::{openclaw_adapter, OpenClawDetector};
use anamnesis_adapter_openviking::{openviking_adapter, OpenVikingDetector};
use anamnesis_adapter_tdai::{tdai_adapter, TdaiDetector};
use anamnesis_core::discovery::{DetectOpts, Discovery};
use anamnesis_embedder::registry;
use anamnesis_importer::{ImportOptions, ImportService};
use anamnesis_search::{pack, ContextBudget, HybridOpts, HybridSearcher, SearchMode};
use anamnesis_store::Store;
use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

/// Round 132 (PR-78ba): pick the dedupe detector. `exact` is the
/// R77 byte-equal `raw_hash` grouper (default, fully back-compat).
/// `near` is the R131 SimHash + LSH + Jaccard cross-adapter
/// near-duplicate detector. Default-on cross-source filter keeps
/// near surfaces aligned with the interoperability mission.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum DedupeMode {
    /// R77 raw_hash byte-equal grouping. Catches only identical
    /// source payloads; misses cross-adapter paraphrases.
    Exact,
    /// R131 SimHash + LSH + Jaccard. Catches cross-adapter
    /// paraphrases (mem0 vs claude-code on the same memory)
    /// that `exact` can't see. Defaults to cross-source-only
    /// groups; pass `--include-near-self` to also surface
    /// within-adapter near-dups.
    Near,
}

impl DedupeMode {
    /// Wire-shape label used in JSON / CSV payloads and human
    /// summaries so a script can branch on `payload.mode`.
    fn wire_label(self) -> &'static str {
        match self {
            DedupeMode::Exact => "exact",
            DedupeMode::Near => "near",
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "anamnesis",
    version,
    about = "Cross-agent memory layer (搜魂术)",
    long_about = None,
)]
struct Cli {
    /// Override data directory (defaults to XDG_DATA_HOME/anamnesis).
    #[arg(long, global = true, env = "ANAMNESIS_DATA_DIR")]
    data_dir: Option<PathBuf>,

    /// Override config file path (defaults to
    /// XDG_CONFIG_HOME/anamnesis/config.toml).
    #[arg(long, global = true, env = "ANAMNESIS_CONFIG")]
    config: Option<PathBuf>,

    /// Log level (trace, debug, info, warn, error).
    #[arg(long, global = true, env = "ANAMNESIS_LOG", default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Initialize a new data directory and database.
    Init {
        /// Override the default embedding model (curated key).
        #[arg(long)]
        model: Option<String>,
    },

    /// Show database stats and active model.
    Status {
        /// Emit JSON instead of the human-friendly table.
        #[arg(long)]
        json: bool,
    },

    /// Scan default paths for known memory sources (read-only).
    Discover,

    /// Manage configured memory sources.
    #[command(subcommand)]
    Source(SourceCmd),

    /// Run an import job for one source.
    Import {
        /// Adapter name, optionally `adapter:instance`.
        target: String,
        /// Full re-scan: ignore any incremental window (`--since` or
        /// the auto-detected `last_import_at`) and ask the adapter to
        /// re-emit every candidate record. Does NOT bypass the store's
        /// `raw_hash` dedup — re-emitting an unchanged record is still
        /// a no-op upsert (PR-#15 fast-path).
        #[arg(long, conflicts_with = "since")]
        full: bool,
        /// RFC3339 timestamp (e.g. `2026-04-01T00:00:00Z`) — only
        /// records modified after this point get imported. When neither
        /// `--full` nor `--since` is given, the importer falls back to
        /// `sources.last_import_at` for incremental imports.
        #[arg(long)]
        since: Option<String>,
        /// Print what would be imported instead of writing.
        #[arg(long)]
        dry_run: bool,
        /// Skip running the embedding worker after the import.
        #[arg(long)]
        no_embed: bool,
        /// Optional path override (e.g. mem0 SQLite file when the default
        /// `~/.mem0/db.sqlite` is wrong).
        #[arg(long)]
        path: Option<PathBuf>,
    },

    /// Search across all imported records.
    Search {
        /// Free-text query.
        query: String,
        /// Restrict to one source (adapter id).
        #[arg(long)]
        source: Option<String>,
        /// Restrict to a specific source instance. Meaningful only when
        /// `--source` is also set; the SQL key is `(adapter, instance)`.
        #[arg(long)]
        instance: Option<String>,
        /// Restrict to one Kind: fact | preference | feedback | reference | episode | skill | unknown.
        #[arg(long)]
        kind: Option<String>,
        /// Restrict to one Scope: user | project | session | ephemeral.
        #[arg(long)]
        scope: Option<String>,
        /// RFC3339 lower bound on records.created_at (inclusive). E.g.
        /// `2026-04-01T00:00:00Z`.
        #[arg(long)]
        since: Option<String>,
        /// RFC3339 upper bound on records.created_at (inclusive).
        #[arg(long)]
        until: Option<String>,
        /// Result limit.
        #[arg(long, default_value_t = 10)]
        limit: u32,
        /// Modality: fulltext | vector | hybrid (default = hybrid).
        #[arg(long, default_value = "hybrid")]
        mode: String,
        /// Emit JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
        /// Append per-stage search timings (`embed_query` / `fts` /
        /// `vec` / `rrf` / `pack`, in milliseconds) and candidate
        /// counts. Mirrors the MCP `search_memories(trace=true)`
        /// payload byte-for-byte; never includes the query text or
        /// snippet content. Default off, keeps the existing wire
        /// shape unchanged.
        #[arg(long)]
        trace: bool,
        /// Round 79 (PR-78b): restrict to records that carry this
        /// user tag (set via `anamnesis tag-record`). The filter
        /// pushes down into FTS, BLOB-vec fallback, and sqlite-vec
        /// at the SQL recall stage, so a single tagged record
        /// surfaces even in a 1700-untagged corpus. Tag is
        /// normalised the same way `tag-record` normalises writes
        /// (`trim().to_lowercase()`).
        #[arg(long)]
        user_tag: Option<String>,
        /// Round 87 (PR-78i): attach a per-result `explain` block
        /// breaking down the ranking arithmetic: best-chunk RRF
        /// score, kind boost, and the FTS / vector stage ranks +
        /// raw scores + `rrf_contribution = 1/(rrf_k + rank)`.
        /// Default off. Orthogonal to `--trace` (which reports
        /// stage *timings* and candidate counts).
        #[arg(long)]
        explain: bool,
    },

    /// Apply or remove user tags on a record (local overlay).
    ///
    /// User tags live in a separate table from the adapter-
    /// derived `records.tags`, so they survive `import` /
    /// re-`import` cycles. Read paths (`search`, `get_record`)
    /// surface them as `user_tags`; the adapter-derived `tags`
    /// stays untouched so source provenance is unambiguous.
    ///
    /// Set semantics: re-adding an existing tag or removing a
    /// missing one is a no-op (the JSON `changed` count tells
    /// the caller how many rows actually moved). Tags are
    /// trimmed + lower-cased + deduped before write.
    TagRecord {
        /// Record id (as returned by `anamnesis search` /
        /// `list-forgotten`).
        record_id: String,
        /// One or more tags to apply. Required for `add` /
        /// `--remove`; optional for `--replace` (an empty list
        /// means "clear all user tags on this record").
        #[arg(required_unless_present = "replace")]
        tags: Vec<String>,
        /// Remove the tags instead of adding them.
        #[arg(long, conflicts_with = "replace")]
        remove: bool,
        /// Round 81: atomically install the given tags as the
        /// **full** post-call set on this record — anything not
        /// in `tags` is deleted, anything new is inserted, all
        /// in one transaction. Passing no tags clears the
        /// overlay. Mutually exclusive with `--remove`.
        #[arg(long)]
        replace: bool,
        /// Emit JSON instead of the human one-liner.
        #[arg(long)]
        json: bool,
        /// Round 96: append a `stats` block (`total_user_tags`)
        /// to the JSON payload and a short stats line to the
        /// human output. Operators paging through tag mutations
        /// can see the post-call tag-count without a second
        /// `get_record` round trip. Default off — back-compat.
        #[arg(long)]
        include_stats: bool,
    },

    /// Report records that share an identical `raw_hash` — the
    /// source-payload byte-fingerprint. Useful when 13 adapters'
    /// worth of imports surface the same Slack message twice, or
    /// the same Markdown file got picked up under two project
    /// paths. Read-only diagnostic — no auto-merge. The operator
    /// picks which sibling to `forget`.
    ///
    /// **Default output is redacted**: `raw_hash` and `native_path`
    /// are omitted; only reported as `has_*` booleans (JSON) /
    /// hidden (human). Pass `--include-sensitive` to reveal.
    ///
    /// Two modes (Round 132): `--mode exact` (default, R77
    /// `raw_hash` byte-equal) and `--mode near` (R131
    /// SimHash + LSH + Jaccard cross-adapter near-duplicate
    /// detection). Near defaults to cross-source-only groups —
    /// the interop-relevant case raw_hash can't catch because
    /// adapters differ in punctuation / prefixes / tokenization.
    Dedupe {
        /// Round 132 (PR-78ba): pick the detector. `exact` is
        /// the R77 raw_hash byte-equal grouper (default, fully
        /// back-compat). `near` is the R131 SimHash + LSH +
        /// Jaccard cross-adapter near-duplicate detector and
        /// defaults to cross-source-only groups (the interop-
        /// relevant case raw_hash can't catch).
        #[arg(long, value_enum, default_value_t = DedupeMode::Exact)]
        mode: DedupeMode,
        /// Scope to duplicate groups that include ≥1 record from
        /// this adapter (e.g. `mem0`, `claude-code`). Round 104:
        /// also accepts a comma-separated OR list
        /// (`--source mem0,claude-code`) — groups whose members
        /// include at least one record from any listed adapter
        /// stay eligible. Tokens are trimmed and empty tokens
        /// dropped, so `--source mem0, , claude-code` is the
        /// same as `--source mem0,claude-code`. The full sibling
        /// set is still returned so you can see which
        /// non-matching records share the same `raw_hash`.
        #[arg(long)]
        source: Option<String>,
        /// Scope to duplicate groups that include ≥1 record from
        /// this instance. Round 115: comma-separated list is also
        /// accepted (`--instance prod,dev`) for an OR filter.
        /// Combines as AND with `--source` when both are set.
        #[arg(long)]
        instance: Option<String>,
        /// Max number of groups to return. Default 20, cap 100.
        #[arg(long, default_value_t = 20)]
        limit: u32,
        /// Emit JSON instead of the human table.
        #[arg(long)]
        json: bool,
        /// Reveal `raw_hash` and `native_path` fields. Default off.
        #[arg(long)]
        include_sensitive: bool,
        /// Round 97: append a `counts` block reporting filter-
        /// scoped totals — `total_groups` (duplicate groups
        /// matching the filter, ignoring `--limit`),
        /// `total_records` (sum of live records across those
        /// whole groups), and `by_source[]` (per-`(adapter,
        /// instance)` record breakdown). Counts respect the
        /// same `--source` / `--instance` filter as the row
        /// list but reflect the full matching set, not the
        /// current page. Default off — back-compat.
        #[arg(long)]
        include_counts: bool,
        /// Round 107: emit redacted CSV. Third tabular-redaction
        /// surface, mirrors R91 `audit tail --csv` + R106
        /// `list-forgotten --csv`. Mutually exclusive with
        /// --json (clap) and --include-sensitive /
        /// --include-counts (runtime). Same privacy contract:
        /// `raw_hash` and `native_path` never appear; the
        /// row's `group_index` carries duplicate-group
        /// membership without leaking the hash.
        #[arg(long, conflicts_with = "json")]
        csv: bool,
        /// Round 132 (PR-78ba): `--mode near` only — opt out of
        /// the default cross-source filter and surface
        /// within-adapter near-duplicates too. Equivalent to
        /// `NearDuplicateFilter::require_cross_source = false`.
        /// Ignored under `--mode exact` (raw_hash detection
        /// has no cross-source notion).
        #[arg(long)]
        include_near_self: bool,
        /// Round 141 (PR-78bj): `--mode near` only — augment
        /// each group with a deterministic ranking heuristic
        /// proposing which record to keep, which to forget, and
        /// the `provenance.derived_from` edge a future merge
        /// would write. Read-only: ranks on existing metadata
        /// (user-tag count, effective_at, has_native_path,
        /// adapter, id) — never reads record content. Operator
        /// action stays `anamnesis forget <record_id>`.
        /// `--csv` + `--merge-preview` is rejected (nested
        /// decision draft doesn't flatten safely).
        #[arg(long)]
        merge_preview: bool,
    },

    /// Round 135 (PR-78bd): list cross-adapter `native_id`
    /// content conflicts — multiple adapters claiming the same
    /// upstream record but disagreeing on what it says. A
    /// different question from `dedupe`: dedupe answers
    /// "are these the same memory?" (raw_hash or near-dup);
    /// `conflicts` answers "do these adapters disagree about the
    /// same identity?".
    ///
    /// Read-only, NOT admin-gated. The action half stays the
    /// existing `forget` workflow — surface a conflict, let the
    /// operator decide which variant to drop.
    Conflicts {
        /// Restrict to groups containing ≥1 record from a given
        /// adapter (`mem0`, `claude-code`). Comma-separated OR
        /// list also accepted, same grammar as `dedupe --source`.
        /// Groups stay whole — siblings outside the filter still
        /// appear so the operator sees the full disagreement set.
        #[arg(long)]
        source: Option<String>,
        /// Restrict to groups containing ≥1 record from a given
        /// instance. Comma-separated OR list also accepted.
        /// Combines as AND with `--source`.
        #[arg(long)]
        instance: Option<String>,
        /// Max number of groups to return. Default 20, cap 100.
        #[arg(long, default_value_t = 20)]
        limit: u32,
        /// Emit JSON instead of the human table.
        #[arg(long)]
        json: bool,
        /// Include a short `content_preview` per record (capped
        /// at 240 chars) so an operator can disambiguate variants
        /// without round-tripping `get-record`. Default off —
        /// keeps the surface redacted by default.
        #[arg(long)]
        include_content: bool,
    },

    /// Remove a tombstone so the source can resurrect the memory
    /// on its next import. Does NOT recreate the record itself —
    /// the tombstone only stored provenance, so "stay forgotten"
    /// becomes "allowed to come back if the source re-emits."
    ///
    /// Idempotency: a typo or unknown id exits non-zero
    /// (`no tombstone for id ... nothing to unforget`) so silent
    /// "success" can't hide a paste mistake from
    /// `list-forgotten`.
    Unforget {
        /// Record id to unforget — same shape as
        /// `list-forgotten.rows[].record_id`.
        record_id: String,
        /// Emit JSON instead of the human one-liner.
        #[arg(long)]
        json: bool,
        /// Round 95: don't actually unforget — print the existing
        /// tombstone, the would-write audit entry count, and exit
        /// 0. The tombstone is NOT deleted and `audit.log` is NOT
        /// appended. Symmetric with R83's `forget --dry-run`.
        #[arg(long)]
        dry_run: bool,
        /// Round 134 (PR-78bc): also unforget every record that
        /// `provenance.derived_from` of this one. Symmetric with
        /// R133 `forget --cascade-derived`. Note: this only deletes
        /// tombstones, never resurrects live records — the
        /// source's re-import (or re-extract for derived rows) is
        /// what actually brings the data back.
        #[arg(long)]
        cascade_derived: bool,
    },

    /// List tombstoned records (audit view).
    ///
    /// Round 74: surfaces what `forget` has tombstoned, scoped by
    /// `--source` / `--instance`, newest-first. **Default output
    /// is redacted** — `native_path`, `raw_hash`, and `reason` are
    /// withheld and reported only as `has_*` booleans. Pass
    /// `--include-sensitive` to reveal them when an operator
    /// actually needs to audit content (e.g. before `unforget`).
    ///
    /// Read-only — never writes to the store or audit log.
    ListForgotten {
        /// Restrict to one adapter id (`claude-code`, `mem0`, …).
        #[arg(long)]
        source: Option<String>,
        /// Restrict to a specific instance.
        #[arg(long)]
        instance: Option<String>,
        /// Result limit. Default 20, cap 100.
        #[arg(long, default_value_t = 20)]
        limit: u32,
        /// Emit JSON instead of the human table.
        #[arg(long)]
        json: bool,
        /// Include `native_path`, `raw_hash`, and `reason` fields
        /// in the output. Default off — these may carry user-
        /// supplied or source-derived content and shouldn't appear
        /// in a casual audit dump.
        #[arg(long)]
        include_sensitive: bool,
        /// Round 90: also report `counts.total` and
        /// `counts.by_source` (per-`(adapter, instance)`
        /// tombstone totals). Counts respect the same
        /// `--source` / `--instance` filter as the row list,
        /// but they reflect the **full** matching set — not
        /// just the current page — so an operator can see the
        /// real shape of the tombstone table at a glance.
        #[arg(long)]
        include_counts: bool,
        /// Round 106: emit redacted CSV (mirrors R91/R92
        /// `audit tail --csv`). Mutually exclusive with --json
        /// (clap) and with --include-sensitive /
        /// --include-counts (runtime check). Re-enabled after
        /// R106 fixed the Windows main-thread stack so CLI
        /// args can grow again.
        #[arg(long, conflicts_with = "json")]
        csv: bool,
    },

    /// Forget a record permanently.
    ///
    /// Writes a tombstone (`record_tombstones`) keyed on
    /// `(adapter, instance, native_id)` and removes the live record
    /// row + its chunks / embeddings / raw artifact. Re-importing
    /// the same source will *not* resurrect this record — the
    /// tombstone gate suppresses it inside `upsert_record` /
    /// `upsert_records_batch` before any chunking work runs.
    ///
    /// Idempotent: re-running on an already-forgotten id exits 0
    /// with `status: already-forgotten`. An id that never existed
    /// exits non-zero with `status: not-found` so a typo in
    /// scripted usage is loud.
    Forget {
        /// Record id (e.g. `claude-code:default:session-42`'s
        /// hashed form, as printed by `anamnesis search`).
        record_id: String,
        /// Optional operator-supplied reason — stored on the
        /// tombstone for the future `list_forgotten` view.
        #[arg(long)]
        reason: Option<String>,
        /// Emit JSON instead of the human one-liner.
        #[arg(long)]
        json: bool,
        /// Round 83: don't actually forget — print a preview
        /// showing how many rows would be deleted, what the
        /// tombstone would carry, and that one audit entry
        /// would be written. The store is **not** touched and
        /// the audit log is **not** appended.
        #[arg(long)]
        dry_run: bool,
        /// Round 133 (PR-78bb): also forget every record that
        /// transitively claims this one in `provenance.derived_from`.
        /// Closes the R72 gap where forgetting an Episode left
        /// Stage-2-extracted Facts / Preferences / Skills live.
        /// Off by default — back-compat. Combine with `--dry-run`
        /// to see the full cascade footprint without writing.
        #[arg(long)]
        cascade_derived: bool,
    },

    /// Score retrieval quality (MRR@k / nDCG@k) over a judged query
    /// set and gate the result on configurable thresholds.
    ///
    /// Round 70: this is the *measurement* primitive — it runs the
    /// same `HybridSearcher` + `pack` path that `search` and the
    /// MCP server use, then scores the ranked records against a
    /// JSONL judgment file. Read-only; never writes to the store.
    EvalQuality {
        /// JSONL judgments file. One [`anamnesis_search::JudgedQuery`]
        /// per line; see the docstring on that type for the schema.
        #[arg(long)]
        judgments: PathBuf,
        /// Retrieval mode the harness should evaluate against:
        /// `fulltext` | `vector` | `hybrid`. Defaults to `fulltext`
        /// so the harness can run in CI without downloading
        /// embedding model files.
        #[arg(long, default_value = "fulltext")]
        mode: String,
        /// Per-query result `limit` handed to the retrieval pipeline.
        #[arg(long, default_value_t = 10)]
        limit: u32,
        /// Depth `k` for MRR@k and nDCG@k. Defaults to `limit`.
        #[arg(long)]
        at: Option<u32>,
        /// Fail (exit code 1) if the aggregate MRR@k is below this.
        #[arg(long)]
        min_mrr: Option<f64>,
        /// Fail (exit code 1) if the aggregate nDCG@k is below this.
        #[arg(long)]
        min_ndcg: Option<f64>,
        /// Emit JSON to stdout instead of the human table.
        #[arg(long)]
        json: bool,
    },

    /// Manage embedding models.
    #[command(subcommand)]
    Model(ModelCmd),

    /// Browse the §-1.5 PR-6 Stage 2 audit log
    /// (`<data_dir>/audit/stage2.jsonl`). Use `list` for an overview,
    /// `show <line-no>` to pretty-print one run.
    #[command(subcommand)]
    Audit(AuditCmd),

    /// Export records as JSONL, CSV, or a mem0-/letta-readable
    /// SQLite DB.
    ///
    /// Round 138 (PR-78bg) added `mem0-sqlite` and R139 (PR-78bh)
    /// added `letta-sqlite` — both close the bidirectional interop
    /// loop: an operator can import → normalise/dedupe/consolidate
    /// in Anamnesis → export back into the schema the original
    /// framework itself reads. Exported DBs are fresh files; we
    /// refuse to overwrite an existing path so a typo can't
    /// clobber the user's real `~/.mem0/history.db` or
    /// `~/.letta/letta.db`.
    Export {
        /// Output file path. For `jsonl` / `csv`: defaults to
        /// stdout. For `mem0-sqlite` / `letta-sqlite`: REQUIRED
        /// — SQLite output can't stream, so we always materialise
        /// a file path.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Format: `jsonl` (one AnamnesisRecord per line), `csv`,
        /// `mem0-sqlite` (R138 — fresh SQLite DB with the
        /// `memories` table mem0 itself reads), or `letta-sqlite`
        /// (R139 — fresh SQLite DB with the `block` table the
        /// Letta SQLite adapter reads). Letta-origin records
        /// round-trip their native `block` columns
        /// (label/description/template_name) from the metadata
        /// the Letta adapter stashed at import time; non-Letta
        /// origins get a stable `anamnesis/<adapter>` label.
        #[arg(long, default_value = "jsonl")]
        format: String,
        /// Restrict to one source (adapter id).
        #[arg(long)]
        source: Option<String>,
    },

    /// §-1.5 PR-6 stage 1: deterministic gate over Episode records.
    ///
    /// Lists which records would be handed to Stage 2 (the LLM step,
    /// not yet implemented). No LLM calls happen during this command —
    /// it's pure inspection. Per §-1.2 #5, Anamnesis never silently
    /// calls an LLM, and this command is the surface that makes the
    /// future Stage 2 explicit.
    Extract {
        /// Target Kind to distill toward: fact | preference | feedback | skill.
        #[arg(long, default_value = "fact")]
        kind: String,
        /// Restrict to one source (adapter id).
        #[arg(long)]
        source: Option<String>,
        /// Restrict to a specific source instance.
        #[arg(long)]
        instance: Option<String>,
        /// Minimum Stage-1 score to surface (0.0–1.0; default 0.4).
        #[arg(long, default_value_t = 0.4)]
        threshold: f32,
        /// Max candidates to list (top-N by score).
        #[arg(long, default_value_t = 25)]
        limit: usize,
        /// Show per-candidate gate rationale.
        #[arg(long)]
        explain: bool,
        /// Emit JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
        /// Opt-out of dry-run mode and actually run Stage 2 against the
        /// configured provider. With `--provider mock` (default) this
        /// is a no-network no-op that exercises the persistence
        /// pipeline. With `--provider openai` it makes real HTTP
        /// requests after printing the cost preview.
        #[arg(long)]
        no_dry_run: bool,
        /// Stage-2 LLM provider: `mock` (default; deterministic, no
        /// network) or `openai` (real OpenAI-compatible HTTP, needs
        /// the `openai-provider` cargo feature).
        #[arg(long, default_value = "mock")]
        provider: String,
        /// Model identifier passed to the provider (e.g.
        /// `gpt-4o-mini`, `llama3.2:3b`). Ignored for `mock`.
        #[arg(long, default_value = "gpt-4o-mini")]
        model: String,
        /// Provider-specific API base URL.
        ///
        /// For `--provider openai`: e.g. `https://api.openai.com/v1`,
        /// `http://localhost:11434/v1` for Ollama. Falls back to
        /// `OPENAI_API_BASE` env.
        ///
        /// For `--provider anthropic`: e.g. `https://api.anthropic.com`.
        /// Falls back to `ANTHROPIC_API_BASE` env.
        ///
        /// Each provider has its own default; omit unless you're
        /// proxying or using a non-stock endpoint.
        #[arg(long)]
        api_base: Option<String>,
        /// Safety cap: refuse to start Stage 2 if Stage 1 surfaced
        /// more than this many candidates. Default 100. Bypass with
        /// a higher value once you've eyeballed `--dry-run` output.
        #[arg(long, default_value_t = 100)]
        max_llm_calls: usize,
        /// Skip the interactive `Proceed? [y/N]` prompt before any
        /// real LLM call. Required when running non-interactively
        /// (CI / scripts). Has no effect with `--provider mock`.
        #[arg(long)]
        yes: bool,
        /// Max concurrent provider calls in flight. Default 1
        /// (sequential). Crank up only when you've checked the
        /// provider's rate-limit budget — paid providers throttle
        /// aggressively.
        #[arg(long, default_value_t = 1)]
        concurrency: usize,
        /// Total retry attempts per provider call (including the
        /// first). Default 3. Set to 1 to disable retry. Honors
        /// `Retry-After` headers when longer than the computed
        /// exponential backoff. `mock` provider ignores this.
        #[arg(long, default_value_t = 3)]
        max_retries: u32,
    },

    /// Show the `provenance.derived_from` lineage of a record. Walks
    /// from the given record id up through its source-Episode chain.
    /// Use the inverse query — direct children of a record — with the
    /// `--children` flag.
    ///
    /// Required for §-1.5 #6 ("结果 provenance 回指原始 Episode 的
    /// record_id，抽取日志可查"): once the Stage-2 extractor starts
    /// writing derived records, this is how users audit which Episode
    /// produced which Fact / Preference / Skill / Feedback.
    Lineage {
        /// Record id to start the walk from.
        record_id: String,
        /// List direct children (records whose `derived_from` points
        /// at the given id) instead of walking ancestors.
        #[arg(long)]
        children: bool,
        /// Max children to list (only meaningful with `--children`).
        #[arg(long, default_value_t = 50)]
        limit: u32,
        /// Emit JSON instead of a human-readable summary.
        #[arg(long)]
        json: bool,
    },

    /// Run as an MCP server. Default = stdio; `--sse <port>` binds the
    /// HTTP/JSON-RPC transport on `127.0.0.1:<port>` (use `0` for an
    /// ephemeral port — the chosen port is printed to stderr).
    Serve {
        /// Bind the HTTP/JSON-RPC transport on `127.0.0.1:<port>`.
        /// Requires the `sse` cargo feature (on by default).
        #[arg(long)]
        sse: Option<u16>,
        /// Pre-shared bearer token for HTTP mode. If omitted a fresh
        /// 64-char token is generated and printed to stderr on startup.
        /// Stdio mode ignores this flag.
        #[arg(long, env = "ANAMNESIS_MCP_TOKEN")]
        token: Option<String>,
    },

    /// Verify database integrity. With --repair, rebuild the FTS index and
    /// re-queue any chunks that have no embeddings under the active model.
    Verify {
        /// Try to fix issues that are auto-repairable.
        #[arg(long)]
        repair: bool,
    },

    /// Run pending schema migrations (no-op after init).
    Migrate,

    /// MCP-client integration helpers. Round-55: print the
    /// copy-and-paste `mcpServers` snippet a Claude Desktop / Cursor /
    /// Continue / Windsurf / generic MCP-aware client needs to talk to
    /// this Anamnesis install.
    #[command(subcommand)]
    Mcp(McpCmd),

    /// §-2.5 per-source health check. Probes each registered source's
    /// `MemoryAdapter::health()` and surfaces what's reachable, what's
    /// stale, and any adapter-specific notes.
    ///
    /// With `--include-unregistered`, also runs the detectors for the
    /// registered first-class adapters so you can see what's available on
    /// the machine but not yet registered.
    Doctor {
        /// Restrict to one adapter id (and optional instance).
        /// Round 110: also accepts a comma-separated OR list
        /// (`--source mem0,claude-code`) — both adapters'
        /// rows survive, everything else drops. Tokens are
        /// trimmed and empty tokens dropped, so
        /// `--source mem0, , claude-code` and
        /// `--source mem0,claude-code` are equivalent.
        #[arg(long)]
        source: Option<String>,
        /// Match an instance within `--source`. Round 114:
        /// also accepts a comma-separated OR list
        /// (`--instance prod,dev`) — any listed instance
        /// matches. Combines as AND with the source OR-set
        /// (`source ∈ [a,b] && instance ∈ [c,d]`). Tokens
        /// trimmed and empty tokens dropped.
        #[arg(long)]
        instance: Option<String>,
        /// Also run the discovery detectors for adapters with no
        /// registered source, so you can see what's installed locally.
        #[arg(long)]
        include_unregistered: bool,
        /// Emit JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
        /// Exit with code 1 if any registered source's `health()`
        /// returned `ok = false`. Useful in CI gates. Off by default —
        /// `doctor` is also an inspection command you might run on a
        /// partially-set-up machine where some sources legitimately
        /// have missing paths yet.
        #[arg(long)]
        strict: bool,
        /// Flag any registered source whose `last_import_at` is older
        /// than this duration. Accepts `Nd` (days), `Nh` (hours),
        /// `Nm` (minutes). Sources never imported are also flagged.
        /// Staleness is reported as a `[!]` row marker; combine with
        /// `--strict-staleness` if you want it to trip the exit code.
        #[arg(long)]
        since: Option<String>,
        /// Also exit non-zero (alongside `--strict`) if any registered
        /// source is stale per `--since`. No effect without `--since`.
        #[arg(long)]
        strict_staleness: bool,
        /// Round 130: emit an enriched JSON envelope
        /// `{ summary, filters, sources, request_metrics }`
        /// instead of the bare-array `--json`. Mutually
        /// exclusive with `--json`. The bare-array shape is
        /// preserved for back-compat — scripts that pin it
        /// keep working; scripts that want operator-facing
        /// counts / filter echo / metrics window opt into
        /// the envelope here.
        #[arg(long, conflicts_with = "json")]
        json_summary: bool,
    },
}

#[derive(Subcommand, Debug)]
enum SourceCmd {
    /// Register a new source.
    ///
    /// File-based adapters take `--path`. URL-based adapters (e.g.
    /// `generic-mcp`) take `--url` and optionally `--token-env`.
    Add {
        /// Adapter name (e.g. `claude-code`, `mem0`, `codex`, `generic-mcp`).
        adapter: String,
        /// Instance discriminator (optional).
        #[arg(long)]
        instance: Option<String>,
        /// Filesystem path, if the adapter takes one.
        #[arg(long, conflicts_with = "url")]
        path: Option<PathBuf>,
        /// Upstream URL, for URL-based adapters like `generic-mcp`
        /// (e.g. `http://127.0.0.1:7878`).
        #[arg(long)]
        url: Option<String>,
        /// Name of the environment variable holding the bearer token
        /// for URL-based adapters. The value is resolved at import time,
        /// not at registration time — only the variable *name* is stored
        /// in the registry, never the token itself.
        #[arg(long, requires = "url")]
        token_env: Option<String>,
    },
    /// List configured sources.
    List {
        /// Round 99: restrict to one adapter id (`mem0`,
        /// `claude-code`, ...). Mirrors the MCP
        /// `list_sources { source }` filter from R96.
        ///
        /// Round 103: comma-separated list is also accepted
        /// (`--source mem0,claude-code`) for an OR filter — both
        /// adapters' rows come back, everything else drops.
        /// Tokens are trimmed and empty tokens dropped, so
        /// `--source mem0, , claude-code ,` and `--source
        /// mem0,claude-code` are equivalent.
        #[arg(long)]
        source: Option<String>,
        /// Round 99: restrict to one instance discriminator.
        ///
        /// Round 115: comma-separated list is also accepted
        /// (`--instance prod,dev`) for an OR filter. Combines as
        /// AND with `--source` when both are set.
        #[arg(long)]
        instance: Option<String>,
        /// Round 88: emit JSON instead of the human table.
        /// Mirrors R86's `source show --json` shape and the
        /// MCP `list_sources` wire format so scripts can
        /// consume the same field set from either side.
        #[arg(long)]
        json: bool,
    },
    /// Round 86: per-source detail view. Same counts as
    /// `source list` (records, chunks, tagged) for one
    /// `(adapter, instance)`, plus the last few `import_errors`
    /// for that source so an operator can answer "why is this
    /// source empty / stale / broken" in one call.
    Show {
        /// `adapter` or `adapter:instance`. Same parser as `remove`.
        target: String,
        /// Max recent import errors to show. Default 5, cap 10.
        #[arg(long, default_value_t = 5)]
        errors: usize,
        /// Emit JSON instead of a human summary.
        #[arg(long)]
        json: bool,
    },
    /// Remove a registered source.
    Remove {
        /// `adapter` or `adapter:instance`.
        target: String,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCmd {
    /// List curated models + which one is active.
    List,
    /// Switch the active embedding model. Re-queues all chunks for embed.
    Use {
        /// Curated key (e.g. `default`, `tiny`, `en`, `multi-strong`).
        key: String,
        /// Skip running the embedding worker after the switch.
        #[arg(long)]
        no_embed: bool,
    },
    /// Pre-download a model into the cache without changing the active one.
    Install {
        /// Curated key.
        key: String,
    },
    /// Re-embed every chunk under the current active model.
    Rebuild {
        /// Skip running the worker; only re-enqueue jobs.
        #[arg(long)]
        no_embed: bool,
    },
}

#[derive(Subcommand, Debug)]
enum AuditCmd {
    /// List every Stage 2 extraction run, newest first.
    ///
    /// Note: this reads `data_dir/audit/stage2.jsonl` (extractor
    /// runs). For the global mutation/search log
    /// (`data_dir/audit.log`), use `audit tail`.
    List {
        /// Maximum number of runs to list.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Emit JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
    },
    /// Show one specific Stage 2 run by its 1-based line number
    /// in `data_dir/audit/stage2.jsonl` (use `list` to find
    /// numbers). Pass `last` for the most recent run.
    Show {
        /// Line number (1-based) or `last`.
        target: String,
        /// Emit JSON instead of a human-readable summary.
        #[arg(long)]
        json: bool,
    },
    /// Round 84: tail the global mutation / search audit log at
    /// `data_dir/audit.log` (the same file `Audit::record` appends
    /// to from every CLI and MCP write). Each line is a JSONL
    /// `AuditEntry { timestamp, action, detail }`.
    ///
    /// Distinct from `audit list/show`, which reads the separate
    /// Stage 2 extractor log.
    Tail {
        /// Maximum number of entries to return.
        #[arg(long = "limit", short = 'n', default_value_t = 20)]
        limit: usize,
        /// Filter to entries whose `action` matches. Pass a
        /// single value (`--action forget`) for exact match, or
        /// a comma-separated list (`--action forget,search`)
        /// for an OR filter — both `forget` and `search` rows
        /// come back, everything else is dropped. Tokens are
        /// trimmed and empty tokens dropped, so `--action
        /// forget,, search ,` and `--action forget,search` are
        /// equivalent. Omit for all actions.
        #[arg(long)]
        action: Option<String>,
        /// Drop entries older than this relative lookback.
        /// Shapes: `Nd` (days), `Nh` (hours), `Nm` (minutes), or
        /// bare integer (seconds). Same grammar as `doctor --since`.
        #[arg(long)]
        since: Option<String>,
        /// Emit full JSON (with `detail`). The human renderer
        /// shows only `line_no / timestamp / action / via /
        /// outcome|status` so a casual tail doesn't dump search
        /// queries or forget reasons into the terminal.
        #[arg(long)]
        json: bool,
        /// Round 91: emit CSV (`line_no,timestamp,action,via,outcome`)
        /// instead of the human table. Same redacted summary
        /// shape as the human output — never includes `detail` /
        /// `reason` / `query`. Mutually exclusive with `--json`
        /// (use `--json` for the structured form).
        #[arg(long, conflicts_with = "json")]
        csv: bool,
    },
}

#[derive(Subcommand, Debug)]
enum McpCmd {
    /// Emit the `mcpServers` JSON snippet for a Claude Desktop / Cursor /
    /// Continue / Windsurf / generic MCP client. Output is the smallest
    /// pasteable wrapper:
    ///
    /// ```json
    /// { "mcpServers": { "<name>": { "command": "...", "args": [...] } } }
    /// ```
    ///
    /// The user merges this with their existing config (e.g. with `jq`,
    /// or by opening the JSON file and pasting into `"mcpServers"`).
    ///
    /// Defaults to stdio transport, which every common client supports.
    /// Pass `--transport sse --sse-port <port>` for HTTP/SSE clients.
    Config {
        /// Server name to register under in the host config. Default
        /// `anamnesis` (the typical convention).
        #[arg(long, default_value = "anamnesis")]
        name: String,
        /// Transport: `stdio` (default — every client supports it) or
        /// `sse` (HTTP/JSON-RPC, requires `--sse-port`).
        #[arg(long, default_value = "stdio")]
        transport: String,
        /// Port for `--transport sse`. Required when transport is sse.
        #[arg(long)]
        sse_port: Option<u16>,
        /// Env-var name holding the bearer token for SSE mode. Default
        /// `ANAMNESIS_MCP_TOKEN` (same name `anamnesis serve` reads). The
        /// emitted config references this by name, never the value.
        #[arg(long, default_value = "ANAMNESIS_MCP_TOKEN")]
        token_env: String,
        /// Override the binary path emitted into `command`. Defaults to
        /// the absolute path of the running `anamnesis` executable so the
        /// snippet works even if `anamnesis` isn't on `$PATH` in the
        /// host client's process.
        #[arg(long)]
        binary: Option<PathBuf>,
    },
}

fn init_tracing(level: &str) {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(level))
        .unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).with_target(false).init();
}

fn resolve_data_dir(override_path: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        return Ok(p);
    }
    let base = if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
        PathBuf::from(xdg)
    } else if cfg!(target_os = "macos") {
        dirs_home()?.join("Library/Application Support")
    } else if cfg!(windows) {
        std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .context("APPDATA not set")?
    } else {
        dirs_home()?.join(".local/share")
    };
    Ok(base.join("anamnesis"))
}

fn dirs_home() -> Result<PathBuf> {
    if let Some(h) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(h));
    }
    // Windows: HOME isn't set by default, but USERPROFILE is.
    if let Some(h) = std::env::var_os("USERPROFILE") {
        return Ok(PathBuf::from(h));
    }
    Err(anyhow!("neither HOME nor USERPROFILE is set"))
}

fn db_path(data_dir: &std::path::Path) -> PathBuf {
    data_dir.join("anamnesis.sqlite")
}

fn models_dir(data_dir: &std::path::Path) -> PathBuf {
    data_dir.join("models")
}

fn resolve_config_path(override_path: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        return Ok(p);
    }
    let home = dirs_home()?;
    Ok(anamnesis_core::Config::default_path(&home))
}

/// Round 106 (PR-78ab): Windows debug builds default to a 1 MB
/// thread stack, and the cumulative clap `Command` tree
/// finally crossed that ceiling around R105 — `anamnesis init`
/// crashed with `STATUS_STACK_OVERFLOW` (`code=-1073741571`)
/// every subprocess. The fix is the standard Rust pattern:
/// move the real work to a worker thread with an 8 MB stack
/// and run the Tokio runtime inside it. The OS-default main
/// thread does almost nothing beyond `Builder::spawn`, and the
/// worker has more than enough headroom for clap parse +
/// long-tail subcommand growth.
///
/// `main` itself stays sync — `#[tokio::main]` is removed so we
/// own the runtime build (and pick where it lives in the stack
/// graph). The pre-R106 body now lives in [`run`] verbatim.
fn main() -> Result<()> {
    let builder = std::thread::Builder::new()
        .name("anamnesis-main".into())
        .stack_size(8 * 1024 * 1024);
    let handle = builder
        .spawn(|| {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("build tokio runtime");
            rt.block_on(run())
        })
        .expect("spawn anamnesis-main thread");
    handle.join().expect("join anamnesis-main thread")
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);
    let data_dir = resolve_data_dir(cli.data_dir)?;
    let config_path = resolve_config_path(cli.config)?;
    let config = anamnesis_core::Config::load(&config_path)
        .with_context(|| format!("load {}", config_path.display()))?;
    tracing::debug!(path = %config_path.display(), "loaded config");

    match cli.command {
        Command::Init { model } => {
            // Precedence: --model > config.embedding.model > registry default.
            let chosen = model
                .clone()
                .or_else(|| Some(config.embedding.model.clone()))
                .filter(|m| !m.is_empty());
            cmd_init(&data_dir, chosen.as_deref())
        }
        Command::Status { json } => cmd_status(&data_dir, json),
        Command::Discover => cmd_discover().await,
        Command::Source(sub) => cmd_source(&data_dir, sub),
        Command::Import {
            target,
            full,
            since,
            dry_run,
            no_embed,
            path,
        } => {
            cmd_import(
                &data_dir,
                &target,
                full,
                since.as_deref(),
                dry_run,
                no_embed,
                path.as_deref(),
            )
            .await
        }
        Command::Search {
            query,
            source,
            instance,
            kind,
            scope,
            since,
            until,
            limit,
            mode,
            json,
            trace,
            user_tag,
            explain,
        } => {
            cmd_search(
                &data_dir,
                &query,
                source.as_deref(),
                instance.as_deref(),
                kind.as_deref(),
                scope.as_deref(),
                since.as_deref(),
                until.as_deref(),
                limit,
                &mode,
                json,
                trace,
                user_tag.as_deref(),
                explain,
            )
            .await
        }
        Command::Extract {
            kind,
            source,
            instance,
            threshold,
            limit,
            explain,
            json,
            no_dry_run,
            provider,
            model,
            api_base,
            max_llm_calls,
            yes,
            concurrency,
            max_retries,
        } => {
            cmd_extract(
                &data_dir,
                &kind,
                source.as_deref(),
                instance.as_deref(),
                threshold,
                limit,
                explain,
                json,
                !no_dry_run,
                &provider,
                &model,
                api_base.as_deref(),
                max_llm_calls,
                yes,
                concurrency,
                max_retries,
            )
            .await
        }
        Command::Lineage {
            record_id,
            children,
            limit,
            json,
        } => cmd_lineage(&data_dir, &record_id, children, limit, json),
        Command::TagRecord {
            record_id,
            tags,
            remove,
            replace,
            json,
            include_stats,
        } => cmd_tag_record(
            &data_dir,
            &record_id,
            &tags,
            remove,
            replace,
            json,
            include_stats,
        ),
        Command::Dedupe {
            mode,
            source,
            instance,
            limit,
            json,
            include_sensitive,
            include_counts,
            csv,
            include_near_self,
            merge_preview,
        } => cmd_dedupe(
            &data_dir,
            mode,
            source.as_deref(),
            instance.as_deref(),
            limit,
            json,
            include_sensitive,
            include_counts,
            csv,
            include_near_self,
            merge_preview,
        ),
        Command::Conflicts {
            source,
            instance,
            limit,
            json,
            include_content,
        } => cmd_conflicts(
            &data_dir,
            source.as_deref(),
            instance.as_deref(),
            limit,
            json,
            include_content,
        ),
        Command::Unforget {
            record_id,
            json,
            dry_run,
            cascade_derived,
        } => cmd_unforget(&data_dir, &record_id, json, dry_run, cascade_derived),
        Command::ListForgotten {
            source,
            instance,
            limit,
            json,
            include_sensitive,
            include_counts,
            csv,
        } => cmd_list_forgotten(
            &data_dir,
            source.as_deref(),
            instance.as_deref(),
            limit,
            json,
            include_sensitive,
            include_counts,
            csv,
        ),
        Command::Forget {
            record_id,
            reason,
            json,
            dry_run,
            cascade_derived,
        } => cmd_forget(
            &data_dir,
            &record_id,
            reason.as_deref(),
            json,
            dry_run,
            cascade_derived,
        ),
        Command::EvalQuality {
            judgments,
            mode,
            limit,
            at,
            min_mrr,
            min_ndcg,
            json,
        } => {
            cmd_eval_quality(
                &data_dir, &judgments, &mode, limit, at, min_mrr, min_ndcg, json,
            )
            .await
        }
        Command::Model(sub) => cmd_model(&data_dir, sub).await,
        Command::Audit(sub) => cmd_audit(&data_dir, sub),
        Command::Serve { sse, token } => {
            cmd_serve(&data_dir, sse, token, config.server.allow_admin_tools).await
        }
        Command::Export {
            out,
            format,
            source,
        } => cmd_export(&data_dir, out.as_deref(), &format, source.as_deref()),
        Command::Verify { repair } => cmd_verify(&data_dir, repair),
        Command::Migrate => {
            let _ = Store::open(db_path(&data_dir))?;
            println!("migrations applied");
            Ok(())
        }
        Command::Mcp(McpCmd::Config {
            name,
            transport,
            sse_port,
            token_env,
            binary,
        }) => cmd_mcp_config(&name, &transport, sse_port, &token_env, binary.as_deref()),
        Command::Doctor {
            source,
            instance,
            include_unregistered,
            json,
            strict,
            since,
            strict_staleness,
            json_summary,
        } => {
            cmd_doctor(
                &data_dir,
                source.as_deref(),
                instance.as_deref(),
                include_unregistered,
                json,
                strict,
                since.as_deref(),
                strict_staleness,
                json_summary,
            )
            .await
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// mcp config — emit the `mcpServers` JSON snippet a host client (Claude
// Desktop / Cursor / Continue / Windsurf / etc.) needs to launch this
// Anamnesis install. Round-55: codex-recommended path to "from CLI to
// closed-loop memory server in one paste" for the 0.1.0 demo.
// ─────────────────────────────────────────────────────────────────────────────

/// Build and print the `mcpServers` JSON wrapper. Pure I/O — nothing
/// touches the data dir or the store. Idempotent; users re-run this
/// after every release to refresh the absolute binary path.
fn cmd_mcp_config(
    name: &str,
    transport: &str,
    sse_port: Option<u16>,
    token_env: &str,
    binary_override: Option<&std::path::Path>,
) -> Result<()> {
    let server = build_mcp_server_entry(transport, sse_port, token_env, binary_override)?;
    let wrapper = serde_json::json!({
        "mcpServers": { name: server },
    });
    println!("{}", serde_json::to_string_pretty(&wrapper)?);
    Ok(())
}

/// Build the JSON value that goes under `mcpServers.<name>` for one
/// host. Two shapes:
///
///   * stdio:  `{"command": "...", "args": ["serve"]}`
///   * sse:    `{"url": "http://127.0.0.1:<port>/mcp", "headers": {"Authorization": "Bearer ${env:TOKEN}"}}`
///
/// Factored out from `cmd_mcp_config` so unit tests don't have to go
/// through stdout — they assert on this `Value` directly.
fn build_mcp_server_entry(
    transport: &str,
    sse_port: Option<u16>,
    token_env: &str,
    binary_override: Option<&std::path::Path>,
) -> Result<serde_json::Value> {
    match transport {
        "stdio" => {
            let bin = resolve_anamnesis_binary(binary_override)?;
            Ok(serde_json::json!({
                "command": bin.to_string_lossy(),
                "args": ["serve"],
            }))
        }
        "sse" => {
            let port = sse_port.ok_or_else(|| {
                anyhow!("--transport sse requires --sse-port <port> (use the same port you'll pass to `anamnesis serve --sse <port>`)")
            })?;
            // Host config consumers (Claude Desktop, Cursor, etc.) all
            // support the `${env:NAME}` placeholder — the value is
            // resolved at request time, not at config-write time, so
            // the secret never lands on disk.
            let auth = format!("Bearer ${{env:{token_env}}}");
            Ok(serde_json::json!({
                "url": format!("http://127.0.0.1:{port}/mcp"),
                "headers": { "Authorization": auth },
            }))
        }
        other => Err(anyhow!(
            "unknown --transport {other:?}; supported: stdio, sse"
        )),
    }
}

/// Return the absolute path to the `anamnesis` binary the host client
/// should launch. Default = the executable that's running this
/// command, so a user who installed via `~/.local/bin/anamnesis` gets
/// a snippet that works from inside their MCP client (which won't
/// necessarily inherit the same `$PATH`).
fn resolve_anamnesis_binary(override_path: Option<&std::path::Path>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        return Ok(p.to_path_buf());
    }
    // current_exe() returns the canonical path to the running process's
    // executable. On macOS / Linux this is what `/proc/self/exe` (or
    // _NSGetExecutablePath) reports — almost always the user's
    // installed-into-PATH binary.
    std::env::current_exe().map_err(|e| anyhow!("could not resolve current binary path: {e}"))
}

// ─────────────────────────────────────────────────────────────────────────────
// serve — embed the MCP server in the CLI process (same code as the
// dedicated `anamnesis-mcp` binary, but one less binary for users to wire up).
// ─────────────────────────────────────────────────────────────────────────────

async fn cmd_serve(
    data_dir: &std::path::Path,
    sse: Option<u16>,
    token: Option<String>,
    allow_admin_tools: bool,
) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let active_model = store.active_model().ok().flatten();
    let provider = open_active_provider_optional(data_dir, &store, active_model.as_deref());
    let server =
        anamnesis_mcp_server::AnamnesisServer::new(store, provider, data_dir.to_path_buf())
            .with_admin_tools(allow_admin_tools);

    match sse {
        Some(port) => {
            eprintln!(
                "anamnesis serve (http) — active model: {}, admin tools: {}",
                active_model.as_deref().unwrap_or("<unset>"),
                if allow_admin_tools {
                    "ENABLED"
                } else {
                    "disabled"
                },
            );
            if allow_admin_tools {
                eprintln!(
                    "  ⚠ admin tools enabled — `import_source` is callable over MCP. \
                     Run only with trusted clients."
                );
            }
            audit(data_dir).record(anamnesis_core::AuditEntry::new(
                "serve.start",
                serde_json::json!({
                    "transport": "http",
                    "port": port,
                    "active_model": active_model,
                    "allow_admin_tools": allow_admin_tools,
                }),
            ));
            run_sse(server, port, token).await
        }
        None => {
            eprintln!(
                "anamnesis serve (stdio) — active model: {}, admin tools: {}",
                active_model.as_deref().unwrap_or("<unset>"),
                if allow_admin_tools {
                    "ENABLED"
                } else {
                    "disabled"
                },
            );
            if allow_admin_tools {
                eprintln!(
                    "  ⚠ admin tools enabled — `import_source` is callable. \
                     Run only with trusted clients."
                );
            }
            audit(data_dir).record(anamnesis_core::AuditEntry::new(
                "serve.start",
                serde_json::json!({
                    "transport": "stdio",
                    "active_model": active_model,
                    "allow_admin_tools": allow_admin_tools,
                }),
            ));
            anamnesis_mcp_server::stdio::run(server).await
        }
    }
}

#[cfg(feature = "sse")]
async fn run_sse(
    server: anamnesis_mcp_server::AnamnesisServer,
    port: u16,
    token: Option<String>,
) -> Result<()> {
    let config = anamnesis_mcp_server::sse::HttpServerConfig { port, token };
    anamnesis_mcp_server::sse::run(server, config).await
}

#[cfg(not(feature = "sse"))]
async fn run_sse(
    _server: anamnesis_mcp_server::AnamnesisServer,
    _port: u16,
    _token: Option<String>,
) -> Result<()> {
    Err(anyhow!(
        "this `anamnesis` build lacks the `sse` cargo feature; \
         rebuild with `--features sse` (on by default)."
    ))
}

#[cfg(feature = "local-fastembed")]
fn open_active_provider_optional(
    data_dir: &std::path::Path,
    _store: &Store,
    active_model: Option<&str>,
) -> Option<Box<dyn anamnesis_core::EmbeddingProvider>> {
    let key = active_model?.split(':').nth(1)?;
    match anamnesis_embedder::LocalFastembedProvider::new(key, models_dir(data_dir)) {
        Ok(p) => Some(Box::new(p)),
        Err(e) => {
            tracing::warn!(
                model = key,
                error = %e,
                "failed to open active embedding model; serve will degrade to FTS-only"
            );
            None
        }
    }
}

#[cfg(not(feature = "local-fastembed"))]
fn open_active_provider_optional(
    _data_dir: &std::path::Path,
    _store: &Store,
    _active_model: Option<&str>,
) -> Option<Box<dyn anamnesis_core::EmbeddingProvider>> {
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// init / status / discover
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_init(data_dir: &std::path::Path, model: Option<&str>) -> Result<()> {
    std::fs::create_dir_all(data_dir)
        .with_context(|| format!("create data dir {}", data_dir.display()))?;
    let store = Store::open(db_path(data_dir))
        .with_context(|| format!("open {}", db_path(data_dir).display()))?;

    let key = model.unwrap_or_else(|| registry::default_model().key);
    if registry::by_key(key).is_none() {
        return Err(anyhow!(
            "unknown model key {key:?} — available: {}",
            registry::available().join(", ")
        ));
    }
    let model_id = format!("local:{key}:1");
    store.set_active_model(&model_id)?;

    println!("initialized at {}", data_dir.display());
    println!("active embedding model: {model_id}");
    Ok(())
}

fn cmd_status(data_dir: &std::path::Path, json: bool) -> Result<()> {
    let db = db_path(data_dir);
    if !db.exists() {
        if json {
            // Round 123 (PR-78ar): top-level `summary` mirrors
            // the MCP discovery-summary pattern (R111-R122) on
            // the operator-facing status surface. NEVER reads
            // `data_dir`/`db_path` so it stays path-free.
            let payload = serde_json::json!({
                "summary": "database not initialized; run `anamnesis init`.",
                "initialized": false,
                "data_dir": data_dir.display().to_string(),
                "db_path": db.display().to_string(),
            });
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            println!(
                "no database found at {} — run `anamnesis init`",
                db.display()
            );
        }
        return Ok(());
    }
    let store = Store::open(&db)?;
    let stats = store.stats()?;
    let active = store.active_model()?;
    // Round-16: surface per-source counts + freshness right on
    // `status`, not just on `source list`. Operators landing on `status`
    // for the first time should see "this source hasn't imported in 30
    // days" without having to know about a second subcommand.
    let per_source = store.list_sources_with_counts()?;
    let now = chrono::Utc::now().timestamp();
    if json {
        // Round 123 (PR-78ar): redacted top-level `summary` +
        // structured `source_summary` rollup. Mirrors the
        // MCP discovery-summary pattern (R111-R122) on the
        // operator side. Summary NEVER reads `location`,
        // `data_dir`, recent import error text/path, or any
        // record content.
        let mut fresh = 0u64;
        let mut stale = 0u64;
        let mut never_imported = 0u64;
        for r in &per_source {
            match source_freshness(r.source.last_import_at, now).label {
                "fresh" => fresh += 1,
                "stale" => stale += 1,
                _ => never_imported += 1,
            }
        }
        let active_label = active.as_deref().unwrap_or("none");
        let summary = format!(
            "database initialized; {} registered source(s); active model: {}; stats reflect whole store ({} records, {} chunks); freshness: {} fresh, {} stale, {} never-imported; import errors: {}.",
            stats.sources,
            active_label,
            stats.records,
            stats.chunks,
            fresh,
            stale,
            never_imported,
            stats.import_errors,
        );
        let payload = serde_json::json!({
            "summary": summary,
            "source_summary": {
                "registered": stats.sources,
                "fresh": fresh,
                "stale": stale,
                "never_imported": never_imported,
            },
            "initialized": true,
            "data_dir": data_dir.display().to_string(),
            "models_dir": models_dir(data_dir).display().to_string(),
            "schema_version": anamnesis_core::SCHEMA_VERSION,
            "active_model": active,
            "stats": {
                "sources": stats.sources,
                "records": stats.records,
                "chunks": stats.chunks,
                "jobs_pending": stats.jobs_pending,
                "jobs_failed": stats.jobs_failed,
                "import_errors": stats.import_errors,
            },
            "recent_import_errors": store
                .recent_import_errors(None, 10)
                .unwrap_or_default()
                .iter()
                .map(|e| serde_json::json!({
                    "adapter": e.adapter,
                    "instance": if e.instance.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::String(e.instance.clone())
                    },
                    "native_id": e.native_id,
                    "native_path": e.native_path,
                    "phase": e.phase,
                    "error": e.error,
                    "occurred_at": e.occurred_at,
                }))
                .collect::<Vec<_>>(),
            "sources": per_source.iter().map(|r| {
                let freshness = source_freshness(r.source.last_import_at, now);
                serde_json::json!({
                    "adapter": r.source.adapter,
                    "instance": if r.source.instance.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::String(r.source.instance.clone())
                    },
                    "location": r.source.location,
                    "added_at": r.source.added_at,
                    "last_import_at": r.source.last_import_at,
                    "record_count": r.record_count,
                    "chunk_count": r.chunk_count,
                    // Round-82: per-source distinct-record count
                    // for `user_record_tags`. Lets `status --json`
                    // consumers spot "where do my keep-forever
                    // records live" without another round trip.
                    "tagged_record_count": r.tagged_record_count,
                    // Round-16 additions — let consumers branch on
                    // staleness without re-computing it from `now`.
                    "freshness": freshness.label,
                    "age_seconds": freshness.age_seconds,
                })
            }).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }
    println!("data_dir        : {}", data_dir.display());
    println!("models dir      : {}", models_dir(data_dir).display());
    println!("schema          : v{}", anamnesis_core::SCHEMA_VERSION);
    println!(
        "active model    : {}",
        active.as_deref().unwrap_or("<unset>")
    );
    println!("sources         : {}", stats.sources);
    println!("records         : {}", stats.records);
    println!("chunks          : {}", stats.chunks);
    println!("jobs pending    : {}", stats.jobs_pending);
    println!("jobs failed     : {}", stats.jobs_failed);
    println!("import errors   : {}", stats.import_errors);

    // Show the 5 most recent import errors so operators see what
    // silently failed during the last import without needing to open
    // the SQLite database. JSON output already includes the full top-10.
    if stats.import_errors > 0 {
        let recent = store.recent_import_errors(None, 5).unwrap_or_default();
        if !recent.is_empty() {
            println!();
            println!("recent import errors (top 5):");
            for e in &recent {
                let inst = if e.instance.is_empty() {
                    String::new()
                } else {
                    format!(":{}", e.instance)
                };
                println!(
                    "  [{adapter}{inst} · {phase}] {error}",
                    adapter = e.adapter,
                    inst = inst,
                    phase = e.phase,
                    error = e.error,
                );
            }
        }
    }

    // Per-source health table. Always emitted (even when empty) so the
    // operator sees the "no sources yet — run `anamnesis source add`"
    // affordance right after `init`.
    println!();
    if per_source.is_empty() {
        println!("sources by health:");
        println!("  (no sources registered — try `anamnesis discover` or `anamnesis source add`)");
    } else {
        println!(
            "{:<14} {:<14} {:<8} {:<8} {:<16} {}",
            "adapter", "instance", "records", "chunks", "last_import", "status"
        );
        for r in &per_source {
            let freshness = source_freshness(r.source.last_import_at, now);
            println!(
                "{:<14} {:<14} {:<8} {:<8} {:<16} {}",
                r.source.adapter,
                if r.source.instance.is_empty() {
                    "-".to_string()
                } else {
                    r.source.instance.clone()
                },
                r.record_count,
                r.chunk_count,
                freshness.age_human,
                freshness.label,
            );
        }
    }
    Ok(())
}

/// Human-readable freshness summary for a source, derived from
/// `last_import_at` (Unix seconds, `None` for never-imported sources).
///
/// `label` is one of `"never-imported" | "fresh" | "stale"`:
///   - `never-imported`: `last_import_at is None` (registered but no
///     import has landed)
///   - `fresh`: imported within the last 24 hours
///   - `stale`: imported more than 24 hours ago
///
/// `age_seconds` is `None` for never-imported, otherwise the gap from
/// `last_import_at` to `now`. `age_human` is the same gap rounded to a
/// short string: `<1m`, `5m`, `3h`, `2d`, `30d+`.
struct Freshness {
    label: &'static str,
    age_seconds: Option<i64>,
    age_human: String,
}

fn source_freshness(last_import_at: Option<i64>, now: i64) -> Freshness {
    match last_import_at {
        None => Freshness {
            label: "never-imported",
            age_seconds: None,
            age_human: "<never>".into(),
        },
        Some(t) => {
            let age = now.saturating_sub(t).max(0);
            let label = if age < 24 * 3600 { "fresh" } else { "stale" };
            Freshness {
                label,
                age_seconds: Some(age),
                age_human: human_age_short(age),
            }
        }
    }
}

/// Compact human-readable age: `<1m`, `5m`, `3h`, `2d`, `30d+`.
fn human_age_short(age_seconds: i64) -> String {
    if age_seconds < 60 {
        "<1m".into()
    } else if age_seconds < 3600 {
        format!("{}m", age_seconds / 60)
    } else if age_seconds < 24 * 3600 {
        format!("{}h", age_seconds / 3600)
    } else if age_seconds < 30 * 24 * 3600 {
        format!("{}d", age_seconds / (24 * 3600))
    } else {
        "30d+".into()
    }
}

async fn cmd_discover() -> Result<()> {
    let discovery = Discovery::new()
        .register(Box::new(ClaudeCodeDetector::new()))
        .register(Box::new(Mem0SqliteDetector::new()))
        .register(Box::new(CodexDetector::new()))
        .register(Box::new(LettaSqliteDetector::new()))
        .register(Box::new(HermesDetector::new()))
        .register(Box::new(OpenClawDetector::new()))
        .register(Box::new(TdaiDetector::new()))
        .register(Box::new(OpenVikingDetector::new()))
        .register(Box::new(MempalaceDetector::new()))
        .register(Box::new(MemoriDetector::new()))
        .register(Box::new(MemosDetector::new()))
        .register(Box::new(MemaryDetector::new()));
    let found = discovery.detect_all(&DetectOpts::default()).await;
    if found.is_empty() {
        println!("no known memory sources found at default locations");
        return Ok(());
    }
    println!(
        "{:<14} {:<8} {:<48} {}",
        "adapter", "conf", "location", "note"
    );
    for s in &found {
        let conf = match s.confidence {
            anamnesis_core::Confidence::High => "high",
            anamnesis_core::Confidence::Medium => "medium",
            anamnesis_core::Confidence::Low => "low",
        };
        println!(
            "{:<14} {:<8} {:<48} {}",
            s.adapter,
            conf,
            s.location,
            s.note.as_deref().unwrap_or("")
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// source add / list / remove
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_source(data_dir: &std::path::Path, sub: SourceCmd) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    match sub {
        SourceCmd::Add {
            adapter,
            instance,
            path,
            url,
            token_env,
        } => {
            // Round-17 (§19.3 PR-1): generic-mcp is now a first-class
            // CLI source — `source add generic-mcp --url ... [--token-env
            // ENV]` registers the upstream MCP HTTP server and lets the
            // subsequent `import generic-mcp:<instance>` pull from it
            // without any test-only construction code. The token *name*,
            // never the token *value*, lives in the registry (resolved
            // at import time via the operator's env).
            let location: Option<String> = match (path.as_ref(), url.as_ref()) {
                (Some(_), Some(_)) => {
                    // clap's `conflicts_with` should prevent this, but
                    // belt-and-braces.
                    return Err(anyhow!("--path and --url are mutually exclusive"));
                }
                (Some(p), None) => Some(p.display().to_string()),
                (None, Some(u)) => Some(u.clone()),
                (None, None) => None,
            };
            let config_json: Option<String> = token_env
                .as_deref()
                .map(|env| serde_json::json!({ "token_env": env }).to_string());
            // Sanity: for URL-based adapters, refuse to register without
            // a URL. This keeps `import generic-mcp:<i>` from later
            // failing with the confusing "no default path" error.
            if adapter == anamnesis_adapter_generic_mcp::ADAPTER_ID && url.is_none() {
                return Err(anyhow!(
                    "source add generic-mcp requires --url <upstream-mcp-url>"
                ));
            }
            store.register_source(
                &adapter,
                instance.as_deref(),
                location.as_deref(),
                config_json.as_deref(),
            )?;
            println!(
                "registered: {adapter}{}{}",
                instance
                    .as_deref()
                    .map(|i| format!(":{i}"))
                    .unwrap_or_default(),
                location.map(|l| format!(" @ {l}")).unwrap_or_default(),
            );
            Ok(())
        }
        SourceCmd::List {
            source,
            instance,
            json,
        } => {
            // Round-9: show per-source counts alongside last_import so
            // operators can spot "registered but empty" sources at a
            // glance — same signal MCP agents get from list_sources.
            // Round-82: add `tagged` column so operators can see
            // where their curated `user_tags` actually live.
            // Round-99: optional `--source` / `--instance` filter
            // mirrors R96 MCP `list_sources { source, instance }`.
            // Round-103: `--source` now also accepts comma-separated
            // OR (`--source mem0,claude-code`) via core's shared
            // `parse_csv_filter`, symmetric with R102 audit-tail
            // multi-value. Empty parse = no filter (back-compat).
            // Round 115: `--instance` now follows the same
            // comma-separated OR parser, symmetric with doctor.
            let rows = store.list_sources_with_counts()?;
            let total_registered = rows.len() as u64;
            let sources = anamnesis_core::parse_csv_filter(source.as_deref());
            let instances = anamnesis_core::parse_csv_filter(instance.as_deref());
            let filter_applied = !sources.is_empty() || !instances.is_empty();
            let rows: Vec<_> = rows
                .into_iter()
                .filter(|r| sources.is_empty() || sources.iter().any(|s| s == &r.source.adapter))
                .filter(|r| {
                    instances.is_empty() || instances.iter().any(|i| i == &r.source.instance)
                })
                .collect();
            if json {
                // Round 88: shape mirrors R86 `source_show.source`
                // + MCP `list_sources.sources[]`. Empty registry
                // returns `{ "sources": [] }` (not human prose)
                // so scripts can branch uniformly.
                //
                // Round 124 (PR-78as): top-level redacted summary
                // mirrors the MCP `list_sources` summary (R117)
                // and the CLI `status --json` summary (R123).
                // NEVER reads `r.source.location` (path) — only
                // counts, filter clauses, active model, and
                // whole-store stats.
                let stats = store.stats()?;
                let active_model_label = store
                    .active_model()
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| "none".to_string());
                let source_clause = if sources.is_empty() {
                    "source filter: all sources".to_string()
                } else {
                    format!("source filter: {}", sources.join(" OR "))
                };
                let instance_clause = if instances.is_empty() {
                    "instance filter: all instances".to_string()
                } else {
                    format!("instance filter: {}", instances.join(" OR "))
                };
                let summary = format!(
                    "{} source(s) returned (filtered from {} registered); {}; {}; active model: {}; stats reflect whole store ({} records, {} chunks).",
                    rows.len(),
                    total_registered,
                    source_clause,
                    instance_clause,
                    active_model_label,
                    stats.records,
                    stats.chunks,
                );
                let payload = serde_json::json!({
                    "summary": summary,
                    "sources": rows.iter().map(render_source_with_counts_json).collect::<Vec<_>>(),
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
                return Ok(());
            }
            if rows.is_empty() {
                if filter_applied {
                    println!("no sources matched filter");
                } else {
                    println!("no sources registered");
                }
            } else {
                println!(
                    "{:<14} {:<14} {:<8} {:<8} {:<8} {:<20} {}",
                    "adapter", "instance", "records", "chunks", "tagged", "last_import", "location"
                );
                for r in rows {
                    let last = r
                        .source
                        .last_import_at
                        .map(|t| {
                            chrono::DateTime::<chrono::Utc>::from_timestamp(t, 0)
                                .map(|d| d.format("%Y-%m-%dT%H:%MZ").to_string())
                                .unwrap_or_else(|| t.to_string())
                        })
                        .unwrap_or_else(|| "<never>".into());
                    println!(
                        "{:<14} {:<14} {:<8} {:<8} {:<8} {:<20} {}",
                        r.source.adapter,
                        r.source.instance,
                        r.record_count,
                        r.chunk_count,
                        r.tagged_record_count,
                        last,
                        r.source.location.unwrap_or_default(),
                    );
                }
            }
            Ok(())
        }
        SourceCmd::Remove { target } => {
            let (adapter, instance) = split_target(&target);
            store.deregister_source(adapter, instance)?;
            println!("removed: {target}");
            Ok(())
        }
        SourceCmd::Show {
            target,
            errors,
            json,
        } => cmd_source_show(&store, &target, errors, json),
    }
}

/// Round 88 (PR-78j): render a `SourceWithCounts` as the shared
/// `source list --json` / `source show --json` source-object
/// shape. Same field set the MCP `list_sources` wire emits, so
/// any tool that already consumes one can parse the other.
fn render_source_with_counts_json(swc: &anamnesis_store::SourceWithCounts) -> serde_json::Value {
    serde_json::json!({
        "adapter": swc.source.adapter,
        "instance": if swc.source.instance.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(swc.source.instance.clone())
        },
        "location": swc.source.location,
        "added_at": swc.source.added_at,
        "last_import_at": swc.source.last_import_at,
        "record_count": swc.record_count,
        "chunk_count": swc.chunk_count,
        "tagged_record_count": swc.tagged_record_count,
    })
}

/// Round 86 (PR-78h): `anamnesis source show <adapter[:instance]>`
/// — per-source detail. Missing source is a loud non-zero exit so
/// a typo in scripted usage doesn't pass silently.
fn cmd_source_show(store: &Store, target: &str, errors: usize, json: bool) -> Result<()> {
    let errors = errors.clamp(1, 10);
    let (adapter, instance) = split_target(target);
    let swc = match store.get_source_with_counts(adapter, instance)? {
        Some(s) => s,
        None => return Err(anyhow!("source not found: {target}")),
    };
    let recent = store.recent_import_errors_for_source(adapter, instance, errors)?;

    if json {
        // Round 128 (PR-78aw): top-level redacted summary
        // mirrors MCP R118 source_show summary. NEVER reads
        // `source.location`, `recent_import_errors[].error`,
        // `native_path`, `native_id`, or `raw_hash`.
        let target_label = match instance {
            Some(i) => format!("{adapter}:{i}"),
            None => adapter.to_string(),
        };
        let last_import_label = match swc.source.last_import_at {
            Some(ts) => ts.to_string(),
            None => "never".to_string(),
        };
        let summary = format!(
            "{target_label} source_show: {} record(s), {} chunk(s), {} tagged record(s); recent import errors: {} returned (limit {}); last import: {}.",
            swc.record_count,
            swc.chunk_count,
            swc.tagged_record_count,
            recent.len(),
            errors,
            last_import_label,
        );

        let payload = serde_json::json!({
            "summary": summary,
            "error_limit": errors,
            "source": {
                "adapter": swc.source.adapter,
                "instance": if swc.source.instance.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(swc.source.instance.clone())
                },
                "location": swc.source.location,
                "added_at": swc.source.added_at,
                "last_import_at": swc.source.last_import_at,
                "record_count": swc.record_count,
                "chunk_count": swc.chunk_count,
                "tagged_record_count": swc.tagged_record_count,
            },
            "recent_import_errors": recent.iter().map(|e| serde_json::json!({
                "adapter": e.adapter,
                "instance": if e.instance.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(e.instance.clone())
                },
                "native_id": e.native_id,
                "native_path": e.native_path,
                "phase": e.phase,
                "error": e.error,
                "occurred_at": e.occurred_at,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        let inst_label = instance_label(&swc.source.instance);
        let last_import = swc
            .source
            .last_import_at
            .map(|t| {
                chrono::DateTime::<chrono::Utc>::from_timestamp(t, 0)
                    .map(|d| d.format("%Y-%m-%dT%H:%MZ").to_string())
                    .unwrap_or_else(|| t.to_string())
            })
            .unwrap_or_else(|| "<never>".into());
        println!("adapter      : {}", swc.source.adapter);
        println!("instance     : {inst_label}");
        if let Some(loc) = &swc.source.location {
            println!("location     : {loc}");
        }
        println!("added_at     : {}", swc.source.added_at);
        println!("last_import  : {last_import}");
        println!("records      : {}", swc.record_count);
        println!("chunks       : {}", swc.chunk_count);
        println!("tagged       : {}", swc.tagged_record_count);
        if recent.is_empty() {
            println!("recent import errors : (none)");
        } else {
            println!("recent import errors (top {}):", recent.len());
            for e in &recent {
                println!(
                    "  [{phase} @ {occurred_at}] {error} (native_id={native_id})",
                    phase = e.phase,
                    occurred_at = e.occurred_at,
                    error = e.error,
                    native_id = e.native_id.as_deref().unwrap_or("-"),
                );
                if let Some(p) = &e.native_path {
                    println!("       native_path: {p}");
                }
            }
        }
    }
    Ok(())
}

fn split_target(t: &str) -> (&str, Option<&str>) {
    match t.split_once(':') {
        Some((a, i)) => (a, Some(i)),
        None => (t, None),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// import
// ─────────────────────────────────────────────────────────────────────────────

async fn cmd_import(
    data_dir: &std::path::Path,
    target: &str,
    full: bool,
    since_arg: Option<&str>,
    dry_run: bool,
    no_embed: bool,
    path_override: Option<&std::path::Path>,
) -> Result<()> {
    let (adapter_id, instance) = split_target(target);

    // Reject unknown adapters before doing any registry / filesystem work
    // so the error message is "not wired" rather than the more confusing
    // "no default path".
    if !is_known_adapter(adapter_id) {
        return Err(anyhow!(
            "adapter {adapter_id:?} not wired; supported: claude-code, codex, mem0, letta, hermes, openclaw, tdai, generic-mcp"
        ));
    }

    // Round-19 (§-1.5 PR-4a): resolve the effective `ScanOpts`.
    //
    //   --full         → forced full scan, `since = None`.
    //   --since <ts>   → explicit incremental bound (RFC3339 / chrono parse).
    //   neither        → default to the source's registry `last_import_at`
    //                    (auto-incremental). On a fresh source, this is
    //                    `None` and the run is effectively a full scan.
    //
    // `--full` and `--since` are mutually exclusive at the clap layer.
    let scan_opts = resolve_scan_opts(data_dir, adapter_id, instance, full, since_arg)?;
    if !dry_run {
        match (full, scan_opts.since.as_ref()) {
            (true, _) => eprintln!("import: --full → ignoring any incremental window"),
            (false, Some(t)) => eprintln!(
                "import: incremental since {t} (override with --full for a complete re-scan)"
            ),
            (false, None) => {}
        }
    }

    // PR-B (BLUEPRINT §18.4 F5): the source registry is the canonical
    // truth for "where does X live". Resolution order:
    //
    //   1. --path P    → trusted override; we'll register/overwrite P
    //                    so the registry catches up to the explicit user
    //                    intent.
    //   2. registry    → use the location the user registered earlier via
    //                    `source add`.
    //   3. fallback    → adapter default path; auto-registered on success
    //                    so the next `import` is no longer ambiguous.
    //
    // We never silently fall back from a registered (but missing) path
    // to the adapter default — that would mask a misconfiguration; the
    // adapter's health check will report the failure instead.
    let store_for_lookup = Store::open(db_path(data_dir))?;
    let registered = store_for_lookup.get_source(adapter_id, instance)?;
    let registered_location = registered.as_ref().and_then(|r| r.location.clone());
    let registered_config = registered.as_ref().and_then(|r| r.config_json.clone());
    drop(store_for_lookup);

    // Round-17 (§19.3 PR-1): generic-mcp is URL-based, not path-based.
    // It does NOT use `default_path_for` (URLs have no useful default —
    // the operator must register one via `source add generic-mcp --url`)
    // and it does NOT pass a PathBuf to `run_import`.
    if adapter_id == anamnesis_adapter_generic_mcp::ADAPTER_ID {
        if path_override.is_some() {
            return Err(anyhow!(
                "generic-mcp is URL-based; use `anamnesis source add generic-mcp --url <url>` \
                 instead of `--path`"
            ));
        }
        let url = registered_location.ok_or_else(|| {
            anyhow!(
                "generic-mcp source {target:?} is not registered. Run `anamnesis source add \
             generic-mcp{instance_suffix} --url <upstream-mcp-url> [--token-env ENV]` first.",
                target = target,
                instance_suffix = instance
                    .as_ref()
                    .map(|i| format!(":{i}"))
                    .unwrap_or_default(),
            )
        })?;
        let token = resolve_generic_mcp_token(registered_config.as_deref())?;
        let adapter = anamnesis_adapter_generic_mcp::generic_mcp_adapter(
            url.clone(),
            token.as_deref(),
            instance,
        );
        return run_import(data_dir, &adapter, dry_run, no_embed, None, true, scan_opts).await;
    }

    let (location, source_was_explicit) = match path_override {
        Some(p) => (p.to_path_buf(), true),
        None => match registered_location {
            Some(loc) => (PathBuf::from(loc), true),
            None => (default_path_for(adapter_id)?, false),
        },
    };

    match adapter_id {
        anamnesis_adapter_claude_code::ADAPTER_ID => {
            let adapter = ClaudeCodeAdapter::new(ClaudeCodeConfig {
                projects_root: location.clone(),
                instance: instance.map(str::to_owned),
            });
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_mem0::ADAPTER_ID => {
            let adapter = mem0_sqlite_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_codex::ADAPTER_ID => {
            let adapter = codex_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_letta::ADAPTER_ID => {
            let adapter = letta_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_hermes::ADAPTER_ID => {
            let adapter = hermes_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_openclaw::ADAPTER_ID => {
            let adapter = openclaw_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_tdai::ADAPTER_ID => {
            let adapter = tdai_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_openviking::ADAPTER_ID => {
            let adapter = openviking_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_mempalace::ADAPTER_ID => {
            let adapter = mempalace_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_memori::ADAPTER_ID => {
            let adapter = memori_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_memos::ADAPTER_ID => {
            let adapter = memos_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_memary::ADAPTER_ID => {
            let adapter = memary_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        other => Err(anyhow!(
            "adapter {other:?} not wired; supported: claude-code, codex, mem0, letta, hermes, openclaw, tdai, openviking, mempalace, memori, memos, memary, generic-mcp"
        )),
    }
}

/// Resolve the effective `ScanOpts` for a CLI `import` invocation.
///
/// Order of precedence:
///   1. `--full`             → `ScanOpts { since: None, full: true }`
///   2. `--since "<ts>"`     → `ScanOpts { since: Some(parsed), full: false }`
///   3. neither → look up the source's registry `last_import_at` and use
///      it as the increment bound. On a fresh source (`last_import_at`
///      is `None`) this is effectively a full scan.
///
/// `--full` and `--since` are mutually exclusive at the clap layer;
/// belt-and-braces check here too.
fn resolve_scan_opts(
    data_dir: &std::path::Path,
    adapter_id: &str,
    instance: Option<&str>,
    full: bool,
    since_arg: Option<&str>,
) -> Result<anamnesis_core::adapter::ScanOpts> {
    use anamnesis_core::adapter::ScanOpts;
    if full && since_arg.is_some() {
        return Err(anyhow!("--full and --since are mutually exclusive"));
    }
    if full {
        return Ok(ScanOpts {
            since: None,
            full: true,
        });
    }
    if let Some(s) = since_arg {
        let parsed = chrono::DateTime::parse_from_rfc3339(s)
            .map_err(|e| anyhow!("--since must be RFC3339 (e.g. 2026-04-01T00:00:00Z): {e}"))?
            .with_timezone(&chrono::Utc);
        return Ok(ScanOpts {
            since: Some(parsed),
            full: false,
        });
    }
    // Fall back to the source's registered `last_import_at`.
    let store = Store::open(db_path(data_dir))?;
    let row = store.get_source(adapter_id, instance)?;
    let since = row
        .and_then(|r| r.last_import_at)
        .and_then(|t| chrono::DateTime::<chrono::Utc>::from_timestamp(t, 0));
    Ok(ScanOpts { since, full: false })
}

/// Read the bearer token for a generic-mcp source from the operator's
/// environment, looking up the env var name stored in
/// `sources.config_json` (`{"token_env": "ANAMNESIS_FOO_TOKEN"}`).
///
/// Returns `Ok(None)` when no `token_env` was registered (the upstream
/// allows unauthenticated access). Errors when the env var name is
/// registered but the variable is unset — that's a misconfiguration the
/// operator probably wants to know about *before* the import hits 401.
fn resolve_generic_mcp_token(config_json: Option<&str>) -> Result<Option<String>> {
    let Some(raw) = config_json else {
        return Ok(None);
    };
    let parsed: serde_json::Value = serde_json::from_str(raw)
        .map_err(|e| anyhow!("source.config_json is not valid JSON: {e}"))?;
    let Some(env) = parsed.get("token_env").and_then(|v| v.as_str()) else {
        return Ok(None);
    };
    match std::env::var(env) {
        Ok(v) if !v.is_empty() => Ok(Some(v)),
        Ok(_) => Err(anyhow!(
            "generic-mcp source's token_env={env:?} is set but empty"
        )),
        Err(_) => Err(anyhow!(
            "generic-mcp source requires env var {env:?} to be set with the bearer token"
        )),
    }
}

/// Whether `cmd_import` knows how to drive this adapter id. Used as the
/// up-front gate so unknown adapters get a clear "not wired" error before
/// any registry / filesystem work happens.
fn is_known_adapter(adapter_id: &str) -> bool {
    matches!(
        adapter_id,
        anamnesis_adapter_claude_code::ADAPTER_ID
            | anamnesis_adapter_mem0::ADAPTER_ID
            | anamnesis_adapter_codex::ADAPTER_ID
            | anamnesis_adapter_letta::ADAPTER_ID
            | anamnesis_adapter_hermes::ADAPTER_ID
            | anamnesis_adapter_openclaw::ADAPTER_ID
            | anamnesis_adapter_tdai::ADAPTER_ID
            | anamnesis_adapter_openviking::ADAPTER_ID
            | anamnesis_adapter_mempalace::ADAPTER_ID
            | anamnesis_adapter_memori::ADAPTER_ID
            | anamnesis_adapter_memos::ADAPTER_ID
            | anamnesis_adapter_memary::ADAPTER_ID
            | anamnesis_adapter_generic_mcp::ADAPTER_ID
    )
}

/// Adapter default discovery paths — used when neither `--path` nor a
/// registered location is available. Keep in sync with each adapter's
/// detector. Callers must gate on `is_known_adapter` first.
fn default_path_for(adapter_id: &str) -> Result<PathBuf> {
    match adapter_id {
        anamnesis_adapter_claude_code::ADAPTER_ID => home_join(&[".claude", "projects"]),
        anamnesis_adapter_mem0::ADAPTER_ID => home_join(&[".mem0", "db.sqlite"]),
        anamnesis_adapter_codex::ADAPTER_ID => home_join(&[".codex"]),
        anamnesis_adapter_letta::ADAPTER_ID => home_join(&[".letta", "letta.db"]),
        anamnesis_adapter_hermes::ADAPTER_ID => home_join(&[".hermes"]),
        anamnesis_adapter_openclaw::ADAPTER_ID => home_join(&[".openclaw"]),
        anamnesis_adapter_tdai::ADAPTER_ID => home_join(&[".openclaw", "memory-tdai"]),
        anamnesis_adapter_openviking::ADAPTER_ID => home_join(&[".openviking", "data"]),
        anamnesis_adapter_mempalace::ADAPTER_ID => home_join(&[".mempalace"]),
        anamnesis_adapter_memori::ADAPTER_ID => home_join(&[".memori", "memori.db"]),
        anamnesis_adapter_memos::ADAPTER_ID => home_join(&[".memos"]),
        anamnesis_adapter_memary::ADAPTER_ID => home_join(&[".memary", "data"]),
        other => Err(anyhow!("no default path for adapter {other:?}")),
    }
}

async fn run_import<A: anamnesis_core::adapter::MemoryAdapter>(
    data_dir: &std::path::Path,
    adapter: &A,
    dry_run: bool,
    no_embed: bool,
    canonical_location: Option<&std::path::Path>,
    source_was_explicit: bool,
    scan_opts: anamnesis_core::adapter::ScanOpts,
) -> Result<()> {
    // Round-18 (§-1.5 PR-3): the side effects of an import — preserving
    // registry `config_json`, stamping `last_import_at`, appending to
    // `audit.log` — used to live inline here. They now live in
    // `ImportService` so the MCP `tool_import_source` path produces the
    // exact same system-state delta. The CLI keeps the local pieces
    // that MCP shouldn't do: a) the human-readable progress line, b)
    // running the embedding worker (which can take minutes — fine for
    // CLI, but would block an MCP JSON-RPC request indefinitely).
    //
    // Round-19 (§-1.5 PR-4a): `scan_opts` now flows from the CLI
    // resolver (--full / --since / auto-from-`last_import_at`) into
    // `ImportOptions.scan_opts` and through to `adapter.scan(opts)`.
    let store = Store::open(db_path(data_dir))?;
    let service = ImportService::new(&store, audit(data_dir));
    let summary = service
        .import(
            adapter,
            ImportOptions {
                dry_run,
                canonical_location: canonical_location.map(|p| p.display().to_string()),
                source_was_explicit,
                scan_opts,
            },
        )
        .await
        .map_err(|e| anyhow!("import: {e}"))?;

    if dry_run {
        println!("dry-run: would import {} raw record(s)", summary.raw_seen);
        return Ok(());
    }

    println!(
        "import done: {} raw, {} upserted, {} chunks, {} errors",
        summary.raw_seen, summary.records_upserted, summary.chunks_written, summary.errors
    );

    if !no_embed {
        run_embed_worker(&store).await?;
    }
    Ok(())
}

fn audit(data_dir: &std::path::Path) -> anamnesis_core::Audit {
    anamnesis_core::Audit::new(data_dir)
}

// ─────────────────────────────────────────────────────────────────────────────
// export
// ─────────────────────────────────────────────────────────────────────────────

/// Round 140 (PR-78bi): refactored to delegate to the shared
/// `anamnesis-export` crate. Same wire behaviour as R138/R139 (CLI
/// `--format` default `jsonl`, `out` defaults to stdout for
/// jsonl/csv, REQUIRED for `mem0-sqlite` / `letta-sqlite`).
/// The shared crate hosts the format dispatch, filter parsing,
/// SQLite safety guard, and provenance metadata convention — so
/// the MCP `export_memories` tool runs through the same writers.
fn cmd_export(
    data_dir: &std::path::Path,
    out: Option<&std::path::Path>,
    format: &str,
    source: Option<&str>,
) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let fmt = anamnesis_export::ExportFormat::parse(format).map_err(|e| anyhow!("{e}"))?;
    let filter = anamnesis_export::ExportFilter {
        source: source.map(str::to_owned),
        instance: None,
        kind: None,
    };

    // jsonl/csv with no `--out` writes to stdout (R0 behaviour);
    // SQLite formats are guarded by the shared validate_sqlite_output.
    let outcome = if matches!(
        fmt,
        anamnesis_export::ExportFormat::Jsonl | anamnesis_export::ExportFormat::Csv
    ) && out.is_none()
    {
        let mut writer = std::io::stdout();
        anamnesis_export::run_export(&store, &filter, fmt, None, Some(&mut writer))
            .map_err(|e| anyhow!("{e}"))?
    } else {
        anamnesis_export::run_export(&store, &filter, fmt, out, None).map_err(|e| anyhow!("{e}"))?
    };

    if let Some(p) = &outcome.out {
        eprintln!("exported {} record(s) to {}", outcome.records, p.display());
    } else {
        eprintln!("exported {} record(s)", outcome.records);
    }
    audit(data_dir).record(anamnesis_core::AuditEntry::new(
        "export",
        serde_json::json!({
            "format":  fmt.as_token(),
            "source":  source,
            "out":     outcome.out.as_ref().map(|p| p.display().to_string()),
            "records": outcome.records,
        }),
    ));
    Ok(())
}

fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// verify
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_verify(data_dir: &std::path::Path, repair: bool) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let mut problems = 0u64;

    // All raw-conn diagnostic queries are scoped tightly so the
    // parking_lot Mutex guard never overlaps with calls back into Store
    // (active_model / rebuild_embedding_jobs would otherwise deadlock).

    // 1. SQLite integrity_check.
    let integrity: String = {
        let conn = store.conn();
        conn.query_row("PRAGMA integrity_check(1)", [], |r| r.get(0))?
    };
    if integrity == "ok" {
        println!("integrity_check : ok");
    } else {
        println!("integrity_check : {integrity}");
        problems += 1;
    }

    // 2. records → record_chunks consistency.
    let (records_count, records_with_chunks): (i64, i64) = {
        let conn = store.conn();
        let rc: i64 = conn.query_row("SELECT COUNT(1) FROM records", [], |r| r.get(0))?;
        let rwc: i64 = conn.query_row(
            "SELECT COUNT(1) FROM records r WHERE EXISTS (SELECT 1 FROM record_chunks c WHERE c.record_id = r.id)",
            [],
            |r| r.get(0),
        )?;
        (rc, rwc)
    };
    let orphan_records = records_count - records_with_chunks;
    if orphan_records == 0 {
        println!("orphan records  : 0");
    } else {
        println!("orphan records  : {orphan_records} (no chunks)");
        problems += 1;
    }

    // 3. FTS index vs record_chunks row count.
    let (chunks_count, fts_count): (i64, i64) = {
        let conn = store.conn();
        let cc: i64 = conn.query_row("SELECT COUNT(1) FROM record_chunks", [], |r| r.get(0))?;
        let fc: i64 = conn.query_row("SELECT COUNT(1) FROM chunks_fts", [], |r| r.get(0))?;
        (cc, fc)
    };
    if chunks_count == fts_count {
        println!("FTS index       : ok ({chunks_count} rows)");
    } else {
        println!("FTS index       : drift ({chunks_count} chunks vs {fts_count} FTS rows)");
        problems += 1;
        if repair {
            println!("FTS index       : rebuilding…");
            let conn = store.conn();
            conn.execute("DELETE FROM chunks_fts", [])?;
            conn.execute(
                "INSERT INTO chunks_fts(rowid, content) SELECT rowid, content FROM record_chunks",
                [],
            )?;
            println!("FTS index       : rebuilt");
        }
    }

    // 4. embeddings vs active model — count chunks that lack an embedding
    //    under the current model.
    if let Some(active) = store.active_model()? {
        let missing: i64 = {
            let conn = store.conn();
            conn.query_row(
                "SELECT COUNT(1) FROM record_chunks c \
                 WHERE NOT EXISTS (SELECT 1 FROM chunk_embeddings e \
                    WHERE e.chunk_id = c.id AND e.model_id = ?1)",
                [&active],
                |r| r.get(0),
            )?
        };
        println!("missing embeds  : {missing} (model: {active})");
        if missing > 0 && repair {
            let n = store.rebuild_embedding_jobs(&active)?;
            println!("missing embeds  : re-queued {n} embedding job(s)");
        }
    } else {
        println!("missing embeds  : skipped (no active model)");
    }

    if problems == 0 {
        println!("status          : healthy");
    } else if repair {
        println!("status          : repair attempted on {problems} issue(s)");
    } else {
        println!("status          : {problems} issue(s) found (run with --repair)");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Kind / Scope parsers (shared by search filters)
// ─────────────────────────────────────────────────────────────────────────────

fn parse_kind(s: &str) -> Result<anamnesis_core::Kind> {
    use anamnesis_core::Kind;
    Ok(match s {
        "fact" => Kind::Fact,
        "preference" => Kind::Preference,
        "feedback" => Kind::Feedback,
        "reference" => Kind::Reference,
        "episode" => Kind::Episode,
        "skill" => Kind::Skill,
        "unknown" => Kind::Unknown,
        other => return Err(anyhow!("unknown kind: {other}")),
    })
}

fn parse_scope(s: &str) -> Result<anamnesis_core::Scope> {
    use anamnesis_core::Scope;
    Ok(match s {
        "user" => Scope::User,
        "project" => Scope::Project,
        "session" => Scope::Session,
        "ephemeral" => Scope::Ephemeral,
        other => return Err(anyhow!("unknown scope: {other}")),
    })
}

fn home_join(parts: &[&str]) -> Result<PathBuf> {
    let mut p = dirs_home()?;
    for part in parts {
        p = p.join(part);
    }
    Ok(p)
}

// ─────────────────────────────────────────────────────────────────────────────
// search
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn cmd_search(
    data_dir: &std::path::Path,
    query: &str,
    source: Option<&str>,
    instance: Option<&str>,
    kind: Option<&str>,
    scope: Option<&str>,
    since: Option<&str>,
    until: Option<&str>,
    limit: u32,
    mode_str: &str,
    json: bool,
    trace: bool,
    user_tag: Option<&str>,
    explain: bool,
) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let mode = match mode_str {
        "fulltext" => SearchMode::Fulltext,
        "vector" => SearchMode::Vector,
        _ => SearchMode::Hybrid,
    };
    let kind_filter = match kind {
        Some(k) => Some(parse_kind(k)?),
        None => None,
    };
    let scope_filter = match scope {
        Some(s) => Some(parse_scope(s)?),
        None => None,
    };
    // Round-22 (§-1.5 PR-5): RFC3339 → unix-seconds for the
    // `SearchFilter.time_from` / `time_to` SQL pushdown. Clap-level
    // validation (`Option<String>`) keeps the wire ergonomic; the
    // adapter-level parse error here is what users see if they hand-
    // type a malformed timestamp.
    let time_from = match since {
        Some(s) => Some(
            chrono::DateTime::parse_from_rfc3339(s)
                .map_err(|e| anyhow!("--since must be RFC3339 (e.g. 2026-04-01T00:00:00Z): {e}"))?
                .with_timezone(&chrono::Utc)
                .timestamp(),
        ),
        None => None,
    };
    let time_to = match until {
        Some(s) => Some(
            chrono::DateTime::parse_from_rfc3339(s)
                .map_err(|e| anyhow!("--until must be RFC3339 (e.g. 2026-04-30T23:59:59Z): {e}"))?
                .with_timezone(&chrono::Utc)
                .timestamp(),
        ),
        None => None,
    };

    // Embedding provider needed for Vector/Hybrid modes.
    let provider = match mode {
        SearchMode::Fulltext => None,
        _ => Some(open_active_provider(data_dir, &store)?),
    };

    // Round 79: normalise the user-tag through the same code
    // `tag_record` writes through, so `--user-tag Keep` finds
    // tags stored as `keep`.
    let user_tag_normalised = match user_tag {
        Some(raw) => Some(anamnesis_store::normalize_user_tag_name(raw)?),
        None => None,
    };

    // PR-C: build the SQL-level filter from the same CLI knobs the
    // post-filter used to consume. We turn `Kind` / `Scope` back into
    // their lower-case string form to match how the store writes them.
    let store_filter = anamnesis_store::SearchFilter {
        source: source.map(str::to_owned),
        instance: instance.map(str::to_owned),
        kind: kind_filter.map(|k| format!("{k:?}").to_lowercase()),
        scope: scope_filter.map(|s| format!("{s:?}").to_lowercase()),
        time_from,
        time_to,
        user_tag: user_tag_normalised,
    };

    // Round 76: always use the traced primitive so the live search
    // and the optional --trace output can never drift. The trace
    // struct is dropped when `trace=false`, so this is free at the
    // default wire — same pattern the MCP `search_memories` tool
    // uses since R71.
    let traced =
        run_search_traced(&store, query, &store_filter, limit, mode, provider.as_ref()).await?;
    let hits = traced.hits;
    let search_trace = traced.trace;

    let t_pack = std::time::Instant::now();
    let packed = pack(
        &store,
        &hits,
        &ContextBudget {
            max_records: limit as usize,
            ..ContextBudget::default()
        },
    )?;
    let pack_ms = t_pack.elapsed().as_millis() as u64;

    let filtered: Vec<_> = packed
        .into_iter()
        .filter(|p| source.is_none_or(|src| p.record.source.adapter == src))
        .filter(|p| kind_filter.is_none_or(|k| p.record.kind == k))
        .filter(|p| scope_filter.is_none_or(|s| p.record.scope == s))
        .collect();

    audit(data_dir).record(anamnesis_core::AuditEntry::new(
        "search",
        serde_json::json!({
            "query": query,
            "source": source,
            "kind": kind,
            "scope": scope,
            "mode": mode_str,
            "limit": limit,
            "hits": filtered.len(),
        }),
    ));

    if json {
        // Round 129 (PR-78ax): top-level redacted summary
        // mirrors MCP R119 search_memories summary. Closes
        // the CLI summary mirror campaign (7/7).
        // NEVER reads `query` arg body or snippet/record_id/
        // chunk_id/native_path. `query: redacted` is explicit.
        let effective_mode = format!("{:?}", search_trace.effective_mode).to_lowercase();
        let source_clause = match source {
            Some(s) if !s.is_empty() => format!("source filter: {s}"),
            _ => "source filter: all sources".to_string(),
        };
        let instance_clause = match instance {
            Some(s) if !s.is_empty() => format!("instance filter: {s}"),
            _ => "instance filter: all instances".to_string(),
        };
        let kind_clause = match store_filter.kind.as_deref() {
            Some(s) if !s.is_empty() => format!("kind filter: {s}"),
            _ => "kind filter: all kinds".to_string(),
        };
        let scope_clause = match store_filter.scope.as_deref() {
            Some(s) if !s.is_empty() => format!("scope filter: {s}"),
            _ => "scope filter: all scopes".to_string(),
        };
        let user_tag_clause = match store_filter.user_tag.as_deref() {
            Some(s) if !s.is_empty() => format!("user_tag filter: {s}"),
            _ => "user_tag filter: absent".to_string(),
        };
        let summary_text = format!(
            "{} result(s) returned; query: redacted; effective mode: {}; limit {}; {}; {}; {}; {}; {}; since: {}; until: {}; trace: {}; explain: {}.",
            filtered.len(),
            effective_mode,
            limit,
            source_clause,
            instance_clause,
            kind_clause,
            scope_clause,
            user_tag_clause,
            if store_filter.time_from.is_some() { "set" } else { "unset" },
            if store_filter.time_to.is_some() { "set" } else { "unset" },
            if trace { "included" } else { "omitted" },
            if explain { "included" } else { "omitted" },
        );

        let mut payload = serde_json::json!({
            "summary": summary_text,
            "query": query,
            "mode": mode_str,
            // Round-8: same expanded wire format as the MCP server so
            // CLI and MCP consumers can rely on identical JSON shapes.
            "results": filtered.iter().map(|p| {
                let best = p.matched_chunks.first();
                let mut row = serde_json::json!({
                    "record_id": p.record.id.0,
                    "trace_id": p.record.id.0,
                    "chunk_id": best.map(|c| c.chunk_id.clone()),
                    "adapter": p.record.source.adapter,
                    "instance": p.record.source.instance,
                    "kind": format!("{:?}", p.record.kind).to_lowercase(),
                    "scope": format!("{:?}", p.record.scope).to_lowercase(),
                    "score": p.score,
                    "rrf_score": p.score,
                    "fts_score": best.and_then(|c| c.fts_score),
                    "vector_score": best.and_then(|c| c.vector_score),
                    "from_fts": best.map(|c| c.from_fts).unwrap_or(false),
                    "from_vec": best.map(|c| c.from_vec).unwrap_or(false),
                    "snippet": best.map(|c| c.content.clone()).unwrap_or_default(),
                    "native_path": p.record.provenance.native_path,
                    "created_at": p.record.created_at.timestamp(),
                    "updated_at": p.record.updated_at.map(|t| t.timestamp()),
                    // Round 78: user-tag overlay. Always emitted —
                    // empty array when the record has no user tags.
                    "user_tags": p.record.user_tags,
                });
                // Round 87 (PR-78i): opt-in score breakdown.
                // Attached only when --explain is set, keeping
                // the default wire shape byte-stable.
                if explain {
                    row["explain"] = render_score_explain(&p.score_explain());
                }
                row
            }).collect::<Vec<_>>(),
        });
        if trace {
            // Same byte-shape as the MCP `search_memories(trace=true)`
            // payload from R71 — so any tooling that already parses
            // one can parse the other. Strictly numeric stage shape +
            // resolved mode; query text and snippet stay outside the
            // trace object.
            let returned_records = filtered.len() as u32;
            payload["trace"] = serde_json::json!({
                "effective_mode": search_trace.effective_mode,
                "candidate_pool": search_trace.candidate_pool,
                "stages_ms": {
                    "embed_query": search_trace.stages_ms.embed_query_ms,
                    "fts":         search_trace.stages_ms.fts_ms,
                    "vec":         search_trace.stages_ms.vec_ms,
                    "rrf":         search_trace.stages_ms.rrf_ms,
                    "pack":        pack_ms,
                },
                "counts": {
                    "fts_hits":         search_trace.counts.fts_hits,
                    "vec_hits":         search_trace.counts.vec_hits,
                    "ranked_chunks":    search_trace.counts.ranked_chunks,
                    "returned_records": returned_records,
                },
            });
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else if filtered.is_empty() {
        println!("no results");
        if trace {
            print_human_search_trace(&search_trace, pack_ms, 0);
        }
    } else {
        // Round-12: human-readable card mirrors the JSON wire format
        // (PR-#16) — same fields, same names, same semantics. CLI
        // operators see what MCP agents see; nothing is invented or
        // recomputed.
        for (i, p) in filtered.iter().enumerate() {
            let best = p.matched_chunks.first();
            let kind = format!("{:?}", p.record.kind).to_lowercase();
            let scope = format!("{:?}", p.record.scope).to_lowercase();

            // Line 1: rank, RRF score, adapter[:instance], kind/scope.
            let inst = p
                .record
                .source
                .instance
                .as_deref()
                .filter(|s| !s.is_empty())
                .map(|s| format!(":{s}"))
                .unwrap_or_default();
            println!(
                "[{:>2}] rrf={:.3}  {}{}  ({kind}, {scope})",
                i + 1,
                p.score,
                p.record.source.adapter,
                inst,
            );

            // Line 2: per-modality score breakdown + timestamps. Same
            // raw values the JSON exposes; null modality scores rendered
            // as `-` so the column line stays parseable visually.
            let fts = best
                .and_then(|c| c.fts_score)
                .map(|s| format!("{s:.3}"))
                .unwrap_or_else(|| "-".into());
            let vec = best
                .and_then(|c| c.vector_score)
                .map(|s| format!("{s:.3}"))
                .unwrap_or_else(|| "-".into());
            let created = p.record.created_at.format("%Y-%m-%dT%H:%MZ");
            let updated = p
                .record
                .updated_at
                .map(|t| t.format("%Y-%m-%dT%H:%MZ").to_string())
                .unwrap_or_else(|| "-".into());
            println!("     fts={fts}  vec={vec}  created={created}  updated={updated}");

            // Line 3: trace ids — exactly what an agent / a follow-up
            // CLI invocation would feed into `trace_provenance` /
            // `get_record`. Surface both record_id and chunk_id so the
            // operator can copy-paste either.
            let chunk_id = best
                .map(|c| c.chunk_id.clone())
                .unwrap_or_else(|| "-".into());
            println!(
                "     record_id={}  chunk_id={}  trace_id={}",
                p.record.id.0, chunk_id, p.record.id.0,
            );

            // Line 4: native_path (full, so operators can `cat $path`).
            println!(
                "     native_path={}",
                p.record.provenance.native_path.as_deref().unwrap_or("-"),
            );

            // Line 5: snippet (truncated on char boundary to stay terminal-safe).
            if let Some(c) = best {
                let snippet = c.content.replace('\n', " ");
                let snippet = if snippet.chars().count() > 180 {
                    let mut s: String = snippet.chars().take(180).collect();
                    s.push('…');
                    s
                } else {
                    snippet
                };
                println!("     snippet: {snippet}");
            }
            // Round 78: user-tag overlay. Only printed when non-
            // empty so untagged records stay quiet — most records
            // have zero tags and noise would dilute the card.
            if !p.record.user_tags.is_empty() {
                println!("     user_tags: {}", p.record.user_tags.join(", "));
            }
            println!();
        }
        if trace {
            print_human_search_trace(&search_trace, pack_ms, filtered.len() as u32);
        }
    }
    Ok(())
}

/// Round 87 (PR-78i): render `RecordScoreExplain` as JSON for
/// the `--explain` payload. Shared between CLI and MCP via the
/// `anamnesis-search` crate's struct, so the two surfaces can't
/// drift on field names or arithmetic.
fn render_score_explain(e: &anamnesis_search::RecordScoreExplain) -> serde_json::Value {
    let stages = match &e.best_chunk_stages {
        Some(s) => {
            let fts = s.fts.as_ref().map(|st| {
                serde_json::json!({
                    "rank": st.rank,
                    "raw_score": st.raw_score,
                    "rrf_contribution": st.rrf_contribution,
                })
            });
            let vector = s.vector.as_ref().map(|st| {
                serde_json::json!({
                    "rank": st.rank,
                    "raw_score": st.raw_score,
                    "rrf_contribution": st.rrf_contribution,
                })
            });
            serde_json::json!({
                "fts": fts,
                "vector": vector,
                "rrf_k": s.rrf_k,
            })
        }
        None => serde_json::Value::Null,
    };
    serde_json::json!({
        "record_score": e.record_score,
        "best_chunk_rrf_score": e.best_chunk_rrf_score,
        "kind_boost": e.kind_boost,
        "stages": stages,
    })
}

/// Render the search trace as a compact block under the human
/// results card. Same numeric shape as the JSON payload (and the
/// MCP `search_memories(trace=true)` wire shape from R71).
fn print_human_search_trace(
    t: &anamnesis_search::SearchTrace,
    pack_ms: u64,
    returned_records: u32,
) {
    let fmt = |v: Option<u64>| v.map(|n| n.to_string()).unwrap_or_else(|| "-".into());
    println!("Search trace:");
    println!(
        "  effective_mode={mode:?} candidate_pool={pool} returned_records={ret}",
        mode = t.effective_mode,
        pool = t.candidate_pool,
        ret = returned_records,
    );
    println!(
        "  stages_ms embed_query={eq} fts={fts} vec={vec} rrf={rrf} pack={pack}",
        eq = fmt(t.stages_ms.embed_query_ms),
        fts = fmt(t.stages_ms.fts_ms),
        vec = fmt(t.stages_ms.vec_ms),
        rrf = fmt(t.stages_ms.rrf_ms),
        pack = pack_ms,
    );
    println!(
        "  counts fts_hits={fh} vec_hits={vh} ranked_chunks={rc}",
        fh = t.counts.fts_hits,
        vh = t.counts.vec_hits,
        rc = t.counts.ranked_chunks,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// conflicts (Round 135 PR-78bd)
// ─────────────────────────────────────────────────────────────────────────────

/// `anamnesis conflicts` — list groups of records that share the
/// same `provenance.native_id` across adapters but disagree on
/// `content`. Distinct from `dedupe`: dedupe surfaces "same memory
/// captured twice" (raw_hash or near-dup); conflicts surface
/// "same identity, different content".
///
/// Read-only. Default output is redacted — `content_preview` is
/// only attached when the operator passes `--include-content`.
fn cmd_conflicts(
    data_dir: &std::path::Path,
    source: Option<&str>,
    instance: Option<&str>,
    limit: u32,
    json: bool,
    include_content: bool,
) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let filter = anamnesis_store::NativeConflictFilter {
        source: source.map(str::to_owned),
        instance: instance.map(str::to_owned),
        limit,
        include_content,
    };
    let groups = store.list_native_content_conflicts_filtered(&filter)?;
    let effective_limit = limit.clamp(1, anamnesis_store::LIST_NATIVE_CONFLICTS_MAX_LIMIT);

    if json {
        let source_tokens = anamnesis_core::parse_csv_filter(source);
        let instance_tokens = anamnesis_core::parse_csv_filter(instance);
        let source_clause = if source_tokens.is_empty() {
            "source filter: all sources".to_string()
        } else {
            format!("source filter: {}", source_tokens.join(" OR "))
        };
        let instance_clause = if instance_tokens.is_empty() {
            "instance filter: all instances".to_string()
        } else {
            format!("instance filter: {}", instance_tokens.join(" OR "))
        };
        let summary = format!(
            "{} cross-adapter `native_id` content conflict group(s) returned; limit {}; {}; {}; content_preview: {}.",
            groups.len(),
            effective_limit,
            source_clause,
            instance_clause,
            if include_content { "included" } else { "redacted" },
        );

        let payload = serde_json::json!({
            "summary":            summary,
            "format":             "json",
            "count":              groups.len(),
            "limit":              effective_limit,
            "content_included":   include_content,
            "filter": {
                "source":   source,
                "instance": instance,
            },
            "groups": groups.iter().map(|g| serde_json::json!({
                "native_id":             g.native_id,
                "record_count":          g.records.len(),
                "content_variant_count": g.content_variant_count,
                "records": g.records.iter().map(|r| {
                    let mut row = serde_json::json!({
                        "record_id":       r.record_id.0,
                        "adapter":         r.adapter,
                        "instance":        if r.instance.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(r.instance.clone()) },
                        "native_id":       r.native_id,
                        "created_at":      r.created_at,
                        "updated_at":      r.updated_at,
                        "has_native_path": r.has_native_path,
                        "content_variant": r.content_variant,
                    });
                    if let Some(prev) = &r.content_preview {
                        row["content_preview"] = serde_json::json!(prev);
                    }
                    row
                }).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    if groups.is_empty() {
        let scope = filter_label(source, instance);
        if scope.is_empty() {
            println!("no cross-adapter `native_id` content conflicts");
        } else {
            println!("no cross-adapter `native_id` content conflicts (filter: {scope})");
        }
        return Ok(());
    }
    let scope = filter_label(source, instance);
    let scope_suffix = if scope.is_empty() {
        String::new()
    } else {
        format!(" (filter: {scope})")
    };
    println!(
        "{} conflict group(s){} (cross-adapter `native_id` disagreement)",
        groups.len(),
        scope_suffix,
    );
    for (idx, g) in groups.iter().enumerate() {
        println!(
            "[{rank}] native_id={nid}  variants={vc}  record_count={n}",
            rank = idx + 1,
            nid = g.native_id,
            vc = g.content_variant_count,
            n = g.records.len(),
        );
        for r in &g.records {
            let inst = if r.instance.is_empty() {
                String::new()
            } else {
                format!(":{}", r.instance)
            };
            println!(
                "    variant {v}: {} ({}{inst}, created_at={})",
                r.record_id.0,
                r.adapter,
                r.created_at,
                v = r.content_variant,
            );
            if let Some(prev) = &r.content_preview {
                println!("       preview: {prev}");
            }
        }
        println!();
    }
    println!(
        "Resolve a conflict by picking which variant to keep, then `anamnesis forget <record_id>` the others."
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// forget (Round 72 PR-72a)
// ─────────────────────────────────────────────────────────────────────────────

/// `anamnesis forget <record_id>` — write a tombstone + clear the
/// live record. Future imports of the same `(adapter, instance,
/// native_id)` triple are suppressed at the store layer.
///
/// Exit codes:
///   - 0 on `Forgotten` or `AlreadyForgotten` (idempotent success).
///   - non-zero on `NotFound` so a typo in scripted usage is loud.
fn cmd_forget(
    data_dir: &std::path::Path,
    record_id: &str,
    reason: Option<&str>,
    json: bool,
    dry_run: bool,
    cascade_derived: bool,
) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let id = anamnesis_core::model::RecordId(record_id.to_string());
    if dry_run {
        return cmd_forget_dry_run(&store, record_id, &id, reason, json, cascade_derived);
    }
    let opts = anamnesis_store::ForgetCascadeOptions { cascade_derived };
    let cascade_outcome = store.forget_record_with_options(&id, reason, &opts)?;
    let outcome = cascade_outcome.root;
    let derived = cascade_outcome.derived;

    // §-1.5 PR-6 audit: every state-mutating CLI action lands in
    // the stage-2 audit log so `anamnesis audit` can reconstruct
    // who-forgot-what-when. Round 133 adds `cascade_derived` +
    // the derived record ids so the audit chain captures the full
    // blast radius (not just the named root).
    let mut audit_detail = serde_json::json!({
        "record_id": record_id,
        "reason": reason,
        "outcome": match &outcome {
            anamnesis_store::ForgetRecordOutcome::Forgotten(_) => "forgotten",
            anamnesis_store::ForgetRecordOutcome::AlreadyForgotten(_) => "already-forgotten",
            anamnesis_store::ForgetRecordOutcome::NotFound => "not-found",
        },
    });
    if cascade_derived {
        audit_detail["cascade_derived"] = serde_json::json!(true);
        audit_detail["derived_record_ids"] = serde_json::json!(derived
            .iter()
            .map(|d| d.record_id.0.clone())
            .collect::<Vec<_>>());
    }
    audit(data_dir).record(anamnesis_core::AuditEntry::new("forget", audit_detail));

    if json {
        let mut payload = match &outcome {
            anamnesis_store::ForgetRecordOutcome::Forgotten(r)
            | anamnesis_store::ForgetRecordOutcome::AlreadyForgotten(r) => serde_json::json!({
                "status": match outcome {
                    anamnesis_store::ForgetRecordOutcome::Forgotten(_) => "forgotten",
                    _ => "already-forgotten",
                },
                "record_id": r.record_id.0,
                "adapter": r.adapter,
                "instance": if r.instance.is_empty() { None } else { Some(&r.instance) },
                "native_id": r.native_id,
                "native_path": r.native_path,
                "reason": r.reason,
                "forgotten_at": r.forgotten_at,
            }),
            anamnesis_store::ForgetRecordOutcome::NotFound => serde_json::json!({
                "status": "not-found",
                "record_id": record_id,
            }),
        };
        if cascade_derived {
            // R133: always emit `cascade` when the flag was set,
            // even with `derived: []` so a script can distinguish
            // "I asked for cascade and there were no derivations"
            // from "cascade was never asked."
            payload["cascade"] = render_forget_cascade_json(&derived);
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        match &outcome {
            anamnesis_store::ForgetRecordOutcome::Forgotten(r) => {
                let inst = if r.instance.is_empty() {
                    String::new()
                } else {
                    format!(":{}", r.instance)
                };
                println!(
                    "forgotten {} (adapter={}{inst}, native_id={})",
                    r.record_id.0, r.adapter, r.native_id
                );
                if let Some(reason) = &r.reason {
                    println!("  reason: {reason}");
                }
            }
            anamnesis_store::ForgetRecordOutcome::AlreadyForgotten(r) => {
                println!(
                    "already forgotten at {}: {} (adapter={}, native_id={})",
                    r.forgotten_at, r.record_id.0, r.adapter, r.native_id,
                );
            }
            anamnesis_store::ForgetRecordOutcome::NotFound => {}
        }
        if cascade_derived {
            print_forget_cascade_human(&derived);
        }
    }

    if matches!(outcome, anamnesis_store::ForgetRecordOutcome::NotFound) {
        return Err(anyhow!(
            "no record with id {record_id:?} — nothing to forget"
        ));
    }
    Ok(())
}

/// Round 133 (PR-78bb): render the derived-records block of a
/// cascade forget as the `cascade` JSON object on `forget --json`.
/// `derived_count` is the cardinality the audit log mirrors; the
/// per-row shape carries adapter/instance/native_id so an operator
/// can confirm "yes, those were the extractor-derived facts."
fn render_forget_cascade_json(
    derived: &[anamnesis_store::DerivedForgetRecord],
) -> serde_json::Value {
    let derived_records: Vec<serde_json::Value> = derived
        .iter()
        .map(|d| {
            serde_json::json!({
                "record_id":            d.record_id.0,
                "adapter":              d.adapter,
                "instance":             if d.instance.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(d.instance.clone())
                },
                "native_id":            d.native_id,
                "forgotten_at":         d.forgotten_at,
                "was_already_forgotten": d.was_already_forgotten,
            })
        })
        .collect();
    serde_json::json!({
        "derived_count":      derived.len(),
        "derived_records":    derived_records,
    })
}

/// R133: human one-liner per derived record beneath the root
/// summary. Mirrors the structured `cascade` JSON shape so an
/// operator gets the same information without `--json`.
fn print_forget_cascade_human(derived: &[anamnesis_store::DerivedForgetRecord]) {
    if derived.is_empty() {
        println!("  cascade-derived: no descendants");
        return;
    }
    println!("  cascade-derived: forgot {} descendant(s)", derived.len());
    for d in derived {
        let inst = if d.instance.is_empty() {
            String::new()
        } else {
            format!(":{}", d.instance)
        };
        let state = if d.was_already_forgotten {
            " (was already forgotten)"
        } else {
            ""
        };
        println!(
            "    {} ({}{inst}, native_id={}){state}",
            d.record_id.0, d.adapter, d.native_id
        );
    }
}

/// Round 83 (PR-78e): `anamnesis forget --dry-run` — preview the
/// cascade without writing anything. Does NOT call
/// `store.forget_record` and does NOT append to `audit.log`. Same
/// exit-code policy as the real path: `NotFound` is loud.
fn cmd_forget_dry_run(
    store: &Store,
    record_id: &str,
    id: &anamnesis_core::model::RecordId,
    reason: Option<&str>,
    json: bool,
    cascade_derived: bool,
) -> Result<()> {
    let opts = anamnesis_store::ForgetCascadeOptions { cascade_derived };
    let cascade_preview = store.preview_forget_record_with_options(id, reason, &opts)?;
    let preview = cascade_preview.root;
    let derived = cascade_preview.derived;

    if json {
        let mut payload = match &preview {
            anamnesis_store::ForgetRecordPreview::WouldForget {
                would_delete,
                tombstone_preview,
            } => serde_json::json!({
                "dry_run": true,
                "status": "would-forget",
                "record_id": tombstone_preview.record_id.0,
                "adapter": tombstone_preview.adapter,
                "instance": if tombstone_preview.instance.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(tombstone_preview.instance.clone())
                },
                "native_id": tombstone_preview.native_id,
                "native_path": tombstone_preview.native_path,
                "raw_hash": tombstone_preview.raw_hash,
                "reason": tombstone_preview.reason,
                "would_delete": {
                    "records": would_delete.records,
                    "raw_artifacts": would_delete.raw_artifacts,
                    "record_chunks": would_delete.record_chunks,
                    "chunk_embeddings": would_delete.chunk_embeddings,
                    "embedding_jobs": would_delete.embedding_jobs,
                    "user_record_tags": would_delete.user_record_tags,
                    "vec0_rows": would_delete.vec0_rows,
                },
                "would_insert": {
                    "record_tombstones": 1,
                    // 1 audit entry: this dry-run does NOT write
                    // one, but the real forget would.
                    "audit_log_entries": 1,
                },
            }),
            anamnesis_store::ForgetRecordPreview::AlreadyForgotten(r) => serde_json::json!({
                "dry_run": true,
                "status": "already-forgotten",
                "record_id": r.record_id.0,
                "adapter": r.adapter,
                "instance": if r.instance.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(r.instance.clone())
                },
                "native_id": r.native_id,
                "native_path": r.native_path,
                "raw_hash": r.raw_hash,
                "reason": r.reason,
                "forgotten_at": r.forgotten_at,
                "would_delete": {
                    "records": 0u64,
                    "raw_artifacts": 0u64,
                    "record_chunks": 0u64,
                    "chunk_embeddings": 0u64,
                    "embedding_jobs": 0u64,
                    "user_record_tags": 0u64,
                    "vec0_rows": 0u64,
                },
                "would_insert": {
                    "record_tombstones": 0,
                    "audit_log_entries": 0,
                },
            }),
            anamnesis_store::ForgetRecordPreview::NotFound => serde_json::json!({
                "dry_run": true,
                "status": "not-found",
                "record_id": record_id,
            }),
        };
        if cascade_derived {
            // Mirror the real-run cascade JSON shape on dry-run so a
            // script can preview the full blast radius before
            // committing. `would_*` counts inside per-derived rows
            // keep the preview self-describing.
            payload["cascade"] = render_forget_cascade_preview_json(&derived);
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        match &preview {
            anamnesis_store::ForgetRecordPreview::WouldForget {
                would_delete,
                tombstone_preview,
            } => {
                let inst = if tombstone_preview.instance.is_empty() {
                    String::new()
                } else {
                    format!(":{}", tombstone_preview.instance)
                };
                println!(
                    "DRY-RUN — would forget {} (adapter={}{inst}, native_id={})",
                    tombstone_preview.record_id.0,
                    tombstone_preview.adapter,
                    tombstone_preview.native_id
                );
                println!(
                    "  would delete: {} record, {} chunks, {} embeddings, {} vec0 rows, \
                     {} embedding jobs, {} user tags, {} raw artifacts",
                    would_delete.records,
                    would_delete.record_chunks,
                    would_delete.chunk_embeddings,
                    would_delete.vec0_rows,
                    would_delete.embedding_jobs,
                    would_delete.user_record_tags,
                    would_delete.raw_artifacts,
                );
                println!("  would write: 1 tombstone, 1 audit entry");
                if let Some(r) = &tombstone_preview.reason {
                    println!("  reason: {r}");
                }
                println!("  (no changes applied — re-run without --dry-run to commit)");
            }
            anamnesis_store::ForgetRecordPreview::AlreadyForgotten(r) => {
                println!(
                    "DRY-RUN — already forgotten at {}: {} (adapter={}, native_id={})",
                    r.forgotten_at, r.record_id.0, r.adapter, r.native_id,
                );
                println!("  no changes would be made (re-run without --dry-run is a no-op too)");
            }
            anamnesis_store::ForgetRecordPreview::NotFound => {}
        }
        if cascade_derived {
            print_forget_cascade_preview_human(&derived);
        }
    }

    if matches!(preview, anamnesis_store::ForgetRecordPreview::NotFound) {
        return Err(anyhow!(
            "no record with id {record_id:?} — nothing to forget (dry-run)"
        ));
    }
    Ok(())
}

/// R133: render the cascade preview block on `forget --dry-run
/// --cascade-derived --json`. Each per-row entry carries
/// `would_delete` (the same per-table count shape as the root) and
/// `already_forgotten_at` (`null` = cascade would write a fresh
/// tombstone, integer = tombstone already exists).
fn render_forget_cascade_preview_json(
    derived: &[anamnesis_store::DerivedForgetPreview],
) -> serde_json::Value {
    let derived_records: Vec<serde_json::Value> = derived
        .iter()
        .map(|d| {
            serde_json::json!({
                "record_id":             d.record_id.0,
                "adapter":               d.adapter,
                "instance":              if d.instance.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(d.instance.clone())
                },
                "native_id":             d.native_id,
                "would_delete": {
                    "records":          d.would_delete.records,
                    "raw_artifacts":    d.would_delete.raw_artifacts,
                    "record_chunks":    d.would_delete.record_chunks,
                    "chunk_embeddings": d.would_delete.chunk_embeddings,
                    "embedding_jobs":   d.would_delete.embedding_jobs,
                    "user_record_tags": d.would_delete.user_record_tags,
                    "vec0_rows":        d.would_delete.vec0_rows,
                },
                "already_forgotten_at": d.already_forgotten_at,
            })
        })
        .collect();
    serde_json::json!({
        "derived_count":   derived.len(),
        "derived_records": derived_records,
    })
}

/// R133: human cascade preview lines under the root summary.
fn print_forget_cascade_preview_human(derived: &[anamnesis_store::DerivedForgetPreview]) {
    if derived.is_empty() {
        println!("  cascade-derived (DRY-RUN): no descendants");
        return;
    }
    println!(
        "  cascade-derived (DRY-RUN): would touch {} descendant(s)",
        derived.len()
    );
    for d in derived {
        let inst = if d.instance.is_empty() {
            String::new()
        } else {
            format!(":{}", d.instance)
        };
        match d.already_forgotten_at {
            Some(ts) => println!(
                "    {} ({}{inst}, native_id={}) — already forgotten at {ts}",
                d.record_id.0, d.adapter, d.native_id
            ),
            None => println!(
                "    {} ({}{inst}, native_id={}) — would write tombstone (cascade-delete {} chunks, {} vec0 rows)",
                d.record_id.0,
                d.adapter,
                d.native_id,
                d.would_delete.record_chunks,
                d.would_delete.vec0_rows,
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// unforget (Round 75 PR-75)
// ─────────────────────────────────────────────────────────────────────────────

/// `anamnesis unforget <record_id>` — remove a tombstone so a future
/// import is allowed to bring the record back.
///
/// Critically: does NOT recreate the live record. The tombstone
/// only stored provenance, not content, so resurrecting from it
/// would let `unforget` synthesise data — Anamnesis is a read-only
/// mirror of source data. The truthful design is "remove the gate;
/// the source itself decides whether to re-emit on next import."
///
/// Exit codes:
///   - 0 on `Unforgotten` (tombstone removed).
///   - non-zero on `NotForgotten` so a paste mistake from
///     `list-forgotten` is loud rather than silently a no-op.
fn cmd_unforget(
    data_dir: &std::path::Path,
    record_id: &str,
    json: bool,
    dry_run: bool,
    cascade_derived: bool,
) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let id = anamnesis_core::model::RecordId(record_id.to_string());
    if dry_run {
        return cmd_unforget_dry_run(&store, record_id, &id, json, cascade_derived);
    }
    let opts = anamnesis_store::UnforgetCascadeOptions { cascade_derived };
    let cascade_outcome = store.unforget_record_with_options(&id, &opts)?;
    let outcome = cascade_outcome.root;
    let derived = cascade_outcome.derived;

    let mut audit_detail = serde_json::json!({
        "record_id": record_id,
        "outcome": match &outcome {
            anamnesis_store::UnforgetRecordOutcome::Unforgotten(_) => "unforgotten",
            anamnesis_store::UnforgetRecordOutcome::NotForgotten   => "not-forgotten",
        },
    });
    if cascade_derived {
        audit_detail["cascade_derived"] = serde_json::json!(true);
        audit_detail["derived_record_ids"] = serde_json::json!(derived
            .iter()
            .map(|d| d.record_id.0.clone())
            .collect::<Vec<_>>());
    }
    audit(data_dir).record(anamnesis_core::AuditEntry::new("unforget", audit_detail));

    if json {
        let mut payload = match &outcome {
            anamnesis_store::UnforgetRecordOutcome::Unforgotten(r) => serde_json::json!({
                "status": "unforgotten",
                "record_id": r.record_id.0,
                "adapter": r.adapter,
                "instance": if r.instance.is_empty() { None } else { Some(&r.instance) },
                "native_id": r.native_id,
                "forgotten_at": r.forgotten_at,
                "record_resurrected": false,
                "requires_reimport": true,
            }),
            anamnesis_store::UnforgetRecordOutcome::NotForgotten => serde_json::json!({
                "status": "not-forgotten",
                "record_id": record_id,
            }),
        };
        if cascade_derived {
            payload["cascade"] = render_unforget_cascade_json(&derived);
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        if let anamnesis_store::UnforgetRecordOutcome::Unforgotten(r) = &outcome {
            let inst = if r.instance.is_empty() {
                String::new()
            } else {
                format!(":{}", r.instance)
            };
            println!(
                "unforgotten {} (adapter={}{inst}, native_id={})",
                r.record_id.0, r.adapter, r.native_id
            );
            println!(
                "  tombstone removed — record itself is NOT resurrected; \
                 re-import the source to bring it back."
            );
        }
        if cascade_derived {
            print_unforget_cascade_human(&derived);
        }
    }

    if matches!(
        outcome,
        anamnesis_store::UnforgetRecordOutcome::NotForgotten
    ) {
        return Err(anyhow!(
            "no tombstone for id {record_id:?} — nothing to unforget"
        ));
    }
    Ok(())
}

/// Round 134 (PR-78bc): render the cascade block on `unforget --json`.
/// Mirrors the R133 `forget` cascade renderer — `derived_count` +
/// per-row record snapshot (record_id / adapter / instance /
/// native_id / forgotten_at). Pre-R134 tombstones with NULL
/// `derived_from` produce an empty list, but the empty `cascade`
/// block still distinguishes "I asked for cascade" from "I didn't."
fn render_unforget_cascade_json(
    derived: &[anamnesis_store::DerivedUnforgetRecord],
) -> serde_json::Value {
    let derived_records: Vec<serde_json::Value> = derived
        .iter()
        .map(|d| {
            serde_json::json!({
                "record_id":    d.record_id.0,
                "adapter":      d.adapter,
                "instance":     if d.instance.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(d.instance.clone())
                },
                "native_id":    d.native_id,
                "forgotten_at": d.forgotten_at,
            })
        })
        .collect();
    serde_json::json!({
        "derived_count":   derived.len(),
        "derived_records": derived_records,
    })
}

/// R134: human one-liner per descendant tombstone beneath the root
/// summary. Mirrors the structured cascade JSON shape so operators
/// get the same information without `--json`.
fn print_unforget_cascade_human(derived: &[anamnesis_store::DerivedUnforgetRecord]) {
    if derived.is_empty() {
        println!("  cascade-derived: no descendant tombstones");
        return;
    }
    println!(
        "  cascade-derived: removed {} descendant tombstone(s)",
        derived.len()
    );
    for d in derived {
        let inst = if d.instance.is_empty() {
            String::new()
        } else {
            format!(":{}", d.instance)
        };
        println!(
            "    {} ({}{inst}, native_id={})",
            d.record_id.0, d.adapter, d.native_id
        );
    }
}

/// Round 95 (PR-78q): `anamnesis unforget --dry-run` — preview
/// the tombstone the real `unforget` would remove. Does NOT
/// call `store.unforget_record` and does NOT append to
/// `audit.log`. Same exit-code policy as the real path:
/// missing tombstone exits non-zero so a typo'd id stays loud.
fn cmd_unforget_dry_run(
    store: &Store,
    record_id: &str,
    id: &anamnesis_core::model::RecordId,
    json: bool,
    cascade_derived: bool,
) -> Result<()> {
    let opts = anamnesis_store::UnforgetCascadeOptions { cascade_derived };
    let cascade_preview = store.preview_unforget_record_with_options(id, &opts)?;
    let preview = cascade_preview.root;
    let derived = cascade_preview.derived;
    if json {
        let mut payload = match &preview {
            anamnesis_store::UnforgetRecordOutcome::Unforgotten(r) => serde_json::json!({
                "dry_run": true,
                "status": "would-unforget",
                "record_id": r.record_id.0,
                "adapter": r.adapter,
                "instance": if r.instance.is_empty() { None } else { Some(&r.instance) },
                "native_id": r.native_id,
                "forgotten_at": r.forgotten_at,
                "record_resurrected": false,
                "requires_reimport": true,
                "would_delete": { "record_tombstones": 1 },
                "would_insert": { "audit_log_entries": 1 },
            }),
            anamnesis_store::UnforgetRecordOutcome::NotForgotten => serde_json::json!({
                "dry_run": true,
                "status": "not-forgotten",
                "record_id": record_id,
            }),
        };
        if cascade_derived {
            payload["cascade"] = render_unforget_cascade_preview_json(&derived);
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        if let anamnesis_store::UnforgetRecordOutcome::Unforgotten(r) = &preview {
            let inst = if r.instance.is_empty() {
                String::new()
            } else {
                format!(":{}", r.instance)
            };
            println!(
                "DRY-RUN — would unforget {} (adapter={}{inst}, native_id={})",
                r.record_id.0, r.adapter, r.native_id
            );
            println!("  would write: 1 audit entry, delete 1 tombstone");
            println!(
                "  (the record itself is NOT resurrected; re-import the source to bring it back)"
            );
        }
        if cascade_derived {
            print_unforget_cascade_preview_human(&derived);
        }
    }

    if matches!(
        preview,
        anamnesis_store::UnforgetRecordOutcome::NotForgotten
    ) {
        return Err(anyhow!(
            "no tombstone for id {record_id:?} — nothing to unforget (dry-run)"
        ));
    }
    Ok(())
}

/// R134: render the cascade-preview block on `unforget --dry-run
/// --cascade-derived --json`. Same per-row shape as the post-commit
/// renderer; no `would_delete` counts here (descendants are one-row
/// tombstone DELETEs each).
fn render_unforget_cascade_preview_json(
    derived: &[anamnesis_store::DerivedUnforgetPreview],
) -> serde_json::Value {
    let derived_records: Vec<serde_json::Value> = derived
        .iter()
        .map(|d| {
            serde_json::json!({
                "record_id":    d.record_id.0,
                "adapter":      d.adapter,
                "instance":     if d.instance.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(d.instance.clone())
                },
                "native_id":    d.native_id,
                "forgotten_at": d.forgotten_at,
            })
        })
        .collect();
    serde_json::json!({
        "derived_count":   derived.len(),
        "derived_records": derived_records,
    })
}

/// R134: human cascade-preview lines under the root summary.
fn print_unforget_cascade_preview_human(derived: &[anamnesis_store::DerivedUnforgetPreview]) {
    if derived.is_empty() {
        println!("  cascade-derived (DRY-RUN): no descendant tombstones");
        return;
    }
    println!(
        "  cascade-derived (DRY-RUN): would remove {} descendant tombstone(s)",
        derived.len()
    );
    for d in derived {
        let inst = if d.instance.is_empty() {
            String::new()
        } else {
            format!(":{}", d.instance)
        };
        println!(
            "    {} ({}{inst}, native_id={}) — forgotten at {}",
            d.record_id.0, d.adapter, d.native_id, d.forgotten_at
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// tag-record (Round 78 PR-78)
// ─────────────────────────────────────────────────────────────────────────────

/// `anamnesis tag-record <record_id> <tag>...` — apply or remove
/// user tags on a record. Writes to the `user_record_tags`
/// overlay (R78), which is distinct from the adapter-derived
/// `records.tags`. Re-import will not overwrite the user tags.
///
/// Audit-logged so the §-1.5 stage-2 audit trail captures
/// who-tagged-what-when, parity with `forget` / `unforget`.
#[allow(clippy::too_many_arguments)]
fn cmd_tag_record(
    data_dir: &std::path::Path,
    record_id: &str,
    tags: &[String],
    remove: bool,
    replace: bool,
    json: bool,
    include_stats: bool,
) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let id = anamnesis_core::model::RecordId(record_id.to_string());
    // clap already enforces `replace` and `remove` are mutually
    // exclusive (`conflicts_with = "replace"` on `--remove`), so
    // the precedence here is deterministic.
    let op = if replace {
        anamnesis_store::UserTagOperation::Replace
    } else if remove {
        anamnesis_store::UserTagOperation::Remove
    } else {
        anamnesis_store::UserTagOperation::Add
    };
    let op_label = operation_label(op);
    let mutation = store.tag_record(&id, tags, op)?;

    audit(data_dir).record(anamnesis_core::AuditEntry::new(
        "tag_record",
        serde_json::json!({
            "record_id": record_id,
            "operation": op_label,
            "requested": mutation.requested,
            "changed": mutation.changed,
        }),
    ));

    if json {
        let mut payload = serde_json::json!({
            "record_id": mutation.record_id.0,
            "operation": op_label,
            "requested": mutation.requested,
            "changed": mutation.changed,
            "user_tags": mutation.user_tags,
        });
        if include_stats {
            payload["stats"] = serde_json::json!({
                "total_user_tags": mutation.user_tags.len(),
            });
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        let verb = match op {
            anamnesis_store::UserTagOperation::Add => "added",
            anamnesis_store::UserTagOperation::Remove => "removed",
            anamnesis_store::UserTagOperation::Replace => "replaced to",
        };
        let req = if mutation.requested.is_empty() {
            "(empty set — cleared)".to_string()
        } else {
            mutation.requested.join(", ")
        };
        println!(
            "{verb} {n} tag(s) on {id} (requested: {req})",
            n = mutation.changed,
            id = mutation.record_id.0,
        );
        if mutation.user_tags.is_empty() {
            println!("  user_tags: (none)");
        } else {
            println!("  user_tags: {}", mutation.user_tags.join(", "));
        }
        if include_stats {
            println!("  stats: total_user_tags={}", mutation.user_tags.len());
        }
    }
    Ok(())
}

/// Stable string label for a `UserTagOperation`. Shared between
/// the CLI's JSON payload and audit log so the wire shape matches
/// the MCP surface exactly.
fn operation_label(op: anamnesis_store::UserTagOperation) -> &'static str {
    match op {
        anamnesis_store::UserTagOperation::Add => "add",
        anamnesis_store::UserTagOperation::Remove => "remove",
        anamnesis_store::UserTagOperation::Replace => "replace",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// dedupe (Round 77 PR-77, Round 80 source/instance filter)
// ─────────────────────────────────────────────────────────────────────────────

/// Compact label for the dedupe filter, shared by the human
/// "no groups" / "N groups" headers so the operator can see at
/// a glance whether the empty result is "nothing's duplicated"
/// or "your filter knocked everything out."
fn filter_label(source: Option<&str>, instance: Option<&str>) -> String {
    match (source, instance) {
        (Some(s), Some(i)) => format!("source={s}, instance={i}"),
        (Some(s), None) => format!("source={s}"),
        (None, Some(i)) => format!("instance={i}"),
        (None, None) => String::new(),
    }
}

/// `anamnesis dedupe` — read-only exact-duplicate report keyed on
/// `records.raw_hash`. Groups records with identical source payload
/// bytes and prints them newest-first within each group so the
/// operator can decide which sibling to `forget`.
///
/// Privacy: default output redacts `raw_hash` and `native_path`
/// (only `has_*` booleans in JSON). `--include-sensitive` opts in
/// to the full fields.
///
/// Read-only — no audit log, no store writes. Composes with
/// `anamnesis forget <record_id>` for the action half.
#[allow(clippy::too_many_arguments)]
fn cmd_dedupe(
    data_dir: &std::path::Path,
    mode: DedupeMode,
    source: Option<&str>,
    instance: Option<&str>,
    limit: u32,
    json: bool,
    include_sensitive: bool,
    include_counts: bool,
    csv: bool,
    include_near_self: bool,
    merge_preview: bool,
) -> Result<()> {
    // Round 141 (PR-78bj): `--merge-preview` is near-only and
    // JSON/human-only. Refuse `exact` (no per-group ranking to
    // propose) and refuse `csv` (the nested decision draft
    // doesn't flatten safely into a flat row format).
    if merge_preview && matches!(mode, DedupeMode::Exact) {
        return Err(anyhow!(
            "--merge-preview requires --mode near: the exact path has no per-group ranking to propose."
        ));
    }
    if merge_preview && csv {
        return Err(anyhow!(
            "--merge-preview and --csv are mutually exclusive — the per-group ranking draft is a nested object that doesn't flatten safely. Use --merge-preview --json or omit --csv."
        ));
    }
    // Round 132 (PR-78ba): near mode is privacy-safe by
    // construction — it never touches raw_hash or native_path —
    // so `--include-sensitive` and `--include-counts` have
    // nothing meaningful to add. Refuse them loudly so an
    // operator notices the mismatch instead of seeing a
    // silently-ignored flag.
    if matches!(mode, DedupeMode::Near) && include_sensitive {
        return Err(anyhow!(
            "--mode near and --include-sensitive are mutually exclusive — near-dedupe never reads `raw_hash` / `native_path`, so there is nothing sensitive to reveal."
        ));
    }
    if matches!(mode, DedupeMode::Near) && include_counts {
        return Err(anyhow!(
            "--mode near and --include-counts are mutually exclusive — the `counts` aggregate is exact-dedupe specific. Drop --include-counts (group cardinality is already on each near group)."
        ));
    }
    if matches!(mode, DedupeMode::Exact) && include_near_self {
        return Err(anyhow!(
            "--include-near-self only applies to --mode near (it opts out of the cross-source filter that is unique to near-dedupe)."
        ));
    }

    // Round 107 (PR-78ac): runtime guards for the CSV
    // conflicts clap can't carry without flirting with the
    // Windows stack-overflow we fixed in R106. `--csv --json`
    // is clap-rejected; the other two combinations are checked
    // here. CSV is the redacted-summary form — sensitive
    // smuggling or nested counts have no place in flat rows.
    if csv && include_sensitive {
        return Err(anyhow!(
            "--csv and --include-sensitive are mutually exclusive — CSV is the redacted-summary form (never carries `raw_hash` / `native_path`)."
        ));
    }
    if csv && include_counts {
        return Err(anyhow!(
            "--csv and --include-counts are mutually exclusive — CSV is flat redacted rows. Drop --csv to get the counts block back."
        ));
    }

    if matches!(mode, DedupeMode::Near) {
        // Defer to the dedicated near-dedupe path — it doesn't
        // share the raw_hash projection so we keep the branches
        // separate rather than papering over with `if` ladders.
        return cmd_dedupe_near(
            data_dir,
            source,
            instance,
            limit,
            json,
            csv,
            include_near_self,
            merge_preview,
        );
    }

    let store = Store::open(db_path(data_dir))?;
    let filter = anamnesis_store::DuplicateRawHashFilter {
        source: source.map(str::to_owned),
        instance: instance.map(str::to_owned),
        limit,
    };
    let groups = store.list_duplicate_raw_hashes_filtered(&filter)?;
    let effective_limit = limit.clamp(1, anamnesis_store::LIST_DUPLICATE_RAW_HASHES_MAX_LIMIT);
    // Round 97: filter-scoped aggregate. Counts reflect the
    // full matching set; `limit` only affects `groups[]`.
    let counts = if include_counts {
        Some(store.count_duplicate_raw_hashes_by_source(&filter)?)
    } else {
        None
    };

    if csv {
        // Round 107 (PR-78ac): CSV is the redacted-summary
        // form, mirroring R91 `audit tail --csv` + R106
        // `list-forgotten --csv`. Header is fixed so scripts
        // can branch on column count; empty result still
        // prints the header. `group_index` carries duplicate-
        // group membership without leaking `raw_hash`: rows
        // sharing the same index belong to the same group.
        // `record_count` is per-group size, repeated on each
        // row for spreadsheet-friendly downstream filtering.
        println!(
            "group_index,record_id,adapter,instance,native_id,created_at,updated_at,has_native_path,record_count"
        );
        for (gi, g) in groups.iter().enumerate() {
            let record_count = g.records.len();
            for r in &g.records {
                let created_iso = chrono::DateTime::<chrono::Utc>::from_timestamp(r.created_at, 0)
                    .map(|d| d.format("%Y-%m-%dT%H:%M:%SZ").to_string())
                    .unwrap_or_else(|| r.created_at.to_string());
                let updated_iso = match r.updated_at {
                    Some(t) => chrono::DateTime::<chrono::Utc>::from_timestamp(t, 0)
                        .map(|d| d.format("%Y-%m-%dT%H:%M:%SZ").to_string())
                        .unwrap_or_else(|| t.to_string()),
                    None => String::new(),
                };
                println!(
                    "{group_index},{rid},{adapter},{instance},{native_id},{created},{updated},{has_native_path},{record_count}",
                    group_index = gi,
                    rid = csv_field(&r.record_id.0),
                    adapter = csv_field(&r.adapter),
                    instance = csv_field(&r.instance),
                    native_id = csv_field(&r.native_id),
                    created = csv_field(&created_iso),
                    updated = csv_field(&updated_iso),
                    has_native_path = r.native_path.is_some(),
                    record_count = record_count,
                );
            }
        }
        return Ok(());
    }

    if json {
        // Round 108 (PR-78ad): `"format": "json"` marker
        // mirrors R107's `"format": "csv"` on the MCP CSV
        // path. Lets a script that supports both shapes branch
        // on `payload.format` instead of probing for `csv` vs
        // `groups[]`. Position is alphabetical-ish next to
        // `count` so the structural keys cluster.
        //
        // Round 125 (PR-78at): top-level redacted summary
        // mirrors MCP `dedupe` R117 + CLI `source list --json`
        // R124 + CLI `status --json` R123. NEVER reads
        // `raw_hash` or `native_path` — only counts, filter
        // clauses, sensitive/counts state.
        let source_tokens = anamnesis_core::parse_csv_filter(source);
        let instance_tokens = anamnesis_core::parse_csv_filter(instance);
        let source_clause = if source_tokens.is_empty() {
            "source filter: all sources".to_string()
        } else {
            format!("source filter: {}", source_tokens.join(" OR "))
        };
        let instance_clause = if instance_tokens.is_empty() {
            "instance filter: all instances".to_string()
        } else {
            format!("instance filter: {}", instance_tokens.join(" OR "))
        };
        let summary = format!(
            "{} duplicate group(s) returned; limit {}; {}; {}; sensitive: {}; counts: {}.",
            groups.len(),
            effective_limit,
            source_clause,
            instance_clause,
            if include_sensitive {
                "included"
            } else {
                "redacted"
            },
            if include_counts {
                "included"
            } else {
                "omitted"
            },
        );

        let mut payload = serde_json::json!({
            "summary": summary,
            "count": groups.len(),
            "format": "json",
            // Round 132 (PR-78ba): wire-shape mode discriminator
            // pairs with the new `dedupe --mode near` branch so a
            // script can switch on `payload.mode` without inspecting
            // which fields are present. Always emitted; default
            // back-compat value is `"exact"`.
            "mode": mode.wire_label(),
            "limit": effective_limit,
            "sensitive_included": include_sensitive,
            "filter": {
                "source": source,
                "instance": instance,
            },
            "groups": groups.iter().map(|g| {
                let mut group = serde_json::json!({
                    "record_count": g.records.len(),
                    "records": g.records.iter().map(|r| {
                        let mut row = serde_json::json!({
                            "record_id": r.record_id.0,
                            "adapter": r.adapter,
                            "instance": if r.instance.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(r.instance.clone()) },
                            "native_id": r.native_id,
                            "created_at": r.created_at,
                            "updated_at": r.updated_at,
                            "has_native_path": r.native_path.is_some(),
                        });
                        if include_sensitive {
                            row["native_path"] = serde_json::json!(r.native_path);
                        }
                        row
                    }).collect::<Vec<_>>(),
                });
                if include_sensitive {
                    group["raw_hash"] = serde_json::json!(g.raw_hash);
                }
                group
            }).collect::<Vec<_>>(),
        });
        if let Some(c) = &counts {
            payload["counts"] = render_dedupe_counts_json(c);
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else if groups.is_empty() {
        let scope = filter_label(source, instance);
        if scope.is_empty() {
            println!("no duplicate raw_hash groups");
        } else {
            println!("no duplicate raw_hash groups (filter: {scope})");
        }
    } else {
        let scope = filter_label(source, instance);
        let scope_suffix = if scope.is_empty() {
            String::new()
        } else {
            format!(" (filter: {scope})")
        };
        println!(
            "{} duplicate raw_hash group(s){}{}",
            groups.len(),
            scope_suffix,
            if include_sensitive {
                " (including sensitive fields)"
            } else {
                ""
            }
        );
        for (idx, g) in groups.iter().enumerate() {
            let hash_label = if include_sensitive {
                format!(" raw_hash={}", g.raw_hash)
            } else {
                String::new()
            };
            println!(
                "[{rank}] {n} record(s){hash_label}",
                rank = idx + 1,
                n = g.records.len()
            );
            for r in &g.records {
                let inst = if r.instance.is_empty() {
                    String::new()
                } else {
                    format!(":{}", r.instance)
                };
                println!(
                    "    {} ({}{inst}, native_id={}, created_at={})",
                    r.record_id.0, r.adapter, r.native_id, r.created_at
                );
                if include_sensitive {
                    if let Some(p) = &r.native_path {
                        println!("       native_path: {p}");
                    }
                } else if r.native_path.is_some() {
                    println!("       (native_path hidden — pass --include-sensitive to reveal)");
                }
            }
            println!();
        }
        println!(
            "Pick one record per group to keep, then `anamnesis forget <record_id>` the others."
        );
        if let Some(c) = &counts {
            println!();
            println!("Duplicate totals (filter-scoped):");
            println!("  total_groups : {}", c.total_groups);
            println!("  total_records: {}", c.total_records);
            if !c.by_source.is_empty() {
                println!("  by_source:");
                for b in &c.by_source {
                    let inst = if b.instance.is_empty() {
                        "(default)".to_string()
                    } else {
                        b.instance.clone()
                    };
                    println!(
                        "    {adapter:<14} {inst:<14} {n}",
                        adapter = b.adapter,
                        inst = inst,
                        n = b.duplicate_record_count,
                    );
                }
            }
        }
    }
    Ok(())
}

/// Round 97 (PR-78s): render `count_duplicate_raw_hashes_by_source`
/// as the `counts` JSON block for `dedupe --include-counts`. The
/// `by_source` array counts records, not group memberships, so
/// mixed-source groups don't double-count. Default instance
/// serialises as JSON `null` matching the rest of the surface.
fn render_dedupe_counts_json(c: &anamnesis_store::DuplicateRawHashCounts) -> serde_json::Value {
    serde_json::json!({
        "total_groups": c.total_groups,
        "total_records": c.total_records,
        "by_source": c.by_source.iter().map(|b| serde_json::json!({
            "adapter": b.adapter,
            "instance": if b.instance.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(b.instance.clone())
            },
            "duplicate_record_count": b.duplicate_record_count,
        })).collect::<Vec<_>>(),
    })
}

/// Round 132 (PR-78ba): `anamnesis dedupe --mode near` — wraps the
/// R131 cross-source near-duplicate algorithm (SimHash + LSH +
/// Jaccard) in the CLI's standard redacted-summary discipline. No
/// `--include-sensitive` / `--include-counts` paths: the algorithm
/// never reads `raw_hash` / `native_path` and there is no
/// raw_hash-style aggregate to surface.
///
/// JSON wire shape:
/// ```text
/// {
///   "format": "json",
///   "mode": "near",
///   "summary": "<human discovery summary>",
///   "count": <groups returned>,
///   "limit": <clamped limit>,
///   "filter": {
///     "source": <raw input | null>,
///     "instance": <raw input | null>,
///     "require_cross_source": <bool>
///   },
///   "groups": [
///     {
///       "record_count": N,
///       "min_similarity": <f64 in [0.6, 1.0]>,
///       "max_distance": <u32 in [0, 8]>,
///       "records": [{
///         "record_id", "adapter", "instance" (null for default),
///         "native_id", "created_at", "updated_at",
///         "has_native_path"
///       }, ...]
///     }, ...
///   ]
/// }
/// ```
#[allow(clippy::too_many_arguments)]
fn cmd_dedupe_near(
    data_dir: &std::path::Path,
    source: Option<&str>,
    instance: Option<&str>,
    limit: u32,
    json: bool,
    csv: bool,
    include_near_self: bool,
    merge_preview: bool,
) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let filter = anamnesis_store::NearDuplicateFilter {
        source: source.map(str::to_owned),
        instance: instance.map(str::to_owned),
        require_cross_source: !include_near_self,
        limit,
    };
    let groups = anamnesis_store::list_near_duplicates(&store, &filter)?;
    let effective_limit = limit.clamp(1, anamnesis_store::NEAR_DEDUPE_MAX_LIMIT);

    // Round 141 (PR-78bj): batch-fetch user-tag counts for every
    // record across every group. One round-trip instead of N, and
    // the count map is what `build_merge_preview` consumes. Always
    // computed (cheap) so the human path can also render the
    // ranking; the JSON path attaches a `merge_preview` block per
    // group only when the operator explicitly asked.
    let all_ids: Vec<anamnesis_core::model::RecordId> = groups
        .iter()
        .flat_map(|g| g.records.iter().map(|r| r.record_id.clone()))
        .collect();
    let tags_map = if all_ids.is_empty() {
        std::collections::HashMap::new()
    } else {
        store.user_tags_by_ids(&all_ids)?
    };
    let tag_counts: std::collections::HashMap<String, u32> = tags_map
        .into_iter()
        .map(|(id, tags)| (id.0, tags.len() as u32))
        .collect();

    if csv {
        // R107-style flat CSV. Header carries the per-group
        // similarity stats so a script doesn't need a second
        // round-trip for ranking. `group_index` (not raw_hash —
        // near-dedupe has none) keeps membership recoverable.
        println!(
            "group_index,record_id,adapter,instance,native_id,created_at,updated_at,has_native_path,record_count,min_similarity,max_distance"
        );
        for (gi, g) in groups.iter().enumerate() {
            let record_count = g.records.len();
            for r in &g.records {
                let created_iso = chrono::DateTime::<chrono::Utc>::from_timestamp(r.created_at, 0)
                    .map(|d| d.format("%Y-%m-%dT%H:%M:%SZ").to_string())
                    .unwrap_or_else(|| r.created_at.to_string());
                let updated_iso = match r.updated_at {
                    Some(t) => chrono::DateTime::<chrono::Utc>::from_timestamp(t, 0)
                        .map(|d| d.format("%Y-%m-%dT%H:%M:%SZ").to_string())
                        .unwrap_or_else(|| t.to_string()),
                    None => String::new(),
                };
                println!(
                    "{group_index},{rid},{adapter},{instance},{native_id},{created},{updated},{has_np},{rc},{sim:.4},{dist}",
                    group_index = gi,
                    rid = csv_field(&r.record_id.0),
                    adapter = csv_field(&r.adapter),
                    instance = csv_field(&r.instance),
                    native_id = csv_field(&r.native_id),
                    created = csv_field(&created_iso),
                    updated = csv_field(&updated_iso),
                    has_np = r.has_native_path,
                    rc = record_count,
                    sim = g.min_similarity,
                    dist = g.max_distance,
                );
            }
        }
        return Ok(());
    }

    if json {
        let source_tokens = anamnesis_core::parse_csv_filter(source);
        let instance_tokens = anamnesis_core::parse_csv_filter(instance);
        let source_clause = if source_tokens.is_empty() {
            "source filter: all sources".to_string()
        } else {
            format!("source filter: {}", source_tokens.join(" OR "))
        };
        let instance_clause = if instance_tokens.is_empty() {
            "instance filter: all instances".to_string()
        } else {
            format!("instance filter: {}", instance_tokens.join(" OR "))
        };
        let summary = format!(
            "{} near-duplicate group(s) returned (mode=near); limit {}; {}; {}; cross-source-only: {}.",
            groups.len(),
            effective_limit,
            source_clause,
            instance_clause,
            !include_near_self,
        );

        let payload = serde_json::json!({
            "summary": summary,
            "count": groups.len(),
            "format": "json",
            "mode": "near",
            "limit": effective_limit,
            "merge_preview_included": merge_preview,
            "filter": {
                "source": source,
                "instance": instance,
                "require_cross_source": !include_near_self,
            },
            "groups": groups.iter().map(|g| {
                let mut group_json = serde_json::json!({
                    "record_count": g.records.len(),
                    "min_similarity": g.min_similarity,
                    "max_distance": g.max_distance,
                    "records": g.records.iter().map(|r| serde_json::json!({
                        "record_id":       r.record_id.0,
                        "adapter":         r.adapter,
                        "instance":        if r.instance.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(r.instance.clone()) },
                        "native_id":       r.native_id,
                        "created_at":      r.created_at,
                        "updated_at":      r.updated_at,
                        "has_native_path": r.has_native_path,
                    })).collect::<Vec<_>>(),
                });
                if merge_preview {
                    if let Some(preview) = near_merge_preview::build_merge_preview(g, &tag_counts) {
                        // proposed_derived_from: every loser gets
                        // an edge pointing at the keeper, modelling
                        // what a future merge mutation would write
                        // into `provenance.derived_from`.
                        let proposed_edges: Vec<serde_json::Value> = preview
                            .forget_record_ids
                            .iter()
                            .map(|loser| serde_json::json!({
                                "from": loser.0,
                                "to":   preview.keep_record_id.0,
                            }))
                            .collect();
                        let ranking_json: Vec<serde_json::Value> = preview
                            .ranking
                            .iter()
                            .map(|r| serde_json::json!({
                                "rank":            r.rank,
                                "record_id":       r.record.record_id.0,
                                "adapter":         r.record.adapter,
                                "decision":        r.decision,
                                "user_tag_count":  r.user_tag_count,
                                "effective_at":    r.effective_at,
                                "has_native_path": r.has_native_path,
                            }))
                            .collect();
                        group_json["merge_preview"] = serde_json::json!({
                            "keep_record_id":         preview.keep_record_id.0,
                            "forget_record_ids":      preview.forget_record_ids.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
                            "proposed_derived_from":  proposed_edges,
                            "ranking":                ranking_json,
                        });
                    }
                }
                group_json
            }).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    // Human form: mirror exact-dedupe structure but call out the
    // similarity/distance per group so the operator can sort by
    // confidence at a glance.
    if groups.is_empty() {
        let scope = filter_label(source, instance);
        if scope.is_empty() {
            println!("no near-duplicate groups (mode=near)");
        } else {
            println!("no near-duplicate groups (mode=near, filter: {scope})");
        }
        if !include_near_self {
            println!(
                "  (cross-source-only filter is ON by default; pass --include-near-self to also see within-adapter near-dups)"
            );
        }
        return Ok(());
    }
    let scope = filter_label(source, instance);
    let scope_suffix = if scope.is_empty() {
        String::new()
    } else {
        format!(" (filter: {scope})")
    };
    println!(
        "{} near-duplicate group(s){} (mode=near; cross-source-only: {})",
        groups.len(),
        scope_suffix,
        !include_near_self,
    );
    for (idx, g) in groups.iter().enumerate() {
        println!(
            "[{rank}] {n} record(s); min_similarity={sim:.3}, max_distance={dist}",
            rank = idx + 1,
            n = g.records.len(),
            sim = g.min_similarity,
            dist = g.max_distance,
        );
        for r in &g.records {
            let inst = if r.instance.is_empty() {
                String::new()
            } else {
                format!(":{}", r.instance)
            };
            println!(
                "    {} ({}{inst}, native_id={}, created_at={})",
                r.record_id.0, r.adapter, r.native_id, r.created_at
            );
        }
        if merge_preview {
            if let Some(preview) = near_merge_preview::build_merge_preview(g, &tag_counts) {
                println!("    merge-preview:");
                println!("      keep:   {}", preview.keep_record_id.0);
                for loser in &preview.forget_record_ids {
                    println!("      forget: {}", loser.0);
                }
                println!(
                    "      proposed derived_from: {} edge(s) → {}",
                    preview.forget_record_ids.len(),
                    preview.keep_record_id.0
                );
            }
        }
        println!();
    }
    println!(
        "These groups are *candidates* — open the records and decide before running `anamnesis forget`."
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// list-forgotten (Round 74 PR-74)
// ─────────────────────────────────────────────────────────────────────────────

/// `anamnesis list-forgotten` — audit view over `record_tombstones`.
///
/// Default output is *redacted*: sensitive fields (`native_path`,
/// `raw_hash`, `reason`) are reported only as `has_*` booleans so a
/// quick "what did I forget" check doesn't spray potentially user-
/// supplied content into the operator's terminal or a log scrape.
/// Pass `--include-sensitive` to opt-in to the full fields.
///
/// Read-only — never writes to the store or audit log. Pure audit
/// surface, distinct from the `forget` mutation.
#[allow(clippy::too_many_arguments)]
fn cmd_list_forgotten(
    data_dir: &std::path::Path,
    source: Option<&str>,
    instance: Option<&str>,
    limit: u32,
    json: bool,
    include_sensitive: bool,
    include_counts: bool,
    csv: bool,
) -> Result<()> {
    // Round 106 (PR-78ab): runtime guards for the conflicts
    // clap can't enforce cheaply on Windows (using multiple
    // clap `conflicts_with_all` entries was implicated in the
    // R105 Windows stack-overflow, so we keep the clap surface
    // small). `--csv --json` is rejected by clap; the other
    // two combinations get checked here so the CSV path never
    // has to reason about leaking `reason` / `native_path` /
    // `raw_hash` or attaching nested counts.
    if csv && include_sensitive {
        return Err(anyhow!(
            "--csv and --include-sensitive are mutually exclusive — CSV is the redacted-summary form (never carries `reason` / `native_path` / `raw_hash`)."
        ));
    }
    if csv && include_counts {
        return Err(anyhow!(
            "--csv and --include-counts are mutually exclusive — CSV is flat redacted rows. Drop --csv to get the counts block back."
        ));
    }

    let store = Store::open(db_path(data_dir))?;
    let filter = anamnesis_store::ListForgottenFilter {
        source: source.map(str::to_owned),
        instance: instance.map(str::to_owned),
        limit,
    };
    let rows = store.list_forgotten(&filter)?;
    // Round 90: opt-in tombstone aggregation. The `counts`
    // block uses the same source/instance filter as the row
    // list but reflects the full matching set (not just the
    // current page).
    let counts = if include_counts {
        Some(store.count_forgotten_by_source(&filter)?)
    } else {
        None
    };

    if csv {
        // Round 106 (PR-78ab): CSV is the redacted-summary
        // form, mirroring R91 `audit tail --csv` + R105 MCP
        // `list_forgotten { csv: true }`. Fixed header so
        // scripts can branch on column count; empty result
        // still prints the header. `--include-sensitive` and
        // `--include-counts` are runtime-rejected above so
        // this branch never has to reason about leaking
        // `reason` / `native_path` / `raw_hash`.
        println!("record_id,adapter,instance,native_id,forgotten_at,has_reason,has_native_path");
        for r in &rows {
            // forgotten_at is stored as unix-epoch i64; render
            // ISO-8601 to match audit_tail's CSV format and
            // stay human-readable.
            let at_iso = chrono::DateTime::<chrono::Utc>::from_timestamp(r.forgotten_at, 0)
                .map(|d| d.format("%Y-%m-%dT%H:%M:%SZ").to_string())
                .unwrap_or_else(|| r.forgotten_at.to_string());
            println!(
                "{rid},{adapter},{instance},{native_id},{at},{has_reason},{has_native_path}",
                rid = csv_field(&r.record_id.0),
                adapter = csv_field(&r.adapter),
                instance = csv_field(&r.instance),
                native_id = csv_field(&r.native_id),
                at = csv_field(&at_iso),
                has_reason = r.reason.is_some(),
                has_native_path = r.native_path.is_some(),
            );
        }
        return Ok(());
    }

    if json {
        // Round 126 (PR-78au): top-level redacted summary,
        // mirroring MCP `list_forgotten` R117 + CLI R123-R125
        // operator summary pattern. Source/instance are scalar
        // here (R74 store API). NEVER reads `reason`,
        // `native_path`, or `raw_hash`.
        let effective_limit = limit.clamp(1, anamnesis_store::LIST_FORGOTTEN_MAX_LIMIT);
        let source_clause = match source {
            Some(v) if !v.is_empty() => format!("source filter: {v}"),
            _ => "source filter: all sources".to_string(),
        };
        let instance_clause = match instance {
            Some(v) if !v.is_empty() => format!("instance filter: {v}"),
            _ => "instance filter: all instances".to_string(),
        };
        let summary = format!(
            "{} tombstone row(s) returned; limit {}; {}; {}; sensitive: {}; counts: {}.",
            rows.len(),
            effective_limit,
            source_clause,
            instance_clause,
            if include_sensitive {
                "included"
            } else {
                "redacted"
            },
            if include_counts {
                "included"
            } else {
                "omitted"
            },
        );

        let mut payload = serde_json::json!({
            "summary": summary,
            "count": rows.len(),
            "limit": effective_limit,
            "sensitive_included": include_sensitive,
            "rows": rows.iter().map(|r| {
                let mut row = serde_json::json!({
                    "record_id": r.record_id.0,
                    "adapter": r.adapter,
                    "instance": if r.instance.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(r.instance.clone()) },
                    "native_id": r.native_id,
                    "forgotten_at": r.forgotten_at,
                    "has_reason": r.reason.is_some(),
                    "has_native_path": r.native_path.is_some(),
                });
                if include_sensitive {
                    row["reason"] = serde_json::json!(r.reason);
                    row["native_path"] = serde_json::json!(r.native_path);
                    row["raw_hash"] = serde_json::json!(r.raw_hash);
                }
                row
            }).collect::<Vec<_>>(),
        });
        if let Some(buckets) = &counts {
            payload["counts"] = render_forgotten_counts_json(buckets);
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else if rows.is_empty() {
        println!("no forgotten records");
    } else {
        println!(
            "{} forgotten record(s){}",
            rows.len(),
            if include_sensitive {
                " (including sensitive fields)"
            } else {
                ""
            }
        );
        for r in &rows {
            let inst = if r.instance.is_empty() {
                String::new()
            } else {
                format!(":{}", r.instance)
            };
            println!(
                "  {} ({}{inst}, native_id={}, at={})",
                r.record_id.0, r.adapter, r.native_id, r.forgotten_at,
            );
            if include_sensitive {
                if let Some(p) = &r.native_path {
                    println!("    native_path : {p}");
                }
                println!("    raw_hash    : {}", r.raw_hash);
                if let Some(reason) = &r.reason {
                    println!("    reason      : {reason}");
                }
            } else {
                let mut flags = Vec::new();
                if r.native_path.is_some() {
                    flags.push("native_path");
                }
                if r.reason.is_some() {
                    flags.push("reason");
                }
                if !flags.is_empty() {
                    println!("    (sensitive fields hidden: {})", flags.join(", "));
                }
            }
        }
        if let Some(buckets) = &counts {
            println!();
            let total: u64 = buckets.iter().map(|b| b.forgotten_count).sum();
            println!("Tombstone totals (filter-scoped):");
            println!("  total: {total}");
            for b in buckets {
                let inst = if b.instance.is_empty() {
                    "(default)".to_string()
                } else {
                    b.instance.clone()
                };
                println!(
                    "  {adapter:<14} {inst:<14} {count}",
                    adapter = b.adapter,
                    inst = inst,
                    count = b.forgotten_count,
                );
            }
        }
    }
    Ok(())
}

/// Round 90 (PR-78l): render the `count_forgotten_by_source`
/// result as the shared `counts` JSON block. CLI and MCP both
/// emit this shape so scripts can branch on the same field set.
fn render_forgotten_counts_json(
    buckets: &[anamnesis_store::ForgottenSourceCount],
) -> serde_json::Value {
    let total: u64 = buckets.iter().map(|b| b.forgotten_count).sum();
    serde_json::json!({
        "total": total,
        "by_source": buckets.iter().map(|b| serde_json::json!({
            "adapter": b.adapter,
            "instance": if b.instance.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(b.instance.clone())
            },
            "forgotten_count": b.forgotten_count,
        })).collect::<Vec<_>>(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// eval-quality (Round 70)
// ─────────────────────────────────────────────────────────────────────────────

/// `anamnesis eval-quality` — score the live retrieval pipeline against
/// a JSONL judgment set. Read-only; never writes the store.
///
/// One line of `--judgments` is one [`anamnesis_search::JudgedQuery`].
/// For each query the harness builds a `SearchFilter` from the
/// optional `source / instance / kind / scope` fields, runs the
/// same `HybridSearcher` + `pack` path `search` uses, converts the
/// resulting `PackedRecord`s into [`RankedRecordRef`]s, and scores
/// them with `evaluate_query_at`. The aggregate is gated by
/// `--min-mrr` / `--min-ndcg` — exit 1 on threshold violation so
/// CI can fail loudly.
#[allow(clippy::too_many_arguments)]
async fn cmd_eval_quality(
    data_dir: &std::path::Path,
    judgments_path: &std::path::Path,
    mode_str: &str,
    limit: u32,
    at: Option<u32>,
    min_mrr: Option<f64>,
    min_ndcg: Option<f64>,
    json: bool,
) -> Result<()> {
    use std::io::BufRead;

    let depth = at.unwrap_or(limit);
    let mode = match mode_str {
        "vector" => SearchMode::Vector,
        "hybrid" => SearchMode::Hybrid,
        _ => SearchMode::Fulltext,
    };

    let file = std::fs::File::open(judgments_path)
        .map_err(|e| anyhow!("open {}: {e}", judgments_path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut judged: Vec<anamnesis_search::JudgedQuery> = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| anyhow!("read line {}: {e}", idx + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let q: anamnesis_search::JudgedQuery = serde_json::from_str(trimmed)
            .map_err(|e| anyhow!("judgments line {}: {e}", idx + 1))?;
        judged.push(q);
    }
    if judged.is_empty() {
        return Err(anyhow!(
            "judgments file {} contained no queries",
            judgments_path.display()
        ));
    }

    let store = Store::open(db_path(data_dir))?;
    // Vector / Hybrid need a real embedding provider; Fulltext doesn't.
    // Keep the default at Fulltext (codex's plan) so CI runs without
    // downloading model files.
    let provider = match mode {
        SearchMode::Fulltext => None,
        _ => Some(open_active_provider(data_dir, &store)?),
    };

    let mut per_query_evals = Vec::with_capacity(judged.len());
    for q in &judged {
        let kind_filter = match q.kind.as_deref() {
            Some(k) => Some(parse_kind(k)?),
            None => None,
        };
        let scope_filter = match q.scope.as_deref() {
            Some(s) => Some(parse_scope(s)?),
            None => None,
        };
        let store_filter = anamnesis_store::SearchFilter {
            source: q.source.clone(),
            instance: q.instance.clone(),
            kind: kind_filter.map(|k| format!("{k:?}").to_lowercase()),
            scope: scope_filter.map(|s| format!("{s:?}").to_lowercase()),
            time_from: None,
            time_to: None,
            // Round 79: eval-quality intentionally doesn't gate on
            // user_tag (a judgment is about query relevance, not
            // tag membership). Future PR-78c could add it.
            user_tag: None,
        };
        let hits = run_search(
            &store,
            &q.query,
            &store_filter,
            limit,
            mode,
            provider.as_ref(),
        )
        .await?;
        let packed = pack(
            &store,
            &hits,
            &ContextBudget {
                max_records: limit as usize,
                ..ContextBudget::default()
            },
        )?;
        let ranked: Vec<anamnesis_search::RankedRecordRef> = packed
            .iter()
            .map(|p| anamnesis_search::RankedRecordRef {
                record_id: p.record.id.0.clone(),
                adapter: p.record.source.adapter.clone(),
                instance: p.record.source.instance.clone().unwrap_or_default(),
                native_id: p.record.provenance.native_id.clone(),
            })
            .collect();
        per_query_evals.push(anamnesis_search::evaluate_query_at(depth, &ranked, q));
    }

    let summary = anamnesis_search::summarize_quality(depth, per_query_evals);
    let mrr_fail = min_mrr.is_some_and(|m| summary.mrr_at_k < m);
    let ndcg_fail = min_ndcg.is_some_and(|m| summary.ndcg_at_k < m);

    if json {
        let mut payload = serde_json::to_value(&summary)?;
        // Surface threshold deltas so a JSON consumer can render
        // failures the same way the human path does.
        let mut failures = Vec::<serde_json::Value>::new();
        if let Some(m) = min_mrr {
            if summary.mrr_at_k < m {
                failures.push(serde_json::json!({
                    "metric": "mrr_at_k",
                    "min":    m,
                    "actual": summary.mrr_at_k,
                }));
            }
        }
        if let Some(m) = min_ndcg {
            if summary.ndcg_at_k < m {
                failures.push(serde_json::json!({
                    "metric": "ndcg_at_k",
                    "min":    m,
                    "actual": summary.ndcg_at_k,
                }));
            }
        }
        payload["failed_thresholds"] = serde_json::Value::Array(failures);
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!(
            "eval-quality: mode={mode_str} queries={n} k={k}",
            n = summary.queries,
            k = summary.at,
        );
        println!(
            "  MRR@{k}  = {v:.4}{thr}",
            k = summary.at,
            v = summary.mrr_at_k,
            thr = min_mrr
                .map(|m| format!("  (min {m:.4})"))
                .unwrap_or_default(),
        );
        println!(
            "  nDCG@{k} = {v:.4}{thr}",
            k = summary.at,
            v = summary.ndcg_at_k,
            thr = min_ndcg
                .map(|m| format!("  (min {m:.4})"))
                .unwrap_or_default(),
        );
        // Per-query break-out for failure triage: anything with
        // reciprocal_rank == 0 in the topk is interesting.
        let misses: Vec<_> = summary
            .per_query
            .iter()
            .filter(|e| e.judged_relevant > 0 && e.reciprocal_rank == 0.0)
            .collect();
        if !misses.is_empty() {
            println!();
            println!("queries with no relevant hit in top-{}:", summary.at);
            for e in misses {
                println!(
                    "  - {id}  (judged_relevant={n})",
                    id = e.id,
                    n = e.judged_relevant
                );
            }
        }
    }

    if mrr_fail || ndcg_fail {
        return Err(anyhow!(
            "quality below threshold: MRR@{k}={mrr:.4} (min {mrr_min:?}), nDCG@{k}={ndcg:.4} (min {ndcg_min:?})",
            k = summary.at,
            mrr = summary.mrr_at_k,
            mrr_min = min_mrr,
            ndcg = summary.ndcg_at_k,
            ndcg_min = min_ndcg,
        ));
    }
    Ok(())
}

/// `anamnesis extract` — §-1.5 PR-6 Stage 1 deterministic gate.
///
/// Today this is **inspection-only**. It enumerates every `Episode` record
/// in the local store, runs the default Stage-1 gate over each, and prints
/// the top-N surviving candidates with their scores. No LLM calls happen.
///
/// Per §-1.5 #6 + §-1.2 #5: any future Stage 2 (the actual LLM-driven
/// distillation) must remain a separate explicit command, must show a
/// cost preview up front, and must never run inside `anamnesis import`.
#[allow(clippy::too_many_arguments)]
async fn cmd_extract(
    data_dir: &std::path::Path,
    kind_str: &str,
    source: Option<&str>,
    instance: Option<&str>,
    threshold: f32,
    limit: usize,
    explain: bool,
    json: bool,
    dry_run: bool,
    provider_id: &str,
    model: &str,
    api_base: Option<&str>,
    max_llm_calls: usize,
    yes: bool,
    concurrency: usize,
    max_retries: u32,
) -> Result<()> {
    let target_kind = anamnesis_extractor::ExtractKind::parse(kind_str).ok_or_else(|| {
        anyhow!("unknown extract kind {kind_str:?}; supported: fact, preference, feedback, skill")
    })?;

    use anamnesis_extractor::Stage1Gate as _;
    let store = Store::open(db_path(data_dir))?;
    let episodes = load_episode_records(&store, source, instance)?;
    let total_seen = episodes.len();
    let gate = anamnesis_extractor::default_gate();
    let candidates = anamnesis_extractor::stage1_select(episodes, &gate, threshold, limit);

    if !dry_run {
        // §-1.5 #6: "运行前向用户展示'将使用模型 X 做 N 次 LLM 调用'".
        // Stage 2 runs through whichever provider the user picked.
        return run_stage2_path(
            data_dir,
            &store,
            &candidates,
            target_kind,
            instance,
            total_seen,
            json,
            provider_id,
            model,
            api_base,
            max_llm_calls,
            yes,
            concurrency,
            max_retries,
        )
        .await;
    }

    if json {
        let rows: Vec<_> = candidates
            .iter()
            .map(|c| {
                serde_json::json!({
                    "record_id": c.record.id.0,
                    "adapter": c.record.source.adapter,
                    "instance": c.record.source.instance,
                    "created_at": c.record.created_at.to_rfc3339(),
                    "score": c.score,
                    "rationale": if explain { c.rationale.clone() } else { vec![] },
                    "content_preview": preview(&c.record.content, 240),
                })
            })
            .collect();
        let plan = anamnesis_extractor::plan_stage2(
            candidates.clone(),
            target_kind,
            "(stage-2 not configured)",
        );
        let out = serde_json::json!({
            "stage1": {
                "gate": gate.name(),
                "threshold": threshold,
                "limit": limit,
                "candidates_total_scanned": total_seen,
                "candidates_surfaced": candidates.len(),
                "candidates": rows,
            },
            "stage2_plan": {
                "target_kind": target_kind.as_str(),
                "estimated_llm_calls": plan.estimated_llm_calls,
                "summary": plan.summary(),
                "status": "not-yet-implemented",
            },
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!("anamnesis extract --dry-run  (Stage 2 not yet wired; this is an inspection)");
    println!(
        "  target kind   : {} → Kind::{:?}",
        target_kind.as_str(),
        target_kind.target_kind()
    );
    println!("  gate          : {}", gate.name());
    println!("  threshold     : {threshold}");
    println!("  limit         : {limit}");
    if let Some(s) = source {
        println!(
            "  source filter : {s}{}",
            instance.map(|i| format!("::{i}")).unwrap_or_default()
        );
    }
    println!();
    if candidates.is_empty() {
        println!(
            "no Episode records survived the Stage-1 gate ({total_seen} scanned, threshold={threshold})"
        );
        println!();
        println!("Stage 2 plan: nothing to extract.");
        return Ok(());
    }
    println!(
        "{} of {} Episode records survived the Stage-1 gate. Top {}:",
        candidates.len(),
        total_seen,
        candidates.len().min(limit)
    );
    println!();
    for (i, c) in candidates.iter().enumerate() {
        println!(
            "[{:>2}] score={:.2}  {}  ({})",
            i + 1,
            c.score,
            c.record.source.adapter,
            c.record.created_at.to_rfc3339(),
        );
        println!("     id={}", c.record.id.0);
        println!("     {}", preview(&c.record.content, 240));
        if explain {
            for line in &c.rationale {
                println!("       ▸ {line}");
            }
        }
        println!();
    }
    let plan = anamnesis_extractor::plan_stage2(
        candidates.clone(),
        target_kind,
        "(stage-2 not configured)",
    );
    println!("{}", plan.summary());
    println!("(Stage 2 is not yet implemented; run again once it lands.)");
    Ok(())
}

/// Read every Episode record from the local store, optionally filtered by
/// adapter / instance. Used by `cmd_extract`. Uses paged id listing
/// followed by per-id `get_record`; fine for ≤10k-record installs which
/// is what we target in Phase 1.
/// `anamnesis extract --no-dry-run` execution path.
///
/// Builds the configured `LlmProvider` (mock or openai), prints the
/// §-1.5 #6 cost preview, runs Stage 2, and persists every derived
/// record with `provenance.derived_from = source_episode_id`. The
/// `anamnesis lineage <derived-id>` audit trail starts working the
/// instant a single record gets written.
#[allow(clippy::too_many_arguments)]
async fn run_stage2_path(
    data_dir: &std::path::Path,
    store: &Store,
    candidates: &[anamnesis_extractor::Candidate],
    target_kind: anamnesis_extractor::ExtractKind,
    instance: Option<&str>,
    total_seen: usize,
    json: bool,
    provider_id: &str,
    model: &str,
    api_base: Option<&str>,
    max_llm_calls: usize,
    yes: bool,
    concurrency: usize,
    max_retries: u32,
) -> Result<()> {
    use anamnesis_extractor::cost_preview_line;

    // §-1.5 #6 safety cap — refuse before constructing the provider so
    // we don't even instantiate an HTTP client when over-budget.
    if candidates.len() > max_llm_calls {
        return Err(anyhow!(
            "Stage 1 surfaced {} candidates which exceeds --max-llm-calls={}. \
             Either re-run with `--limit {}` to bound the input, \
             or `--max-llm-calls {}` if you really want that many.",
            candidates.len(),
            max_llm_calls,
            max_llm_calls,
            candidates.len(),
        ));
    }

    let (provider, banner) = build_provider(provider_id, model, api_base, max_retries)?;

    // §-1.5 #6 plan banner — always printed BEFORE any provider call.
    let estimated_tokens: usize = candidates
        .iter()
        .map(|c| {
            let prompt = anamnesis_extractor::build_prompt(target_kind, &c.record.content);
            provider.estimate_tokens(&prompt)
        })
        .sum();
    let preview = cost_preview_line(provider.model_id(), candidates.len(), estimated_tokens);
    if !json {
        eprintln!("{preview}");
        eprintln!("{banner}");
        eprintln!();
    }

    // Interactive confirmation gate. Mock is offline+deterministic →
    // no surprise → no prompt. `--yes` skips for scripts. JSON mode
    // also skips (calling code is non-interactive by construction).
    let needs_confirm = provider_id != "mock" && !yes && !json;
    if needs_confirm {
        eprintln!(
            "About to send {} LLM request(s). Proceed? [y/N]",
            candidates.len()
        );
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).ok();
        let answer = buf.trim().to_ascii_lowercase();
        if answer != "y" && answer != "yes" {
            return Err(anyhow!(
                "aborted by user — pass `--yes` to skip this prompt in scripts"
            ));
        }
    }

    let run_started_at = chrono::Utc::now();
    let report = anamnesis_extractor::run_stage2_concurrent(
        provider.as_ref(),
        candidates,
        target_kind,
        instance,
        concurrency,
    )
    .await?;

    // Persist every derived record. Chunker is needed because
    // upsert_record demands at least one chunk row for the FTS index.
    use anamnesis_core::chunker::Chunker;
    let chunker = Chunker::default();
    let mut written = 0usize;
    for record in &report.records {
        let chunks = chunker.chunk(&record.id, &record.content);
        store.upsert_record(record, &chunks, None)?;
        written += 1;
    }

    // §-1.5 #6 audit trail: write one JSONL line per Stage 2 run to
    // `<data_dir>/audit/stage2.jsonl`. Lets `anamnesis lineage` users
    // cross-reference WHICH extract run produced a given record, and
    // makes it possible to dump cost / token / error stats over time.
    let run_finished_at = chrono::Utc::now();
    let audit_entry = serde_json::json!({
        "ts_started": run_started_at.to_rfc3339(),
        "ts_finished": run_finished_at.to_rfc3339(),
        "stage": "stage2",
        "provider_id": provider_id,
        "provider_model": provider.model_id(),
        "target_kind": target_kind.as_str(),
        "concurrency": concurrency,
        "max_retries": max_retries,
        "candidates_total_scanned": total_seen,
        "candidates_processed": candidates.len(),
        "records_written": written,
        "records_skipped": report.skipped,
        "estimated_input_tokens": report.estimated_input_tokens,
        "errors": report.errors,
        "derived_record_ids": report.records.iter().map(|r| r.id.0.clone()).collect::<Vec<_>>(),
        "source_record_ids": report.records.iter()
            .filter_map(|r| r.provenance.derived_from.as_ref().map(|p| p.0.clone()))
            .collect::<Vec<_>>(),
    });
    if let Err(e) = append_stage2_audit(data_dir, &audit_entry) {
        eprintln!("⚠ audit log write failed: {e}");
    }

    if json {
        let rows: Vec<_> = report
            .records
            .iter()
            .map(|r| {
                serde_json::json!({
                    "record_id": r.id.0,
                    "kind": format!("{:?}", r.kind).to_lowercase(),
                    "scope": format!("{:?}", r.scope).to_lowercase(),
                    "content_preview": preview_text(&r.content, 240),
                    "derived_from": r.provenance.derived_from.as_ref().map(|p| p.0.clone()),
                })
            })
            .collect();
        let out = serde_json::json!({
            "stage": "stage2",
            "provider_model": provider.model_id(),
            "candidates_total_scanned": total_seen,
            "candidates_processed": candidates.len(),
            "records_written": written,
            "records_skipped": report.skipped,
            "errors": report.errors,
            "estimated_input_tokens": report.estimated_input_tokens,
            "records": rows,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!(
        "anamnesis extract --no-dry-run  (Stage 2 via {} — mock; no network)",
        provider.model_id()
    );
    println!(
        "  scanned       : {total_seen} Episode record(s); {} survived Stage-1 gate",
        candidates.len()
    );
    println!("  records made  : {written}");
    println!(
        "  records skip  : {} (provider returned empty)",
        report.skipped
    );
    println!(
        "  llm errors    : {}{}",
        report.errors.len(),
        if report.errors.is_empty() {
            ""
        } else {
            " — see audit log"
        }
    );
    println!("  tokens (est)  : {}", report.estimated_input_tokens);
    for err in &report.errors {
        eprintln!("  ⚠ {err}");
    }
    println!();
    if written == 0 {
        println!("(no derived records persisted — nothing to extract)");
    } else {
        println!(
            "{written} derived record(s) written. Use \
             `anamnesis lineage <record-id>` to inspect the derivation chain."
        );
    }
    Ok(())
}

fn preview_text(s: &str, max_chars: usize) -> String {
    preview(s, max_chars)
}

/// Build the `LlmProvider` the user requested. Returns the provider
/// plus a short human-readable banner the cost preview prints right
/// after the "Stage 2 plan: …" line.
///
/// Provider selection rules:
/// - `mock`  → `MockProvider::default_instance()` (no network)
/// - `openai` → `OpenAiProvider::new(model)` configured from
///   `OPENAI_API_KEY` (env) and `--api-base` / `OPENAI_API_BASE`
///   (default `https://api.openai.com/v1`).
///
/// Unknown ids return an error before any candidate is processed.
fn build_provider(
    provider_id: &str,
    model: &str,
    api_base: Option<&str>,
    max_retries: u32,
) -> Result<(Box<dyn anamnesis_extractor::LlmProvider>, String)> {
    match provider_id {
        "mock" => Ok((
            Box::new(anamnesis_extractor::MockProvider::default_instance()),
            "Stage 2 will run via the built-in MockProvider — zero network requests, \
             deterministic output."
                .into(),
        )),
        "openai" => build_openai_provider(model, api_base, max_retries),
        "anthropic" => build_anthropic_provider(model, api_base, max_retries),
        other => Err(anyhow!(
            "unknown --provider {other:?}; supported: mock, openai, anthropic"
        )),
    }
}

#[cfg(feature = "anthropic-provider")]
fn build_anthropic_provider(
    model: &str,
    api_base: Option<&str>,
    max_retries: u32,
) -> Result<(Box<dyn anamnesis_extractor::LlmProvider>, String)> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
        anyhow!(
            "ANTHROPIC_API_KEY environment variable is required for `--provider anthropic`. \
             Set it before running, or use `--provider mock`."
        )
    })?;
    let mut p = anamnesis_extractor::AnthropicProvider::new(model)
        .with_api_key(api_key)
        .with_max_retries(max_retries);
    // Priority: --api-base CLI flag > ANTHROPIC_API_BASE env > default.
    let resolved_base = api_base
        .map(|s| s.to_string())
        .or_else(|| std::env::var("ANTHROPIC_API_BASE").ok());
    if let Some(base) = resolved_base {
        p = p.with_api_base(base);
    }
    let banner = format!(
        "Stage 2 will run via Anthropic Messages API at {} (model={}, max_retries={}). \
         Each candidate is one HTTP POST.",
        p.api_base(),
        p.model_name(),
        p.retry_policy().max_attempts,
    );
    Ok((Box::new(p), banner))
}

#[cfg(not(feature = "anthropic-provider"))]
fn build_anthropic_provider(
    _model: &str,
    _api_base: Option<&str>,
    _max_retries: u32,
) -> Result<(Box<dyn anamnesis_extractor::LlmProvider>, String)> {
    Err(anyhow!(
        "`--provider anthropic` requires the `anthropic-provider` cargo feature, \
         which is on by default. Rebuild with `cargo build --features anthropic-provider` \
         (or `--all-features`) and try again."
    ))
}

#[cfg(feature = "openai-provider")]
fn build_openai_provider(
    model: &str,
    api_base: Option<&str>,
    max_retries: u32,
) -> Result<(Box<dyn anamnesis_extractor::LlmProvider>, String)> {
    let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
        anyhow!(
            "OPENAI_API_KEY environment variable is required for `--provider openai`. \
             Set it before running, or use `--provider mock`."
        )
    })?;
    let mut p = anamnesis_extractor::OpenAiProvider::new(model)
        .with_api_key(api_key)
        .with_max_retries(max_retries);
    // Priority: --api-base CLI flag > OPENAI_API_BASE env > library default.
    let resolved_base = api_base
        .map(|s| s.to_string())
        .or_else(|| std::env::var("OPENAI_API_BASE").ok());
    if let Some(base) = resolved_base {
        p = p.with_api_base(base);
    }
    let banner = format!(
        "Stage 2 will run via OpenAI-compatible provider at {} (model={}, max_retries={}). \
         Each candidate is one HTTP POST.",
        p.api_base(),
        p.model_name(),
        p.retry_policy().max_attempts,
    );
    Ok((Box::new(p), banner))
}

#[cfg(not(feature = "openai-provider"))]
fn build_openai_provider(
    _model: &str,
    _api_base: Option<&str>,
    _max_retries: u32,
) -> Result<(Box<dyn anamnesis_extractor::LlmProvider>, String)> {
    Err(anyhow!(
        "`--provider openai` requires the `openai-provider` cargo feature, \
         which is on by default. Rebuild with `cargo build --features openai-provider` \
         (or `--all-features`) and try again."
    ))
}

/// Append one Stage 2 audit entry to `<data_dir>/audit/stage2.jsonl`.
///
/// One file (not per-run) so `jq -s '.' stage2.jsonl` slurps the whole
/// history; per-run timestamps live in the entry. Append-only; we
/// never rewrite or compact this file from inside the CLI. Operators
/// who want rotation can wire it to logrotate externally.
/// Read the entire audit log into memory and return one JSON value per
/// line. Lines that fail to parse are skipped with a `tracing::warn`
/// so a single corrupted entry doesn't make the whole log unreadable.
fn read_audit_log(data_dir: &std::path::Path) -> Result<Vec<serde_json::Value>> {
    let path = data_dir.join("audit").join("stage2.jsonl");
    if !path.is_file() {
        return Ok(vec![]);
    }
    let body =
        std::fs::read_to_string(&path).map_err(|e| anyhow!("read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(v) => out.push(v),
            Err(e) => {
                tracing::warn!(
                    line_no = i + 1,
                    path = %path.display(),
                    error = %e,
                    "audit log: skipping unparseable line"
                );
            }
        }
    }
    Ok(out)
}

/// `anamnesis audit list` / `anamnesis audit show <target>`.
/// `anamnesis doctor` — §-2.5 per-source health check.
///
/// For each registered source, instantiates the adapter at its
/// configured location and calls `MemoryAdapter::health().await`. Also
/// folds in the store's per-source record/chunk counts and
/// `last_import_at` so the output answers two questions at a glance:
///
///   1. Is the upstream still reachable? (the adapter's `health()` ok)
///   2. Is what's in the store fresh? (record_count + last_import_at)
///
/// With `--include-unregistered`, also runs the discovery detectors so
/// the output names adapters that *could* be wired up but aren't yet.
#[allow(clippy::too_many_arguments)]
async fn cmd_doctor(
    data_dir: &std::path::Path,
    filter_source: Option<&str>,
    filter_instance: Option<&str>,
    include_unregistered: bool,
    json: bool,
    strict: bool,
    since: Option<&str>,
    strict_staleness: bool,
    json_summary: bool,
) -> Result<()> {
    let stale_threshold = match since {
        Some(spec) => Some(parse_doctor_since(spec)?),
        None => None,
    };
    let now = chrono::Utc::now().timestamp();
    let store = Store::open(db_path(data_dir))?;
    let registered = store.list_sources_with_counts()?;

    // Round 110 (PR-78af): `--source` now accepts a comma-
    // separated OR list (`--source mem0,claude-code`) via
    // core's shared `parse_csv_filter`, symmetric with R102
    // audit-tail / R103 list-sources / R104 dedupe multi-
    // value. Empty parse = no filter (back-compat with R74's
    // single-value semantic). `--instance` stays single-value
    // AND-combined with the adapter set.
    let sources = anamnesis_core::parse_csv_filter(filter_source);
    let source_matches =
        |adapter: &str| -> bool { sources.is_empty() || sources.iter().any(|s| s == adapter) };

    // Round 114 (PR-78aj): `--instance` now also accepts a
    // comma-separated OR list, symmetric with `--source`'s
    // R110 behaviour. Combined as AND with the source set:
    // `source ∈ [a,b] && instance ∈ [c,d]`. Empty parse on
    // either dimension = no filter on that dimension. The
    // registered-source path uses both predicates; the
    // unregistered detector path doesn't carry instance, so
    // it stays unchanged.
    let instances = anamnesis_core::parse_csv_filter(filter_instance);
    let instance_matches =
        |inst: &str| -> bool { instances.is_empty() || instances.iter().any(|i| i == inst) };

    // Round-64 follow-up: one `GROUP BY` query instead of N row-
    // materializing scans. For 13 registered sources this turns
    // 13 × `SELECT * FROM import_errors WHERE adapter = ?` (which
    // could materialize huge result sets on bad-import storms) into
    // one `SELECT adapter, COUNT(1) FROM import_errors GROUP BY adapter`.
    let error_counts = store.count_import_errors_by_adapter().unwrap_or_default();

    let mut rows = Vec::new();
    for swc in &registered {
        let src = &swc.source;
        if !source_matches(&src.adapter) {
            continue;
        }
        if !instance_matches(&src.instance) {
            continue;
        }
        let health = run_adapter_health(src).await;
        let stale = stale_threshold.map(|t| match src.last_import_at {
            Some(ts) => (now - ts) > t,
            // Never imported → counts as stale when a threshold is set.
            None => true,
        });
        let import_errors_n = error_counts.get(&src.adapter).copied().unwrap_or(0);
        rows.push(DoctorRow {
            adapter: src.adapter.clone(),
            instance: instance_label(&src.instance),
            location: src.location.clone(),
            registered: true,
            ok: health.as_ref().map(|h| h.ok).unwrap_or(false),
            detail: match &health {
                Some(h) => h.detail.clone(),
                None => "adapter not wired into doctor; see `import` dispatch".to_string(),
            },
            record_count: Some(swc.record_count),
            chunk_count: Some(swc.chunk_count),
            last_import_at: src.last_import_at,
            stale,
            import_errors: import_errors_n,
        });
    }

    if include_unregistered {
        // Probe every detector. Any (adapter, instance=None) that
        // doesn't already appear in `registered` is appended as an
        // unregistered candidate row.
        let registered_pairs: std::collections::HashSet<(String, String)> = registered
            .iter()
            .map(|s| (s.source.adapter.clone(), s.source.instance.clone()))
            .collect();
        let detected = run_all_detectors().await;
        for d in detected {
            // Detector results don't carry an instance (always None).
            // Treat them as the default-instance row for that adapter.
            let key = (d.adapter.clone(), String::new());
            if registered_pairs.contains(&key) {
                continue;
            }
            if !source_matches(&d.adapter) {
                continue;
            }
            rows.push(DoctorRow {
                adapter: d.adapter,
                instance: "(default)".into(),
                location: Some(d.location),
                registered: false,
                ok: matches!(d.confidence, anamnesis_core::Confidence::High),
                detail: d.note.unwrap_or_else(|| "detector hit".to_string()),
                record_count: d.estimated_records,
                chunk_count: None,
                last_import_at: None,
                import_errors: 0,
                // Unregistered rows aren't subject to staleness — there
                // was never an import to begin with.
                stale: None,
            });
        }
    }

    if json || json_summary {
        let rows_json: Vec<_> = rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "adapter": r.adapter,
                    "instance": r.instance,
                    "location": r.location,
                    "registered": r.registered,
                    "ok": r.ok,
                    "detail": r.detail,
                    "record_count": r.record_count,
                    "chunk_count": r.chunk_count,
                    "last_import_at": r.last_import_at,
                    "stale": r.stale,
                    "import_errors": r.import_errors,
                })
            })
            .collect();

        if json_summary {
            // Round 130 (PR-78ay): additive enrichment
            // envelope. The bare-array `--json` shape stays
            // for back-compat with existing CI scripts.
            // `summary` / `filters` NEVER read `location` or
            // `detail` text — only counts and parsed filter
            // tokens.
            let total = rows.len() as u64;
            let registered_count = rows.iter().filter(|r| r.registered).count() as u64;
            let ok_count = rows.iter().filter(|r| r.ok && r.registered).count() as u64;
            let unhealthy_count = rows.iter().filter(|r| !r.ok && r.registered).count() as u64;
            let stale_count = rows.iter().filter(|r| r.stale.unwrap_or(false)).count() as u64;
            let unregistered_count = rows.iter().filter(|r| !r.registered).count() as u64;

            let envelope = serde_json::json!({
                "summary": {
                    "total": total,
                    "registered": registered_count,
                    "ok": ok_count,
                    "unhealthy": unhealthy_count,
                    "stale": stale_count,
                    "unregistered": unregistered_count,
                },
                "filters": {
                    "source": anamnesis_core::parse_csv_filter(filter_source),
                    "instance": anamnesis_core::parse_csv_filter(filter_instance),
                    "since_seconds": stale_threshold,
                    "include_unregistered": include_unregistered,
                },
                "sources": rows_json,
            });
            println!("{}", serde_json::to_string_pretty(&envelope)?);
        } else {
            println!("{}", serde_json::to_string_pretty(&rows_json)?);
        }
        // Honor --strict and --strict-staleness in json mode too — print
        // the JSON first so the caller can still parse stdout, then exit
        // non-zero. Lets CI gates do `doctor --json --strict | tee report.json`.
        return apply_doctor_exit_gate(&rows, strict, strict_staleness);
    }

    if rows.is_empty() {
        println!(
            "No registered sources{}. Run `anamnesis discover` and `anamnesis source add` first.",
            if include_unregistered {
                " — and no detectors found anything either"
            } else {
                ""
            }
        );
        return Ok(());
    }

    println!(
        "Anamnesis doctor — per-source health check{}",
        if include_unregistered {
            " (incl. unregistered detector hits)"
        } else {
            ""
        }
    );
    println!();
    for row in &rows {
        let tag = if !row.registered {
            "?"
        } else if row.stale == Some(true) && row.ok {
            // Registered + reachable but data is older than --since.
            "!"
        } else if row.ok {
            "✓"
        } else {
            "✗"
        };
        let title = if row.instance == "(default)" {
            row.adapter.clone()
        } else {
            format!("{} :: {}", row.adapter, row.instance)
        };
        let status = if row.registered {
            if row.ok {
                if row.stale == Some(true) {
                    "registered, healthy, STALE"
                } else {
                    "registered, healthy"
                }
            } else {
                "registered, NOT HEALTHY"
            }
        } else {
            "NOT REGISTERED (detected locally)"
        };
        println!("[{tag}] {title}  — {status}");
        if let Some(loc) = &row.location {
            println!("    location          : {loc}");
        }
        if let Some(count) = row.record_count {
            print!("    records in store  : {count}");
            if let Some(ts) = row.last_import_at {
                let when = chrono::DateTime::<chrono::Utc>::from_timestamp(ts, 0)
                    .map(|dt| dt.to_rfc3339())
                    .unwrap_or_else(|| ts.to_string());
                print!(" · last_import_at: {when}");
            }
            println!();
        }
        if let Some(chunks) = row.chunk_count {
            println!("    chunks            : {chunks}");
        }
        if row.registered && row.import_errors > 0 {
            println!("    import errors     : {}", row.import_errors);
        }
        println!("    detail            : {}", row.detail);
        println!();
    }
    let ok_count = rows.iter().filter(|r| r.registered && r.ok).count();
    let bad_count = rows.iter().filter(|r| r.registered && !r.ok).count();
    let unreg_count = rows.iter().filter(|r| !r.registered).count();
    let stale_count = rows.iter().filter(|r| r.stale == Some(true)).count();
    let stale_blurb = if stale_threshold.is_some() {
        format!(", {stale_count} stale")
    } else {
        String::new()
    };
    println!(
        "Summary: {ok_count} healthy, {bad_count} unhealthy, {unreg_count} unregistered{stale_blurb}."
    );
    // Round-69: append a small "MCP request latency" footer if the
    // local store has any recorded MCP tool calls in the last 24h.
    // The JSON output path keeps its existing shape (a wire change is
    // a separate PR — see Round 69 PR body).
    let metrics_window_secs: i64 = 86_400;
    if let Ok(summaries) = store.summarize_mcp_request_metrics(Some(now - metrics_window_secs)) {
        if !summaries.is_empty() {
            println!();
            println!("Recent MCP tool latency (24h)");
            for s in summaries {
                let last_rc = s
                    .last_result_count
                    .map(|n| format!(" n_hits={n}"))
                    .unwrap_or_default();
                println!(
                    "  {tool} n={count} err={errors} p50={p50}ms p95={p95}ms p99={p99}ms last={last}ms{last_rc}",
                    tool = s.tool,
                    count = s.count,
                    errors = s.errors,
                    p50 = s.p50_ms,
                    p95 = s.p95_ms,
                    p99 = s.p99_ms,
                    last = s.last_ms,
                );
            }
        }
    }
    apply_doctor_exit_gate(&rows, strict, strict_staleness)
}

/// Decide whether `doctor` should exit non-zero based on the flags.
/// Both human and JSON paths call this; the JSON path already printed
/// the report so we just need to propagate the exit signal.
fn apply_doctor_exit_gate(rows: &[DoctorRow], strict: bool, strict_staleness: bool) -> Result<()> {
    let bad_count = rows.iter().filter(|r| r.registered && !r.ok).count();
    let stale_count = rows.iter().filter(|r| r.stale == Some(true)).count();
    if strict && bad_count > 0 {
        return Err(anyhow!(
            "{bad_count} registered source(s) reported unhealthy under --strict"
        ));
    }
    if strict_staleness && stale_count > 0 {
        return Err(anyhow!(
            "{stale_count} registered source(s) are stale under --strict-staleness"
        ));
    }
    Ok(())
}

struct DoctorRow {
    adapter: String,
    instance: String,
    location: Option<String>,
    registered: bool,
    ok: bool,
    detail: String,
    record_count: Option<u64>,
    chunk_count: Option<u64>,
    last_import_at: Option<i64>,
    /// `Some(true)` if this registered row is older than `--since`
    /// or never-imported. `Some(false)` if it's fresh enough.
    /// `None` when `--since` wasn't passed.
    stale: Option<bool>,
    /// Count of `import_errors` rows scoped to this adapter (across
    /// every instance — `import_errors` doesn't denormalize the
    /// instance discriminator into the row count yet). 0 when clean.
    /// Only populated for `registered` rows.
    import_errors: u64,
}

/// Parse the `--since` value into seconds. Accepts shapes:
///   - `7d`  → 7 * 86_400
///   - `12h` → 12 * 3_600
///   - `30m` → 30 * 60
///   - bare integer → seconds
fn parse_doctor_since(spec: &str) -> Result<i64> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err(anyhow!("--since cannot be empty"));
    }
    let (num_str, mult) = match spec.chars().last() {
        Some('d') | Some('D') => (&spec[..spec.len() - 1], 86_400_i64),
        Some('h') | Some('H') => (&spec[..spec.len() - 1], 3_600_i64),
        Some('m') | Some('M') => (&spec[..spec.len() - 1], 60_i64),
        _ => (spec, 1_i64),
    };
    let n: i64 = num_str.parse().map_err(|_| {
        anyhow!("--since must be of the form `Nd`, `Nh`, `Nm`, or bare seconds; got {spec:?}")
    })?;
    if n < 0 {
        return Err(anyhow!("--since must be non-negative; got {spec:?}"));
    }
    Ok(n.saturating_mul(mult))
}

fn instance_label(instance: &str) -> String {
    if instance.is_empty() {
        "(default)".into()
    } else {
        instance.to_string()
    }
}

/// Build the adapter for a registered source and call `.health().await`.
/// Returns `None` when the adapter id isn't one we recognize (so the
/// caller can render a friendly "not wired" message).
async fn run_adapter_health(
    src: &anamnesis_store::SourceRow,
) -> Option<anamnesis_core::adapter::HealthStatus> {
    use anamnesis_core::adapter::MemoryAdapter;
    let location_path = src.location.as_deref().map(std::path::PathBuf::from);
    let instance = if src.instance.is_empty() {
        None
    } else {
        Some(src.instance.as_str())
    };

    match src.adapter.as_str() {
        anamnesis_adapter_claude_code::ADAPTER_ID => {
            let path = location_path
                .unwrap_or_else(|| home_join(&[".claude", "projects"]).unwrap_or_default());
            let adapter = ClaudeCodeAdapter::new(ClaudeCodeConfig {
                projects_root: path,
                instance: instance.map(str::to_owned),
            });
            Some(adapter.health().await)
        }
        anamnesis_adapter_codex::ADAPTER_ID => {
            let path = location_path.unwrap_or_else(|| home_join(&[".codex"]).unwrap_or_default());
            Some(codex_adapter(path, instance).health().await)
        }
        anamnesis_adapter_mem0::ADAPTER_ID => {
            let path = location_path
                .unwrap_or_else(|| home_join(&[".mem0", "db.sqlite"]).unwrap_or_default());
            Some(mem0_sqlite_adapter(path, instance).health().await)
        }
        anamnesis_adapter_letta::ADAPTER_ID => {
            let path = location_path
                .unwrap_or_else(|| home_join(&[".letta", "letta.db"]).unwrap_or_default());
            Some(letta_adapter(path, instance).health().await)
        }
        anamnesis_adapter_hermes::ADAPTER_ID => {
            let path = location_path.unwrap_or_else(|| home_join(&[".hermes"]).unwrap_or_default());
            Some(hermes_adapter(path, instance).health().await)
        }
        anamnesis_adapter_openclaw::ADAPTER_ID => {
            let path =
                location_path.unwrap_or_else(|| home_join(&[".openclaw"]).unwrap_or_default());
            Some(openclaw_adapter(path, instance).health().await)
        }
        anamnesis_adapter_tdai::ADAPTER_ID => {
            let path = location_path
                .unwrap_or_else(|| home_join(&[".openclaw", "memory-tdai"]).unwrap_or_default());
            Some(tdai_adapter(path, instance).health().await)
        }
        anamnesis_adapter_openviking::ADAPTER_ID => {
            let path = location_path
                .unwrap_or_else(|| home_join(&[".openviking", "data"]).unwrap_or_default());
            Some(openviking_adapter(path, instance).health().await)
        }
        anamnesis_adapter_mempalace::ADAPTER_ID => {
            let path =
                location_path.unwrap_or_else(|| home_join(&[".mempalace"]).unwrap_or_default());
            Some(mempalace_adapter(path, instance).health().await)
        }
        anamnesis_adapter_memori::ADAPTER_ID => {
            let path = location_path
                .unwrap_or_else(|| home_join(&[".memori", "memori.db"]).unwrap_or_default());
            Some(memori_adapter(path, instance).health().await)
        }
        anamnesis_adapter_memos::ADAPTER_ID => {
            let path = location_path.unwrap_or_else(|| home_join(&[".memos"]).unwrap_or_default());
            Some(memos_adapter(path, instance).health().await)
        }
        anamnesis_adapter_memary::ADAPTER_ID => {
            let path = location_path
                .unwrap_or_else(|| home_join(&[".memary", "data"]).unwrap_or_default());
            Some(memary_adapter(path, instance).health().await)
        }
        anamnesis_adapter_generic_mcp::ADAPTER_ID => {
            // `MemoryAdapter::health()` on generic-mcp is a single GET
            // to `<url>/healthz` — much cheaper than the resources/list
            // pull that import does. The bearer token (if any) is
            // resolved from `src.config_json` the same way `cmd_import`
            // resolves it, so token-env discrepancies show up here too.
            let Some(url) = src.location.clone() else {
                return Some(anamnesis_core::adapter::HealthStatus {
                    ok: false,
                    detail: "generic-mcp registered without --url".to_string(),
                });
            };
            let token = match resolve_generic_mcp_token(src.config_json.as_deref()) {
                Ok(t) => t,
                Err(e) => {
                    return Some(anamnesis_core::adapter::HealthStatus {
                        ok: false,
                        detail: format!("generic-mcp token resolution failed: {e}"),
                    });
                }
            };
            let adapter =
                anamnesis_adapter_generic_mcp::generic_mcp_adapter(url, token.as_deref(), instance);
            Some(adapter.health().await)
        }
        _ => None,
    }
}

/// Run every detector we know about. Single place to keep this list in
/// sync with `cmd_discover` and the §-2.5 adapter roster.
async fn run_all_detectors() -> Vec<anamnesis_core::discovery::DetectedSource> {
    let discovery = Discovery::new()
        .register(Box::new(ClaudeCodeDetector::new()))
        .register(Box::new(Mem0SqliteDetector::new()))
        .register(Box::new(CodexDetector::new()))
        .register(Box::new(LettaSqliteDetector::new()))
        .register(Box::new(HermesDetector::new()))
        .register(Box::new(OpenClawDetector::new()))
        .register(Box::new(TdaiDetector::new()))
        .register(Box::new(OpenVikingDetector::new()))
        .register(Box::new(MempalaceDetector::new()))
        .register(Box::new(MemoriDetector::new()))
        .register(Box::new(MemosDetector::new()))
        .register(Box::new(MemaryDetector::new()));
    discovery.detect_all(&DetectOpts::default()).await
}

fn cmd_audit(data_dir: &std::path::Path, sub: AuditCmd) -> Result<()> {
    match sub {
        // Stage 2 audit (extractor runs) lives in a separate file
        // — `data_dir/audit/stage2.jsonl`. These two arms route
        // there.
        AuditCmd::List { limit, json } => {
            let entries = read_audit_log(data_dir)?;
            audit_list(&entries, limit, json)
        }
        AuditCmd::Show { target, json } => {
            let entries = read_audit_log(data_dir)?;
            audit_show(&entries, &target, json)
        }
        // Round 84: global mutation/search audit at
        // `data_dir/audit.log`. Different file, different reader.
        AuditCmd::Tail {
            limit,
            action,
            since,
            json,
            csv,
        } => cmd_audit_tail(
            data_dir,
            limit,
            action.as_deref(),
            since.as_deref(),
            json,
            csv,
        ),
    }
}

/// Round 84 (PR-78f): `anamnesis audit tail` — read
/// `data_dir/audit.log` and print the last N entries with
/// optional action/since filters. CLI is the "operator mode"
/// surface; --json carries full `detail`, the human renderer is
/// summary-only so a casual tail doesn't dump search queries or
/// forget reasons.
#[allow(clippy::too_many_arguments)]
fn cmd_audit_tail(
    data_dir: &std::path::Path,
    limit: usize,
    action: Option<&str>,
    since: Option<&str>,
    json: bool,
    csv: bool,
) -> Result<()> {
    let since_dt: Option<chrono::DateTime<chrono::Utc>> = match since {
        Some(spec) => {
            let lookback_seconds = parse_doctor_since(spec)?;
            Some(chrono::Utc::now() - chrono::Duration::seconds(lookback_seconds))
        }
        None => None,
    };

    // Round 102 (PR-78x): comma-separated `--action` becomes a
    // multi-value OR filter (`["forget", "search"]`). Parsing
    // lives in `anamnesis_core::parse_audit_actions` so CLI +
    // MCP share the split rule byte-for-byte. JSON keeps the
    // existing `filter.action` raw string for back-compat with
    // R84/R91 clients, plus an additive `filter.actions` array
    // carrying the normalised tokens.
    let actions = anamnesis_core::parse_audit_actions(action);
    let opts = anamnesis_core::AuditTailOptions {
        limit: Some(limit),
        since: since_dt,
        actions: actions.clone(),
    };
    let audit = anamnesis_core::Audit::new(data_dir);
    let rows = audit
        .tail(&opts)
        .map_err(|e| anyhow!("read audit.log: {e}"))?;
    let effective_limit = limit.clamp(1, anamnesis_core::AUDIT_TAIL_MAX_LIMIT);

    if json {
        // Round 127 (PR-78av): top-level redacted summary
        // mirroring MCP R116 + CLI R123-R126 operator
        // summary pattern. NEVER reads `entry.detail`,
        // `reason`, or `query`. CLI JSON path historically
        // includes full detail in `entries[]` (operator
        // surface), so the summary clause reports
        // `detail: included` to reflect that.
        let action_clause = if actions.is_empty() {
            "all actions".to_string()
        } else {
            format!("action filter: {}", actions.join(" OR "))
        };
        let since_clause = match since {
            Some(s) => format!("since: {s}"),
            None => "since: all time".to_string(),
        };
        let summary = format!(
            "{} audit entries returned; limit {}; {}; {}; detail: included.",
            rows.len(),
            effective_limit,
            action_clause,
            since_clause,
        );
        let payload = serde_json::json!({
            "summary": summary,
            "count": rows.len(),
            "limit": effective_limit,
            "filter": {
                "action":  action,
                "actions": actions,
                "since":   since,
            },
            "entries": rows.iter().map(|r| serde_json::json!({
                "line_no":   r.line_no,
                "timestamp": r.entry.timestamp,
                "action":    r.entry.action,
                "detail":    r.entry.detail,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else if csv {
        // Round 91: CSV uses the **same redacted summary fields**
        // as the human renderer — never `detail`, `reason`, or
        // `query`. Empty result still emits the header so
        // downstream scripts can branch uniformly.
        println!("line_no,timestamp,action,via,outcome");
        for r in &rows {
            let (via, outcome) = audit_tail_summary(r);
            println!(
                "{line_no},{ts},{action},{via},{outcome}",
                line_no = r.line_no,
                ts = csv_field(&r.entry.timestamp.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
                action = csv_field(&r.entry.action),
                via = csv_field(&via),
                outcome = csv_field(&outcome),
            );
        }
    } else if rows.is_empty() {
        if action.is_some() || since.is_some() {
            println!("no audit entries matched (filter applied)");
        } else {
            println!("no audit entries yet (data_dir/audit.log empty or missing)");
        }
    } else {
        println!(
            "{:>5}  {:<25}  {:<18}  {:<6}  {}",
            "line", "timestamp", "action", "via", "outcome"
        );
        for r in &rows {
            let (via, outcome) = audit_tail_summary(r);
            println!(
                "{:>5}  {:<25}  {:<18}  {:<6}  {}",
                r.line_no,
                r.entry.timestamp.format("%Y-%m-%dT%H:%M:%SZ"),
                r.entry.action,
                via,
                outcome,
            );
        }
        println!();
        println!(
            "({} entries shown — pass --json for full detail including reason/query/etc.)",
            rows.len()
        );
    }
    Ok(())
}

/// Round 91 (PR-78m): shared `(via, outcome)` extraction for the
/// human and CSV `audit tail` renderers. Pulls only the
/// **redacted summary** fields — `via` and one of
/// `outcome`/`status`/`changed` — so neither surface accidentally
/// leaks `reason` / `query` / `detail`.
fn audit_tail_summary(r: &anamnesis_core::AuditTailRow) -> (String, String) {
    let via = r
        .entry
        .detail
        .get("via")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let outcome = r
        .entry
        .detail
        .get("outcome")
        .or_else(|| r.entry.detail.get("status"))
        .or_else(|| r.entry.detail.get("changed"))
        .map(|v| match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default();
    (via, outcome)
}

fn audit_list(entries: &[serde_json::Value], limit: usize, json: bool) -> Result<()> {
    let mut rows: Vec<(usize, &serde_json::Value)> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (i + 1, e))
        .collect();
    rows.reverse(); // newest first
    rows.truncate(limit);

    if json {
        let payload: Vec<_> = rows
            .iter()
            .map(|(line_no, entry)| {
                serde_json::json!({
                    "line_no": line_no,
                    "ts_started": entry.get("ts_started"),
                    "provider_id": entry.get("provider_id"),
                    "provider_model": entry.get("provider_model"),
                    "target_kind": entry.get("target_kind"),
                    "candidates_processed": entry.get("candidates_processed"),
                    "records_written": entry.get("records_written"),
                    "records_skipped": entry.get("records_skipped"),
                    "errors_count": entry
                        .get("errors")
                        .and_then(|e| e.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0),
                    "estimated_input_tokens": entry.get("estimated_input_tokens"),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    if rows.is_empty() {
        println!(
            "No Stage 2 audit entries yet. Run \
             `anamnesis extract --no-dry-run` to populate the log."
        );
        return Ok(());
    }

    println!(
        "{:<5}  {:<25}  {:<14}  {:<14}  {:<8}  {:>5}  {:>5}  {:>6}",
        "#", "started", "provider", "model", "kind", "made", "skip", "tokens"
    );
    for (line_no, entry) in &rows {
        let ts = entry
            .get("ts_started")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let pid = entry
            .get("provider_id")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let model = entry
            .get("provider_model")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let kind = entry
            .get("target_kind")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let made = entry
            .get("records_written")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let skip = entry
            .get("records_skipped")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let tokens = entry
            .get("estimated_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        // Truncate ts to seconds for compactness.
        let ts_short = ts.get(..19).unwrap_or(ts);
        println!(
            "{:<5}  {:<25}  {:<14}  {:<14}  {:<8}  {:>5}  {:>5}  {:>6}",
            line_no,
            ts_short,
            truncate_for_column(pid, 14),
            truncate_for_column(model, 14),
            truncate_for_column(kind, 8),
            made,
            skip,
            tokens,
        );
    }
    println!();
    println!(
        "{} entries shown (newest first). Use `anamnesis audit show <#>` for details.",
        rows.len()
    );
    Ok(())
}

fn audit_show(entries: &[serde_json::Value], target: &str, json: bool) -> Result<()> {
    let line_no = if target.eq_ignore_ascii_case("last") {
        if entries.is_empty() {
            return Err(anyhow!("audit log is empty — nothing to show"));
        }
        entries.len()
    } else {
        target.parse::<usize>().map_err(|_| {
            anyhow!("unrecognized target {target:?}; pass a 1-based line number or `last`")
        })?
    };
    if line_no == 0 || line_no > entries.len() {
        return Err(anyhow!(
            "line number {line_no} out of range; audit log has {} entries",
            entries.len()
        ));
    }
    let entry = &entries[line_no - 1];
    if json {
        println!("{}", serde_json::to_string_pretty(entry)?);
        return Ok(());
    }
    println!("Stage 2 audit entry #{line_no}:");
    println!(
        "  ts_started     : {}",
        entry
            .get("ts_started")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
    );
    println!(
        "  ts_finished    : {}",
        entry
            .get("ts_finished")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
    );
    println!(
        "  provider_id    : {}",
        entry
            .get("provider_id")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
    );
    println!(
        "  provider_model : {}",
        entry
            .get("provider_model")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
    );
    println!(
        "  target_kind    : {}",
        entry
            .get("target_kind")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
    );
    println!(
        "  candidates_total_scanned : {}",
        entry
            .get("candidates_total_scanned")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    );
    println!(
        "  candidates_processed     : {}",
        entry
            .get("candidates_processed")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    );
    println!(
        "  records_written          : {}",
        entry
            .get("records_written")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    );
    println!(
        "  records_skipped          : {}",
        entry
            .get("records_skipped")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    );
    println!(
        "  estimated_input_tokens   : {}",
        entry
            .get("estimated_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    );
    if let Some(errors) = entry.get("errors").and_then(|v| v.as_array()) {
        if !errors.is_empty() {
            println!("  errors ({}):", errors.len());
            for e in errors {
                println!("    ▸ {}", e.as_str().unwrap_or("?"));
            }
        }
    }
    if let Some(derived) = entry.get("derived_record_ids").and_then(|v| v.as_array()) {
        println!("  derived record ids ({}):", derived.len());
        for (i, d) in derived.iter().enumerate() {
            if i >= 10 {
                println!(
                    "    … {} more (use --json for the full list)",
                    derived.len() - 10
                );
                break;
            }
            println!("    ▸ {}", d.as_str().unwrap_or("?"));
        }
    }
    if let Some(sources) = entry.get("source_record_ids").and_then(|v| v.as_array()) {
        println!("  source record ids ({}):", sources.len());
        for (i, s) in sources.iter().enumerate() {
            if i >= 10 {
                println!(
                    "    … {} more (use --json for the full list)",
                    sources.len() - 10
                );
                break;
            }
            println!("    ▸ {}", s.as_str().unwrap_or("?"));
        }
    }
    Ok(())
}

fn truncate_for_column(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{head}…")
}

fn append_stage2_audit(data_dir: &std::path::Path, entry: &serde_json::Value) -> Result<()> {
    let dir = data_dir.join("audit");
    std::fs::create_dir_all(&dir).map_err(|e| anyhow!("create audit dir: {e}"))?;
    let path = dir.join("stage2.jsonl");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| anyhow!("open {}: {e}", path.display()))?;
    let line = serde_json::to_string(entry).map_err(|e| anyhow!("serialize: {e}"))?;
    writeln!(f, "{line}").map_err(|e| anyhow!("write {}: {e}", path.display()))?;
    Ok(())
}

fn load_episode_records(
    store: &Store,
    source: Option<&str>,
    instance: Option<&str>,
) -> Result<Vec<anamnesis_core::AnamnesisRecord>> {
    use anamnesis_core::model::{Kind, RecordId};
    const PAGE: u32 = 500;
    let mut out = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let (ids, next) = store.list_record_ids_paged(cursor.as_deref(), PAGE)?;
        if ids.is_empty() {
            break;
        }
        for id_str in &ids {
            let rid = RecordId(id_str.clone());
            if let Some(record) = store.get_record(&rid)? {
                if record.kind != Kind::Episode {
                    continue;
                }
                if let Some(src) = source {
                    if record.source.adapter != src {
                        continue;
                    }
                }
                if let Some(inst) = instance {
                    if record.source.instance.as_deref() != Some(inst) {
                        continue;
                    }
                }
                out.push(record);
            }
        }
        cursor = next;
        if cursor.is_none() {
            break;
        }
    }
    Ok(out)
}

fn preview(s: &str, max_chars: usize) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    if cleaned.chars().count() <= max_chars {
        return cleaned;
    }
    let head: String = cleaned.chars().take(max_chars).collect();
    format!("{head}…")
}

/// `anamnesis lineage <record-id>` — show the §-1.5 PR-6 derivation chain.
///
/// Default mode walks `provenance.derived_from` upward from the given record
/// until it hits a root (a record with `derived_from = None`) or a dangling
/// parent reference. With `--children`, lists *direct* derivations of the
/// record instead — useful right after running the Stage-2 extractor to see
/// which Facts/Preferences/Skills got distilled out of one Episode.
fn cmd_lineage(
    data_dir: &std::path::Path,
    record_id: &str,
    children: bool,
    limit: u32,
    json: bool,
) -> Result<()> {
    use anamnesis_core::model::RecordId;
    let store = Store::open(db_path(data_dir))?;
    let rid = RecordId(record_id.to_string());

    if children {
        let kids = store.list_derivations(&rid, limit)?;
        if json {
            let rows: Vec<_> = kids
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "id": r.id.0,
                        "adapter": r.source.adapter,
                        "instance": r.source.instance,
                        "kind": format!("{:?}", r.kind).to_lowercase(),
                        "scope": format!("{:?}", r.scope).to_lowercase(),
                        "created_at": r.created_at.to_rfc3339(),
                        "content_preview": preview(&r.content, 200),
                    })
                })
                .collect();
            let out = serde_json::json!({
                "parent": record_id,
                "children": rows,
                "count": kids.len(),
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
            return Ok(());
        }
        if kids.is_empty() {
            println!("No records derived from {record_id}.");
            println!("(If you expected results, the §-1.5 PR-6 Stage-2 extractor probably hasn't run on this record yet.)");
            return Ok(());
        }
        println!(
            "Direct derivations of {record_id} ({} record(s)):",
            kids.len()
        );
        println!();
        for (i, r) in kids.iter().enumerate() {
            println!(
                "[{:>2}] {:?}/{:?}  {}  ({})",
                i + 1,
                r.kind,
                r.scope,
                r.source.adapter,
                r.created_at.to_rfc3339(),
            );
            println!("     id={}", r.id.0);
            println!("     {}", preview(&r.content, 200));
            println!();
        }
        return Ok(());
    }

    // Ancestor walk.
    let chain = store.lineage_chain(&rid)?;
    let chain = match chain {
        None => {
            return Err(anyhow!("no record with id {record_id:?} in this store"));
        }
        Some(c) => c,
    };
    if json {
        let nodes: Vec<_> = chain
            .records
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id.0,
                    "adapter": r.source.adapter,
                    "instance": r.source.instance,
                    "kind": format!("{:?}", r.kind).to_lowercase(),
                    "scope": format!("{:?}", r.scope).to_lowercase(),
                    "created_at": r.created_at.to_rfc3339(),
                    "derived_from": r.provenance.derived_from.as_ref().map(|p| p.0.clone()),
                    "content_preview": preview(&r.content, 200),
                })
            })
            .collect();
        let out = serde_json::json!({
            "start": record_id,
            "chain": nodes,
            "depth": chain.records.len(),
            "missing_parent": chain.missing_parent.as_ref().map(|r| r.0.clone()),
            "complete": chain.missing_parent.is_none(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }
    println!(
        "Lineage of {record_id} ({} record(s) in chain, leaf → root):",
        chain.records.len()
    );
    println!();
    for (depth, r) in chain.records.iter().enumerate() {
        let arrow = if depth == 0 { "  " } else { "↑ " };
        println!(
            "{arrow}[{depth}] {:?}/{:?}  {}  ({})",
            r.kind,
            r.scope,
            r.source.adapter,
            r.created_at.to_rfc3339(),
        );
        println!("       id={}", r.id.0);
        println!("       {}", preview(&r.content, 200));
        println!();
    }
    if let Some(missing) = chain.missing_parent {
        println!(
            "⚠ chain ends at a dangling parent reference: {} \
             (the parent record is no longer in the store; lineage is incomplete)",
            missing.0
        );
    } else {
        println!(
            "✓ chain is complete — root is a record with no derived_from \
             (came directly from an upstream adapter)."
        );
    }
    Ok(())
}

async fn run_search(
    store: &Store,
    query: &str,
    filter: &anamnesis_store::SearchFilter,
    limit: u32,
    mode: SearchMode,
    provider: Option<&ProviderHandle>,
) -> Result<Vec<anamnesis_search::RankedChunk>> {
    Ok(
        run_search_traced(store, query, filter, limit, mode, provider)
            .await?
            .hits,
    )
}

/// Round 76: traced variant of `run_search`. Returns both the
/// ranked chunks and the per-stage timing/count breakdown so the
/// CLI `--trace` flag can render the same shape MCP's
/// `search_memories(trace=true)` returns. The plain `run_search`
/// is a `.hits`-projection wrapper around this — same single-path
/// discipline the search crate uses (avoid trace-vs-live drift).
async fn run_search_traced(
    store: &Store,
    query: &str,
    filter: &anamnesis_store::SearchFilter,
    limit: u32,
    mode: SearchMode,
    provider: Option<&ProviderHandle>,
) -> Result<anamnesis_search::TracedSearchResult> {
    // Round 136 (PR-78be): central candidate-pool policy. Replaces
    // the historic `limit * 4` heuristic with a clamped 8× scale
    // (floor 64, ceil 512) so tiny-limit queries don't starve
    // recall and huge-limit queries don't burn ANN cycles on
    // candidates the post-rerank top-K won't use.
    let opts = HybridOpts::for_limit(limit, mode);
    match provider {
        Some(handle) => match handle {
            #[cfg(feature = "local-fastembed")]
            ProviderHandle::Local(p) => Ok(HybridSearcher::new(p.as_ref())
                .search_filtered_traced(store, query, filter, &opts)
                .await?),
            ProviderHandle::None => Ok(HybridSearcher::<DummyProvider>::fulltext_only()
                .search_filtered_traced(store, query, filter, &opts)
                .await?),
        },
        None => Ok(HybridSearcher::<DummyProvider>::fulltext_only()
            .search_filtered_traced(store, query, filter, &opts)
            .await?),
    }
}

/// Type-erased provider handle so the CLI can branch on feature flags.
enum ProviderHandle {
    #[cfg(feature = "local-fastembed")]
    Local(Box<anamnesis_embedder::LocalFastembedProvider>),
    /// No provider available — present only so `match` has a None arm
    /// usable at compile time without fastembed.
    #[allow(dead_code)]
    None,
}

/// Dummy provider for fulltext_only generics — never actually instantiated.
struct DummyProvider;
#[async_trait::async_trait]
impl anamnesis_core::EmbeddingProvider for DummyProvider {
    fn model_id(&self) -> anamnesis_core::ModelId {
        anamnesis_core::ModelId::new("dummy", "dummy", 1)
    }
    fn dim(&self) -> u16 {
        1
    }
    async fn embed_batch(
        &self,
        _texts: &[&str],
        _task: anamnesis_core::EmbeddingTask,
    ) -> anamnesis_core::Result<Vec<Vec<f32>>> {
        Err(anamnesis_core::Error::Other("dummy provider".into()))
    }
}

#[cfg(feature = "local-fastembed")]
fn open_active_provider(data_dir: &std::path::Path, store: &Store) -> Result<ProviderHandle> {
    let active = store.active_model()?.ok_or_else(|| {
        anyhow!(
            "no active embedding model set; run `anamnesis init` or `anamnesis model use <key>`"
        )
    })?;
    let key = parse_model_key(&active)?;
    let provider = anamnesis_embedder::LocalFastembedProvider::new(&key, models_dir(data_dir))
        .map_err(|e| anyhow!("open local embedding model {key}: {e}"))?;
    Ok(ProviderHandle::Local(Box::new(provider)))
}

#[cfg(not(feature = "local-fastembed"))]
fn open_active_provider(_data_dir: &std::path::Path, _store: &Store) -> Result<ProviderHandle> {
    Err(anyhow!(
        "this anamnesis build lacks `local-fastembed` feature; rebuild with `--features local-fastembed`"
    ))
}

fn parse_model_key(model_id: &str) -> Result<String> {
    // model_id format: "<provider>:<key>:<version>"
    let parts: Vec<&str> = model_id.split(':').collect();
    if parts.len() != 3 {
        return Err(anyhow!("malformed active model id: {model_id}"));
    }
    Ok(parts[1].to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// embedding worker — used after import or model switch
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "local-fastembed")]
async fn run_embed_worker(store: &Store) -> Result<()> {
    let Some(active) = store.active_model()? else {
        return Ok(());
    };
    let key = parse_model_key(&active)?;
    let data_dir_guess = home_join(&[".local", "share", "anamnesis"]).unwrap_or_default();
    // The CLI keeps models under {data_dir}/models — get data_dir from
    // the store path. For simplicity, we always use the standard location;
    // the model is downloaded once and re-used.
    let cache = data_dir_guess.join("models");
    let provider = anamnesis_embedder::LocalFastembedProvider::new(&key, &cache)
        .map_err(|e| anyhow!("open local model for embedding worker: {e}"))?;
    let worker = anamnesis_embedder::EmbeddingWorker::new(&provider);
    let summary = worker
        .drain(store)
        .await
        .map_err(|e| anyhow!("worker drain: {e}"))?;
    println!(
        "embedding worker: {} processed, {} failed",
        summary.processed, summary.failed
    );
    Ok(())
}

#[cfg(not(feature = "local-fastembed"))]
async fn run_embed_worker(_store: &Store) -> Result<()> {
    println!("local-fastembed feature disabled; skipping embedding worker");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// model list / use / install / rebuild
// ─────────────────────────────────────────────────────────────────────────────

async fn cmd_model(data_dir: &std::path::Path, sub: ModelCmd) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let active = store.active_model()?;
    let active_key = active.as_deref().and_then(|id| parse_model_key(id).ok());
    match sub {
        ModelCmd::List => {
            println!(
                "{:<14} {:<8} {:<32} {:<8} {:<6}",
                "key", "active", "hf_id", "size_mb", "dim"
            );
            for m in registry::REGISTRY {
                let marker = if Some(m.key.to_string()) == active_key {
                    "yes"
                } else {
                    ""
                };
                println!(
                    "{:<14} {:<8} {:<32} {:<8} {:<6}",
                    m.key,
                    marker,
                    m.hf_id,
                    if m.approx_size_mb == 0 {
                        "cloud".to_string()
                    } else {
                        format!("{}", m.approx_size_mb)
                    },
                    m.dim
                );
            }
            Ok(())
        }
        ModelCmd::Use { key, no_embed } => {
            let m = registry::by_key(&key).ok_or_else(|| {
                anyhow!(
                    "unknown model key {key:?}; available: {}",
                    registry::available().join(", ")
                )
            })?;
            if !m.is_local {
                return Err(anyhow!(
                    "model {key:?} is a cloud provider; not supported in Phase 1"
                ));
            }
            let new_id = format!("local:{}:{}", m.key, 1);
            store.set_active_model(&new_id)?;
            let n = store.rebuild_embedding_jobs(&new_id)?;
            println!("active model now: {new_id} ({n} chunks re-queued)");
            if !no_embed && n > 0 {
                let store = Store::open(db_path(data_dir))?;
                run_embed_worker(&store).await?;
            }
            Ok(())
        }
        ModelCmd::Install { key } => {
            let m = registry::by_key(&key).ok_or_else(|| anyhow!("unknown model key {key:?}"))?;
            if !m.is_local {
                return Err(anyhow!(
                    "model {key:?} is a cloud provider; nothing to download"
                ));
            }
            install_model(data_dir, &key)?;
            println!(
                "installed model {key} to {}",
                models_dir(data_dir).display()
            );
            Ok(())
        }
        ModelCmd::Rebuild { no_embed } => {
            let active = store
                .active_model()?
                .ok_or_else(|| anyhow!("no active embedding model set"))?;
            let n = store.rebuild_embedding_jobs(&active)?;
            println!("re-queued {n} chunks under {active}");
            if !no_embed && n > 0 {
                let store = Store::open(db_path(data_dir))?;
                run_embed_worker(&store).await?;
            }
            Ok(())
        }
    }
}

#[cfg(feature = "local-fastembed")]
fn install_model(data_dir: &std::path::Path, key: &str) -> Result<()> {
    // Constructing the provider triggers a download into models_dir.
    let _ = anamnesis_embedder::LocalFastembedProvider::new(key, models_dir(data_dir))
        .map_err(|e| anyhow!("install {key}: {e}"))?;
    Ok(())
}

#[cfg(not(feature = "local-fastembed"))]
fn install_model(_data_dir: &std::path::Path, _key: &str) -> Result<()> {
    Err(anyhow!(
        "this build lacks `local-fastembed`; rebuild with `--features local-fastembed`"
    ))
}

#[cfg(test)]
mod freshness_tests {
    use super::{build_mcp_server_entry, human_age_short, source_freshness};

    #[test]
    fn freshness_never_imported_when_last_import_is_none() {
        let f = source_freshness(None, 1_000_000);
        assert_eq!(f.label, "never-imported");
        assert!(f.age_seconds.is_none());
        assert_eq!(f.age_human, "<never>");
    }

    #[test]
    fn freshness_fresh_within_24h() {
        let now = 1_000_000_i64;
        // Just imported.
        let f = source_freshness(Some(now), now);
        assert_eq!(f.label, "fresh");
        assert_eq!(f.age_seconds, Some(0));
        assert_eq!(f.age_human, "<1m");

        // 23h59m ago → still fresh.
        let f = source_freshness(Some(now - (24 * 3600 - 1)), now);
        assert_eq!(f.label, "fresh");
    }

    #[test]
    fn freshness_stale_after_24h_boundary() {
        let now = 1_000_000_i64;
        // Exactly 24h → boundary lands on stale.
        let f = source_freshness(Some(now - 24 * 3600), now);
        assert_eq!(f.label, "stale");
        assert_eq!(f.age_seconds, Some(24 * 3600));
        assert_eq!(f.age_human, "1d");

        // 7 days ago.
        let f = source_freshness(Some(now - 7 * 24 * 3600), now);
        assert_eq!(f.label, "stale");
        assert_eq!(f.age_human, "7d");
    }

    #[test]
    fn freshness_clamps_negative_age_to_zero() {
        // A clock skew that makes `last_import_at > now` must not
        // underflow or report negative age. Round to "<1m"/fresh.
        let f = source_freshness(Some(2_000_000), 1_000_000);
        assert_eq!(f.label, "fresh");
        assert_eq!(f.age_seconds, Some(0));
        assert_eq!(f.age_human, "<1m");
    }

    #[test]
    fn human_age_short_buckets() {
        assert_eq!(human_age_short(0), "<1m");
        assert_eq!(human_age_short(30), "<1m");
        assert_eq!(human_age_short(59), "<1m");
        assert_eq!(human_age_short(60), "1m");
        assert_eq!(human_age_short(125), "2m");
        assert_eq!(human_age_short(3600), "1h");
        assert_eq!(human_age_short(7200), "2h");
        assert_eq!(human_age_short(24 * 3600), "1d");
        assert_eq!(human_age_short(7 * 24 * 3600), "7d");
        assert_eq!(human_age_short(29 * 24 * 3600), "29d");
        assert_eq!(human_age_short(30 * 24 * 3600), "30d+");
        assert_eq!(human_age_short(365 * 24 * 3600), "30d+");
    }

    // ─── Round-55: `anamnesis mcp config` ───────────────────────────────

    /// Stdio default is the shape Claude Desktop / Cursor / Continue /
    /// Windsurf all consume — a `command` + `args` pair.
    #[test]
    fn mcp_config_stdio_default_shape() {
        let server = build_mcp_server_entry(
            "stdio",
            None,
            "ANAMNESIS_MCP_TOKEN",
            Some(std::path::Path::new("/usr/local/bin/anamnesis")),
        )
        .unwrap();
        assert_eq!(server["command"], "/usr/local/bin/anamnesis");
        assert_eq!(server["args"], serde_json::json!(["serve"]));
        // Stdio mode must NOT emit url/headers — those are SSE-only and
        // confuse strict host parsers (e.g. Claude Desktop rejects mixed
        // command+url entries).
        assert!(server.get("url").is_none());
        assert!(server.get("headers").is_none());
    }

    /// SSE mode emits a URL + bearer-token header. The token env-var
    /// name is interpolated via the `${env:NAME}` placeholder host
    /// clients resolve at request time — the value never lands on disk.
    #[test]
    fn mcp_config_sse_emits_bearer_env_placeholder() {
        let server = build_mcp_server_entry(
            "sse",
            Some(7878),
            "MY_TOKEN_VAR",
            Some(std::path::Path::new("/usr/local/bin/anamnesis")),
        )
        .unwrap();
        assert_eq!(server["url"], "http://127.0.0.1:7878/mcp");
        assert_eq!(
            server["headers"]["Authorization"],
            "Bearer ${env:MY_TOKEN_VAR}"
        );
        // No command/args for SSE — that's the discriminator hosts use
        // to pick the transport.
        assert!(server.get("command").is_none());
        assert!(server.get("args").is_none());
    }

    /// SSE without --sse-port is the one user-facing error this command
    /// can produce. Must surface a hint pointing at `anamnesis serve --sse`.
    #[test]
    fn mcp_config_sse_without_port_errors_clearly() {
        let err = build_mcp_server_entry(
            "sse",
            None,
            "ANAMNESIS_MCP_TOKEN",
            Some(std::path::Path::new("/usr/local/bin/anamnesis")),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("--sse-port"),
            "error must name the flag: {err}"
        );
        assert!(
            err.contains("serve --sse"),
            "error should point at the matching `serve` flag: {err}"
        );
    }

    /// Unknown transport must reject — don't emit a half-built config.
    #[test]
    fn mcp_config_unknown_transport_errors() {
        let err = build_mcp_server_entry(
            "websocket",
            None,
            "ANAMNESIS_MCP_TOKEN",
            Some(std::path::Path::new("/usr/local/bin/anamnesis")),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("unknown --transport"),
            "unhelpful error: {err}"
        );
    }

    /// Binary override is honored verbatim — `current_exe()` is only
    /// the fallback, and operators packaging anamnesis under a custom
    /// path (e.g. a Nix store) need to be able to pin it.
    #[test]
    fn mcp_config_honors_binary_override() {
        let server = build_mcp_server_entry(
            "stdio",
            None,
            "ANAMNESIS_MCP_TOKEN",
            Some(std::path::Path::new(
                "/nix/store/abcd-anamnesis/bin/anamnesis",
            )),
        )
        .unwrap();
        assert_eq!(server["command"], "/nix/store/abcd-anamnesis/bin/anamnesis");
    }
}
