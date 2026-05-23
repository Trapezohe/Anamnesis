//! The Anamnesis MCP server — dispatcher + tool/resource handlers.
//!
//! Per BLUEPRINT §6.3 we expose:
//!
//! Tools:
//!   search_memories(query, source?, kind?, scope?, limit?, mode?)
//!   get_record(id)
//!   list_sources()
//!   import_source(adapter, instance?, path?, dry_run?)
//!   trace_provenance(id)
//!   doctor(source?, instance?, since?)
//!
//! Resources:
//!   anamnesis://record/{id}
//!   anamnesis://source/{adapter}[:instance]
//!   anamnesis://timeline/{YYYY-MM-DD}

use std::path::PathBuf;

use anamnesis_adapter_claude_code::{ClaudeCodeAdapter, ClaudeCodeConfig};
use anamnesis_adapter_codex::codex_adapter;
use anamnesis_adapter_generic_mcp::generic_mcp_adapter;
use anamnesis_adapter_hermes::hermes_adapter;
use anamnesis_adapter_letta::letta_adapter;
use anamnesis_adapter_mem0::sqlite_adapter as mem0_sqlite_adapter;
use anamnesis_adapter_memary::memary_adapter;
use anamnesis_adapter_memori::memori_adapter;
use anamnesis_adapter_memos::memos_adapter;
use anamnesis_adapter_mempalace::mempalace_adapter;
use anamnesis_adapter_openclaw::openclaw_adapter;
use anamnesis_adapter_openviking::openviking_adapter;
use anamnesis_adapter_tdai::tdai_adapter;
use anamnesis_core::embedding::EmbeddingProvider;
use anamnesis_core::model::RecordId;
use anamnesis_importer::{ImportOptions, ImportService};
use anamnesis_search::{pack, ContextBudget, HybridOpts, HybridSearcher, SearchMode};
use anamnesis_store::{McpRequestMetric, Store};
use serde_json::{json, Value};

use crate::protocol::{JsonRpcRequest, JsonRpcResponse};

/// Server protocol version we report on initialize.
pub const SERVER_NAME: &str = "anamnesis";
/// Spec version we target — clients should validate compatibility.
pub const PROTOCOL_VERSION: &str = "2025-03-26";

/// Set of admin-only tool names. These mutate state, touch arbitrary
/// filesystem paths, or otherwise stray outside read-only memory access.
/// Hidden from `tools/list` and rejected by `tools/call` unless
/// `AnamnesisServer::allow_admin_tools` is true. See BLUEPRINT §17.5 PR-A.
pub const ADMIN_TOOLS: &[&str] = &[
    "import_source",
    "forget_record",
    "unforget_record",
    "list_forgotten",
    "tag_record",
    // Round 84 (PR-78f): `audit_tail` is read-only but the
    // entries it surfaces carry search queries, forget reasons,
    // source locations — non-admin agents shouldn't be able to
    // back-door read those.
    "audit_tail",
    // Round 86 (PR-78h): `source_show` includes `recent_import_errors`
    // which carry `native_path` + adapter-side error text (e.g.
    // file paths, parse traces). Same gating logic as audit_tail.
    // `list_sources` (no per-row error detail) stays non-admin.
    "source_show",
];

/// Was this tool tagged as admin?
fn is_admin_tool(name: &str) -> bool {
    ADMIN_TOOLS.contains(&name)
}

/// Render a `ForgottenRecord` into the MCP wire shape used by
/// `forget_record`. Outcome is `"forgotten"` or `"already-forgotten"`.
fn forget_payload(outcome: &str, r: anamnesis_store::ForgottenRecord) -> Value {
    json!({
        "outcome":      outcome,
        "record_id":    r.record_id.0,
        "adapter":      r.adapter,
        "instance":     if r.instance.is_empty() { Value::Null } else { Value::String(r.instance) },
        "native_id":    r.native_id,
        "native_path":  r.native_path,
        "raw_hash":     r.raw_hash,
        "reason":       r.reason,
        "forgotten_at": r.forgotten_at,
    })
}

/// Round 105 (PR-78aa): CSV-shape rules shared by every MCP
/// CSV renderer (audit_tail in R92, list_forgotten in R105).
/// Quote + double inner-quote when the field contains `,`,
/// `"`, or `\n`; otherwise the raw value passes through.
fn csv_escape(s: &str) -> String {
    if s.chars().any(|c| c == ',' || c == '"' || c == '\n') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

/// Round 105 (PR-78aa): render `Store::list_forgotten` rows as
/// the MCP CSV string. Same columns and redaction discipline
/// as the CLI R105 helper —
/// `record_id,adapter,instance,native_id,forgotten_at,
/// has_reason,has_native_path` only, never `reason` /
/// `native_path` / `raw_hash`. Empty rows still emit the
/// header so scripts can branch uniformly. `forgotten_at` is
/// rendered as ISO-8601 UTC to match audit_tail's CSV format.
fn render_list_forgotten_csv(rows: &[anamnesis_store::ForgottenRecord]) -> String {
    let mut out = String::from(
        "record_id,adapter,instance,native_id,forgotten_at,has_reason,has_native_path\n",
    );
    for r in rows {
        let at_iso = chrono::DateTime::<chrono::Utc>::from_timestamp(r.forgotten_at, 0)
            .map(|d| d.format("%Y-%m-%dT%H:%M:%SZ").to_string())
            .unwrap_or_else(|| r.forgotten_at.to_string());
        out.push_str(&format!(
            "{rid},{adapter},{instance},{native_id},{at},{has_reason},{has_native_path}\n",
            rid = csv_escape(&r.record_id.0),
            adapter = csv_escape(&r.adapter),
            instance = csv_escape(&r.instance),
            native_id = csv_escape(&r.native_id),
            at = csv_escape(&at_iso),
            has_reason = r.reason.is_some(),
            has_native_path = r.native_path.is_some(),
        ));
    }
    out
}

/// Round 107 (PR-78ac): render dedupe groups as the MCP CSV
/// string. Same columns and redaction discipline as the CLI
/// R107 helper —
/// `group_index,record_id,adapter,instance,native_id,
/// created_at,updated_at,has_native_path,record_count` only,
/// never `raw_hash` / `native_path`. `group_index` carries
/// duplicate-group membership without leaking the hash; rows
/// sharing the same index belong to the same group.
/// `record_count` is per-group size, repeated on each row for
/// spreadsheet-friendly downstream filtering. Empty input still
/// emits the header so scripts can branch uniformly. Timestamps
/// render ISO-8601 to match audit_tail + list_forgotten CSV.
fn render_dedupe_csv(groups: &[anamnesis_store::DuplicateRawHashGroup]) -> String {
    let mut out = String::from(
        "group_index,record_id,adapter,instance,native_id,created_at,updated_at,has_native_path,record_count\n",
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
            out.push_str(&format!(
                "{group_index},{rid},{adapter},{instance},{native_id},{created},{updated},{has_native_path},{record_count}\n",
                group_index = gi,
                rid = csv_escape(&r.record_id.0),
                adapter = csv_escape(&r.adapter),
                instance = csv_escape(&r.instance),
                native_id = csv_escape(&r.native_id),
                created = csv_escape(&created_iso),
                updated = csv_escape(&updated_iso),
                has_native_path = r.native_path.is_some(),
                record_count = record_count,
            ));
        }
    }
    out
}

/// Round 92 (PR-78n): render `Audit::tail` results as the
/// MCP CSV string. Same columns and redaction discipline as the
/// CLI helper in R91 — `line_no,timestamp,action,via,outcome`
/// only, never `detail` / `reason` / `query`. Empty rows still
/// emit the header so scripts can branch uniformly. CSV
/// escaping reuses the same simple rule (quote + double inner
/// quotes when the field contains `,`, `"`, or `\n`).
fn render_audit_tail_csv(rows: &[anamnesis_core::AuditTailRow]) -> String {
    let mut out = String::from("line_no,timestamp,action,via,outcome\n");
    for r in rows {
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
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .unwrap_or_default();
        out.push_str(&format!(
            "{line_no},{ts},{action},{via},{outcome}\n",
            line_no = r.line_no,
            ts = csv_escape(&r.entry.timestamp.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
            action = csv_escape(&r.entry.action),
            via = csv_escape(&via),
            outcome = csv_escape(&outcome),
        ));
    }
    out
}

/// Round 98 (PR-78t): render `count_duplicate_raw_hashes_by_source`
/// as the MCP `counts` block for `dedupe { include_counts: true }`.
/// Same shape as the CLI `--include-counts` JSON in R97 — sources
/// share `(adapter, instance)` keys, instance serialises as
/// JSON `null` for the default. Operators can read either surface
/// without per-tool field translation.
fn render_dedupe_counts(c: &anamnesis_store::DuplicateRawHashCounts) -> Value {
    json!({
        "total_groups":  c.total_groups,
        "total_records": c.total_records,
        "by_source": c.by_source.iter().map(|b| json!({
            "adapter": b.adapter,
            "instance": if b.instance.is_empty() {
                Value::Null
            } else {
                Value::String(b.instance.clone())
            },
            "duplicate_record_count": b.duplicate_record_count,
        })).collect::<Vec<_>>(),
    })
}

/// Round 90 (PR-78l): render `count_forgotten_by_source` as
/// the shared `counts` block for MCP `list_forgotten { include_counts: true }`.
/// Same shape as the CLI JSON payload — `total` plus
/// `by_source[]` with each `(adapter, instance, forgotten_count)`.
/// Default instance serialises as JSON `null`.
fn render_forgotten_counts(buckets: &[anamnesis_store::ForgottenSourceCount]) -> Value {
    let total: u64 = buckets.iter().map(|b| b.forgotten_count).sum();
    json!({
        "total": total,
        "by_source": buckets.iter().map(|b| json!({
            "adapter": b.adapter,
            "instance": if b.instance.is_empty() {
                Value::Null
            } else {
                Value::String(b.instance.clone())
            },
            "forgotten_count": b.forgotten_count,
        })).collect::<Vec<_>>(),
    })
}

/// Round 89 (PR-78k): compact text rendering of
/// `RecordScoreExplain` for the `find_related { explain: true }`
/// prompt. Same numeric fields as `render_score_explain` but
/// flattened into a single line — JSON inside a prompt would
/// burn LLM context for no readability gain.
///
/// `best_chunk_stages = None` (e.g. test fixtures that bypass
/// RRF) collapses to just the record-level totals.
fn format_score_explain_for_prompt(e: &anamnesis_search::RecordScoreExplain) -> String {
    let mut parts = vec![
        format!("record_score={:.4}", e.record_score),
        format!("best_chunk_rrf_score={:.4}", e.best_chunk_rrf_score),
        format!("kind_boost={:.4}", e.kind_boost),
    ];
    if let Some(stages) = &e.best_chunk_stages {
        if let Some(fts) = &stages.fts {
            parts.push(format!(
                "fts_rank={} fts_contribution={:.4}",
                fts.rank, fts.rrf_contribution
            ));
        }
        if let Some(vec) = &stages.vector {
            parts.push(format!(
                "vec_rank={} vec_contribution={:.4}",
                vec.rank, vec.rrf_contribution
            ));
        }
        parts.push(format!("rrf_k={:.0}", stages.rrf_k));
    }
    format!("explain: {}", parts.join(", "))
}

/// Round 87 (PR-78i): render `RecordScoreExplain` as the
/// `search_memories({ explain: true })` per-result block. Same
/// shape as the CLI `--explain` payload — they share the
/// `anamnesis-search::RecordScoreExplain` struct so they can't
/// drift.
fn render_score_explain(e: &anamnesis_search::RecordScoreExplain) -> Value {
    let stages = match &e.best_chunk_stages {
        Some(s) => {
            let fts = s.fts.as_ref().map(|st| {
                json!({
                    "rank": st.rank,
                    "raw_score": st.raw_score,
                    "rrf_contribution": st.rrf_contribution,
                })
            });
            let vector = s.vector.as_ref().map(|st| {
                json!({
                    "rank": st.rank,
                    "raw_score": st.raw_score,
                    "rrf_contribution": st.rrf_contribution,
                })
            });
            json!({
                "fts": fts,
                "vector": vector,
                "rrf_k": s.rrf_k,
            })
        }
        None => Value::Null,
    };
    json!({
        "record_score": e.record_score,
        "best_chunk_rrf_score": e.best_chunk_rrf_score,
        "kind_boost": e.kind_boost,
        "stages": stages,
    })
}

/// Round 84 (PR-78f): parse the MCP `audit_tail.since` arg —
/// same grammar as CLI `parse_doctor_since` (Nd / Nh / Nm /
/// bare seconds), returns the wall-clock instant `now - spec`
/// so the caller can compare with `entry.timestamp` directly.
fn parse_audit_since(spec: &str) -> Result<chrono::DateTime<chrono::Utc>, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err("cannot be empty".into());
    }
    let (num_str, mult): (&str, i64) = match spec.chars().last() {
        Some('d') | Some('D') => (&spec[..spec.len() - 1], 86_400),
        Some('h') | Some('H') => (&spec[..spec.len() - 1], 3_600),
        Some('m') | Some('M') => (&spec[..spec.len() - 1], 60),
        _ => (spec, 1),
    };
    let n: i64 = num_str
        .parse()
        .map_err(|_| format!("must be `Nd` / `Nh` / `Nm` / bare seconds; got {spec:?}"))?;
    if n < 0 {
        return Err(format!("must be non-negative; got {spec:?}"));
    }
    Ok(chrono::Utc::now() - chrono::Duration::seconds(n.saturating_mul(mult)))
}

/// Round 83 (PR-78e): MCP wire shape for `forget_record { dry_run: true }`.
/// Mirrors the CLI dry-run JSON: top-level `dry_run: true`,
/// `status: "would-forget"`, the would-be tombstone fields, and
/// `would_delete` / `would_insert` count blocks.
fn forget_dry_run_payload(
    status: &str,
    would_delete: &anamnesis_store::ForgetCascadeCounts,
    tombstone_preview: &anamnesis_store::ForgetTombstonePreview,
) -> Value {
    json!({
        "dry_run":    true,
        "status":     status,
        "record_id":  tombstone_preview.record_id.0,
        "adapter":    tombstone_preview.adapter,
        "instance":   if tombstone_preview.instance.is_empty() {
            Value::Null
        } else {
            Value::String(tombstone_preview.instance.clone())
        },
        "native_id":   tombstone_preview.native_id,
        "native_path": tombstone_preview.native_path,
        "raw_hash":    tombstone_preview.raw_hash,
        "reason":      tombstone_preview.reason,
        "would_delete": {
            "records":           would_delete.records,
            "raw_artifacts":     would_delete.raw_artifacts,
            "record_chunks":     would_delete.record_chunks,
            "chunk_embeddings":  would_delete.chunk_embeddings,
            "embedding_jobs":    would_delete.embedding_jobs,
            "user_record_tags":  would_delete.user_record_tags,
            "vec0_rows":         would_delete.vec0_rows,
        },
        "would_insert": {
            "record_tombstones":  1,
            // 1 audit entry: the real forget would write one,
            // this dry-run did not.
            "audit_log_entries":  1,
        },
    })
}

/// Round 85 (PR-78g): render `Store::lineage_chain` as the MCP
/// `get_record { include_lineage: true }` payload. Each entry is
/// a **summary** — record_id + provenance + classification but
/// NOT the heavy `content` / `metadata` blob, so a deep chain
/// doesn't bloat the get_record response. Agents that want full
/// ancestor content re-call `get_record` for that id.
///
/// `chain[0]` is the leaf (the record the caller asked for);
/// `chain[last]` is the furthest ancestor we could resolve.
/// `complete: false` means `missing_parent` was hit before
/// reaching a real root.
fn build_lineage_payload(chain: &anamnesis_store::LineageChain) -> Value {
    let records: Vec<Value> = chain
        .records
        .iter()
        .map(|r| {
            json!({
                "record_id":    r.id.0,
                "adapter":      r.source.adapter,
                "instance": match &r.source.instance {
                    Some(s) if !s.is_empty() => Value::String(s.clone()),
                    _ => Value::Null,
                },
                "kind":         format!("{:?}", r.kind).to_lowercase(),
                "scope":        format!("{:?}", r.scope).to_lowercase(),
                "derived_from": r.provenance.derived_from.as_ref().map(|p| p.0.clone()),
                "native_id":    r.provenance.native_id,
                "native_path":  r.provenance.native_path,
                "raw_hash":     r.provenance.raw_hash,
                "captured_at":  r.provenance.captured_at.timestamp(),
                "created_at":   r.created_at.timestamp(),
            })
        })
        .collect();
    let start = records
        .first()
        .and_then(|r| r["record_id"].as_str())
        .map(str::to_owned);
    json!({
        "start":          start,
        "depth":          records.len(),
        "complete":       chain.missing_parent.is_none(),
        "missing_parent": chain.missing_parent.as_ref().map(|p| p.0.clone()),
        "chain":          records,
    })
}

/// Round-22 (§-1.5 PR-5): parse an RFC3339 timestamp into the unix
/// seconds form the `SearchFilter` stores. Used by `search_memories`
/// for `since` / `until` parameters. Returns a string-form error so the
/// JSON-RPC layer can wrap it as a `-32602` invalid-params response.
fn parse_rfc3339_to_unix(s: &str) -> std::result::Result<i64, String> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&chrono::Utc).timestamp())
        .map_err(|e| format!("expected RFC3339 (e.g. 2026-04-01T00:00:00Z): {e}"))
}

/// Round-18 (§-1.5 PR-3): read the bearer token for a registered
/// `generic-mcp` source from the operator's environment. The env-var
/// *name* lives in `sources.config_json` (`{"token_env": "..."}`); the
/// value is never stored — see PR-#25 round-17 rationale.
///
/// `Ok(None)` means "no token configured" (the upstream accepts
/// unauthenticated). An empty string or missing env var when a
/// `token_env` *is* registered yields `Err` so a misconfiguration
/// fails fast at the MCP entry point rather than hitting the upstream
/// with no auth.
fn resolve_generic_mcp_token(config_json: Option<&str>) -> Result<Option<String>, String> {
    let Some(raw) = config_json else {
        return Ok(None);
    };
    let parsed: Value =
        serde_json::from_str(raw).map_err(|e| format!("source.config_json invalid JSON: {e}"))?;
    let Some(env) = parsed.get("token_env").and_then(|v| v.as_str()) else {
        return Ok(None);
    };
    match std::env::var(env) {
        Ok(v) if !v.is_empty() => Ok(Some(v)),
        Ok(_) => Err(format!(
            "generic-mcp source's token_env={env:?} is set but empty"
        )),
        Err(_) => Err(format!(
            "generic-mcp source requires env var {env:?} to be set"
        )),
    }
}

/// Stateful MCP server wrapping a `Store` and (optionally) an
/// `EmbeddingProvider`. Built once and reused across all incoming
/// requests on any transport (stdio, SSE/HTTP).
pub struct AnamnesisServer {
    store: Store,
    provider: Option<Box<dyn EmbeddingProvider>>,
    /// Data directory — handlers like trace_provenance and import_source
    /// need it to resolve relative paths.
    pub data_dir: PathBuf,
    /// HOME override — same role as in CLI; lets tests stub paths.
    pub home_override: Option<PathBuf>,
    /// Expose admin tools (currently `import_source`) over MCP.
    /// Defaults to `false` — see `ADMIN_TOOLS` for the list.
    allow_admin_tools: bool,
}

impl AnamnesisServer {
    /// Build a new server wrapping the given store + optional provider.
    /// Admin tools are OFF by default; enable with `with_admin_tools(true)`
    /// only when the server is reachable solely by trusted clients.
    pub fn new(
        store: Store,
        provider: Option<Box<dyn EmbeddingProvider>>,
        data_dir: PathBuf,
    ) -> Self {
        Self {
            store,
            provider,
            data_dir,
            home_override: None,
            allow_admin_tools: false,
        }
    }

    /// Replace HOME for filesystem-dependent handlers (tests).
    pub fn with_home(mut self, home: PathBuf) -> Self {
        self.home_override = Some(home);
        self
    }

    /// Enable or disable admin tools (currently `import_source`).
    pub fn with_admin_tools(mut self, allow: bool) -> Self {
        self.allow_admin_tools = allow;
        self
    }

    /// Read the current admin flag — useful for diagnostics and tests.
    pub fn admin_tools_allowed(&self) -> bool {
        self.allow_admin_tools
    }

    fn home(&self) -> PathBuf {
        self.home_override
            .clone()
            .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("/"))
    }

    /// Dispatch one parsed request. Notifications still go through here
    /// (they get a synthetic null-id response that the caller can drop).
    pub async fn handle(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.clone().unwrap_or(Value::Null);
        match req.method.as_str() {
            "initialize" => JsonRpcResponse::ok(id, self.initialize_result()),
            "notifications/initialized" => JsonRpcResponse::ok(id, Value::Null),
            "ping" => JsonRpcResponse::ok(id, json!({})),
            "tools/list" => JsonRpcResponse::ok(id, self.tools_list_payload()),
            "tools/call" => self.handle_tools_call(id, req.params).await,
            "resources/list" => self.handle_resources_list(id, req.params).await,
            "resources/read" => self.handle_resources_read(id, req.params).await,
            "prompts/list" => JsonRpcResponse::ok(id, prompts_list_payload()),
            "prompts/get" => self.handle_prompts_get(id, req.params).await,
            other => JsonRpcResponse::err(id, -32601, format!("method not found: {other}")),
        }
    }

    fn initialize_result(&self) -> Value {
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {
                "name": SERVER_NAME,
                "version": env!("CARGO_PKG_VERSION"),
            },
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {},
            },
        })
    }

    async fn handle_tools_call(&self, id: Value, params: Value) -> JsonRpcResponse {
        let name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => {
                self.record_metric_safely(None, "unknown", false, 0, None, Some("missing_name"));
                return JsonRpcResponse::err(id, -32602, "missing tools/call.name");
            }
        };
        // BLUEPRINT §17.5 PR-A — the load-bearing check.
        if is_admin_tool(&name) && !self.allow_admin_tools {
            self.record_metric_safely(None, &name, false, 0, None, Some("admin_disabled"));
            return JsonRpcResponse::err(
                id,
                -32601,
                format!(
                    "tool {name:?} is an admin tool and disabled on this server \
                     (set [server.mcp] allow_admin_tools = true in config to enable; \
                     prefer running this operation via the CLI instead)"
                ),
            );
        }
        let args = params.get("arguments").cloned().unwrap_or(Value::Null);
        let started_at = chrono::Utc::now().timestamp();
        let t0 = std::time::Instant::now();
        let result = match name.as_str() {
            "search_memories" => self.tool_search_memories(args.clone()).await,
            "get_record" => self.tool_get_record(args.clone()).await,
            "list_sources" => self.tool_list_sources(args.clone()).await,
            "import_source" => self.tool_import_source(args.clone()).await,
            "trace_provenance" => self.tool_trace_provenance(args.clone()).await,
            "doctor" => self.tool_doctor(args.clone()).await,
            "forget_record" => self.tool_forget_record(args.clone()).await,
            "unforget_record" => self.tool_unforget_record(args.clone()).await,
            "list_forgotten" => self.tool_list_forgotten(args.clone()).await,
            "dedupe" => self.tool_dedupe(args.clone()).await,
            "tag_record" => self.tool_tag_record(args.clone()).await,
            "audit_tail" => self.tool_audit_tail(args.clone()).await,
            "source_show" => self.tool_source_show(args.clone()).await,
            other => {
                self.record_metric_safely(
                    Some(started_at),
                    &name,
                    false,
                    t0.elapsed().as_millis() as i64,
                    None,
                    Some("unknown_tool"),
                );
                return JsonRpcResponse::err(id, -32602, format!("unknown tool: {other}"));
            }
        };
        let duration_ms = t0.elapsed().as_millis() as i64;
        match result {
            Ok(payload) => {
                let result_count = if name == "search_memories" {
                    payload
                        .get("results")
                        .and_then(|v| v.as_array())
                        .map(|a| a.len() as i64)
                } else {
                    None
                };
                self.record_metric(
                    started_at,
                    &name,
                    true,
                    duration_ms,
                    result_count,
                    None,
                    &args,
                );
                JsonRpcResponse::ok(
                    id,
                    json!({
                        "content": [{"type": "text", "text": payload.to_string()}],
                        "structuredContent": payload,
                    }),
                )
            }
            Err(msg) => {
                self.record_metric(
                    started_at,
                    &name,
                    false,
                    duration_ms,
                    None,
                    Some("tool_error"),
                    &args,
                );
                JsonRpcResponse::err(id, -32603, msg)
            }
        }
    }

    /// Persist one MCP request metric. Writes happen *after* the
    /// response payload is built, so metric I/O cannot widen the
    /// latency the client observes. Failures are warned-and-swallowed:
    /// observability must not break MCP request handling.
    ///
    /// Extracts `mode` / `source` / `instance` / `limit` from the
    /// caller's arguments when the tool is `search_memories` — these
    /// are non-PII structured fields the caller already disclosed by
    /// passing them. Query text and raw args are never stored.
    #[allow(clippy::too_many_arguments)]
    fn record_metric(
        &self,
        started_at: i64,
        tool: &str,
        ok: bool,
        duration_ms: i64,
        result_count: Option<i64>,
        error_kind: Option<&str>,
        args: &Value,
    ) {
        let (mode, source, instance, limit_value) = if tool == "search_memories" {
            (
                args.get("mode").and_then(|v| v.as_str()).map(str::to_owned),
                args.get("source")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned),
                args.get("instance")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned),
                args.get("limit").and_then(|v| v.as_i64()),
            )
        } else {
            (None, None, None, None)
        };
        let metric = McpRequestMetric {
            started_at,
            tool: tool.to_string(),
            ok,
            duration_ms,
            result_count,
            error_kind: error_kind.map(str::to_owned),
            mode,
            source,
            instance,
            limit_value,
        };
        if let Err(e) = self.store.record_mcp_request_metric(&metric) {
            tracing::warn!(error = ?e, tool, "failed to record MCP request metric");
        }
    }

    /// Variant for the dispatch-gate paths where we don't have args
    /// to mine and don't want to allocate empty options. Equivalent
    /// to `record_metric` with all arg-derived fields `None`.
    fn record_metric_safely(
        &self,
        started_at: Option<i64>,
        tool: &str,
        ok: bool,
        duration_ms: i64,
        result_count: Option<i64>,
        error_kind: Option<&str>,
    ) {
        self.record_metric(
            started_at.unwrap_or_else(|| chrono::Utc::now().timestamp()),
            tool,
            ok,
            duration_ms,
            result_count,
            error_kind,
            &Value::Null,
        );
    }

    async fn handle_prompts_get(&self, id: Value, params: Value) -> JsonRpcResponse {
        let name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => return JsonRpcResponse::err(id, -32602, "missing prompts/get.name"),
        };
        let args = params.get("arguments").cloned().unwrap_or(Value::Null);
        let result = match name.as_str() {
            "summarize_my_preferences" => self.prompt_summarize_preferences(args).await,
            "find_related" => self.prompt_find_related(args).await,
            other => return JsonRpcResponse::err(id, -32602, format!("unknown prompt: {other}")),
        };
        match result {
            Ok(payload) => JsonRpcResponse::ok(id, payload),
            Err(msg) => JsonRpcResponse::err(id, -32603, msg),
        }
    }

    async fn handle_resources_read(&self, id: Value, params: Value) -> JsonRpcResponse {
        let uri = match params.get("uri").and_then(|v| v.as_str()) {
            Some(u) => u.to_string(),
            None => return JsonRpcResponse::err(id, -32602, "missing resources/read.uri"),
        };
        match self.read_resource(&uri).await {
            Ok(payload) => JsonRpcResponse::ok(
                id,
                json!({
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": payload.to_string(),
                    }],
                }),
            ),
            Err(msg) => JsonRpcResponse::err(id, -32603, msg),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool implementations
// ─────────────────────────────────────────────────────────────────────────────

impl AnamnesisServer {
    async fn tool_search_memories(&self, args: Value) -> Result<Value, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "search_memories.query is required".to_string())?;
        let source = args
            .get("source")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let instance = args
            .get("instance")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let kind = args
            .get("kind")
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase());
        let scope = args
            .get("scope")
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase());
        // Round-22 (§-1.5 PR-5): time-window filters. The store layer
        // already supports `time_from` / `time_to` via SQL pushdown
        // (PR-C); this just plumbs them through from the MCP wire so
        // an agent can ask "what did I learn in the last week".
        let time_from = match args.get("since").and_then(|v| v.as_str()) {
            Some(s) => Some(parse_rfc3339_to_unix(s).map_err(|e| format!("since: {e}"))?),
            None => None,
        };
        let time_to = match args.get("until").and_then(|v| v.as_str()) {
            Some(s) => Some(parse_rfc3339_to_unix(s).map_err(|e| format!("until: {e}"))?),
            None => None,
        };
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(10);
        let mode = match args
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("hybrid")
        {
            "fulltext" => SearchMode::Fulltext,
            "vector" => SearchMode::Vector,
            _ => SearchMode::Hybrid,
        };
        // Round-71: opt-in per-stage breakdown. Default off — when
        // omitted the response shape is byte-identical to pre-R71.
        let trace_requested = args.get("trace").and_then(|v| v.as_bool()).unwrap_or(false);
        // Round 87 (PR-78i): opt-in per-hit score breakdown
        // (record_score / best_chunk_rrf_score / kind_boost / FTS
        // + vector stage ranks + contributions). Orthogonal to
        // `trace`; default off keeps the result-row shape stable.
        let explain_requested = args
            .get("explain")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Round 79 (PR-78b): `--user-tag` filter on the read path.
        // Normalised through the shared `normalize_user_tag_name`
        // so `Keep` written by `tag_record` matches `keep` from
        // the search wire.
        let user_tag = match args.get("user_tag").and_then(|v| v.as_str()) {
            Some(raw) => Some(
                anamnesis_store::normalize_user_tag_name(raw)
                    .map_err(|e| format!("user_tag: {e}"))?,
            ),
            None => None,
        };

        // PR-C: push every filter into the SQL recall stage so a
        // minority-source query (e.g. `source = "mem0"` against a
        // claude-code-dominated corpus) is not silently emptied by
        // post-RRF filtering.
        let filter = anamnesis_store::SearchFilter {
            source: source.clone(),
            instance,
            kind,
            scope,
            time_from,
            time_to,
            user_tag,
        };

        let store = &self.store;
        let opts = HybridOpts {
            limit,
            candidate_pool: (limit * 4).max(limit),
            mode,
        };
        // Always go through the traced primitive so the live search
        // and the trace can never drift. The `trace` field on the
        // response is only included when the caller asked — keeping
        // the default wire shape additive-compatible.
        let traced = match self.provider.as_ref() {
            Some(p) => HybridSearcher::new(p.as_ref())
                .search_filtered_traced(store, query, &filter, &opts)
                .await
                .map_err(|e| format!("search: {e}"))?,
            None => HybridSearcher::<NoProvider>::fulltext_only()
                .search_filtered_traced(store, query, &filter, &opts.fulltext_fallback())
                .await
                .map_err(|e| format!("search: {e}"))?,
        };
        let hits = traced.hits;
        let search_trace = traced.trace;

        let t_pack = std::time::Instant::now();
        let packed = pack(
            store,
            &hits,
            &ContextBudget {
                max_records: limit as usize,
                ..ContextBudget::default()
            },
        )
        .map_err(|e| format!("pack: {e}"))?;
        let pack_ms = t_pack.elapsed().as_millis() as u64;
        // Post-filter is now a defense-in-depth no-op: the SQL stage
        // already excluded non-matching adapters from the candidate
        // pool. We keep it so adding new filter dimensions to the MCP
        // surface stays a one-line change here.
        let filtered: Vec<_> = if let Some(src) = source.as_deref() {
            packed
                .into_iter()
                .filter(|p| p.record.source.adapter == src)
                .collect()
        } else {
            packed
        };
        // Round-8 wire format. Each result carries enough for an
        // MCP agent to:
        //   - chain into `trace_provenance` (use `trace_id` = `record_id`)
        //   - understand *why* a hit surfaced (`from_fts` / `from_vec`,
        //     plus the per-modality raw scores)
        //   - sort / threshold / filter on the client side (`rrf_score`,
        //     `fts_score`, `vector_score`)
        //   - render time-aware UI (`created_at`, `updated_at`)
        //
        // `score` is kept as an alias for `rrf_score` so older agents
        // that pinned the previous field name don't break.
        let mut payload = json!({
            "results": filtered.iter().map(|p| {
                let best = p.matched_chunks.first();
                let mut row = json!({
                    "record_id": p.record.id.0,
                    // Alias the record id as `trace_id` so an agent that
                    // already holds a result can call
                    // `trace_provenance({"id": trace_id})` without
                    // remembering the field-mapping convention.
                    "trace_id": p.record.id.0,
                    "chunk_id": best.map(|c| c.chunk_id.clone()),
                    "adapter": p.record.source.adapter,
                    "instance": p.record.source.instance,
                    "kind": format!("{:?}", p.record.kind).to_lowercase(),
                    "scope": format!("{:?}", p.record.scope).to_lowercase(),
                    // Score breakdown.
                    "score": p.score,            // back-compat alias
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
                if explain_requested {
                    row["explain"] = render_score_explain(&p.score_explain());
                }
                row
            }).collect::<Vec<_>>()
        });
        if trace_requested {
            // Inject the search-layer trace + MCP-layer pack_ms +
            // packed-record counts. Everything here is numeric or the
            // effective mode string; no query text, no snippets, no
            // record/chunk ids, no path strings — same privacy
            // contract as R69's `mcp_request_metrics`.
            let returned_records = payload["results"].as_array().map_or(0, |a| a.len()) as u32;
            payload["trace"] = json!({
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
        Ok(payload)
    }

    async fn tool_get_record(&self, args: Value) -> Result<Value, String> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "get_record.id is required".to_string())?;
        // Round 85 (PR-78g): optional provenance walk. Default
        // false so the wire shape stays back-compatible for every
        // existing MCP agent — they only opt in when they actually
        // want lineage.
        let include_lineage = args
            .get("include_lineage")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let store = &self.store;
        let rec = store
            .get_record(&RecordId(id.to_string()))
            .map_err(|e| format!("store: {e}"))?;
        let Some(r) = rec else {
            // Missing record → null. The same shape `search_memories`
            // would emit for a no-hit query, so agents can branch
            // uniformly.
            return Ok(Value::Null);
        };

        // Round-11: emit a normalised, agent-decision-ready payload
        // instead of the raw serde of `AnamnesisRecord`. Same naming
        // convention as the `search_memories` wire format (PR-#16):
        // lower-case enums, JSON null for absent instance, surface
        // chunk/embedding readiness so the agent can decide whether
        // hybrid retrieval will hit this record right now.
        let summary = store
            .record_summary(&r.id)
            .map_err(|e| format!("store: {e}"))?
            .ok_or_else(|| "record vanished between lookup and summary".to_string())?;
        // Round 78: user-tag overlay. Always returned (empty vec
        // when no user tags) so agents can branch on absence
        // uniformly. Read is NOT admin-gated — only writes are.
        let user_tags = store.user_tags(&r.id).map_err(|e| format!("store: {e}"))?;

        // Round 85 (PR-78g): leaf-to-root provenance chain via
        // R74's existing `LineageChain`. Each chain entry is a
        // *summary* (no `content`, no `metadata`) — agents that
        // want full ancestor content re-call `get_record` for that
        // id. This keeps the get_record payload bounded even when
        // an extractor produced a deep chain. `chain[0]` is the
        // record the caller asked for; `chain[last]` is the
        // furthest ancestor we could resolve.
        let lineage_payload: Option<Value> = if include_lineage {
            let chain = store
                .lineage_chain(&r.id)
                .map_err(|e| format!("get_record.include_lineage: {e}"))?
                .ok_or_else(|| "lineage: record vanished between lookup and walk".to_string())?;
            Some(build_lineage_payload(&chain))
        } else {
            None
        };

        Ok(json!({
            "record_id": r.id.0,
            // trace_id alias mirrors the search wire format — agents
            // can pass it straight back to `trace_provenance`.
            "trace_id": r.id.0,
            "adapter": r.source.adapter,
            // Default-instance records serialise their `instance` as
            // JSON null (the SQL stores "" but that's an implementation
            // detail; PR-#17 fixed the same thing for `list_sources`).
            "instance": match &r.source.instance {
                Some(s) if !s.is_empty() => Value::String(s.clone()),
                _ => Value::Null,
            },
            "kind": format!("{:?}", r.kind).to_lowercase(),
            "scope": format!("{:?}", r.scope).to_lowercase(),
            "content": r.content,
            "tags": r.tags,
            "metadata": r.metadata,
            "native_id": r.provenance.native_id,
            "native_path": r.provenance.native_path,
            "raw_hash": r.provenance.raw_hash,
            "captured_at": r.provenance.captured_at.timestamp(),
            "created_at": r.created_at.timestamp(),
            "updated_at": r.updated_at.map(|t| t.timestamp()),
            "schema_version": r.schema_version,
            // Readiness — the load-bearing piece. An agent that wants
            // to ensure vector retrieval will hit this record can
            // assert `chunk_count == embedded_chunk_count` and
            // `active_model` matches expectations.
            "chunk_count": summary.chunk_count,
            "embedded_chunk_count": summary.embedded_chunk_count,
            "active_model": summary.active_model,
            // Source-vector breadcrumb only. We do NOT return the
            // vector itself — source embeddings are provenance, not
            // retrieval (BLUEPRINT §6.6.1).
            "source_embedding_model": summary.source_embedding_model,
            "source_embedding_dim": summary.source_embedding_dim,
            // Round 78: user-tag overlay. Distinct from `tags`
            // (which is adapter-derived and gets overwritten on
            // re-import). Empty array is the common case.
            "user_tags": user_tags,
            // Round 85: provenance walk. Only present when the
            // caller passed `include_lineage: true`. Default
            // omission keeps the wire shape back-compatible.
            "lineage": lineage_payload,
        }))
    }

    async fn tool_list_sources(&self, args: Value) -> Result<Value, String> {
        let store = &self.store;
        // Round 96 (PR-78r): optional `source` + `instance`
        // filter narrows the `sources[]` array only — the
        // top-level `stats` block still reflects the whole
        // store so existing R0 clients reading `stats.records`
        // see the same values they always have.
        //
        // Round 103 (PR-78y): `source` now also accepts a
        // comma-separated OR list (`"mem0,claude-code"`) via
        // core's shared `parse_csv_filter`, symmetric with R102
        // audit-tail multi-value. Empty parse = no filter
        // (back-compat with R96). `instance` stays single-value
        // AND-combined with the adapter set.
        let source_raw = args.get("source").and_then(|v| v.as_str());
        let sources = anamnesis_core::parse_csv_filter(source_raw);
        let instance_filter = args
            .get("instance")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty());

        let stats = store.stats().map_err(|e| format!("stats: {e}"))?;
        // Round-9: per-source counts + last_import_at let an agent
        // distinguish "bad retrieval" from "stale source" without a
        // second round trip. LEFT JOIN means registered-but-empty
        // sources still appear (record_count=0) — which is the signal
        // the agent needs to detect a misconfigured adapter.
        let rows = store
            .list_sources_with_counts()
            .map_err(|e| format!("list: {e}"))?;
        let filtered: Vec<&anamnesis_store::SourceWithCounts> = rows
            .iter()
            .filter(|r| sources.is_empty() || sources.iter().any(|s| s == &r.source.adapter))
            .filter(|r| match instance_filter {
                Some(i) => r.source.instance == i,
                None => true,
            })
            .collect();
        Ok(json!({
            "sources": filtered.iter().map(|r| json!({
                "adapter": r.source.adapter,
                "instance": if r.source.instance.is_empty() {
                    Value::Null
                } else {
                    Value::String(r.source.instance.clone())
                },
                "location": r.source.location,
                "added_at": r.source.added_at,
                "last_import_at": r.source.last_import_at,
                "record_count": r.record_count,
                "chunk_count": r.chunk_count,
                // Round-82 PR-78d: distinct-record count for the
                // user-tag overlay. Lets an MCP agent answer
                // "where do my keep-forever records live" without
                // a second call. NOT the number of tag rows.
                "tagged_record_count": r.tagged_record_count,
            })).collect::<Vec<_>>(),
            "active_model": store.active_model().ok().flatten(),
            "stats": {
                "records": stats.records,
                "chunks": stats.chunks,
                "jobs_pending": stats.jobs_pending,
                "jobs_failed": stats.jobs_failed,
            }
        }))
    }

    /// Round-18 (§-1.5 PR-3): MCP admin import now flows through the
    /// shared `ImportService` so the system-state delta (registry +
    /// `last_import_at` + `audit.log`) matches the CLI `import` path
    /// exactly.
    ///
    /// Wire schema (after PR-3):
    ///   `{ adapter: string, instance?: string, dry_run?: bool }`
    ///
    /// Removed: `path` (and the never-implemented `url`). Allowing
    /// MCP clients to supply arbitrary filesystem paths bypassed the
    /// `source add` registry and let any admin-tools-enabled client
    /// read any path the server process could read — that was the
    /// §-1.2.2 / §-1.6.8 boundary we tightened in this PR. To import
    /// a new location, register it with CLI `anamnesis source add`
    /// first, then call `import_source` over MCP.
    ///
    /// Skips the embedding worker (CLI does it; MCP must not block the
    /// JSON-RPC request on minutes of embedding work — codex round-18
    /// trap #2). The next CLI run or scheduled embed sweep will pick up
    /// the freshly-imported chunks.
    async fn tool_import_source(&self, args: Value) -> Result<Value, String> {
        let adapter_id = args
            .get("adapter")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "import_source.adapter is required".to_string())?;
        let instance = args.get("instance").and_then(|v| v.as_str());
        let dry_run = args
            .get("dry_run")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Tighten the boundary: MCP must not be a path-arbitrary import.
        if args.get("path").is_some() || args.get("url").is_some() {
            return Err("MCP import_source no longer accepts `path` or `url`; \
                 register the source with `anamnesis source add` first, \
                 then call this tool with just `adapter` (+ optional `instance`)."
                .into());
        }

        // Look up the registry; refuse to invent a location.
        let registered = self
            .store
            .get_source(adapter_id, instance)
            .map_err(|e| format!("store: {e}"))?
            .ok_or_else(|| {
                format!(
                    "source {adapter_id}{} is not registered; use CLI \
                     `anamnesis source add {adapter_id}{} ...` first.",
                    instance.map(|i| format!(":{i}")).unwrap_or_default(),
                    instance
                        .map(|i| format!(" --instance {i}"))
                        .unwrap_or_default(),
                )
            })?;

        let service = ImportService::new(&self.store, anamnesis_core::Audit::new(&self.data_dir));
        let opts = ImportOptions {
            dry_run,
            // The CLI fills `canonical_location` from --path / --url so
            // a fresh import can re-anchor the registry. MCP never gets
            // a fresh location (we just refused `path` and `url`), so we
            // leave the registry's value untouched.
            canonical_location: None,
            source_was_explicit: true,
            // Round-19 (§-1.5 PR-4a): MCP admin import currently always
            // runs as a full scan. Surfacing `since` over MCP is the
            // §-1.5 PR-5 work; until then ScanOpts::default() preserves
            // the pre-PR-4 behavior exactly.
            ..Default::default()
        };

        let summary = match adapter_id {
            anamnesis_adapter_claude_code::ADAPTER_ID => {
                let projects_root = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".claude").join("projects"));
                let adapter = ClaudeCodeAdapter::new(ClaudeCodeConfig {
                    projects_root,
                    instance: instance.map(str::to_owned),
                });
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_mem0::ADAPTER_ID => {
                let db_path = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".mem0").join("db.sqlite"));
                let adapter = mem0_sqlite_adapter(db_path, instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_codex::ADAPTER_ID => {
                let root = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".codex"));
                let adapter = codex_adapter(root, instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_letta::ADAPTER_ID => {
                let db_path = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".letta").join("letta.db"));
                let adapter = letta_adapter(db_path, instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_hermes::ADAPTER_ID => {
                let data_dir = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".hermes"));
                let adapter = hermes_adapter(data_dir, instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_openclaw::ADAPTER_ID => {
                let data_dir = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".openclaw"));
                let adapter = openclaw_adapter(data_dir, instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_tdai::ADAPTER_ID => {
                let data_dir = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".openclaw").join("memory-tdai"));
                let adapter = tdai_adapter(data_dir, instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_openviking::ADAPTER_ID => {
                let workspace_dir = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".openviking").join("data"));
                let adapter = openviking_adapter(workspace_dir, instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_mempalace::ADAPTER_ID => {
                let home_dir = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".mempalace"));
                let adapter = mempalace_adapter(home_dir, instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_memori::ADAPTER_ID => {
                let db_path = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".memori").join("memori.db"));
                let adapter = memori_adapter(db_path, instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_memos::ADAPTER_ID => {
                let root_dir = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".memos"));
                let adapter = memos_adapter(root_dir, instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_memary::ADAPTER_ID => {
                let data_dir = registered
                    .location
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.home().join(".memary").join("data"));
                let adapter = memary_adapter(data_dir, instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            anamnesis_adapter_generic_mcp::ADAPTER_ID => {
                let url = registered.location.as_deref().ok_or_else(|| {
                    "generic-mcp source has no URL in the registry; \
                     run `anamnesis source add generic-mcp --url ...` first"
                        .to_string()
                })?;
                let token = resolve_generic_mcp_token(registered.config_json.as_deref())
                    .map_err(|e| format!("token: {e}"))?;
                let adapter = generic_mcp_adapter(url.to_string(), token.as_deref(), instance);
                service
                    .import(&adapter, opts)
                    .await
                    .map_err(|e| format!("import: {e}"))?
            }
            other => return Err(format!("unknown adapter: {other}")),
        };

        serde_json::to_value(summary).map_err(|e| e.to_string())
    }

    async fn tool_trace_provenance(&self, args: Value) -> Result<Value, String> {
        let store = &self.store;

        // Round-10: accept EITHER `id` (= record_id) OR `chunk_id`.
        // `search_memories` now returns `chunk_id` in every hit (PR-#16
        // wire format), so an agent that wants to ask "what does this
        // exact matched chunk look like in the raw source?" can chain
        // directly without first resolving record → chunk.
        let chunk_id_arg = args.get("chunk_id").and_then(|v| v.as_str());
        let record_id_arg = args.get("id").and_then(|v| v.as_str());

        let (rec, chunk_extras) = match (chunk_id_arg, record_id_arg) {
            (Some(cid), _) => {
                let lookup = store
                    .get_chunk(cid)
                    .map_err(|e| format!("store: {e}"))?
                    .ok_or_else(|| format!("chunk not found: {cid}"))?;
                let rec = store
                    .get_record(&lookup.record_id)
                    .map_err(|e| format!("store: {e}"))?
                    .ok_or_else(|| {
                        format!(
                            "chunk {} found but parent record {} missing — db corruption?",
                            cid, lookup.record_id.0
                        )
                    })?;
                let tokenized = anamnesis_store::cjk::tokenize_indexing(&lookup.content);
                let extras = json!({
                    "chunk_id": lookup.chunk_id,
                    "chunk_seq": lookup.seq,
                    "chunk_content": lookup.content,
                    "chunk_content_tokenized": tokenized,
                    "chunk_token_estimate": lookup.token_estimate,
                });
                (rec, Some(extras))
            }
            (None, Some(id)) => {
                let rec = store
                    .get_record(&RecordId(id.to_string()))
                    .map_err(|e| format!("store: {e}"))?
                    .ok_or_else(|| format!("record not found: {id}"))?;
                (rec, None)
            }
            (None, None) => {
                return Err(
                    "trace_provenance requires either `id` (record_id) or `chunk_id`".into(),
                );
            }
        };

        // Base provenance — same shape as before for back-compat.
        let mut out = json!({
            "record_id": rec.id.0,
            "adapter": rec.source.adapter,
            "instance": rec.source.instance,
            "native_id": rec.provenance.native_id,
            "native_path": rec.provenance.native_path,
            "captured_at": rec.provenance.captured_at,
            "raw_hash": rec.provenance.raw_hash,
        });

        // Chunk-level extras only when the caller asked by chunk_id.
        if let (Some(obj), Some(extras)) = (out.as_object_mut(), chunk_extras) {
            if let Some(extra_obj) = extras.as_object() {
                for (k, v) in extra_obj {
                    obj.insert(k.clone(), v.clone());
                }
            }
        }
        Ok(out)
    }

    /// Round-54: expose `anamnesis doctor` over MCP.
    ///
    /// Per-source health check — for each registered source, runs the
    /// adapter's `health()` probe (cheap: e.g. directory exists for
    /// file-backed adapters, GET `/healthz` for generic-mcp) and joins
    /// against the store's per-source counts. Agents call this when
    /// "search_memories returned nothing relevant" so they can tell the
    /// user "your mem0 source is unreachable" instead of silently
    /// continuing with stale assumptions.
    ///
    /// Wire shape:
    ///   request:  `{ source?: string, instance?: string, since?: "Nd"|"Nh"|"Nm"|"<int seconds>" }`
    ///   response: `{ summary: { total, ok, unhealthy, stale }, sources: [...] }`
    ///
    /// Not admin-gated — `list_sources` already exposes locations; this
    /// tool just adds liveness + staleness signal. The CLI's
    /// `--include-unregistered` mode (which runs filesystem detectors
    /// for every adapter) is intentionally NOT exposed here: detectors
    /// walk the user's home dir, and the MCP boundary should not be a
    /// path-arbitrary probe surface (see `import_source` rationale).
    async fn tool_doctor(&self, args: Value) -> Result<Value, String> {
        // Round 110 (PR-78af): `source` now accepts a comma-
        // separated OR list (`"mem0,claude-code"`) via core's
        // shared `parse_csv_filter`, symmetric with R102
        // audit-tail / R103 list-sources / R104 dedupe. Empty
        // parse = no filter (back-compat). `instance` stays
        // single-value AND-combined with the adapter set.
        let filter_source_raw = args.get("source").and_then(|v| v.as_str());
        let sources = anamnesis_core::parse_csv_filter(filter_source_raw);
        let filter_instance = args
            .get("instance")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let stale_threshold = match args.get("since").and_then(|v| v.as_str()) {
            Some(spec) => Some(parse_doctor_since(spec)?),
            None => None,
        };
        // Round-69: `metrics_since` defaults to 24h (86_400 s). The
        // arg accepts the same `"24h" / "7d" / raw seconds` grammar as
        // `since` so users don't have to remember two conventions.
        let metrics_window_secs: i64 = match args.get("metrics_since").and_then(|v| v.as_str()) {
            Some(spec) => parse_doctor_since(spec)?,
            None => 86_400,
        };
        let now = chrono::Utc::now().timestamp();

        let store = &self.store;
        let registered = store
            .list_sources_with_counts()
            .map_err(|e| format!("list: {e}"))?;

        let mut rows = Vec::new();
        for swc in &registered {
            let src = &swc.source;
            if !sources.is_empty() && !sources.iter().any(|s| s == &src.adapter) {
                continue;
            }
            if let Some(inst) = filter_instance.as_deref() {
                if src.instance != inst {
                    continue;
                }
            }
            let health = self.run_adapter_health_for_source(src).await;
            let stale = stale_threshold.map(|t| match src.last_import_at {
                Some(ts) => (now - ts) > t,
                // Never-imported counts as stale once a threshold is set.
                None => true,
            });
            let ok = health.as_ref().map(|h| h.ok).unwrap_or(false);
            let detail = match &health {
                Some(h) => h.detail.clone(),
                None => "adapter not wired into doctor; see `import` dispatch".to_string(),
            };
            rows.push(json!({
                "adapter": src.adapter,
                "instance": if src.instance.is_empty() {
                    Value::Null
                } else {
                    Value::String(src.instance.clone())
                },
                "location": src.location,
                "ok": ok,
                "detail": detail,
                "record_count": swc.record_count,
                "chunk_count": swc.chunk_count,
                "last_import_at": src.last_import_at,
                "stale": stale,
            }));
        }

        let ok_count = rows
            .iter()
            .filter(|r| r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
            .count();
        let bad_count = rows.len() - ok_count;
        let stale_count = rows
            .iter()
            .filter(|r| r.get("stale").and_then(|v| v.as_bool()).unwrap_or(false))
            .count();

        // Round-69: per-tool MCP request latency summary, additive
        // field. Existing `summary` / `sources` shape is unchanged.
        let metric_since_ts = now - metrics_window_secs;
        let tool_metrics_payload = match self
            .store
            .summarize_mcp_request_metrics(Some(metric_since_ts))
        {
            Ok(summaries) => summaries
                .into_iter()
                .map(|s| {
                    json!({
                        "tool": s.tool,
                        "count": s.count,
                        "errors": s.errors,
                        "p50_ms": s.p50_ms,
                        "p95_ms": s.p95_ms,
                        "p99_ms": s.p99_ms,
                        "last_ms": s.last_ms,
                        "last_result_count": s.last_result_count,
                        "last_started_at": s.last_started_at,
                    })
                })
                .collect::<Vec<_>>(),
            Err(e) => {
                tracing::warn!(error = ?e, "doctor: failed to summarize mcp request metrics");
                Vec::new()
            }
        };

        Ok(json!({
            "summary": {
                "total": rows.len(),
                "ok": ok_count,
                "unhealthy": bad_count,
                "stale": stale_count,
            },
            "sources": rows,
            "request_metrics": {
                "window_seconds": metrics_window_secs,
                "tools": tool_metrics_payload,
            },
        }))
    }

    /// Round 73 (PR-72b): MCP-side counterpart to `anamnesis forget`.
    /// Admin-gated — the dispatcher rejects this tool unless
    /// `allow_admin_tools = true` so only operator-trusted clients
    /// can mutate state. Reuses `Store::forget_record` from R72,
    /// which writes the tombstone + cascades cleanup + suppresses
    /// future re-imports.
    ///
    /// `NotFound` is surfaced as a tool error, not success: if no
    /// tombstone was written, Anamnesis cannot guarantee "stay
    /// forgotten" — the caller probably has the wrong id and
    /// should be told loudly.
    async fn tool_forget_record(&self, args: Value) -> Result<Value, String> {
        let record_id = args
            .get("record_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "forget_record.record_id is required".to_string())?;
        let reason = args
            .get("reason")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        // Round 83 (PR-78e): dry_run=true returns the cascade
        // preview without writing a tombstone and without
        // appending an audit entry. Still admin-gated — the
        // preview reveals raw_hash/native_path and destructive
        // intent that should match the real forget's ACL.
        let dry_run = args
            .get("dry_run")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if dry_run {
            let preview = self
                .store
                .preview_forget_record(&RecordId(record_id.to_string()), reason.as_deref())
                .map_err(|e| format!("forget_record: {e}"))?;
            return match preview {
                anamnesis_store::ForgetRecordPreview::WouldForget {
                    would_delete,
                    tombstone_preview,
                } => Ok(forget_dry_run_payload(
                    "would-forget",
                    &would_delete,
                    &tombstone_preview,
                )),
                anamnesis_store::ForgetRecordPreview::AlreadyForgotten(r) => {
                    let mut p = forget_payload("already-forgotten", r);
                    p["dry_run"] = json!(true);
                    Ok(p)
                }
                anamnesis_store::ForgetRecordPreview::NotFound => Err(format!(
                    "forget_record: no record with id {record_id:?} — nothing to forget (dry-run)"
                )),
            };
        }

        let outcome = self
            .store
            .forget_record(&RecordId(record_id.to_string()), reason.as_deref())
            .map_err(|e| format!("forget_record: {e}"))?;

        // Mirror the CLI's audit entry shape so operator history is
        // entrypoint-agnostic (`anamnesis audit list` shows MCP and
        // CLI forgets the same way). Same on-demand `Audit::new` the
        // import path uses — keeps the server struct slim.
        anamnesis_core::Audit::new(&self.data_dir).record(anamnesis_core::AuditEntry::new(
            "forget",
            json!({
                "record_id": record_id,
                "reason":    reason,
                "outcome": match &outcome {
                    anamnesis_store::ForgetRecordOutcome::Forgotten(_)        => "forgotten",
                    anamnesis_store::ForgetRecordOutcome::AlreadyForgotten(_) => "already-forgotten",
                    anamnesis_store::ForgetRecordOutcome::NotFound            => "not-found",
                },
                "via": "mcp",
            }),
        ));

        match outcome {
            anamnesis_store::ForgetRecordOutcome::Forgotten(r) => {
                Ok(forget_payload("forgotten", r))
            }
            anamnesis_store::ForgetRecordOutcome::AlreadyForgotten(r) => {
                Ok(forget_payload("already-forgotten", r))
            }
            anamnesis_store::ForgetRecordOutcome::NotFound => Err(format!(
                "forget_record: no record with id {record_id:?} — nothing to forget"
            )),
        }
    }

    /// Round 75 (PR-75): MCP-side counterpart to `anamnesis unforget`.
    /// Admin-gated. Removes the tombstone so the source can resurrect
    /// the memory on its next `import_source`. Does NOT recreate the
    /// `records` row — same truthful-design constraint as the CLI:
    /// Anamnesis is a read-only mirror of source data.
    ///
    /// `NotForgotten` is a tool error (not a silent success) because
    /// the operator almost certainly typoed an id from
    /// `list_forgotten` — better to surface the mistake than to log
    /// a successful no-op.
    async fn tool_unforget_record(&self, args: Value) -> Result<Value, String> {
        let record_id = args
            .get("record_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "unforget_record.record_id is required".to_string())?;
        // Round 95 (PR-78q): dry_run preview. Doesn't delete the
        // tombstone, doesn't append to audit. Still admin-gated
        // because the response includes raw_hash/native_path/reason.
        let dry_run = args
            .get("dry_run")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if dry_run {
            let preview = self
                .store
                .preview_unforget_record(&RecordId(record_id.to_string()))
                .map_err(|e| format!("unforget_record: {e}"))?;
            return match preview {
                anamnesis_store::UnforgetRecordOutcome::Unforgotten(r) => Ok(json!({
                    "dry_run":             true,
                    "outcome":             "would-unforget",
                    "record_id":           r.record_id.0,
                    "adapter":             r.adapter,
                    "instance":            if r.instance.is_empty() { Value::Null } else { Value::String(r.instance) },
                    "native_id":           r.native_id,
                    "forgotten_at":        r.forgotten_at,
                    "record_resurrected":  false,
                    "requires_reimport":   true,
                    "would_delete":        { "record_tombstones": 1 },
                    "would_insert":        { "audit_log_entries": 1 },
                })),
                anamnesis_store::UnforgetRecordOutcome::NotForgotten => Err(format!(
                    "unforget_record: no tombstone for id {record_id:?} — nothing to unforget (dry-run)"
                )),
            };
        }

        let outcome = self
            .store
            .unforget_record(&RecordId(record_id.to_string()))
            .map_err(|e| format!("unforget_record: {e}"))?;

        anamnesis_core::Audit::new(&self.data_dir).record(anamnesis_core::AuditEntry::new(
            "unforget",
            json!({
                "record_id": record_id,
                "outcome": match &outcome {
                    anamnesis_store::UnforgetRecordOutcome::Unforgotten(_) => "unforgotten",
                    anamnesis_store::UnforgetRecordOutcome::NotForgotten   => "not-forgotten",
                },
                "via": "mcp",
            }),
        ));

        match outcome {
            anamnesis_store::UnforgetRecordOutcome::Unforgotten(r) => Ok(json!({
                "outcome":             "unforgotten",
                "record_id":           r.record_id.0,
                "adapter":             r.adapter,
                "instance":            if r.instance.is_empty() { Value::Null } else { Value::String(r.instance) },
                "native_id":           r.native_id,
                "forgotten_at":        r.forgotten_at,
                "record_resurrected":  false,
                "requires_reimport":   true,
            })),
            anamnesis_store::UnforgetRecordOutcome::NotForgotten => Err(format!(
                "unforget_record: no tombstone for id {record_id:?} — nothing to unforget"
            )),
        }
    }

    /// Round 74 (PR-74): MCP audit view for `record_tombstones`.
    /// Admin-gated and redaction-by-default — `native_path`,
    /// `raw_hash`, `reason` are reported as `has_*` booleans
    /// unless the caller opts in with `include_sensitive=true`.
    /// `limit` is clamped by the store to
    /// `[1, LIST_FORGOTTEN_MAX_LIMIT]`.
    ///
    /// Read-only: never writes the store or audit log.
    async fn tool_list_forgotten(&self, args: Value) -> Result<Value, String> {
        let source = args
            .get("source")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let instance = args
            .get("instance")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(20);
        let include_sensitive = args
            .get("include_sensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Round 90 (PR-78l): opt-in aggregate counts.
        let include_counts = args
            .get("include_counts")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Round 105 (PR-78aa): opt-in CSV form, mirroring R92's
        // `audit_tail.csv`. CSV is the redacted-summary form;
        // we refuse `csv + include_sensitive` and
        // `csv + include_counts` because the operator intent is
        // contradictory — CSV is flat redacted rows by design,
        // so smuggling sensitive fields or attaching a counts
        // block would either leak content or pretend the CSV
        // carried more shape than it does.
        let csv_requested = args.get("csv").and_then(|v| v.as_bool()).unwrap_or(false);
        if csv_requested && include_sensitive {
            return Err(
                "list_forgotten: `csv: true` and `include_sensitive: true` are mutually exclusive \
                 — CSV is the redacted-summary form (never carries `reason` / `native_path` / \
                 `raw_hash`)."
                    .to_string(),
            );
        }
        if csv_requested && include_counts {
            return Err(
                "list_forgotten: `csv: true` and `include_counts: true` are mutually exclusive \
                 — CSV is flat redacted rows. Drop `csv` to get the `counts` block back."
                    .to_string(),
            );
        }

        let filter = anamnesis_store::ListForgottenFilter {
            source: source.clone(),
            instance: instance.clone(),
            limit,
        };
        let rows = self
            .store
            .list_forgotten(&filter)
            .map_err(|e| format!("list_forgotten: {e}"))?;
        let counts = if include_counts {
            Some(
                self.store
                    .count_forgotten_by_source(&filter)
                    .map_err(|e| format!("list_forgotten counts: {e}"))?,
            )
        } else {
            None
        };

        let effective_limit = limit.clamp(1, anamnesis_store::LIST_FORGOTTEN_MAX_LIMIT);

        if csv_requested {
            // CSV path returns a single flat string instead of
            // `rows[]`. Header + redacted summary columns
            // mirror the CLI R105 output exactly so a scripted
            // consumer can switch between transports without
            // reparsing.
            let csv = render_list_forgotten_csv(&rows);
            return Ok(json!({
                "count":              rows.len(),
                "limit":              effective_limit,
                "format":             "csv",
                "sensitive_included": false,
                "filter":             {
                    "source":   source,
                    "instance": instance,
                },
                "csv":                csv,
            }));
        }

        let rows_payload: Vec<Value> = rows
            .iter()
            .map(|r| {
                let mut row = json!({
                    "record_id": r.record_id.0,
                    "adapter":   r.adapter,
                    "instance":  if r.instance.is_empty() { Value::Null } else { Value::String(r.instance.clone()) },
                    "native_id": r.native_id,
                    "forgotten_at":    r.forgotten_at,
                    "has_reason":      r.reason.is_some(),
                    "has_native_path": r.native_path.is_some(),
                });
                if include_sensitive {
                    row["reason"]      = json!(r.reason);
                    row["native_path"] = json!(r.native_path);
                    row["raw_hash"]    = json!(r.raw_hash);
                }
                row
            })
            .collect();
        // Round 109 (PR-78ae): `"format": "json"` marker
        // pairs with R105's `"format": "csv"` on the CSV
        // branch (`return Ok(json!({...}))` above). MCP
        // clients can switch on `payload.format` instead of
        // probing for `rows[]` vs `csv`. Completes the trio
        // started in R108: dedupe, list_forgotten, audit_tail
        // all now carry symmetric format markers on both
        // branches.
        let mut payload = json!({
            "count":              rows.len(),
            "format":             "json",
            "limit":              effective_limit,
            "sensitive_included": include_sensitive,
            "rows":               rows_payload,
        });
        if let Some(buckets) = &counts {
            payload["counts"] = render_forgotten_counts(buckets);
        }
        Ok(payload)
    }

    /// Round 77 (PR-77): MCP audit view for raw-hash duplicates.
    /// NOT admin-gated — this is a read-only diagnostic with the
    /// same redaction discipline as `list_forgotten`. The action
    /// half (`forget_record`) still requires admin.
    async fn tool_dedupe(&self, args: Value) -> Result<Value, String> {
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(20);
        let include_sensitive = args
            .get("include_sensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Round 80: scope duplicates to groups containing
        // ≥1 record from a given source/instance. Empty
        // strings normalised to None so the caller can pass
        // `""` without accidentally filtering on the empty
        // instance (which is a real value for single-instance
        // adapters).
        let source = args
            .get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        let instance = args
            .get("instance")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        // Round 98 (PR-78t): opt-in filter-scoped aggregate.
        // Symmetric with R97's CLI `--include-counts`. Counts
        // ignore `limit` and reflect the full matching set; per-
        // source breakdown counts records, not group memberships.
        let include_counts = args
            .get("include_counts")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Round 107 (PR-78ac): opt-in CSV form, mirroring R92
        // audit_tail.csv + R105 list_forgotten.csv. CSV is the
        // redacted-summary form by design; we refuse
        // `csv + include_sensitive` (would leak raw_hash /
        // native_path) and `csv + include_counts` (CSV is flat,
        // no nested counts block).
        let csv_requested = args.get("csv").and_then(|v| v.as_bool()).unwrap_or(false);
        if csv_requested && include_sensitive {
            return Err(
                "dedupe: `csv: true` and `include_sensitive: true` are mutually exclusive — \
                 CSV is the redacted-summary form (never carries `raw_hash` / `native_path`)."
                    .to_string(),
            );
        }
        if csv_requested && include_counts {
            return Err(
                "dedupe: `csv: true` and `include_counts: true` are mutually exclusive — \
                 CSV is flat redacted rows. Drop `csv` to get the `counts` block back."
                    .to_string(),
            );
        }

        let filter = anamnesis_store::DuplicateRawHashFilter {
            source: source.clone(),
            instance: instance.clone(),
            limit,
        };
        let groups = self
            .store
            .list_duplicate_raw_hashes_filtered(&filter)
            .map_err(|e| format!("dedupe: {e}"))?;
        let counts = if include_counts {
            Some(
                self.store
                    .count_duplicate_raw_hashes_by_source(&filter)
                    .map_err(|e| format!("dedupe counts: {e}"))?,
            )
        } else {
            None
        };
        let effective_limit = limit.clamp(1, anamnesis_store::LIST_DUPLICATE_RAW_HASHES_MAX_LIMIT);

        if csv_requested {
            // CSV path returns a flat string, not `groups[]`.
            // Same redacted-summary discipline as the CLI R107
            // helper: `group_index` (not `raw_hash`) carries
            // membership; `has_native_path` boolean (not the
            // path itself).
            let csv = render_dedupe_csv(&groups);
            return Ok(json!({
                "count":              groups.len(),
                "limit":              effective_limit,
                "format":             "csv",
                "sensitive_included": false,
                "filter":             {
                    "source":   source,
                    "instance": instance,
                },
                "csv":                csv,
            }));
        }

        let payload_groups: Vec<Value> = groups
            .iter()
            .map(|g| {
                let mut group = json!({
                    "record_count": g.records.len(),
                    "records": g.records.iter().map(|r| {
                        let mut row = json!({
                            "record_id":       r.record_id.0,
                            "adapter":         r.adapter,
                            "instance":        if r.instance.is_empty() { Value::Null } else { Value::String(r.instance.clone()) },
                            "native_id":       r.native_id,
                            "created_at":      r.created_at,
                            "updated_at":      r.updated_at,
                            "has_native_path": r.native_path.is_some(),
                        });
                        if include_sensitive {
                            row["native_path"] = json!(r.native_path);
                        }
                        row
                    }).collect::<Vec<_>>(),
                });
                if include_sensitive {
                    group["raw_hash"] = json!(g.raw_hash);
                }
                group
            })
            .collect();
        // Round 108 (PR-78ad): `"format": "json"` marker
        // pairs with R107's `"format": "csv"` on the CSV
        // branch. MCP clients can switch on `payload.format`
        // without probing for `csv` vs `groups[]`. The CSV
        // branch already returned early above, so reaching
        // here means the structured form.
        let mut payload = json!({
            "count":              groups.len(),
            "format":             "json",
            "limit":              effective_limit,
            "sensitive_included": include_sensitive,
            "filter": {
                "source":   source,
                "instance": instance,
            },
            "groups":             payload_groups,
        });
        if let Some(c) = &counts {
            payload["counts"] = render_dedupe_counts(c);
        }
        Ok(payload)
    }

    /// Round 78 (PR-78): MCP-side `tag_record`. Admin-gated
    /// because it mutates local state. Reads (`user_tags`
    /// surfaced in `search_memories` / `get_record`) are NOT
    /// admin-gated. Set semantics — re-adding is a no-op.
    /// Same audit-log shape as the CLI.
    async fn tool_tag_record(&self, args: Value) -> Result<Value, String> {
        let record_id = args
            .get("record_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "tag_record.record_id is required".to_string())?;
        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "tag_record.tags must be an array".to_string())?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        let operation = match args
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("add")
        {
            "add" => anamnesis_store::UserTagOperation::Add,
            "remove" => anamnesis_store::UserTagOperation::Remove,
            // Round 81 PR-78c: atomic full-set replace.
            "replace" => anamnesis_store::UserTagOperation::Replace,
            other => {
                return Err(format!(
                    "tag_record.operation must be \"add\", \"remove\", or \"replace\"; \
                     got {other:?}"
                ))
            }
        };
        let op_label = match operation {
            anamnesis_store::UserTagOperation::Add => "add",
            anamnesis_store::UserTagOperation::Remove => "remove",
            anamnesis_store::UserTagOperation::Replace => "replace",
        };
        // Round 96 (PR-78r): opt-in stats block — `total_user_tags`
        // after the mutation. Default off; existing R78 / R81
        // consumers see the same wire shape.
        let include_stats = args
            .get("include_stats")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mutation = self
            .store
            .tag_record(&RecordId(record_id.to_string()), &tags, operation)
            .map_err(|e| format!("tag_record: {e}"))?;

        anamnesis_core::Audit::new(&self.data_dir).record(anamnesis_core::AuditEntry::new(
            "tag_record",
            json!({
                "record_id": record_id,
                "operation": op_label,
                "requested": mutation.requested,
                "changed":   mutation.changed,
                "via":       "mcp",
            }),
        ));

        let mut payload = json!({
            "record_id": mutation.record_id.0,
            "operation": op_label,
            "requested": mutation.requested,
            "changed":   mutation.changed,
            "user_tags": mutation.user_tags,
        });
        if include_stats {
            payload["stats"] = json!({
                "total_user_tags": mutation.user_tags.len(),
            });
        }
        Ok(payload)
    }

    /// Round 84 (PR-78f): MCP-side `audit_tail`. Admin-gated
    /// (the entries can carry search queries / forget reasons /
    /// source locations); even then, the default response shape
    /// is the **redacted summary** — `line_no`, `timestamp`,
    /// `action`, `via`, `outcome`/`status` only. The full
    /// per-entry `detail` payload is opt-in via
    /// `include_detail: true` so an MCP agent doesn't
    /// accidentally feed user-typed search text back into its
    /// own context.
    async fn tool_audit_tail(&self, args: Value) -> Result<Value, String> {
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        let since_spec = args
            .get("since")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty());
        let since_dt: Option<chrono::DateTime<chrono::Utc>> = match since_spec {
            Some(spec) => {
                Some(parse_audit_since(spec).map_err(|e| format!("audit_tail.since: {e}"))?)
            }
            None => None,
        };
        let include_detail = args
            .get("include_detail")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Round 92 (PR-78n): MCP-side parity with R91's CLI
        // `audit tail --csv`. CSV is the redacted-summary form
        // (no `detail`); we refuse `csv + include_detail` because
        // the operator intent is contradictory — mixing the two
        // would either smuggle raw `reason` / `query` into a CSV
        // export or pretend the CSV was full-detail.
        let csv_requested = args.get("csv").and_then(|v| v.as_bool()).unwrap_or(false);
        if csv_requested && include_detail {
            return Err(
                "audit_tail: `csv: true` and `include_detail: true` are mutually exclusive — \
                 CSV is the redacted-summary form (no `detail`)."
                    .to_string(),
            );
        }

        // Round 102 (PR-78x): comma-separated `action` becomes
        // a multi-value OR filter (`["forget", "search"]`).
        // Parsing lives in core's `parse_audit_actions` so CLI
        // + MCP share the split rule byte-for-byte. Response
        // keeps the existing `filter.action` raw string for
        // back-compat with R84/R91 clients and adds an additive
        // `filter.actions` array of normalised tokens.
        let actions = anamnesis_core::parse_audit_actions(action.as_deref());
        let opts = anamnesis_core::AuditTailOptions {
            limit,
            since: since_dt,
            actions: actions.clone(),
        };
        let audit = anamnesis_core::Audit::new(&self.data_dir);
        let rows = audit
            .tail(&opts)
            .map_err(|e| format!("audit_tail: read audit.log: {e}"))?;
        let effective_limit = limit
            .unwrap_or(anamnesis_core::AUDIT_TAIL_DEFAULT_LIMIT)
            .clamp(1, anamnesis_core::AUDIT_TAIL_MAX_LIMIT);

        if csv_requested {
            // CSV path returns a single `csv` string instead of
            // an `entries[]` array. Same redacted summary fields
            // CLI uses (`line_no,timestamp,action,via,outcome`),
            // never `detail` / `reason` / `query`. Empty result
            // still emits the header so downstream scripts can
            // branch uniformly.
            let csv = render_audit_tail_csv(&rows);
            return Ok(json!({
                "count":           rows.len(),
                "limit":           effective_limit,
                "format":          "csv",
                "include_detail":  false,
                "filter": {
                    "action":  action,
                    "actions": actions,
                    "since":   since_spec,
                },
                "csv": csv,
            }));
        }

        let entries: Vec<Value> = rows
            .iter()
            .map(|r| {
                let mut row = json!({
                    "line_no":   r.line_no,
                    "timestamp": r.entry.timestamp,
                    "action":    r.entry.action,
                    "via":       r.entry.detail.get("via").cloned().unwrap_or(Value::Null),
                    "outcome": r
                        .entry
                        .detail
                        .get("outcome")
                        .or_else(|| r.entry.detail.get("status"))
                        .cloned()
                        .unwrap_or(Value::Null),
                });
                if include_detail {
                    row["detail"] = r.entry.detail.clone();
                }
                row
            })
            .collect();

        // Round 109 (PR-78ae): `"format": "json"` marker
        // pairs with R92's `"format": "csv"` on the CSV
        // branch above. Same symmetry as R108 (dedupe) and
        // R109 list_forgotten — MCP clients can branch on
        // `payload.format` without probing for `entries[]`
        // vs `csv`.
        Ok(json!({
            "count":           rows.len(),
            "format":          "json",
            "limit":           effective_limit,
            "include_detail":  include_detail,
            "filter": {
                "action":  action,
                "actions": actions,
                "since":   since_spec,
            },
            "entries": entries,
        }))
    }

    /// Round 86 (PR-78h): MCP-side `source_show`. Admin-gated
    /// because `recent_import_errors` rows carry `native_path` +
    /// adapter-side error text, same sensitivity contract as
    /// `audit_tail`. The non-admin `list_sources` already
    /// surfaces the counts (without per-row error detail) so a
    /// read-only agent doesn't lose anything by being held out
    /// of this one.
    ///
    /// Wire shape:
    ///   * `source`: same fields as one row of `list_sources` plus
    ///     `tagged_record_count` (R82).
    ///   * `recent_import_errors`: newest-first, capped at the
    ///     caller's `error_limit` (default 5, hard cap 10 — kept
    ///     small so the response stays bounded even on a noisy
    ///     adapter).
    ///   * Missing `(adapter, instance)` → `Err(format!(..))`,
    ///     not `null`, so MCP clients exit on typo'd ids.
    async fn tool_source_show(&self, args: Value) -> Result<Value, String> {
        const ERROR_LIMIT_DEFAULT: usize = 5;
        const ERROR_LIMIT_MAX: usize = 10;
        let adapter = args
            .get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "source_show.source is required".to_string())?;
        let instance = args
            .get("instance")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty());
        let error_limit = args
            .get("error_limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(ERROR_LIMIT_DEFAULT)
            .clamp(1, ERROR_LIMIT_MAX);

        let swc = self
            .store
            .get_source_with_counts(adapter, instance)
            .map_err(|e| format!("source_show: {e}"))?
            .ok_or_else(|| {
                let target = match instance {
                    Some(i) => format!("{adapter}:{i}"),
                    None => adapter.to_string(),
                };
                format!("source_show: source not found: {target}")
            })?;
        let recent = self
            .store
            .recent_import_errors_for_source(adapter, instance, error_limit)
            .map_err(|e| format!("source_show: {e}"))?;

        let recent_payload: Vec<Value> = recent
            .iter()
            .map(|e| {
                json!({
                    "adapter": e.adapter,
                    "instance": if e.instance.is_empty() {
                        Value::Null
                    } else {
                        Value::String(e.instance.clone())
                    },
                    "native_id":   e.native_id,
                    "native_path": e.native_path,
                    "phase":       e.phase,
                    "error":       e.error,
                    "occurred_at": e.occurred_at,
                })
            })
            .collect();

        Ok(json!({
            "source": {
                "adapter": swc.source.adapter,
                "instance": if swc.source.instance.is_empty() {
                    Value::Null
                } else {
                    Value::String(swc.source.instance.clone())
                },
                "location":             swc.source.location,
                "added_at":             swc.source.added_at,
                "last_import_at":       swc.source.last_import_at,
                "record_count":         swc.record_count,
                "chunk_count":          swc.chunk_count,
                "tagged_record_count":  swc.tagged_record_count,
            },
            "recent_import_errors": recent_payload,
            "error_limit": error_limit,
        }))
    }

    /// Build the right adapter for a registered source and call
    /// `MemoryAdapter::health().await`. Mirrors the CLI's
    /// `run_adapter_health` — the dispatch table stays in lockstep with
    /// `tool_import_source` so a source that can be imported can also
    /// be doctored. Returns `None` if the adapter id is unknown.
    async fn run_adapter_health_for_source(
        &self,
        src: &anamnesis_store::SourceRow,
    ) -> Option<anamnesis_core::adapter::HealthStatus> {
        use anamnesis_core::adapter::MemoryAdapter;
        let location_path = src.location.as_deref().map(PathBuf::from);
        let instance = if src.instance.is_empty() {
            None
        } else {
            Some(src.instance.as_str())
        };

        match src.adapter.as_str() {
            anamnesis_adapter_claude_code::ADAPTER_ID => {
                let path =
                    location_path.unwrap_or_else(|| self.home().join(".claude").join("projects"));
                let adapter = ClaudeCodeAdapter::new(ClaudeCodeConfig {
                    projects_root: path,
                    instance: instance.map(str::to_owned),
                });
                Some(adapter.health().await)
            }
            anamnesis_adapter_codex::ADAPTER_ID => {
                let path = location_path.unwrap_or_else(|| self.home().join(".codex"));
                Some(codex_adapter(path, instance).health().await)
            }
            anamnesis_adapter_mem0::ADAPTER_ID => {
                let path =
                    location_path.unwrap_or_else(|| self.home().join(".mem0").join("db.sqlite"));
                Some(mem0_sqlite_adapter(path, instance).health().await)
            }
            anamnesis_adapter_letta::ADAPTER_ID => {
                let path =
                    location_path.unwrap_or_else(|| self.home().join(".letta").join("letta.db"));
                Some(letta_adapter(path, instance).health().await)
            }
            anamnesis_adapter_hermes::ADAPTER_ID => {
                let path = location_path.unwrap_or_else(|| self.home().join(".hermes"));
                Some(hermes_adapter(path, instance).health().await)
            }
            anamnesis_adapter_openclaw::ADAPTER_ID => {
                let path = location_path.unwrap_or_else(|| self.home().join(".openclaw"));
                Some(openclaw_adapter(path, instance).health().await)
            }
            anamnesis_adapter_tdai::ADAPTER_ID => {
                let path = location_path
                    .unwrap_or_else(|| self.home().join(".openclaw").join("memory-tdai"));
                Some(tdai_adapter(path, instance).health().await)
            }
            anamnesis_adapter_openviking::ADAPTER_ID => {
                let path =
                    location_path.unwrap_or_else(|| self.home().join(".openviking").join("data"));
                Some(openviking_adapter(path, instance).health().await)
            }
            anamnesis_adapter_mempalace::ADAPTER_ID => {
                let path = location_path.unwrap_or_else(|| self.home().join(".mempalace"));
                Some(mempalace_adapter(path, instance).health().await)
            }
            anamnesis_adapter_memori::ADAPTER_ID => {
                let path =
                    location_path.unwrap_or_else(|| self.home().join(".memori").join("memori.db"));
                Some(memori_adapter(path, instance).health().await)
            }
            anamnesis_adapter_memos::ADAPTER_ID => {
                let path = location_path.unwrap_or_else(|| self.home().join(".memos"));
                Some(memos_adapter(path, instance).health().await)
            }
            anamnesis_adapter_memary::ADAPTER_ID => {
                let path =
                    location_path.unwrap_or_else(|| self.home().join(".memary").join("data"));
                Some(memary_adapter(path, instance).health().await)
            }
            anamnesis_adapter_generic_mcp::ADAPTER_ID => {
                // Same logic the CLI uses — read URL from the registry,
                // resolve token env-var, then fire a single `/healthz` GET.
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
                let adapter = generic_mcp_adapter(url, token.as_deref(), instance);
                Some(adapter.health().await)
            }
            _ => None,
        }
    }
}

/// Parse the MCP `doctor.since` argument into seconds. Same shapes the
/// CLI's `--since` accepts (`Nd`, `Nh`, `Nm`, bare integer). Returns a
/// string error so the JSON-RPC layer can wrap it as `-32602`.
fn parse_doctor_since(spec: &str) -> Result<i64, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err("since cannot be empty".to_string());
    }
    let (num_str, mult) = match spec.chars().last() {
        Some('d') | Some('D') => (&spec[..spec.len() - 1], 86_400_i64),
        Some('h') | Some('H') => (&spec[..spec.len() - 1], 3_600_i64),
        Some('m') | Some('M') => (&spec[..spec.len() - 1], 60_i64),
        _ => (spec, 1_i64),
    };
    let n: i64 = num_str
        .parse()
        .map_err(|_| format!("since must be `Nd`/`Nh`/`Nm`/bare seconds; got {spec:?}"))?;
    if n < 0 {
        return Err(format!("since must be non-negative; got {spec:?}"));
    }
    Ok(n.saturating_mul(mult))
}

// ─────────────────────────────────────────────────────────────────────────────
// Prompt implementations (BLUEPRINT §6.3 — convenience prompts)
// ─────────────────────────────────────────────────────────────────────────────

impl AnamnesisServer {
    async fn prompt_summarize_preferences(&self, args: Value) -> Result<Value, String> {
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as i64)
            .unwrap_or(20);
        // Round 94 (PR-78p): symmetric with R93's
        // `find_related.user_tag` — let the operator narrow the
        // preference summary to records carrying a specific
        // user tag (e.g. `keep-forever`). Normalised through the
        // same helper `tag_record` writes through, so
        // `Keep-Forever` matches a tag stored as `keep-forever`.
        let user_tag = match args.get("user_tag").and_then(|v| v.as_str()) {
            Some(raw) => Some(
                anamnesis_store::normalize_user_tag_name(raw)
                    .map_err(|e| format!("summarize_my_preferences.user_tag: {e}"))?,
            ),
            None => None,
        };
        let store = &self.store;
        // Filter pushes down at the SQL recall stage (same R79
        // discipline as search): the store helper does the
        // EXISTS subquery before LIMIT so a single tagged
        // record surfaces under a heavy untagged-majority corpus.
        let rows = store
            .summarize_preferences_records(limit, user_tag.as_deref())
            .map_err(|e| format!("summarize_my_preferences query: {e}"))?;

        let mut bullets = String::new();
        for row in &rows {
            bullets.push_str(&format!(
                "- [{kind}] {content_short}  (id={id_short}, source={src})\n",
                kind = row.kind,
                content_short = trim_for_prompt(&row.content, 240),
                id_short = &row.id[..row.id.len().min(12)],
                src = row.native_path.as_deref().unwrap_or("?"),
            ));
        }
        if bullets.is_empty() {
            bullets.push_str("(no user-scope records yet)\n");
        }

        // Round 101 (PR-78w): summary line symmetric with R100
        // `find_related`. The store helper already filtered by
        // `user_tag` at the SQL recall stage, so every row in
        // `rows` is by definition a match — `matched_user_tag =
        // bullet_count` when a tag is set. Keeps the LLM-facing
        // shape identical to find_related so an agent reading
        // either prompt sees the same structured prelude.
        let bullet_count = rows.len();
        let summary_line = match user_tag.as_deref() {
            Some(tag) => format!(
                "Summary: bullets={bullet_count}; user_tag=\"{tag}\"; matched_user_tag={bullet_count}\n\n"
            ),
            None => format!("Summary: bullets={bullet_count}\n\n"),
        };

        let user_text = format!(
            "Below are the user's stable preferences and personal facts that we have on file. \
             Summarize them into 5–8 concise bullet points capturing what an AI assistant \
             should consistently keep in mind when collaborating with this user. Group related \
             items, preserve any explicit dos/don'ts, and surface contradictions if any.\n\n\
             ---\n{summary_line}{bullets}",
        );

        Ok(json!({
            "description": "Summarize the user's stable preferences from Anamnesis records.",
            "messages": [{
                "role": "user",
                "content": {"type": "text", "text": user_text}
            }]
        }))
    }

    async fn prompt_find_related(&self, args: Value) -> Result<Value, String> {
        let text = args
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "find_related.text is required".to_string())?;
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .unwrap_or(5);
        // Round 89 (PR-78k): opt-in compact score breakdown
        // appended to each bullet. Mirrors R87's
        // `search_memories({ explain: true })` shape but
        // rendered as a short numeric line — JSON inside a
        // prompt would burn LLM context for no readability
        // gain.
        let explain_requested = args
            .get("explain")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Round-65: optional filter args. SearchFilter has been complete
        // on the store side since round-22 (api.rs::SearchFilter); the
        // MCP prompt just hadn't exposed any way to set it, so every
        // find_related call ran against the entire corpus. With a real
        // multi-adapter install (claude-code + codex + letta + mem0 …)
        // this dilutes the top-N. These args let the caller scope the
        // recall to one source / kind / scope without breaking any
        // existing text/limit-only invocation.
        //
        // Round 93 (PR-78o): also honour `user_tag` so an agent can
        // narrow related-memory lookup to records the operator has
        // curated. Normalised through the same helper `tag_record`
        // writes through, so `--user-tag Keep` finds tags stored
        // as `keep`.
        let user_tag = match args.get("user_tag").and_then(|v| v.as_str()) {
            Some(raw) => Some(
                anamnesis_store::normalize_user_tag_name(raw)
                    .map_err(|e| format!("find_related.user_tag: {e}"))?,
            ),
            None => None,
        };
        let filter = anamnesis_store::SearchFilter {
            source: args
                .get("source")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            instance: args
                .get("instance")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            kind: args.get("kind").and_then(|v| v.as_str()).map(str::to_owned),
            scope: args
                .get("scope")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            user_tag,
            ..Default::default()
        };

        let store = &self.store;
        let opts = HybridOpts {
            limit,
            candidate_pool: (limit * 4).max(limit),
            mode: SearchMode::Hybrid,
        };
        let hits = match self.provider.as_ref() {
            Some(p) => HybridSearcher::new(p.as_ref())
                .search_filtered(store, text, &filter, &opts)
                .await
                .map_err(|e| format!("search: {e}"))?,
            None => HybridSearcher::<NoProvider>::fulltext_only()
                .search_filtered(store, text, &filter, &opts.fulltext_fallback())
                .await
                .map_err(|e| format!("search: {e}"))?,
        };
        let packed = pack(
            store,
            &hits,
            &ContextBudget {
                max_records: limit as usize,
                ..ContextBudget::default()
            },
        )
        .map_err(|e| format!("pack: {e}"))?;

        let mut bullets = String::new();
        for p in &packed {
            let snippet = p
                .matched_chunks
                .first()
                .map(|c| trim_for_prompt(&c.content, 240))
                .unwrap_or_default();
            bullets.push_str(&format!(
                "- [{adapter}] {snippet}  (score={score:.3})\n",
                adapter = p.record.source.adapter,
                score = p.score,
            ));
            // Round 89: opt-in numeric breakdown on the next
            // indented line. Compact text — `explain: ...`,
            // not JSON — so the LLM sees the structure but the
            // token cost stays bounded.
            if explain_requested {
                bullets.push_str(&format!(
                    "  {}\n",
                    format_score_explain_for_prompt(&p.score_explain())
                ));
            }
        }
        if bullets.is_empty() {
            bullets.push_str("(no related memories found)\n");
        }

        // Round 100 (PR-78v): compact summary line so the LLM
        // sees at a glance how many memories were attached and
        // (when a user_tag filter was set) how many actually
        // carried that tag. Cheap — `packed.len()` is in hand
        // and `user_tags` already lives on `RecordHeader` from
        // R78. Text-only, doesn't bloat the prompt token cost
        // even when bullets is large.
        let bullet_count = packed.len();
        let matched_user_tag_count = match filter.user_tag.as_deref() {
            Some(tag) => packed
                .iter()
                .filter(|p| p.record.user_tags.iter().any(|t| t == tag))
                .count(),
            None => 0,
        };
        let summary_line = match filter.user_tag.as_deref() {
            Some(tag) => format!(
                "Summary: bullets={bullet_count}; user_tag=\"{tag}\"; matched_user_tag={matched_user_tag_count}\n\n"
            ),
            None => format!("Summary: bullets={bullet_count}\n\n"),
        };

        let user_text = format!(
            "The user is currently working on / discussing the following:\n\n{text}\n\n\
             Here are the most relevant memories Anamnesis has on file. Cite them when they \
             contradict or reinforce what the user is asking. Don't repeat verbatim; weave them \
             into your reply where useful.\n\n---\n{summary_line}{bullets}",
        );
        Ok(json!({
            "description": "Inject the top-N related Anamnesis memories into the LLM's context.",
            "messages": [{
                "role": "user",
                "content": {"type": "text", "text": user_text}
            }]
        }))
    }
}

fn trim_for_prompt(s: &str, max_chars: usize) -> String {
    let collapsed: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    if collapsed.chars().count() > max_chars {
        let cut: String = collapsed.chars().take(max_chars).collect();
        format!("{cut}…")
    } else {
        collapsed
    }
}

fn prompts_list_payload() -> Value {
    // Round 111 (PR-78ag): top-level `summary` field is an
    // additive enrichment on top of the MCP `prompts/list`
    // shape so an agent doing first-time discovery can decide
    // which prompt to fetch without parsing every per-arg
    // description. The per-prompt entries are unchanged
    // (back-compat with every R0-R110 client). Counted from
    // `prompts[].len()` would be more elegant if the array
    // were built first; kept hard-coded here because the
    // array is small and the summary text needs to be hand-
    // authored anyway.
    json!({
        "summary": "2 prompts: `summarize_my_preferences` distills the user's stable user-scope preferences (with optional `user_tag` overlay); `find_related` injects top-N memories related to a free-text query (filterable by source / instance / kind / scope / user_tag, optional score `explain`).",
        "prompts": [
            {
                "name": "summarize_my_preferences",
                "description": "Summarize the user's stable preferences from Anamnesis user-scope records.",
                "arguments": [
                    {"name": "limit", "description": "Max records to include (default 20)", "required": false},
                    {"name": "user_tag", "description": "Round 94: restrict to records carrying this user tag (overlay table from R78). Tag is normalised (`trim().to_lowercase()`) to match `tag_record` writes. Filter pushes down at the SQL recall stage before LIMIT.", "required": false}
                ]
            },
            {
                "name": "find_related",
                "description": "Inject the top-N Anamnesis memories related to a free-text description.",
                "arguments": [
                    {"name": "text", "description": "What the user is working on or asking about", "required": true},
                    {"name": "limit", "description": "Max related memories to include (default 5)", "required": false},
                    {"name": "source", "description": "Restrict to one adapter id, e.g. `claude-code`, `codex`, `mem0` (default: all sources)", "required": false},
                    {"name": "instance", "description": "Restrict to one instance discriminator (only meaningful when `source` is also set)", "required": false},
                    {"name": "kind", "description": "Restrict to one Kind: fact | preference | feedback | reference | episode | skill | unknown", "required": false},
                    {"name": "scope", "description": "Restrict to one Scope: user | project | session | ephemeral", "required": false},
                    {"name": "user_tag", "description": "Round 93: restrict to records carrying this user tag (overlay table from R78). Tag is normalised (`trim().to_lowercase()`) to match `tag_record` writes — `Keep-Forever` matches `keep-forever`. Filter pushes down at the SQL recall stage, so a single tagged record surfaces under a heavy untagged-majority corpus.", "required": false},
                    {"name": "explain", "description": "Round 89: append a compact numeric score breakdown (record_score / best_chunk_rrf_score / kind_boost / fts_rank / vec_rank / rrf_k) to each bullet. Default off — keeps prompt token cost bounded.", "required": false}
                ]
            }
        ]
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Resource implementations
// ─────────────────────────────────────────────────────────────────────────────

impl AnamnesisServer {
    async fn read_resource(&self, uri: &str) -> Result<Value, String> {
        let stripped = uri
            .strip_prefix("anamnesis://")
            .ok_or_else(|| format!("uri must start with anamnesis://: {uri}"))?;
        let (kind, rest) = stripped
            .split_once('/')
            .ok_or_else(|| format!("uri missing path: {uri}"))?;
        match kind {
            "record" => self.read_record_resource(rest).await,
            "source" => self.read_source_resource(rest).await,
            "timeline" => self.read_timeline_resource(rest).await,
            other => Err(format!("unknown resource kind: {other}")),
        }
    }

    async fn read_record_resource(&self, id: &str) -> Result<Value, String> {
        let store = &self.store;
        let rec = store
            .get_record(&RecordId(id.to_string()))
            .map_err(|e| format!("store: {e}"))?
            .ok_or_else(|| format!("record not found: {id}"))?;
        serde_json::to_value(&rec).map_err(|e| format!("serialize: {e}"))
    }

    async fn read_source_resource(&self, spec: &str) -> Result<Value, String> {
        let (adapter, instance) = match spec.split_once(':') {
            Some((a, i)) => (a, Some(i)),
            None => (spec, None),
        };
        let store = &self.store;
        let rows = store.list_sources().map_err(|e| format!("list: {e}"))?;
        let matching: Vec<_> = rows
            .into_iter()
            .filter(|(a, i, _)| {
                a == adapter
                    && match instance {
                        Some(want) => i == want,
                        None => true,
                    }
            })
            .collect();
        // Inline a small recent-record sample (≤5) so consumers can preview
        // without running search_memories.
        let conn = store.conn();
        let mut stmt = conn
            .prepare(
                "SELECT id, kind, scope, native_path, created_at FROM records \
                 WHERE adapter = ?1 \
                 ORDER BY created_at DESC LIMIT 5",
            )
            .map_err(|e| format!("recent prepare: {e}"))?;
        let recent: Vec<Value> = stmt
            .query_map([adapter], |r| {
                Ok(json!({
                    "id": r.get::<_, String>(0)?,
                    "kind": r.get::<_, String>(1)?,
                    "scope": r.get::<_, String>(2)?,
                    "native_path": r.get::<_, Option<String>>(3)?,
                    "created_at": r.get::<_, i64>(4)?,
                }))
            })
            .map_err(|e| format!("recent query: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(json!({
            "sources": matching.iter().map(|(a, i, loc)| json!({
                "adapter": a,
                "instance": if i.is_empty() { Value::Null } else { Value::String(i.clone()) },
                "location": loc,
            })).collect::<Vec<_>>(),
            "recent": recent,
        }))
    }

    async fn read_timeline_resource(&self, date: &str) -> Result<Value, String> {
        // Accept YYYY-MM-DD; return records whose created_at falls in that
        // UTC day.
        let date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
            .map_err(|e| format!("invalid date {date}: {e}"))?;
        let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
        let end = start + 24 * 3600;
        let store = &self.store;
        let conn = store.conn();
        let mut stmt = conn
            .prepare(
                "SELECT id, adapter, instance, kind, scope, native_path, created_at \
                 FROM records WHERE created_at >= ?1 AND created_at < ?2 \
                 ORDER BY created_at ASC",
            )
            .map_err(|e| format!("timeline prepare: {e}"))?;
        let events: Vec<Value> = stmt
            .query_map([start, end], |r| {
                Ok(json!({
                    "id": r.get::<_, String>(0)?,
                    "adapter": r.get::<_, String>(1)?,
                    "instance": r.get::<_, String>(2)?,
                    "kind": r.get::<_, String>(3)?,
                    "scope": r.get::<_, String>(4)?,
                    "native_path": r.get::<_, Option<String>>(5)?,
                    "created_at": r.get::<_, i64>(6)?,
                }))
            })
            .map_err(|e| format!("timeline query: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(json!({
            "date": date.format("%Y-%m-%d").to_string(),
            "count": events.len(),
            "events": events,
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool / resource catalogues
// ─────────────────────────────────────────────────────────────────────────────

impl AnamnesisServer {
    /// Build the `tools/list` payload, hiding admin tools (see
    /// [`ADMIN_TOOLS`]) when `allow_admin_tools` is false. The dispatcher
    /// in `handle_tools_call` enforces the same gate at call time — this
    /// filter is *cosmetic* protection against discovery, never the only
    /// line of defense.
    fn tools_list_payload(&self) -> Value {
        let mut payload = tools_list_payload_all();
        if !self.allow_admin_tools {
            if let Some(arr) = payload.get_mut("tools").and_then(|v| v.as_array_mut()) {
                arr.retain(|t| {
                    t.get("name")
                        .and_then(|n| n.as_str())
                        .map(|n| !is_admin_tool(n))
                        .unwrap_or(true)
                });
            }
        }
        payload
    }
}

/// The full catalogue of MCP tools the server knows about, before any
/// admin gating. Used as the seed by `AnamnesisServer::tools_list_payload`.
fn tools_list_payload_all() -> Value {
    json!({
        "tools": [
            {
                "name": "search_memories",
                "description": "Hybrid search across all imported records (FTS + vector + RRF). \
                                All filters push down to the SQL recall stage (PR-C / §-1.5 PR-5) \
                                so they survive minority-source dominance.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "source": {
                            "type": "string",
                            "description": "Restrict to one adapter id (claude-code, codex, mem0, generic-mcp)."
                        },
                        "instance": {
                            "type": "string",
                            "description": "Restrict to a specific source instance (the discriminator passed to \
                                            `source add --instance`). Meaningful only when `source` is also set."
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["fact", "preference", "feedback", "reference", "episode", "skill", "unknown"],
                            "description": "Restrict to one Kind."
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["user", "project", "session", "ephemeral"],
                            "description": "Restrict to one Scope."
                        },
                        "since": {
                            "type": "string",
                            "description": "RFC3339 lower bound on records.created_at (inclusive). \
                                            E.g. \"2026-04-01T00:00:00Z\"."
                        },
                        "until": {
                            "type": "string",
                            "description": "RFC3339 upper bound on records.created_at (inclusive)."
                        },
                        "limit": {"type": "integer", "minimum": 1, "default": 10},
                        "mode": {"type": "string", "enum": ["fulltext", "vector", "hybrid"], "default": "hybrid"},
                        "trace": {
                            "type": "boolean",
                            "default": false,
                            "description": "Return per-stage search timings (embed_query / fts / vec / rrf / pack, in ms) \
                                            and candidate counts under a top-level `trace` field. Never includes query \
                                            text, snippets, or any record / chunk identifiers — strictly numeric \
                                            diagnostic shape. Default off; omitting it keeps the response wire-identical \
                                            to pre-Round-71."
                        },
                        "user_tag": {
                            "type": "string",
                            "description": "Restrict to records carrying this user tag (overlay table from \
                                            Round 78). Filter pushes down into FTS, BLOB-vec fallback, and \
                                            sqlite-vec at the SQL recall stage, so a single tagged record \
                                            surfaces even under a heavy untagged-majority corpus. Tag is \
                                            normalised (`trim().to_lowercase()`) to match `tag_record` writes."
                        },
                        "explain": {
                            "type": "boolean",
                            "default": false,
                            "description": "Round 87: attach a per-result `explain` block breaking down the \
                                            ranking arithmetic — record_score, best_chunk_rrf_score, kind_boost, \
                                            and the FTS / vector stage ranks + raw scores + \
                                            `rrf_contribution = 1/(rrf_k + rank)`. Numeric-only (no record/chunk \
                                            ids, no query, no snippet beyond what `results[]` already exposes). \
                                            Orthogonal to `trace` (stage timings + candidate counts) — both can \
                                            be true."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_record",
                "description": "Fetch one record by id. Read-only, NOT admin-gated. \
                                Pass `include_lineage: true` (Round 85) to attach a `lineage` \
                                block carrying the leaf→root provenance chain — each entry is \
                                a *summary* (provenance + classification, no content) to keep \
                                the payload bounded; agents that want full ancestor content \
                                re-call `get_record` for that id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Record id (as returned by `search_memories` / `list_forgotten`)."
                        },
                        "include_lineage": {
                            "type": "boolean",
                            "default": false,
                            "description": "Round 85: attach `lineage.{start, depth, complete, missing_parent, chain[]}` from `Store::lineage_chain`. Default off — keeps the wire shape back-compatible."
                        }
                    },
                    "required": ["id"]
                }
            },
            {
                "name": "list_sources",
                "description": "List registered sources + active model + counters. Each source carries \
                                `record_count`, `chunk_count`, `tagged_record_count` (R82: distinct \
                                records with ≥1 user_tag), `last_import_at`, and `location`. \
                                Round 96: optional `source` / `instance` narrow the `sources[]` array; \
                                the top-level `stats` block keeps reporting the whole store so existing \
                                consumers reading `stats.records` are unaffected. Round 103: `source` \
                                also accepts a comma-separated OR list — see the arg description.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Restrict the sources array to one or more adapter ids. Single value (`\"mem0\"`) is exact match (R96); comma-separated list (`\"mem0,claude-code\"`) is OR (R103) — both adapters' rows come back, everything else drops. Tokens are trimmed and empty tokens dropped. Omit (or empty string) for all sources."
                        },
                        "instance": {
                            "type": "string",
                            "description": "Round 96: restrict the sources array to one instance discriminator. Only meaningful when `source` is also set."
                        }
                    }
                }
            },
            {
                "name": "source_show",
                "description": "Round 86: per-source detail view for one `(adapter, instance)`. \
                                Returns the same source object as `list_sources` plus a \
                                `recent_import_errors[]` array (newest-first, capped by \
                                `error_limit`, default 5 / max 10). ADMIN-GATED — the import-error \
                                rows carry `native_path` + adapter-side error text. Missing source \
                                returns a tool error.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Adapter id (e.g. `claude-code`, `mem0`)."
                        },
                        "instance": {
                            "type": "string",
                            "description": "Optional instance discriminator. Omit for the default-instance row."
                        },
                        "error_limit": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 5,
                            "description": "Max recent import-error rows. Clamped to [1, 10]."
                        }
                    },
                    "required": ["source"]
                }
            },
            {
                "name": "import_source",
                "description": "Run an import job for one source registered via CLI `anamnesis source add`. \
                                The source's location (path or URL) and credentials (env-var name only — value \
                                never leaves the operator's shell) are taken from the registry; MCP clients \
                                cannot pass `path` or `url` directly. Adapter ids: claude-code, codex, mem0, \
                                letta, hermes, openclaw, tdai, openviking, mempalace, memori, memos, memary, generic-mcp. \
                                Admin-gated — server must be started with --allow-admin-tools \
                                or have it enabled in config.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "adapter": {
                            "type": "string",
                            "description": "claude-code | codex | mem0 | letta | hermes | openclaw | tdai | openviking | mempalace | memori | memos | memary | generic-mcp",
                            "enum": ["claude-code", "codex", "mem0", "letta", "hermes", "openclaw", "tdai", "openviking", "mempalace", "memori", "memos", "memary", "generic-mcp"]
                        },
                        "instance": {
                            "type": "string",
                            "description": "Instance discriminator. Must match an existing source registry row \
                                            for (adapter, instance)."
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "Scan-only: count raw records without writing. Source registry and \
                                            audit log are not touched in dry-run."
                        }
                    },
                    "required": ["adapter"]
                }
            },
            {
                "name": "trace_provenance",
                "description": "Return native_id / native_path / raw_hash for one record. \
                                Pass `id` (record_id from search_memories.record_id) for record-level provenance, \
                                or `chunk_id` (search_memories.chunk_id) to also get the chunk content and its \
                                jieba-tokenized form for debugging retrieval quality.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Record id (= trace_id from search_memories)"
                        },
                        "chunk_id": {
                            "type": "string",
                            "description": "Chunk id from a search_memories hit. Pass this when you want to inspect the exact matched chunk."
                        }
                    },
                    "anyOf": [
                        {"required": ["id"]},
                        {"required": ["chunk_id"]}
                    ]
                }
            },
            {
                "name": "doctor",
                "description": "Per-source health check. For each registered source, runs the adapter's \
                                `health()` probe (cheap — directory existence, single-row SELECT, or \
                                `/healthz` GET) and joins against per-source record counts. Use this when \
                                `search_memories` returns nothing relevant to distinguish \"no memories \
                                yet\" from \"the source is unreachable\". Read-only; never mutates state.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Restrict to one or more adapter ids. Single value (`\"mem0\"`) is exact match; comma-separated list (`\"mem0,claude-code\"`) is OR (R110) — both adapters' rows come back, everything else drops. Tokens trimmed and empty tokens dropped. Omit (or empty string) to check every registered source."
                        },
                        "instance": {
                            "type": "string",
                            "description": "Instance discriminator. Meaningful only when `source` is also set."
                        },
                        "since": {
                            "type": "string",
                            "description": "Mark a source as stale if its last_import_at is older than this \
                                            relative window. Shapes: `Nd` (days), `Nh` (hours), `Nm` (minutes), \
                                            or a bare integer (seconds). Never-imported sources always count \
                                            as stale once `since` is set."
                        },
                        "metrics_since": {
                            "type": "string",
                            "description": "Window for the `request_metrics` summary (per-tool count / errors / \
                                            p50 / p95 / p99 / last_ms). Same grammar as `since`. Defaults to 24h."
                        }
                    }
                }
            },
            {
                "name": "forget_record",
                "description": "Permanently forget a record by id. Writes a tombstone keyed on \
                                `(adapter, instance, native_id)` so the same source can no longer \
                                resurrect the memory through a subsequent `import_source`. \
                                ADMIN-GATED: hidden from `tools/list` and rejected by `tools/call` \
                                unless the server was started with `allow_admin_tools = true`. \
                                Does NOT modify any upstream source data — Anamnesis stays \
                                read-only with respect to the original Claude Code memory file / \
                                mem0 row / etc. Calling with an unknown `record_id` is a tool error \
                                (no tombstone was written, so the guarantee cannot be made).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "record_id": {
                            "type": "string",
                            "description": "Record id to forget — same shape as `search_memories.record_id`."
                        },
                        "reason": {
                            "type": "string",
                            "description": "Optional operator-supplied reason. Stored on the tombstone \
                                            for the future `list_forgotten` view."
                        },
                        "dry_run": {
                            "type": "boolean",
                            "default": false,
                            "description": "Round 83: preview the cascade without writing. Returns `status: \
                                            \"would-forget\"` plus `would_delete` (records, raw_artifacts, \
                                            record_chunks, chunk_embeddings, embedding_jobs, user_record_tags, \
                                            vec0_rows) and `would_insert` (record_tombstones, audit_log_entries). \
                                            Does NOT mutate the store and does NOT append an audit entry."
                        }
                    },
                    "required": ["record_id"]
                }
            },
            {
                "name": "unforget_record",
                "description": "Remove a tombstone so the source can resurrect the memory on its \
                                next `import_source`. Does NOT recreate the record itself — the \
                                tombstone only stored provenance, so 'unforget' means 'allowed to \
                                come back if the source re-emits.' Anamnesis stays a read-only \
                                mirror; resurrection happens through the source's own re-import, \
                                not through this tool. ADMIN-GATED. Calling with an id that has \
                                no tombstone is a tool error (likely typo from `list_forgotten`).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "record_id": {
                            "type": "string",
                            "description": "Record id to unforget — same shape as \
                                            `list_forgotten.rows[].record_id`."
                        },
                        "dry_run": {
                            "type": "boolean",
                            "default": false,
                            "description": "Round 95: preview the tombstone the real unforget would delete. \
                                            Returns `outcome: \"would-unforget\"` plus `would_delete.record_tombstones=1` \
                                            and `would_insert.audit_log_entries=1`. Does NOT mutate the store and does NOT \
                                            append an audit entry. Symmetric with `forget_record { dry_run: true }`."
                        }
                    },
                    "required": ["record_id"]
                }
            },
            {
                "name": "list_forgotten",
                "description": "List tombstoned records, newest-first. Read-only audit view over \
                                `record_tombstones`. ADMIN-GATED — hidden from `tools/list` and \
                                rejected by `tools/call` unless the server allows admin tools. \
                                **Default response is redacted**: `native_path`, `raw_hash`, and \
                                `reason` are reported only as `has_*` booleans. Set \
                                `include_sensitive=true` to opt in to those fields (e.g. before an \
                                `unforget` decision). `limit` is clamped to [1, 100].",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Restrict to one adapter id (`claude-code`, `mem0`, …)."
                        },
                        "instance": {
                            "type": "string",
                            "description": "Restrict to one instance discriminator within `source`."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 20,
                            "description": "Max rows. Clamped to [1, 100] by the store."
                        },
                        "include_sensitive": {
                            "type": "boolean",
                            "default": false,
                            "description": "Reveal `native_path`, `raw_hash`, `reason`. Off by default."
                        },
                        "include_counts": {
                            "type": "boolean",
                            "default": false,
                            "description": "Round 90: also return a `counts` block with `total` + `by_source[]` (per-`(adapter, instance)` tombstone totals). Counts respect the same source/instance filter as the row list but reflect the full matching set — not just the current page."
                        },
                        "csv": {
                            "type": "boolean",
                            "default": false,
                            "description": "Round 105: return a CSV string in `csv` (header `record_id,adapter,instance,native_id,forgotten_at,has_reason,has_native_path`) instead of structured `rows[]`. Same redacted summary discipline as the CLI `list-forgotten --csv` — never carries `reason` / `native_path` / `raw_hash`. Mutually exclusive with `include_sensitive: true` and `include_counts: true` (CSV is flat redacted rows by design)."
                        }
                    }
                }
            },
            {
                "name": "dedupe",
                "description": "Report records sharing identical `raw_hash` (exact source-payload \
                                duplicates). Read-only diagnostic, NOT admin-gated — the action \
                                half is `forget_record` (which is admin-gated). Default response \
                                is redacted: `raw_hash` and `native_path` are omitted unless \
                                `include_sensitive=true`. Limit clamped to [1, 100]. Optional \
                                `source` / `instance` scope the report to groups containing ≥1 \
                                matching record; the full sibling set is still returned so the \
                                operator sees which non-matching records share the same hash. \
                                Only catches byte-identical duplicates; semantic / near-duplicate \
                                detection is out of scope.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Restrict duplicate groups to those containing ≥1 record from a given adapter. Single value (`\"mem0\"`) is exact match (R80); comma-separated list (`\"mem0,claude-code\"`) is OR (R104) — groups whose members include at least one record from any listed adapter stay eligible. Tokens trimmed and empty tokens dropped. Omit (or empty string) for all sources."
                        },
                        "instance": {
                            "type": "string",
                            "description": "Instance discriminator. Only meaningful when `source` is also set."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 20,
                            "description": "Max number of groups. Clamped to [1, 100] by the store."
                        },
                        "include_sensitive": {
                            "type": "boolean",
                            "default": false,
                            "description": "Reveal `raw_hash` and `native_path`. Off by default."
                        },
                        "include_counts": {
                            "type": "boolean",
                            "default": false,
                            "description": "Round 98: attach a `counts` block with `total_groups` (duplicate groups matching the filter), `total_records` (sum of live records across whole groups), and `by_source[]` (per-`(adapter, instance)` duplicate-record breakdown). Counts ignore `limit` and reflect the full filter-scoped set. `by_source` counts records (not group memberships) so mixed-source groups don't double-count: sum(by_source.duplicate_record_count) == total_records."
                        },
                        "csv": {
                            "type": "boolean",
                            "default": false,
                            "description": "Round 107: return a CSV string in `csv` (header `group_index,record_id,adapter,instance,native_id,created_at,updated_at,has_native_path,record_count`) instead of structured `groups[]`. Same redacted-summary discipline as R91 `audit tail --csv` and R106 `list-forgotten --csv` — never carries `raw_hash` / `native_path`. `group_index` carries duplicate-group membership without leaking the hash. Mutually exclusive with `include_sensitive: true` and `include_counts: true`."
                        }
                    }
                }
            },
            {
                "name": "tag_record",
                "description": "Apply, remove, or replace user tags on a record. Tags live in a \
                                separate overlay table from the adapter-derived `tags` field, so \
                                they survive re-import. Read paths (`search_memories`, \
                                `get_record`) surface them as `user_tags`. ADMIN-GATED for write; \
                                reads are not admin-gated. Set semantics — re-adding an existing \
                                tag or removing a missing one is a no-op; `replace` reports the \
                                set delta (re-replacing with the same set = 0). Tags are trimmed, \
                                lower-cased, deduped before write. Limit: 32 tags per call, 64 \
                                bytes per tag. Empty `tags` is **only valid for \
                                `operation=\"replace\"`** (= clear all user tags on the record); \
                                `add`/`remove` reject empty input.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "record_id": {
                            "type": "string",
                            "description": "Record id (as returned by `search_memories` / `list_forgotten`)."
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "maxItems": 32,
                            "description": "Tags to apply/remove/replace with. Normalised before write. Must be non-empty for `add`/`remove`; may be empty for `replace` (= clear)."
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["add", "remove", "replace"],
                            "default": "add",
                            "description": "`add` inserts (set semantic); `remove` deletes (set semantic); `replace` installs `tags` as the full post-call set, deleting anything not in the input."
                        },
                        "include_stats": {
                            "type": "boolean",
                            "default": false,
                            "description": "Round 96: append a `stats` block (`total_user_tags`: post-mutation tag count) to the response. Default off — keeps the wire shape back-compatible."
                        }
                    },
                    "required": ["record_id", "tags"]
                }
            },
            {
                "name": "audit_tail",
                "description": "Round 84: tail the global mutation/search audit log at `data_dir/audit.log` \
                                (every CLI + MCP write appends one JSONL entry via Audit::record). \
                                ADMIN-GATED — the entries can include search queries, forget reasons, \
                                and source locations, so non-admin agents shouldn't be able to \
                                back-door read those. Default response is **redacted summary**: each \
                                entry carries only `line_no / timestamp / action / via / outcome`. \
                                Pass `include_detail: true` to receive the full per-entry `detail` \
                                payload. Limit clamped to [1, 1000]. Distinct from the CLI's stage 2 \
                                `audit list/show`, which reads a separate extractor log.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 20,
                            "description": "Max entries to return. Clamped to [1, 1000] by the server."
                        },
                        "action": {
                            "type": "string",
                            "description": "Filter on `entry.action`. Single value (`\"forget\"`) is exact match; comma-separated list (`\"forget,search\"`) is OR (R102) — both `forget` and `search` rows come back, everything else is dropped. Tokens are trimmed and empty tokens dropped. Omit (or empty string) for all actions. The response echoes both `filter.action` (raw input, back-compat) and `filter.actions` (normalised list)."
                        },
                        "since": {
                            "type": "string",
                            "description": "Relative lookback. Shapes: `Nd` (days), `Nh` (hours), `Nm` (minutes), or bare integer (seconds). Same grammar as `doctor.since`."
                        },
                        "include_detail": {
                            "type": "boolean",
                            "default": false,
                            "description": "When true, attach the full per-entry `detail` payload. Default off — keeps search queries / forget reasons out of the response by default."
                        },
                        "csv": {
                            "type": "boolean",
                            "default": false,
                            "description": "Round 92: return a CSV string in `csv` (header `line_no,timestamp,action,via,outcome`) instead of structured `entries[]`. Same redacted summary discipline as the CLI `audit tail --csv` — never carries `detail` / `reason` / `query`. Mutually exclusive with `include_detail: true`."
                        }
                    }
                }
            }
        ]
    })
}

impl AnamnesisServer {
    /// MCP `resources/list` with cursor-based pagination (round-21,
    /// §-1.5 PR-2).
    ///
    /// Wire shape (per MCP spec):
    ///   * request: `{ "cursor"?: string }` (`limit` is non-standard
    ///     but accepted; defaults to `RESOURCES_LIST_PAGE` and is
    ///     clamped to `[1, MAX_LIST_LIMIT]`).
    ///   * response: `{ "resources": [...], "resourceTemplates": [...],
    ///     "nextCursor"?: string }`
    ///
    /// Pagination ordering is lexicographic ascending by record id
    /// (record id is content-derived → deterministic across hosts).
    /// `nextCursor` is `Some(last_id)` when the page hit the limit and
    /// another page MAY exist; otherwise `None`. The templates are
    /// **only** emitted on the first page (`cursor == None`) so
    /// downstream paginators see them once, not on every page.
    ///
    /// Store errors are propagated as JSON-RPC `-32603` (internal
    /// error) — round-21 also closes the §19.3 finding that the
    /// previous handler swallowed errors via `unwrap_or_default`.
    async fn handle_resources_list(
        &self,
        id: Value,
        params: Value,
    ) -> crate::protocol::JsonRpcResponse {
        let cursor = params.get("cursor").and_then(|v| v.as_str());
        let requested_limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n.min(u32::MAX as u64) as u32)
            .unwrap_or(RESOURCES_LIST_PAGE);

        let (ids, next_cursor) = match self.store.list_record_ids_paged(cursor, requested_limit) {
            Ok(p) => p,
            Err(e) => {
                return crate::protocol::JsonRpcResponse::err(
                    id,
                    -32603,
                    format!("resources/list: {e}"),
                );
            }
        };

        let mut resources: Vec<Value> = ids
            .iter()
            .map(|rid| {
                json!({
                    "uri": format!("anamnesis://record/{rid}"),
                    "name": format!("record {}", &rid[..rid.len().min(12)]),
                    "description": "One AnamnesisRecord as JSON.",
                    "mimeType": "application/json",
                })
            })
            .collect();

        // Templates appear ONLY on the first page. Paginating
        // consumers see them once; later pages stay focused on the
        // continuing record window so `nextCursor` semantics stay
        // clean.
        let templates = static_resource_templates();
        if cursor.is_none() {
            for t in templates.as_array().into_iter().flatten() {
                resources.push(t.clone());
            }
        }

        let mut payload = json!({
            "resources": resources,
            "resourceTemplates": templates,
        });
        if let Some(c) = next_cursor {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("nextCursor".into(), Value::String(c));
            }
        }
        crate::protocol::JsonRpcResponse::ok(id, payload)
    }
}

/// Default page size for `resources/list` when the client doesn't
/// supply `limit`. Round-21 changed the semantics from "recent
/// window" to "first page of full catalogue"; sized so a single
/// page response stays under ~50 KB of JSON.
pub const RESOURCES_LIST_PAGE: u32 = 100;

/// Deprecated alias kept for binary back-compat in any out-of-tree
/// consumer (`anamnesis-mcp-server` is published; renaming the const
/// would be a SemVer break unless we keep the old name in scope). New
/// code should use `RESOURCES_LIST_PAGE`.
#[deprecated(note = "renamed in round-21 to RESOURCES_LIST_PAGE; this alias is for back-compat")]
pub const RESOURCES_LIST_RECENT_LIMIT: u32 = RESOURCES_LIST_PAGE;

fn static_resource_templates() -> Value {
    json!([
        {
            "uri": "anamnesis://record/{id}",
            "name": "Anamnesis record",
            "description": "Fetch one record as JSON.",
            "mimeType": "application/json"
        },
        {
            "uri": "anamnesis://source/{adapter}",
            "name": "Anamnesis source summary",
            "description": "Source description + recent 5 records.",
            "mimeType": "application/json"
        },
        {
            "uri": "anamnesis://timeline/{YYYY-MM-DD}",
            "name": "Anamnesis timeline",
            "description": "All records captured on a UTC day.",
            "mimeType": "application/json"
        }
    ])
}

// (The old static-only `resources_list_payload` was deleted in round-13
// — `AnamnesisServer::resources_list_payload` above replaces it.
// Templates moved into `static_resource_templates`; concrete records
// are now enumerated dynamically from the store.)

/// Placeholder type — only used as a generic argument for HybridSearcher
/// when no provider is wired. The fulltext_only path never invokes its
/// methods.
struct NoProvider;
#[async_trait::async_trait]
impl EmbeddingProvider for NoProvider {
    fn model_id(&self) -> anamnesis_core::ModelId {
        anamnesis_core::ModelId::new("noprovider", "noop", 0)
    }
    fn dim(&self) -> u16 {
        1
    }
    async fn embed_batch(
        &self,
        _texts: &[&str],
        _task: anamnesis_core::EmbeddingTask,
    ) -> anamnesis_core::Result<Vec<Vec<f32>>> {
        Err(anamnesis_core::Error::Other(
            "NoProvider should not be called".into(),
        ))
    }
}

trait HybridOptsFulltextFallback {
    fn fulltext_fallback(&self) -> HybridOpts;
}

impl HybridOptsFulltextFallback for HybridOpts {
    fn fulltext_fallback(&self) -> HybridOpts {
        HybridOpts {
            limit: self.limit,
            candidate_pool: self.candidate_pool,
            mode: SearchMode::Fulltext,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId as Rid, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use chrono::{TimeZone, Utc};

    fn make_record(
        adapter: &str,
        native_id: &str,
        content: &str,
        created_ts: i64,
    ) -> AnamnesisRecord {
        AnamnesisRecord {
            id: Rid::from_parts(adapter, None, native_id),
            source: SourceDescriptor {
                adapter: adapter.into(),
                instance: None,
                version: "0".into(),
            },
            content: content.into(),
            embedding: None,
            scope: Scope::User,
            kind: Kind::Fact,
            created_at: Utc.timestamp_opt(created_ts, 0).unwrap(),
            updated_at: None,
            tags: vec![],
            metadata: Default::default(),
            provenance: Provenance {
                native_id: native_id.into(),
                native_path: Some(format!("/p/{native_id}")),
                captured_at: Utc.timestamp_opt(created_ts, 0).unwrap(),
                raw_hash: "h".into(),
                derived_from: None,
            },
            schema_version: SCHEMA_VERSION,
        }
    }

    fn server_with_records(records: &[AnamnesisRecord]) -> AnamnesisServer {
        let store = Store::open_in_memory().unwrap();
        for r in records {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }
        store
            .register_source("claude-code", Some("default"), Some("/tmp/x"), None)
            .unwrap();
        AnamnesisServer::new(store, None, std::env::temp_dir())
    }

    fn req(method: &str, params: Value) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(Value::from(1)),
            method: method.into(),
            params,
        }
    }

    #[tokio::test]
    async fn initialize_returns_server_info() {
        let s = server_with_records(&[]);
        let resp = s.handle(req("initialize", Value::Null)).await;
        let result = resp.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "anamnesis");
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
    }

    fn tool_names_from(payload: &Value) -> Vec<String> {
        payload["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["name"].as_str().map(str::to_owned))
            .collect()
    }

    #[tokio::test]
    async fn tools_list_hides_admin_tools_by_default() {
        // PR-A: import_source is admin-gated and admin defaults to OFF.
        let s = server_with_records(&[]);
        assert!(!s.admin_tools_allowed(), "admin must default to off");
        let resp = s.handle(req("tools/list", Value::Null)).await;
        let names = tool_names_from(&resp.result.unwrap());
        for expected in [
            "search_memories",
            "get_record",
            "list_sources",
            "trace_provenance",
            "doctor",
            // Round 77: read-only diagnostic, not admin-gated.
            "dedupe",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "missing tool {expected}"
            );
        }
        assert!(
            !names.contains(&"import_source".to_string()),
            "import_source MUST be hidden by default — found in tools/list",
        );
        // R77 added `dedupe` to the non-admin catalogue (5 → 6).
        assert_eq!(names.len(), 6, "expect exactly 6 non-admin tools");
    }

    #[tokio::test]
    async fn tools_list_includes_all_when_admin_enabled() {
        // PR-A: with admin enabled, the full catalogue is back.
        // Round 73 added `forget_record` as a 2nd admin tool, so the
        // total climbed 6 → 7.
        let store = Store::open_in_memory().unwrap();
        let s = AnamnesisServer::new(store, None, std::env::temp_dir()).with_admin_tools(true);
        let resp = s.handle(req("tools/list", Value::Null)).await;
        let names = tool_names_from(&resp.result.unwrap());
        // R73 added forget_record (6→7). R74 added list_forgotten
        // (7→8). R75 added unforget_record (8→9). R77 added
        // dedupe (9→10). R78 added tag_record (10→11). R84 added
        // audit_tail (11→12). R86 added source_show (12→13).
        assert_eq!(names.len(), 13);
        for expected in [
            "search_memories",
            "get_record",
            "list_sources",
            "import_source",
            "trace_provenance",
            "doctor",
            "forget_record",
            "unforget_record",
            "list_forgotten",
            "dedupe",
            "tag_record",
            "audit_tail",
            "source_show",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "missing tool {expected}"
            );
        }
    }

    #[tokio::test]
    async fn import_source_call_rejected_when_admin_disabled() {
        // Codex-flagged trap: clients can cache the schema and call
        // tools/call directly without going through tools/list. The
        // server-side reject is the load-bearing check.
        let s = server_with_records(&[]);
        assert!(!s.admin_tools_allowed());
        let resp = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "import_source",
                    "arguments": {"adapter": "claude-code", "path": "/tmp/nonexistent"}
                }),
            ))
            .await;
        // MUST be an MCP error, not a 200 success with empty payload.
        let err = resp.error.expect("call must error out when admin disabled");
        assert_eq!(err.code, -32601);
        assert!(
            err.message.contains("admin tool"),
            "error message should explain why: {}",
            err.message,
        );
    }

    #[tokio::test]
    async fn import_source_call_dispatches_when_admin_enabled() {
        // Sanity: with admin enabled the dispatch path reaches the
        // handler. We don't run a real import here (no fixtures) — we
        // assert the gate doesn't reject before the handler runs by
        // checking the response shape is NOT the gate-error shape.
        let store = Store::open_in_memory().unwrap();
        let s = AnamnesisServer::new(store, None, std::env::temp_dir()).with_admin_tools(true);
        let resp = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "import_source",
                    "arguments": {"adapter": "claude-code", "path": "/tmp/nonexistent-anamnesis-pr-a-test"}
                }),
            ))
            .await;
        // Either it succeeded (handler ran and reported 0 records) or
        // it returned a handler-level error — neither must be the
        // -32601 admin-gate error.
        if let Some(err) = resp.error {
            assert_ne!(
                err.code, -32601,
                "must not be the admin-gate error when admin is enabled (got: {})",
                err.message,
            );
        }
    }

    #[tokio::test]
    async fn unknown_method_returns_32601() {
        let s = server_with_records(&[]);
        let resp = s.handle(req("does/not/exist", Value::Null)).await;
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
    }

    #[tokio::test]
    async fn list_sources_returns_registered_sources_and_stats() {
        let r = make_record("claude-code", "x", "alpha", 1700000000);
        let s = server_with_records(&[r]);
        let resp = s
            .handle(req("tools/call", json!({"name": "list_sources"})))
            .await;
        let result = resp.result.unwrap();
        let structured = &result["structuredContent"];
        let sources = structured["sources"].as_array().unwrap();
        assert!(sources.iter().any(|s| s["adapter"] == "claude-code"));
        assert_eq!(structured["stats"]["records"], 1);
    }

    /// Round-9: pin the wire format an MCP agent receives from
    /// `list_sources`. Each source object must carry the staleness +
    /// volume signal an agent needs to decide "query this source vs.
    /// flag it as misconfigured".
    #[tokio::test]
    async fn list_sources_wire_format_carries_counts_and_staleness() {
        // We build the store explicitly here (not via server_with_records)
        // so the record's `instance` and the registered source's
        // `instance` agree — that's what real adapters / the CLI do
        // (PR-#9 source-registry-canonical-import landed this).
        let store = Store::open_in_memory().unwrap();
        let r = AnamnesisRecord {
            id: RecordId::from_parts("claude-code", None, "x"),
            source: SourceDescriptor {
                adapter: "claude-code".into(),
                instance: None,
                version: "0.0.1".into(),
            },
            content: "alpha beta gamma".into(),
            embedding: None,
            scope: Scope::User,
            kind: Kind::Fact,
            created_at: chrono::Utc.timestamp_opt(1700000000, 0).unwrap(),
            updated_at: None,
            tags: vec![],
            metadata: Default::default(),
            provenance: Provenance {
                native_id: "x".into(),
                native_path: None,
                captured_at: chrono::Utc.timestamp_opt(1700000000, 0).unwrap(),
                raw_hash: "h-wire".into(),
                derived_from: None,
            },
            schema_version: anamnesis_core::SCHEMA_VERSION,
        };
        let chunks = Chunker::default().chunk(&r.id, &r.content);
        store.upsert_record(&r, &chunks, None).unwrap();
        // Default instance (None) → stored as "" in the sources table,
        // matching record.instance and producing a JOIN-hit.
        store
            .register_source("claude-code", None, Some("/tmp/x"), None)
            .unwrap();
        let s = AnamnesisServer::new(store, None, std::env::temp_dir());

        let resp = s
            .handle(req("tools/call", json!({"name": "list_sources"})))
            .await;
        let result = resp.result.unwrap();
        let structured = &result["structuredContent"];
        let sources = structured["sources"].as_array().unwrap();
        assert_eq!(sources.len(), 1);
        let source = &sources[0];

        // All required wire-format keys (Round 82 added
        // `tagged_record_count`).
        for key in [
            "adapter",
            "instance",
            "location",
            "added_at",
            "last_import_at",
            "record_count",
            "chunk_count",
            "tagged_record_count",
        ] {
            assert!(
                source.get(key).is_some(),
                "missing wire field {key:?} in list_sources response"
            );
        }
        assert_eq!(
            source["tagged_record_count"], 0,
            "no tag_record call yet → 0 tagged records"
        );

        // Specific shapes.
        assert_eq!(source["adapter"], "claude-code");
        // Codex acceptance #3: a default instance (stored as "" in SQL)
        // must serialize as JSON null on the wire — the empty string
        // is an SQL implementation detail, not part of the agent
        // contract.
        assert_eq!(
            source["instance"],
            Value::Null,
            "default instance must serialize as JSON null, not empty string"
        );
        assert!(source["added_at"].as_i64().is_some());
        assert_eq!(
            source["record_count"], 1,
            "the seeded record must be counted under its registered source"
        );
        assert!(
            source["chunk_count"].as_u64().unwrap() >= 1,
            "alpha beta gamma must produce at least one chunk"
        );

        // last_import_at remains null until the source has actually
        // been imported (the test seeded register_source manually
        // without calling update_last_import_at). This pins Codex's
        // acceptance #3 in the wire layer.
        assert_eq!(
            source["last_import_at"],
            Value::Null,
            "registered-but-never-imported source must report null last_import_at"
        );

        // Stats block still present + correct (back-compat).
        assert_eq!(structured["stats"]["records"], 1);
    }

    /// Round 82 PR-78d: `list_sources` reports
    /// `tagged_record_count` per `(adapter, instance)` once a
    /// `tag_record` call has landed. Counts distinct records,
    /// not tag rows.
    #[tokio::test]
    async fn list_sources_tagged_record_count_reflects_tag_record_writes() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("claude-code", None, Some("/tmp/x"), None)
            .unwrap();
        // Seed two records under the same source; tag only one.
        for n in ["a", "b"] {
            let r = AnamnesisRecord {
                id: RecordId::from_parts("claude-code", None, n),
                source: SourceDescriptor {
                    adapter: "claude-code".into(),
                    instance: None,
                    version: "0.0.1".into(),
                },
                content: format!("body for {n}"),
                embedding: None,
                scope: Scope::User,
                kind: Kind::Fact,
                created_at: chrono::Utc.timestamp_opt(1700000000, 0).unwrap(),
                updated_at: None,
                tags: vec![],
                metadata: Default::default(),
                provenance: Provenance {
                    native_id: n.into(),
                    native_path: None,
                    captured_at: chrono::Utc.timestamp_opt(1700000000, 0).unwrap(),
                    raw_hash: format!("h-{n}"),
                    derived_from: None,
                },
                schema_version: anamnesis_core::SCHEMA_VERSION,
            };
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &chunks, None).unwrap();
        }
        let id_a = RecordId::from_parts("claude-code", None, "a");
        // Add TWO tags to a single record — `tagged_record_count`
        // must report `1` (records), not `2` (tag rows).
        store
            .tag_record(
                &id_a,
                &["keep".into(), "todo".into()],
                anamnesis_store::UserTagOperation::Add,
            )
            .unwrap();

        let s = AnamnesisServer::new(store, None, std::env::temp_dir());
        let resp = s
            .handle(req("tools/call", json!({"name": "list_sources"})))
            .await;
        let structured = &resp.result.unwrap()["structuredContent"];
        let source = &structured["sources"][0];
        assert_eq!(source["record_count"], 2);
        assert_eq!(
            source["tagged_record_count"], 1,
            "two tags on one record = 1 tagged record, NOT 2"
        );
    }

    #[tokio::test]
    async fn search_memories_fulltext_returns_hit() {
        let r = make_record(
            "claude-code",
            "x",
            "the marker phrase platypusBanjoComet",
            1700000000,
        );
        let s = server_with_records(&[r]);
        let resp = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "search_memories",
                    "arguments": {"query": "platypusBanjoComet", "mode": "fulltext", "limit": 5}
                }),
            ))
            .await;
        let result = resp.result.unwrap();
        let results = result["structuredContent"]["results"].as_array().unwrap();
        assert!(!results.is_empty());
        assert!(results[0]["snippet"]
            .as_str()
            .unwrap()
            .contains("platypusBanjoComet"));
    }

    /// Round-8: wire-format hardening. Lock the JSON shape an MCP agent
    /// receives from `search_memories` — every field the consumer might
    /// chain on (trace_id for `trace_provenance`, score breakdown for
    /// explain-why, time fields for UI) must be present.
    #[tokio::test]
    async fn search_memories_wire_format_carries_full_schema() {
        let r = make_record(
            "claude-code",
            "wire-test",
            "lorem ipsum dolor sit amet uniqueWireToken",
            1700000000,
        );
        let s = server_with_records(&[r]);
        let resp = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "search_memories",
                    "arguments": {"query": "uniqueWireToken", "mode": "fulltext", "limit": 5}
                }),
            ))
            .await;
        let result = resp.result.unwrap();
        let results = result["structuredContent"]["results"].as_array().unwrap();
        assert!(!results.is_empty(), "expected at least one hit");
        let hit = &results[0];

        // Identity / provenance chain (agent uses these to call
        // trace_provenance or get_record without remapping).
        assert!(hit.get("record_id").and_then(|v| v.as_str()).is_some());
        assert_eq!(hit["trace_id"], hit["record_id"]);
        assert!(hit.get("chunk_id").and_then(|v| v.as_str()).is_some());

        // Classification (agent decides whether to surface this hit).
        for f in ["adapter", "kind", "scope"] {
            assert!(
                hit.get(f).and_then(|v| v.as_str()).is_some(),
                "missing {f} in wire format"
            );
        }

        // Score breakdown — must NOT be silently null when modality
        // contributed.
        assert!(hit["score"].as_f64().is_some(), "score must be numeric");
        assert!(
            hit["rrf_score"].as_f64().is_some(),
            "rrf_score must be numeric"
        );
        assert_eq!(
            hit["score"], hit["rrf_score"],
            "`score` is kept as a back-compat alias for `rrf_score`"
        );
        assert_eq!(
            hit["from_fts"],
            json!(true),
            "fulltext mode hit must flag from_fts"
        );
        assert!(
            hit["fts_score"].as_f64().is_some(),
            "fts_score must be populated when from_fts is true"
        );

        // Time fields (agent can render recency / filter on time).
        assert!(hit["created_at"].as_i64().is_some());

        // Snippet present.
        assert!(hit["snippet"].as_str().unwrap().contains("uniqueWireToken"));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Round-22 (§-1.5 PR-5): MCP filter surface — schema + handler.
    // ─────────────────────────────────────────────────────────────────────

    /// `tools/list` must advertise the full filter set so an MCP agent
    /// learning the schema knows it can narrow by kind / scope /
    /// instance / since / until without trial-and-error.
    #[tokio::test]
    async fn tools_list_search_memories_advertises_filter_schema() {
        let s = server_with_records(&[]);
        let resp = s.handle(req("tools/list", Value::Null)).await;
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        let sm = tools
            .iter()
            .find(|t| t["name"] == "search_memories")
            .expect("search_memories must be in tools/list");
        let props = sm["inputSchema"]["properties"].as_object().unwrap();
        for f in [
            "query", "source", "instance", "kind", "scope", "since", "until", "limit", "mode",
        ] {
            assert!(
                props.contains_key(f),
                "search_memories schema missing `{f}` after PR-5"
            );
        }
        // The enums must enumerate the real options so agents can validate
        // locally.
        let kinds = props["kind"]["enum"].as_array().unwrap();
        assert!(kinds.iter().any(|v| v == "preference"));
        let scopes = props["scope"]["enum"].as_array().unwrap();
        assert!(scopes.iter().any(|v| v == "session"));
    }

    /// `search_memories` with `since` set must filter the recall stage,
    /// not just post-filter. Two records at t=2024 and t=2026 with the
    /// same query token; `since=2025-01-01` keeps only the newer.
    #[tokio::test]
    async fn search_memories_since_filters_by_created_at() {
        // 2024-01-01T00:00:00Z = 1704067200
        // 2026-01-01T00:00:00Z = 1767225600
        let old = make_record(
            "claude-code",
            "r-old",
            "filterTokenAlpha shared snippet content",
            1704067200,
        );
        let new = make_record(
            "claude-code",
            "r-new",
            "filterTokenAlpha shared snippet content",
            1767225600,
        );
        let s = server_with_records(&[old, new]);
        let resp = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "search_memories",
                    "arguments": {
                        "query": "filterTokenAlpha",
                        "mode": "fulltext",
                        "since": "2025-01-01T00:00:00Z",
                        "limit": 10
                    }
                }),
            ))
            .await;
        let results = resp.result.unwrap()["structuredContent"]["results"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(
            results.len(),
            1,
            "since-filter must drop the 2024 record; got {results:#?}"
        );
        assert!(!results[0]["record_id"].as_str().unwrap().is_empty());
    }

    /// `until` is the upper bound — same fixture, `until=2025-01-01`
    /// keeps only the OLDER record.
    #[tokio::test]
    async fn search_memories_until_filters_by_created_at() {
        let old = make_record(
            "claude-code",
            "r-old",
            "filterTokenBeta shared snippet content",
            1704067200,
        );
        let new = make_record(
            "claude-code",
            "r-new",
            "filterTokenBeta shared snippet content",
            1767225600,
        );
        let s = server_with_records(&[old, new]);
        let resp = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "search_memories",
                    "arguments": {
                        "query": "filterTokenBeta",
                        "mode": "fulltext",
                        "until": "2025-01-01T00:00:00Z",
                        "limit": 10
                    }
                }),
            ))
            .await;
        let results = resp.result.unwrap()["structuredContent"]["results"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(results.len(), 1);
    }

    /// Malformed timestamps must surface a clean error, not silently
    /// fall back to "no filter".
    #[tokio::test]
    async fn search_memories_since_rejects_non_rfc3339() {
        let s = server_with_records(&[]);
        let resp = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "search_memories",
                    "arguments": {
                        "query": "anything",
                        "since": "yesterday",
                    }
                }),
            ))
            .await;
        let err_msg = resp
            .error
            .as_ref()
            .map(|e| e.message.clone())
            .unwrap_or_default();
        assert!(
            err_msg.contains("RFC3339") || err_msg.contains("since"),
            "expected clear RFC3339 error; got: {err_msg:?}"
        );
    }

    #[tokio::test]
    async fn get_record_returns_record_or_null() {
        let r = make_record("claude-code", "x", "y", 1700000000);
        let id = r.id.0.clone();
        let s = server_with_records(&[r]);

        let hit = s
            .handle(req(
                "tools/call",
                json!({"name": "get_record", "arguments": {"id": id}}),
            ))
            .await;
        let result = hit.result.unwrap();
        assert_eq!(result["structuredContent"]["content"], "y");

        let miss = s
            .handle(req(
                "tools/call",
                json!({"name": "get_record", "arguments": {"id": "no-such-id"}}),
            ))
            .await;
        assert_eq!(miss.result.unwrap()["structuredContent"], Value::Null);
    }

    /// Round-11: `get_record` wire format. The agent contract must
    /// match `search_memories` (PR-#16) for the overlapping fields and
    /// add chunk/embedding readiness so the agent can decide whether
    /// hybrid retrieval will actually hit this record right now.
    #[tokio::test]
    async fn get_record_emits_normalized_payload_with_readiness() {
        // `make_record` defaults kind=Fact, scope=User, instance=None.
        let r = make_record("claude-code", "abc", "hello world", 1700000000);
        let id = r.id.0.clone();
        let s = server_with_records(&[r]);

        let resp = s
            .handle(req(
                "tools/call",
                json!({"name": "get_record", "arguments": {"id": id.clone()}}),
            ))
            .await;
        let p = &resp.result.unwrap()["structuredContent"];

        // Identity / agent-chain fields.
        assert_eq!(p["record_id"], id);
        assert_eq!(
            p["trace_id"], id,
            "trace_id alias mirrors search_memories so agents can hand it back to trace_provenance"
        );

        // Normalised enums — lower-case strings, not Debug-style Capitalised.
        assert_eq!(p["kind"], "fact", "kind must be lower-case");
        assert_eq!(p["scope"], "user", "scope must be lower-case");

        // Default-instance serialises as null on the wire.
        assert_eq!(
            p["instance"],
            Value::Null,
            "default instance must be JSON null, not empty string"
        );

        // Content + provenance.
        assert_eq!(p["content"], "hello world");
        assert_eq!(p["adapter"], "claude-code");
        assert_eq!(p["native_id"], "abc");

        // Readiness — the round-11 load-bearing piece.
        assert!(
            p["chunk_count"].as_u64().unwrap() >= 1,
            "the upserted record has at least one chunk"
        );
        // No active model was set by the test helper, so the embedded
        // count is 0 and active_model is null. Tested separately below.
        assert_eq!(p["embedded_chunk_count"], 0);
        assert_eq!(p["active_model"], Value::Null);

        // Source-vector breadcrumb absent on a synthetic record.
        assert_eq!(p["source_embedding_model"], Value::Null);
        assert_eq!(p["source_embedding_dim"], Value::Null);

        // Time fields use unix-epoch numbers, not RFC3339 strings.
        assert_eq!(p["created_at"].as_i64(), Some(1700000000));
    }

    #[tokio::test]
    async fn get_record_embedded_chunk_count_tracks_active_model() {
        // The "is this record ready for vector search?" signal only
        // counts embeddings under the CURRENTLY-ACTIVE model. An
        // embedding produced under a previous model must not count.
        //
        // We walk the real path (set model → upsert → claim → complete →
        // switch model) so the test is end-to-end honest — no hand-rolled
        // SQL bypassing the store API.
        let r = make_record("claude-code", "x", "a", 1700000000);
        let id = r.id.0.clone();
        let store = Store::open_in_memory().unwrap();
        let chunks = Chunker::default().chunk(&r.id, &r.content);

        // First: a stale model — set it active, upsert (enqueues
        // jobs under it), claim, complete (writes chunk_embeddings).
        store.set_active_model("stale-model").unwrap();
        store.upsert_record(&r, &chunks, None).unwrap();
        let job = store
            .claim_next_job("stale-model")
            .unwrap()
            .expect("upsert must enqueue at least one job");
        store.complete_job(&job, &[0.1, 0.2, 0.3]).unwrap();

        // Now switch to a different active model. The freshly-written
        // chunk_embeddings row stays — but it's now stale.
        store.set_active_model("active-model").unwrap();
        store
            .register_source("claude-code", None, Some("/tmp/x"), None)
            .unwrap();

        let s = AnamnesisServer::new(store, None, std::env::temp_dir());
        let resp = s
            .handle(req(
                "tools/call",
                json!({"name": "get_record", "arguments": {"id": id}}),
            ))
            .await;
        let p = &resp.result.unwrap()["structuredContent"];
        assert_eq!(p["active_model"], "active-model");
        assert_eq!(
            p["embedded_chunk_count"], 0,
            "embeddings under stale-model must NOT count toward active-model readiness"
        );
    }

    #[tokio::test]
    async fn trace_provenance_returns_native_path_and_hash() {
        let r = make_record("claude-code", "abc", "y", 1700000000);
        let id = r.id.0.clone();
        let s = server_with_records(&[r]);
        let resp = s
            .handle(req(
                "tools/call",
                json!({"name": "trace_provenance", "arguments": {"id": id}}),
            ))
            .await;
        let structured = &resp.result.unwrap()["structuredContent"];
        assert_eq!(structured["adapter"], "claude-code");
        assert_eq!(structured["native_id"], "abc");
        assert_eq!(structured["native_path"], "/p/abc");
        assert_eq!(structured["raw_hash"], "h");
        // Back-compat: record_id call must NOT include chunk-level extras.
        assert!(structured.get("chunk_id").is_none());
        assert!(structured.get("chunk_content").is_none());
    }

    /// Round-10: trace_provenance now accepts `chunk_id` to chain
    /// directly from a `search_memories` hit into the matched chunk's
    /// raw text + jieba-tokenized form. The record-level fields are
    /// still surfaced so an agent gets one canonical provenance shape
    /// either way.
    #[tokio::test]
    async fn trace_provenance_accepts_chunk_id_and_returns_chunk_content() {
        // 含中文，便于断言 tokenized 字符串与 raw 不同（jieba 已切词）。
        let r = make_record("claude-code", "abc", "项目偏好 — 用户喜欢 vim", 1700000000);
        let s = server_with_records(&[r]);

        // First call search_memories to discover the chunk_id the way
        // a real agent would (we don't want to hard-code the format).
        let search_resp = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "search_memories",
                    "arguments": {"query": "项目偏好", "mode": "fulltext", "limit": 1}
                }),
            ))
            .await;
        let chunk_id = search_resp.result.as_ref().unwrap()["structuredContent"]["results"][0]
            ["chunk_id"]
            .as_str()
            .unwrap()
            .to_string();

        // Trace via chunk_id.
        let resp = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "trace_provenance",
                    "arguments": {"chunk_id": chunk_id}
                }),
            ))
            .await;
        let structured = &resp.result.unwrap()["structuredContent"];

        // Record-level provenance is still there.
        assert_eq!(structured["adapter"], "claude-code");
        assert_eq!(structured["native_id"], "abc");
        assert_eq!(structured["native_path"], "/p/abc");
        assert_eq!(structured["raw_hash"], "h");

        // Chunk-level extras are now there too.
        assert!(structured["chunk_id"].as_str().unwrap().contains(':'));
        assert!(structured["chunk_seq"].as_u64().is_some());
        let chunk_content = structured["chunk_content"].as_str().unwrap();
        assert!(chunk_content.contains("项目偏好"));
        let tokenized = structured["chunk_content_tokenized"].as_str().unwrap();
        // Jieba should have segmented the Chinese — tokenized must
        // be space-delimited and contain the multi-char Chinese token
        // somewhere.
        assert!(
            tokenized.contains(' ') || tokenized.chars().count() < chunk_content.chars().count(),
            "tokenized form should be space-joined jieba tokens, got {tokenized:?}"
        );
        assert!(structured["chunk_token_estimate"].as_u64().is_some());
    }

    #[tokio::test]
    async fn trace_provenance_chunk_id_unknown_returns_error() {
        let r = make_record("claude-code", "abc", "y", 1700000000);
        let s = server_with_records(&[r]);
        let resp = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "trace_provenance",
                    "arguments": {"chunk_id": "no-such-chunk:0"}
                }),
            ))
            .await;
        let err = resp.error.unwrap();
        assert!(err.message.contains("chunk not found"));
    }

    #[tokio::test]
    async fn trace_provenance_requires_id_or_chunk_id() {
        let r = make_record("claude-code", "abc", "y", 1700000000);
        let s = server_with_records(&[r]);
        let resp = s
            .handle(req(
                "tools/call",
                json!({"name": "trace_provenance", "arguments": {}}),
            ))
            .await;
        let err = resp.error.unwrap();
        assert!(err.message.contains("requires either"));
    }

    #[tokio::test]
    async fn resources_list_includes_all_three() {
        let s = server_with_records(&[]);
        let resp = s.handle(req("resources/list", Value::Null)).await;
        let resources = resp.result.unwrap()["resources"]
            .as_array()
            .unwrap()
            .clone();
        let uris: Vec<&str> = resources.iter().filter_map(|r| r["uri"].as_str()).collect();
        assert!(uris.iter().any(|u| u.contains("record")));
        assert!(uris.iter().any(|u| u.contains("source")));
        assert!(uris.iter().any(|u| u.contains("timeline")));
    }

    #[tokio::test]
    async fn resource_record_uri_returns_record() {
        let r = make_record("claude-code", "x", "hi", 1700000000);
        let id = r.id.0.clone();
        let s = server_with_records(&[r]);
        let uri = format!("anamnesis://record/{id}");
        let resp = s.handle(req("resources/read", json!({"uri": uri}))).await;
        let result = resp.result.unwrap();
        let text = result["contents"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["content"], "hi");
    }

    #[tokio::test]
    async fn resource_source_uri_returns_recent_records() {
        let r = make_record("claude-code", "x", "hi", 1700000000);
        let s = server_with_records(&[r]);
        let resp = s
            .handle(req(
                "resources/read",
                json!({"uri": "anamnesis://source/claude-code"}),
            ))
            .await;
        let result = resp.result.unwrap();
        let text = result["contents"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert!(!parsed["sources"].as_array().unwrap().is_empty());
        assert_eq!(parsed["recent"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn resource_timeline_uri_filters_by_date() {
        let r1 = make_record("claude-code", "x", "yesterday", 1700000000);
        let r2 = make_record("claude-code", "y", "yesterday too", 1700001000);
        let r3 = make_record("claude-code", "z", "different day", 1700200000);
        let s = server_with_records(&[r1, r2, r3]);

        let day1 = chrono::Utc
            .timestamp_opt(1700000000, 0)
            .unwrap()
            .format("%Y-%m-%d")
            .to_string();
        let uri = format!("anamnesis://timeline/{day1}");
        let resp = s.handle(req("resources/read", json!({"uri": uri}))).await;
        let text = resp.result.unwrap()["contents"][0]["text"]
            .as_str()
            .unwrap()
            .to_owned();
        let parsed: Value = serde_json::from_str(&text).unwrap();
        // Two records fall in the same UTC day; the third belongs to
        // another day.
        assert_eq!(parsed["count"], 2);
    }

    #[tokio::test]
    async fn prompts_list_includes_both_prompts() {
        let s = server_with_records(&[]);
        let resp = s.handle(req("prompts/list", Value::Null)).await;
        let payload = resp.result.unwrap();
        let names: Vec<&str> = payload["prompts"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|p| p["name"].as_str())
            .collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"summarize_my_preferences"));
        assert!(names.contains(&"find_related"));
    }

    /// Round 111 (PR-78ag): `prompts/list` carries a top-level
    /// `summary` line so an agent doing first-time discovery
    /// can decide which prompt to fetch without parsing every
    /// per-arg description. Additive on top of the MCP spec —
    /// existing R0+ consumers reading `prompts[]` are
    /// unaffected.
    #[tokio::test]
    async fn prompts_list_carries_top_level_summary_line() {
        let s = server_with_records(&[]);
        let resp = s.handle(req("prompts/list", Value::Null)).await;
        let payload = resp.result.unwrap();
        let summary = payload["summary"]
            .as_str()
            .expect("prompts/list must carry a top-level `summary` field for agent discovery");
        // Must mention both prompts by name + the prompt count
        // so the line is self-describing.
        assert!(
            summary.contains("2 prompts"),
            "summary should declare the prompt count: {summary}"
        );
        assert!(
            summary.contains("summarize_my_preferences"),
            "summary should name `summarize_my_preferences`: {summary}"
        );
        assert!(
            summary.contains("find_related"),
            "summary should name `find_related`: {summary}"
        );
        // `prompts[]` must continue to carry both entries
        // (back-compat).
        let names: Vec<&str> = payload["prompts"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|p| p["name"].as_str())
            .collect();
        assert_eq!(names.len(), 2);
    }

    #[tokio::test]
    async fn prompt_summarize_preferences_renders_user_scope_records() {
        let r1 = make_record(
            "claude-code",
            "p1",
            "User prefers thorough error handling",
            1700000000,
        );
        let r2 = make_record(
            "mem0",
            "p2",
            "User likes integration tests against real DB",
            1700001000,
        );
        let s = server_with_records(&[r1, r2]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({"name": "summarize_my_preferences", "arguments": {"limit": 10}}),
            ))
            .await;
        let result = resp.result.unwrap();
        let text = result["messages"][0]["content"]["text"].as_str().unwrap();
        assert!(text.contains("Summarize"));
        assert!(text.contains("thorough error handling"));
        assert!(text.contains("real DB"));
    }

    // ─── Round-94 PR-78p: summarize_my_preferences user_tag ──────

    /// `user_tag` narrows the preference summary to records
    /// carrying the named user tag. Filter is normalised (so
    /// `Keep-Forever` matches `keep-forever`) and pushes down
    /// before the LIMIT clause — a single tagged record
    /// surfaces even under a heavy untagged-majority corpus.
    #[tokio::test]
    async fn prompt_summarize_preferences_honors_user_tag_filter() {
        let r1 = make_record(
            "claude-code",
            "p-tagged",
            "User prefers uniquePrefMarkerTAG",
            1700000000,
        );
        let r2 = make_record(
            "mem0",
            "p-untagged",
            "User likes uniquePrefMarkerUNTAG",
            1700001000,
        );
        let s = server_with_records(&[r1, r2]);
        s.store
            .tag_record(
                &anamnesis_core::model::RecordId::from_parts("claude-code", None, "p-tagged"),
                &["keep-forever".into()],
                anamnesis_store::UserTagOperation::Add,
            )
            .unwrap();

        let resp = s
            .handle(req(
                "prompts/get",
                json!({
                    "name": "summarize_my_preferences",
                    "arguments": {"limit": 10, "user_tag": "Keep-Forever"},
                }),
            ))
            .await;
        let text = resp.result.unwrap()["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            text.contains("uniquePrefMarkerTAG"),
            "tagged record must surface: {text}"
        );
        assert!(
            !text.contains("uniquePrefMarkerUNTAG"),
            "untagged record must be filtered out: {text}"
        );
    }

    /// Bogus `user_tag` (control character) is rejected with a
    /// clear error referencing the parameter name.
    #[tokio::test]
    async fn prompt_summarize_preferences_user_tag_rejects_invalid_input() {
        let s = server_with_records(&[]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({
                    "name": "summarize_my_preferences",
                    "arguments": {"user_tag": "bad\nnewline"},
                }),
            ))
            .await;
        assert!(resp.error.is_some(), "must reject bad user_tag");
        let msg = resp.error.unwrap().message;
        assert!(
            msg.contains("user_tag"),
            "error must mention parameter name; got {msg}"
        );
    }

    /// `prompts/list` advertises the new `user_tag` arg.
    #[tokio::test]
    async fn prompts_list_advertises_summarize_my_preferences_user_tag_arg() {
        let s = server_with_records(&[]);
        let resp = s.handle(req("prompts/list", Value::Null)).await;
        let prompts = resp.result.unwrap()["prompts"].as_array().unwrap().clone();
        let p = prompts
            .iter()
            .find(|p| p["name"] == "summarize_my_preferences")
            .expect("summarize_my_preferences must be in prompts/list");
        let args = p["arguments"].as_array().unwrap();
        assert!(
            args.iter().any(|a| a["name"] == "user_tag"),
            "must advertise user_tag: {args:?}"
        );
    }

    // ─── Round-101 PR-78w: summarize_my_preferences summary line ─

    /// Default `summarize_my_preferences` prompt now carries a
    /// `Summary: bullets=N` line above the bullets — symmetric
    /// with R100's `find_related` summary so the LLM sees the
    /// same structured prelude on both prompts.
    #[tokio::test]
    async fn prompt_summarize_preferences_includes_bullet_count_summary() {
        let r1 = make_record("claude-code", "p1", "thorough error handling", 1700000000);
        let r2 = make_record("mem0", "p2", "integration tests over mocks", 1700001000);
        let s = server_with_records(&[r1, r2]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({"name": "summarize_my_preferences", "arguments": {"limit": 10}}),
            ))
            .await;
        let text = resp.result.unwrap()["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            text.contains("Summary: bullets=2"),
            "summary line missing/wrong: {text}"
        );
        // No user_tag arg → no user_tag clause in the summary.
        assert!(
            !text.contains("user_tag="),
            "user_tag absent should omit the tag clause: {text}"
        );
    }

    /// When `user_tag` is set the summary reports the filter tag
    /// and the matched count. Because the store helper already
    /// filtered by tag at the SQL recall stage, `matched_user_tag
    /// = bullets`.
    #[tokio::test]
    async fn prompt_summarize_preferences_summary_reports_user_tag_matches() {
        let r1 = make_record(
            "claude-code",
            "pref-tagged",
            "User prefers uniquePrefSummary",
            1700000000,
        );
        let r2 = make_record(
            "mem0",
            "pref-untagged",
            "User likes uniquePrefSummaryUntag",
            1700001000,
        );
        let s = server_with_records(&[r1, r2]);
        s.store
            .tag_record(
                &anamnesis_core::model::RecordId::from_parts("claude-code", None, "pref-tagged"),
                &["keep-forever".into()],
                anamnesis_store::UserTagOperation::Add,
            )
            .unwrap();

        let resp = s
            .handle(req(
                "prompts/get",
                json!({
                    "name": "summarize_my_preferences",
                    "arguments": {"limit": 10, "user_tag": "Keep-Forever"},
                }),
            ))
            .await;
        let text = resp.result.unwrap()["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("Summary: bullets=1"), "got {text}");
        assert!(text.contains("user_tag=\"keep-forever\""), "got {text}");
        assert!(text.contains("matched_user_tag=1"), "got {text}");
    }

    /// Empty result still emits `Summary: bullets=0` — same
    /// structured-zero discipline as R100. The original
    /// `(no user-scope records yet)` placeholder is preserved.
    #[tokio::test]
    async fn prompt_summarize_preferences_summary_reports_zero_when_no_records() {
        let s = server_with_records(&[]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({"name": "summarize_my_preferences"}),
            ))
            .await;
        let text = resp.result.unwrap()["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("Summary: bullets=0"), "got {text}");
        assert!(text.contains("(no user-scope records yet)"));
    }

    #[tokio::test]
    async fn prompt_find_related_returns_top_n_with_text_arg() {
        let r = make_record("claude-code", "x", "alpha bright morning", 1700000000);
        let s = server_with_records(&[r]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({"name": "find_related", "arguments": {"text": "alpha", "limit": 3}}),
            ))
            .await;
        let result = resp.result.unwrap();
        let text = result["messages"][0]["content"]["text"].as_str().unwrap();
        assert!(text.contains("most relevant memories"));
        assert!(text.contains("alpha bright morning"));
    }

    /// Round-65: `source` filter scopes find_related to one adapter.
    /// Two records, one per adapter, both containing "alpha"; filtering
    /// by `source = "claude-code"` should only surface the claude-code
    /// memory in the rendered prompt.
    #[tokio::test]
    async fn prompt_find_related_honors_source_filter() {
        let r1 = make_record(
            "claude-code",
            "x1",
            "alpha bright morning from claude",
            1700000000,
        );
        let r2 = make_record("mem0", "x2", "alpha bright morning from mem0", 1700000010);
        let s = server_with_records(&[r1, r2]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({
                    "name": "find_related",
                    "arguments": {
                        "text": "alpha",
                        "limit": 10,
                        "source": "claude-code",
                    }
                }),
            ))
            .await;
        let result = resp.result.unwrap();
        let text = result["messages"][0]["content"]["text"].as_str().unwrap();
        assert!(
            text.contains("from claude"),
            "claude-code memory should be present"
        );
        assert!(
            !text.contains("from mem0"),
            "mem0 memory must be filtered out: {text}"
        );
    }

    // ─── Round-100 PR-78v: find_related summary line ─────────────

    /// Default `find_related` prompt now carries a compact
    /// `Summary: bullets=N` line above the bullets so the LLM
    /// sees at a glance how many memories were attached.
    #[tokio::test]
    async fn prompt_find_related_includes_bullet_count_summary() {
        let r1 = make_record("claude-code", "a", "alpha bright morning", 1700000000);
        let r2 = make_record("mem0", "b", "alpha bright evening", 1700000010);
        let s = server_with_records(&[r1, r2]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({"name": "find_related", "arguments": {"text": "alpha", "limit": 10}}),
            ))
            .await;
        let text = resp.result.unwrap()["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            text.contains("Summary: bullets=2"),
            "summary line missing or wrong: {text}"
        );
        // No user_tag arg → no user_tag clause in the summary.
        assert!(
            !text.contains("user_tag="),
            "user_tag absent should omit the tag clause: {text}"
        );
    }

    /// When `user_tag` is set the summary also reports the
    /// filter tag and how many of the surfaced bullets actually
    /// carry that tag. Useful for the LLM to gauge filter
    /// strength without parsing the bullets.
    #[tokio::test]
    async fn prompt_find_related_summary_reports_user_tag_matches() {
        let r = make_record(
            "claude-code",
            "tagged-x",
            "alpha bright morning",
            1700000000,
        );
        let s = server_with_records(&[r]);
        s.store
            .tag_record(
                &anamnesis_core::model::RecordId::from_parts("claude-code", None, "tagged-x"),
                &["keep-forever".into()],
                anamnesis_store::UserTagOperation::Add,
            )
            .unwrap();

        let resp = s
            .handle(req(
                "prompts/get",
                json!({
                    "name": "find_related",
                    "arguments": {"text": "alpha", "user_tag": "Keep-Forever"},
                }),
            ))
            .await;
        let text = resp.result.unwrap()["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("Summary: bullets=1"), "got {text}");
        assert!(text.contains("user_tag=\"keep-forever\""), "got {text}");
        assert!(text.contains("matched_user_tag=1"), "got {text}");
    }

    /// Empty result still gets a summary line — `bullets=0`
    /// reaches the LLM as a structured zero, not implicit
    /// silence.
    #[tokio::test]
    async fn prompt_find_related_summary_reports_zero_when_no_matches() {
        let s = server_with_records(&[]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({"name": "find_related", "arguments": {"text": "nothing"}}),
            ))
            .await;
        let text = resp.result.unwrap()["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("Summary: bullets=0"), "got {text}");
        assert!(text.contains("(no related memories found)"));
    }

    // ─── Round-89 PR-78k: find_related explain ────────────────────

    /// Default `find_related` (no `explain` arg) does NOT carry
    /// the numeric breakdown — back-compat with every existing
    /// MCP client + every prompt agent that's already wired.
    #[tokio::test]
    async fn prompt_find_related_default_has_no_explain_block() {
        let r = make_record("claude-code", "x", "alpha bright morning", 1700000000);
        let s = server_with_records(&[r]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({"name": "find_related", "arguments": {"text": "alpha"}}),
            ))
            .await;
        let text = resp.result.unwrap()["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            !text.contains("explain:"),
            "default find_related must not emit explain breakdown; got {text}"
        );
    }

    /// `explain: true` appends a compact text breakdown to each
    /// bullet. Asserts the line contains `record_score=` and
    /// `fts_rank=` (the FTS-only test fixture has no vector stage).
    #[tokio::test]
    async fn prompt_find_related_explain_emits_compact_breakdown() {
        let r = make_record("claude-code", "x", "alpha bright morning", 1700000000);
        let s = server_with_records(&[r]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({
                    "name": "find_related",
                    "arguments": {"text": "alpha", "explain": true},
                }),
            ))
            .await;
        let text = resp.result.unwrap()["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            text.contains("explain:"),
            "explain bullet must be present; got {text}"
        );
        assert!(text.contains("record_score="));
        assert!(text.contains("best_chunk_rrf_score="));
        assert!(text.contains("kind_boost="));
        assert!(text.contains("fts_rank="));
        assert!(text.contains("rrf_k="));
    }

    /// `prompts/list` advertises the new `explain` argument so
    /// MCP clients can introspect.
    #[tokio::test]
    async fn prompts_list_advertises_find_related_explain_arg() {
        let s = server_with_records(&[]);
        let resp = s.handle(req("prompts/list", Value::Null)).await;
        let prompts = resp.result.unwrap()["prompts"].as_array().unwrap().clone();
        let find_related = prompts
            .iter()
            .find(|p| p["name"] == "find_related")
            .expect("find_related must be in prompts/list");
        let args = find_related["arguments"].as_array().unwrap();
        assert!(
            args.iter().any(|a| a["name"] == "explain"),
            "find_related must advertise `explain` arg: {args:?}"
        );
    }

    // ─── Round-93 PR-78o: find_related user_tag filter ────────────

    /// `user_tag` narrows `find_related` to records carrying the
    /// named user tag. Without the filter, both records hit; with
    /// the filter, only the tagged record surfaces. Pin the
    /// normalisation: passing `Keep-Forever` matches a record
    /// tagged `keep-forever`.
    #[tokio::test]
    async fn prompt_find_related_honors_user_tag_filter_with_normalisation() {
        // Use distinct body content per record so we can verify
        // which one surfaces in the prompt bullets — the
        // make_record helper plain-passes content through to the
        // bullet snippet via the search/pack path.
        let r1 = make_record(
            "claude-code",
            "tagged-rec",
            "alpha bright morning uniqueTaggedMarker",
            1700000000,
        );
        let r2 = make_record(
            "claude-code",
            "untagged-rec",
            "alpha bright morning uniqueUntaggedMarker",
            1700000010,
        );
        let s = server_with_records(&[r1, r2]);
        // Apply a user tag to the first record directly via the
        // store. The MCP filter pushes the same normalisation
        // through, so passing `Keep-Forever` must match.
        s.store
            .tag_record(
                &anamnesis_core::model::RecordId::from_parts("claude-code", None, "tagged-rec"),
                &["keep-forever".into()],
                anamnesis_store::UserTagOperation::Add,
            )
            .unwrap();

        let resp = s
            .handle(req(
                "prompts/get",
                json!({
                    "name": "find_related",
                    "arguments": {"text": "alpha", "user_tag": "Keep-Forever", "limit": 10},
                }),
            ))
            .await;
        let text = resp.result.unwrap()["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            text.contains("uniqueTaggedMarker"),
            "tagged record should surface: {text}"
        );
        assert!(
            !text.contains("uniqueUntaggedMarker"),
            "untagged record must be filtered out: {text}"
        );
    }

    /// Bogus `user_tag` (control character) is rejected with a
    /// clear error referencing the parameter name — same shape
    /// `search_memories.user_tag` uses.
    #[tokio::test]
    async fn prompt_find_related_user_tag_rejects_invalid_input() {
        let s = server_with_records(&[]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({
                    "name": "find_related",
                    "arguments": {"text": "alpha", "user_tag": "bad\nnewline"},
                }),
            ))
            .await;
        assert!(resp.error.is_some(), "must reject bad user_tag");
        let msg = resp.error.unwrap().message;
        assert!(
            msg.contains("user_tag"),
            "error must mention the parameter name; got {msg}"
        );
    }

    /// `prompts/list` advertises the new `user_tag` arg so MCP
    /// clients can introspect.
    #[tokio::test]
    async fn prompts_list_advertises_find_related_user_tag_arg() {
        let s = server_with_records(&[]);
        let resp = s.handle(req("prompts/list", Value::Null)).await;
        let prompts = resp.result.unwrap()["prompts"].as_array().unwrap().clone();
        let find_related = prompts
            .iter()
            .find(|p| p["name"] == "find_related")
            .expect("find_related must be in prompts/list");
        let args = find_related["arguments"].as_array().unwrap();
        assert!(
            args.iter().any(|a| a["name"] == "user_tag"),
            "find_related must advertise `user_tag` arg: {args:?}"
        );
    }

    /// Round-65: a non-existent `source` filter returns the empty
    /// rendered "(no related memories found)" stub, not the unfiltered
    /// corpus. Guards against silently dropping the filter.
    #[tokio::test]
    async fn prompt_find_related_empty_when_source_does_not_match() {
        let r = make_record("claude-code", "z", "alpha bright morning", 1700000000);
        let s = server_with_records(&[r]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({
                    "name": "find_related",
                    "arguments": {
                        "text": "alpha",
                        "limit": 5,
                        "source": "nonexistent-adapter",
                    }
                }),
            ))
            .await;
        let result = resp.result.unwrap();
        let text = result["messages"][0]["content"]["text"].as_str().unwrap();
        assert!(
            text.contains("(no related memories found)"),
            "unmatched source must short-circuit to empty bullets, got: {text}"
        );
    }

    /// Round-65: the new filter args are advertised in `prompts/list`
    /// so MCP clients know they exist.
    #[tokio::test]
    async fn prompts_list_advertises_find_related_filter_args() {
        let s = server_with_records(&[]);
        let resp = s.handle(req("prompts/list", json!({}))).await;
        let result = resp.result.unwrap();
        let find_related = result["prompts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == "find_related")
            .expect("find_related prompt present");
        let arg_names: Vec<String> = find_related["arguments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["name"].as_str().unwrap_or_default().to_owned())
            .collect();
        for expected in ["text", "limit", "source", "instance", "kind", "scope"] {
            assert!(
                arg_names.iter().any(|n| n == expected),
                "prompts/list should advertise `{expected}`; got {arg_names:?}"
            );
        }
    }

    #[tokio::test]
    async fn prompt_find_related_requires_text_arg() {
        let s = server_with_records(&[]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({"name": "find_related", "arguments": {}}),
            ))
            .await;
        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn unknown_prompt_errors() {
        let s = server_with_records(&[]);
        let resp = s
            .handle(req(
                "prompts/get",
                json!({"name": "nonsense", "arguments": {}}),
            ))
            .await;
        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn unknown_resource_kind_errors() {
        let s = server_with_records(&[]);
        let resp = s
            .handle(req(
                "resources/read",
                json!({"uri": "anamnesis://nonsense/x"}),
            ))
            .await;
        assert!(resp.error.is_some());
    }

    // ─── Round-54: `doctor` tool ──────────────────────────────────────

    /// Smoke: doctor returns one row per registered source, with the
    /// summary fields and per-source health detail.
    #[tokio::test]
    async fn doctor_returns_per_source_health_for_registered_sources() {
        let store = Store::open_in_memory().unwrap();
        // Two registered sources — one with a deliberately bogus location
        // so its `health()` flips `ok=false`, plus another with a path
        // that doesn't exist either (claude-code's health checks for
        // projects_root existence).
        store
            .register_source("mem0", None, Some("/nonexistent/mem0.sqlite"), None)
            .unwrap();
        store
            .register_source(
                "claude-code",
                None,
                Some("/nonexistent/claude/projects"),
                None,
            )
            .unwrap();
        let s = AnamnesisServer::new(store, None, std::env::temp_dir());
        let resp = s.handle(req("tools/call", json!({"name": "doctor"}))).await;
        let result = resp.result.expect("doctor must succeed");
        let body = &result["structuredContent"];
        let sources = body["sources"].as_array().unwrap();
        assert_eq!(sources.len(), 2, "expected one row per registered source");
        // Both sources point at nonexistent paths → both unhealthy.
        assert_eq!(body["summary"]["total"], 2);
        assert_eq!(body["summary"]["ok"], 0);
        assert_eq!(body["summary"]["unhealthy"], 2);
        // Every row carries the wire-format keys downstream agents rely on.
        for src in sources {
            for key in [
                "adapter",
                "instance",
                "location",
                "ok",
                "detail",
                "record_count",
                "chunk_count",
                "last_import_at",
                "stale",
            ] {
                assert!(src.get(key).is_some(), "doctor row missing {key}: {src}",);
            }
            // No `since` was passed → stale must be JSON null.
            assert!(src["stale"].is_null(), "stale must be null without since");
        }
    }

    /// `source` filter narrows the result set to one adapter — the
    /// equivalent of CLI `doctor mem0`.
    #[tokio::test]
    async fn doctor_source_filter_narrows_to_one_adapter() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("mem0", None, Some("/nonexistent/mem0.sqlite"), None)
            .unwrap();
        store
            .register_source("claude-code", None, Some("/nonexistent/claude"), None)
            .unwrap();
        let s = AnamnesisServer::new(store, None, std::env::temp_dir());
        let resp = s
            .handle(req(
                "tools/call",
                json!({"name": "doctor", "arguments": {"source": "mem0"}}),
            ))
            .await;
        let sources = resp.result.unwrap()["structuredContent"]["sources"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0]["adapter"], "mem0");
    }

    /// Round 110 (PR-78af): comma-separated `source` is the OR
    /// filter — both adapter rows survive, third drops.
    /// Symmetric with R102 audit-tail / R103 list-sources /
    /// R104 dedupe multi-value pattern.
    #[tokio::test]
    async fn doctor_source_multi_value_or_filters_matching_adapters() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("mem0", None, Some("/nonexistent/mem0.sqlite"), None)
            .unwrap();
        store
            .register_source("claude-code", None, Some("/nonexistent/claude"), None)
            .unwrap();
        store
            .register_source("codex", None, Some("/nonexistent/codex"), None)
            .unwrap();
        let s = AnamnesisServer::new(store, None, std::env::temp_dir());
        let resp = s
            .handle(req(
                "tools/call",
                json!({"name": "doctor", "arguments": {"source": "mem0, claude-code"}}),
            ))
            .await;
        let sources = resp.result.unwrap()["structuredContent"]["sources"]
            .as_array()
            .unwrap()
            .clone();
        let adapters: std::collections::BTreeSet<&str> = sources
            .iter()
            .map(|s| s["adapter"].as_str().unwrap())
            .collect();
        assert_eq!(
            adapters,
            ["claude-code", "mem0"].into_iter().collect(),
            "codex must drop under multi-source OR: got {sources:?}"
        );
    }

    /// Multi-source OR combined with `instance` is AND: row
    /// matches iff `adapter ∈ source-set` AND `instance ==
    /// instance-arg`. mem0:dev survives; mem0:prod drops
    /// (instance mismatch); claude-code with default instance
    /// drops (instance mismatch).
    #[tokio::test]
    async fn doctor_source_multi_value_with_instance_is_and_filter() {
        let store = Store::open_in_memory().unwrap();
        store
            .register_source("mem0", Some("prod"), Some("/nonexistent/a"), None)
            .unwrap();
        store
            .register_source("mem0", Some("dev"), Some("/nonexistent/b"), None)
            .unwrap();
        store
            .register_source("claude-code", None, Some("/nonexistent/cc"), None)
            .unwrap();
        let s = AnamnesisServer::new(store, None, std::env::temp_dir());
        let resp = s
            .handle(req(
                "tools/call",
                json!({"name": "doctor", "arguments": {
                    "source": "mem0,claude-code",
                    "instance": "dev",
                }}),
            ))
            .await;
        let sources = resp.result.unwrap()["structuredContent"]["sources"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(sources.len(), 1, "only mem0:dev should match: {sources:?}");
        assert_eq!(sources[0]["adapter"], "mem0");
        assert_eq!(sources[0]["instance"], "dev");
    }

    /// `since=Nd` marks a source whose `last_import_at` is older than
    /// the threshold (or was never imported) as `stale=true`.
    #[tokio::test]
    async fn doctor_since_marks_stale_when_last_import_older_than_threshold() {
        let store = Store::open_in_memory().unwrap();
        // Source A: never imported → must be stale once `since` is set.
        store
            .register_source("mem0", None, Some("/nonexistent/mem0.sqlite"), None)
            .unwrap();
        // Source B: imported "1 year ago" (way past 7d threshold below).
        store
            .register_source("claude-code", None, Some("/nonexistent/claude"), None)
            .unwrap();
        let one_year_ago = chrono::Utc::now().timestamp() - 365 * 86_400;
        store
            .conn()
            .execute(
                "UPDATE sources SET last_import_at = ?1 WHERE adapter = 'claude-code'",
                rusqlite::params![one_year_ago],
            )
            .unwrap();
        let s = AnamnesisServer::new(store, None, std::env::temp_dir());
        let resp = s
            .handle(req(
                "tools/call",
                json!({"name": "doctor", "arguments": {"since": "7d"}}),
            ))
            .await;
        let body = resp.result.unwrap()["structuredContent"].clone();
        let sources = body["sources"].as_array().unwrap();
        for src in sources {
            assert_eq!(
                src["stale"], true,
                "every source must be stale under since=7d here: {src}",
            );
        }
        assert_eq!(body["summary"]["stale"], 2);
    }

    /// Unknown `since` shape is a -32603 error (wrapped from the
    /// handler's `Err`) — agents must learn the contract not silently
    /// get an all-stale or all-fresh report.
    #[tokio::test]
    async fn doctor_rejects_garbage_since() {
        let s = server_with_records(&[]);
        let resp = s
            .handle(req(
                "tools/call",
                json!({"name": "doctor", "arguments": {"since": "next tuesday"}}),
            ))
            .await;
        let err = resp.error.expect("garbage since must error");
        assert!(
            err.message.contains("since"),
            "error must mention 'since': {}",
            err.message,
        );
    }

    /// Empty registry → empty rows + zeroed summary. Agents should not
    /// confuse "no rows" with "tool failed".
    #[tokio::test]
    async fn doctor_returns_zero_summary_on_empty_registry() {
        let store = Store::open_in_memory().unwrap();
        let s = AnamnesisServer::new(store, None, std::env::temp_dir());
        let resp = s.handle(req("tools/call", json!({"name": "doctor"}))).await;
        let body = resp.result.unwrap()["structuredContent"].clone();
        assert_eq!(body["summary"]["total"], 0);
        assert_eq!(body["summary"]["ok"], 0);
        assert_eq!(body["summary"]["unhealthy"], 0);
        assert_eq!(body["summary"]["stale"], 0);
        assert!(body["sources"].as_array().unwrap().is_empty());
    }

    /// `doctor` is in the default tool catalogue (not admin-gated) so
    /// any MCP client can discover it without `--allow-admin-tools`.
    #[tokio::test]
    async fn doctor_is_visible_to_non_admin_clients() {
        let s = server_with_records(&[]);
        assert!(!s.admin_tools_allowed());
        let resp = s.handle(req("tools/list", Value::Null)).await;
        let names = tool_names_from(&resp.result.unwrap());
        assert!(
            names.contains(&"doctor".to_string()),
            "doctor must be in non-admin tools/list",
        );
    }

    #[test]
    fn parse_doctor_since_accepts_known_shapes() {
        assert_eq!(parse_doctor_since("7d").unwrap(), 7 * 86_400);
        assert_eq!(parse_doctor_since("12h").unwrap(), 12 * 3_600);
        assert_eq!(parse_doctor_since("30m").unwrap(), 30 * 60);
        assert_eq!(parse_doctor_since("90").unwrap(), 90);
        assert!(parse_doctor_since("").is_err());
        assert!(parse_doctor_since("xyz").is_err());
        assert!(parse_doctor_since("-1d").is_err());
    }

    // ─── Round-69: MCP request metrics surfaced via doctor ──────────

    /// After N `search_memories` calls, `doctor` must report exactly
    /// those N requests under `request_metrics.tools[]` — proving the
    /// dispatcher-side instrumentation actually records.
    #[tokio::test]
    async fn doctor_reports_recent_search_memories_metrics() {
        let s = server_with_records(&[make_record("claude-code", "r1", "alpha bench", 1_700)]);
        for _ in 0..3 {
            let _ = s
                .handle(req(
                    "tools/call",
                    json!({
                        "name": "search_memories",
                        "arguments": {"query": "alpha", "mode": "fulltext", "limit": 5}
                    }),
                ))
                .await;
        }
        let resp = s.handle(req("tools/call", json!({"name": "doctor"}))).await;
        let body = resp.result.unwrap();
        let metrics = body["structuredContent"]["request_metrics"].clone();
        assert_eq!(metrics["window_seconds"], 86_400);
        let tools = metrics["tools"].as_array().unwrap();
        // Find search_memories — at minimum present; doctor invocation
        // itself also lands in metrics (and is fine to ignore here).
        let sm = tools
            .iter()
            .find(|t| t["tool"] == "search_memories")
            .expect("search_memories must appear in metrics");
        assert_eq!(sm["count"], 3, "should record exactly 3 search calls");
        assert_eq!(sm["errors"], 0);
        // Percentiles are non-decreasing.
        let p50 = sm["p50_ms"].as_u64().unwrap();
        let p95 = sm["p95_ms"].as_u64().unwrap();
        let p99 = sm["p99_ms"].as_u64().unwrap();
        assert!(p50 <= p95 && p95 <= p99, "p50={p50} p95={p95} p99={p99}");
        assert!(sm["last_result_count"].as_i64().unwrap() >= 1);
    }

    /// Calling an unknown tool name must record an error metric without
    /// throwing — confirms the metric write itself never breaks the
    /// dispatcher's error path.
    #[tokio::test]
    async fn doctor_reports_unknown_tool_as_error() {
        let s = server_with_records(&[]);
        let resp = s
            .handle(req("tools/call", json!({"name": "no_such_tool"})))
            .await;
        assert!(
            resp.error.is_some(),
            "unknown tool must return JSON-RPC error"
        );

        let body = s
            .handle(req("tools/call", json!({"name": "doctor"})))
            .await
            .result
            .unwrap();
        let tools = body["structuredContent"]["request_metrics"]["tools"]
            .as_array()
            .unwrap()
            .clone();
        let bad = tools
            .iter()
            .find(|t| t["tool"] == "no_such_tool")
            .expect("unknown_tool error must be recorded in metrics");
        assert_eq!(bad["errors"], 1);
        assert_eq!(bad["count"], 1);
    }

    /// Existing `doctor` shape (`summary` + `sources`) must remain
    /// unchanged — this is the back-compat guard for any agent that
    /// already pinned to the pre-Round-69 wire format.
    #[tokio::test]
    async fn doctor_preserves_existing_summary_and_sources_shape() {
        let s = server_with_records(&[]);
        let body = s
            .handle(req("tools/call", json!({"name": "doctor"})))
            .await
            .result
            .unwrap();
        let body = &body["structuredContent"];
        assert!(body["summary"].is_object());
        assert!(body["sources"].is_array());
        for key in ["total", "ok", "unhealthy", "stale"] {
            assert!(
                body["summary"][key].is_number(),
                "summary.{key} must still be present"
            );
        }
    }

    // ─── Round-71: search_memories(trace=true) ──────────────────────

    /// Without `trace=true`, the response shape must be byte-identical
    /// to the pre-Round-71 wire — guards against accidentally always
    /// emitting the trace and silently inflating MCP payloads for
    /// every existing client.
    #[tokio::test]
    async fn search_trace_omitted_by_default() {
        let s = server_with_records(&[make_record("claude-code", "r1", "alpha bench", 1_700)]);
        let body = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "search_memories",
                    "arguments": {"query": "alpha", "mode": "fulltext", "limit": 5}
                }),
            ))
            .await
            .result
            .unwrap();
        let payload = &body["structuredContent"];
        assert!(payload["results"].is_array());
        assert!(
            payload.get("trace").is_none(),
            "trace must be absent unless explicitly requested; got {payload}"
        );
    }

    /// With `trace=true`, the response must carry per-stage `stages_ms`
    /// and `counts` blocks alongside the usual `results` array.
    #[tokio::test]
    async fn search_trace_reports_stage_breakdown_when_requested() {
        let s = server_with_records(&[
            make_record("claude-code", "r1", "alpha bench memory", 1_700),
            make_record("claude-code", "r2", "alpha duplicate memory", 1_701),
        ]);
        let body = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "search_memories",
                    "arguments": {
                        "query": "alpha",
                        "mode": "fulltext",
                        "limit": 5,
                        "trace": true
                    }
                }),
            ))
            .await
            .result
            .unwrap();
        let payload = &body["structuredContent"];
        assert!(payload["results"].is_array());
        let trace = &payload["trace"];
        assert_eq!(trace["effective_mode"], "fulltext");
        assert!(trace["candidate_pool"].is_u64());
        // Fulltext mode: fts and rrf and pack ran; embed/vec did not.
        assert!(trace["stages_ms"]["fts"].is_u64());
        assert!(trace["stages_ms"]["embed_query"].is_null());
        assert!(trace["stages_ms"]["vec"].is_null());
        assert!(trace["stages_ms"]["rrf"].is_u64());
        assert!(trace["stages_ms"]["pack"].is_u64());
        // Counts cover both modalities + final shape.
        assert!(trace["counts"]["fts_hits"].as_u64().unwrap() >= 1);
        assert_eq!(trace["counts"]["vec_hits"].as_u64().unwrap(), 0);
        assert!(trace["counts"]["ranked_chunks"].as_u64().unwrap() >= 1);
        assert!(trace["counts"]["returned_records"].as_u64().unwrap() >= 1);
    }

    /// The trace payload must never include the query text or any
    /// snippet/record/chunk identifier. This is the same privacy
    /// contract enforced for R69's `mcp_request_metrics`.
    #[tokio::test]
    async fn search_trace_payload_excludes_user_content() {
        let s = server_with_records(&[make_record(
            "claude-code",
            "r1",
            "the marker phrase wombatFluteSafari should not appear anywhere",
            1_700,
        )]);
        let body = s
            .handle(req(
                "tools/call",
                json!({
                    "name": "search_memories",
                    "arguments": {
                        "query": "wombatFluteSafari",
                        "mode": "fulltext",
                        "limit": 5,
                        "trace": true
                    }
                }),
            ))
            .await
            .result
            .unwrap();
        let trace_str = body["structuredContent"]["trace"].to_string();
        assert!(
            !trace_str.contains("wombatFluteSafari"),
            "query text must not appear in trace payload: {trace_str}"
        );
        assert!(
            !trace_str.contains("marker phrase"),
            "memory content must not appear in trace payload: {trace_str}"
        );
        // Whitelist of allowed top-level keys inside `trace`.
        let trace = &body["structuredContent"]["trace"];
        let allowed = ["effective_mode", "candidate_pool", "stages_ms", "counts"];
        let obj = trace.as_object().unwrap();
        for k in obj.keys() {
            assert!(
                allowed.contains(&k.as_str()),
                "unexpected trace top-level field {k:?}: needs a privacy review"
            );
        }
    }
}
