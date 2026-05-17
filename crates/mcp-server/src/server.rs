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
//!
//! Resources:
//!   anamnesis://record/{id}
//!   anamnesis://source/{adapter}[:instance]
//!   anamnesis://timeline/{YYYY-MM-DD}

use std::path::PathBuf;

use anamnesis_adapter_claude_code::{ClaudeCodeAdapter, ClaudeCodeConfig};
use anamnesis_adapter_codex::codex_adapter;
use anamnesis_adapter_mem0::sqlite_adapter as mem0_sqlite_adapter;
use anamnesis_core::embedding::EmbeddingProvider;
use anamnesis_core::model::RecordId;
use anamnesis_importer::ImportRunner;
use anamnesis_search::{pack, ContextBudget, HybridOpts, HybridSearcher, SearchMode};
use anamnesis_store::Store;
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
pub const ADMIN_TOOLS: &[&str] = &["import_source"];

/// Was this tool tagged as admin?
fn is_admin_tool(name: &str) -> bool {
    ADMIN_TOOLS.contains(&name)
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
            "resources/list" => JsonRpcResponse::ok(id, resources_list_payload()),
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
            None => return JsonRpcResponse::err(id, -32602, "missing tools/call.name"),
        };
        // BLUEPRINT §17.5 PR-A — the load-bearing check.
        //
        // We don't only filter admin tools from `tools/list` because a
        // client may have cached the schema (or hard-coded the name) and
        // call `tools/call` directly. Rejecting at the dispatcher is the
        // only thing that genuinely closes the hole.
        if is_admin_tool(&name) && !self.allow_admin_tools {
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
        let result = match name.as_str() {
            "search_memories" => self.tool_search_memories(args).await,
            "get_record" => self.tool_get_record(args).await,
            "list_sources" => self.tool_list_sources().await,
            "import_source" => self.tool_import_source(args).await,
            "trace_provenance" => self.tool_trace_provenance(args).await,
            other => return JsonRpcResponse::err(id, -32602, format!("unknown tool: {other}")),
        };
        match result {
            Ok(payload) => JsonRpcResponse::ok(
                id,
                json!({
                    "content": [{"type": "text", "text": payload.to_string()}],
                    "structuredContent": payload,
                }),
            ),
            Err(msg) => JsonRpcResponse::err(id, -32603, msg),
        }
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
        let kind = args
            .get("kind")
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase());
        let scope = args
            .get("scope")
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase());
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

        // PR-C: push every filter into the SQL recall stage so a
        // minority-source query (e.g. `source = "mem0"` against a
        // claude-code-dominated corpus) is not silently emptied by
        // post-RRF filtering.
        let filter = anamnesis_store::SearchFilter {
            source: source.clone(),
            instance: None,
            kind,
            scope,
            time_from: None,
            time_to: None,
        };

        let store = &self.store;
        let opts = HybridOpts {
            limit,
            candidate_pool: (limit * 4).max(limit),
            mode,
        };
        let hits = match self.provider.as_ref() {
            Some(p) => HybridSearcher::new(p.as_ref())
                .search_filtered(store, query, &filter, &opts)
                .await
                .map_err(|e| format!("search: {e}"))?,
            None => HybridSearcher::<NoProvider>::fulltext_only()
                .search_filtered(store, query, &filter, &opts.fulltext_fallback())
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
        Ok(json!({
            "results": filtered.iter().map(|p| {
                let best = p.matched_chunks.first();
                json!({
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
                })
            }).collect::<Vec<_>>()
        }))
    }

    async fn tool_get_record(&self, args: Value) -> Result<Value, String> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "get_record.id is required".to_string())?;
        let store = &self.store;
        let rec = store
            .get_record(&RecordId(id.to_string()))
            .map_err(|e| format!("store: {e}"))?;
        match rec {
            Some(r) => serde_json::to_value(&r).map_err(|e| format!("serialize: {e}")),
            None => Ok(Value::Null),
        }
    }

    async fn tool_list_sources(&self) -> Result<Value, String> {
        let store = &self.store;
        let stats = store.stats().map_err(|e| format!("stats: {e}"))?;
        let rows = store.list_sources().map_err(|e| format!("list: {e}"))?;
        Ok(json!({
            "sources": rows.iter().map(|(adapter, instance, location)| json!({
                "adapter": adapter,
                "instance": if instance.is_empty() { Value::Null } else { Value::String(instance.clone()) },
                "location": location,
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

    async fn tool_import_source(&self, args: Value) -> Result<Value, String> {
        let adapter_id = args
            .get("adapter")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "import_source.adapter is required".to_string())?;
        let instance = args.get("instance").and_then(|v| v.as_str());
        let path_override = args.get("path").and_then(|v| v.as_str()).map(PathBuf::from);

        match adapter_id {
            anamnesis_adapter_claude_code::ADAPTER_ID => {
                let projects_root =
                    path_override.unwrap_or_else(|| self.home().join(".claude").join("projects"));
                let adapter = ClaudeCodeAdapter::new(ClaudeCodeConfig {
                    projects_root,
                    instance: instance.map(str::to_owned),
                });
                let store = &self.store;
                let summary = ImportRunner::new(&adapter)
                    .run(store)
                    .await
                    .map_err(|e| format!("import: {e}"))?;
                Ok(serde_json::to_value(summary).map_err(|e| e.to_string())?)
            }
            anamnesis_adapter_mem0::ADAPTER_ID => {
                let db_path =
                    path_override.unwrap_or_else(|| self.home().join(".mem0").join("db.sqlite"));
                let adapter = mem0_sqlite_adapter(db_path, instance);
                let store = &self.store;
                let summary = ImportRunner::new(&adapter)
                    .run(store)
                    .await
                    .map_err(|e| format!("import: {e}"))?;
                Ok(serde_json::to_value(summary).map_err(|e| e.to_string())?)
            }
            anamnesis_adapter_codex::ADAPTER_ID => {
                let root = path_override.unwrap_or_else(|| self.home().join(".codex"));
                let adapter = codex_adapter(root, instance);
                let store = &self.store;
                let summary = ImportRunner::new(&adapter)
                    .run(store)
                    .await
                    .map_err(|e| format!("import: {e}"))?;
                Ok(serde_json::to_value(summary).map_err(|e| e.to_string())?)
            }
            other => Err(format!("unknown adapter: {other}")),
        }
    }

    async fn tool_trace_provenance(&self, args: Value) -> Result<Value, String> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "trace_provenance.id is required".to_string())?;
        let store = &self.store;
        let rec = store
            .get_record(&RecordId(id.to_string()))
            .map_err(|e| format!("store: {e}"))?
            .ok_or_else(|| format!("record not found: {id}"))?;
        Ok(json!({
            "record_id": rec.id.0,
            "adapter": rec.source.adapter,
            "instance": rec.source.instance,
            "native_id": rec.provenance.native_id,
            "native_path": rec.provenance.native_path,
            "captured_at": rec.provenance.captured_at,
            "raw_hash": rec.provenance.raw_hash,
        }))
    }
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
        let store = &self.store;
        let conn = store.conn();
        let mut stmt = conn
            .prepare(
                "SELECT id, content, kind, native_path, created_at FROM records \
                 WHERE scope = 'user' ORDER BY created_at DESC LIMIT ?1",
            )
            .map_err(|e| format!("prepare: {e}"))?;
        let rows: Vec<(String, String, String, Option<String>, i64)> = stmt
            .query_map([limit], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, i64>(4)?,
                ))
            })
            .map_err(|e| format!("query: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        let mut bullets = String::new();
        for (id, content, kind, path, _) in &rows {
            bullets.push_str(&format!(
                "- [{kind}] {content_short}  (id={id_short}, source={src})\n",
                content_short = trim_for_prompt(content, 240),
                id_short = &id[..id.len().min(12)],
                src = path.as_deref().unwrap_or("?"),
            ));
        }
        if bullets.is_empty() {
            bullets.push_str("(no user-scope records yet)\n");
        }

        let user_text = format!(
            "Below are the user's stable preferences and personal facts that we have on file. \
             Summarize them into 5–8 concise bullet points capturing what an AI assistant \
             should consistently keep in mind when collaborating with this user. Group related \
             items, preserve any explicit dos/don'ts, and surface contradictions if any.\n\n\
             ---\n{bullets}",
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

        let store = &self.store;
        let opts = HybridOpts {
            limit,
            candidate_pool: (limit * 4).max(limit),
            mode: SearchMode::Hybrid,
        };
        let hits = match self.provider.as_ref() {
            Some(p) => HybridSearcher::new(p.as_ref())
                .search(store, text, &opts)
                .await
                .map_err(|e| format!("search: {e}"))?,
            None => HybridSearcher::<NoProvider>::fulltext_only()
                .search(store, text, &opts.fulltext_fallback())
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
        }
        if bullets.is_empty() {
            bullets.push_str("(no related memories found)\n");
        }

        let user_text = format!(
            "The user is currently working on / discussing the following:\n\n{text}\n\n\
             Here are the most relevant memories Anamnesis has on file. Cite them when they \
             contradict or reinforce what the user is asking. Don't repeat verbatim; weave them \
             into your reply where useful.\n\n---\n{bullets}",
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
    json!({
        "prompts": [
            {
                "name": "summarize_my_preferences",
                "description": "Summarize the user's stable preferences from Anamnesis user-scope records.",
                "arguments": [
                    {"name": "limit", "description": "Max records to include (default 20)", "required": false}
                ]
            },
            {
                "name": "find_related",
                "description": "Inject the top-N Anamnesis memories related to a free-text description.",
                "arguments": [
                    {"name": "text", "description": "What the user is working on or asking about", "required": true},
                    {"name": "limit", "description": "Max related memories to include (default 5)", "required": false}
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
                "description": "Hybrid search across all imported records (FTS + vector + RRF).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "source": {"type": "string", "description": "Restrict to one adapter (e.g. claude-code, mem0)"},
                        "limit": {"type": "integer", "minimum": 1, "default": 10},
                        "mode": {"type": "string", "enum": ["fulltext", "vector", "hybrid"], "default": "hybrid"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_record",
                "description": "Fetch one record by id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"]
                }
            },
            {
                "name": "list_sources",
                "description": "List registered sources + active model + counters.",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "import_source",
                "description": "Run an import job for one source (claude-code or mem0).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "adapter": {"type": "string"},
                        "instance": {"type": "string"},
                        "path": {"type": "string", "description": "Path override (mem0 sqlite file or claude projects root)"}
                    },
                    "required": ["adapter"]
                }
            },
            {
                "name": "trace_provenance",
                "description": "Return native_id / native_path / raw_hash for one record.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"]
                }
            }
        ]
    })
}

fn resources_list_payload() -> Value {
    json!({
        "resources": [
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
        ]
    })
}

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
        assert_eq!(names.len(), 4, "expect exactly 4 non-admin tools");
    }

    #[tokio::test]
    async fn tools_list_includes_all_five_when_admin_enabled() {
        // PR-A: with admin enabled, the full catalogue is back.
        let store = Store::open_in_memory().unwrap();
        let s = AnamnesisServer::new(store, None, std::env::temp_dir()).with_admin_tools(true);
        let resp = s.handle(req("tools/list", Value::Null)).await;
        let names = tool_names_from(&resp.result.unwrap());
        assert_eq!(names.len(), 5);
        for expected in [
            "search_memories",
            "get_record",
            "list_sources",
            "import_source",
            "trace_provenance",
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
}
