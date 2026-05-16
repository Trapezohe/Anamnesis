---
status: blueprint
stage: 蓝图（pre-implementation）
created: 2026-05-16
updated: 2026-05-16
tags:
  - ghast
  - memory
  - anamnesis
  - architecture
---

# 36 - 搜魂术（Anamnesis）

> **One-liner（英）**: A user-sovereign memory layer that imports, unifies, and serves agent memory across Claude Code, Codex, mem0, Hermes, and any MCP-aware tool.
>
> **One-liner（中）**: 把散落在各家 agent 里的记忆引渡到一个统一的、用户主权的本地记忆层；通过 MCP 暴露给任何 agent 消费。

---

## 0. 锁定决策（已拍板）

| 项 | 决策 | 备注 |
|---|---|---|
| 中文名 | **搜魂术** | |
| 英文名 | **Anamnesis** | crate / 二进制 / repo 均使用 |
| License | **Apache 2.0** | 含专利条款，企业友好 |
| 实现语言 | **Rust** | 见 §6.2 |
| 协议 | **MCP** | server 模式，stdio + SSE |
| 存储 | **本地 SQLite + FTS5 + sqlite-vec** | 单文件，零运维 |
| 一期模型 | **只读导入、本地优先、无云同步** | |
| 仓库结构 | **cargo workspace**（多 crate） | 见 §5 |
| 数据目录 | XDG 标准：`$XDG_DATA_HOME/anamnesis/` | macOS 默认 `~/Library/Application Support/anamnesis/` |

---

## 1. 项目定位

搜魂术是**独立开源项目**，不是 ghast 的内置模块。ghast 是它的第一个消费者，但不是唯一。

### 为什么独立
1. **生态可信度** — 跨 agent 记忆桥必须看起来中立
2. **协议优先** — 别的 agent 愿意暴露记忆给搜魂术，前提是社区信任
3. **复用面** — CLI 工具的用户基数远大于单个 desktop app
4. **合规** — 开源更容易处理「读用户在第三方系统中的数据」

### 三句话价值主张
1. **拿回主权**：你的 agent 记忆是你的，不该被锁在某家产品里
2. **跨工具检索**：在 Claude Code 里记下的偏好，在 Codex / ghast / Cursor 里都能用
3. **可携带、可审计**：所有数据本地，schema 开放，导出/迁移随时可做

---

## 2. 愿景与范围

### MVP（一期）必须达成
- [ ] CLI: `anamnesis init / import / search / export / status / serve`
- [ ] MCP server：暴露 5 个核心 tool + 3 类 resource
- [ ] 2 个 adapter：Claude Code（本地 JSONL+MD）、mem0（self-hosted SQLite）
- [ ] 本地存储：SQLite + FTS5 全文检索 + sqlite-vec 向量列
- [ ] 增量同步（基于文件 mtime + native_id 去重）
- [ ] ghast 通过 MCP server config 注册即用，零代码

### 非目标（一期不做）
- ❌ 云端同步 / 多设备同步
- ❌ 记忆编辑、合并、覆盖原始数据
- ❌ GUI（GUI 由 ghast 等下游负责）
- ❌ 实时双向同步
- ❌ 自动 re-embedding（默认保留原向量）

---

## 3. 系统架构

### 3.1 总体分层

```
┌──────────────────────────────────────────────────────────────┐
│                    Consumers（消费者层）                       │
│   ghast │ Claude Code │ Cursor │ Zed │ CLI 用户 │ 自定义脚本    │
└──────────┬──────────────────────────┬─────────────────────────┘
           │ MCP (stdio/SSE)          │ CLI
           ▼                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  anamnesis 二进制（Rust）                      │
│  ┌────────────────────┐    ┌───────────────────────┐         │
│  │  mcp-server crate  │    │     cli crate         │         │
│  └─────────┬──────────┘    └──────────┬────────────┘         │
│            └──────────────┬───────────┘                       │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────┐        │
│  │             core crate（领域逻辑）                │        │
│  │  Record / Source / Query / IndexBuilder / ...    │        │
│  └──────┬───────────────────────────────────┬───────┘        │
│         │                                   │                 │
│         ▼                                   ▼                 │
│  ┌────────────────┐               ┌──────────────────────┐   │
│  │  store crate   │               │  adapters/* crates   │   │
│  │ SQLite+FTS+vec │               │ claude-code / mem0   │   │
│  └────────────────┘               │ codex / hermes / ... │   │
│                                   └──────────┬───────────┘    │
└──────────────────────────────────────────────┼───────────────┘
                                               │
                          ┌────────────────────┼────────────────┐
                          ▼                    ▼                ▼
                  ~/.claude/projects   mem0 SQLite/API    其他源
```

### 3.2 数据流

**导入流（initial / incremental）**：
```
Adapter.scan()
  → Stream<RawRecord>
    → Normalizer（映射到 AnamnesisRecord）
      → Dedup（hash(source.adapter + native_id)）
        → Store.upsert()
          → FTS5 索引 + 向量索引（如有）
```

**查询流（MCP `search_memories`）**：
```
Query{text?, source?, kind?, time_range?, limit}
  → Store.search()
    ├─ FTS5 BM25（默认）
    ├─ Vector kNN（如启用 + 已嵌入）
    └─ Hybrid rerank（reciprocal rank fusion）
      → Vec<RecordWithScore>
        → MCP response
```

**MCP server 模式**：
```
Consumer ─ stdio/SSE ─→ rmcp dispatcher ─→ tool handler ─→ core ─→ store
                                       └─→ resource handler ─→ store
```

---

## 4. 核心 Schema（v0.1，锁定）

```rust
// crate: anamnesis-core::model

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnamnesisRecord {
    /// blake3(source.adapter + ":" + source.instance? + ":" + provenance.native_id)
    pub id: RecordId,
    pub source: SourceDescriptor,
    pub content: String,
    pub embedding: Option<Embedding>,
    pub scope: Scope,
    pub kind: Kind,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
    pub tags: Vec<String>,
    pub metadata: serde_json::Map<String, Value>,
    pub provenance: Provenance,
    /// schema 版本，向后兼容窗口至少 2 个 minor
    pub schema_version: u32,  // = 1
}

#[derive(...)] pub struct SourceDescriptor {
    pub adapter: String,        // "claude-code" / "mem0" / "codex" / "hermes" / ...
    pub instance: Option<String>,  // 多 vault 区分
    pub version: String,        // adapter 自报版本，便于排错
}

#[derive(...)] pub struct Embedding {
    pub vector: Vec<f32>,
    pub model: String,          // e.g. "voyage-3" / "text-embedding-3-small"
    pub dim: u16,
}

#[derive(...)] pub enum Scope { User, Project, Session, Ephemeral }

#[derive(...)] pub enum Kind {
    Fact,         // 客观事实（"用户偏好 zsh"）
    Preference,   // 用户偏好（"少用 emoji"）
    Feedback,     // 对 agent 的修正
    Reference,    // 外部资源指针（"bug 跟踪在 Linear INGEST"）
    Episode,      // 一段对话/事件
    Skill,        // 学到的工作方式
    Unknown,
}

#[derive(...)] pub struct Provenance {
    pub native_id: String,      // 原系统 ID
    pub native_path: Option<String>,  // 文件路径 / DB row
    pub captured_at: DateTime<Utc>,
    pub raw_hash: String,       // blake3(原始内容)，用于增量去重
}
```

### Schema 演进策略

- `schema_version: u32` 字段必填
- minor 升级保证读旧版能 work；major 升级提供 migration tool
- DB 端单独存 `meta.schema_version`，启动时对比，自动升级

---

## 5. Cargo Workspace 结构

```
anamnesis/
├── Cargo.toml                    # workspace 根
├── LICENSE-APACHE                # Apache 2.0
├── README.md
├── crates/
│   ├── core/                     # 领域逻辑、trait、model（无 IO）
│   │   ├── src/model.rs
│   │   ├── src/adapter.rs        # Adapter trait
│   │   ├── src/query.rs
│   │   └── src/error.rs
│   ├── store/                    # SQLite + FTS5 + sqlite-vec
│   │   ├── src/lib.rs
│   │   ├── src/schema.sql
│   │   └── src/migrations/
│   ├── cli/                      # anamnesis CLI（clap）
│   │   └── src/main.rs
│   ├── mcp-server/               # rmcp 实现
│   │   └── src/main.rs
│   ├── adapter-claude-code/      # 第一个 adapter
│   ├── adapter-mem0/             # 第二个 adapter
│   ├── adapter-codex/            # Phase 4
│   └── adapter-hermes/           # Phase 4
└── xtask/                        # 构建辅助任务（release / migration）
```

**关键约定**：
- `core` 零 IO，纯类型 + trait → 易测试、不锁定后端
- `store` 是唯一可写持久层
- `mcp-server` 和 `cli` 都只依赖 `core` + `store`，互不感知
- 每个 `adapter-*` 独立 crate，可由社区贡献而不污染核心

---

## 6. 实现细节

### 6.1 二进制形态

| 模式 | 调用方式 | 用途 |
|---|---|---|
| **CLI** | `anamnesis <cmd>` | 脚本、CI、初始导入、人工查询 |
| **MCP stdio server** | `anamnesis serve --stdio` | ghast 等 desktop client 子进程方式启动 |
| **MCP SSE server** | `anamnesis serve --sse --port 7878` | 远程/共享场景、调试 |
| **library / FFI** | Phase 5+ | C ABI、Node/Swift binding |

### 6.2 语言 — Rust

| 维度 | 选择理由 |
|---|---|
| 长期基础设施 | 参照 ripgrep / fd / bat / tauri / bun，全是 Rust |
| 性能 | tantivy（FTS）、sqlite-vec、HNSW 全在 Rust 舒适区 |
| MCP SDK | 官方 `rmcp` 成熟 |
| 二进制 | ~5-10MB，daemon 启动 <50ms |
| 安全审计 | 内存安全 + 类型驱动 → 用户更愿意信任 |

调 Python-only SDK（如 mem0 自托管初版）时走 subprocess 或后续 PyO3。

### 6.3 协议 — MCP

#### MCP Tools（5 个核心）

| Tool 名 | 用途 | 输入 schema 关键字段 | 输出 |
|---|---|---|---|
| `search_memories` | 跨源检索 | `query: string, source?: string, kind?: Kind, scope?: Scope, time_range?, limit: int, mode?: 'fulltext'\|'vector'\|'hybrid'` | `Vec<RecordWithScore>` |
| `get_record` | 按 id 取详情 | `id: string` | `AnamnesisRecord` |
| `list_sources` | 已配置源 + 健康状态 | — | `Vec<SourceStatus>` |
| `import_source` | 触发导入（同步/异步） | `adapter: string, full?: bool` | `ImportJobStatus` |
| `trace_provenance` | 还原原始上下文 | `id: string` | `{native_path, native_id, source_quote, surrounding_records}` |

#### MCP Resources（3 类 URI 模式）

| URI 模式 | 含义 |
|---|---|
| `anamnesis://record/{id}` | 单条记忆（JSON） |
| `anamnesis://source/{adapter}` | 某源摘要 + 最近记录 |
| `anamnesis://timeline/{date}` | 某天/某区间的记忆时间线 |

#### MCP Prompts（可选，便利项）

- `summarize_my_preferences` — 总结用户在指定 scope 下的偏好
- `find_related` — 给定一段文本，找最相关历史记忆

### 6.4 CLI 规范

```
anamnesis init [--data-dir PATH]
anamnesis source add <adapter> [--instance NAME] [--path PATH] [--api-key ...]
anamnesis source list
anamnesis source remove <adapter>[:instance]

anamnesis import <adapter>[:instance] [--full] [--since 7d] [--dry-run]
anamnesis search "query" [--source X] [--kind preference] [--limit 20] [--mode hybrid]
anamnesis export [--format jsonl|ndjson|csv] [--filter ...] [--out PATH]
anamnesis status                  # 数据库统计、最近导入、健康
anamnesis serve [--stdio | --sse --port N]
anamnesis verify [--repair]       # schema 校验 + 索引重建
anamnesis migrate                 # 跨 schema_version 升级

# 全局 flag
--config PATH --data-dir PATH --log-level info --json
```

### 6.5 配置文件

路径：`$XDG_CONFIG_HOME/anamnesis/config.toml`（macOS: `~/Library/Application Support/anamnesis/config.toml`）

```toml
[core]
data_dir = "~/.local/share/anamnesis"
log_level = "info"

[index]
embedding_enabled = false   # 默认关闭，避免 surprise cost
embedding_model = "voyage-3"
embedding_provider = "voyage"
hybrid_alpha = 0.5          # FTS 与向量分数加权

[server.mcp]
allowed_clients = ["*"]     # 后续可白名单
require_token = false       # 本地默认信任，远程 SSE 时强制 true

[[sources]]
adapter = "claude-code"
instance = "default"
path = "~/.claude/projects"
include_globs = ["**/*.jsonl", "**/memory/*.md"]
exclude_globs = []
watch = true

[[sources]]
adapter = "mem0"
instance = "self-hosted"
mode = "sqlite"
path = "~/.mem0/db.sqlite"

[[sources]]
adapter = "mem0"
instance = "cloud"
mode = "api"
api_key_env = "MEM0_API_KEY"
```

### 6.6 存储层（store crate）

```sql
-- crates/store/src/schema.sql
CREATE TABLE records (
  id              TEXT PRIMARY KEY,
  adapter         TEXT NOT NULL,
  instance        TEXT,
  content         TEXT NOT NULL,
  scope           TEXT NOT NULL,
  kind            TEXT NOT NULL,
  created_at      INTEGER NOT NULL,    -- unix epoch
  updated_at      INTEGER,
  tags            TEXT,                -- JSON array
  metadata        TEXT,                -- JSON
  native_id       TEXT NOT NULL,
  native_path     TEXT,
  captured_at     INTEGER NOT NULL,
  raw_hash        TEXT NOT NULL,
  schema_version  INTEGER NOT NULL DEFAULT 1,
  UNIQUE(adapter, instance, native_id)
);

CREATE VIRTUAL TABLE records_fts USING fts5(
  content,
  tags,
  content_rowid='id',
  tokenize='unicode61'
);

-- 用 sqlite-vec 扩展
CREATE VIRTUAL TABLE records_vec USING vec0(
  id TEXT PRIMARY KEY,
  embedding float[1024]    -- 维度由 config 决定
);

CREATE TABLE meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- meta('schema_version', '1')

CREATE TABLE import_jobs (
  id              TEXT PRIMARY KEY,
  adapter         TEXT NOT NULL,
  instance        TEXT,
  started_at      INTEGER NOT NULL,
  finished_at     INTEGER,
  status          TEXT NOT NULL,        -- 'running' / 'done' / 'failed'
  records_seen    INTEGER DEFAULT 0,
  records_added   INTEGER DEFAULT 0,
  records_updated INTEGER DEFAULT 0,
  error           TEXT
);

CREATE INDEX idx_records_adapter ON records(adapter, instance);
CREATE INDEX idx_records_created ON records(created_at DESC);
CREATE INDEX idx_records_kind ON records(kind);
```

混合检索（hybrid）伪代码：
```rust
let fts_hits = fts5_search(query, limit*2);
let vec_hits = if embedding_enabled {
    vec_search(embed(query)?, limit*2)
} else { vec![] };
rrf_merge(fts_hits, vec_hits, limit)  // reciprocal rank fusion
```

### 6.7 Adapter trait

```rust
#[async_trait]
pub trait MemoryAdapter: Send + Sync {
    fn descriptor(&self) -> SourceDescriptor;

    /// 全量扫描，流式产出原始记录
    fn scan(&self, opts: ScanOpts) -> BoxStream<'_, Result<RawRecord>>;

    /// 可选：实时监听增量（fs notify / 轮询 / SSE）
    fn watch(&self, opts: WatchOpts) -> Option<BoxStream<'_, Result<RawDelta>>>;

    /// 把原始记录规范化为 AnamnesisRecord
    fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>>;

    /// 凭据 / 路径 / 权限检查
    async fn health(&self) -> HealthStatus;
}

pub struct RawRecord {
    pub native_id: String,
    pub native_path: Option<String>,
    pub payload: serde_json::Value,
    pub captured_at: DateTime<Utc>,
}
```

### 6.8 Claude Code Adapter（第一个 adapter，详细）

数据源：
- `~/.claude/projects/<hash>/*.jsonl` — 对话历史
- `~/.claude/projects/<hash>/memory/MEMORY.md` — 索引
- `~/.claude/projects/<hash>/memory/*.md` — 单条记忆（frontmatter + body）

抽取规则：
1. **memory/*.md → Kind 直接由 frontmatter `type` 字段映射**
   - `user` → `Kind::Fact`（同时 `scope=User`）
   - `feedback` → `Kind::Feedback`
   - `project` → `Kind::Fact`（`scope=Project`）
   - `reference` → `Kind::Reference`
2. **conversation.jsonl → 默认 `Kind::Episode`、`scope=Session`**
   - 一个 session 一个 record（content = 摘要 + 关键消息），避免炸 records 表
   - `native_id = session_id`，`raw_hash = blake3(整个 jsonl)`
3. **MEMORY.md 不进 records 表**，作为 source-of-truth 的索引，仅用于交叉验证

增量：基于 `mtime + raw_hash`。
- `mtime` 没变 → 跳过
- `mtime` 变了但 `raw_hash` 同 → 跳过
- `mtime` 和 `raw_hash` 都变 → upsert（按 unique(adapter, instance, native_id)）

### 6.9 mem0 Adapter（第二个 adapter）

两种模式：
- **`mode = "sqlite"`**：直读 self-hosted SQLite，`SELECT id, memory, user_id, metadata, created_at FROM memories`
- **`mode = "api"`**：调 mem0 REST API，分页拉

字段映射：
- `memory` → `content`
- `user_id` → `metadata.mem0_user_id`，`scope=User`
- `created_at` → `created_at`
- `metadata.categories` → `tags`
- `id` → `provenance.native_id`
- 默认 `kind=Fact`（mem0 没有细分类型）

### 6.10 新 Adapter 模板（贡献者指南）

```
crates/adapter-<name>/
  Cargo.toml         # 仅依赖 anamnesis-core
  src/lib.rs         # impl MemoryAdapter
  src/normalize.rs   # raw → AnamnesisRecord
  tests/contract.rs  # 跑统一的 adapter contract test 套件
  fixtures/          # 一份匿名化的 sample data
```

`anamnesis-core` 导出 `adapter_contract_test!(YourAdapter)` 宏，强制每个 adapter 通过同一组黑盒测试（idempotent / dedup / schema_version / watch lifecycle）。

---

## 7. 安全与权限模型

### 威胁面
1. **本地 daemon 端口被其他进程窃听**（SSE 模式）
2. **MCP stdio 子进程的环境变量泄漏 API key**
3. **导入时把敏感对话明文落地**（mem0 cloud 也存了一份在云端，但本地 SQLite 是用户应控制的）
4. **多用户机器**：A 用户的 daemon 可能被 B 读

### 防御
- **默认 stdio 模式**：父进程 fd 隔离，最安全
- **SSE 模式强制本地 token**：`anamnesis serve --sse` 启动时生成 64-byte token，client 必须在 header 带 `Authorization: Bearer <token>`
- **数据目录权限**：`chmod 700`，单用户独占
- **API key 来源**：仅从环境变量读，禁止配置文件明文存
- **PII 脱敏 hook**：可配置 regex 替换（默认关闭，用户开启）
- **审计日志**：所有 import / search / export 写 `~/.local/share/anamnesis/audit.log`

### Apache 2.0 + 数据主权声明
README + LICENSE 里明确写明：
- 软件本身 Apache 2.0
- 用户的记忆数据**不属于本项目**，搜魂术只是搬运工
- 不收集 telemetry（一期）

---

## 8. 测试策略

| 层 | 范围 | 工具 |
|---|---|---|
| 单元测试 | core 内 model / query / hash | `cargo test` |
| Adapter contract test | 每个 adapter 必跑 8 个标准用例 | 共享宏 + fixtures |
| 集成测试 | store + 单 adapter 端到端 | `tempfile` + sample fixtures |
| MCP 协议测试 | server 行为对照 spec | rmcp 自带 + 录制响应 |
| Fuzzing | normalize 路径（防恶意 JSONL） | `cargo-fuzz` |
| 性能基准 | 10k / 100k / 1M records 检索 P50/P99 | `criterion` |

**关键不变式**（合并所有测试覆盖）：
1. 同一 source 同 native_id 的两次导入 → 不产生重复 row
2. `raw_hash` 不变 → 不重写 record
3. schema_version 升级后，旧 record 仍可读
4. MCP search 返回的 record id 一定能 get_record 取到

---

## 9. 可观测性

- **日志**：tracing + tracing-subscriber，env `ANAMNESIS_LOG=debug`
- **结构化错误**：`thiserror` 自定义 + 错误码
- **指标**：`anamnesis status` 输出 JSON（records 数 / 各 source 数 / 最近 import / 健康）
- **telemetry**：**一期完全不发**（信任建立期）

---

## 10. 关键 Tradeoff（决策记录）

| # | 问题 | 选择 | 反方 | 复审条件 |
|---|---|---|---|---|
| 1 | Embedding 是否统一重算 | 默认保留原向量，跨源搜索时按需 re-embed | 一律重 embed | 用户反馈跨源搜索不准 |
| 2 | 只读 vs 双向同步 | 只读 | 双向 | 至少 6 个月成熟后 |
| 3 | 本地 vs 云端 | 本地优先 | SaaS | 多设备同步需求达临界 |
| 4 | Rust vs Go | Rust | Go | 上线速度成瓶颈时 |
| 5 | 协议 | MCP | 自定义 JSON-RPC | MCP 演进破坏兼容 |
| 6 | 向量库 | sqlite-vec | qdrant / lancedb | 单机超 1M 条变慢 |
| 7 | 全文检索 | SQLite FTS5 | tantivy | FTS5 性能瓶颈 |
| 8 | License | Apache 2.0 | MIT / AGPL | — |

---

## 11. 路线图

### Phase 0：项目奠基（1 周）
- [x] 决定语言：Rust
- [x] 决定协议：MCP
- [x] 决定 License：Apache 2.0
- [x] 决定名称：anamnesis（搜魂术）
- [ ] 创建 GitHub repo `anamnesis` + LICENSE + README + CONTRIBUTING + CODE_OF_CONDUCT
- [ ] 占坑：crates.io、Homebrew tap、组织/账户
- [ ] cargo workspace 骨架（core / store / cli / mcp-server）
- [ ] 锁定 `AnamnesisRecord` schema v1
- [ ] CI：fmt / clippy / test / cross-compile（macOS aarch64+x64, Linux x64+aarch64, Windows x64）
- [ ] Release Drafter + cargo-release 配置

### Phase 1：核心引擎（2 周）
- [ ] `store`：SQLite + FTS5 + sqlite-vec + migrations
- [ ] `core`：Adapter trait + Normalizer + Query
- [ ] `cli`：`init / status / search / serve` 子命令
- [ ] Adapter contract test 框架
- [ ] **claude-code adapter**（含 watcher）
- [ ] E2E 测试：导入真实 `~/.claude/projects` → search 命中

### Phase 2：第二个 Adapter + MCP（1-2 周）
- [ ] **mem0 adapter**（sqlite + api 双模式）
- [ ] `mcp-server`：stdio + SSE，5 tools + 3 resources
- [ ] 跨 adapter 检索 demo
- [ ] 验证 schema 抽象是否扛得住 → 若不行回 Phase 1 修订

### Phase 3：ghast 集成 + 0.1 发布（1 周）
- [ ] ghast 添加 anamnesis 作为默认 MCP server config（用户启用即用）
- [ ] ghast 内 UX：源管理、时间线、溯源 UI
- [ ] Homebrew formula + Linux tarball + Windows zip
- [ ] `cargo install anamnesis`
- [ ] 0.1.0 release blog + demo 视频

### Phase 4：生态扩展（持续）
- [ ] codex adapter
- [ ] hermes adapter（参见 [[Hermes 借鉴清单 - 12 项功能差距与优先级]]）
- [ ] generic MCP adapter（订阅任意 memory-providing MCP server）
- [ ] **反向**：anamnesis 自身作为 MCP memory server，让别的 agent 也能读聚合记忆 → 闭环
- [ ] 选择性导入 UI / regex 过滤 / PII 脱敏 hook 默认值
- [ ] embedding 自动化（可选）：voyage / openai / local（nomic / bge）
- [ ] HN / Reddit / Twitter 发布

### Phase 5：远期
- [ ] FFI（C ABI + Node/Swift binding）
- [ ] 跨设备同步（end-to-end encrypted）
- [ ] 推动「Agent Memory Interchange Format」社区标准
- [ ] 商业化：paid hosted（可选，按使用量），开源核心永久免费

---

## 12. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Claude Code 修改 JSONL 格式 | 高 | 中 | 把 adapter 版本化 + 标 `schema_version`，多版本并存 |
| sqlite-vec 不稳定 | 中 | 中 | 向量列可降级到全表 cosine（性能差但可用） |
| MCP spec 演进破坏兼容 | 中 | 高 | pin rmcp 版本，跟踪 spec 更新做适配 |
| 用户数据隐私事故 | 低 | 极高 | 默认无 telemetry / 文档强调本地 / 审计日志 / Apache 2.0 |
| 维护人手不足 | 高 | 高 | adapter 独立 crate 降低社区贡献门槛 |

---

## 13. 开放问题（待后续讨论）

- [ ] 隐私边界：是否提供 GUI 帮用户选择性导入？还是仅 CLI flag？
- [ ] 冲突解决：同一事实在两个 source 里描述不同——以哪个为准，还是都保留？
- [ ] 时间衰减：是否给每条记忆一个 relevance decay（影响检索排序）？
- [ ] 多用户机器：是否支持系统级安装 + 多用户独立数据库？
- [ ] 商业模式：paid hosted 是否独立子项目还是同 repo？
- [ ] 协议标准化：先 own 自己的 schema，还是早期就发 RFC 拉社区？
- [ ] Telemetry：1.0 之后是否引入匿名 opt-in 使用统计？

---

## 14. 立即可执行的下一步

1. **占名**：GitHub org / repo（`anamnesis`）、crates.io、Homebrew tap、域名（可选 `anamnesis.dev`）
2. **写 README v0**：用「Three sentences value prop + Quick start gif」结构
3. **cargo workspace init**：搭骨架 + CI + LICENSE-APACHE
4. **第一个 PR**：core crate 的 `AnamnesisRecord` + 单元测试
5. **第二个 PR**：store crate 骨架 + migration 0001

---

## 15. 关联文档

- [[35-Coding模式实施计划]]
- [[Hermes 借鉴清单 - 12 项功能差距与优先级]]
- [[Partner Mode - 主动反思与自我评估闭环]]
- [[32-数据模型速查]]

---

*Status: 蓝图完整，等待 Phase 0 执行启动。命名 / License / 语言 / 协议 / Schema / 架构 / 路线图 / 安全 / 测试均已落定。*
