<p align="center">
  <img src="./banner.png" alt="Anamnesis banner" width="920">
</p>

<p align="center">
  <img src="./logo.png" alt="Anamnesis logo" width="96">
</p>

<h1 align="center">Anamnesis · 搜魂术</h1>

<p align="center">
  <strong>把散落在各家 Agent 里的记忆，引渡到一个统一、可审计、本地优先的记忆层。</strong>
</p>

<p align="center">
  <a href="https://github.com/Trapezohe/Anamnesis"><img src="https://img.shields.io/badge/version-v0.0.1-0ea5e9?style=for-the-badge" alt="version"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-22c55e?style=for-the-badge" alt="license"></a>
  <img src="https://img.shields.io/badge/rust-%3E%3D1.85-f97316?style=for-the-badge&logo=rust&logoColor=white" alt="rust">
  <img src="https://img.shields.io/badge/MCP-stdio%20%2B%20SSE-8b5cf6?style=for-the-badge" alt="MCP">
  <img src="https://img.shields.io/badge/RAG-local%20hybrid-14b8a6?style=for-the-badge" alt="local rag">
  <a href="https://x.com/Ghast_AI"><img src="https://img.shields.io/badge/X-@Ghast__AI-000000?style=for-the-badge&logo=x&logoColor=white" alt="X"></a>
  <a href="https://discord.gg/ghastai"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <a href="./README.md">English</a>
  · <a href="#概览">概览</a>
  · <a href="#当前支持的记忆框架与-agent">当前支持</a>
  · <a href="#系统架构">系统架构</a>
  · <a href="#快速开始">快速开始</a>
  · <a href="./docs/BLUEPRINT.md">蓝图</a>
  · <a href="https://discord.gg/ghastai">Discord</a>
</p>

---

## 概览

**Anamnesis（搜魂术）** 是一个开源、本地优先的跨 Agent 记忆层。它读取 Claude Code、mem0、Codex、Generic MCP resource 以及未来更多 Agent / Memory Framework 中的记忆数据，把它们归一化到同一套 schema、SQLite 存储和 Anamnesis 自有 RAG 索引中，再通过 CLI 与 MCP 提供给任何可信 Agent 使用。

它不是另一个聊天应用，而是 Agent 时代的本地记忆基础设施：

- **用户主权**：记忆默认留在本地，不默认上传、不默认遥测。
- **跨工具连续性**：一个 Agent 学到的用户偏好、项目规则和历史上下文，可以被其他可信 Agent 继续使用。
- **统一检索栈**：不代理 mem0 search，不混用 Claude / Codex / Hermes 的向量空间；统一由 Anamnesis chunk、embedding、FTS、vector search、rerank。
- **可审计 provenance**：每条记录都保留 `adapter / instance / native_id / native_path / raw_hash`，可以回到原始来源。

> 当前状态：`v0.0.1` pre-release。导入、存储、本地 RAG、CLI、MCP 主链路已经可运行，但 CLI/API/schema 在 `0.1.0` 前仍可能调整。

## 技术概览

| 维度 | 当前实现 |
|---|---|
| 语言 | Rust 2021，MSRV `1.85` |
| 二进制 | `anamnesis` CLI、`anamnesis-mcp` MCP server |
| 存储 | SQLite + FTS5 + chunk-level tables；当前向量为 BLOB-backed cosine fallback，sqlite-vec 是目标替换层 |
| 检索 | FTS5 BM25 + vector kNN + Reciprocal Rank Fusion + ContextPacker |
| Embedding | 默认本地 `fastembed-rs`；内置 curated model registry；Voyage cloud provider 必须显式开启 |
| 协议 | MCP stdio；`anamnesis-mcp --sse` 支持 loopback HTTP/SSE |
| 当前 Adapter | Claude Code、mem0 SQLite、Codex 基础版、Generic MCP 基础版 |
| 安全姿态 | 本地优先、source provenance、cloud provider 显式 opt-in；MCP admin tool gate 是下一轮 P0 收口项 |

## 当前支持的记忆框架与 Agent

### 可导入的记忆源

| 类型 | Source / Agent | 状态 | 当前读取内容 | 精准度 |
|---|---|---|---|---|
| Agent | Claude Code | 可用 | `~/.claude/projects/*/memory/*.md`、项目 `*.jsonl` session | memory md 中高；session 中低 |
| Memory Framework | mem0 | 可用 | self-hosted SQLite `memories` 表 | 中高 |
| Agent | Codex | 基础版 | `.codex` 下 JSON / JSONL session | 低，待真实 schema 收口 |
| Protocol | Generic MCP Server | 基础版 | `resources/list` + `resources/read` | 低，当前按 opaque resource 处理 |

### 可消费 Anamnesis 的工具

| Consumer | 接入方式 | 状态 | 说明 |
|---|---|---|---|
| ghast | MCP server config | 计划集成 | ghast 是第一消费者，但 Anamnesis 保持独立开源项目 |
| Claude Desktop / Claude Code MCP client | `anamnesis-mcp` stdio | 可接入 | 适合本地检索和 provenance 查询 |
| Codex / CLI Agent | MCP stdio 或 CLI | 可接入 | 可以通过 MCP 或 shell 命令消费 |
| Cursor / Zed / 其他 MCP-aware tools | MCP stdio / SSE | 可接入 | 取决于各客户端 MCP 能力 |
| 自定义脚本 | CLI + JSON 输出 | 可接入 | `search --json`、`export`、`status --json` |

### 计划支持

| Source / Consumer | 类型 | 计划 |
|---|---|---|
| Hermes | Agent / Memory system | 独立 adapter，复用统一 schema 与本地 RAG |
| OpenAI / Voyage / 其他云 embedding | Embedding provider | 只做显式 opt-in，永不默认外发 |
| Agent Memory Interchange Format | 标准化方向 | 后续 RFC，推动跨 Agent 记忆交换 |

## 为什么需要 Anamnesis

今天每个 Agent 都在各自保存“记忆”：

- Claude Code 有项目 JSONL session、markdown memory 和本地状态。
- mem0 有 SQLite / API 中的结构化 memory。
- Codex 有本地 session 和 rollout 历史。
- ghast、Hermes、Cursor、Zed 未来也可能各有一套上下文和记忆系统。

如果这些记忆只待在原系统里，用户每换一个工具就要重新训练一次 Agent。Anamnesis 做的是把这些分散资产统一成一个本地、开放、可检索、可迁移的记忆层。

## 系统架构

```mermaid
flowchart TB
  subgraph Consumers["消费者"]
    Ghast["ghast"]
    ClaudeDesktop["Claude Desktop"]
    CodexClient["Codex / CLI Agent"]
    Cursor["Cursor / Zed"]
    Scripts["自定义脚本"]
  end

  subgraph Runtime["Anamnesis 本地运行层"]
    CLI["anamnesis CLI"]
    MCP["anamnesis-mcp<br/>stdio / SSE"]
    Search["search crate<br/>Hybrid RAG + ContextPacker"]
    Importer["importer crate<br/>scan -> normalize -> chunk -> upsert"]
    Store["store crate<br/>SQLite + FTS5 + embeddings"]
    Embedder["embedder crate<br/>fastembed / optional cloud"]
  end

  subgraph Sources["记忆源"]
    ClaudeCode["Claude Code<br/>MD + JSONL"]
    Mem0["mem0<br/>SQLite"]
    Codex["Codex<br/>JSON / JSONL"]
    GenericMCP["Generic MCP<br/>resources/read"]
    Future["Hermes / 更多 adapter"]
  end

  Ghast --> MCP
  ClaudeDesktop --> MCP
  CodexClient --> MCP
  Cursor --> MCP
  Scripts --> CLI

  MCP --> Search
  CLI --> Search
  CLI --> Importer
  MCP -. "admin tools (计划 gate)" .-> Importer

  Importer --> Store
  Search --> Store
  Store --> Embedder
  Embedder --> Store

  ClaudeCode --> Importer
  Mem0 --> Importer
  Codex --> Importer
  GenericMCP --> Importer
  Future --> Importer
```

## 导入运行逻辑

Anamnesis 把“读取载体”和“理解记忆语义”拆开。Adapter 不直接写数据库，只负责发现、读取和规范化；所有持久化统一由 store transaction 完成。

```mermaid
flowchart LR
  A["Discovery<br/>只读路径/表结构/数量"] --> B["用户确认 / Source Registry"]
  B --> C["Adapter.scan()<br/>RawRecord stream"]
  C --> D["Parser<br/>Markdown / JSONL / SQLite / MCP resource"]
  D --> E["Normalizer<br/>AnamnesisRecord"]
  E --> F["Chunker<br/>record -> chunks"]
  F --> G["Store.upsert_transaction()<br/>records + raw_artifacts + chunks"]
  G --> H["FTS5 index<br/>chunks_fts"]
  G --> I["Embedding jobs<br/>content_hash + model_id"]
  I --> J["Embedding worker<br/>local fastembed"]
  J --> K["chunk_embeddings"]
```

### Adapter 精准度矩阵

| Source | 当前读取方式 | 统一结果 | 精准度 | 说明 |
|---|---|---|---|---|
| Claude Code memory markdown | `~/.claude/projects/*/memory/*.md` | frontmatter type -> `Kind/Scope`，body -> `content` | 中高 | 结构化 memory 已可用；frontmatter parser 仍需增强 |
| Claude Code JSONL | 项目 `*.jsonl` 文件 | `Episode / Session` | 中低 | 当前是历史对话召回，不等于稳定偏好抽取 |
| mem0 SQLite | read-only `memories` 表 | `memory` -> content，默认 `Fact / User` | 中高 | SQLite 模式可用；API 模式和 source embedding provenance 待补 |
| Codex | 基础 `.json/.jsonl` 扫描 | `Episode / Session` | 低 | 需要识别真实 Codex session schema 和路径白名单 |
| Generic MCP | `resources/list` + `resources/read` | `Unknown / Ephemeral` | 低 | 适合 opaque resource；精准语义需定义 memory MCP 约定 |

## RAG 检索运行逻辑

Anamnesis 的检索路径完全由自己控制。源系统的向量、搜索 API 或排序逻辑不会进入统一检索结果。

```mermaid
flowchart LR
  Q["Query<br/>text + source/kind/scope/time filters"] --> Filter["SearchFilter<br/>计划下推 store"]

  Filter --> FTS["FTS5 BM25<br/>record_chunks"]
  Filter --> QEmbed["embed_query()<br/>active model"]
  QEmbed --> Vec["Vector kNN<br/>chunk_embeddings"]

  FTS --> RRF["RRF merge<br/>K = 60"]
  Vec --> RRF
  RRF --> Agg["按 record_id 聚合 chunk"]
  Agg --> Pack["ContextPacker<br/>budget + diversity + provenance"]
  Pack --> Resp["MCP / CLI response<br/>record + matched snippets"]
```

检索原则：

- **source embedding 只作 provenance**：源系统自带向量只能存到 `raw_artifacts`，永远不参与跨源搜索。
- **index embedding 统一生成**：所有 chunk 使用 Anamnesis 当前 active model 重新 embedding。
- **chunk 是检索单元，record 是语义单元**：长 session 可切多个 chunk，但对外聚合回 record。
- **ContextPacker 控制最终上下文**：预算、provenance、source diversity、matched snippets 都在返回给 Agent 前处理。

## 存储模型

```mermaid
erDiagram
  SOURCES ||--o{ RECORDS : registers
  RECORDS ||--o{ RECORD_CHUNKS : splits_into
  RECORDS ||--|| RAW_ARTIFACTS : preserves
  RECORD_CHUNKS ||--o{ CHUNK_EMBEDDINGS : indexed_by
  RECORD_CHUNKS ||--o{ EMBEDDING_JOBS : queues
  SOURCES ||--o{ IMPORT_ERRORS : reports

  SOURCES {
    text adapter
    text instance
    text location
    text config_json
    integer last_import_at
  }

  RECORDS {
    text id
    text adapter
    text instance
    text content
    text scope
    text kind
    text native_id
    text native_path
    text raw_hash
  }

  RECORD_CHUNKS {
    text id
    text record_id
    integer seq
    text content
    text content_hash
    integer token_estimate
  }

  CHUNK_EMBEDDINGS {
    text chunk_id
    text model_id
    text content_hash
    integer dim
    blob embedding
  }

  RAW_ARTIFACTS {
    text record_id
    text payload_json
    blob source_embedding
    text source_embedding_model
    integer captured_at
  }
```

## MCP 运行逻辑

```mermaid
sequenceDiagram
  participant Agent as MCP Client / Agent
  participant Server as anamnesis-mcp
  participant Search as HybridSearcher
  participant Store as SQLite Store
  participant Embed as EmbeddingProvider

  Agent->>Server: tools/call search_memories
  Server->>Search: query + filters + mode
  Search->>Store: FTS5 chunk search
  Search->>Embed: embed_query (if vector/hybrid)
  Embed-->>Search: query vector
  Search->>Store: vector chunk search
  Search->>Search: RRF merge + pack
  Search-->>Server: packed records + snippets + provenance
  Server-->>Agent: MCP response
```

当前 MCP 能力：

| 类型 | 能力 |
|---|---|
| Tools | `search_memories`、`get_record`、`list_sources`、`import_source`、`trace_provenance` |
| Resources | `anamnesis://record/{id}`、`anamnesis://source/{adapter}`、`anamnesis://timeline/{date}` |
| Prompts | `summarize_my_preferences`、`find_related` |

> 安全说明：`import_source` 属于 admin 能力。pre-release 阶段只建议在可信本地 client 中使用；下一轮 P0 会默认关闭 MCP admin tools。

## 快速开始

### 从源码安装

```bash
git clone https://github.com/Trapezohe/Anamnesis
cd Anamnesis

cargo install --path crates/cli
cargo install --path crates/mcp-server
```

### 初始化与导入

```bash
anamnesis init
anamnesis discover
anamnesis source add claude-code --path ~/.claude/projects
anamnesis import claude-code
anamnesis search "用户偏好怎么写测试？"
anamnesis status
```

### 作为 MCP server 使用

```bash
anamnesis-mcp
anamnesis-mcp --sse 8787
```

MCP client 配置示例：

```json
{
  "mcpServers": {
    "anamnesis": {
      "command": "anamnesis-mcp",
      "args": []
    }
  }
}
```

## CLI 一览

```bash
anamnesis init [--model KEY]
anamnesis discover
anamnesis source add/list/remove
anamnesis import <adapter>[:instance] [--full] [--dry-run] [--no-embed] [--path PATH]
anamnesis search <query> [--source X] [--kind K] [--scope S] [--limit N] [--mode hybrid|fulltext|vector] [--json]
anamnesis export [--format jsonl|csv] [--out FILE] [--source X]
anamnesis verify [--repair]
anamnesis model list/use/install/rebuild
anamnesis serve
anamnesis migrate
```

## 当前限制

Anamnesis 已经具备统一入库和统一检索闭环，但还不能宣称“精准理解所有 Agent 的记忆语义”。

- Codex adapter 仍是基础 episode 导入。
- Generic MCP adapter 仍是 opaque resource 导入。
- `source add` 与 `import` 的 canonical registry 链路需要继续收口。
- `--full / --since` 与 `ScanOpts` 需要真正接入 adapter scan。
- MCP admin tools 需要默认关闭。
- Session 到 stable memory 的二阶段 extractor 仍在设计中。

## 路线图

| 阶段 | 状态 | 重点 |
|---|---|---|
| Phase 0 | 基本完成 | Rust workspace、Apache-2.0、CI、README/CONTRIBUTING、schema v1/v2 |
| Phase 1 | 大部分完成 | core/store/importer/search/embedder、Claude Code、mem0 SQLite、本地 hybrid RAG |
| Phase 2 | 收口中 | MCP admin gate、source registry import、filter 下推、ScanOpts、streaming scan |
| Phase 3 | 计划中 | ghast 集成、Homebrew/cargo release、真实 dogfood 质量评估 |
| Phase 4 | 计划中 | Hermes adapter、精准 Codex adapter、memory MCP convention、Agent Memory Interchange Format |

## 贡献

最有价值的贡献是新增高质量 adapter。每个 adapter 应遵守：

- discovery 只读 metadata；
- scan 流式产出 raw records；
- normalize 是确定性纯函数；
- 每条记录保留 provenance；
- 通过 shared adapter contract tests。

详见 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## 社区

- X: [@Ghast_AI](https://x.com/Ghast_AI)
- Discord: [discord.gg/ghastai](https://discord.gg/ghastai)

## License

[Apache License 2.0](./LICENSE)

用户导入的记忆数据不属于本项目 license 范围，始终属于用户自己。

## Star History

<a href="https://www.star-history.com/#Trapezohe/Anamnesis&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Trapezohe/Anamnesis&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Trapezohe/Anamnesis&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Trapezohe/Anamnesis&type=Date" />
  </picture>
</a>
