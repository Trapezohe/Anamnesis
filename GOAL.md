---
status: active
audience: Claude Code（实施方）× codex（方案设计 + 代码审查）协作自迭代
created: 2026-05-27
anchor: docs/BLUEPRINT.md · Phase 4「生态扩展（持续）」
---

# GOAL — Anamnesis 自迭代宪章

> 这份文件是 Claude Code × codex 协作自迭代的**唯一目标来源**。每开始一轮（Round）之前先读它。
>
> **分工速记（详见 §6）**：codex **只做两件事 —— 方案设计 + 代码审查**，不碰其余任何环节。Claude Code 负责**其余全部**：选题、向 codex 要方案、实施方案、测试、修复 codex 审查发现的问题、开 PR、合并。
> 它的作用不是告诉你"做什么具体功能"，而是**圈定边界**：什么是北极星、什么算跑偏、一轮怎么算完成。
> 当一项工作与这里的北极星冲突时，**停下来问人，不要自作主张扩张范围**。

---

## 0. 一句话目标

> 把散落在各家 agent 里的记忆，引渡到一个**用户主权、本地优先**的统一记忆层，
> 并通过 MCP/CLI **安全地服务回任意 agent** —— 让记忆能在工具之间干净地流动。

Anamnesis 不是又一个记忆产品，它是**已存在的记忆之间的桥**。原始记忆的所有权永远属于产生它的 agent / 框架；Anamnesis 只做聚合、规范化、检索、回流。

---

## 1. 北极星（锁定，不可漂移）

这些是 Blueprint §1/§2/§7 已拍板的决策。**任何一轮迭代都不得违反**：

| # | 约束 | 含义 |
|---|---|---|
| N1 | **本地优先（local-first）** | 数据落地在 XDG/SQLite 单文件，零运维、无强制云。 |
| N2 | **用户主权（user-sovereign）** | 默认无 telemetry；所有数据可导出、可审计、可迁移；schema 开放。 |
| N3 | **不改写源数据** | 对上游记忆**只读导入**。Anamnesis 永远不回写、不删除、不"修复"原始 agent 的记忆文件。 |
| N4 | **协议优先 / 可被任意 agent 消费** | 能力通过 MCP（stdio+SSE）与 CLI 暴露，保持中立，不绑定单一下游。 |
| N5 | **provenance 是一等公民** | 每条记录都带来源身份、scope、时间、内容 hash、lineage；任何新能力都不能弄丢溯源。 |
| N6 | **adapter 契约不可破** | 14 个 adapter 共享 `MemoryAdapter` 契约（descriptor 稳定、scan 幂等、normalize 纯函数、schema_version 正确、native_id 存在、raw_hash 非平凡）。新增/改动 adapter 必须继续过契约测试套件。 |

---

## 2. 当前主线（本阶段聚焦，**不要离开这条线**）

近 ~30 轮的演进轨迹清晰，本阶段继续沿它走：

```
round-trip 导出（mem0-sqlite / letta-sqlite / memos-dir）
   → reconcile / drift 诊断（reconcile_sources, reconcile_export_bucket）
      → 导入后回填闭环（import_source --reconcile-export）
         → capability discovery（discover_adapters.round_trip_export_format）
            → 〔下一步在这条线上〕
```

**主题** = 把 Blueprint Phase 4 那条「**反向：anamnesis 自身作为 MCP memory server，让别的 agent 也能读聚合记忆 → 闭环**」一步步做扎实：

- 让"记忆能干净地从 A agent 流到 B agent"这件事**端到端可用、可诊断、可发现、可信赖**。
- 优先级排序原则：**先闭合已开的环，再开新环**。
  1. 补齐现有 round-trip / reconcile / discovery 链路上的**缺口与边界 case**（最高优先）。
  2. 提升这条链路的**可观测性 / 可诊断性 / 自描述能力**（agent 端能程序化发现并安全调用）。
  3. 在不破坏 N1–N6 的前提下，**扩展 round-trip 覆盖面**（更多 adapter 获得可信的回流目标）。

> 在动手前问自己一句：**"这一轮是在闭合互操作的环，还是在偏离它？"** 偏离就别做。

---

## 3. 迭代单元 —— Round 范式（沿用现有节奏）

每一轮 = **一个原子能力**，对应一个 PR。这是从现有 commit 历史里提炼的纪律，必须照做。
一轮的标准时序见 §6；下面是每轮在**代码/产出层面**必须满足的约束（无论由谁产出）：

1. **单一关注点**：一轮只动一个能力面。不要把无关重构、改名、依赖升级混进同一个 PR。
2. **CLI ↔ MCP 对等**：若能力同时存在于两侧，必须**两侧一起加**并各自有测试（历史上每个 `feat(cli,mcp)` 都这样）。
3. **测试随能力增长**：每轮新增针对性测试，workspace 测试总数只增不减；新增的契约/不变量要有断言守护（参考 R149 的 parity test）。
4. **provenance / 审计不回退**：新能力若触及记录，必须保留 lineage 与 audit event。
5. **文档同步**：面向用户的能力更新 `CHANGELOG.md [Unreleased]`；必要时更新 README / INTEGRATIONS.md。
6. **提交规范**：conventional commit，标题形如 `feat(<scope>): <能力> (Round N)`，结尾带
   `Co-Authored-By: ...`。从 `main` 切新分支，**不在 main 上直接提交**。
7. **开 PR → CI 全绿 → Claude Code 合并**：PR body 含 `## Summary` + `## Test plan`（勾选 fmt/clippy/test）
   及 codex 方案/审查结论摘要。开 PR、跑测试、确认 CI 全绿、squash 合并、删分支 —— **全部由 Claude Code 执行**。
   codex 不参与开 PR / 测试 / 合并。
8. **代码克制**：实现代码不写啰嗦的注释和文档块。只在意图不显然处留一行短注释；契约/不变量靠**测试断言**表达而非长段注释。命名自解释优先于注解。面向用户的说明放 CHANGELOG/README，不灌进源码。

### 每轮的 Definition of Done（缺一不可）

- [ ] `~/.cargo/bin/cargo fmt --all`
- [ ] `~/.cargo/bin/cargo clippy --workspace --all-targets --no-default-features -- -D warnings`
- [ ] `~/.cargo/bin/cargo test --workspace --lib --bins --tests --no-default-features -- --test-threads=2`（全绿，计数 ≥ 上一轮）
- [ ] 若涉及 SSE / default-features 行为，对应 feature 组合也跑过
- [ ] CLI↔MCP 对等已确认（或在 PR 里说明为何只需单侧）
- [ ] CHANGELOG `[Unreleased]` 已更新
- [ ] 已交 codex 审查并修复其指出的问题（直到 codex pass）
- [ ] PR 已开、CI 全绿、由 Claude Code squash 合并并删分支

> 注：本机 cargo 在 `~/.cargo/bin/cargo`，不在默认 PATH，命令一律用全路径。

---

## 4. 硬护栏 —— 非目标（做了就是跑偏）

以下来自 Blueprint §2「非目标」与 §7「安全」。**未经人工明确同意，不得触碰**：

- ❌ **云同步 / 多设备同步**（属 Phase 5 远期，需人工决策才启动）
- ❌ **回写、编辑、合并、覆盖原始 agent 的记忆数据**（违反 N3）
- ❌ **GUI**（交给 ghast 等下游）
- ❌ **实时双向同步**
- ❌ **默认开启自动 re-embedding**（默认保留原向量）
- ❌ **引入 telemetry / 任何默认外发网络调用**（违反 N2）
- ❌ **为单一下游 agent 做特例化**而破坏协议中立性（违反 N4）
- ❌ **新增重量级运行时依赖**而不在 PR 里说明理由与体积影响
- ❌ **破坏性 API 变更**且不标注、不给迁移路径（schema 用 `schema_version` 多版本并存）

---

## 5. 停下来问人的信号（不要硬闯）

遇到下列任一情况，**暂停当前轮，写清问题交给人**，而不是替用户拍板：

- 要做的事落在第 4 节任一非目标上，但你觉得"这次应该例外"。
- 需要改动共享 schema、`MemoryAdapter` 契约，或任何会让既有测试**必须修改**才能过的接口。
- 一轮无法保持原子（开始膨胀成多个能力），说明范围没切好 —— 退回来重新切分。
- 与本机/外部产生不可逆副作用（删文件、写用户真实记忆目录、发网络请求）。
- 连续两轮在同一处卡壳或测试反复红 —— 交给人，附最小复现。

---

## 6. 协作分工（codex × Claude Code）

**职责边界（严格，不越界）：**

| 角色 | 做（仅此） | 明确不做 |
|---|---|---|
| **codex** | ① **方案设计**：针对一轮的选题，产出技术方案/接口设计/取舍分析/测试要点。② **代码审查**：对 Claude Code 实现的 diff 做独立 review，给出 pass/fail 与具体问题。 | ❌ 不写/改实现代码 ❌ 不修自己审出的问题 ❌ 不跑测试 ❌ 不开 PR ❌ 不合并 |
| **Claude Code（我）** | **除 codex 那两件事之外的全部**：① 锚定本宪章**选题**、把需求讲清交 codex 要方案。② 评估方案、必要时回问。③ **实施方案**（写码 + 加测试）。④ **跑测试**、过 §3 DoD。⑤ 交 codex 审查，并**亲自修复 codex 审出的所有问题**，直到 codex pass。⑥ **开 PR、确认 CI 全绿、squash 合并、删分支**。 | ❌ 不自行拍板北极星/护栏的例外（见 §5） |
| **人** | 监督；裁决 §5 上报的需人工拍板的开放问题；可随时叫停。 | — |

**一轮的标准时序（每一步的执行者已标注）：**

```
[Claude Code] 选题（依 §2 主线 + 下方优先级）
  → [Claude Code] 交 codex：要「方案设计」（接口/取舍/测试要点）
    → [codex] 产出方案
      → [Claude Code] 评估方案；有疑问回问 codex，直到方案对齐北极星与护栏
        → [Claude Code] 实施方案（写码 + 测试 + 跑 §3 DoD）
          → [Claude Code] 交 codex：要「代码审查」
            → [codex] review diff，给出问题清单
              → [Claude Code] 亲自修复每个问题、重跑测试，直到 codex pass
                → [Claude Code] 开 PR（含 codex 方案与 review 结论摘要）
                  → [Claude Code] CI 全绿后 squash 合并 + 删分支 → 进入下一轮
```

- **codex 只输出文字（方案 / 审查意见），从不产出或修改代码、不执行任何动作。** 一切代码动作（写、改、修复、测试、合并）都由 Claude Code 落地。
- **选题优先级**：① §2 链路上的已知缺口 → ② Blueprint Phase 4 清单 → ③ CHANGELOG/TODO 里标注的债。
- **方案先行**：没有 codex 的方案、或方案未对齐北极星/护栏之前，**不动实现代码**。
- **审查闭环**：代码必须经 codex 至少一轮 review、且其指出的问题由 Claude Code 修复到 pass，才进 PR；review 关键结论写进 PR body。
- **每轮独立**：一个 PR 一个能力，互不阻塞；不要堆叠未合并的依赖链。
- **遇到 §5 信号**：由 Claude Code 暂停并上报人工，不让 codex 或自己替用户拍板。

---

*本宪章随主线推进可更新，但 §1 北极星与 §4 护栏的任何改动都必须经人工确认。*
