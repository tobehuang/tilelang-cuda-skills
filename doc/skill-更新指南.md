# Skill 更新操作指南

> 面向场景：你想新增、修改、删除 `skills/` 下的 TileLang / CUDA skill 包，或是只改某一份 reference。本文给出**目录约定 → 改前准备 → 修改流程 → 验证 → 提交**全套动作。

---

## 0. 仓库快速画像

```
tilelang-cuda-skills/
├── README.md                   ← 顶层索引；新增 skill 必同步这里
├── doc/                        ← 中文解读 / 学习笔记（本文档所在地）
├── evals/                      ← 评测脚本（如有）
└── skills/
    ├── tilelang/
    │   ├── writing-tilelang-kernels/
    │   │   ├── SKILL.md                ← 主入口（必有）
    │   │   └── references/
    │   │       ├── api-quick-reference.md
    │   │       ├── kernel-templates.md
    │   │       ├── language-docs.md
    │   │       └── tilelang-examples.md
    │   ├── debugging-tilelang-programs/
    │   ├── profiling-tilelang-programs/
    │   ├── optimizing-tilelang-programs/
    │   └── testing-fwd-bwd-kernels/
    └── cuda_skill/
        └── SKILL.md
```

git 远端已设两个：

| Remote | URL | 用途 |
|---|---|---|
| `origin` | `tobehuang/tilelang-cuda-skills` | 你的 fork，PR 推这里 |
| `upstream` | `sablin39/tilelang-cuda-skills` | 上游，拉同步用 |

**对应版本基线**：TileLang v0.1.9，CUDA 13.1，PyTorch 2.11，验证 GPU = RTX PRO 6000 Blackwell (sm_120)。

---

## 1. Skill 包约定（写之前必读）

每个 skill 包的标准形态：

```
skills/<category>/<skill-name>/
├── SKILL.md                    # 必有，含 YAML frontmatter
└── references/                 # 可选，长文档拆这里
    ├── *.md
    └── ...
```

### 1.1 `SKILL.md` 头部 frontmatter

```markdown
---
name: writing-tilelang-kernels
description: >
  How to write TileLang GPU kernels from scratch ...
  Use this skill whenever the user wants to ...
---
```

**关键字段**：

| 字段 | 作用 | 注意 |
|---|---|---|
| `name` | 必须**等于目录名**（agent 靠这个加载） | 改名要同步改目录 |
| `description` | 触发条件描述 | 写满**触发词**：用户可能说什么、关键字、API 名 |

> agent 是靠 `description` 里的关键词来决定要不要加载这个 skill 的，所以**关键词要写全**：用户口语 + API 名 + 算子名 + 同义词。

### 1.2 SKILL.md 主体推荐结构

照 `writing-tilelang-kernels` 这套模式：

1. **Pre-flight Checklist**（动作清单）
2. **Anatomy / 心智模型**（必懂概念）
3. **Step-by-Step Workflow**（操作流程）
4. **Quick Templates**（最小可跑模板，2~3 个）
5. **Key API Patterns**（速查）
6. **Common Pitfalls**（症状 → 原因 → 修法 表格）
7. **Escalation**（指向其他 skill）

> 保持**一条贯穿主线** + **可表格化的速查 / 坑表**。LLM 用得到，人也好读。

### 1.3 references/ 拆分规则

什么放主 SKILL.md、什么拆 references？

| 内容 | 位置 | 理由 |
|---|---|---|
| 心智模型、checklist、坑表、最小模板 | `SKILL.md` | 高频；首次加载就要看到 |
| 完整 API 表（几十条） | `references/api-quick-reference.md` | 长，按需查 |
| 完整可跑模板（带 host code/校验/profiling） | `references/kernel-templates.md` | 长，复制粘贴用 |
| DSL 详细语义、边界情况 | `references/language-docs.md` | 进阶 |
| 完整算子示例（attention 等） | `references/tilelang-examples.md` | 复杂参考 |

**经验法则**：SKILL.md 控制在 ~250 行内，超了就拆。

---

## 2. 三种典型更新场景

### 场景 A：仅修改某一份 reference（最常见）

例：TileLang 升级了，新增了某个 API。

**步骤**：

1. 拉最新 upstream：
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git checkout -b update/api-quick-ref-v0.1.10
   ```
2. 改 `skills/tilelang/<skill>/references/<file>.md`
3. **如果新增/删除了 API，反向检查 `SKILL.md`**：
   - 主入口里有没有引用这个 API？
   - "Common Pitfalls"是不是要补一行？
4. 跑验证（见 §3）
5. commit + push + PR

### 场景 B：新增 / 修改一整个 skill 包

例：写一个新的 `quantization-tilelang-kernels` skill。

**步骤**：

1. 建目录：
   ```bash
   mkdir -p skills/tilelang/quantization-tilelang-kernels/references
   ```
2. 写 `SKILL.md`，照 §1.2 推荐结构。frontmatter 的 `name` = 目录名。
3. 写 references，每份聚焦一个主题。
4. **改顶层 README.md**：在 "TileLang Skills" 表格里加一行；如果该 skill 在工作流图里有位置，也要更新工作流图。
5. **跨 skill 一致性检查**：
   - 其他 skill 的 "Escalation" 段是否要指向新 skill？
   - 触发词有没有跟已有 skill 重叠？重叠会导致 agent 选错 skill。
6. 跑验证。
7. commit + PR。

### 场景 C：修改 SKILL.md 主入口

例：发现某个坑漏写了。

**步骤**：

1. 改 `SKILL.md`
2. 想想这条信息有没有更适合放 references 的部分（例如详细 API 表更新放 `api-quick-reference.md`）
3. 检查 references 之间没有矛盾
4. commit + PR

---

## 3. 验证清单（提 PR 前必跑）

### 3.1 静态检查

```bash
# 1) frontmatter name 与目录名一致
for d in skills/tilelang/*/; do
  name=$(basename "$d")
  fm=$(awk '/^name:/{print $2; exit}' "$d/SKILL.md")
  echo "$name  vs  $fm"
done

# 2) 找死链 (相对路径)
grep -rn "references/" skills/

# 3) 找拼写错误（粗筛）
grep -rn "TODO\|FIXME\|XXX" skills/
```

### 3.2 代码块跑得通

skill 里的代码片段要保证**复制即跑**。建议：

1. 把 `references/kernel-templates.md` 里**所有完整模板**拷到 `evals/` 下当成测试，跑一遍：
   ```bash
   cd evals && python test_<skill>_templates.py
   ```
2. 涉及 GPU 的可以挂 CI 跳过，但**至少手动跑一次**。

### 3.3 触发词不冲突

写新 skill 后，问 agent 一些问题，看会不会路由错：

- 用 `writing` 类触发词试 → 应该路由到 writing 包
- 用 `slow / latency` → profiling
- 用 `nan / wrong result` → debugging
- 用 `tune / faster` → optimizing

如果新 skill 的 `description` 抢了别家的触发词，要么改触发词措辞，要么在两边都加 cross-link。

### 3.4 文档一致性

| 检查项 | 怎么查 |
|---|---|
| 顶层 README 表格有这条 | `grep <skill-name> README.md` |
| Escalation 链路完整 | 在 SKILL.md 末尾搜 "Escalation" |
| `doc/` 下解读文档要不要同步 | 看 `doc/*.md` 有没有引用变更内容 |
| TileLang 版本号一致 | `grep -rn "v0.1." skills/` |

---

## 4. Git 工作流

### 4.1 标准 PR 流程

```bash
# 同步 upstream
git fetch upstream
git checkout main
git merge --ff-only upstream/main
git push origin main

# 起分支
git checkout -b update/<topic>

# 改文件 ...

# 提交
git add skills/ README.md doc/
git commit -m "skill: <动作> <skill 名> -- <一句话原因>"

# 推 fork
git push origin update/<topic>

# 在 GitHub 开 PR：origin/update/<topic> -> upstream/main
```

### 4.2 commit message 建议

照仓库已有风格（见 `git log`）：

```
skill: <动词> <skill 名> -- <原因>
references: update <文件> for <版本/原因>
README: <动作>
doc: <动作>
```

例子（仓库已有）：

- `Revise TileLang skills: portability, self-contained refs, ncu bottleneck guide`
- `Remove device-specific benchmarks, add compare_backward as recommended gradient test`
- `Revise README for cuda_skill`

> 一句话点出**改了什么**和**为什么**，方便回看。

---

## 5. TileLang 版本升级 checklist

> TileLang 还在快速迭代。每次升 minor 版本（如 v0.1.9 → v0.2.0）都按这套走。

```
[ ] 跑 pip install -U tilelang
[ ] python -c "import tilelang; print(tilelang.__version__)"
[ ] 把 evals/ 下所有模板跑一遍，记录失败列表
[ ] 对照 TileLang CHANGELOG，找出 API 变化（新增/废弃/重命名）
[ ] 改动清单逐项处理：
    [ ] api-quick-reference.md   ← 增删 API 表行
    [ ] language-docs.md         ← 语义/边界变化
    [ ] kernel-templates.md      ← 重跑、调容差、改默认 tile
    [ ] tilelang-examples.md     ← 重跑
    [ ] SKILL.md (各 skill)      ← 心智模型 / 坑表
    [ ] README.md                ← 顶部"Based on TileLang v0.x.x"那一行
[ ] 触发词回归测试 (§3.3)
[ ] 顶层一致性检查 (§3.4)
[ ] PR
```

---

## 6. 内容编辑规范

### 6.1 风格

- **代码块标语言**：```python / ```bash / ```cuda
- **表格优于长段落**：症状/原因/修法 一律表格
- **最小可跑代码**：每段 Python 都能复制粘贴跑（imports + host code + 校验）
- **避免空话**：删掉 "this is very important", "as we all know"
- **不要 emoji**（除非用户明确要）

### 6.2 术语统一

| 优先 | 避免 |
|---|---|
| `T.alloc_shared` / `T.alloc_fragment` | "shared mem alloc" |
| `T.Pipelined` | "software pipeline loop" |
| `block_M / block_N / block_K` | `BM/BN/BK` 仅在文字描述时用 |
| `out_idx=[-1]` | "last is output" |

### 6.3 模板的"可跑五件套"

每个完整模板**必须**包含：

```
1. imports                # torch / tilelang / language as T
2. kernel definition       # @tilelang.jit + @T.prim_func
3. parameters              # M, N, K, block_*, dtype
4. host code               # 准备 tensor、调用 kernel
5. validation              # ref_program + assert_close + 容差注释
6. (可选) profiling         # do_bench + tflops 或 bandwidth
```

---

## 7. 常见坑（更新文档时）

| 症状 | 原因 | 修法 |
|---|---|---|
| agent 加载不到新 skill | `name` 字段和目录名不一致 | 同步 |
| agent 把简单问题路由到大 skill | `description` 写得太宽 | 收紧关键词，加场景描述 |
| 模板跑不通 | API 升级后 signature 变了 | 跑 evals 回归 |
| README 表格里多一行但 skill 删了 | 删 skill 没删索引 | `grep` 全仓 |
| references 里互相打架 | 新版改了一处忘了改另一处 | 拉清单逐项扫 |
| 容差太严 / 太松 | 没区分 fp16 GEMM vs transcendental | 见 writing-tilelang-kernels 的容差表 |

---

## 8. 推荐的更新节奏

```
日常 (周/月级)         TileLang 版本升 (季级)        Skill 重构 (半年)
─────────────         ─────────────────────         ────────────────
+ 修小坑               + §5 完整跑一遍                + 重审心智模型
+ 补 API 例子          + 重跑所有模板                  + 拆/合 references
+ 同步 upstream        + 改默认 tile                   + 重写触发词
+ 改 doc/ 解读         + 更新 README 版本行            + 工作流图重画
```

---

## 9. 一页纸 SOP（贴墙版）

```
要改 skill？
  │
  ├── 只改 reference?
  │     1. fetch upstream → checkout -b
  │     2. 改文件
  │     3. 反查 SKILL.md 是否要同步
  │     4. evals 跑模板 → push → PR
  │
  ├── 改 SKILL.md 主入口?
  │     1. 同上 1
  │     2. 改文件，控制 ~250 行
  │     3. references 一致性扫
  │     4. 触发词回归测试
  │     5. push → PR
  │
  └── 新增整个 skill 包?
        1. mkdir + SKILL.md(frontmatter) + references/
        2. 改 README.md 索引表 + 工作流图
        3. 其他 skill 的 Escalation 反向 link
        4. 触发词冲突检查
        5. evals 加测试
        6. push → PR
```

---

## 10. 给后续维护者的几条经验

1. **先跑再改**：TileLang 迭代快，先用现有模板验证当前 skill 是否还正确，再决定改什么。
2. **模板就是文档**：`kernel-templates.md` 里的模板本身就是最好的教学。**保它们能跑**比写多少话都重要。
3. **触发词比内容更影响 agent 体验**：description 里多列同义词、API 名、用户口语。
4. **坑表是金子**：每次 debug 出新坑，立刻补 SKILL.md 的 "Common Pitfalls"，未来自己受益。
5. **doc/ 是自留地**：`skills/` 是给 agent 吃的，`doc/` 是给人看的。两边可以风格不同，但不要矛盾。
6. **小步提交**：一次 PR 改一类事（光改 reference / 光升版本 / 光重构）。混在一起难 review。

---

> 本文档保存路径：`doc/skill-更新指南.md`  
> 配套阅读：`doc/writing-tilelang-kernels-解读.md`
