# `optimizing-tilelang-programs` Skill 详细解读

> 给"kernel 已经能跑、想榨更多性能"的同学的进阶指南。把 skill 包里 5 份原始文件压成一篇 —— **方法论 + 8 项 checklist + AutoTuner + 坑表**。

---

## 0. 一图：优化的整体定位

```
   写出 kernel              算对了                  跑得慢
─────────────────       ─────────────────       ──────────────────
writing-tilelang-       debugging-              profiling-tilelang-
kernels                 tilelang-programs       programs
                                                 │
                                                 │ 测出基线 + 找瓶颈类型
                                                 ▼
                                          ┌──────────────────────┐
                                          │  optimizing-tilelang │ ← 本 skill
                                          │  -programs           │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                                              再回到 profiling 验证
```

**这个 skill 的前置条件**：
1. **正确性已过**：`profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)`
2. **有基线数据**：知道当前的 latency / TFLOPS（来自 profiling skill）
3. **知道是 compute-bound 还是 memory-bound**：决定要按哪条路优化
4. **已用 ncu 定位具体瓶颈**：tensor core / CUDA core / memory / latency bound

> 没有这 4 条就开始优化 = 瞎猜。

---

## 1. Skill 包文件结构

```
skills/tilelang/optimizing-tilelang-programs/
├── SKILL.md                              ← 主入口（本文重点）
└── references/
    ├── optimization-checklist.md         ← 8 项 checklist 详细版
    ├── autotuning-guide.md               ← AutoTuner 完整教程
    ├── autotuning-doc.md                 ← 官方 doc 摘录
    └── autotune-example.md               ← TileLang 仓库 GEMM autotune 完整 example
```

| 文件 | 什么时候看 |
|---|---|
| `SKILL.md` | 每次优化先扫一遍 |
| `optimization-checklist.md` | 每条优化要看代码 / 公式时 |
| `autotuning-guide.md` | 用 AutoTuner |
| `autotune-example.md` | 想用 roller 自动生成 config / 看完整脚本 |

---

## 2. 优化方法论：Change-One-Thing

GPU 性能里 tile size、pipeline 深度、线程数、硬件约束**互相耦合**。唯一可靠的方法 = **一次只改一个参数**。

```
1. 测基线
2. 改一个参数
3. 再测
4. 记表
5. 好就留，坏就 revert
6. 回到第 2 步
```

> 一次改两个，永远不知道是谁起的作用。

### 2.1 结果记录模板

```
| Config                                           | Latency (ms) | TFLOPS | Notes      |
|--------------------------------------------------|--------------|--------|------------|
| baseline: bM=128 bN=128 bK=32 stages=2 thr=128   | 1.20         | 56     |            |
| change bK=64                                     | 1.05         | 64     | +14% ← 留  |
| change thr=256                                   | 1.00         | 67     | +5% 累计   |
| change stages=3                                  | 1.10         | 61     | -10% revert|
```

记表的**核心价值**：以后回看每一步的因果关系，不用重新猜。

---

## 3. 八项优化 Checklist（按优先级）

> 全部从 SKILL.md + `optimization-checklist.md` 提炼。**按顺序做**，前面的影响最大。

### 3.1 Tile 大小（影响最大）

`block_M × block_N` 决定每个 block 处理多少工作。tile 越大，**摊薄**了搬运 shared memory 的成本，但**吃**更多 SRAM 和寄存器。

```
shared_bytes = (block_M * block_K + block_K * block_N) * dtype_bytes * num_stages
```

例：`128×128×32`，fp16，2 stages → `(128*32 + 32*128) * 2 * 2 = 32768 = 32 KB`。

| GPU | 默认 SRAM | opt-in 上限 |
|---|---|---|
| Blackwell sm_120 | 48 KB | 228 KB |
| Hopper sm_90 | 48 KB | 228 KB |
| Ampere sm_80 | 48 KB | 164 KB |

**选 tile 的经验**：

| 场景 | 推荐 tile | 理由 |
|---|---|---|
| GEMM (compute-bound) | 128×128 | 多数 GPU 的 sweet spot |
| 长方形 GEMM (M≫N or N≫M) | 256×128 / 128×256 | 匹配形状 |
| 内存受限（elementwise / reduction） | 64×64 / 32×128 | tile 大也没用 |
| SRAM 紧 | ≥ 64×64 | 别更小 |

> 64×64 → 128×128 通常是**单步增益最大**的优化。

### 3.2 内层 tile `block_K`

`block_K` = 每个 pipelined 循环迭代里 reduction 维度的步长。

```
block_K 大 → 循环次数少 → 每个 stage 多用 SRAM
```

| block_K | 128×128 / 2 stages 占用 |
|---|---|
| 32 | 32 KB |
| 64 | 64 KB |

**何时增大**：compute-bound + SRAM 还有空 + K 维度大。  
**何时不动**：memory-bound（增大也没用）/ K 已经很小。

### 3.3 Pipeline 深度 `num_stages`

`T.Pipelined(iters, num_stages=N)` —— 让搬运和计算 overlap。

```
num_stages=0    →  无 pipeline。算完一块再搬下一块。debug 用。
num_stages=2    →  双缓冲。算 N 块时搬 N+1 块。安全默认。
num_stages=3    →  三缓冲。N 块算，N+1 块在路上，N+2 块发请求。
```

| num_stages | 128×128×32 占用 |
|---|---|
| 2 | 32 KB |
| 3 | 48 KB |

> `2 → 3` 通常变化很小，**SRAM 紧时反而更慢**。先 2，遇到 memory-latency-bound 再尝试 3。

### 3.4 线程数 `threads`

`T.Kernel(..., threads=N)`：每 block 多少线程。

| 默认 | 升级条件 |
|---|---|
| 128 | tile ≥ 128×128 时可以试 256 |

线程多 → 寄存器压力大 → occupancy 可能反而降。**跟 ncu 看实际 occupancy**。

### 3.5 L2 Swizzle

```python
T.use_swizzle(panel_size=10, enable=True)
```

把 block 的执行顺序重排，让相邻 block 共用同一段 B 矩阵 → **L2 命中率↑**。

| 何时有效 | 何时没用 |
|---|---|
| N ≥ 8192 | N ≤ 4096，B 已经在 L2 |
| 多行输出共享 B 块 | memory-bound（瓶颈不在 L2） |

`panel_size`：试 4 / 10 / 16，跟 L2 大小和 N 相关。

### 3.6 内存访问模式

**三件套**：

1. **Coalesced reads**：行主序 tensor，线程沿最内维读。`T.copy` 自动处理。前提是 `assert A.is_contiguous()`。
2. **Vectorized loads**：内层维度 fp16 是 8 的倍数 / fp32 是 4 的倍数 → 能用 128-bit load。
3. **Bank conflicts**：32 个 bank，warp 内多线程命中同 bank 会序列化。一般 TileLang layout inference 帮搞定。怀疑时：
   ```bash
   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum python script.py
   ```

### 3.7 Epilogue Fusion（融合后处理）

每多一次 kernel launch：
- ~5–10 μs 启动开销
- 一次 global memory 写 + 下次 kernel 再读

**做法**：把 GEMM 后的激活 / bias / scaling 写到同一个 kernel 的最后一段 `T.Parallel`，**写回前在寄存器里改完**。

```python
# 不要这样（2 个 kernel，2 次 global 写）
C = matmul(A, B); C = sigmoid(C)

# 要这样（1 个 kernel，1 次写）
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
    T.copy(A[by*BM, ko*BK], A_shared)
    T.copy(B[ko*BK, bx*BN], B_shared)
    T.gemm(A_shared, B_shared, C_local)

# epilogue：在 fragment 寄存器上做完
for i, j in T.Parallel(BM, BN):
    C_local[i, j] = T.sigmoid(C_local[i, j])

T.copy(C_local, C[by*BM, bx*BN])     # fp32 → fp16 cast 也在这一步发生
```

**典型可融合操作**：activation（ReLU / sigmoid / GELU）、bias add、scaling、fp32→fp16 cast、ReLU + bias + cast 多合一。

### 3.8 Layout Annotation

针对 `T.atomic_add` 类 kernel（**反向传播 dQ 累加**最典型）：

```python
from tilelang.utils import make_dq_layout
T.annotate_layout({dQ: make_dq_layout(dQ)})
```

不写的话，atomic 写会散乱命中 → 慢。`ncu` 看到 store efficiency 低就上。

> 主要是 backward kernel 的事，详见 `testing-fwd-bwd-kernels` skill。

---

## 4. 不同 kernel 的优先级速查

| Kernel 类型 | Priority 1 | Priority 2 | Priority 3 |
|---|---|---|---|
| GEMM | tile 128×128 | block_K=64 | threads=256 |
| GEMM + epilogue | **融合 epilogue** | tile sizes | pipeline 深度 |
| Elementwise | 向量化 load | tile sizes | shared staging |
| Reduction | block_N（宽 reduce） | coalescing | shared memory |
| Attention fwd | block_M / block_N | num_stages | causal mask |
| Attention bwd | dQ atomic layout | Split-K | 线程数 |

---

## 5. 进阶技巧（一般用不到）

### 5.1 Split-K

M 或 N 很小但 K 很大时，单 tile 占住整个 M/N → 大量 SM idle。

```
方法：把 K 切成 split_k 段，每段独立做局部 reduction，
最后 atomic_add 合并结果。
通常由 autotuner 搜出来。
```

### 5.2 Persistent Kernel

很多个小 GEMM 串行启动时，让线程**一直存活、连续吃多个 tile**，省 launch 开销。Flash attention 类用得多。

### 5.3 Warp Specialization

block 内不同 warp 干不同事（一些 warp 搬数据，另一些算）。Hopper / Blackwell 的 TMA 内核常用。

---

## 6. AutoTuner：自动搜 config

手调累的时候用 `AutoTuner`。**用编程式 API**，不要用 `@tilelang.autotune` 装饰器（带 `ref_prog` 函数时有 cache 序列化 bug，见 §8）。

### 6.1 何时该用 AutoTuner

| 用 | 不用 |
|---|---|
| kernel 能跑但不知道 tile 怎么定 | memory-bound（tile 影响小） |
| 多 size 部署，每个 size 最优可能不同 | 已经知道最优 config |
| 手调到瓶颈想试更多 | 一次性实验（< 5 个 config 手测更快） |
| 换 GPU 架构 | |

### 6.2 编程式 API 五步

```python
from tilelang.autotuner import AutoTuner
import tilelang as tl

# 1) kernel 函数（必须 return @T.prim_func，参数包含要调的 tile/stages/threads）
@tl.jit(out_idx=[-1])
def my_kernel(M, N, K, block_M, block_N, block_K,
              num_stages=2, threads=128, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype),
               B: T.Tensor((K, N), dtype),
               C: T.Tensor((M, N), dtype)):
        ...
    return kernel

# 2) 候选 config
configs = [
    {"block_M": 64,  "block_N": 64,  "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 3, "threads": 256},
]

# 3) 参考实现（对数）
def ref_program(A, B):
    return A @ B

# 4) 装配
autotuner = (
    AutoTuner.from_kernel(my_kernel, configs)
    .set_compile_args(out_idx=[-1], target="auto")
    .set_profile_args(
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program,
        warmup=3,
        rep=20,
    )
)

# 5) 跑
result = autotuner.run()
print(f"Best config: {result.config}")
print(f"Best latency: {result.latency:.4f} ms")
best_kernel = result.kernel       # 直接拿来用
```

### 6.3 Config 搜索空间生成

**快速搜索**（~5 秒，3–5 个）：

```python
configs = [
    {"block_M": 64,  "block_N": 64,  "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 128},
]
```

**全面搜索**（~30–60 秒，16–32 个）：

```python
configs = [
    {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": ns, "threads": thr}
    for bm in [64, 128]
    for bn in [64, 128]
    for bk in [32, 64]
    for ns in [2, 3]
    for thr in [128, 256]
]
```

**预过滤**（省时间）：

```python
def is_valid_config(cfg, dtype_bytes=2):
    shared = (cfg["block_M"]*cfg["block_K"] + cfg["block_K"]*cfg["block_N"]) \
             * dtype_bytes * cfg["num_stages"]
    return shared <= 228 * 1024     # Blackwell 上限

configs = [c for c in configs if is_valid_config(c)]
```

### 6.4 Roller：自动出 tensor-core 友好的 config

来自 `autotune-example.md`，靠 TileLang 的 `MatmulTemplate.recommend_hints()`：

```python
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA, CDNA
from tilelang.carver.roller.rasterization import NoRasterization

arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
carve = MatmulTemplate(M=M, N=N, K=K, in_dtype=T.float16,
                       out_dtype=T.float16, accum_dtype=T.float32).with_arch(arch)
roller_hints = carve.recommend_hints(topk=20)

configs = []
for hint in roller_hints:
    bm, bn = hint.block
    wm, wn = hint.warp
    configs.append({
        "block_M": bm,
        "block_N": bn,
        "block_K": hint.rstep[0],
        "num_stages": hint.pipeline_stage if hint.pipeline_stage > 1 else 0,
        "thread_num": (bm // wm) * (bn // wn) * 32,
        "enable_rasteration": hint.rasterization_plan is not NoRasterization,  # 注意拼写历史包袱
    })
```

> Roller 是设备感知的，**比手写网格通常更靠谱**，但搜索空间小。

### 6.5 启发式：免搜，按 SM 版本直选

来自 example：

```python
def get_heuristic_config():
    sm = sm_major * 10 + sm_minor
    if sm == 80:    # Ampere
        return {"block_M": 128, "block_N": 256, "block_K": 32, "num_stages": 2,
                "thread_num": 128, "enable_rasteration": True}
    elif sm == 90:  # Hopper
        return {"block_M": 128, "block_N": 256, "block_K": 64, "num_stages": 3,
                "thread_num": 256, "enable_rasteration": True}
    else:           # Blackwell 等
        return {"block_M": 128, "block_N": 256, "block_K": 32, "num_stages": 0,
                "thread_num": 128, "enable_rasteration": True}
```

跑一次 ncu 就够，**生产里直接用启发式**比每次 autotune 快得多。

### 6.6 解读 AutoTuner 结果

| 信号 | 含义 |
|---|---|
| 最佳比默认快 10–50% | 正常（compute-bound） |
| 多个 config 互相在 5% 内 | 最优面平稳，可放心用 |
| 所有 config 几乎一样快 | **memory-bound**，调 tile 没用 |
| 最优 tile 特别大 | 可能不通用到其他 size |
| 4096 最优 ≠ 1024 最优 | 必须**按 size 分别 autotune** |

### 6.7 Per-Size Tuning

```python
for M, N, K in [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192)]:
    autotuner = (
        AutoTuner.from_kernel(
            lambda bM, bN, bK, **kw: my_kernel(M, N, K, bM, bN, bK, **kw),
            configs,
        )
        .set_compile_args(out_idx=[-1], target="auto")
        .set_profile_args(supply_type=tl.TensorSupplyType.Integer, ref_prog=ref_program)
    )
    result = autotuner.run()
    print(f"{M}×{N}×{K}: best={result.config}, lat={result.latency:.4f} ms")
```

---

## 7. Before / After 基准模板

每次优化都要套这个：

```python
import tilelang
import torch

M, N, K = 4096, 4096, 4096
def ref_program(A, B): return A @ B

# 基线
baseline = my_kernel(M, N, K, block_M=128, block_N=128, block_K=32)
prof = baseline.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
prof.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)
base_lat = prof.do_bench(warmup=25, rep=100, return_mode="median")
base_tflops = 2*M*N*K / base_lat * 1e-9

# 优化后
opt = my_kernel(M, N, K, block_M=128, block_N=128, block_K=64)
prof2 = opt.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
prof2.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)
opt_lat = prof2.do_bench(warmup=25, rep=100, return_mode="median")
opt_tflops = 2*M*N*K / opt_lat * 1e-9

# 参考（cuBLAS via PyTorch）
ref_lat = prof.do_bench(ref_program, warmup=25, rep=100, return_mode="median")
ref_tflops = 2*M*N*K / ref_lat * 1e-9

print(f"Baseline:  {base_lat:.4f} ms ({base_tflops:.1f} TFLOPS)")
print(f"Optimized: {opt_lat:.4f} ms ({opt_tflops:.1f} TFLOPS) [{opt_tflops/base_tflops:.2f}x]")
print(f"cuBLAS:    {ref_lat:.4f} ms ({ref_tflops:.1f} TFLOPS)")
```

固定流程：**正确性 → 基线 → 改 → 正确性 → 优化值 → vs cuBLAS**。

---

## 8. 重要 Caveats

| Caveat | 后果 |
|---|---|
| **最优 config 随 size 变** | 4096 上比 cuBLAS 快，1024 上可能输 → 在目标生产 size 上 bench |
| **每变一次 tile 就重编一次** | 单 kernel < 1 秒，autotune 几十 config 就要算 |
| **SRAM 上限**：默认 48 KB / opt-in 164–228 KB | 超了：要么 launch 失败，要么静默走慢内存 |
| **TMA 对齐**（Blackwell / Hopper） | 内层 fp16 必须 8 倍数；fp32 必须 4 倍数。否则报 invalid TMA descriptor |

---

## 9. 常见坑

| 坑 | 后果 | 修法 |
|---|---|---|
| 一次改多个参数 | 不知谁起作用 | Change-One-Thing |
| 在错的 size 上调 | 生产环境其实更慢 | 用目标 size bench |
| 改完不验正确性 | 错而快 = 没用 | 每次都 `assert_allclose` |
| Pipeline stages 太多 | SRAM 溢出，反而慢 | 从 2 起，看 ncu 再加 |
| 用 `@tilelang.autotune` 装饰器 + `ref_prog` 函数 | `TypeError: not JSON serializable` | 用 `AutoTuner.from_kernel()` 编程式 API |
| 觉得 tile 越大越好 | 寄存器压力↑、occupancy↓ | 128×128 是 GEMM 起点，再大要测 |

---

## 10. 升级路径

```
优化后变错  →  debugging-tilelang-programs
要更准的基线 →  profiling-tilelang-programs
从零写 kernel →  writing-tilelang-kernels
反向传播 kernel
   先看架构   →  testing-fwd-bwd-kernels
   再回来调   →  本 skill
```

---

## 11. 一页纸 SOP（贴墙版）

```
                    ┌─────────────────────────────┐
                    │ 0. 正确性已过 + 有基线        │
                    │    + 知道是 compute/memory   │
                    │      bound + ncu 瓶颈分类    │
                    └────────────┬─────────────────┘
                                 ▼
              ┌──────── Change-One-Thing 循环 ────────┐
              │                                        │
              │  按优先级试这 8 项：                    │
              │   1. tile size  (128×128 起)          │
              │   2. block_K    (32 → 64)             │
              │   3. num_stages (2 → 3)               │
              │   4. threads    (128 → 256)           │
              │   5. swizzle    (N≥8192 才上)         │
              │   6. memory: vec load + coalesce      │
              │   7. epilogue 融合                     │
              │   8. layout annotation (atomic add)   │
              │                                        │
              │  每次：改 → assert_allclose → bench    │
              │        → 记表 → 留 / revert            │
              └─────┬──────────────────────────────────┘
                    │ 手调到 plateau 或 config 超 10 个
                    ▼
                ┌──────────────────────────────┐
                │ AutoTuner.from_kernel(...)   │
                │ + Roller (tensor-core hint)  │
                │ 或固化为启发式 by SM         │
                └──────────────────────────────┘
                    │
                    ▼
              对比 cuBLAS，记最终 config
```

---

> 源 skill 路径：`skills/tilelang/optimizing-tilelang-programs/`  
> TileLang 版本：v0.1.9  
> 配套阅读：
> - `doc/writing-tilelang-kernels-解读.md`（写 kernel 的基础）
> - `doc/skill-更新指南.md`（改这套 skill 的方法）
