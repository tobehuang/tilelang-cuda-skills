# `writing-tilelang-kernels` Skill 详细解读

> 一份给"想用 TileLang 写 GPU 算子"的同学的入门 + 速查文档。把 skill 包里 5 份原始文件（SKILL.md + 4 份 references）压缩成一篇，**心智模型 + 模板 + 速查 + 坑** 一条龙。

---

## 0. 一图总览：TileLang 是什么

TileLang 是一个 **tile 级别的 GPU DSL**，写起来像 Python，编译产物是 CUDA / HIP。它把"线程 × 内存层级"两件事抽象成两套原语：

```
                     ┌─────────────────────────────────────┐
                     │           Global Memory             │  ← T.Tensor 参数住这里
                     └────────────┬────────────────────────┘
                                  │ T.copy / T.async_copy
                     ┌────────────▼────────────────────────┐
                     │         Shared Memory (SRAM)        │  ← T.alloc_shared
                     └────────────┬────────────────────────┘
                                  │ T.copy / T.gemm
                     ┌────────────▼────────────────────────┐
                     │       Register / Fragment           │  ← T.alloc_fragment
                     └─────────────────────────────────────┘

         block 网格 = T.Kernel(grid_x, grid_y, threads=N)
                  ↑                ↑
            T.ceildiv(N, BN)  T.ceildiv(M, BM)
```

写 kernel 的固定 4 步骨架：

1. **alloc**：开 shared / fragment
2. **clear**：累加器清零
3. **Pipelined 循环**：copy + 计算（`T.gemm` 或 `T.Parallel` 算式）
4. **copy 写回**：fragment → global

只要记住这 4 步，90% 的 kernel 都是改第 3 步的算式而已。

---

## 1. Skill 包文件结构

技能根目录：`skills/tilelang/writing-tilelang-kernels/`

| 文件 | 角色 | 什么时候读 |
|---|---|---|
| `SKILL.md` | **总入口**：checklist + 心智模型 + 工作流 + 模板 + 坑 | 每次写新 kernel 先扫一遍 |
| `references/kernel-templates.md` | **7 个完整可跑模板**（含 host code、校验、profiling） | 找最像的复制改 |
| `references/api-quick-reference.md` | 全 API 速查表（v0.1.9） | 写到不会的算子查 |
| `references/language-docs.md` | DSL 语言文档：async、TMA、warp 原语、dynamic shape | 进阶/边界 |
| `references/tilelang-examples.md` | 真实大 kernel（attention、softmax 等） | 复杂场景参考 |

读这套 skill 的推荐顺序：**SKILL.md → kernel-templates.md → 写代码时再翻 api/language 速查**。

---

## 2. 写 kernel 之前的 Checklist（4 步）

> 这是 SKILL.md 的"飞行前检查"。别跳。

1. **环境验证**
   ```bash
   python -c "import tilelang; print(tilelang.__version__)"
   ```
2. **明确目标算子**：input/output 的 shape、dtype、数学定义。
3. **先写 PyTorch 参考实现** —— 后面对数全靠它：
   ```python
   def ref_program(A, B):
       return A @ B
   ```
4. **找最近的 pattern**：照下表对号入座。

| Pattern | 看什么关键字 |
|---|---|
| GEMM | `T.gemm`、shared memory、pipelining |
| GEMM + fusion | GEMM 后接 ReLU/sigmoid 等 epilogue |
| Elementwise | `T.Parallel`、shared staging |
| Reduction / Norm | `T.reduce_sum`，两遍扫描 |
| Online Softmax | running max、exp、rescale |
| Flash Attention | 多 fragment、循环里做 softmax |
| Linear Attention | 分 chunk、运行态 |

---

## 3. Kernel 解剖（核心心智模型）

### 3.1 两层结构

每个 TileLang kernel 都是 **Python 外层 + TIR 内层** 的双层套娃：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])                    # ← 外层：编译期参数
def my_kernel(M, N, K, block_M, block_N, block_K,
              dtype=T.float16, accum_dtype=T.float32):

    @T.prim_func                                # ← 内层：真正的 GPU kernel
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N),
                      T.ceildiv(M, block_M),
                      threads=128) as (bx, by):

            # 1) 分配片上内存
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 2) 清零累加器
            T.clear(C_local)

            # 3) Pipelined 主循环：copy + 计算
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # 4) 写回 global
            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel                               # ← 必须 return 内层
```

记忆口诀：**外层吃 shape、dtype、tile，内层做 kernel**。

### 3.2 三层内存

```
T.alloc_shared(shape, dtype)    # 片上 SRAM；block 内所有线程共享
                                # 用途：从 global 暂存 tile

T.alloc_fragment(shape, dtype)  # 寄存器；形状写起来像完整 tile
                                # 编译器靠 layout inference 自动分到各线程
                                # 用途：累加器、中间结果

T.alloc_local(shape, dtype)     # 显式 thread-private
                                # 用途：标量累加
```

> ⚠️ `fragment` 的 shape 是"虚拟 tile 视角"，**不是每个线程真的拿到那么大的寄存器**，编译器会切。

### 3.3 数据搬运：一招 `T.copy` 走天下

```python
T.copy(A[by * BM, ko * BK], A_shared)  # global  → shared
T.copy(C_local, C[by * BM, bx * BN])   # fragment → global
```

`T.copy` 自动做 coalescing 和 scope 适配，**extents 从 dst 形状反推**。

需要手动重叠 cp.async + 计算时用 `T.async_copy`（要自己配 `T.ptx_wait_group`，见 §7）。

### 3.4 `@tilelang.jit` 参数

| 写法 | 行为 |
|---|---|
| `out_idx=[-1]` | 最后一个参数是输出，调用时不传，kernel 返回它 |
| `out_idx=[2, 3]` | 第 2、3 个参数是输出 |
| 不写 `out_idx` | 调用方自己 alloc 全部 buffer 并传入；kernel 返回 None |
| `target="cuda"` | 可省，从输入 tensor 推 |

```python
# 有 out_idx=[-1]
c = kernel(a, b)        # 传 N-1 个，返回 c

# 无 out_idx
kernel(a, b, c)         # 传 N 个，自己 alloc c
```

### 3.5 `T.Kernel` 网格

```python
T.Kernel(grid_x, grid_y, threads=128) as (bx, by)         # 2D grid
T.Kernel(grid_x, grid_y, grid_z, threads=128) as (bx, by, bz)  # 3D
```

- `bx, by` 就是 `blockIdx.x / .y`
- `threads` 一般 128 或 256
- grid 维度永远用 `T.ceildiv(dim, block_dim)`，**别用 `dim // block_dim`**

---

## 4. Step-by-Step 工作流

```
1. 写 PyTorch ref     →  ref_program(*args)
2. 选 pattern         →  elementwise / reduction / gemm / fused
3. 选 tile size       →  见下表默认值
4. 写 kernel          →  抄最近的模板
5. compile + run      →  kernel = my_kernel(...); c = kernel(a, b)
6. 数值校验           →  torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
7. 看生成 CUDA(可选)  →  print(kernel.get_kernel_source())
```

**默认 tile 参数**（先 work 起来再调）：

| 算子类型 | block_M | block_N | block_K | num_stages | threads |
|---|---|---|---|---|---|
| GEMM | 128 | 128 | 32 | 2~3 | 128 |
| Elementwise | 64 | 64 | — | — | 128 |
| Reduction（沿 N） | 64 | 128 | — | — | 128 |

---

## 5. 模板速览（共 7 个）

完整代码在 `references/kernel-templates.md`，每个模板都带 host code + 校验 + profiling，可直接 copy-paste。

### 5.1 1D Vector Scale

`Y = X * scale`，最简模板，演示 1D grid + `T.Parallel`。

```python
with T.Kernel(T.ceildiv(N, block_size), threads=128) as (bx,):
    X_shared = T.alloc_shared((block_size,), dtype)
    Y_local  = T.alloc_fragment((block_size,), dtype)
    T.copy(X[bx * block_size], X_shared)
    for i in T.Parallel(block_size):
        Y_local[i] = X_shared[i] * T.cast(scale_val, dtype)
    T.copy(Y_local, Y[bx * block_size])
```

### 5.2 2D Elementwise — Softplus

`Y = log(1 + exp(X))`，演示**跨 dtype 计算**：fp16 输入 → fp32 算 → fp16 写回。

```python
T.copy(X[by * BM, bx * BN], X_shared)
for i, j in T.Parallel(BM, BN):
    val = T.cast(X_shared[i, j], accum_dtype)
    Y_local[i, j] = T.log(T.cast(1.0, accum_dtype) + T.exp(val))
T.copy(Y_local, Y[by * BM, bx * BN])
```

### 5.3 Row Sum（Reduction）

每行求和，演示 `T.reduce_sum(..., dim=1)` + 多块分批累加：

```python
T.clear(acc)
for ko in T.serial(T.ceildiv(N, block_N)):
    T.copy(A[bx * BM, ko * BN], A_shared)
    T.reduce_sum(A_shared, local_sum, dim=1)
    for i in T.Parallel(BM):
        acc[i] = acc[i] + local_sum[i]
T.copy(acc, Out[bx * BM])
```

### 5.4 GEMM Minimal

经典模板，§3.1 已贴。**这是所有 GEMM 系算子的母模板**。

### 5.5 GEMM + Sigmoid Epilogue（融合）

GEMM 算完不要直接写回，**先在 fragment 里 elementwise 改一下**再 copy：

```python
for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=num_stages):
    T.copy(A[...], A_shared); T.copy(B[...], B_shared)
    T.gemm(A_shared, B_shared, C_local)

# 融合的 epilogue（in-register）
for i, j in T.Parallel(BM, BN):
    C_local[i, j] = T.sigmoid(C_local[i, j])

T.copy(C_local, C[...])
```

> 这是把 `(A @ B)` 跟 sigmoid 融成单 kernel 的关键：epilogue 必须放在最后那次 `T.copy` 之前。

### 5.6 Dynamic Shape GEMM

编译一次，跑多种 size：

```python
@tilelang.jit(out_idx=[-1])
def dynamic_matmul(block_M, block_N, block_K, num_stages=2, ...):
    M = T.dynamic("M")
    N = T.dynamic("N")
    K = T.dynamic("K")
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype), ...): ...
    return kernel

kernel = dynamic_matmul(128, 128, 32)
for (M, N, K) in [(256,256,256), (512,1024,256), (1024,1024,1024)]:
    a, b = ...
    c = kernel(a, b)            # 不重新编译
```

bench 时要传 `dynamic_symbolic_constraints={"M":M, "N":N, "K":K}`。

### 5.7 GEMM with Transpose B（Q @ K^T 模式）

B 存为 (N, K)，**逻辑上转置**：

```python
A_shared = T.alloc_shared((BM, BK), dtype)
B_shared = T.alloc_shared((BN, BK), dtype)        # ← 注意 (BN, BK)
T.copy(A[by * BM, ko * BK], A_shared)
T.copy(B[bx * BN, ko * BK], B_shared)              # 取整行
T.gemm(A_shared, B_shared, C_local, transpose_B=True)
```

attention 的 Q@K^T 全靠这个模式。

---

## 6. API 速查（v0.1.9）

按用途分组，更全的版本见 `references/api-quick-reference.md`。

### 6.1 结构

| API | 说明 |
|---|---|
| `@tilelang.jit(out_idx=[], target="cuda")` | JIT 装饰器 |
| `@T.prim_func` | 声明 kernel 函数 |
| `T.Tensor((shape), dtype)` | 声明 buffer 参数 |
| `T.Kernel(gx, gy, threads=N)` | launch context |
| `tilelang.compile(func, target, backend)` | AOT 编译（替代 jit） |

### 6.2 内存

| API | 说明 |
|---|---|
| `T.alloc_shared(shape, dtype)` | 片上 SRAM |
| `T.alloc_fragment(shape, dtype)` | 寄存器 fragment |
| `T.alloc_local(shape, dtype)` | thread-private |
| `T.alloc_var(dtype)` | 单标量 |
| `T.alloc_global(shape, dtype)` | 动态 global |

### 6.3 数据 / 计算

| API | 说明 |
|---|---|
| `T.copy(src, dst)` | 通用 copy（同步语义） |
| `T.async_copy(src, dst)` | 显式 cp.async；要配 `T.ptx_wait_group` |
| `T.gemm(A_s, B_s, C_f, transpose_A, transpose_B)` | 张量核 GEMM |
| `T.reduce_sum/max/min(src, dst, dim=N)` | tile 级 reduction |
| `T.cumsum(src, dst, dim=N)` | 前缀和 |

### 6.4 数学

`T.exp / T.exp2 / T.log / T.log2 / T.rsqrt / T.sigmoid / T.max / T.min / T.abs / T.cast / T.if_then_else`

### 6.5 Buffer 操作

| API | 说明 |
|---|---|
| `T.clear(buf)` | 清零 |
| `T.fill(buf, value)` | 填值 |
| `T.atomic_add(dst, value)` | 原子加（global） |

### 6.6 循环

| API | 说明 |
|---|---|
| `T.Pipelined(iters, num_stages=N)` | 软件流水，自动 overlap copy+compute |
| `T.Parallel(d0, d1, ...)` | 并行 elementwise loop（映射到线程） |
| `T.serial(start, stop)` | 顺序循环 |
| `T.unroll(start, stop)` | 编译期 unroll |
| `T.Persistent(domain, wave_size, idx)` | persistent block |

### 6.7 优化提示

| API | 说明 |
|---|---|
| `T.use_swizzle(panel_size=10)` | L2 cache swizzle |
| `T.annotate_layout({buf: layout})` | 显式 layout |
| `T.annotate_l2_hit_ratio(buf, ratio)` | L2 命中率提示 |

### 6.8 常量 / 工具 / dtypes

- `T.infinity(dtype)`：softmax 初始用 `-T.infinity(dtype)`
- `T.ceildiv(a, b)`
- `T.dynamic("name")` / `T.dyn["name"]`：动态维度
- dtypes：`T.float16/32/64`、`T.bfloat16`、`T.int8/16/32/64`、`T.uint8/16/32`、`T.float8_e4m3 / e5m2`

### 6.9 Debug / Profiling

```python
T.print(obj, msg='')            # kernel 内部打印（thread 0）
T.device_assert(cond, msg='')

print(kernel.get_kernel_source())               # 看生成 CUDA
profiler = kernel.get_profiler()
profiler.do_bench(warmup=25, rep=100)           # 跑延迟
profiler.assert_allclose(ref_program, rtol, atol)
profiler.assert_consistent(repeat=N)            # 检查 race
```

---

## 7. `T.copy` vs `T.async_copy`（容易踩的语义差）

```
T.copy(src, dst)           T.async_copy(src, dst)
─────────────              ─────────────────────────
同步语义                    显式 cp.async
调用后立即可用 dst          调用后 dst **未必到位**
编译器自动选机制             一定走 cp.async；降不下去就编译失败
   - SIMT copy             不自动插 wait
   - ldmatrix              ↓ 必须自己写：
   - TMA                   T.ptx_wait_group(0)
   - cp.async（自动 commit + wait）
```

何时用 `async_copy`：手写流水 / warp-specialized 内核，要把 copy 和 compute 显式 overlap。一般 case 写 `T.copy` 就够。

---

## 8. 常见坑（高价值！）

| 症状 | 原因 | 修法 |
|---|---|---|
| 输出 NaN / 垃圾 | 累加前忘记 `T.clear(C_local)` | 加 clear |
| `T.gemm K shape check failed: K_A=X, K_B=Y` | shape 不匹配 | 必须 A=(BM,BK), B=(BK,BN), C=(BM,BN) |
| 输出全是看似随机的值 | 漏掉最后 `T.copy(result, Out[...])` 写回 | 补 copy |
| `Kernel expected N inputs, but M are provided` | `out_idx` 用错 | 有 `out_idx=[-1]`：传 N-1 个；无：传全部 N 个 |
| `Invalid TMA descriptor: globalStrides must be multiple of 16 bytes` | Hopper/Blackwell 内层维度未对齐 | fp16 内层 = 8 的倍数；fp32 = 4 的倍数 |
| 编译过但 hang | 循环次数算错 | `T.Pipelined(T.ceildiv(K, block_K), ...)`，**不是 `K`** |

---

## 9. 进阶语言特性（来自 `language-docs.md`）

> 进阶玩家专区。一般 kernel 用不到。

### 9.1 Dynamic Shape 两种写法

```python
# 写法 1：注解级（推荐写在签名里）
K = T.dyn['K']
@T.prim_func
def f(A: T.Tensor((K,), 'float32')):
    N = A.shape[0]                # 用 buffer.shape 取实际值
    for i in T.serial(N): ...

# 写法 2：表达式级（要在 body 里用 K 这个符号）
K = T.dynamic('K', 'int32')
@T.prim_func
def g(A: T.Tensor((K,), 'float32')):
    for i in T.serial(K): ...     # 直接用 K
```

> `T.symbolic` 已 deprecated，统一用 `T.dynamic`。

### 9.2 Warp 级原语

- **Vote / Ballot**：`T.any_sync`、`T.all_sync`、`T.ballot_sync`、`T.activemask`
- **Shuffle**：`T.shfl_sync`、`T.shfl_xor`、`T.shfl_down`、`T.shfl_up`
- **Match**（仅 sm_70+，HIP 无）：`T.match_any_sync`、`T.match_all_sync`
- **Warp reducer**：`T.warp_reduce_sum / max / min / bitand / bitor`
- **Lane/warp idx**：`T.get_lane_idx()`、`T.get_warp_idx_sync()`

### 9.3 同步

```python
T.sync_threads()        # __syncthreads()
T.sync_warp()           # __syncwarp()
T.sync_grid()           # cooperative grid barrier
T.syncthreads_count(p)  # __syncthreads_count
T.syncthreads_and(p)
T.syncthreads_or(p)
```

### 9.4 Atomic

`T.atomic_add(_x2/_x4)`、`T.atomic_max/min`、`T.atomic_load/store`

### 9.5 Hopper 专属

- TMA：`T.create_tma_descriptor`、`T.tma_load`、`T.tma_store_arrive/wait`
- Warp-group MMA：`T.warpgroup_arrive`、`T.warpgroup_commit_batch`、`T.warpgroup_wait(num_mma)`、`T.wait_wgmma(id)`
- TMEM：`T.alloc_tmem`、`T.deallocate_tmem`
- 寄存器控制：`T.set_max_nreg`、`T.inc/dec_max_nreg`、`T.annotate_producer_reg_dealloc`、`T.annotate_consumer_reg_alloc`

---

## 10. 校验容差 & TFLOPS 计算

### 10.1 容差表

| 操作 | rtol | atol | 原因 |
|---|---|---|---|
| fp16 GEMM | 1e-2 | 1e-2 | tensor core 累加舍入 |
| fp16 GEMM + fusion | 1e-2 | 1e-2 | 同上 + epilogue |
| Elementwise（exp/log/sigmoid） | 5e-2 | 5e-2 | fp16 超越函数近似 |
| Reduction | 5e-2 | 5e-2 | 累加顺序与 PyTorch 不同 |
| 简单 elementwise（scale） | 1e-2 | 1e-2 | 仅基本舍入 |

### 10.2 TFLOPS

```
tflops = 2 * M * N * K / latency_ms * 1e-9
```

`2` = 一次 mul + 一次 add；`1e-9` 把 ms 化成 s 再换算 TFLOPS。

---

## 11. 升级路径（出问题去找谁）

```
写完 kernel
   │
   ├── 编译挂        →  看错误信息，对照 §8 坑表
   │
   ├── 编译过但数值错 →  debugging-tilelang-programs  skill
   │
   ├── 数值对但慢    →  profiling-tilelang-programs  skill
   │                  └→ optimizing-tilelang-programs skill
   │
   └── 需要 fwd+bwd  →  testing-fwd-bwd-kernels      skill
```

---

## 12. 一页纸速查（贴墙版）

```
@tilelang.jit(out_idx=[-1])
def my_kernel(M, N, K, BM, BN, BK, dtype=T.float16, accum=T.float32):

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype),
               B: T.Tensor((K, N), dtype),
               C: T.Tensor((M, N), dtype)):

        with T.Kernel(T.ceildiv(N, BN),
                      T.ceildiv(M, BM), threads=128) as (bx, by):

            A_s = T.alloc_shared((BM, BK), dtype)        # SRAM
            B_s = T.alloc_shared((BK, BN), dtype)        # SRAM
            C_f = T.alloc_fragment((BM, BN), accum)      # registers

            T.clear(C_f)                                  # ① 清零

            for ko in T.Pipelined(T.ceildiv(K, BK),
                                  num_stages=3):
                T.copy(A[by*BM, ko*BK], A_s)              # ② 搬 A
                T.copy(B[ko*BK, bx*BN], B_s)              # ③ 搬 B
                T.gemm(A_s, B_s, C_f)                     # ④ 累加

            T.copy(C_f, C[by*BM, bx*BN])                  # ⑤ 写回

    return kernel
```

记住 5 步：**clear → copy A → copy B → gemm → copy 写回**。改 epilogue/算式只改第 4 步前后。

---

> 源 skill 路径：`skills/tilelang/writing-tilelang-kernels/`  
> TileLang 版本：v0.1.9
