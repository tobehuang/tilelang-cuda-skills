# GPU Performance Metrics Reference

## Key Metrics for TileLang Kernels

### Throughput Metrics

| Metric | Formula | Unit | What it tells you |
|--------|---------|------|-------------------|
| TFLOPS | `2 * M * N * K / latency_ms * 1e-9` | TFLOPS | Compute throughput for GEMM |
| Effective Bandwidth | `bytes_moved / latency_ms * 1e-6` | GB/s | Memory throughput |
| Arithmetic Intensity | `FLOPs / bytes_moved` | FLOPs/byte | Compute vs memory bound |

### Occupancy Metrics (from ncu)

| Metric | Target | Impact |
|--------|--------|--------|
| SM Occupancy | >50% | More warps to hide latency |
| Theoretical Occupancy | Check | Limited by registers, shared mem, or block size |
| Achieved Occupancy | Close to theoretical | If much lower, check for load imbalance |

### Memory Metrics (from ncu)

| Metric | What it means |
|--------|---------------|
| Global Load Throughput | Bytes/sec read from DRAM |
| Global Store Throughput | Bytes/sec written to DRAM |
| Shared Memory Throughput | Bytes/sec through shared memory |
| L2 Cache Hit Rate | Fraction of requests served by L2 |
| Shared Memory Bank Conflicts | Wasted cycles from bank conflicts |

### Compute Metrics (from ncu)

| Metric | What it means |
|--------|---------------|
| SM Active Cycles | Fraction of cycles with active warps |
| Tensor Core Utilization | Fraction of eligible cycles using tensor cores |
| Warp Stall Reasons | Why warps are waiting (memory, sync, etc.) |

## How to Calculate Bytes Moved

| Operation | Read bytes | Write bytes | Total |
|-----------|-----------|-------------|-------|
| Elementwise (C = A op B) | 2 * M * N * elem_size | M * N * elem_size | 3 * M * N * elem_size |
| Reduction (row sum) | M * N * elem_size | M * elem_size | M * (N + 1) * elem_size |
| GEMM (C = A @ B) | (M*K + K*N) * elem_size | M * N * elem_size | (M*K + K*N + M*N) * elem_size |

For fp16: elem_size = 2 bytes. For fp32: elem_size = 4 bytes.

## Roofline Model

A kernel is either **compute-bound** or **memory-bound**:

```
If arithmetic_intensity > peak_flops / peak_bandwidth:
    kernel is compute-bound → optimize compute (tensor cores, tiling)
else:
    kernel is memory-bound → optimize memory (coalescing, vectorization)
```

### Typical GPU Peak Numbers

These are approximate -- check your specific GPU with `nvidia-smi -q` or ncu:

| GPU | Peak FP16 TFLOPS | Peak BW (GB/s) | Ridge Point (FLOPs/byte) |
|-----|-------------------|----------------|--------------------------|
| A100 | 312 | 2039 | 153 |
| H100 | 990 | 3350 | 296 |
| RTX 4090 | 330 | 1008 | 327 |
| RTX PRO 6000 Blackwell | ~varies | ~varies | Check with ncu |

### GEMM Arithmetic Intensity

For GEMM C = A @ B with M, N, K:
```
FLOPs = 2 * M * N * K
Bytes = (M*K + K*N + M*N) * elem_size
AI = 2*M*N*K / ((M*K + K*N + M*N) * elem_size)
```

For large square GEMM (M=N=K=4096, fp16):
```
AI = 2 * 4096^3 / (3 * 4096^2 * 2) ≈ 1365 FLOPs/byte → compute-bound
```

For tall-skinny GEMM (M=4096, N=1, K=4096, fp16):
```
AI = 2 * 4096 * 1 * 4096 / ((4096*4096 + 4096 + 4096) * 2) ≈ 1 FLOP/byte → memory-bound
```

## Interpreting do_bench Results

### Expected Latency Ranges (1024x1024 GEMM fp16)

Validated on RTX PRO 6000 Blackwell:
- `do_bench(backend="event")`: ~0.020 ms (105 TFLOPS)
- `do_bench(backend="cupti")`: ~0.020 ms (110 TFLOPS)
- `do_bench(backend="cudagraph")`: ~0.017 ms (125 TFLOPS)

The `cudagraph` backend eliminates kernel launch overhead, giving the lowest (most optimistic) latency. Use `"event"` for realistic numbers.

### When Numbers Don't Make Sense

- **TFLOPS exceeds GPU peak**: Check your FLOP formula. For GEMM it's `2*M*N*K`, not `M*N*K`.
- **Latency is 0.0**: Problem too small, within measurement noise. Increase problem size.
- **First run is much slower**: JIT compilation happens on first call. Ensure warmup is sufficient.
- **High variance between runs**: Check that no other GPU workloads are running. Pin to a specific GPU with `CUDA_VISIBLE_DEVICES=0`.

## ncu Command Cheat Sheet

```bash
# Quick summary
ncu --target-processes all .venv/bin/python script.py

# Full analysis, save report
ncu --set full -o report .venv/bin/python script.py

# Specific metrics only
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second \
    .venv/bin/python script.py

# Skip warmup kernels (profile only the Nth kernel)
ncu --kernel-id ::N: .venv/bin/python script.py

# Profile specific kernel by name regex
ncu --kernel-name "my_kernel" .venv/bin/python script.py
```

## nsys Command Cheat Sheet

```bash
# Basic timeline
nsys profile -o timeline .venv/bin/python script.py

# With CUDA API trace
nsys profile --trace=cuda,nvtx -o timeline .venv/bin/python script.py

# Report summary
nsys stats timeline.nsys-rep
```
