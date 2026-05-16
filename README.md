# TileLang Development Skills

Agent skills for writing, debugging, profiling, and optimizing [TileLang](https://github.com/tile-ai/tilelang) GPU kernels.

Contents are created by Claude opus 4.6 in claude code with the `skill-creator` plugin. Based on TileLang `v0.1.9` docs and examples, validated on RTX PRO 6000 Blackwell GPUs (sm_120) with CUDA 13.1 and pytorch 2.11.

The `.vscode` directory is intentionally tracked as a reference to configure highlighting for humans.

## Skills

| Skill | Description | Key Topics |
|-------|-------------|------------|
| [writing-tilelang-kernels](skills/tilelang/writing-tilelang-kernels/SKILL.md) | Write TileLang GPU kernels from scratch or by adapting patterns | Kernel anatomy, templates (GEMM, elementwise, reduction), memory scopes, T.copy/T.gemm, dynamic shapes |
| [debugging-tilelang-programs](skills/tilelang/debugging-tilelang-programs/SKILL.md) | Diagnose and fix errors in TileLang programs | Failure taxonomy, T.print, AutoDD, compute-sanitizer, numerical drift, race detection |
| [profiling-tilelang-programs](skills/tilelang/profiling-tilelang-programs/SKILL.md) | Benchmark and profile TileLang kernels | do_bench backends, TFLOPS/bandwidth, ncu bottleneck diagnosis (pipe utilization, warp stalls), roofline |
| [optimizing-tilelang-programs](skills/tilelang/optimizing-tilelang-programs/SKILL.md) | Optimize TileLang kernels for performance | Tile sizes, pipeline stages, threads, AutoTuner, epilogue fusion, swizzle, ncu-guided tuning |
| [testing-fwd-bwd-kernels](skills/tilelang/testing-fwd-bwd-kernels/SKILL.md) | Test kernels with forward and backward passes | torch.autograd.Function, compare_backward (not gradcheck), mixed-precision, atomicAdd, attention bwd |

## Workflow

The skills form a natural progression:

```
writing → debugging → profiling → optimizing
                                      ↓
                              testing-fwd-bwd (for differentiable ops)
```

1. **Write** a kernel using templates from the writing skill
2. **Debug** if it fails to compile or produces wrong results
3. **Profile** to measure baseline performance
4. **Optimize** to improve performance
5. **Test fwd+bwd** if the kernel needs gradients

