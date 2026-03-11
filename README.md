# mlx-cce-runtime

`mlx-cce-runtime` provides a runtime implementation of chunked cross-entropy for
upstream MLX. It installs a compatibility shim that exposes
`mx.fast.cce_loss(...)` without requiring a forked MLX build.

## What it does

- Computes cross-entropy directly from hidden states and LM-head weights
- Avoids full-logit materialization by processing the vocabulary in chunks
- Supports quantized LM-head weights through `mx.quantized_matmul(...)`
- Patches `mx.fast.cce_loss` for integrations that already expect the fork API

## Public API

- `install_mlx_fast_cce_loss(*, override: bool = False)`
- `make_chunked_cross_entropy_loss(...)`
- `make_runtime_cce_loss_fused_finalize(...)`

## Current limitations versus forked MLX CCE

- The runtime path is built from MLX custom functions and Metal kernels, not a
  dedicated backend primitive.
- Dense backward still relies on chunked recomputation and is expected to use
  more memory than the forked backend kernel.
- Benchmarking must be done from a clean cwd such as `/tmp` to avoid namespace
  package contamination from sibling `mlx` checkouts.

## Benchmarking

Use the included benchmark from a neutral working directory:

```bash
cd /tmp
/Users/mathew/repos/mlx_parent/worktrees/runtime_env/.venv/bin/python \
  /Users/mathew/repos/mlx_parent/mlx-cce-runtime/benchmarks/unsloth_8k_bf16_bench.py
```

The benchmark prints JSON containing:

- import provenance (`mlx.__path__`, `mlx.core.__file__`)
- baseline step time and peak memory
- CCE step time and peak memory
- final losses for both runs
