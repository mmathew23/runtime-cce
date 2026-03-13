# mlx-cce-runtime

`mlx-cce-runtime` provides a runtime implementation of chunked cross-entropy for
upstream MLX. It installs a compatibility shim that exposes
`mx.fast.cce_loss(...)` without requiring a forked MLX build.

It does not replace `mlx`. The supported runtime code paths run on top of the
`mlx` already installed in your Python environment.

## What it does

- Computes cross-entropy directly from hidden states and LM-head weights
- Avoids full-logit materialization by processing the vocabulary in chunks
- Supports quantized LM-head weights through `mx.quantized_matmul(...)`
- Patches `mx.fast.cce_loss` for integrations that already expect the fork API

## Public API

- `install_mlx_fast_cce_loss(*, override: bool = False)`
- `make_chunked_cross_entropy_loss(...)`
- `make_runtime_cce_loss_fused_finalize(...)`
- `SUPPORTED_RUNTIME_VARIANTS`
- `RUNTIME_VARIANT_INFO`

## Install

Install into an environment that already has upstream `mlx`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install mlx
pip install -e .
```

This keeps the environment on stock `mlx`. `mlx-cce-runtime` is a separate
package layered on top.

## Tested integration stack

The package itself only depends on stock `mlx`, but the end-to-end training
benchmarks in this repository were run against Manan's MLX branches:

- `Manan17/unsloth` on `feature/mlx-apple-silicon`
- `Manan17/unsloth-zoo` on `feature/mlx-cce-runtime`

That `unsloth-zoo` branch is the integration layer that knows how to call
`mlx-cce-runtime` directly when it is installed, while still falling back to
`mx.fast.cce_loss(...)` when needed.

## Basic usage

Use the runtime loss directly:

```python
from mlx_cce_runtime import make_chunked_cross_entropy_loss

loss_fn, _ = make_chunked_cross_entropy_loss(runtime_variant="balanced")
```

Or install the compatibility shim for integrations that expect
`mx.fast.cce_loss(...)`:

```python
from mlx_cce_runtime import install_mlx_fast_cce_loss

install_mlx_fast_cce_loss()
```

## Integration styles

There are three practical ways to use the package:

1. Direct runtime API
   Use `make_chunked_cross_entropy_loss(...)` in your own MLX training code.
2. Compatibility shim
   Call `install_mlx_fast_cce_loss()` to patch `mx.fast.cce_loss(...)` at
   runtime for code that already expects the fork-style API.
3. Unsloth / Unsloth Zoo integration
   Use Manan's MLX branches above. The `feature/mlx-cce-runtime` branch in
   `unsloth-zoo` can call `mlx_cce_runtime` directly without requiring a forked
   MLX build.

## Supported runtime variants

These are the supported non-native runtime code paths for general testing:

- `balanced`
  - Default path. Stable chunked forward/backward composition with full-precision
    backward chunks.
- `compact_backward`
  - Stores bf16 backward chunks more compactly to reduce the isolated loss-step peak.
- `simd_reduction`
  - Builds on `compact_backward` and swaps in SIMD-group forward reduction kernels.

Legacy aliases are still accepted:

- `clean` -> `balanced`
- `iter` -> `compact_backward`
- `simd` -> `simd_reduction`
- `fused_finalize` -> `balanced`

## Native extension

The repository also contains an optional out-of-tree native extension in
`native/`. This is experimental and is not required for the supported
`balanced`, `compact_backward`, or `simd_reduction` paths.

If you build it, it links against the `mlx` already installed in the active
environment. It does not replace `mlx`.

```bash
source .venv/bin/activate
pip install nanobind cmake ninja
cmake -S native -B build-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/src/mlx_cce_runtime
cmake --build build-native
```

That build places the optional `_ext` module into `src/mlx_cce_runtime`.

### Experimental native variants

If the native extension is built, the package also exposes experimental runtime
variants intended for investigation rather than normal use:

- `native`
- `native_bridge`
- `native_custom_vjp`

These are not currently part of the supported runtime surface. They were used
to explore whether an out-of-tree native primitive can close the memory gap with
the forked MLX CCE path. At the time of writing, they are not reliable enough
to recommend for real training, but they are still useful if you want to
investigate MLX primitive/autograd behavior further.

To experiment with them:

```bash
source .venv/bin/activate
pip install -e .
pip install nanobind cmake ninja
cmake -S native -B build-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/src/mlx_cce_runtime
cmake --build build-native
```

Then select a variant explicitly, for example:

```bash
MLX_CCE_RUNTIME_VARIANT=native
```

or in Python:

```python
from mlx_cce_runtime import make_chunked_cross_entropy_loss

loss_fn, _ = make_chunked_cross_entropy_loss(runtime_variant="native")
```

Treat the native variants as research code paths. They are kept in-tree so
others can inspect, profile, and continue the investigation.

## Current limitations versus forked MLX CCE

- The runtime path is built from MLX custom functions and Metal kernels, not a
  dedicated backend primitive.
- Dense backward still relies on chunked recomputation and is expected to use
  more memory than the forked backend kernel.
- Benchmarking must be done from a clean cwd such as `/tmp` to avoid namespace
  package contamination from sibling `mlx` checkouts.

## Benchmarking

This repository contains two kinds of benchmark scripts:

- End-to-end Unsloth training benchmark
  - `benchmarks/unsloth_8k_bf16_bench.py`
- Isolated loss / memory microbenchmarks
  - `benchmarks/compare_loss_memory.py`
  - `benchmarks/profile_unsloth_memory_breakdown.py`
  - `benchmarks/profile_unsloth_train_step_components.py`
  - `benchmarks/profile_unsloth_layer_groups.py`
  - `benchmarks/profile_unsloth_late_blocks.py`
- Native extension repro / debugging scripts
  - `benchmarks/repro_native_vjp_retention.py`
  - `benchmarks/repro_pair_square_probe.py`

The end-to-end benchmark depends on the Unsloth MLX branches described above.
The loss and profiling scripts are useful even without that trainer stack.

Use the included benchmark from a neutral working directory:

```bash
REPO_ROOT=/path/to/mlx-cce-runtime
VENV_PYTHON=/path/to/venv/bin/python
cd /tmp
"$VENV_PYTHON" "$REPO_ROOT/benchmarks/unsloth_8k_bf16_bench.py"
```

To select a runtime variant:

```bash
REPO_ROOT=/path/to/mlx-cce-runtime
VENV_PYTHON=/path/to/venv/bin/python
cd /tmp
MLX_CCE_RUNTIME_VARIANT=balanced \
"$VENV_PYTHON" "$REPO_ROOT/benchmarks/unsloth_8k_bf16_bench.py"
```

To run the isolated loss benchmark instead:

```bash
REPO_ROOT=/path/to/mlx-cce-runtime
VENV_PYTHON=/path/to/venv/bin/python
cd /tmp
"$VENV_PYTHON" "$REPO_ROOT/benchmarks/compare_loss_memory.py" \
  --impl runtime \
  --runtime-variant balanced \
  --grads full
```

The benchmark prints JSON containing:

- import provenance (`mlx.__path__`, `mlx.core.__file__`)
- baseline step time and peak memory
- CCE step time and peak memory
- final losses for both runs
