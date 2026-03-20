# NxCL Benchmarks

Standalone benchmark scripts comparing NxCL against other Nx backends
and compilers. Each script uses `Mix.install` so it can be run directly:

```bash
# Run individual benchmarks
elixir benchmarks/elementwise.exs
elixir benchmarks/matmul.exs
elixir benchmarks/mlp_inference.exs
elixir benchmarks/reduction.exs
elixir benchmarks/compiler_vs_eager.exs
```

## What's compared

| Script | What it measures |
|---|---|
| `elementwise.exs` | Unary/binary elementwise ops at various sizes |
| `matmul.exs` | Matrix multiply sweep (32x32 to 512x512) |
| `mlp_inference.exs` | Dense layer and full MLP inference |
| `reduction.exs` | Sum reduction at various sizes |
| `compiler_vs_eager.exs` | NxCL.Compiler vs NxCL.Backend vs Evaluator |

## Backends tested

- **BinaryBackend** — Nx's pure-Elixir CPU backend (baseline)
- **NxCL.Backend** — NxCL eager backend (OpenCL GPU)
- **NxCL.Compiler** — NxCL defn compiler (OpenCL GPU)
- **Evaluator** — Nx's default defn evaluator (CPU)
- **EXLA** — Google XLA compiler (CPU/GPU, if installed)
- **NxEigen** — Eigen-based CPU backend (if installed)

EXLA and NxEigen are optional. If they fail to install the benchmarks
still run with the available backends.
