# Matrix multiply benchmarks.
#
# Sweeps square matrix sizes from small (where launch overhead dominates)
# to large (where GPU compute wins). This is the most important benchmark
# for real inference workloads.
#
# Usage: elixir benchmarks/matmul.exs

Code.require_file("bench_helper.exs", __DIR__)

sizes = [32, 64, 128, 256, 512]
backends = BenchHelper.eager_backends()

IO.puts("=== Matrix Multiply (square) ===\n")

matmul_benchmarks =
  for {name, backend_fn} <- backends, n <- sizes, into: %{} do
    backend = backend_fn.()
    a = BenchHelper.rand({n, n}, backend)
    b = BenchHelper.rand({n, n}, backend)
    flops = 2 * n * n * n

    {"#{name} #{n}x#{n} (#{div(flops, 1_000_000)}M FLOPs)",
     fn -> Nx.dot(a, b) end}
  end

Benchee.run(matmul_benchmarks,
  warmup: 2,
  time: 5,
  memory_time: 0,
  print: [configuration: false]
)

IO.puts("\n=== Matrix Multiply (non-square) ===\n")

nonsquare = [{32, 128, 64}, {128, 256, 64}, {256, 512, 128}]

nonsquare_benchmarks =
  for {name, backend_fn} <- backends, {m, k, n} <- nonsquare, into: %{} do
    backend = backend_fn.()
    a = BenchHelper.rand({m, k}, backend)
    b = BenchHelper.rand({k, n}, backend)

    {"#{name} #{m}x#{k} * #{k}x#{n}",
     fn -> Nx.dot(a, b) end}
  end

Benchee.run(nonsquare_benchmarks,
  warmup: 2,
  time: 5,
  memory_time: 0,
  print: [configuration: false]
)
