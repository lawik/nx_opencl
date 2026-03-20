# Reduction benchmarks.
#
# Tests sum, max, and min reductions at various sizes. Reductions are
# memory-bandwidth-bound and have different GPU characteristics than
# compute-bound ops like matmul.
#
# Usage: elixir benchmarks/reduction.exs

Code.require_file("bench_helper.exs", __DIR__)

sizes = [1_000, 10_000, 100_000, 1_000_000]
backends = BenchHelper.eager_backends()

IO.puts("=== Nx.sum (full reduction) ===\n")

sum_benchmarks =
  for {name, backend_fn} <- backends, n <- sizes, into: %{} do
    backend = backend_fn.()
    input = BenchHelper.rand({n}, backend)
    {"#{name} (n=#{n})", fn -> Nx.sum(input) end}
  end

Benchee.run(sum_benchmarks,
  warmup: 1,
  time: 3,
  memory_time: 0,
  print: [configuration: false]
)

IO.puts("\n=== Nx.reduce_max (full reduction) ===\n")

max_benchmarks =
  for {name, backend_fn} <- backends, n <- [1_000, 100_000, 1_000_000], into: %{} do
    backend = backend_fn.()
    input = BenchHelper.rand({n}, backend)
    {"#{name} (n=#{n})", fn -> Nx.reduce_max(input) end}
  end

Benchee.run(max_benchmarks,
  warmup: 1,
  time: 3,
  memory_time: 0,
  print: [configuration: false]
)

IO.puts("\n=== 2D sum (reduce all axes) ===\n")

shapes_2d = [{100, 100}, {256, 256}, {1000, 1000}]

sum2d_benchmarks =
  for {name, backend_fn} <- backends, shape <- shapes_2d, into: %{} do
    backend = backend_fn.()
    input = BenchHelper.rand(shape, backend)
    {m, n} = shape
    {"#{name} (#{m}x#{n})", fn -> Nx.sum(input) end}
  end

Benchee.run(sum2d_benchmarks,
  warmup: 1,
  time: 3,
  memory_time: 0,
  print: [configuration: false]
)
