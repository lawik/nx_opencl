# Elementwise operation benchmarks.
#
# Compares unary and binary elementwise ops across backends at
# various tensor sizes to show where GPU dispatch overhead pays off.
#
# Usage: elixir benchmarks/elementwise.exs

Code.require_file("bench_helper.exs", __DIR__)

sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
backends = BenchHelper.eager_backends()

IO.puts("=== Elementwise: Nx.exp ===\n")

exp_benchmarks =
  for {name, backend_fn} <- backends, n <- sizes, into: %{} do
    backend = backend_fn.()
    input = BenchHelper.rand({n}, backend)
    {"#{name} (n=#{n})", fn -> Nx.exp(input) end}
  end

Benchee.run(exp_benchmarks,
  warmup: 1,
  time: 3,
  memory_time: 0,
  print: [configuration: false]
)

IO.puts("\n=== Elementwise: Nx.add ===\n")

add_benchmarks =
  for {name, backend_fn} <- backends, n <- [1_000, 100_000, 1_000_000], into: %{} do
    backend = backend_fn.()
    a = BenchHelper.rand({n}, backend)
    b = BenchHelper.rand({n}, backend)
    {"#{name} (n=#{n})", fn -> Nx.add(a, b) end}
  end

Benchee.run(add_benchmarks,
  warmup: 1,
  time: 3,
  memory_time: 0,
  print: [configuration: false]
)

IO.puts("\n=== Elementwise: Nx.sigmoid ===\n")

sigmoid_benchmarks =
  for {name, backend_fn} <- backends, n <- [1_000, 100_000, 1_000_000], into: %{} do
    backend = backend_fn.()
    input = BenchHelper.rand({n}, backend)
    {"#{name} (n=#{n})", fn -> Nx.sigmoid(input) end}
  end

Benchee.run(sigmoid_benchmarks,
  warmup: 1,
  time: 3,
  memory_time: 0,
  print: [configuration: false]
)
