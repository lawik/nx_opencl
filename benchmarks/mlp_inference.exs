# MLP inference benchmarks.
#
# Tests the pattern that matters most for on-device inference:
# dot + bias add + activation, composed into multi-layer networks.
#
# Usage: elixir benchmarks/mlp_inference.exs

Code.require_file("bench_helper.exs", __DIR__)

backends = BenchHelper.eager_backends()

defmodule MLPBench do
  def dense_relu(x, w, b) do
    x |> Nx.dot(w) |> Nx.add(b) |> Nx.max(Nx.tensor(0.0))
  end

  def dense(x, w, b) do
    x |> Nx.dot(w) |> Nx.add(b)
  end

  def mlp_2layer(x, w1, b1, w2, b2) do
    x |> dense_relu(w1, b1) |> dense(w2, b2)
  end

  def mlp_3layer(x, w1, b1, w2, b2, w3, b3) do
    x |> dense_relu(w1, b1) |> dense_relu(w2, b2) |> dense(w3, b3)
  end
end

IO.puts("=== Single dense layer (batch=1, 784 -> 256) ===\n")

single_layer =
  for {name, backend_fn} <- backends, into: %{} do
    backend = backend_fn.()
    x = BenchHelper.rand({1, 784}, backend)
    w = BenchHelper.rand({784, 256}, backend)
    b = BenchHelper.rand({1, 256}, backend)
    {name, fn -> MLPBench.dense_relu(x, w, b) end}
  end

Benchee.run(single_layer,
  warmup: 2,
  time: 5,
  memory_time: 0,
  print: [configuration: false]
)

IO.puts("\n=== 2-layer MLP (batch=1, 784 -> 256 -> 10) ===\n")

two_layer =
  for {name, backend_fn} <- backends, into: %{} do
    backend = backend_fn.()
    x = BenchHelper.rand({1, 784}, backend)
    w1 = BenchHelper.rand({784, 256}, backend)
    b1 = BenchHelper.rand({1, 256}, backend)
    w2 = BenchHelper.rand({256, 10}, backend)
    b2 = BenchHelper.rand({1, 10}, backend)
    {name, fn -> MLPBench.mlp_2layer(x, w1, b1, w2, b2) end}
  end

Benchee.run(two_layer,
  warmup: 2,
  time: 5,
  memory_time: 0,
  print: [configuration: false]
)

IO.puts("\n=== 3-layer MLP (batch=16, 784 -> 256 -> 128 -> 10) ===\n")

three_layer =
  for {name, backend_fn} <- backends, into: %{} do
    backend = backend_fn.()
    x = BenchHelper.rand({16, 784}, backend)
    w1 = BenchHelper.rand({784, 256}, backend)
    b1 = BenchHelper.rand({1, 256}, backend)
    w2 = BenchHelper.rand({256, 128}, backend)
    b2 = BenchHelper.rand({1, 128}, backend)
    w3 = BenchHelper.rand({128, 10}, backend)
    b3 = BenchHelper.rand({1, 10}, backend)
    {name, fn -> MLPBench.mlp_3layer(x, w1, b1, w2, b2, w3, b3) end}
  end

Benchee.run(three_layer,
  warmup: 2,
  time: 5,
  memory_time: 0,
  print: [configuration: false]
)

IO.puts("\n=== Batch size scaling (2-layer MLP, 784 -> 256 -> 10) ===\n")

batch_sizes = [1, 4, 16, 64]

batch_benchmarks =
  for {name, backend_fn} <- backends, bs <- batch_sizes, into: %{} do
    backend = backend_fn.()
    x = BenchHelper.rand({bs, 784}, backend)
    w1 = BenchHelper.rand({784, 256}, backend)
    b1 = BenchHelper.rand({1, 256}, backend)
    w2 = BenchHelper.rand({256, 10}, backend)
    b2 = BenchHelper.rand({1, 10}, backend)
    {"#{name} batch=#{bs}", fn -> MLPBench.mlp_2layer(x, w1, b1, w2, b2) end}
  end

Benchee.run(batch_benchmarks,
  warmup: 2,
  time: 5,
  memory_time: 0,
  print: [configuration: false]
)
