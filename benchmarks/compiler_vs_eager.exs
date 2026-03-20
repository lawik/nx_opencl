# Compiler vs eager backend benchmarks.
#
# Compares NxCL.Compiler (whole-graph) against NxCL.Backend (per-op)
# and the default Nx.Defn.Evaluator. This shows the overhead of per-op
# dispatch vs compiled execution, and where the compiler wins even
# without fusion.
#
# Usage: elixir benchmarks/compiler_vs_eager.exs

Code.require_file("bench_helper.exs", __DIR__)

has_opencl = BenchHelper.has_opencl?()
has_exla = BenchHelper.has_exla?()

# ── Single op: Nx.exp ────────────────────────────────

IO.puts("=== Single op: exp (n=100_000) ===\n")

single_op_input = BenchHelper.rand({100_000})

single_op = %{
  "Evaluator (CPU)" => fn ->
    Nx.Defn.jit(fn x -> Nx.exp(x) end).(single_op_input)
  end
}

single_op =
  if has_opencl do
    gpu_input = Nx.backend_transfer(single_op_input, NxCL.Backend)

    Map.merge(single_op, %{
      "NxCL.Backend (eager)" => fn ->
        Nx.exp(gpu_input)
      end,
      "NxCL.Compiler (jit)" => fn ->
        Nx.Defn.jit(fn x -> Nx.exp(x) end, compiler: NxCL.Compiler).(single_op_input)
      end
    })
  else
    single_op
  end

single_op =
  if has_exla do
    Map.put(single_op, "EXLA (jit)", fn ->
      Nx.Defn.jit(fn x -> Nx.exp(x) end, compiler: EXLA).(single_op_input)
    end)
  else
    single_op
  end

Benchee.run(single_op, warmup: 2, time: 5, memory_time: 0, print: [configuration: false])

# ── Elementwise chain (3 ops) ────────────────────────

IO.puts("\n=== Elementwise chain: exp -> negate -> sigmoid (n=100_000) ===\n")

chain_input = BenchHelper.rand({100_000})
chain_fn = fn x -> x |> Nx.exp() |> Nx.negate() |> Nx.sigmoid() end

chain = %{
  "Evaluator (CPU)" => fn -> Nx.Defn.jit(chain_fn).(chain_input) end
}

chain =
  if has_opencl do
    gpu_chain = Nx.backend_transfer(chain_input, NxCL.Backend)

    Map.merge(chain, %{
      "NxCL.Backend (eager, 3 dispatches)" => fn ->
        gpu_chain |> Nx.exp() |> Nx.negate() |> Nx.sigmoid()
      end,
      "NxCL.Compiler (jit, 3 dispatches)" => fn ->
        Nx.Defn.jit(chain_fn, compiler: NxCL.Compiler).(chain_input)
      end
    })
  else
    chain
  end

chain =
  if has_exla do
    Map.put(chain, "EXLA (jit, fused)", fn ->
      Nx.Defn.jit(chain_fn, compiler: EXLA).(chain_input)
    end)
  else
    chain
  end

Benchee.run(chain, warmup: 2, time: 5, memory_time: 0, print: [configuration: false])

# ── Dense layer (dot + add + sigmoid) ────────────────

IO.puts("\n=== Dense layer: dot + add + sigmoid (32x64 -> 128) ===\n")

dense_x = BenchHelper.rand({32, 64})
dense_w = BenchHelper.rand({64, 128})
dense_b = BenchHelper.rand({1, 128})

dense_fn = fn x, w, b ->
  x |> Nx.dot(w) |> Nx.add(b) |> Nx.sigmoid()
end

dense = %{
  "Evaluator (CPU)" => fn ->
    Nx.Defn.jit(dense_fn).(dense_x, dense_w, dense_b)
  end
}

dense =
  if has_opencl do
    gx = Nx.backend_transfer(dense_x, NxCL.Backend)
    gw = Nx.backend_transfer(dense_w, NxCL.Backend)
    gb = Nx.backend_transfer(dense_b, NxCL.Backend)

    Map.merge(dense, %{
      "NxCL.Backend (eager)" => fn ->
        dense_fn.(gx, gw, gb)
      end,
      "NxCL.Compiler (jit)" => fn ->
        Nx.Defn.jit(dense_fn, compiler: NxCL.Compiler).(dense_x, dense_w, dense_b)
      end
    })
  else
    dense
  end

dense =
  if has_exla do
    Map.put(dense, "EXLA (jit)", fn ->
      Nx.Defn.jit(dense_fn, compiler: EXLA).(dense_x, dense_w, dense_b)
    end)
  else
    dense
  end

Benchee.run(dense, warmup: 2, time: 5, memory_time: 0, print: [configuration: false])

# ── 2-layer MLP ──────────────────────────────────────

IO.puts("\n=== 2-layer MLP: 784 -> 256 -> 10 (batch=16) ===\n")

mlp_x = BenchHelper.rand({16, 784})
mlp_w1 = BenchHelper.rand({784, 256})
mlp_b1 = BenchHelper.rand({1, 256})
mlp_w2 = BenchHelper.rand({256, 10})
mlp_b2 = BenchHelper.rand({1, 10})

mlp_fn = fn x, w1, b1, w2, b2 ->
  x
  |> Nx.dot(w1) |> Nx.add(b1) |> Nx.sigmoid()
  |> Nx.dot(w2) |> Nx.add(b2) |> Nx.sigmoid()
end

mlp = %{
  "Evaluator (CPU)" => fn ->
    Nx.Defn.jit(mlp_fn).(mlp_x, mlp_w1, mlp_b1, mlp_w2, mlp_b2)
  end
}

mlp =
  if has_opencl do
    Map.merge(mlp, %{
      "NxCL.Backend (eager, incl transfer)" => fn ->
        gx = Nx.backend_transfer(mlp_x, NxCL.Backend)
        gw1 = Nx.backend_transfer(mlp_w1, NxCL.Backend)
        gb1 = Nx.backend_transfer(mlp_b1, NxCL.Backend)
        gw2 = Nx.backend_transfer(mlp_w2, NxCL.Backend)
        gb2 = Nx.backend_transfer(mlp_b2, NxCL.Backend)
        mlp_fn.(gx, gw1, gb1, gw2, gb2)
      end,
      "NxCL.Compiler (jit)" => fn ->
        Nx.Defn.jit(mlp_fn, compiler: NxCL.Compiler).(mlp_x, mlp_w1, mlp_b1, mlp_w2, mlp_b2)
      end
    })
  else
    mlp
  end

mlp =
  if has_exla do
    Map.put(mlp, "EXLA (jit)", fn ->
      Nx.Defn.jit(mlp_fn, compiler: EXLA).(mlp_x, mlp_w1, mlp_b1, mlp_w2, mlp_b2)
    end)
  else
    mlp
  end

Benchee.run(mlp, warmup: 2, time: 5, memory_time: 0, print: [configuration: false])
