# NxCL

Nx backend and `defn` compiler for OpenCL GPUs. Targets OpenCL 1.2+ devices
including mobile GPUs (Adreno, Mali, PowerVR) and desktop GPUs via a Rust
NIF using the `opencl3` crate.

## Prerequisites

An OpenCL runtime must be installed:

```bash
# Ubuntu/Debian - PoCL (CPU OpenCL, good for development/testing)
sudo apt install pocl-opencl-icd ocl-icd-libopencl1 ocl-icd-opencl-dev

# Verify
clinfo --list
```

NVIDIA, AMD, and Intel GPU drivers typically include OpenCL support.
Rust toolchain is required for building the NIF.

## Installation

Add `nx_opencl` to your dependencies:

```elixir
def deps do
  [
    {:nx_opencl, "~> 0.1.0"}
  ]
end
```

## Usage

NxCL provides two ways to run Nx computations on the GPU: the **eager
backend** for ad-hoc tensor operations, and the **defn compiler** for
whole-graph execution of `defn` functions.

### Eager backend

Set `NxCL.Backend` as the default and use Nx as normal. Operations
with GPU kernels run on the device; everything else falls back to CPU
transparently.

```elixir
Nx.default_backend(NxCL.Backend)

a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
Nx.dot(a, b)
```

### Defn compiler

Use `NxCL.Compiler` with `Nx.Defn.jit/2` to trace a `defn` function
into a flat op sequence and execute the whole graph on the GPU in one
pass.

```elixir
defmodule MyModel do
  import Nx.Defn

  defn predict(x, w1, b1, w2, b2) do
    x
    |> Nx.dot(w1) |> Nx.add(b1) |> Nx.negate() |> Nx.negate()
    |> Nx.dot(w2) |> Nx.add(b2) |> Nx.sigmoid()
  end
end

jitted = Nx.Defn.jit(&MyModel.predict/5, compiler: NxCL.Compiler)
result = jitted.(input, w1, b1, w2, b2)
```

### Transfer tensors to/from GPU

```elixir
gpu_tensor = Nx.backend_transfer(cpu_tensor, NxCL.Backend)
cpu_tensor = Nx.backend_transfer(gpu_tensor, Nx.BinaryBackend)
```

### With Axon models

Axon models are pure Nx under the hood, so they work with the eager
backend by transferring params and inputs to the GPU:

```elixir
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(128, activation: :relu)
  |> Axon.dense(10, activation: :softmax)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 784}, :f32), %{})

# Transfer params to GPU (walk the ModelState struct)
gpu_params = %{params |
  data: Map.new(params.data, fn {layer, layer_params} ->
    {layer, Map.new(layer_params, fn {k, v} ->
      {k, Nx.backend_transfer(v, NxCL.Backend)}
    end)}
  end)
}

gpu_input = Nx.backend_transfer(input, NxCL.Backend)
output = predict_fn.(gpu_params, gpu_input)
```

### Query device info

```elixir
ctx = NxCL.Native.device_ctx_create(0)
NxCL.Native.device_info(ctx)
# [{"name", "NVIDIA GeForce RTX 3090 Ti"}, {"vendor", "NVIDIA"}, ...]
```

## GPU-Accelerated Operations

The following ops run directly on the GPU for f32 tensors:

| Category | Operations |
|---|---|
| Elementwise binary | `add`, `subtract`, `multiply`, `divide` |
| Elementwise unary | `negate`, `exp`, `log`, `tanh`, `sigmoid`, `abs` |
| Linear algebra | `dot` (2D matrix multiply, tiled GEMM) |
| Reductions | `sum`, `reduce_max`, `reduce_min` (full reduction) |
| Shape | `reshape`, `squeeze`, `bitcast` (zero-copy) |

All other Nx operations fall back to the CPU backend automatically
when using `NxCL.Backend`. The compiler (`NxCL.Compiler`) raises on
unsupported ops.

## Running Tests

```bash
# All tests (requires OpenCL runtime)
mix test

# Skip OpenCL tests
mix test --exclude opencl
```
