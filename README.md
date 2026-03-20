# NxCL

Nx backend for OpenCL GPUs. Targets OpenCL 1.2+ devices including mobile
GPUs (Adreno, Mali, PowerVR) and desktop GPUs via a Rust NIF using the
`opencl3` crate.

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

### As default backend

```elixir
Nx.default_backend(NxCL.Backend)

a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
Nx.dot(a, b)
```

### Transfer tensors to/from GPU

```elixir
# Move a tensor to the GPU
gpu_tensor = Nx.backend_transfer(cpu_tensor, NxCL.Backend)

# Move it back to CPU
cpu_tensor = Nx.backend_transfer(gpu_tensor, Nx.BinaryBackend)
```

### With Axon models

```elixir
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(128, activation: :relu)
  |> Axon.dense(10, activation: :softmax)

{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 784}, :f32), %{})

# Transfer params to GPU
gpu_params = transfer_to_gpu(params)
gpu_input = Nx.backend_transfer(input, NxCL.Backend)

# Run inference on GPU
output = predict_fn.(gpu_params, gpu_input)
```

### Query device info

```elixir
ctx = NxCL.Native.device_ctx_create(0)
NxCL.Native.device_info(ctx)
# [{"name", "NVIDIA GeForce RTX 3090 Ti"}, {"vendor", "NVIDIA"}, ...]
```

## GPU-Accelerated Operations

The following ops run directly on the GPU (f32 tensors, matching shapes):

| Category | Operations |
|---|---|
| Elementwise binary | `add`, `subtract`, `multiply`, `divide` |
| Elementwise unary | `negate`, `exp`, `log`, `tanh`, `sigmoid`, `abs` |
| Linear algebra | `dot` (2D matrix multiply, tiled GEMM) |
| Reductions | `sum`, `reduce_max`, `reduce_min` (full reduction) |
| Shape | `reshape`, `squeeze`, `bitcast` (zero-copy) |

All other Nx operations fall back to the CPU backend automatically.

## Running Tests

```bash
# All tests (requires OpenCL runtime)
mix test

# Skip OpenCL tests
mix test --exclude opencl
```
