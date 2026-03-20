defmodule NxOpencl do
  @moduledoc """
  Nx backend for OpenCL GPUs.

  NxCL targets OpenCL 1.2+ devices for running inference workloads.
  It provides a `Nx.Backend` implementation that dispatches tensor
  operations to OpenCL kernels via a Rust NIF.

  ## Quick start

      Nx.default_backend(NxCL.Backend)

      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
      Nx.dot(a, b)

  ## Transferring tensors

      gpu_tensor = Nx.backend_transfer(cpu_tensor, NxCL.Backend)
      cpu_tensor = Nx.backend_transfer(gpu_tensor, Nx.BinaryBackend)

  See `NxCL.Backend` for the full list of GPU-accelerated operations.
  """
end
