# Common setup for all benchmarks.
# Installs dependencies and detects available backends.

Mix.install(
  [
    {:nx, "~> 0.9"},
    {:nx_opencl, path: Path.expand("..", __DIR__)},
    {:benchee, "~> 1.0"},
    {:exla, "~> 0.11", optional: true},
    {:nx_eigen, "~> 0.1", optional: true}
  ],
  force: true
)

defmodule BenchHelper do
  @moduledoc false

  def has_exla? do
    Code.ensure_loaded?(EXLA.Backend) and
      function_exported?(EXLA.Backend, :__info__, 1)
      and exla_nif_loaded?()
  end

  defp exla_nif_loaded? do
    try do
      Code.ensure_loaded!(EXLA.NIF)
      true
    rescue
      _ -> false
    end
  end

  def has_eigen? do
    Code.ensure_loaded?(NxEigen)
  end

  def has_opencl? do
    case NxCL.Native.device_ctx_create(0) do
      ref when is_reference(ref) -> true
      _ -> false
    end
  end

  def device_name do
    ctx = NxCL.Native.device_ctx_create(0)
    info = NxCL.Native.device_info(ctx) |> Map.new()
    info["name"]
  end

  def report_backends do
    IO.puts("\n=== Available backends ===")
    IO.puts("  BinaryBackend: always")
    IO.puts("  NxCL.Backend:  #{if has_opencl?(), do: "yes (#{device_name()})", else: "no"}")
    IO.puts("  EXLA:          #{if has_exla?(), do: "yes", else: "no"}")
    IO.puts("  NxEigen:       #{if has_eigen?(), do: "yes", else: "no"}")
    IO.puts("")
  end

  @doc "Build a map of backend name => setup function, filtering to what's available"
  def eager_backends do
    backends = %{"BinaryBackend" => fn -> Nx.BinaryBackend end}

    backends =
      if has_opencl?() do
        Map.put(backends, "NxCL.Backend", fn -> NxCL.Backend end)
      else
        backends
      end

    backends =
      if has_exla?() do
        Map.put(backends, "EXLA", fn -> EXLA.Backend end)
      else
        backends
      end

    backends =
      if has_eigen?() do
        Map.put(backends, "NxEigen", fn -> NxEigen end)
      else
        backends
      end

    backends
  end

  @doc "Run a function with a given backend set as default"
  def with_backend(backend_mod, fun) do
    Nx.with_default_backend(backend_mod, fun)
  end

  @doc "Generate a random f32 tensor with a given shape on a specific backend"
  def rand(shape, backend \\ Nx.BinaryBackend) do
    key = Nx.Random.key(:rand.uniform(1_000_000))
    {t, _key} = Nx.Random.uniform(key, shape: shape, type: :f32)
    Nx.backend_transfer(t, backend)
  end
end

BenchHelper.report_backends()
