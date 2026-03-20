defmodule NxCL.BackendTest do
  use ExUnit.Case, async: false
  import NxCL.TestHelpers

  @moduletag :opencl

  describe "from_binary / to_binary round-trip" do
    test "f32 tensor round-trip" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      bin = Nx.to_binary(t)

      gpu_t = Nx.backend_transfer(t, NxCL.Backend)
      assert %NxCL.Backend{} = gpu_t.data

      result_bin = Nx.to_binary(gpu_t)
      assert result_bin == bin
    end

    test "2D tensor round-trip" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      gpu_t = Nx.backend_transfer(t, NxCL.Backend)
      cpu_t = Nx.backend_transfer(gpu_t, Nx.BinaryBackend)

      assert_tensors_close(cpu_t, t)
    end
  end

  describe "elementwise binary ops via backend" do
    test "add" do
      a = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([10.0, 20.0, 30.0]) |> Nx.backend_transfer(NxCL.Backend)

      result = Nx.add(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([11.0, 22.0, 33.0])

      assert_tensors_close(result, expected)
    end

    test "subtract" do
      a = Nx.tensor([10.0, 20.0, 30.0]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)

      result = Nx.subtract(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([9.0, 18.0, 27.0])

      assert_tensors_close(result, expected)
    end

    test "multiply" do
      a = Nx.tensor([2.0, 3.0, 4.0]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([5.0, 5.0, 5.0]) |> Nx.backend_transfer(NxCL.Backend)

      result = Nx.multiply(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([10.0, 15.0, 20.0])

      assert_tensors_close(result, expected)
    end

    test "divide" do
      a = Nx.tensor([10.0, 20.0, 30.0]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([2.0, 4.0, 5.0]) |> Nx.backend_transfer(NxCL.Backend)

      result = Nx.divide(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([5.0, 5.0, 6.0])

      assert_tensors_close(result, expected)
    end
  end

  describe "elementwise unary ops via backend" do
    test "negate" do
      t = Nx.tensor([1.0, -2.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.negate(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([-1.0, 2.0, -3.0])
      assert_tensors_close(result, expected)
    end

    test "exp" do
      t = Nx.tensor([0.0, 1.0, 2.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.exp(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.exp(Nx.tensor([0.0, 1.0, 2.0]))
      assert_tensors_close(result, expected, 1.0e-5)
    end

    test "sigmoid" do
      t = Nx.tensor([0.0, 1.0, -1.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.sigmoid(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.sigmoid(Nx.tensor([0.0, 1.0, -1.0]))
      assert_tensors_close(result, expected, 1.0e-5)
    end
  end

  describe "dot (matmul) via backend" do
    test "2D matmul matches CPU" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]]) |> Nx.backend_transfer(NxCL.Backend)

      result = Nx.dot(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.dot(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), Nx.tensor([[5.0, 6.0], [7.0, 8.0]]))

      assert_tensors_close(result, expected)
    end

    test "non-square matmul" do
      key = Nx.Random.key(42)
      {a_cpu, key} = Nx.Random.uniform(key, shape: {4, 8})
      {b_cpu, _key} = Nx.Random.uniform(key, shape: {8, 6})

      a = Nx.backend_transfer(a_cpu, NxCL.Backend)
      b = Nx.backend_transfer(b_cpu, NxCL.Backend)

      result = Nx.dot(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.dot(a_cpu, b_cpu)

      assert_tensors_close(result, expected, 1.0e-4)
    end
  end

  describe "reductions via backend" do
    test "sum all elements" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.sum(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor(10.0)
      assert_tensors_close(result, expected, 1.0e-4)
    end

    test "reduce_max" do
      t = Nx.tensor([3.0, 1.0, 5.0, 2.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.reduce_max(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor(5.0)
      assert_tensors_close(result, expected, 1.0e-5)
    end

    test "reduce_min" do
      t = Nx.tensor([3.0, 1.0, 5.0, 2.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.reduce_min(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor(1.0)
      assert_tensors_close(result, expected, 1.0e-5)
    end
  end

  describe "fallback to CPU for unimplemented ops" do
    test "greater comparison works via fallback" do
      a = Nx.tensor([1.0, 5.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)
      b = Nx.tensor([2.0, 4.0, 3.0]) |> Nx.backend_transfer(NxCL.Backend)

      result = Nx.greater(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([0, 1, 0], type: :u8)

      assert result == expected
    end

    test "sort works via fallback" do
      t = Nx.tensor([3.0, 1.0, 2.0]) |> Nx.backend_transfer(NxCL.Backend)
      result = Nx.sort(t) |> Nx.backend_transfer(Nx.BinaryBackend)
      expected = Nx.tensor([1.0, 2.0, 3.0])
      assert_tensors_close(result, expected)
    end
  end

  describe "differential test: GPU vs CPU" do
    test "chain of ops matches CPU" do
      key = Nx.Random.key(42)
      {input, _key} = Nx.Random.uniform(key, shape: {4, 4})

      # CPU reference
      cpu_result =
        input
        |> Nx.multiply(Nx.tensor(2.0))
        |> Nx.add(Nx.tensor(1.0))
        |> Nx.negate()

      # GPU computation
      gpu_input = Nx.backend_transfer(input, NxCL.Backend)

      gpu_result =
        gpu_input
        |> Nx.multiply(Nx.backend_transfer(Nx.tensor(2.0), NxCL.Backend))
        |> Nx.add(Nx.backend_transfer(Nx.tensor(1.0), NxCL.Backend))
        |> Nx.negate()
        |> Nx.backend_transfer(Nx.BinaryBackend)

      assert_tensors_close(gpu_result, cpu_result, 1.0e-5)
    end
  end
end
