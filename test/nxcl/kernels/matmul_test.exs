defmodule NxCL.Kernels.MatmulTest do
  use ExUnit.Case, async: false
  import NxCL.TestHelpers

  @moduletag :opencl
  setup do
    ctx = NxCL.Native.device_ctx_create(0)
    %{ctx: ctx}
  end

  defp run_matmul(ctx, a_tensor, b_tensor) do
    {m, k} = Nx.shape(a_tensor)
    {_k, n} = Nx.shape(b_tensor)

    a_tensor = Nx.as_type(a_tensor, :f32)
    b_tensor = Nx.as_type(b_tensor, :f32)

    a_bin = Nx.to_binary(a_tensor)
    b_bin = Nx.to_binary(b_tensor)

    a_buf = NxCL.Native.buffer_write(ctx, a_bin)
    b_buf = NxCL.Native.buffer_write(ctx, b_bin)

    result_buf = NxCL.Native.matmul_op(ctx, a_buf, b_buf, m, n, k)
    result_bin = NxCL.Native.buffer_read(result_buf, m * n * 4)

    result_bin |> Nx.from_binary(:f32) |> Nx.reshape({m, n})
  end

  describe "matmul correctness" do
    test "2x2 simple", %{ctx: ctx} do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])

      result = run_matmul(ctx, a, b)
      expected = Nx.dot(a, b)

      assert_tensors_close(result, expected)
    end

    test "square matrices 8x8", %{ctx: ctx} do
      a = Nx.iota({8, 8}, type: :f32)
      b = Nx.iota({8, 8}, type: :f32)

      result = run_matmul(ctx, a, b)
      expected = Nx.dot(a, b)

      assert_tensors_close(result, expected, 1.0e-3)
    end

    test "non-square M != N != K", %{ctx: ctx} do
      key = Nx.Random.key(42)
      {a, key} = Nx.Random.uniform(key, shape: {4, 8})
      {b, _key} = Nx.Random.uniform(key, shape: {8, 6})

      result = run_matmul(ctx, a, b)
      expected = Nx.dot(a, b)

      assert_tensors_close(result, expected, 1.0e-4)
    end

    test "dimensions not divisible by tile size (13x7 * 7x11)", %{ctx: ctx} do
      key = Nx.Random.key(123)
      {a, key} = Nx.Random.uniform(key, shape: {13, 7})
      {b, _key} = Nx.Random.uniform(key, shape: {7, 11})

      result = run_matmul(ctx, a, b)
      expected = Nx.dot(a, b)

      assert_tensors_close(result, expected, 1.0e-4)
    end

    test "single element", %{ctx: ctx} do
      a = Nx.tensor([[3.0]])
      b = Nx.tensor([[7.0]])

      result = run_matmul(ctx, a, b)
      expected = Nx.tensor([[21.0]])

      assert_tensors_close(result, expected)
    end

    test "larger matrix 32x32", %{ctx: ctx} do
      key = Nx.Random.key(99)
      {a, key} = Nx.Random.uniform(key, shape: {32, 32})
      {b, _key} = Nx.Random.uniform(key, shape: {32, 32})

      result = run_matmul(ctx, a, b)
      expected = Nx.dot(a, b)

      assert_tensors_close(result, expected, 1.0e-4)
    end
  end

  describe "matmul edge cases" do
    test "zeros", %{ctx: ctx} do
      a = Nx.broadcast(Nx.tensor(0.0), {8, 8})
      key = Nx.Random.key(42)
      {b, _key} = Nx.Random.uniform(key, shape: {8, 8})

      result = run_matmul(ctx, a, b)
      expected = Nx.broadcast(Nx.tensor(0.0), {8, 8})

      assert_tensors_close(result, expected)
    end

    test "identity matrix", %{ctx: ctx} do
      key = Nx.Random.key(42)
      {a, _key} = Nx.Random.uniform(key, shape: {8, 8})
      b = Nx.eye(8)

      result = run_matmul(ctx, a, b)
      assert_tensors_close(result, a, 1.0e-5)
    end
  end
end
