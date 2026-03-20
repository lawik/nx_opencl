defmodule NxCL.Kernels.ReduceTest do
  use ExUnit.Case, async: false
  import NxCL.TestHelpers

  @moduletag :opencl
  setup do
    ctx = NxCL.Native.device_ctx_create(0)
    %{ctx: ctx}
  end

  describe "reduce_sum" do
    test "sum of small vector", %{ctx: ctx} do
      data = floats_to_binary([1.0, 2.0, 3.0, 4.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.reduce_op(ctx, "sum", buf, 4)
      [result] = binary_to_floats(NxCL.Native.buffer_read(result_buf, 4))

      assert_in_delta result, 10.0, 1.0e-5
    end

    test "sum of larger vector", %{ctx: ctx} do
      n = 1000
      vals = for i <- 1..n, do: 1.0
      buf = NxCL.Native.buffer_write(ctx, floats_to_binary(vals))

      result_buf = NxCL.Native.reduce_op(ctx, "sum", buf, n)
      [result] = binary_to_floats(NxCL.Native.buffer_read(result_buf, 4))

      assert_in_delta result, 1000.0, 1.0e-2
    end

    test "sum of vector with negatives", %{ctx: ctx} do
      data = floats_to_binary([1.0, -1.0, 2.0, -2.0, 3.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.reduce_op(ctx, "sum", buf, 5)
      [result] = binary_to_floats(NxCL.Native.buffer_read(result_buf, 4))

      assert_in_delta result, 3.0, 1.0e-5
    end

    test "sum of single element", %{ctx: ctx} do
      data = floats_to_binary([42.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.reduce_op(ctx, "sum", buf, 1)
      [result] = binary_to_floats(NxCL.Native.buffer_read(result_buf, 4))

      assert_in_delta result, 42.0, 1.0e-5
    end
  end

  describe "reduce_max" do
    test "max of small vector", %{ctx: ctx} do
      data = floats_to_binary([1.0, 5.0, 3.0, 2.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.reduce_op(ctx, "max", buf, 4)
      [result] = binary_to_floats(NxCL.Native.buffer_read(result_buf, 4))

      assert_in_delta result, 5.0, 1.0e-5
    end

    test "max with negative values", %{ctx: ctx} do
      data = floats_to_binary([-10.0, -5.0, -20.0, -1.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.reduce_op(ctx, "max", buf, 4)
      [result] = binary_to_floats(NxCL.Native.buffer_read(result_buf, 4))

      assert_in_delta result, -1.0, 1.0e-5
    end
  end

  describe "reduce_min" do
    test "min of small vector", %{ctx: ctx} do
      data = floats_to_binary([3.0, 1.0, 5.0, 2.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.reduce_op(ctx, "min", buf, 4)
      [result] = binary_to_floats(NxCL.Native.buffer_read(result_buf, 4))

      assert_in_delta result, 1.0, 1.0e-5
    end

    test "min with negative values", %{ctx: ctx} do
      data = floats_to_binary([-10.0, -5.0, -20.0, -1.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.reduce_op(ctx, "min", buf, 4)
      [result] = binary_to_floats(NxCL.Native.buffer_read(result_buf, 4))

      assert_in_delta result, -20.0, 1.0e-5
    end
  end
end
