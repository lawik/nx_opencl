defmodule NxCL.Kernels.ElementwiseTest do
  use ExUnit.Case, async: false
  import NxCL.TestHelpers

  @moduletag :opencl
  setup do
    ctx = NxCL.Native.device_ctx_create(0)
    %{ctx: ctx}
  end

  describe "binary elementwise ops" do
    test "add two vectors", %{ctx: ctx} do
      a_data = floats_to_binary([1.0, 2.0, 3.0, 4.0])
      b_data = floats_to_binary([10.0, 20.0, 30.0, 40.0])

      a = NxCL.Native.buffer_write(ctx, a_data)
      b = NxCL.Native.buffer_write(ctx, b_data)

      result_buf = NxCL.Native.elementwise_binary_op(ctx, "add", a, b, 4)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 16))

      assert_floats_close(result, [11.0, 22.0, 33.0, 44.0])
    end

    test "subtract two vectors", %{ctx: ctx} do
      a_data = floats_to_binary([10.0, 20.0, 30.0, 40.0])
      b_data = floats_to_binary([1.0, 2.0, 3.0, 4.0])

      a = NxCL.Native.buffer_write(ctx, a_data)
      b = NxCL.Native.buffer_write(ctx, b_data)

      result_buf = NxCL.Native.elementwise_binary_op(ctx, "subtract", a, b, 4)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 16))

      assert_floats_close(result, [9.0, 18.0, 27.0, 36.0])
    end

    test "multiply two vectors", %{ctx: ctx} do
      a_data = floats_to_binary([2.0, 3.0, 4.0, 5.0])
      b_data = floats_to_binary([10.0, 10.0, 10.0, 10.0])

      a = NxCL.Native.buffer_write(ctx, a_data)
      b = NxCL.Native.buffer_write(ctx, b_data)

      result_buf = NxCL.Native.elementwise_binary_op(ctx, "multiply", a, b, 4)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 16))

      assert_floats_close(result, [20.0, 30.0, 40.0, 50.0])
    end

    test "divide two vectors", %{ctx: ctx} do
      a_data = floats_to_binary([10.0, 20.0, 30.0, 40.0])
      b_data = floats_to_binary([2.0, 4.0, 5.0, 8.0])

      a = NxCL.Native.buffer_write(ctx, a_data)
      b = NxCL.Native.buffer_write(ctx, b_data)

      result_buf = NxCL.Native.elementwise_binary_op(ctx, "divide", a, b, 4)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 16))

      assert_floats_close(result, [5.0, 5.0, 6.0, 5.0])
    end

    test "add with larger vector (non-multiple of workgroup size)", %{ctx: ctx} do
      n = 100
      a_vals = for i <- 1..n, do: i * 1.0
      b_vals = for i <- 1..n, do: i * 2.0
      expected = for i <- 1..n, do: i * 3.0

      a = NxCL.Native.buffer_write(ctx, floats_to_binary(a_vals))
      b = NxCL.Native.buffer_write(ctx, floats_to_binary(b_vals))

      result_buf = NxCL.Native.elementwise_binary_op(ctx, "add", a, b, n)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, n * 4))

      assert_floats_close(result, expected)
    end
  end

  describe "unary elementwise ops" do
    test "negate", %{ctx: ctx} do
      a = NxCL.Native.buffer_write(ctx, floats_to_binary([1.0, -2.0, 3.0, -4.0]))
      result_buf = NxCL.Native.elementwise_unary_op(ctx, "negate", a, 4)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 16))
      assert_floats_close(result, [-1.0, 2.0, -3.0, 4.0])
    end

    test "abs", %{ctx: ctx} do
      a = NxCL.Native.buffer_write(ctx, floats_to_binary([-1.0, 2.0, -3.0, 4.0]))
      result_buf = NxCL.Native.elementwise_unary_op(ctx, "abs", a, 4)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 16))
      assert_floats_close(result, [1.0, 2.0, 3.0, 4.0])
    end

    test "relu", %{ctx: ctx} do
      a = NxCL.Native.buffer_write(ctx, floats_to_binary([-2.0, -1.0, 0.0, 1.0, 2.0]))
      result_buf = NxCL.Native.elementwise_unary_op(ctx, "relu", a, 5)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 20))
      assert_floats_close(result, [0.0, 0.0, 0.0, 1.0, 2.0])
    end

    test "sigmoid", %{ctx: ctx} do
      a = NxCL.Native.buffer_write(ctx, floats_to_binary([0.0, 1.0, -1.0, 10.0]))
      result_buf = NxCL.Native.elementwise_unary_op(ctx, "sigmoid", a, 4)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 16))

      expected = [0.5, 0.7310586, 0.26894143, 0.9999546]
      assert_floats_close(result, expected, 1.0e-4)
    end

    test "exp", %{ctx: ctx} do
      a = NxCL.Native.buffer_write(ctx, floats_to_binary([0.0, 1.0, 2.0]))
      result_buf = NxCL.Native.elementwise_unary_op(ctx, "exp", a, 3)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 12))

      expected = [1.0, :math.exp(1.0), :math.exp(2.0)]
      assert_floats_close(result, expected, 1.0e-5)
    end

    test "log", %{ctx: ctx} do
      a = NxCL.Native.buffer_write(ctx, floats_to_binary([1.0, :math.exp(1.0), :math.exp(2.0)]))
      result_buf = NxCL.Native.elementwise_unary_op(ctx, "log", a, 3)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 12))

      assert_floats_close(result, [0.0, 1.0, 2.0], 1.0e-5)
    end

    test "tanh", %{ctx: ctx} do
      a = NxCL.Native.buffer_write(ctx, floats_to_binary([0.0, 1.0, -1.0]))
      result_buf = NxCL.Native.elementwise_unary_op(ctx, "tanh", a, 3)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 12))

      expected = [0.0, :math.tanh(1.0), :math.tanh(-1.0)]
      assert_floats_close(result, expected, 1.0e-5)
    end
  end
end
