defmodule NxCL.Kernels.SoftmaxTest do
  use ExUnit.Case, async: false
  import NxCL.TestHelpers

  @moduletag :opencl
  setup do
    ctx = NxCL.Native.device_ctx_create(0)
    %{ctx: ctx}
  end

  describe "softmax" do
    test "single row sums to 1.0", %{ctx: ctx} do
      data = floats_to_binary([1.0, 2.0, 3.0, 4.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.softmax_op(ctx, buf, 1, 4)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 16))

      sum = Enum.sum(result)
      assert_in_delta sum, 1.0, 1.0e-5

      # All values should be positive
      assert Enum.all?(result, &(&1 > 0.0))

      # Values should be monotonically increasing (since input is)
      assert result == Enum.sort(result)
    end

    test "uniform input gives uniform output", %{ctx: ctx} do
      data = floats_to_binary([1.0, 1.0, 1.0, 1.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.softmax_op(ctx, buf, 1, 4)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 16))

      for val <- result do
        assert_in_delta val, 0.25, 1.0e-5
      end
    end

    test "multiple rows", %{ctx: ctx} do
      # 2 rows x 3 cols
      data = floats_to_binary([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.softmax_op(ctx, buf, 2, 3)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 24))

      row1 = Enum.slice(result, 0, 3)
      row2 = Enum.slice(result, 3, 3)

      assert_in_delta Enum.sum(row1), 1.0, 1.0e-5
      assert_in_delta Enum.sum(row2), 1.0, 1.0e-5
    end

    test "matches Nx softmax", %{ctx: ctx} do
      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      expected = Nx.divide(Nx.exp(input), Nx.sum(Nx.exp(input)))

      data = Nx.to_binary(input)
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.softmax_op(ctx, buf, 1, 4)
      result_bin = NxCL.Native.buffer_read(result_buf, 16)
      result = Nx.from_binary(result_bin, :f32) |> Nx.reshape({1, 4})

      assert_tensors_close(result, expected, 1.0e-5)
    end

    test "large logits (numerical stability)", %{ctx: ctx} do
      # Large values - should not overflow thanks to max subtraction
      data = floats_to_binary([1000.0, 1001.0, 1002.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.softmax_op(ctx, buf, 1, 3)
      result = binary_to_floats(NxCL.Native.buffer_read(result_buf, 12))

      sum = Enum.sum(result)
      assert_in_delta sum, 1.0, 1.0e-5

      # Should not contain NaN or Inf
      for val <- result do
        assert is_float(val) and val == val and abs(val) != :infinity
      end
    end

    test "single class", %{ctx: ctx} do
      data = floats_to_binary([5.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result_buf = NxCL.Native.softmax_op(ctx, buf, 1, 1)
      [result] = binary_to_floats(NxCL.Native.buffer_read(result_buf, 4))

      assert_in_delta result, 1.0, 1.0e-5
    end
  end
end
