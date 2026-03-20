defmodule NxCL.BufferTest do
  use ExUnit.Case, async: false
  import NxCL.TestHelpers

  @moduletag :opencl
  describe "buffer round-trip" do
    setup do
      ctx = NxCL.Native.device_ctx_create(0)
      %{ctx: ctx}
    end

    test "write and read back f32 data", %{ctx: ctx} do
      data = floats_to_binary([1.0, 2.0, 3.0, 4.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result = NxCL.Native.buffer_read(buf, byte_size(data))
      assert result == data
    end

    test "write and read back zeros", %{ctx: ctx} do
      data = floats_to_binary([0.0, 0.0, 0.0, 0.0])
      buf = NxCL.Native.buffer_write(ctx, data)

      result = NxCL.Native.buffer_read(buf, byte_size(data))
      assert result == data
    end

    test "write and read back negative values", %{ctx: ctx} do
      data = floats_to_binary([-1.0, -2.5, -100.0, 0.001])
      buf = NxCL.Native.buffer_write(ctx, data)

      result = NxCL.Native.buffer_read(buf, byte_size(data))
      assert result == data
    end

    test "write and read back larger buffer", %{ctx: ctx} do
      floats = for i <- 1..1024, do: i * 0.1
      data = floats_to_binary(floats)
      buf = NxCL.Native.buffer_write(ctx, data)

      result = NxCL.Native.buffer_read(buf, byte_size(data))
      result_floats = binary_to_floats(result)
      assert_floats_close(result_floats, floats)
    end

    test "create empty buffer", %{ctx: ctx} do
      buf = NxCL.Native.buffer_create(ctx, 16)
      assert is_reference(buf)
    end
  end
end
