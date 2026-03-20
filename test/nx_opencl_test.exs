defmodule NxOpenclTest do
  use ExUnit.Case

  test "NIF module loads" do
    ctx = NxCL.Native.device_ctx_create(0)
    assert is_reference(ctx)
  end
end
