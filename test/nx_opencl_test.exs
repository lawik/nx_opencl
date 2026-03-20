defmodule NxOpenclTest do
  use ExUnit.Case
  doctest NxOpencl

  test "greets the world" do
    assert NxOpencl.hello() == :world
  end
end
