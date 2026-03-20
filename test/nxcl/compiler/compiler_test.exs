defmodule NxCL.CompilerTest do
  @moduledoc """
  Full compiler round-trip tests: trace → flatten → execute → verify.
  Compares compiled results against CPU evaluation.
  """
  use ExUnit.Case, async: false
  import NxCL.TestHelpers

  @moduletag :opencl

  defp assert_compiler_matches_cpu(fun, args, atol \\ 1.0e-4) do
    # CPU reference
    cpu_result = apply(fun, args)

    # Compiled
    compiled = Nx.Defn.jit(fun, compiler: NxCL.Compiler)
    compiled_result = apply(compiled, args)

    assert_tensors_close(compiled_result, cpu_result, atol)
  end

  describe "single ops" do
    test "exp" do
      assert_compiler_matches_cpu(fn x -> Nx.exp(x) end, [Nx.tensor([0.0, 1.0, 2.0])])
    end

    test "log" do
      assert_compiler_matches_cpu(fn x -> Nx.log(x) end, [Nx.tensor([1.0, 2.0, 3.0])])
    end

    test "negate" do
      assert_compiler_matches_cpu(fn x -> Nx.negate(x) end, [Nx.tensor([1.0, -2.0, 3.0])])
    end

    test "sigmoid" do
      assert_compiler_matches_cpu(fn x -> Nx.sigmoid(x) end, [Nx.tensor([0.0, 1.0, -1.0])])
    end

    test "tanh" do
      assert_compiler_matches_cpu(fn x -> Nx.tanh(x) end, [Nx.tensor([0.0, 1.0, -1.0])])
    end

    test "abs" do
      assert_compiler_matches_cpu(fn x -> Nx.abs(x) end, [Nx.tensor([-1.0, 2.0, -3.0])])
    end
  end

  describe "binary ops" do
    test "add" do
      assert_compiler_matches_cpu(
        fn x, y -> Nx.add(x, y) end,
        [Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([10.0, 20.0, 30.0])]
      )
    end

    test "subtract" do
      assert_compiler_matches_cpu(
        fn x, y -> Nx.subtract(x, y) end,
        [Nx.tensor([10.0, 20.0, 30.0]), Nx.tensor([1.0, 2.0, 3.0])]
      )
    end

    test "multiply" do
      assert_compiler_matches_cpu(
        fn x, y -> Nx.multiply(x, y) end,
        [Nx.tensor([2.0, 3.0, 4.0]), Nx.tensor([5.0, 5.0, 5.0])]
      )
    end

    test "divide" do
      assert_compiler_matches_cpu(
        fn x, y -> Nx.divide(x, y) end,
        [Nx.tensor([10.0, 20.0, 30.0]), Nx.tensor([2.0, 4.0, 5.0])]
      )
    end
  end

  describe "chains" do
    test "exp then log (identity)" do
      assert_compiler_matches_cpu(
        fn x -> x |> Nx.exp() |> Nx.log() end,
        [Nx.tensor([1.0, 2.0, 3.0])]
      )
    end

    test "add then negate" do
      assert_compiler_matches_cpu(
        fn x, y -> Nx.add(x, y) |> Nx.negate() end,
        [Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])]
      )
    end

    test "multiply then sigmoid" do
      assert_compiler_matches_cpu(
        fn x, y -> Nx.multiply(x, y) |> Nx.sigmoid() end,
        [Nx.tensor([0.5, 1.0, 1.5]), Nx.tensor([2.0, 2.0, 2.0])]
      )
    end
  end

  describe "matmul" do
    test "simple 2x2 matmul" do
      assert_compiler_matches_cpu(
        fn x, w -> Nx.dot(x, w) end,
        [
          Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
          Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
        ]
      )
    end

    test "non-square matmul" do
      key = Nx.Random.key(42)
      {a, key} = Nx.Random.uniform(key, shape: {4, 8})
      {b, _key} = Nx.Random.uniform(key, shape: {8, 6})

      assert_compiler_matches_cpu(fn x, w -> Nx.dot(x, w) end, [a, b])
    end
  end

  describe "dense layer pattern" do
    test "dot + add" do
      assert_compiler_matches_cpu(
        fn x, w, b -> x |> Nx.dot(w) |> Nx.add(b) end,
        [
          Nx.tensor([[1.0, 2.0]]),
          Nx.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
          Nx.tensor([[0.1, 0.2, 0.3]])
        ]
      )
    end
  end

  describe "determinism" do
    test "same result on repeated calls" do
      fun = fn x -> Nx.sigmoid(x) end
      jitted = Nx.Defn.jit(fun, compiler: NxCL.Compiler)
      input = Nx.tensor([1.0, 2.0, 3.0])

      r1 = jitted.(input) |> Nx.to_binary()
      r2 = jitted.(input) |> Nx.to_binary()
      assert r1 == r2
    end
  end
end
