defmodule NxCL.Compiler.FlattenTest do
  use ExUnit.Case, async: true

  alias NxCL.Compiler.Flatten

  # Helper: use our compiler's flatten path to get flat ops from a defn-style function
  # We trace by actually calling the compiler with the Evaluator's tracing mechanism
  defp trace_and_flatten(fun, args) do
    # Jit the function with our compiler, but intercept the flatten output.
    # Simplest: just call the compiler's internal flatten on the traced expr.
    # Use Nx.Defn.jit which handles all the tracing properly.
    {_fun, params, _templates, _flatten} =
      Nx.Defn.Compiler.to_lazy_params(fun, args)

    # params is a list of Expr-backed tensors, fun takes individual args
    expr = apply(fun, params)

    Flatten.flatten(expr)
  end

  describe "linear chain" do
    test "single unary op" do
      ops = trace_and_flatten(fn x -> Nx.exp(x) end, [Nx.tensor([1.0, 2.0, 3.0])])

      op_types = Enum.map(ops, & &1.op)
      assert :parameter in op_types
      assert :exp in op_types
    end

    test "chain of unary ops in correct order" do
      ops = trace_and_flatten(fn x -> x |> Nx.exp() |> Nx.log() end, [Nx.tensor([1.0])])

      op_types = Enum.map(ops, & &1.op)

      param_idx = Enum.find_index(op_types, &(&1 == :parameter))
      exp_idx = Enum.find_index(op_types, &(&1 == :exp))
      log_idx = Enum.find_index(op_types, &(&1 == :log))

      assert param_idx < exp_idx
      assert exp_idx < log_idx
    end

    test "binary op" do
      ops =
        trace_and_flatten(
          fn x, y -> Nx.add(x, y) end,
          [Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])]
        )

      op_types = Enum.map(ops, & &1.op)
      assert Enum.count(op_types, &(&1 == :parameter)) == 2
      assert :add in op_types
    end
  end

  describe "DAG (shared nodes)" do
    test "diamond: shared input appears once" do
      ops =
        trace_and_flatten(
          fn x ->
            a = Nx.exp(x)
            b = Nx.log(x)
            Nx.add(a, b)
          end,
          [Nx.tensor([1.0])]
        )

      param_ops = Enum.filter(ops, &(&1.op == :parameter))
      assert length(param_ops) == 1
    end
  end

  describe "consumer counts" do
    test "parameter used twice has 2 consumers" do
      ops =
        trace_and_flatten(
          fn x ->
            a = Nx.exp(x)
            b = Nx.log(x)
            Nx.add(a, b)
          end,
          [Nx.tensor([1.0])]
        )

      param_op = Enum.find(ops, &(&1.op == :parameter))
      assert length(param_op.consumers) == 2
    end

    test "intermediate with single consumer has 1 consumer" do
      ops = trace_and_flatten(fn x -> x |> Nx.exp() |> Nx.log() end, [Nx.tensor([1.0])])

      exp_op = Enum.find(ops, &(&1.op == :exp))
      assert length(exp_op.consumers) == 1
    end

    test "final op has 0 consumers" do
      ops = trace_and_flatten(fn x -> Nx.exp(x) end, [Nx.tensor([1.0])])

      exp_op = Enum.find(ops, &(&1.op == :exp))
      assert length(exp_op.consumers) == 0
    end
  end

  describe "dot op" do
    test "matmul has contract axes" do
      ops =
        trace_and_flatten(
          fn x, w -> Nx.dot(x, w) end,
          [Nx.tensor([[1.0, 2.0]]), Nx.tensor([[3.0], [4.0]])]
        )

      dot_op = Enum.find(ops, &(&1.op == :dot))
      assert dot_op != nil
      assert dot_op.shape == {1, 1}
      assert match?({[_], [_]}, dot_op.contract_axes)
    end
  end

  describe "shapes" do
    test "ops carry correct output shapes" do
      ops = trace_and_flatten(fn x -> Nx.exp(x) end, [Nx.tensor([[1.0, 2.0], [3.0, 4.0]])])

      exp_op = Enum.find(ops, &(&1.op == :exp))
      assert exp_op.shape == {2, 2}
    end
  end
end
