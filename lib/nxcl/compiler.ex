defmodule NxCL.Compiler do
  @moduledoc """
  `Nx.Defn.Compiler` implementation for OpenCL.

  Receives a computation graph from `defn`, flattens it into a linear
  op sequence, and executes each op on the GPU via the NIF layer.

  Unlike the eager backend (`NxCL.Backend`) which dispatches one op at
  a time, the compiler sees the entire computation graph up front. This
  enables future optimisations like elementwise fusion, GEMM+bias+activation
  fusion, and buffer reuse via liveness analysis.

  ## Usage

  Use with `Nx.Defn.jit/2`:

      defn predict(x, w, b) do
        x |> Nx.dot(w) |> Nx.add(b) |> Nx.sigmoid()
      end

      jitted = Nx.Defn.jit(&predict/3, compiler: NxCL.Compiler)
      result = jitted.(input, weights, bias)

  Or set as the default compiler:

      Nx.Defn.default_options(compiler: NxCL.Compiler)

  ## Supported operations

  The compiler supports the same GPU-accelerated ops as `NxCL.Backend`:
  elementwise arithmetic (`add`, `subtract`, `multiply`, `divide`),
  unary math (`exp`, `log`, `tanh`, `sigmoid`, `negate`, `abs`),
  `dot` (2D matmul), plus `reshape` and `squeeze` as zero-copy ops.

  Unsupported ops raise `NxCL.CompilerError` at runtime. For full op
  coverage with automatic CPU fallback, use `NxCL.Backend` instead.
  """

  @behaviour Nx.Defn.Compiler

  alias NxCL.Compiler.{FlatOp, Flatten}

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl true
  def __compile__(_key, vars, fun, _opts) do
    # Trace: build the expression tree
    expr = fun.(vars)

    # Flatten the expr tree into a linear op sequence
    flat_ops = Flatten.flatten(expr)

    # Return a function that executes the plan
    fn [args] ->
      [execute(flat_ops, args, expr)]
    end
  end

  @impl true
  def __partitions_options__(opts) do
    [opts]
  end

  @impl true
  def __to_backend__(_opts) do
    {NxCL.Backend, []}
  end

  # Execute the flat op sequence naively (Phase C0: no fusion, no buffer reuse)
  # Buffer map stores {nif_ref, shape} tuples
  defp execute(flat_ops, args, output_expr) do
    ctx = get_ctx()

    buffers =
      Enum.reduce(flat_ops, %{}, fn op, bufs ->
        {ref, shape} = execute_op(ctx, op, bufs, args)
        Map.put(bufs, op.id, {ref, shape})
      end)

    # Read back the output
    output_id = output_expr.data.id
    {output_ref, _shape} = buffers[output_id]
    output_type = output_expr.type
    output_shape = output_expr.shape

    bin = NxCL.Native.buffer_read(output_ref, byte_size(output_shape, output_type))

    Nx.from_binary(bin, output_type)
    |> Nx.reshape(output_shape)
  end

  defp execute_op(ctx, %FlatOp{op: :parameter, param_index: idx}, _bufs, args) do
    tensor = Enum.at(args, idx).()
    tensor = Nx.as_type(tensor, :f32)
    ref = NxCL.Native.buffer_write(ctx, Nx.to_binary(tensor))
    {ref, tensor.shape}
  end

  defp execute_op(
         ctx,
         %FlatOp{op: :constant, value: value, shape: shape, type: type},
         _bufs,
         _args
       ) do
    tensor = Nx.broadcast(Nx.tensor(value, type: type), shape)
    tensor = Nx.as_type(tensor, :f32)
    ref = NxCL.Native.buffer_write(ctx, Nx.to_binary(tensor))
    {ref, shape}
  end

  defp execute_op(ctx, %FlatOp{op: :tensor, tensor: tensor, shape: shape}, _bufs, _args) do
    tensor = Nx.as_type(tensor, :f32)
    ref = NxCL.Native.buffer_write(ctx, Nx.to_binary(tensor))
    {ref, shape}
  end

  # Unary elementwise ops with GPU kernels
  @gpu_unary %{
    negate: "negate",
    exp: "exp",
    log: "log",
    tanh: "tanh",
    sigmoid: "sigmoid",
    abs: "abs"
  }

  for {op, kernel_name} <- @gpu_unary do
    defp execute_op(
           ctx,
           %FlatOp{op: unquote(op), args: [{:ref, input_id}], shape: shape},
           bufs,
           _args
         ) do
      {input_ref, _} = bufs[input_id]
      n = tuple_product(shape)
      ref = NxCL.Native.elementwise_unary_op(ctx, unquote(kernel_name), input_ref, n)
      {ref, shape}
    end
  end

  # Binary elementwise ops with GPU kernels
  @gpu_binary %{
    add: "add",
    subtract: "subtract",
    multiply: "multiply",
    divide: "divide"
  }

  for {op, kernel_name} <- @gpu_binary do
    defp execute_op(
           ctx,
           %FlatOp{op: unquote(op), args: [{:ref, left_id}, {:ref, right_id}], shape: shape},
           bufs,
           _args
         ) do
      {left_ref, _} = bufs[left_id]
      {right_ref, _} = bufs[right_id]
      n = tuple_product(shape)
      ref = NxCL.Native.elementwise_binary_op(ctx, unquote(kernel_name), left_ref, right_ref, n)
      {ref, shape}
    end
  end

  # Matrix multiply (2D only for C0)
  defp execute_op(
         ctx,
         %FlatOp{
           op: :dot,
           args: [{:ref, left_id}, {:ref, right_id}],
           shape: {m, n} = shape,
           contract_axes: {_, _},
           batch_axes: {[], []}
         },
         bufs,
         _args
       ) do
    {left_ref, left_shape} = bufs[left_id]
    {right_ref, _right_shape} = bufs[right_id]

    # K is the contracted dimension
    k = elem(left_shape, tuple_size(left_shape) - 1)

    ref = NxCL.Native.matmul_op(ctx, left_ref, right_ref, m, n, k)
    {ref, shape}
  end

  # Reshape: zero-copy, just pass through the buffer with new shape
  defp execute_op(
         _ctx,
         %FlatOp{op: :reshape, args: [{:ref, input_id}], shape: shape},
         bufs,
         _args
       ) do
    {ref, _old_shape} = bufs[input_id]
    {ref, shape}
  end

  # Squeeze: same as reshape
  defp execute_op(
         _ctx,
         %FlatOp{op: :squeeze, args: [{:ref, input_id} | _], shape: shape},
         bufs,
         _args
       ) do
    {ref, _old_shape} = bufs[input_id]
    {ref, shape}
  end

  # Fallback: unsupported ops
  defp execute_op(_ctx, %FlatOp{op: op}, _bufs, _args) do
    raise NxCL.CompilerError,
      message: "Unsupported op in NxCL.Compiler: #{inspect(op)}"
  end

  defp get_ctx() do
    case :persistent_term.get(:nxcl_device_ctx, nil) do
      nil ->
        ctx = NxCL.Native.device_ctx_create(0)
        :persistent_term.put(:nxcl_device_ctx, ctx)
        ctx

      ctx ->
        ctx
    end
  end

  defp byte_size(shape, {_, bit_size}) do
    tuple_product(shape) * div(bit_size, 8)
  end

  defp tuple_product(shape) when is_tuple(shape) do
    shape |> Tuple.to_list() |> Enum.product()
  end
end

defmodule NxCL.CompilerError do
  defexception [:message]
end
