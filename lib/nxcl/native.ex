defmodule NxCL.Native do
  @moduledoc false

  use Rustler,
    otp_app: :nx_opencl,
    crate: "nxcl_nif"

  @spec device_ctx_create(non_neg_integer()) :: reference() | {:error, String.t()}
  def device_ctx_create(_device_id), do: :erlang.nif_error(:nif_not_loaded)

  @spec device_info(reference()) :: [{String.t(), String.t()}]
  def device_info(_ctx), do: :erlang.nif_error(:nif_not_loaded)

  @spec buffer_create(reference(), non_neg_integer()) :: reference()
  def buffer_create(_ctx, _size), do: :erlang.nif_error(:nif_not_loaded)

  @spec buffer_write(reference(), binary()) :: reference()
  def buffer_write(_ctx, _data), do: :erlang.nif_error(:nif_not_loaded)

  @spec buffer_read(reference(), non_neg_integer()) :: binary()
  def buffer_read(_buf, _size), do: :erlang.nif_error(:nif_not_loaded)

  @spec elementwise_binary_op(
          reference(),
          String.t(),
          reference(),
          reference(),
          non_neg_integer()
        ) ::
          reference()
  def elementwise_binary_op(_ctx, _op, _a, _b, _n), do: :erlang.nif_error(:nif_not_loaded)

  @spec elementwise_unary_op(reference(), String.t(), reference(), non_neg_integer()) ::
          reference()
  def elementwise_unary_op(_ctx, _op, _a, _n), do: :erlang.nif_error(:nif_not_loaded)

  @spec matmul_op(
          reference(),
          reference(),
          reference(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer()
        ) :: reference()
  def matmul_op(_ctx, _a, _b, _m, _n, _k), do: :erlang.nif_error(:nif_not_loaded)

  @spec reduce_op(reference(), String.t(), reference(), non_neg_integer()) :: reference()
  def reduce_op(_ctx, _op, _input, _n), do: :erlang.nif_error(:nif_not_loaded)

  @spec softmax_op(reference(), reference(), non_neg_integer(), non_neg_integer()) :: reference()
  def softmax_op(_ctx, _input, _rows, _cols), do: :erlang.nif_error(:nif_not_loaded)
end
