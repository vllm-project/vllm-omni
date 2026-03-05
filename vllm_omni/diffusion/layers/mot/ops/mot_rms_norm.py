# ruff: noqa: N803, E741
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _mot_rms_norm_kernel(
    input_ptr,
    text_weight_ptr,
    vae_weight_ptr,
    text_indices_ptr,  # MoT Routing Info
    vae_indices_ptr,  # MoT Routing Info
    M_text,  # MoT Routing Info
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    input shape: (batch_size*seq_len, hidden_size)
    RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
    """
    # Step 0: MoT Routing
    pid = tl.program_id(0).to(tl.int64)

    # dummy init (must be scalar int64[] to match tl.load return type)
    row_idx = tl.cast(0, tl.int64)
    weight_ptr = text_weight_ptr

    if pid < M_text:
        # --- Text Path ---
        row_idx = tl.load(text_indices_ptr + pid)
        weight_ptr = text_weight_ptr
    else:
        # --- VAE Path ---
        vae_pid = pid - M_text
        row_idx = tl.load(vae_indices_ptr + vae_pid)
        weight_ptr = vae_weight_ptr

    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Compute sum of squares in float32 to avoid overflow
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        # Convert to float32 for accumulation to prevent overflow
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

    # Step 2: Compute RMS (root mean square) in float32
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    # Step 3: Normalize and apply weight
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        # Compute in float32 then convert back to input dtype
        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        output = output_f32.to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


@triton.jit
def _mot_rms_norm_qk_kernel(
    input_ptr,
    text_weight_ptr,
    vae_weight_ptr,
    text_indices_ptr,
    vae_indices_ptr,
    M_text,
    output_ptr,
    stride_tok,
    stride_head,
    out_stride_tok,
    out_stride_head,
    num_heads,
    head_dim,
    eps,
    BLOCK_DIM: tl.constexpr,  # for head_dim=128,should be 128
    SHARED_WEIGHT: tl.constexpr,  # weights are shared across heads
):
    pid_tok = tl.program_id(0).to(tl.int64)

    # MoT Routing
    if pid_tok < M_text:
        row_idx = tl.load(text_indices_ptr + pid_tok)
        weight_ptr = text_weight_ptr
    else:
        vae_pid = pid_tok - M_text
        row_idx = tl.load(vae_indices_ptr + vae_pid)
        weight_ptr = vae_weight_ptr

    dim_offsets = tl.arange(0, BLOCK_DIM)
    mask = dim_offsets < head_dim

    # input.shape=(4098,28,128)=(seq_len,num_heads,head_dim)
    # weight.shape=(28,128)=(num_heads,head_dim) or (128,)=(head_dim)
    tok_in_ptr = input_ptr + row_idx * stride_tok
    tok_out_ptr = output_ptr + row_idx * out_stride_tok

    # if weight.shape=(128,)
    if SHARED_WEIGHT:
        weight = tl.load(weight_ptr + dim_offsets, mask=mask, other=1.0)
        weight_f32 = weight.to(tl.float32)

    # one program loop over all heads
    for h in range(num_heads):
        head_in_ptr = tok_in_ptr + h * stride_head
        head_out_ptr = tok_out_ptr + h * out_stride_head

        # if weight.shape=(28,128)
        if not SHARED_WEIGHT:
            weight_offset = h * head_dim + dim_offsets
            weight = tl.load(weight_ptr + weight_offset, mask=mask, other=1.0)
            weight_f32 = weight.to(tl.float32)

        vals = tl.load(head_in_ptr + dim_offsets, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)

        sq_vals = vals_f32 * vals_f32
        sum_sq = tl.sum(tl.where(mask, sq_vals, 0.0))
        mean_sq = sum_sq / head_dim
        rms = tl.sqrt(mean_sq + eps)
        inv_rms = 1.0 / rms

        output_f32 = vals_f32 * inv_rms * weight_f32
        tl.store(head_out_ptr + dim_offsets, output_f32.to(vals.dtype), mask=mask)


def mot_rms_norm(
    input: torch.Tensor,
    text_weight: torch.Tensor,
    vae_weight: torch.Tensor,
    text_indices: torch.Tensor,
    vae_indices: torch.Tensor,
    head_norm: bool = False,
    eps: float = 1e-6,
    block_size: int | None = None,
) -> torch.Tensor:
    """
    Compute RMS normalization using Triton kernel.

    RMS Norm normalizes the input by the root mean square and scales by weight:
    output = input / sqrt(mean(input^2) + eps) * weight

    Args:
        input: Input tensor of shape  (batch_size*seq_len, hidden_size) or (batch_size,seq_len, hidden_size)
        text_weight: Weight for text tokens tensor of shape (hidden_size,)
        vae_weight: Weight for vae tokens tensor of shape (hidden_size,)
        text_indices: indices of text tokens, (batch_size*2,)
        vae_indices: indices of vae tokens, (batch_size*(seq_len-2),)
        eps: Small constant for numerical stability

    Returns:
        Tensor with RMS normalization applied along the last dimension
    """

    assert input.shape[-1] == text_weight.shape[-1], (
        f"Input last dimension ({input.shape[-1]}) must match Text weight dimension ({text_weight.shape[-1]})"
    )
    assert input.shape[-1] == vae_weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match VAE weight dimension ({vae_weight.shape[-1]})"
    )

    original_shape = input.shape
    text_indices = text_indices.reshape(-1)
    vae_indices = vae_indices.reshape(-1)
    M_text = text_indices.shape[0]
    M_vae = vae_indices.shape[0]
    num_tokens = M_text + M_vae

    if not head_norm:
        input_2d = input.reshape(-1, input.shape[-1])
        assert input_2d.shape[0] == num_tokens, (
            f"batch_size={input_2d.shape[0]}, len(text_indices)={M_text}, len(vae_indices)={M_vae}"
            f"for layer norm, batched_token_length should match the sum of indices_length"
        )
        input_2d = input_2d.contiguous()
        text_indices = text_indices.contiguous()
        vae_indices = vae_indices.contiguous()

        text_weight = text_weight.contiguous()
        vae_weight = vae_weight.contiguous()

        n_rows, n_cols = input_2d.shape

        output = torch.empty_like(input_2d)

        if block_size is None:
            block_size = triton.next_power_of_2(n_cols)
            block_size = min(block_size, 4096)

        num_warps = 4
        if block_size >= 2048:
            num_warps = 8
        elif block_size >= 1024:
            num_warps = 4
        else:
            num_warps = 2

        grid = (n_rows,)
        _mot_rms_norm_kernel[grid](
            input_2d,
            text_weight,
            vae_weight,
            text_indices,
            vae_indices,
            M_text,
            output,
            input_2d.stride(0),
            output.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return output.reshape(original_shape)
    else:
        # qk norm scenarios:
        # input.shape=(batch_size, seq_len,head_num, head_dim) or
        # input.shape=(batch_size*seq_len,head_num, head_dim)
        assert len(original_shape) > 2, (
            "If head_norm=True,input shape be 3D or 3D, last 2 dimensions should be head_num and head_dim"
        )
        num_heads = input.shape[-2]
        head_dim = input.shape[-1]

        is_shared_weight = text_weight.dim() == 1
        if not is_shared_weight:
            assert num_heads == text_weight.shape[0] and num_heads == vae_weight.shape[0], (
                "when weights are not shared across heads, the first dimension of "
                "weights should be num of heads and match input.shape[-2]"
            )

        # reshape to 3D
        input_3d = input.view(-1, num_heads, head_dim)

        input_3d = input_3d.contiguous()
        text_indices = text_indices.contiguous()
        vae_indices = vae_indices.contiguous()
        text_weight = text_weight.contiguous()
        vae_weight = vae_weight.contiguous()

        output_3d = torch.empty_like(input_3d)

        block_dim = triton.next_power_of_2(head_dim)
        num_warps = 4 if block_dim > 128 else 2
        num_stages = 3

        _mot_rms_norm_qk_kernel[(num_tokens,)](
            input_3d,
            text_weight,
            vae_weight,
            text_indices,
            vae_indices,
            M_text,
            output_3d,
            input_3d.stride(0),
            input_3d.stride(1),
            output_3d.stride(0),
            output_3d.stride(1),
            num_heads,
            head_dim,
            eps,
            BLOCK_DIM=block_dim,
            SHARED_WEIGHT=is_shared_weight,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output_3d.view(original_shape)
