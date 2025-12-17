import torch
import triton
import triton.language as tl


# adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/layers/triton_ops.py
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HS_HALF": 32}, num_warps=2),
        triton.Config({"BLOCK_HS_HALF": 64}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 128}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 256}, num_warps=8),
    ],
    key=["head_size", "interleaved"],
)
@triton.jit
def _rotary_embedding_kernel(
    output_ptr,
    x_ptr,
    cos_ptr,
    sin_ptr,
    num_heads,
    head_size,
    num_tokens,
    stride_x_row,
    stride_cos_row,
    stride_sin_row,
    interleaved: tl.constexpr,
    BLOCK_HS_HALF: tl.constexpr,  # noqa N803
):
    row_idx = tl.program_id(0)
    token_idx = (row_idx // num_heads) % num_tokens

    x_row_ptr = x_ptr + row_idx * stride_x_row
    cos_row_ptr = cos_ptr + token_idx * stride_cos_row
    sin_row_ptr = sin_ptr + token_idx * stride_sin_row
    output_row_ptr = output_ptr + row_idx * stride_x_row

    # half size for x1 and x2
    head_size_half = head_size // 2

    for block_start in range(0, head_size_half, BLOCK_HS_HALF):
        offsets_half = block_start + tl.arange(0, BLOCK_HS_HALF)
        mask = offsets_half < head_size_half

        cos_vals = tl.load(cos_row_ptr + offsets_half, mask=mask, other=0.0)
        sin_vals = tl.load(sin_row_ptr + offsets_half, mask=mask, other=0.0)

        offsets_x1 = 2 * offsets_half
        offsets_x2 = 2 * offsets_half + 1

        x1_vals = tl.load(x_row_ptr + offsets_x1, mask=mask, other=0.0)
        x2_vals = tl.load(x_row_ptr + offsets_x2, mask=mask, other=0.0)

        x1_fp32 = x1_vals.to(tl.float32)
        x2_fp32 = x2_vals.to(tl.float32)
        cos_fp32 = cos_vals.to(tl.float32)
        sin_fp32 = sin_vals.to(tl.float32)
        o1_vals = tl.fma(-x2_fp32, sin_fp32, x1_fp32 * cos_fp32)
        o2_vals = tl.fma(x1_fp32, sin_fp32, x2_fp32 * cos_fp32)

        tl.store(output_row_ptr + offsets_x1, o1_vals.to(x1_vals.dtype), mask=mask)
        tl.store(output_row_ptr + offsets_x2, o2_vals.to(x2_vals.dtype), mask=mask)


def _apply_rotary_embedding(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    output = torch.empty_like(x)

    if x.dim() > 3:
        bsz, num_tokens, num_heads, head_size = x.shape
    else:
        num_tokens, num_heads, head_size = x.shape
        bsz = 1

    assert head_size % 2 == 0, "head_size must be divisible by 2"

    x_reshaped = x.view(-1, head_size)
    output_reshaped = output.view(-1, head_size)

    # num_tokens per head, 1 token per block
    grid = (bsz * num_tokens * num_heads,)

    if interleaved and cos.shape[-1] == head_size:
        cos = cos[..., ::2].contiguous()
        sin = sin[..., ::2].contiguous()
    else:
        cos = cos.contiguous()
        sin = sin.contiguous()

    _rotary_embedding_kernel[grid](
        output_reshaped,
        x_reshaped,
        cos,
        sin,
        num_heads,
        head_size,
        num_tokens,
        x_reshaped.stride(0),
        cos.stride(0),
        sin.stride(0),
        interleaved,
    )

    return output


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = False,
    interleaved: bool = False,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size] or [num_tokens, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    if is_neox_style:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        o1 = (x1.float() * cos - x2.float() * sin).type_as(x)
        o2 = (x2.float() * cos + x1.float() * sin).type_as(x)
        return torch.cat((o1, o2), dim=-1)
    else:
        return _apply_rotary_embedding(x, cos, sin, interleaved)
