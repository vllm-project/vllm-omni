from importlib.util import find_spec

import torch
from einops import rearrange, repeat
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.custom_op import CustomOp

logger = init_logger(__name__)


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def apply_rotary_emb_mindiesd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    half_head_dim: bool = True,  # if true, size of sin and cos is (B, S, D/2), otherwise (B, S, D)
) -> torch.Tensor:
    from mindiesd import rotary_position_embedding

    if cos.dim() == 3:
        # (B, S, D/2) -> (S, D/2)
        cos = cos[0]
        sin = sin[0]

    if interleaved:
        # if last dim of sin and cos is D/2, expand to (S, D) to adapt to mindiesd operators
        if half_head_dim:
            seqlen = cos.shape[0]
            sin = sin.unsqueeze(0).unsqueeze(2).unsqueeze(-1).expand(-1, -1, -1, -1, 2).reshape(1, seqlen, 1, -1)
            cos = cos.unsqueeze(0).unsqueeze(2).unsqueeze(-1).expand(-1, -1, -1, -1, 2).reshape(1, seqlen, 1, -1)
        return rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", head_first=False, fused=True)
    else:
        if half_head_dim:
            seqlen = cos.shape[0]
            sin = sin.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
            cos = cos.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        return rotary_position_embedding(x, cos, sin, rotated_mode="rotated_half", head_first=False, fused=True)


class RotaryEmbedding(CustomOp):
    """
    rotary positional embedding.
    interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
           of 1st half and 2nd half (GPT-NeoX style).
    """

    def __init__(self, is_neox_style: bool = False) -> None:
        super().__init__()
        self.is_neox_style = is_neox_style
        self.interleaved = not is_neox_style
        self.apply_rotary_emb_flash_attn = None
        self.has_mindie = False
        if find_spec("flash_attn") is not None:
            from flash_attn.ops.triton.rotary import apply_rotary

            self.apply_rotary_emb_flash_attn = apply_rotary
        if find_spec("mindiesd") is not None:
            self.has_mindie = True

    def forward_cuda(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

        if cos.dim() == 3:
            # (B, S, D/2) -> (S, D/2)
            cos = cos[0]
            sin = sin[0]

        return apply_rotary_emb(
            x,
            cos,
            sin,
            interleaved=self.interleaved,
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.apply_rotary_emb_flash_attn is None:
            return self.forward_cuda(x, cos, sin)

        if cos.dim() == 3:
            # (B, S, D/2) -> (S, D/2)
            cos = cos[0]
            sin = sin[0]

        return self.apply_rotary_emb_flash_attn(
            x,
            cos,
            sin,
            interleaved=self.interleaved,
        )

    def forward_npu(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_mindie:
            return apply_rotary_emb_mindiesd(x, cos, sin, self.interleaved)
        else:
            return self.forward_native(x, cos, sin)

    def forward_xpu(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, cos, sin)

    def forward_musa(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, cos, sin)

    def forward_native(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return apply_rotary_emb_torch(
            x,
            cos,
            sin,
            interleaved=self.interleaved,
        )


class RotaryEmbeddingWan(RotaryEmbedding):
    """
    rotary positional embedding for Wan.
    interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
           of 1st half and 2nd half (GPT-NeoX style).
    """

    def __init__(self, is_neox_style: bool = False, half_head_dim: bool = False) -> None:
        super().__init__(is_neox_style=is_neox_style)
        self.half_head_dim = half_head_dim

    def forward_cuda(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, cos, sin)

    def forward_npu(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_mindie:
            if cos.dim() > 2:
                cos = cos.reshape(-1, cos.shape[-1])
                sin = sin.reshape(-1, sin.shape[-1])
            return apply_rotary_emb_mindiesd(x, cos, sin, self.interleaved, self.half_head_dim)
        else:
            return self.forward_native(x, cos, sin)

    def forward_native(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        rotated = torch.stack(
            (
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos,
            ),
            dim=-1,
        )
        return rotated.flatten(-2, -1).to(x.dtype)


def apply_rope_to_qk(
    rope: RotaryEmbedding,
    query: torch.Tensor,
    key: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors.

    Args:
        rope: RotaryEmbedding instance for applying position embeddings
        query: Query tensor [B, S, H, D]
        key: Key tensor [B, S, H, D]
        image_rotary_emb: Tuple of (cos, sin) tensors or None

    Returns:
        Tuple of (query, key) with RoPE applied if rotary embeddings provided
    """
    if image_rotary_emb is not None:
        cos, sin = image_rotary_emb
        cos = cos.to(query.dtype)
        sin = sin.to(query.dtype)
        query = rope(query, cos, sin)
        key = rope(key, cos, sin)
    return query, key
