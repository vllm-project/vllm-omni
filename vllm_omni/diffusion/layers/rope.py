from importlib.util import find_spec

import torch
from einops import rearrange, repeat
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.platforms import current_omni_platform

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

    x, squeezed = _ensure_batch_dim(x)

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
        out = rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", head_first=False, fused=True)
    else:
        if half_head_dim:
            seqlen = cos.shape[0]
            sin = sin.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
            cos = cos.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        out = rotary_position_embedding(x, cos, sin, rotated_mode="rotated_half", head_first=False, fused=True)
    return _restore_batch_dim(out, squeezed)


def _ensure_batch_dim(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    # Upstream fused rotary kernels expect ``x`` shaped as
    # ``[batch_size, seq_len, nheads, headdim]``. Some omni diffusion call
    # sites pass ``[seq_len, nheads, headdim]`` instead, so normalize to 4D
    # here before entering the fused path.
    if x.dim() == 3:
        return x.unsqueeze(0), True
    return x, False


def _restore_batch_dim(x: torch.Tensor, squeezed: bool) -> torch.Tensor:
    if squeezed:
        return x.squeeze(0)
    return x


class RotaryEmbedding(CustomOp):
    """
    rotary positional embedding.
    interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
           of 1st half and 2nd half (GPT-NeoX style).
    """

    def __init__(self, is_neox_style: bool = False, half_head_dim: bool = True) -> None:
        super().__init__()
        self.is_neox_style = is_neox_style
        self.interleaved = not is_neox_style
        self.half_head_dim = half_head_dim
        self.apply_rotary_emb_flash_attn = None
        self.has_mindie = False
        # ``find_spec("flash_attn")`` is True as long as *any* package publishes
        # the ``flash_attn`` namespace — including ``flash-attn-4``, which ships
        # only ``flash_attn.cute`` and no ``flash_attn.ops``. Guard the import
        # so a partial namespace doesn't crash the RoPE init; the CUDA forward
        # path uses ``vllm.vllm_flash_attn.layers.rotary`` anyway.
        if find_spec("flash_attn") is not None:
            try:
                from flash_attn.ops.triton.rotary import apply_rotary

                self.apply_rotary_emb_flash_attn = apply_rotary
            except ImportError:
                pass
        if find_spec("mindiesd") is not None:
            self.has_mindie = True

    def _prepare_half_head_dim_cos_sin(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize shared cos/sin to the half-dim layout used by CUDA/native kernels."""
        if cos.dim() == 3:
            # All batch elements share the same rotary position encoding.
            cos = cos[0]
            sin = sin[0]
        if not self.half_head_dim:
            cos = cos[..., ::2] if self.interleaved else cos[..., : cos.shape[-1] // 2]
            sin = sin[..., ::2] if self.interleaved else sin[..., : sin.shape[-1] // 2]
        return cos, sin

    def forward_cuda(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

        cos, sin = self._prepare_half_head_dim_cos_sin(cos, sin)

        x, squeezed = _ensure_batch_dim(x)
        output = apply_rotary_emb(
            x,
            cos,
            sin,
            interleaved=self.interleaved,
        )
        return _restore_batch_dim(output, squeezed)

    def forward_hip(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.apply_rotary_emb_flash_attn is None:
            return self.forward_cuda(x, cos, sin)

        cos, sin = self._prepare_half_head_dim_cos_sin(cos, sin)

        x, squeezed = _ensure_batch_dim(x)
        output = self.apply_rotary_emb_flash_attn(
            x,
            cos,
            sin,
            interleaved=self.interleaved,
        )
        return _restore_batch_dim(output, squeezed)

    def forward_npu(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_mindie:
            return apply_rotary_emb_mindiesd(x, cos, sin, self.interleaved, self.half_head_dim)
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
        # Keep the full-dimension NeoX form inline for MUSA compilation.
        # Other layouts use the shared native implementation.
        if self.half_head_dim or self.interleaved:
            return self.forward_native(x, cos, sin)
        if cos.dim() == 3:
            cos, sin = cos[0], sin[0]
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    def forward_native(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # All batch elements share the same rotary position encoding.
        # Strip the batch dim so the underlying op broadcasts over the batch,
        # consistent with forward_cuda / forward_hip / apply_rotary_emb_mindiesd.
        cos, sin = self._prepare_half_head_dim_cos_sin(cos, sin)
        return apply_rotary_emb_torch(
            x,
            cos,
            sin,
            interleaved=self.interleaved,
        )


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


class RotaryEmbeddingS2VGrid(torch.nn.Module):
    """Grid-based RoPE for S2V motioner/init attention.

    Applies complex-valued rotary embeddings using 3D grid sampling
    (frame, height, width). Used by SimpleSelfAttention, SwinSelfAttention,
    CausalSelfAttention in motioner blocks.
    """

    @staticmethod
    @torch.amp.autocast(current_omni_platform.device_type, enabled=False)
    def precompute(
        seq_len: int, num_heads: int, head_dim: int, grid_sizes, freqs: torch.Tensor, device: torch.device, start=None
    ) -> torch.Tensor:
        """Precompute position frequency tensor from grid specification.

        Returns a complex tensor that can be reused across layers via apply_precomputed().
        """
        c = head_dim // 2

        trainable_freqs = None
        if isinstance(freqs, list):
            trainable_freqs = freqs[1]
            freqs = freqs[0]
        freqs = freqs.to(device)
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        if not isinstance(grid_sizes, list):
            grid_sizes = [grid_sizes]

        batch_size = grid_sizes[0][0].shape[0] if isinstance(grid_sizes[0], list) else grid_sizes[0].shape[0]
        precomputed = torch.empty((batch_size, seq_len, 1, c), device=device, dtype=torch.complex64)
        seq_bucket = [0]

        for g in grid_sizes:
            if not isinstance(g, list):
                g = [torch.zeros_like(g), g]
            g_batch_size = g[0].shape[0]
            for i in range(g_batch_size):
                if start is None:
                    f_o, h_o, w_o = g[0][i]
                else:
                    f_o, h_o, w_o = start[i]

                f, h, w = g[1][i]
                t_f, t_h, t_w = g[2][i]
                seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
                seg_len = int(seq_f * seq_h * seq_w)
                if seg_len > 0:
                    if t_f > 0:
                        seq_f_int = int(seq_f)
                        seq_h_int = int(seq_h)
                        seq_w_int = int(seq_w)

                        if f_o >= 0:
                            f_sam = torch.linspace(int(f_o), int(t_f + f_o) - 1, seq_f_int, device=device).long()
                        else:
                            f_sam = torch.linspace(int(-f_o), int(-t_f - f_o) + 1, seq_f_int, device=device).long()
                        h_sam = torch.linspace(int(h_o), int(t_h + h_o) - 1, seq_h_int, device=device).long()
                        w_sam = torch.linspace(int(w_o), int(t_w + w_o) - 1, seq_w_int, device=device).long()

                        freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                        freqs_0 = freqs_0.view(seq_f_int, 1, 1, -1)

                        freqs_i = torch.cat(
                            [
                                freqs_0.expand(seq_f_int, seq_h_int, seq_w_int, -1),
                                freqs[1][h_sam].view(1, seq_h_int, 1, -1).expand(seq_f_int, seq_h_int, seq_w_int, -1),
                                freqs[2][w_sam].view(1, 1, seq_w_int, -1).expand(seq_f_int, seq_h_int, seq_w_int, -1),
                            ],
                            dim=-1,
                        ).reshape(seg_len, 1, -1)
                    elif t_f < 0:
                        freqs_i = trainable_freqs.unsqueeze(1)
                    precomputed[i, seq_bucket[-1] : seq_bucket[-1] + seg_len] = freqs_i
            seq_bucket.append(seq_bucket[-1] + seg_len)
        return precomputed

    @staticmethod
    @torch.amp.autocast(current_omni_platform.device_type, enabled=False)
    def apply_precomputed(x: torch.Tensor, precomputed_freqs: torch.Tensor) -> torch.Tensor:
        """Apply precomputed position frequencies to input tensor."""
        n = x.size(2)
        input_dtype = x.dtype
        seq_len = x.size(1)
        precomputed_freqs = precomputed_freqs[:, :seq_len]
        x_c = torch.view_as_complex(x.to(torch.float64).reshape(x.size(0), seq_len, n, -1, 2))
        x_c = torch.view_as_real(x_c * precomputed_freqs).flatten(3)
        return x_c.reshape(x.size(0), seq_len, n, -1).to(input_dtype)

    @staticmethod
    @torch.amp.autocast(current_omni_platform.device_type, enabled=False)
    def forward(x: torch.Tensor, grid_sizes, freqs: torch.Tensor, start=None) -> torch.Tensor:
        n, d = x.size(2), x.size(3)
        precomputed = RotaryEmbeddingS2VGrid.precompute(x.size(1), n, d, grid_sizes, freqs, x.device, start=start)
        return RotaryEmbeddingS2VGrid.apply_precomputed(x, precomputed)
