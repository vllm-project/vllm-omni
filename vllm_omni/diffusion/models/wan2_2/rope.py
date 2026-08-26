# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wan-specific rotary positional embedding layers."""

import torch

from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rotary_emb_mindiesd
from vllm_omni.platforms import current_omni_platform


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
        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

        if cos.dim() > 2:
            cos = cos.reshape(-1, cos.shape[-1])
            sin = sin.reshape(-1, sin.shape[-1])

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
            return self.forward_native(x, cos, sin)

        if cos.dim() > 2:
            cos = cos.reshape(-1, cos.shape[-1])
            sin = sin.reshape(-1, sin.shape[-1])

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


class WanS2VRotaryPosEmbed(torch.nn.Module):
    """Precompute complex-valued RoPE embeddings for S2V multi-grid positions.

    Owns the base frequency buffer and provides forward() to compute position
    embeddings given hidden_states (for shape) and grid_sizes.
    """

    def __init__(self, num_heads: int, head_dim: int, max_seq_len: int = 1024):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        d = head_dim
        freqs = torch.cat(
            [
                self._rope_params(max_seq_len, d - 4 * (d // 6)),
                self._rope_params(max_seq_len, 2 * (d // 6)),
                self._rope_params(max_seq_len, 2 * (d // 6)),
            ],
            dim=1,
        )
        self.register_buffer("freqs", freqs.to(torch.complex64), persistent=False)

    @staticmethod
    @torch.amp.autocast(current_omni_platform.device_type, enabled=False)
    def _rope_params(max_seq_len, dim, theta=10000):
        if dim % 2 != 0:
            raise ValueError(f"dim ({dim}) must be even")
        freqs = torch.outer(
            torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_sizes: list,
        trainable_freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Precompute RoPE embeddings for the given grid layout.

        Args:
            hidden_states: Tensor [B, S, ...] (used for batch/seq shape and device)
            grid_sizes: Grid specification (list of [offsets, sizes, totals])
            trainable_freqs: Optional trainable frequency overrides for t_f < 0

        Returns:
            Complex tensor [B, S, 1, head_dim//2] of precomputed position embeddings
        """
        b, s = hidden_states.shape[0], hidden_states.shape[1]
        c = self.head_dim // 2
        device = hidden_states.device

        freqs = self.freqs.to(device)
        if trainable_freqs is not None:
            freqs_input = [freqs, trainable_freqs]
        else:
            freqs_input = freqs

        if isinstance(freqs_input, list):
            trainable_f = freqs_input[1]
            freqs_split = freqs_input[0]
        else:
            trainable_f = None
            freqs_split = freqs_input
        freqs_split = freqs_split.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        output = torch.empty((b, s, 1, c), device=device, dtype=torch.complex64)
        seq_bucket = [0]
        if not isinstance(grid_sizes, list):
            grid_sizes = [grid_sizes]
        for g in grid_sizes:
            if not isinstance(g, list):
                g = [torch.zeros_like(g), g]
            batch_size = g[0].shape[0]
            for i in range(batch_size):
                f_o, h_o, w_o = g[0][i]
                f, h, w = g[1][i]
                t_f, t_h, t_w = g[2][i]
                seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
                seq_len = int(seq_f * seq_h * seq_w)
                if seq_len > 0:
                    if t_f > 0:
                        assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                        seq_f_int = int(seq_f)
                        seq_h_int = int(seq_h)
                        seq_w_int = int(seq_w)

                        if f_o >= 0:
                            f_sam = torch.linspace(int(f_o), int(t_f + f_o) - 1, seq_f_int, device=device).long()
                        else:
                            f_sam = torch.linspace(int(-f_o), int(-t_f - f_o) + 1, seq_f_int, device=device).long()
                        h_sam = torch.linspace(int(h_o), int(t_h + h_o) - 1, seq_h_int, device=device).long()
                        w_sam = torch.linspace(int(w_o), int(t_w + w_o) - 1, seq_w_int, device=device).long()

                        freqs_0 = torch.index_select(freqs_split[0] if f_o >= 0 else freqs_split[0].conj(), 0, f_sam)
                        freqs_0 = freqs_0.view(seq_f_int, 1, 1, -1)

                        freqs_i = torch.cat(
                            [
                                freqs_0.expand(seq_f_int, seq_h_int, seq_w_int, -1),
                                torch.index_select(freqs_split[1], 0, h_sam)
                                .view(1, seq_h_int, 1, -1)
                                .expand(seq_f_int, seq_h_int, seq_w_int, -1),
                                torch.index_select(freqs_split[2], 0, w_sam)
                                .view(1, 1, seq_w_int, -1)
                                .expand(seq_f_int, seq_h_int, seq_w_int, -1),
                            ],
                            dim=-1,
                        ).reshape(seq_len, 1, -1)
                    elif t_f < 0:
                        freqs_i = trainable_f.unsqueeze(1)
                    output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = freqs_i
            seq_bucket.append(seq_bucket[-1] + seq_len)
        return output


class RotaryEmbeddingWanS2V(RotaryEmbeddingWan):
    """Apply RoPE using precomputed complex freqs for Wan S2V main transformer.

    Converts complex freqs (from WanS2VRotaryPosEmbed) to cos/sin and delegates
    to RotaryEmbeddingWan for platform-optimized application (float32 kernel).
    Under TP, freqs has 1 head — broadcasts automatically via cos/sin.
    """

    def __init__(self) -> None:
        super().__init__(is_neox_style=False, half_head_dim=True)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        freqs_sliced = freqs[:, : x.size(1)]
        cos = freqs_sliced.real.to(x.dtype)
        sin = freqs_sliced.imag.to(x.dtype)
        return super().forward(x, cos, sin)
