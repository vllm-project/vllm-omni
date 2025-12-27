# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from einops import repeat
from torch import nn


def apply_real_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """
    Apply real-valued rotary embeddings to input tensor.

    Args:
        x: Input tensor of shape [..., seq_len, num_heads, dim] or [..., seq_len, dim]
        freqs_cos: Cosine frequencies of shape [batch, seq_len, dim] or [seq_len, dim]
        freqs_sin: Sine frequencies of shape [batch, seq_len, dim] or [seq_len, dim]

    Returns:
        Tensor with rotary embeddings applied
    """
    # x: [batch, seq_len, num_heads, dim] or [batch, seq_len, dim]
    # freqs_cos: [batch, seq_len, dim] or [seq_len, dim]
    # freqs_sin: [batch, seq_len, dim] or [seq_len, dim]

    x_shape = x.shape
    if len(x_shape) == 4:
        batch, seq_len, num_heads, dim = x_shape
        x_reshaped = x.view(batch, seq_len, num_heads, dim // 2, 2)
    elif len(x_shape) == 3:
        batch, seq_len, dim = x_shape
        num_heads = None
        x_reshaped = x.view(batch, seq_len, dim // 2, 2)
    else:
        raise ValueError(f"Unsupported x shape: {x.shape}")

    # freqs_cos/sin: [batch, seq_len, dim] or [seq_len, dim]
    # Expand freqs_cos/sin to [batch, seq_len, dim] if needed
    if freqs_cos.dim() == 2:
        # [seq_len, dim] -> [1, seq_len, dim]
        freqs_cos = freqs_cos.unsqueeze(0)
        freqs_sin = freqs_sin.unsqueeze(0)
    if freqs_cos.shape[0] == 1 and batch > 1:
        freqs_cos = freqs_cos.expand(batch, -1, -1)
        freqs_sin = freqs_sin.expand(batch, -1, -1)

    # Reshape freqs to [batch, seq_len, dim//2, 2]
    freqs_cos_reshaped = freqs_cos.view(batch, seq_len, dim // 2, 2)
    freqs_sin_reshaped = freqs_sin.view(batch, seq_len, dim // 2, 2)

    cos_1 = freqs_cos_reshaped[..., 0]  # [batch, seq_len, dim//2]
    cos_2 = freqs_cos_reshaped[..., 1]  # [batch, seq_len, dim//2]
    sin_1 = freqs_sin_reshaped[..., 0]  # [batch, seq_len, dim//2]
    sin_2 = freqs_sin_reshaped[..., 1]  # [batch, seq_len, dim//2]

    # Broadcast cos/sin to match x_reshaped
    if len(x_shape) == 4:
        # [batch, seq_len, 1, dim//2]
        cos_1 = cos_1.unsqueeze(2)
        cos_2 = cos_2.unsqueeze(2)
        sin_1 = sin_1.unsqueeze(2)
        sin_2 = sin_2.unsqueeze(2)

    x1 = x_reshaped[..., 0]  # [..., seq_len, num_heads, dim//2] or [..., seq_len, dim//2]
    x2 = x_reshaped[..., 1]  # same

    out1 = x1 * cos_1 - x2 * sin_1
    out2 = x1 * sin_2 + x2 * cos_2

    out = torch.stack([out1, out2], dim=-1)
    return out.view(*x_shape)


def get_1d_rotary_pos_embed_real(
    dim: int,
    pos: np.ndarray | int,
    theta: float = 10000.0,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    freqs_dtype: torch.dtype = torch.float32,
):
    """
    Precompute the frequency tensor for real-valued rotary embeddings.

    This function calculates frequency tensors using cosine and sine functions, avoiding complex numbers entirely.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `Tuple[torch.Tensor, torch.Tensor]`: Precomputed cosine and sine frequency tensors. [S, D]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore # [S, D/2]

    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()

    # Create interleaved cos and sin patterns for real rotation
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]

    return freqs_cos, freqs_sin


def complex_to_real_equivalent(freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert complex frequency embeddings to equivalent real cos/sin representation.

    Args:
        freqs_cis: Complex frequency tensor of shape [S, D/2]

    Returns:
        Tuple of (cos, sin) tensors of shape [S, D]
    """
    # Extract real and imaginary parts
    cos_part = freqs_cis.real  # [S, D/2]
    sin_part = freqs_cis.imag  # [S, D/2]

    # Create interleaved cos and sin patterns
    freqs_cos = cos_part.repeat_interleave(2, dim=1, output_size=cos_part.shape[1] * 2).float()  # [S, D]
    freqs_sin = sin_part.repeat_interleave(2, dim=1, output_size=sin_part.shape[1] * 2).float()  # [S, D]

    return freqs_cos, freqs_sin


class RotaryPosEmbedReal(nn.Module):
    def __init__(
        self, theta: int, axes_dim: tuple[int, int, int], axes_lens: tuple[int, int, int], patch_size: int = 2
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

    @staticmethod
    def get_freqs_real(
        axes_dim: tuple[int, int, int], axes_lens: tuple[int, int, int], theta: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        freqs_real = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
            cos_emb, sin_emb = get_1d_rotary_pos_embed_real(d, e, theta=theta, freqs_dtype=freqs_dtype)
            freqs_real.append((cos_emb, sin_emb))
        return freqs_real

    def _get_freqs_real(
        self, freqs_real: list[tuple[torch.Tensor, torch.Tensor]], ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        cos_result = []
        sin_result = []
        for i in range(len(self.axes_dim)):
            freqs_cos, freqs_sin = freqs_real[i]
            freqs_cos = freqs_cos.to(ids.device)
            freqs_sin = freqs_sin.to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs_cos.shape[-1]).to(torch.int64)
            cos_result.append(torch.gather(freqs_cos.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
            sin_result.append(torch.gather(freqs_sin.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))

        combined_cos = torch.cat(cos_result, dim=-1).to(device)
        combined_sin = torch.cat(sin_result, dim=-1).to(device)
        return combined_cos, combined_sin

    def forward(
        self, freqs_real, attention_mask, l_effective_ref_img_len, l_effective_img_len, ref_img_sizes, img_sizes, device
    ):
        batch_size = len(attention_mask)
        p = self.patch_size

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()
        l_effective_cap_len = [int(len) for len in l_effective_cap_len]
        seq_lengths = [
            int(cap_len + sum(ref_img_len) + img_len)
            for cap_len, ref_img_len, img_len in zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len)
        ]

        max_seq_len = max(seq_lengths)
        max_ref_img_len = max([int(sum(ref_img_len)) for ref_img_len in l_effective_ref_img_len])
        max_img_len = int(max(l_effective_img_len))

        # Create position IDs
        position_ids = torch.zeros(batch_size, int(max_seq_len), 3, dtype=torch.int32, device=device)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            # add text position ids
            position_ids[i, :cap_seq_len] = repeat(
                torch.arange(cap_seq_len, dtype=torch.int32, device=device), "l -> l 3"
            )
            pe_shift = cap_seq_len
            pe_shift_len = cap_seq_len

            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
                    H, W = ref_img_size
                    ref_H_tokens, ref_W_tokens = H // p, W // p
                    assert ref_H_tokens * ref_W_tokens == ref_img_len
                    # add image position ids

                    row_ids = repeat(
                        torch.arange(ref_H_tokens, dtype=torch.int32, device=device), "h -> h w", w=ref_W_tokens
                    ).flatten()
                    col_ids = repeat(
                        torch.arange(ref_W_tokens, dtype=torch.int32, device=device), "w -> h w", h=ref_H_tokens
                    ).flatten()
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 0] = pe_shift
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 1] = row_ids
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 2] = col_ids

                    pe_shift += max(ref_H_tokens, ref_W_tokens)
                    pe_shift_len += ref_img_len

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // p, W // p
            assert H_tokens * W_tokens == l_effective_img_len[i]

            row_ids = repeat(torch.arange(H_tokens, dtype=torch.int32, device=device), "h -> h w", w=W_tokens).flatten()
            col_ids = repeat(torch.arange(W_tokens, dtype=torch.int32, device=device), "w -> h w", h=H_tokens).flatten()

            assert pe_shift_len + l_effective_img_len[i] == seq_len
            position_ids[i, pe_shift_len:seq_len, 0] = pe_shift
            position_ids[i, pe_shift_len:seq_len, 1] = row_ids
            position_ids[i, pe_shift_len:seq_len, 2] = col_ids

        # Get combined rotary embeddings (real version)
        freqs_cos, freqs_sin = self._get_freqs_real(freqs_real, position_ids)

        # create separate rotary embeddings for captions and images
        cap_freqs_cos = torch.zeros(
            batch_size, encoder_seq_len, freqs_cos.shape[-1], device=device, dtype=freqs_cos.dtype
        )
        cap_freqs_sin = torch.zeros(
            batch_size, encoder_seq_len, freqs_sin.shape[-1], device=device, dtype=freqs_sin.dtype
        )
        ref_img_freqs_cos = torch.zeros(
            batch_size, max_ref_img_len, freqs_cos.shape[-1], device=device, dtype=freqs_cos.dtype
        )
        ref_img_freqs_sin = torch.zeros(
            batch_size, max_ref_img_len, freqs_sin.shape[-1], device=device, dtype=freqs_sin.dtype
        )
        img_freqs_cos = torch.zeros(batch_size, max_img_len, freqs_cos.shape[-1], device=device, dtype=freqs_cos.dtype)
        img_freqs_sin = torch.zeros(batch_size, max_img_len, freqs_sin.shape[-1], device=device, dtype=freqs_sin.dtype)

        for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(
            zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len, seq_lengths)
        ):
            cap_freqs_cos[i, :cap_seq_len] = freqs_cos[i, :cap_seq_len]
            cap_freqs_sin[i, :cap_seq_len] = freqs_sin[i, :cap_seq_len]
            ref_img_freqs_cos[i, : sum(ref_img_len)] = freqs_cos[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            ref_img_freqs_sin[i, : sum(ref_img_len)] = freqs_sin[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            img_freqs_cos[i, :img_len] = freqs_cos[
                i, cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len
            ]
            img_freqs_sin[i, :img_len] = freqs_sin[
                i, cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len
            ]

        return (
            (cap_freqs_cos, cap_freqs_sin),
            (ref_img_freqs_cos, ref_img_freqs_sin),
            (img_freqs_cos, img_freqs_sin),
            (freqs_cos, freqs_sin),
            l_effective_cap_len,
            seq_lengths,
        )


# For backward compatibility - wrapper that mimics the original interface
class RotaryPosEmbed(RotaryPosEmbedReal):
    """Backward compatible wrapper that returns the same interface as the original complex version"""

    @staticmethod
    def get_freqs_cis(
        axes_dim: tuple[int, int, int], axes_lens: tuple[int, int, int], theta: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return RotaryPosEmbedReal.get_freqs_real(axes_dim, axes_lens, theta)

    def forward(
        self,
        freqs_cis,  # This will actually be freqs_real (cos, sin tuples)
        attention_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
        device,
    ):
        # Call the real version implementation
        return super().forward(
            freqs_cis,  # freqs_real
            attention_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )


# Original function for compatibility with mixed approach
def get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray | int,
    theta: float = 10000.0,
    use_real: bool = True,  # Default to real version
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float32,
):
    """
    Enhanced version that defaults to real computation but can still produce complex if needed.
    """
    if use_real:
        return get_1d_rotary_pos_embed_real(dim, pos, theta, linear_factor, ntk_factor, freqs_dtype)
    else:
        # Fallback to complex version for compatibility
        assert dim % 2 == 0

        if isinstance(pos, int):
            pos = torch.arange(pos)
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)

        theta = theta * ntk_factor
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
        freqs = torch.outer(pos, freqs)

        is_npu = freqs.device.type == "npu"
        if is_npu:
            freqs = freqs.float()

        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
