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


import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_flash_attention_utils import (
    is_flash_attn_available,
    is_torch_npu_available,
)

from ..mammothmoda2_qwen2_5_vl.modeling_mammothmoda2_qwen2_5_vl import Qwen2RMSNorm


def _swiglu(x, y):
    return F.silu(x.float(), inplace=False).to(x.dtype) * y


if is_flash_attn_available() and not is_torch_npu_available():
    from flash_attn.ops.activations import swiglu
else:
    swiglu = _swiglu


class LuminaFeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        hidden_size (`int`):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        intermediate_size (`int`): The intermediate dimension of the feedforward layer.
        multiple_of (`int`, *optional*): Value to ensure hidden dimension is a multiple
            of this value.
        ffn_dim_multiplier (float, *optional*): Custom multiplier for hidden
            dimension. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: int | None = 256,
        ffn_dim_multiplier: float | None = None,
    ):
        super().__init__()
        self.swiglu = swiglu

        # custom hidden_size factor multiplier
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.linear_1 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            inner_dim,
            dim,
            bias=False,
        )
        self.linear_3 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )

    def forward(self, x):
        h1, h2 = self.linear_1(x), self.linear_3(x)
        return self.linear_2(self.swiglu(h1, h2))


class SimpleQFormerImageRefiner(nn.Module):
    """
    A lightweight Q-Former-like module that maps a variable-length sequence of
    input features to a fixed number of query tokens (default 128).
    Inputs are first projected to `hidden_size`, then a stack of decoder blocks
    performs self-attention over learnable queries and cross-attention to inputs.
    Output shape: (batch, num_queries, hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 128,
        num_layers: int = 2,
        num_heads: int | None = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        # ensure num_heads divides hidden_size
        if num_heads is None:
            num_heads = max(1, hidden_size // 128)
        self.num_heads = self._choose_valid_num_heads(hidden_size, num_heads)
        self.input_proj = nn.Sequential(
            Qwen2RMSNorm(hidden_size, eps=norm_eps),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # Learnable query embeddings
        scale = hidden_size**-0.5
        self.query = nn.Parameter(scale * torch.randn(1, num_queries, hidden_size))

        # Decoder layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    dict(
                        ln_q1=Qwen2RMSNorm(hidden_size, eps=norm_eps),
                        self_attn=nn.MultiheadAttention(
                            embed_dim=hidden_size, num_heads=self.num_heads, dropout=dropout, batch_first=True
                        ),
                        ln_q2=Qwen2RMSNorm(hidden_size, eps=norm_eps),
                        cross_attn=nn.MultiheadAttention(
                            embed_dim=hidden_size, num_heads=self.num_heads, dropout=dropout, batch_first=True
                        ),
                        ln_ffn=Qwen2RMSNorm(hidden_size, eps=norm_eps),
                        ffn=LuminaFeedForward(dim=hidden_size, inner_dim=4 * hidden_size),
                    )
                )
            )

    @staticmethod
    def _choose_valid_num_heads(hidden_size: int, proposed_heads: int, preferred_head_dim: int = 128) -> int:
        """Pick a number of heads that divides hidden_size, close to proposed or preferred."""
        # If proposed is valid, use it
        if proposed_heads > 0 and hidden_size % proposed_heads == 0:
            return proposed_heads
        # target based on preferred head dim
        target = max(1, round(hidden_size / preferred_head_dim))
        # collect divisors up to 128 heads (more than enough)
        max_heads_cap = min(128, hidden_size)
        divisors = [d for d in range(1, max_heads_cap + 1) if hidden_size % d == 0]
        # choose closest to target
        best = min(divisors, key=lambda d: (abs(d - target), -d))
        return best

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            Tensor of shape (batch, num_queries, hidden_size)
        """
        batch, _, _ = x.shape
        kv = self.input_proj(x)
        q = self.query.repeat(batch, 1, 1).to(kv.dtype)

        for layer in self.layers:
            # Self-attention on queries
            q_norm = layer["ln_q1"](q)
            attn_out, _ = layer["self_attn"](q_norm, q_norm, q_norm, need_weights=False)
            q = q + attn_out

            # Cross-attention: queries attend to inputs
            q_norm = layer["ln_q2"](q)
            cross_out, _ = layer["cross_attn"](q_norm, kv, kv, need_weights=False, key_padding_mask=attention_mask)
            q = q + cross_out

            # Feed-forward
            q = q + layer["ffn"](layer["ln_ffn"](q))

        return q
