# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, repeat

"""
DiT-v5
- Add convolution in DiTBlock to increase high-freq component
"""


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.proj = nn.Linear(self.inner_dim, dim)

    def to_heads(self, ts: torch.Tensor):
        b, t, c = ts.shape
        # (b, t, nh, c)
        ts = ts.reshape(b, t, self.num_heads, c // self.num_heads)
        ts = ts.transpose(1, 2)
        return ts

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Args:
        x(torch.Tensor): shape (b, t, c)
        attn_mask(torch.Tensor): shape (b, t, t)
        """
        b, t, c = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.to_heads(q)  # (b, nh, t, c)
        k = self.to_heads(k)
        v = self.to_heads(v)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )  # (b, nh, t, c)
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_chunk(self, x: torch.Tensor, att_cache: torch.Tensor = None, attn_mask: torch.Tensor = None):
        """
        Args:
            x: shape (b, dt, c)
            att_cache: shape (b, nh, t, c*2)
        """
        b, t, c = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.to_heads(q)  # (b, nh, t, c)
        k = self.to_heads(k)
        v = self.to_heads(v)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # unpack {k,v}_cache
        if att_cache is not None:
            if attn_mask is not None:
                k_cache, v_cache = att_cache.chunk(2, dim=3)
                k = torch.cat([k, k_cache], dim=2)
                v = torch.cat([v, v_cache], dim=2)

            else:
                k_cache, v_cache = att_cache.chunk(2, dim=3)
                k = torch.cat([k, k_cache], dim=2)
                v = torch.cat([v, v_cache], dim=2)

        # new {k,v}_cache
        new_att_cache = torch.cat([k, v], dim=3)
        # attn_mask = torch.ones((b, 1, t, t1), dtype=torch.bool, device=x.device)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (b, nh, t, c)
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, new_att_cache


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # from SinusoidalPosEmb
        self.scale = 1000

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half).to(t)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * self.scale, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# Convolution related
class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size)
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.causal_padding)
        x = super().forward(x)
        return x

    def forward_chunk(self, x: torch.Tensor, cnn_cache: torch.Tensor = None):
        if cnn_cache is None:
            cnn_cache = x.new_zeros((x.shape[0], self.in_channels, self.causal_padding[0]))
        x = torch.cat([cnn_cache, x], dim=2)
        new_cnn_cache = x[..., -self.causal_padding[0] :]
        x = super().forward(x)
        return x, new_cnn_cache


class CausalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.block = torch.nn.Sequential(
            # norm
            # conv1
            Transpose(1, 2),
            CausalConv1d(in_channels, out_channels, kernel_size),
            Transpose(1, 2),
            # norm & act
            nn.LayerNorm(out_channels),
            nn.Mish(),
            # conv2
            Transpose(1, 2),
            CausalConv1d(out_channels, out_channels, kernel_size),
            Transpose(1, 2),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: shape (b, t, c)
            mask: shape (b, t, 1)
        """
        if mask is not None:
            x = x * mask
        x = self.block(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_chunk(self, x: torch.Tensor, cnn_cache: torch.Tensor = None):
        """
        Args:
            x: shape (b, dt, c)
            cnn_cache: shape (b, c1+c2, 2)
        """
        if cnn_cache is not None:
            cnn_cache1, cnn_cache2 = cnn_cache.split((self.in_channels, self.out_channels), dim=1)
        else:
            cnn_cache1, cnn_cache2 = None, None
        x = self.block[0](x)
        x, new_cnn_cache1 = self.block[1].forward_chunk(x, cnn_cache1)
        x = self.block[2:6](x)
        x, new_cnn_cache2 = self.block[6].forward_chunk(x, cnn_cache2)
        x = self.block[7](x)
        new_cnn_cache = torch.cat((new_cnn_cache1, new_cnn_cache2), dim=1)
        return x, new_cnn_cache


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, head_dim, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, qk_norm=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = MLP(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.conv = CausalConvBlock(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor):
        """Args
        x: shape (b, t, c)
        c: shape (b, 1, c)
        attn_mask: shape (b, t, t), bool type attention mask
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv = (
            self.adaLN_modulation(c).chunk(9, dim=-1)
        )
        # attention
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)
        # conv
        x = x + gate_conv * self.conv(modulate(self.norm3(x), shift_conv, scale_conv))
        # mlp
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward_chunk(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            x: shape (b, dt, c)
            c: shape (b, 1, c)
            cnn_cache: shape (b, c1+c2, 2)
            att_cache: shape (b, nh, t, c * 2)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv = (
            self.adaLN_modulation(c).chunk(9, dim=-1)
        )
        # attention
        x_att, new_att_cache = self.attn.forward_chunk(modulate(self.norm1(x), shift_msa, scale_msa), att_cache, mask)
        x = x + gate_msa * x_att
        # conv
        x_conv, new_cnn_cache = self.conv.forward_chunk(modulate(self.norm3(x), shift_conv, scale_conv), cnn_cache)
        x = x + gate_conv * x_conv
        # mlp
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, new_cnn_cache, new_att_cache


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        depth: int = 28,
        num_heads: int = 8,
        head_dim: int = 64,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.in_proj = nn.Linear(in_channels, hidden_size)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, head_dim, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

        self.initialize_weights()

        self.enable_cuda_graph = False
        self.use_cuda_graph = False

        self.graph_chunk = {}
        self.inference_buffers_chunk = {}
        self.max_size_chunk = {}

        self.register_buffer("att_cache_buffer", torch.zeros((16, 2, 8, 1000, 128)), persistent=False)
        self.register_buffer("cnn_cache_buffer", torch.zeros((16, 2, 1024, 2)), persistent=False)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _init_cuda_graph_chunk(self):
        # get dtype, device from registered buffer
        dtype, device = self.cnn_cache_buffer.dtype, self.cnn_cache_buffer.device
        # init cuda graph for streaming forward
        with torch.no_grad():
            for chunk_size in [30, 48, 96]:
                if chunk_size == 30 or chunk_size == 48:
                    max_size = 500
                    self.max_size_chunk[chunk_size] = max_size
                else:
                    max_size = 1000
                    self.max_size_chunk[chunk_size] = max_size
                static_x1 = torch.zeros((2, 320, chunk_size), dtype=dtype, device=device)
                static_t1 = torch.zeros((2, 1, 512), dtype=dtype, device=device)
                static_mask1 = torch.ones((2, chunk_size, max_size + chunk_size), dtype=torch.bool, device=device)
                static_att_cache = torch.zeros((16, 2, 8, max_size, 128), dtype=dtype, device=device)
                static_cnn_cache = torch.zeros((16, 2, 1024, 2), dtype=dtype, device=device)
                static_inputs1 = [
                    static_x1,
                    static_t1,
                    static_mask1,
                    static_cnn_cache,
                    static_att_cache,
                ]
                static_new_cnn_cache = torch.zeros((16, 2, 1024, 2), dtype=dtype, device=device)
                static_new_att_cache = torch.zeros((16, 2, 8, max_size + chunk_size, 128), dtype=dtype, device=device)
                self.blocks_forward_chunk(
                    static_inputs1[0],
                    static_inputs1[1],
                    static_inputs1[2],
                    static_inputs1[3],
                    static_inputs1[4],
                    static_new_cnn_cache,
                    static_new_att_cache,
                )
                graph_chunk = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_chunk):
                    static_out1 = self.blocks_forward_chunk(
                        static_x1,
                        static_t1,
                        static_mask1,
                        static_cnn_cache,
                        static_att_cache,
                        static_new_cnn_cache,
                        static_new_att_cache,
                    )
                static_outputs1 = [static_out1, static_new_cnn_cache, static_new_att_cache]
                self.inference_buffers_chunk[chunk_size] = {
                    "static_inputs": static_inputs1,
                    "static_outputs": static_outputs1,
                }
                self.graph_chunk[chunk_size] = graph_chunk

    def _init_cuda_graph_all(self):
        self._init_cuda_graph_chunk()
        self.use_cuda_graph = True
        print("CUDA Graph initialized successfully for chunk decoder")

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """Args:
        x: shape (b, c, t)
        mask: shape (b, 1, t)
        t: shape (b,)
        spks: shape (b, c)
        cond: shape (b, c, t)
        """
        # (sfy) chunk training strategy should not be open-sourced

        # time
        t = self.t_embedder(t).unsqueeze(1)  # (b, 1, c)
        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        return self.blocks_forward(x, t, mask)

    def blocks_forward(self, x, t, mask):
        x = x.transpose(1, 2)
        attn_mask = mask.bool()
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, t, attn_mask)
        x = self.final_layer(x, t)
        x = x.transpose(1, 2)
        return x

    def forward_chunk(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
    ):
        """
        Args:
            x: shape (b, dt, c)
            mu: shape (b, dt, c)
            t: shape (b,)
            spks: shape (b, c)
            cond: shape (b, dt, c)
            cnn_cache: shape (depth, b, c1+c2, 2)
            att_cache: shape (depth, b, nh, t, c * 2)
        """

        # time
        t = self.t_embedder(t).unsqueeze(1)  # (b, 1, c)
        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        # create fake cache
        if cnn_cache is None:
            cnn_cache = [None] * len(self.blocks)
        if att_cache is None:
            att_cache = [None] * len(self.blocks)
        if att_cache[0] is not None:
            last_att_len = att_cache.shape[3]
        else:
            last_att_len = 0
        chunk_size = x.shape[2]
        mask = torch.ones(x.shape[0], chunk_size, last_att_len + chunk_size, dtype=torch.bool, device=x.device)
        if (
            self.use_cuda_graph
            and att_cache[0] is not None
            and chunk_size in self.graph_chunk
            and last_att_len <= self.max_size_chunk[chunk_size]
        ):
            padded_mask = torch.zeros(
                (2, chunk_size, self.max_size_chunk[chunk_size] + chunk_size), dtype=mask.dtype, device=mask.device
            )
            padded_mask[:, :, : mask.shape[-1]] = mask
            padded_att_cache = torch.zeros(
                (16, 2, 8, self.max_size_chunk[chunk_size], 128), dtype=att_cache.dtype, device=att_cache.device
            )
            padded_att_cache[:, :, :, :last_att_len, :] = att_cache
            self.inference_buffers_chunk[chunk_size]["static_inputs"][0].copy_(x)
            self.inference_buffers_chunk[chunk_size]["static_inputs"][1].copy_(t)
            self.inference_buffers_chunk[chunk_size]["static_inputs"][2].copy_(padded_mask)
            self.inference_buffers_chunk[chunk_size]["static_inputs"][3].copy_(cnn_cache)
            self.inference_buffers_chunk[chunk_size]["static_inputs"][4].copy_(padded_att_cache)
            self.graph_chunk[chunk_size].replay()
            x = self.inference_buffers_chunk[chunk_size]["static_outputs"][0][:, :, :chunk_size]
            new_cnn_cache = self.inference_buffers_chunk[chunk_size]["static_outputs"][1]
            new_att_cache = self.inference_buffers_chunk[chunk_size]["static_outputs"][2][
                :, :, :, : chunk_size + last_att_len, :
            ]
        else:
            mask = None
            x = self.blocks_forward_chunk(
                x, t, mask, cnn_cache, att_cache, self.cnn_cache_buffer, self.att_cache_buffer
            )
            new_cnn_cache = self.cnn_cache_buffer
            new_att_cache = self.att_cache_buffer[:, :, :, : last_att_len + chunk_size, :]

        return x, new_cnn_cache, new_att_cache

    def blocks_forward_chunk(
        self, x, t, mask, cnn_cache=None, att_cache=None, cnn_cache_buffer=None, att_cache_buffer=None
    ):
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        for b_idx, block in enumerate(self.blocks):
            x, this_new_cnn_cache, this_new_att_cache = block.forward_chunk(
                x, t, cnn_cache[b_idx], att_cache[b_idx], mask
            )
            cnn_cache_buffer[b_idx] = this_new_cnn_cache
            att_cache_buffer[b_idx][:, :, : this_new_att_cache.shape[2], :] = this_new_att_cache
        x = self.final_layer(x, t)
        x = x.transpose(1, 2)
        return x
