# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
from torch import nn
from torch.nn.utils import weight_norm

from vllm_omni.model_executor.models.indextts2.s2mel.modules.commons import sequence_mask
from vllm_omni.model_executor.models.indextts2.s2mel.modules.gpt_fast.model import ModelArgs, Transformer
from vllm_omni.model_executor.models.indextts2.s2mel.modules.wavenet import WN


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


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
        self.max_period = 10000
        self.scale = 1000

        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        args = self.scale * t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.time_as_token = getattr(args.DiT, "time_as_token", False)
        self.style_as_token = getattr(args.DiT, "style_as_token", False)
        self.uvit_skip_connection = getattr(args.DiT, "uvit_skip_connection", False)
        model_args = ModelArgs(
            block_size=16384,  # args.DiT.block_size,
            n_layer=args.DiT.depth,
            n_head=args.DiT.num_heads,
            dim=args.DiT.hidden_dim,
            head_dim=args.DiT.hidden_dim // args.DiT.num_heads,
            vocab_size=1024,
            uvit_skip_connection=self.uvit_skip_connection,
            time_as_token=self.time_as_token,
        )
        self.transformer = Transformer(model_args)
        self.in_channels = args.DiT.in_channels
        self.out_channels = args.DiT.in_channels
        self.num_heads = args.DiT.num_heads

        self.x_embedder = weight_norm(nn.Linear(args.DiT.in_channels, args.DiT.hidden_dim, bias=True))

        self.content_type = args.DiT.content_type  # 'discrete' or 'continuous'
        self.content_codebook_size = args.DiT.content_codebook_size  # for discrete content
        self.content_dim = args.DiT.content_dim  # for continuous content
        self.cond_projection = nn.Linear(args.DiT.content_dim, args.DiT.hidden_dim, bias=True)  # continuous content

        self.is_causal = args.DiT.is_causal

        self.t_embedder = TimestepEmbedder(args.DiT.hidden_dim)

        input_pos = torch.arange(16384)
        self.register_buffer("input_pos", input_pos)

        self.final_layer_type = args.DiT.final_layer_type  # mlp or wavenet
        if self.final_layer_type == "wavenet":
            self.t_embedder2 = TimestepEmbedder(args.wavenet.hidden_dim)
            self.conv1 = nn.Linear(args.DiT.hidden_dim, args.wavenet.hidden_dim)
            self.conv2 = nn.Conv1d(args.wavenet.hidden_dim, args.DiT.in_channels, 1)
            self.wavenet = WN(
                hidden_channels=args.wavenet.hidden_dim,
                kernel_size=args.wavenet.kernel_size,
                dilation_rate=args.wavenet.dilation_rate,
                n_layers=args.wavenet.num_layers,
                gin_channels=args.wavenet.hidden_dim,
                p_dropout=args.wavenet.p_dropout,
                causal=False,
            )
            self.final_layer = FinalLayer(args.wavenet.hidden_dim, 1, args.wavenet.hidden_dim)
            self.res_projection = nn.Linear(
                args.DiT.hidden_dim, args.wavenet.hidden_dim
            )  # residual connection from transformer output to final output
            self.wavenet_style_condition = args.wavenet.style_condition
            assert args.DiT.style_condition == args.wavenet.style_condition
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(args.DiT.hidden_dim, args.DiT.hidden_dim),
                nn.SiLU(),
                nn.Linear(args.DiT.hidden_dim, args.DiT.in_channels),
            )
        self.transformer_style_condition = args.DiT.style_condition

        self.class_dropout_prob = args.DiT.class_dropout_prob
        self.content_mask_embedder = nn.Embedding(1, args.DiT.hidden_dim)

        self.long_skip_connection = args.DiT.long_skip_connection
        self.skip_linear = nn.Linear(args.DiT.hidden_dim + args.DiT.in_channels, args.DiT.hidden_dim)

        self.cond_x_merge_linear = nn.Linear(
            args.DiT.hidden_dim
            + args.DiT.in_channels * 2
            + args.style_encoder.dim * self.transformer_style_condition * (not self.style_as_token),
            args.DiT.hidden_dim,
        )
        if self.style_as_token:
            self.style_in = nn.Linear(args.style_encoder.dim, args.DiT.hidden_dim)

    def setup_caches(self, max_batch_size, max_seq_length):
        self.transformer.setup_caches(max_batch_size, max_seq_length, use_kv_cache=False)

    def forward(self, x, prompt_x, x_lens, t, style, cond, mask_content=False, pre_mask=None):
        """
        x (torch.Tensor): random noise
        prompt_x (torch.Tensor): reference mel + zero mel
            shape: (batch_size, 80, 795+1068)
        x_lens (torch.Tensor): mel frames output
            shape: (batch_size, mel_timesteps)
        t (torch.Tensor): radshape:
            shape: (batch_size)
        style (torch.Tensor): reference global style
            shape: (batch_size, 192)
        cond (torch.Tensor): semantic info of reference audio and altered audio
            shape: (batch_size, mel_timesteps(795+1069), 512)
        pre_mask (torch.Tensor | None): precomputed (x_mask, x_mask_expanded) to
            avoid recomputation across ODE steps. Tuple of (x_mask, x_mask_expanded).
        """
        class_dropout = False
        if self.training and torch.rand(1) < self.class_dropout_prob:
            class_dropout = True
        if not self.training and mask_content:
            class_dropout = True
        cond_in_module = self.cond_projection

        B, _, T = x.size()

        t1 = self.t_embedder(t)  # (N, D) # t1 [2, 512]
        cond = cond_in_module(cond)  # cond [2,1863,512]->[2,1863,512]

        x = x.transpose(1, 2)  # [2,1863,80]
        prompt_x = prompt_x.transpose(1, 2)  # [2,1863,80]

        x_in = torch.cat([x, prompt_x, cond], dim=-1)  # 80+80+512=672 [2, 1863, 672]

        if self.transformer_style_condition and not self.style_as_token:  # True and True
            x_in = torch.cat([x_in, style[:, None, :].expand(-1, T, -1)], dim=-1)  # [2, 1863, 864]

        if class_dropout:  # False
            x_in[..., self.in_channels :] = x_in[..., self.in_channels :] * 0  # 80维后全置为0

        x_in = self.cond_x_merge_linear(x_in)  # (N, T, D) [2, 1863, 512]

        if self.style_as_token:  # False
            style = self.style_in(style)
            style = torch.zeros_like(style) if class_dropout else style
            x_in = torch.cat([style.unsqueeze(1), x_in], dim=1)

        if self.time_as_token:  # False
            x_in = torch.cat([t1.unsqueeze(1), x_in], dim=1)

        if pre_mask is not None:
            x_mask, x_mask_expanded = pre_mask
        else:
            x_mask = (
                sequence_mask(x_lens + self.style_as_token + self.time_as_token, max_length=x_in.size(1))
                .to(x.device)
                .unsqueeze(1)
            )
            input_pos = self.input_pos[: x_in.size(1)]
            x_mask_expanded = x_mask[:, None, :].expand(-1, -1, x_in.size(1), -1) if not self.is_causal else None

        input_pos = self.input_pos[: x_in.size(1)]  # (T,) range（0，1863）
        # Route through per-shape CUDA graph runner if enabled.
        graph_runner = getattr(self, "_cuda_graph_runner", None)
        if graph_runner is not None:
            x_res = graph_runner(x_in, t1.unsqueeze(1), input_pos, x_mask_expanded)
        else:
            x_res = self.transformer(x_in, t1.unsqueeze(1), input_pos, x_mask_expanded)  # [2, 1863, 512]
        x_res = x_res[:, 1:] if self.time_as_token else x_res
        x_res = x_res[:, 1:] if self.style_as_token else x_res

        if self.long_skip_connection:  # True
            x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))
        if self.final_layer_type == "wavenet":
            x = self.conv1(x_res)
            x = x.transpose(1, 2)
            t2 = self.t_embedder2(t)
            x = self.wavenet(x, x_mask, g=t2.unsqueeze(2)).transpose(1, 2) + self.res_projection(
                x_res
            )  # long residual connection
            x = self.final_layer(x, t1).transpose(1, 2)
            x = self.conv2(x)
        else:
            x = self.final_mlp(x_res)
            x = x.transpose(1, 2)
        # x [2,80,1863]
        return x
