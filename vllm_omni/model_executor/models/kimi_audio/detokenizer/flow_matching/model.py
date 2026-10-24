# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from MoonshotAI/Kimi-Audio (MIT), revision
# 349251e1d8f4f98d58fda59246381faecd7392e0, kimia_infer/models/detokenizer/.
# See vllm_omni/model_executor/models/kimi_audio/NOTICE for the upstream license.

import math

import torch
import torch.nn as nn

from .dit_block import DiTBlock, FinalLayer


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    interpolation_factor: int = 1,
    max_seq_length: int = 4096,
):
    print(f"using rope base theta = {theta}, interpolation factor = {interpolation_factor}")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # ROPE type-A extension
    # we choose to use interpolation rather than extrapolation for better position encoding
    # for scale purposes, t should be a float tensor
    t = torch.arange(end, device=freqs.device).float()
    scale = 1.0 / float(interpolation_factor)
    t *= scale

    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    # Sometimes, we don't need so many rope emb as seq_len is smaller than max_pos_emb
    # e.g. rope 1M but seqlen 32k, this will cause gpu memory waste
    if max_seq_length < end:
        freqs_cis = freqs_cis[:max_seq_length,].clone()
    return freqs_cis


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
        freqs = (
            torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
            .float()
            .to(device=t.device)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2  # d/2
        emb = math.log(10000) / (half_dim - 1)  # 2*log(10000)/(d-2)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float) * -emb
        )  # -2i/(d-2)*log(10000); i from 0 to (d-2)/2; shape: (d/2, )
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(
            0
        )  # pos/[1000 ** (2i/(d-2))]; shape: (num_embeddings, d/2)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)  # shape: (num_embeddings, d)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = self.make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()  # (B, T, dim)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

    def make_positions(self, tensor, padding_idx):
        """Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class DiTPrefix(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size,
        output_size,
        semantic_vocab_size,
        hidden_size=1024,
        depth=12,
        num_heads=4,
        # mlp related
        mlp_ratio=4.0,
        ffn_type="conv1d_conv1d",
        ffn_gated_glu=True,
        ffn_act_layer="gelu",
        ffn_conv_kernel_size=5,
        # rope
        use_rope=False,
        rope_params={
            "max_position_embeddings": 4096,
            "rope_base": 10000.0,
            "rope_interpolation_factor": 1.0,
        },
        position_embedding_type="sincos",
        max_seq_len=4096,
        prompt_cfg_dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.prompt_cfg_dropout = prompt_cfg_dropout

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.semantic_token_embedding = nn.Embedding(semantic_vocab_size, hidden_size)

        self.input_linear = nn.Linear(input_size, hidden_size)

        # position embedding
        if position_embedding_type == "learnable":
            self.position_embedding = nn.Embedding(max_seq_len + 1, hidden_size)
        elif position_embedding_type == "sincos":
            self.position_embedding = SinusoidalPositionalEmbedding(hidden_size, 0, max_seq_len + 1)
        elif position_embedding_type == "skip":
            self.position_embedding = None
        else:
            raise NotImplementedError(f"Position embedding type: {position_embedding_type} not implemented.")

        self.use_rope = use_rope

        if self.use_rope:
            assert hidden_size % num_heads == 0, (
                "Hidden size must be divisible by num_heads for rope position embedding."
            )
            rope_dim = hidden_size // num_heads

            self.rotary_pos_emb = precompute_freqs_cis(
                rope_dim,
                rope_params["max_position_embeddings"],
                theta=rope_params["rope_base"],
                interpolation_factor=rope_params["rope_interpolation_factor"],
                max_seq_length=max_seq_len,
            )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    ffn_type=ffn_type,
                    ffn_conv_kernel_size=ffn_conv_kernel_size,
                    ffn_gated_glu=ffn_gated_glu,
                    ffn_act_layer=ffn_act_layer,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, output_size)
        self.initialize_weights()

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

    def forward(
        self,
        x,
        position_ids,
        t,
        condition,
        seq_len,
        cu_seqlens,
        cu_maxlen,
        cu_seqlens_k,
        cu_maxlen_k,
        mask,
        incremental_state=None,
        nopadding=True,
    ):
        """
        Forward pass of DiT.
        x: (N, T, C) tensor of inputs (latent representations of speech)
        position_ids: (N, T) tensor of positional indices
        t: (N,) tensor of diffusion timesteps
        condition: (N, T) tensor of semantic tokens
        seq_len: (N,) tensor of sequence lengths
        """

        condition = self.semantic_token_embedding(condition)  # (N, T, D)

        x = self.input_linear(x)

        if self.position_embedding is not None:
            position_emb = self.position_embedding(position_ids)
            x = x + position_emb

        # ROPE
        if self.use_rope:
            bsz, seqlen = position_ids.shape
            if self.rotary_pos_emb.device != position_ids.device:
                self.rotary_pos_emb = self.rotary_pos_emb.to(position_ids.device)
            rotary_pos_emb = torch.zeros(
                (bsz, seqlen, self.rotary_pos_emb.shape[1]),
                dtype=self.rotary_pos_emb.dtype,
                device=self.rotary_pos_emb.device,
            )
            for b in range(bsz):
                cur_rope = rotary_pos_emb[b]
                cur_position_ids = position_ids[b]
                cur_rope[:] = self.rotary_pos_emb[cur_position_ids]
        else:
            rotary_pos_emb = None

        t = self.t_embedder(t)  # (N, D)
        c = t.unsqueeze(1) + condition  # (N, T, D)

        for block_idx, block in enumerate(self.blocks):
            # x = block(x, c, attn_mask)  # (N, T, D)
            # XXX mask could be None because we always use full mask

            if incremental_state is not None:
                if block_idx not in incremental_state:
                    incremental_state[block_idx] = {}
                incr = incremental_state[block_idx]
            else:
                incr = None

            x = block(
                x=x,
                c=c,
                seq_len=seq_len,
                cu_seqlens=cu_seqlens,
                cu_maxlen=cu_maxlen,
                cu_seqlens_k=cu_seqlens_k,
                cu_maxlen_k=cu_maxlen_k,
                mask=mask,
                rotary_pos_emb=rotary_pos_emb,
                incremental_state=incr,
                nopadding=nopadding,
            )

        x = self.final_layer(x, c)  # (N, T, C)
        return x
