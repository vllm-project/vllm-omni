# Copyright 2025 vLLM-Omni Team
"""Kimi Audio flow-matching DiT (non-streaming, SDPA-based).

Ported from the reference implementation in Kimi-Audio/kimia_infer/models/detokenizer/flow_matching
so that the Stage 1 detokenizer loads real checkpoint weights.  The original
uses flash-attention; this version uses PyTorch's scaled_dot_product_attention
to avoid the flash_attn dependency.

Attention locality: the reference runs the DiT in chunks of 30 semantic tokens
(= 120 mel frames) with bidirectional attention WITHIN a chunk (and a KV-cache
prefix across chunks).  Running bidirectional attention over the whole sequence
over-dilutes context vs. the ~120-frame training window and collapses the DiT
output to its conditional mean (under-dispersed mel -> muffled/gibberish audio).
We therefore restrict attention to a local block-diagonal window by default
(``attn_block_size`` mel frames).  Set to 0/None to disable (global attention).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Large negative sentinel used to mask out-of-block attention logits.  Small
# enough to be representable in bfloat16, large enough that softmax -> ~0.
_ATTN_MASK_NEG = -1e4


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Precompute RoPE frequency cis matrices.

    Returns a complex tensor of shape [end, dim // 2].
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + small MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: float = 10000.0,
    ) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            t: 1-D tensor of timestep indices (may be scaled).
            dim: Output dimension.
        Returns:
            [len(t), dim] embedding.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half,
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key tensors.

    Args:
        xq, xk: [B, N, num_heads, head_dim] where head_dim is even.
        freqs_cis: [B, N, head_dim // 2] complex tensor.
    Returns:
        Rotated xq, xk with same shape as inputs.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Broadcast freqs_cis across heads.
    freqs_cis = freqs_cis.unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation."""
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.q_norm(q), self.k_norm(k)

        if rotary_pos_emb is not None:
            q, k = apply_rotary_emb(q, k, rotary_pos_emb)

        # scaled_dot_product_attention expects [B, num_heads, N, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Mlp(nn.Module):
    """Standard transformer FFN (used when ffn_type == 'vanilla_mlp')."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class FinalLayer(nn.Module):
    """Final output layer with adaptive layer-norm zero conditioning."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiTBlock(nn.Module):
    """DiT block with adaptive layer-norm zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        x_ = modulate(self.norm1(x), shift_msa, scale_msa)
        x_ = self.attn(x_, rotary_pos_emb=rotary_pos_emb, attn_mask=attn_mask)
        x = x + gate_msa * x_

        x_ = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_ = self.mlp(x_)
        x = x + gate_mlp * x_
        return x


class KimiAudioFlowMatchingDiT(nn.Module):
    """Flow-matching DiT for Kimi Audio semantic tokens -> mel spectrogram."""

    def __init__(self, config: dict):
        super().__init__()
        dit_config = config.get("model", {}).get("dit", {})

        self.hidden_size = dit_config.get("hidden_size", 2304)
        self.input_size = dit_config.get("input_size", 80)
        self.output_size = dit_config.get("output_size", self.input_size)
        self.semantic_vocab_size = dit_config.get("semantic_vocab_size", 16384)
        self.depth = dit_config.get("depth", 16)
        self.num_heads = dit_config.get("num_heads", 18)
        self.mlp_ratio = dit_config.get("mlp_ratio", 4.0)
        self.use_rope = dit_config.get("use_rope", True)
        self.position_embedding_type = dit_config.get("position_embedding_type", "skip")
        self.max_seq_len = dit_config.get("max_seq_len", 4096)
        rope_params = dit_config.get("rope_params", {})
        self.rope_max_position_embeddings = rope_params.get("max_position_embeddings", 4096)
        self.rope_base = rope_params.get("rope_base", 10000.0)

        # Local/block-diagonal attention window, in MEL frames.  The reference
        # processes the DiT in chunks of 30 semantic tokens (= 120 mel frames at
        # upsample x4) with a KV-cache prefix across chunks; without that prefix
        # a slightly smaller window (~60 frames) generalizes better in one-shot.
        # KIMI_DIT_BLOCK_SIZE overrides config for live A/B (<=0 disables masking).
        env_block = os.environ.get("KIMI_DIT_BLOCK_SIZE")
        if env_block is not None and env_block != "":
            self.attn_block_size = int(env_block)
        else:
            self.attn_block_size = int(dit_config.get("attn_block_size", 60))
        self._attn_mask_cache: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

        # Reference checkpoint stores semantic_vocab_size + 1 embeddings.
        self.semantic_token_embedding = nn.Embedding(
            self.semantic_vocab_size + 1,
            self.hidden_size,
        )
        self.input_linear = nn.Linear(self.input_size, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size, frequency_embedding_size=256)

        if self.position_embedding_type == "skip":
            self.position_embedding = None
        else:
            raise NotImplementedError(f"position_embedding_type={self.position_embedding_type} is not supported")

        if self.use_rope:
            assert self.hidden_size % self.num_heads == 0
            rope_dim = self.head_dim = self.hidden_size // self.num_heads
            self.register_buffer(
                "rotary_pos_emb",
                precompute_freqs_cis(
                    rope_dim,
                    self.rope_max_position_embeddings,
                    theta=self.rope_base,
                ),
                persistent=False,
            )
        else:
            self.rotary_pos_emb = None

        self.blocks = nn.ModuleList(
            [DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.depth)]
        )
        self.final_layer = FinalLayer(self.hidden_size, self.output_size)

    def _build_rotary_pos_emb(self, position_ids: torch.Tensor) -> torch.Tensor | None:
        """Gather RoPE frequencies for the given positions."""
        if not self.use_rope or self.rotary_pos_emb is None:
            return None
        B, N = position_ids.shape
        # position_ids: [B, N]; rotary_pos_emb: [max_pos, head_dim//2]
        rotary = torch.zeros(
            (B, N, self.rotary_pos_emb.shape[1]),
            dtype=self.rotary_pos_emb.dtype,
            device=position_ids.device,
        )
        for b in range(B):
            rotary[b] = self.rotary_pos_emb[position_ids[b]]
        return rotary

    def _build_block_mask(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        """Build an additive block-diagonal attention mask for local attention.

        Returns a [length, length] tensor with 0 inside each block of
        ``attn_block_size`` frames and a large negative value outside, so each
        position attends only within its local window.  Returns None when local
        attention is disabled (attn_block_size <= 0) -> global attention.
        Cached per (length, block, device, dtype); generate() calls forward()
        once per ODE step so the mask is built only once per sequence.
        """
        if self.attn_block_size <= 0 or length <= self.attn_block_size:
            return None
        key = (length, self.attn_block_size, device, dtype)
        mask = self._attn_mask_cache.get(key)
        if mask is None:
            idx = torch.arange(length, device=device)
            same_block = (idx // self.attn_block_size).unsqueeze(1) == (idx // self.attn_block_size).unsqueeze(0)
            mask = torch.where(
                same_block,
                torch.zeros((), device=device, dtype=dtype),
                torch.full((), _ATTN_MASK_NEG, device=device, dtype=dtype),
            )
            self._attn_mask_cache[key] = mask
        return mask

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, L, input_size] noisy mel.
            condition: [B, L] semantic token IDs.
            t: [B] timestep indices (already scaled if needed).
            position_ids: [B, L] position indices. Defaults to arange.
        Returns:
            [B, L, output_size] predicted velocity/mel.
        """
        B, L, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        cond_emb = self.semantic_token_embedding(condition)
        x = self.input_linear(x)

        rotary_pos_emb = self._build_rotary_pos_emb(position_ids)
        attn_mask = self._build_block_mask(L, x.device, x.dtype)

        t_emb = self.t_embedder(t)
        c = t_emb.unsqueeze(1) + cond_emb

        for block in self.blocks:
            x = block(x, c, rotary_pos_emb=rotary_pos_emb, attn_mask=attn_mask)

        x = self.final_layer(x, c)
        return x

    def generate(
        self,
        audio_token_ids: torch.Tensor,
        ode_steps: int = 30,
        cfg_scale: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        normalize_mel: bool = False,
        mel_mean: float = 0.0,
        mel_std: float = 1.0,
    ) -> torch.Tensor:
        """Flow-matching inference with Euler ODE solver.

        Args:
            audio_token_ids: [B, L] or [L] semantic token IDs in [0, semantic_vocab_size).
            ode_steps: Number of Euler steps.
            cfg_scale: Classifier-free guidance scale (<=1 disables CFG).
            dtype: Data type for the diffusion latents.
            normalize_mel: Whether to denormalize the predicted mel.
            mel_mean, mel_std: Mel normalization statistics.
        Returns:
            [B, L, 80] mel spectrogram.
        """
        if audio_token_ids.dim() == 1:
            audio_token_ids = audio_token_ids.unsqueeze(0)

        device = audio_token_ids.device
        audio_token_ids = audio_token_ids.clamp(0, self.semantic_vocab_size)

        B, L = audio_token_ids.shape
        x = torch.randn(B, L, self.input_size, device=device, dtype=dtype)

        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        dt = 1.0 / ode_steps

        use_cfg = cfg_scale > 1.0

        for step in range(ode_steps):
            t_val = step / ode_steps
            # Match the reference timestep scaling: t * 1000 cast to long.
            t_tensor = torch.full(
                (B,),
                int(t_val * 1000),
                device=device,
                dtype=torch.long,
            )

            if use_cfg:
                x_in = torch.cat([x, x], dim=0)
                cond_in = torch.cat([audio_token_ids, audio_token_ids], dim=0)
                pos_in = torch.cat([position_ids, position_ids], dim=0)
                t_in = torch.cat([t_tensor, t_tensor], dim=0)
                pred = self.forward(x_in, cond_in, t_in, pos_in)
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                v = self.forward(x, audio_token_ids, t_tensor, position_ids)

            x = x + v.to(dtype) * dt

        if normalize_mel:
            x = x * mel_std + mel_mean

        return x
