# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/OpenMOSS/MOVA
"""
MOVA Dual-Tower Conditional Bridge.

Provides bidirectional cross-modal conditioning between visual and audio
DiT towers. Includes Aligned RoPE for temporal synchronization,
per-frame attention pooling, and configurable interaction strategies.

Strategy: minimal semantic migration from upstream interactionv2.py.
PerFrameAttentionPooling uses nn.MultiheadAttention (not vllm-omni Attention).
ConditionalCrossAttention uses vllm-omni Attention for consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.layer import Attention

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# RoPE for cross-modal alignment
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """RoPE implementation (based on Qwen3 pattern)."""

    inv_freq: torch.Tensor

    def __init__(self, base: float, dim: int, device: torch.device | None = None):
        super().__init__()
        self.base = base
        self.dim = dim
        self.attention_scaling = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Per-frame attention pooling
# ---------------------------------------------------------------------------


class PerFrameAttentionPooling(nn.Module):
    """
    Per-frame multi-head attention pooling.

    Given [B, L, D] where L = T*H*W, pools H*W tokens per frame to produce [B, T, D].
    Uses nn.MultiheadAttention (not DiT attention -- this is a functional pooling module).
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        self.probe = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.normal_(self.probe, std=0.02)

        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        """
        Args:
            x: [B, L, D], where L = T*H*W
            grid_size: (T, H, W)
        Returns:
            pooled: [B, T, D]
        """
        B, L, D = x.shape
        T, H, W = grid_size

        S = H * W
        x_bt_s_d = x.view(B, T, S, D).contiguous().view(B * T, S, D)

        probe = self.probe.expand(B * T, -1, -1)
        pooled_bt_1_d = self.attention(probe, x_bt_s_d, x_bt_s_d, need_weights=False)[0]
        pooled_bt_d = pooled_bt_1_d.squeeze(1)

        pooled = pooled_bt_d.view(B, T, D)
        pooled = self.layernorm(pooled)
        return pooled


# ---------------------------------------------------------------------------
# Interaction controller
# ---------------------------------------------------------------------------


class CrossModalInteractionController:
    """
    Strategy class controlling which layers interact between visual and audio towers.

    Strategies:
        - "shallow_focus": first ~1/3 of layers
        - "distributed": every 3 layers
        - "progressive": dense first 8, sparse after
        - "custom": explicit layer indices
        - "full": all layers
    """

    def __init__(self, visual_layers: int = 30, audio_layers: int = 30):
        self.visual_layers = visual_layers
        self.audio_layers = audio_layers
        self.min_layers = min(visual_layers, audio_layers)

    def get_interaction_layers(self, strategy: str = "shallow_focus") -> dict[str, list[tuple[int, int]]]:
        if strategy == "shallow_focus":
            num_interact = min(10, self.min_layers // 3)
            interact_layers = list(range(0, num_interact))
        elif strategy == "distributed":
            interact_layers = list(range(0, self.min_layers, 3))
        elif strategy == "progressive":
            shallow = list(range(0, min(8, self.min_layers)))
            if self.min_layers > 8:
                deep = list(range(8, self.min_layers, 3))
                interact_layers = shallow + deep
            else:
                interact_layers = shallow
        elif strategy == "custom":
            interact_layers = [i for i in [0, 2, 4, 6, 8, 12, 16, 20] if i < self.min_layers]
        elif strategy == "full":
            interact_layers = list(range(0, self.min_layers))
        else:
            raise ValueError(f"Unknown interaction strategy: {strategy}")

        return {
            "v2a": [(i, i) for i in interact_layers],
            "a2v": [(i, i) for i in interact_layers],
        }

    def should_interact(self, layer_idx: int, direction: str, interaction_mapping: dict) -> bool:
        if direction not in interaction_mapping:
            return False
        return any(src == layer_idx for src, _ in interaction_mapping[direction])


# ---------------------------------------------------------------------------
# Cross-attention modules
# ---------------------------------------------------------------------------


class ConditionalCrossAttention(nn.Module):
    """Cross-attention between two modalities with RMSNorm on Q/K and optional RoPE."""

    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.q_dim = dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
        y_freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)

        b = q.shape[0]
        n, d = self.num_heads, self.head_dim

        if x_freqs is not None:
            x_cos, x_sin = x_freqs
            q_view = q.view(b, -1, n, d)
            x_cos = x_cos.to(q_view.dtype).to(q_view.device)
            x_sin = x_sin.to(q_view.dtype).to(q_view.device)
            q_view, _ = apply_rotary_pos_emb(q_view, q_view, x_cos, x_sin, unsqueeze_dim=2)
            q = q_view.reshape(b, -1, n * d)

        if y_freqs is not None:
            y_cos, y_sin = y_freqs
            k_view = k.view(b, -1, n, d)
            y_cos = y_cos.to(k_view.dtype).to(k_view.device)
            y_sin = y_sin.to(k_view.dtype).to(k_view.device)
            _, k_view = apply_rotary_pos_emb(k_view, k_view, y_cos, y_sin, unsqueeze_dim=2)
            k = k_view.reshape(b, -1, n * d)

        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)

        x = self.attn(q, k, v)
        x = x.flatten(2)
        return self.o(x)


# ---------------------------------------------------------------------------
# Adaptive LayerNorm (inlined from diffusers to avoid import)
# ---------------------------------------------------------------------------


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm incorporating timestep/condition embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int | None = None,
        output_dim: int | None = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 2:
            scale, shift = temb.chunk(2, dim=2)
        elif self.chunk_dim == 1:
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)

        return self.norm(x) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Cross-attention block wrapper
# ---------------------------------------------------------------------------


class ConditionalCrossAttentionBlock(nn.Module):
    """Wrapper: LayerNorm on conditioning input + optional pooled AdaLN."""

    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6, pooled_adaln: bool = False):
        super().__init__()
        self.y_norm = nn.LayerNorm(kv_dim, eps=eps)
        self.inner = ConditionalCrossAttention(dim=dim, kv_dim=kv_dim, num_heads=num_heads, eps=eps)
        self.pooled_adaln = pooled_adaln
        if pooled_adaln:
            self.per_frame_pooling = PerFrameAttentionPooling(kv_dim, num_heads=num_heads, eps=eps)
            self.adaln = AdaLayerNorm(kv_dim, output_dim=dim * 2, chunk_dim=2)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
        y_freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
        video_grid_size: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        if self.pooled_adaln:
            assert video_grid_size is not None, "video_grid_size must not be None"
            pooled_y = self.per_frame_pooling(y, video_grid_size)
            if pooled_y.shape[1] != x.shape[1]:
                pooled_y = F.interpolate(
                    pooled_y.permute(0, 2, 1),
                    size=x.shape[1],
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
            x = self.adaln(x, temb=pooled_y)
        y = self.y_norm(y)
        return self.inner(x=x, y=y, x_freqs=x_freqs, y_freqs=y_freqs)


# ---------------------------------------------------------------------------
# Main bridge module
# ---------------------------------------------------------------------------


class MovaBridge(nn.Module):
    """
    Dual-tower conditional bridge for MOVA.

    Provides bidirectional cross-modal conditioning between visual and audio
    DiT towers at designated layer indices, with Aligned RoPE for temporal
    synchronization.
    """

    _repeated_blocks = ("ConditionalCrossAttentionBlock",)

    def __init__(
        self,
        visual_layers: int = 30,
        audio_layers: int = 30,
        visual_hidden_dim: int = 3072,
        audio_hidden_dim: int = 1536,
        audio_fps: float = 44100.0 / 2048.0,
        head_dim: int = 128,
        interaction_strategy: str = "shallow_focus",
        apply_cross_rope: bool = False,
        apply_first_frame_bias_in_rope: bool = False,
        trainable_condition_scale: bool = False,
        pooled_adaln: bool = False,
    ):
        super().__init__()

        self.visual_hidden_dim = visual_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.audio_fps = audio_fps
        self.head_dim = head_dim
        self.apply_cross_rope = apply_cross_rope
        self.apply_first_frame_bias_in_rope = apply_first_frame_bias_in_rope
        self.trainable_condition_scale = trainable_condition_scale
        self.pooled_adaln = pooled_adaln

        if self.trainable_condition_scale:
            self.condition_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        else:
            self.condition_scale = 1.0

        self.controller = CrossModalInteractionController(visual_layers, audio_layers)
        self.interaction_mapping = self.controller.get_interaction_layers(interaction_strategy)

        # Audio hidden states -> visual DiT conditioning
        self.rotary = RotaryEmbedding(base=10000.0, dim=head_dim)
        self.audio_to_video_conditioners = nn.ModuleDict()
        for v_layer, _ in self.interaction_mapping["a2v"]:
            self.audio_to_video_conditioners[str(v_layer)] = ConditionalCrossAttentionBlock(
                dim=visual_hidden_dim,
                kv_dim=audio_hidden_dim,
                num_heads=visual_hidden_dim // head_dim,
                pooled_adaln=False,
            )

        # Visual hidden states -> audio DiT conditioning
        self.video_to_audio_conditioners = nn.ModuleDict()
        for a_layer, _ in self.interaction_mapping["v2a"]:
            self.video_to_audio_conditioners[str(a_layer)] = ConditionalCrossAttentionBlock(
                dim=audio_hidden_dim,
                kv_dim=visual_hidden_dim,
                num_heads=audio_hidden_dim // head_dim,
                pooled_adaln=self.pooled_adaln,
            )

    @torch.no_grad()
    def build_aligned_freqs(
        self,
        video_fps: float,
        grid_size: tuple[int, int, int],
        audio_steps: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        Build temporally aligned RoPE (cos, sin) pairs for video and audio.

        Returns:
            visual_freqs: (cos_v, sin_v), shape [1, f_v*h*w, head_dim]
            audio_freqs:  (cos_a, sin_a), shape [1, audio_steps, head_dim]
        """
        f_v, h, w = grid_size
        L_v = f_v * h * w
        L_a = int(audio_steps)

        device = device or next(self.parameters()).device
        dtype = dtype or torch.float32

        # Audio positions: simple arange
        audio_pos = torch.arange(L_a, device=device, dtype=torch.float32).unsqueeze(0)

        # Video positions: aligned to audio-step units
        # Hard-coded VAE temporal stride = 4
        if self.apply_first_frame_bias_in_rope:
            video_effective_fps = float(video_fps) / 4.0
            if f_v > 0:
                t_starts = torch.zeros((f_v,), device=device, dtype=torch.float32)
                if f_v > 1:
                    t_starts[1:] = (1.0 / float(video_fps)) + torch.arange(
                        f_v - 1, device=device, dtype=torch.float32
                    ) * (1.0 / video_effective_fps)
            else:
                t_starts = torch.zeros((0,), device=device, dtype=torch.float32)
            video_pos_per_frame = t_starts * float(self.audio_fps)
        else:
            scale = float(self.audio_fps) / float(video_fps / 4.0)
            video_pos_per_frame = torch.arange(f_v, device=device, dtype=torch.float32) * scale

        # Tokens within the same frame share the same time position
        video_pos = video_pos_per_frame.repeat_interleave(h * w).unsqueeze(0)

        # Build cos/sin via RotaryEmbedding
        dummy_v = torch.zeros((1, L_v, self.head_dim), device=device, dtype=dtype)
        dummy_a = torch.zeros((1, L_a, self.head_dim), device=device, dtype=dtype)

        cos_v, sin_v = self.rotary(dummy_v, position_ids=video_pos)
        cos_a, sin_a = self.rotary(dummy_a, position_ids=audio_pos)

        return (cos_v, sin_v), (cos_a, sin_a)

    def should_interact(self, layer_idx: int, direction: str) -> bool:
        return self.controller.should_interact(layer_idx, direction, self.interaction_mapping)

    def apply_conditional_control(
        self,
        layer_idx: int,
        direction: str,
        primary_hidden_states: torch.Tensor,
        condition_hidden_states: torch.Tensor,
        x_freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
        y_freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
        condition_scale: float | None = None,
        video_grid_size: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Apply conditional control at the DiT hidden-state level."""
        if not self.controller.should_interact(layer_idx, direction, self.interaction_mapping):
            return primary_hidden_states

        if direction == "a2v":
            conditioner = self.audio_to_video_conditioners[str(layer_idx)]
        elif direction == "v2a":
            conditioner = self.video_to_audio_conditioners[str(layer_idx)]
        else:
            raise ValueError(f"Invalid direction: {direction}")

        conditioned_features = conditioner(
            x=primary_hidden_states,
            y=condition_hidden_states,
            x_freqs=x_freqs,
            y_freqs=y_freqs,
            video_grid_size=video_grid_size,
        )

        if self.trainable_condition_scale and condition_scale is not None:
            logger.warning(
                "This model has a trainable condition_scale, but an external "
                "condition_scale=%s was provided. The trainable condition_scale "
                "will be ignored.",
                condition_scale,
            )

        scale = condition_scale if condition_scale is not None else self.condition_scale
        return primary_hidden_states + conditioned_features * scale

    def forward(
        self,
        layer_idx: int,
        visual_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        *,
        x_freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
        y_freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
        a2v_condition_scale: float | None = None,
        v2a_condition_scale: float | None = None,
        condition_scale: float | None = None,
        video_grid_size: tuple[int, int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional conditional control for both visual and audio towers.

        Returns:
            (visual_conditioned, audio_conditioned)
        """
        visual_conditioned = self.apply_conditional_control(
            layer_idx=layer_idx,
            direction="a2v",
            primary_hidden_states=visual_hidden_states,
            condition_hidden_states=audio_hidden_states,
            x_freqs=x_freqs,
            y_freqs=y_freqs,
            condition_scale=a2v_condition_scale if a2v_condition_scale is not None else condition_scale,
            video_grid_size=video_grid_size,
        )

        audio_conditioned = self.apply_conditional_control(
            layer_idx=layer_idx,
            direction="v2a",
            primary_hidden_states=audio_hidden_states,
            condition_hidden_states=visual_hidden_states,
            x_freqs=y_freqs,
            y_freqs=x_freqs,
            condition_scale=v2a_condition_scale if v2a_condition_scale is not None else condition_scale,
            video_grid_size=video_grid_size,
        )

        return visual_conditioned, audio_conditioned
