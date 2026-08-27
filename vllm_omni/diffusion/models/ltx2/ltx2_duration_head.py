# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.5 caption-conditioned duration prediction."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class LTX2DurationAttentionPooler(nn.Module):
    """Pool connector tokens with learned cross-attention queries."""

    def __init__(self, hidden_dim: int = 256, num_queries: int = 1, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        query = self.to_q(queries).unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        key = self.to_k(tokens).unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        value = self.to_v(tokens).unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        return self.to_out(hidden_states)


class LTX2DurationHead(nn.Module):
    """Predict a shot duration in seconds from LTX connector outputs."""

    def __init__(
        self,
        video_cross_attention_dim: int = 4096,
        audio_cross_attention_dim: int = 2048,
        pooler_hidden_dim: int = 256,
        num_queries: int = 1,
        num_pooler_heads: int = 4,
        mlp_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim)
        self.video_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)
        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim)
        self.audio_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)
        self.attention_pooler = LTX2DurationAttentionPooler(
            hidden_dim=pooler_hidden_dim,
            num_queries=num_queries,
            num_heads=num_pooler_heads,
        )
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden_dim)
        self.mlp_out = nn.Linear(mlp_hidden_dim, 1)

    def forward(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if video_tokens is None and audio_tokens is None:
            raise ValueError("LTX2DurationHead requires video_tokens and/or audio_tokens.")

        dtype = self.video_input_proj.weight.dtype
        token_groups = []
        if video_tokens is not None:
            token_groups.append(self.video_input_proj(video_tokens.to(dtype)) + self.video_modality_emb)
        if audio_tokens is not None:
            token_groups.append(self.audio_input_proj(audio_tokens.to(dtype)) + self.audio_modality_emb)

        pooled = self.attention_pooler(torch.cat(token_groups, dim=1)).flatten(1)
        hidden_states = F.gelu(self.mlp_hidden(pooled), approximate="tanh")
        return self.mlp_out(hidden_states).squeeze(-1).exp()

    def predict_num_frames(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
        *,
        frame_rate: float,
        temporal_compression_ratio: int,
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
    ) -> int:
        """Clamp one duration prediction and snap it to the VAE frame grid."""
        predicted_seconds = self(video_tokens, audio_tokens)
        if predicted_seconds.numel() != 1:
            raise ValueError(
                f"predict_num_frames supports one prompt at a time, but got shape {tuple(predicted_seconds.shape)}."
            )
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive.")
        if temporal_compression_ratio < 1:
            raise ValueError("temporal_compression_ratio must be positive.")
        if min_seconds >= max_seconds:
            raise ValueError("min_seconds must be less than max_seconds.")

        seconds = predicted_seconds.item()
        min_frames = max(1, round(min_seconds * frame_rate))
        max_frames = round(max_seconds * frame_rate)
        clamped_frames = max(min_frames, min(round(seconds * frame_rate), max_frames))
        num_frames = ((clamped_frames - 1) // temporal_compression_ratio) * temporal_compression_ratio + 1

        if num_frames < min_frames:
            snapped_up = num_frames + temporal_compression_ratio
            if snapped_up <= max_frames:
                num_frames = snapped_up
            else:
                if abs(snapped_up - clamped_frames) < abs(num_frames - clamped_frames):
                    num_frames = snapped_up
                logger.warning(
                    "Duration bounds [%.2fs, %.2fs] at %.2f fps contain no frame count on the %dk + 1 grid; "
                    "using %d frames.",
                    min_seconds,
                    max_seconds,
                    frame_rate,
                    temporal_compression_ratio,
                    num_frames,
                )

        if seconds < min_seconds or seconds > max_seconds:
            logger.warning(
                "Duration prediction %.2fs was clamped to %.2fs (%d frames at %.2f fps).",
                seconds,
                num_frames / frame_rate,
                num_frames,
                frame_rate,
            )
        else:
            logger.info(
                "Predicted duration %.2fs (%d frames at %.2f fps).",
                seconds,
                num_frames,
                frame_rate,
            )
        return num_frames
