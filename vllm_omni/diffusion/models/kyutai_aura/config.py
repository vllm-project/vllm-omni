# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration of the released Kyutai AURA-1.3B checkpoint.

AURA (https://arxiv.org/abs/2611.26540) is a 4-step causal speech-to-video student of a Wan
diffusion transformer. It generates a talking-head video chunk by chunk: each chunk is
``frames_per_chunk`` latent frames, sampled at ``timesteps`` without CFG, with a rolling K/V
history that keeps the first chunk as an anchor plus the latest ``window_chunks`` chunks.
AURA-5B (wav2vec2 speech features, Wan2.2 VAE, 720p at 25 fps) is not supported yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

# Used when the request gives no prompt. The checkpoints were distilled with one fixed description of a
# person talking to the camera; a request may pass its own description as the prompt.
DEFAULT_PROMPT = (
    "Close-up video of someone speaking to the camera with lively facial expressions and slight head motion."
)


@dataclass(frozen=True)
class KyutaiAuraConfig:
    """Architecture and sampling facts of one released AURA checkpoint."""

    name: str
    hub_repo: str
    checkpoint: str  # standalone DiT weights (LoRA merged), relative to ``hub_repo``
    base_repo: str  # Wan base: VAE, umT5 encoder and tokenizer
    vae_file: str
    vae_stride: tuple[int, int, int]  # (time, height, width)
    vae_fp32: bool  # VAE compute dtype: fp32 if True, else bf16 (weights are fp32 either way)
    vae_decode_group: int  # latent frames per VAE decoder call after the first frame
    audio_dim: int  # width of the speech features
    audio_layers: int  # feature layers mixed by the learned weights of the speech encoder
    fps: int
    buckets: Mapping[float, tuple[int, int]]  # height / width -> (height, width)
    dim: int
    ffn_dim: int
    num_heads: int
    latent_channels: int
    segment_frames: int  # latent frames of one training clip (RoPE layout)
    ref_rope_offset: int  # temporal RoPE position of the reference frame inside a segment
    rope_reindex: bool  # freeze positions after the last chunk that fits in a training clip
    num_layers: int = 30
    patch_size: tuple[int, int, int] = (1, 2, 2)
    audio_inject_layers: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29)
    audio_local_tokens: int = 4  # local speech tokens per latent frame (plus one padding token)
    audio_motion_frames: int = 73  # first feature repeated before each chunk's speech encoder input
    frames_per_chunk: int = 4  # latent frames per streamed chunk
    window_chunks: int = 2  # cached chunks besides the anchor, the current one included
    timesteps: tuple[int, ...] = (1000, 745, 498, 255)
    schedule_steps: int = 50  # flow-matching grid of the teacher
    schedule_shift: float = 5.0
    text_len: int = 512
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""
        return self.dim // self.num_heads

    @property
    def video_frames_per_chunk(self) -> int:
        """Decoded frames of every chunk but the first, which has ``vae_stride[0] - 1`` fewer."""
        return self.vae_stride[0] * self.frames_per_chunk

    @property
    def rope_max_chunk(self) -> int | None:
        """Last chunk index whose positions fit in one training segment, or None (no freeze)."""
        if not self.rope_reindex:
            return None
        return self.segment_frames // self.frames_per_chunk - 1


KYUTAI_AURA_1_3B = KyutaiAuraConfig(
    name="kyutai-aura-1.3b",
    hub_repo="kyutai/AURA-1.3B",
    checkpoint="stage2_dmd/step-6500_ema_merged.safetensors",
    base_repo="Wan-AI/Wan2.1-T2V-1.3B",
    vae_file="Wan2.1_VAE.pth",
    vae_stride=(4, 8, 8),
    vae_fp32=False,
    vae_decode_group=4,
    audio_dim=512,  # Mimi: the dequantized first (semantic) codebook
    audio_layers=1,
    fps=16,
    buckets=MappingProxyType({0.67: (512, 768), 1.00: (576, 576)}),
    dim=1536,
    ffn_dim=8960,
    num_heads=12,
    latent_channels=16,
    segment_frames=20,
    ref_rope_offset=30,
    rope_reindex=True,
)

SUPPORTED: Mapping[str, KyutaiAuraConfig] = MappingProxyType({KYUTAI_AURA_1_3B.hub_repo.lower(): KYUTAI_AURA_1_3B})
NOT_SUPPORTED_YET: Mapping[str, str] = MappingProxyType(
    {"kyutai/aura-5b": "Kyutai AURA-5B (wav2vec2 speech features, Wan2.2 VAE) is not supported yet"}
)


def _hub_id(model: str) -> str:
    return str(model).removeprefix("https://huggingface.co/").strip("/").lower()


def resolve_kyutai_aura_config(model: str) -> KyutaiAuraConfig | None:
    """The supported checkpoint behind a Hub id or a local snapshot directory, or None.

    The AURA repos have no root config.json or model_index.json; a local directory is matched by
    the standalone DiT file it holds."""
    repo = _hub_id(model)
    if repo in SUPPORTED:
        return SUPPORTED[repo]
    path = Path(model)
    if path.is_dir():
        return next((cfg for cfg in SUPPORTED.values() if (path / cfg.checkpoint).is_file()), None)
    return None


def unsupported_kyutai_aura(model: str) -> str | None:
    """Why a released AURA checkpoint is not supported yet, or None for any other model."""
    return NOT_SUPPORTED_YET.get(_hub_id(model))


def closest_bucket(buckets: Mapping[float, tuple[int, int]], height: int, width: int) -> tuple[int, int]:
    """Training resolution (height, width) whose aspect key is closest to ``height / width``."""
    ratio = height / width
    return buckets[min(buckets, key=lambda key: abs(key - ratio))]
