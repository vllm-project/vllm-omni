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


@dataclass(frozen=True, kw_only=True)
class KyutaiAuraConfig:
    """Facts of one released AURA checkpoint, grouped by the part of the pipeline that reads them."""

    # Where the weights come from.
    name: str
    hub_repo: str
    checkpoint: str  # merged DiT weights inside ``hub_repo``
    base_repo: str  # the Wan repository that provides the VAE, umT5 and its tokenizer
    vae_file: str

    # Diffusion transformer (Wan layout).
    dim: int
    ffn_dim: int
    num_heads: int
    num_layers: int = 30
    latent_channels: int
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6

    # Speech conditioning.
    audio_dim: int  # channels of one speech feature frame
    audio_layers: int  # how many feature layers the speech encoder mixes
    audio_inject_layers: tuple[int, ...] = (
        0,
        3,
        6,
        9,
        12,
        15,
        18,
        21,
        24,
        27,
        29,
    )  # blocks followed by speech attention
    audio_local_tokens: int = 4  # speech tokens per latent frame, not counting the padding token
    audio_motion_frames: int = 73  # copies of the first feature put before each chunk's features

    # Video and VAE.
    fps: int
    buckets: Mapping[float, tuple[int, int]]  # aspect ratio (height / width) -> (height, width)
    vae_stride: tuple[int, int, int]  # downsampling of the VAE in (time, height, width)
    vae_fp32: bool  # compute the VAE in fp32 (True) or bf16 (False); its weights are fp32
    vae_decode_group: int  # how many latent frames one decoder call takes, after the first frame

    # Streaming: chunks, history and temporal positions.
    frames_per_chunk: int = 4  # latent frames generated together
    window_chunks: int = 2  # chunks kept after the anchor chunk, counting the one being generated
    segment_frames: int  # length in latent frames of the clips the model was trained on
    ref_rope_offset: int  # where the reference frame sits in time inside such a clip
    rope_reindex: bool  # stop advancing temporal positions once a clip's length is reached

    # Sampling.
    timesteps: tuple[int, ...] = (1000, 745, 498, 255)
    schedule_steps: int = 50  # points of the teacher's flow-matching grid
    schedule_shift: float = 5.0

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
    dim=1536,
    ffn_dim=8960,
    num_heads=12,
    latent_channels=16,
    audio_dim=512,  # Mimi: the dequantized first (semantic) codebook
    audio_layers=1,
    fps=16,
    buckets=MappingProxyType({0.67: (512, 768), 1.00: (576, 576)}),
    vae_stride=(4, 8, 8),
    vae_fp32=False,
    vae_decode_group=4,
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
