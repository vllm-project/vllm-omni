# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration for the FastWAM family of world-action models.

FastWAM (Yuan et al., MIT) is a mixture-of-transformers policy: a Wan2.2-TI2V-5B
video expert encodes the current camera frame once, and a small action expert
denoises an action chunk while attending to the cached video keys/values.

The family members share one architecture and differ only in

* the action expert's width (``action_hidden_dim`` / ``action_ffn_dim``),
* the width of the tokens it denoises (``action_token_dim``): raw 7-D actions for
  the original FastWAM checkpoints, or 16-D latents for CAE-FastWAM,
* an optional action codec (CAE's suite-conditioned action VAE decoder) that maps
  the denoised latents back to the action chunk, and
* the flow-matching shift of the action schedule.

The shift cannot be recovered from the weights, so it always comes from the
preset; every shape that *can* be read from the checkpoint is checked against
the preset at load time (:meth:`FastWAMConfig.check_policy_state_dict`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# Prompt template the FastWAM LIBERO checkpoints were trained with.
DEFAULT_PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


@dataclass(frozen=True)
class ActionCodecConfig:
    """Decoder half of CAE-FastWAM's action VAE (``action_vae.pt``)."""

    action_dim: int = 7
    action_horizon: int = 32
    num_tokens: int = 8
    latent_dim: int = 16
    decoder_hidden: int = 128
    num_suites: int = 4

    @classmethod
    def from_checkpoint_config(cls, cfg: Mapping[str, Any]) -> ActionCodecConfig:
        """Build from the ``config`` dict stored in ``action_vae.pt``.

        Encoder-only fields (``encoder_hidden``, ``variational``, loss weights, ...)
        are training-time settings and are ignored.
        """
        if int(cfg.get("video_cond_dim", 0)) != 0:
            raise ValueError("Video-conditioned action VAEs are not supported (video_cond_dim must be 0).")
        codec = cls(
            action_dim=int(cfg["action_dim"]),
            action_horizon=int(cfg["action_horizon"]),
            num_tokens=int(cfg["num_tokens"]),
            latent_dim=int(cfg["latent_dim"]),
            decoder_hidden=int(cfg["decoder_hidden"]),
            num_suites=int(cfg.get("num_suites", 0)),
        )
        if codec.action_horizon % codec.num_tokens:
            raise ValueError(
                f"action_horizon={codec.action_horizon} must be divisible by num_tokens={codec.num_tokens}."
            )
        return codec


@dataclass(frozen=True)
class FastWAMConfig:
    """Resolved configuration of one FastWAM-family checkpoint."""

    variant: str
    # Action expert (the only part whose shape differs across the family).
    action_hidden_dim: int
    action_ffn_dim: int
    action_token_dim: int
    action_shift: float
    action_codec: ActionCodecConfig | None
    # Checkpoint assets, relative to the model directory.
    policy_checkpoint: str
    action_codec_checkpoint: str | None
    # Normalisation statistics; ``{suite}`` is replaced by the task suite name.
    stats_path: str
    suite_conditioned_stats: bool
    # Environment action chunk.
    action_dim: int = 7
    action_horizon: int = 32
    proprio_dim: int = 8
    # Video expert: Wan2.2-TI2V-5B DiT.
    video_hidden_dim: int = 3072
    video_ffn_dim: int = 14336
    video_in_dim: int = 48
    video_out_dim: int = 48
    patch_size: tuple[int, int, int] = (1, 2, 2)
    # Shared by both experts (mixed attention needs equal heads and depth).
    num_layers: int = 30
    num_heads: int = 24
    attn_head_dim: int = 128
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6
    num_train_timesteps: int = 1000
    # Observation layout: two cameras, each resized to ``camera_size`` (H, W)
    # and concatenated side by side.
    camera_size: tuple[int, int] = (224, 224)
    num_cameras: int = 2
    text_len: int = 128
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    task_suites: tuple[str, ...] = LIBERO_SUITES
    suite_ids: Mapping[str, int] = field(default_factory=lambda: {s: i for i, s in enumerate(LIBERO_SUITES)})
    validated: bool = False

    @property
    def num_action_tokens(self) -> int:
        """Number of tokens the action expert denoises per chunk."""
        return self.action_codec.num_tokens if self.action_codec is not None else self.action_horizon

    @property
    def image_size(self) -> tuple[int, int]:
        """Model input size (H, W) after concatenating the cameras horizontally."""
        height, width = self.camera_size
        return height, width * self.num_cameras

    def stats_file(self, suite: str) -> str:
        return self.stats_path.format(suite=suite)

    def check_policy_state_dict(self, mot_state: Mapping[str, torch.Tensor]) -> None:
        """Fail fast when the checkpoint does not belong to this preset.

        A FastWAM checkpoint loaded with the CAE preset (or the other way round)
        would otherwise run with the wrong action schedule shift and silently
        produce bad actions.
        """
        prefix = "mixtures.action."
        encoder = mot_state.get(prefix + "action_encoder.weight")
        ffn = mot_state.get(prefix + "blocks.0.ffn.0.weight")
        if encoder is None or ffn is None:
            raise ValueError("Policy checkpoint has no FastWAM action expert (mixtures.action.*).")
        hidden, token_dim = (int(d) for d in encoder.shape)
        ffn_dim = int(ffn.shape[0])
        expected = (self.action_hidden_dim, self.action_token_dim, self.action_ffn_dim)
        if (hidden, token_dim, ffn_dim) != expected:
            raise ValueError(
                f"Policy checkpoint action expert has hidden={hidden}, token_dim={token_dim}, ffn={ffn_dim}, "
                f"but the {self.variant!r} preset expects hidden={expected[0]}, token_dim={expected[1]}, "
                f"ffn={expected[2]}. Pick the preset that matches the checkpoint (fastwam_variant)."
            )
        if (self.action_codec is None) != (self.action_token_dim == self.action_dim):
            raise ValueError(
                "An action codec is required exactly when the denoised tokens are not raw actions "
                f"(action_token_dim={self.action_token_dim}, action_dim={self.action_dim})."
            )

    def with_codec(self, codec: ActionCodecConfig) -> FastWAMConfig:
        """Return a copy using the codec read from the checkpoint, after checking it."""
        if codec.latent_dim != self.action_token_dim:
            raise ValueError(f"Action codec latent_dim={codec.latent_dim} != action_token_dim={self.action_token_dim}.")
        if (codec.action_dim, codec.action_horizon) != (self.action_dim, self.action_horizon):
            raise ValueError(
                f"Action codec decodes [{codec.action_horizon}, {codec.action_dim}] chunks, expected "
                f"[{self.action_horizon}, {self.action_dim}]."
            )
        if codec.num_suites and codec.num_suites != len(self.suite_ids):
            raise ValueError(f"Action codec has {codec.num_suites} suites, expected {len(self.suite_ids)}.")
        return replace(self, action_codec=codec)


PRESETS: dict[str, FastWAMConfig] = {
    # amd/CAE-FastWAM (Apache-2.0): 32-wide action expert denoising 8 latent tokens
    # of width 16, decoded by a suite-conditioned action VAE; per-suite stats.
    "cae_fastwam_libero": FastWAMConfig(
        variant="cae_fastwam_libero",
        action_hidden_dim=32,
        action_ffn_dim=256,
        action_token_dim=16,
        action_shift=5.0,
        action_codec=ActionCodecConfig(),
        policy_checkpoint="policy.pt",
        action_codec_checkpoint="action_vae.pt",
        stats_path="stats/{suite}.json",
        suite_conditioned_stats=True,
        validated=True,
    ),
    # yuanty/fastwam ``libero_uncond_2cam224.pt``: 1024-wide action expert denoising
    # 32 raw action tokens; one stats file pooled over the four suites.
    # Loads through the same code but is NOT validated yet.
    "fastwam_libero_uncond": FastWAMConfig(
        variant="fastwam_libero_uncond",
        action_hidden_dim=1024,
        action_ffn_dim=4096,
        action_token_dim=7,
        action_shift=1.0,
        action_codec=None,
        policy_checkpoint="libero_uncond_2cam224.pt",
        action_codec_checkpoint=None,
        stats_path="libero_uncond_2cam224_dataset_stats.json",
        suite_conditioned_stats=False,
        validated=False,
    ),
}
DEFAULT_VARIANT = "cae_fastwam_libero"

# Keys of ``model_config`` that may override a preset's asset paths.
_ASSET_OVERRIDES = ("policy_checkpoint", "action_codec_checkpoint", "stats_path")


def resolve_config(model_config: Mapping[str, Any]) -> FastWAMConfig:
    """Pick the preset named by ``model_config['fastwam_variant']`` and apply asset overrides."""
    variant = str(model_config.get("fastwam_variant") or DEFAULT_VARIANT)
    if variant not in PRESETS:
        raise ValueError(f"Unknown fastwam_variant {variant!r}; choose one of {sorted(PRESETS)}.")
    overrides = {key: model_config[key] for key in _ASSET_OVERRIDES if model_config.get(key) is not None}
    return replace(PRESETS[variant], **overrides)
