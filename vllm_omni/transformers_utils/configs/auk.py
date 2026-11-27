# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Config for the AuK instruction-driven audio generation model.

The upstream checkpoint ships ``config.yaml`` plus loose safetensors, so an
assembled directory is produced by ``tools/prepare_auk_checkpoint.py``: the
Qwen2.5-Omni-3B ``config.json`` with this ``model_type``, AuK's architecture,
and the five sections below added at the top level.

Subclassing ``Qwen2_5OmniConfig`` is load-bearing, not stylistic. The encoder
stage reuses the in-tree Qwen2.5-Omni thinker processor, whose
``get_hf_config`` resolves ``self.ctx.get_hf_config(Qwen2_5OmniConfig)`` and
asserts the type, so a bare ``PretrainedConfig`` subclass would fail before
the first request. Inheriting also leaves the parent's nested
``thinker_config`` / ``talker_config`` / ``token2wav_config`` handling and its
``get_text_config`` in place, which is what the encoder stage reads.
"""

from typing import Any

from transformers import AutoConfig
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniConfig

# The distilled AuK-Flash checkpoint pins a 4-step time grid recorded in its
# metadata. The values are exact and must never be rounded: the student was
# trained on them, and a rounded grid changes the output amplitude.
AUK_FLASH_T_GRID: tuple[float, ...] = (
    0.0,
    0.07612049579620361,
    0.2928932309150696,
    0.6173166036605835,
    1.0,
)

DEFAULT_DIT: dict[str, Any] = {
    "dim": 1536,
    "heads": 24,
    "dim_head": 64,
    "ff_mult": 2,
    "text_hidden_dim": 2048,
    "num_layers": 10,
    "num_single_layers": 20,
    "attn_mask_enabled": True,
}

DEFAULT_VAE: dict[str, Any] = {
    "latent_dim": 64,
    "downsample_rate": 480,
    "target_sample_rate": 24000,
    "model_init_kwargs": {
        "upsample_rates": [5, 4, 3, 2, 2, 2],
        "upsample_kernel_sizes": [10, 8, 6, 4, 4, 4],
        "upsample_initial_channel": 1536,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "downsample_rates": [2, 2, 2, 3, 4, 5],
        "downsample_channels": [12, 24, 48, 96, 192, 384, 768],
        "snake_logscale": True,
        "latent_dim": 64,
        "use_vae": True,
        "causal": True,
        "flow_hidden_channels": 256,
        "act_causal": True,
    },
}

# Base AuK sampling recipe; AuK-Flash replaces nfe and cfg with its own.
DEFAULT_SAMPLING: dict[str, Any] = {"nfe": 32, "cfg": 2.0, "sway": -1.0}

VARIANTS = ("base", "flash")


class AuKConfig(Qwen2_5OmniConfig):
    """Assembled AuK checkpoint config: a Qwen2.5-Omni config plus DiT and VAE."""

    model_type = "auk"

    def __init__(
        self,
        dit: dict[str, Any] | None = None,
        vae: dict[str, Any] | None = None,
        variant: str = "base",
        flash_t_grid: list[float] | None = None,
        defaults: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"AuK variant must be one of {VARIANTS}; got {variant!r}")

        # Assigned before the parent's init so nothing it runs during
        # validation sees a half-built config.
        self.dit = {**DEFAULT_DIT, **(dit or {})}
        self.vae = {**DEFAULT_VAE, **(vae or {})}
        self.variant = variant
        self.flash_t_grid = list(flash_t_grid) if flash_t_grid is not None else list(AUK_FLASH_T_GRID)
        self.defaults = {**DEFAULT_SAMPLING, **(defaults or {})}

        kwargs.setdefault("architectures", ["AuKForConditionalGeneration"])
        # The parent owns thinker_config / talker_config / token2wav_config and
        # get_text_config; nothing here overrides them.
        super().__init__(**kwargs)

    @property
    def is_flash(self) -> bool:
        """Whether this checkpoint is the distilled 4-step student."""
        return getattr(self, "variant", "base") == "flash"

    @property
    def latent_dim(self) -> int:
        return int(self.vae["latent_dim"])

    @property
    def downsample_rate(self) -> int:
        return int(self.vae["downsample_rate"])

    @property
    def target_sample_rate(self) -> int:
        return int(self.vae["target_sample_rate"])


AutoConfig.register("auk", AuKConfig)

__all__ = [
    "AUK_FLASH_T_GRID",
    "DEFAULT_DIT",
    "DEFAULT_SAMPLING",
    "DEFAULT_VAE",
    "VARIANTS",
    "AuKConfig",
]
