# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HuggingFace config classes for the Alpamayo VLA models (R1 + 1.5).

Alpamayo is built on top of Qwen3-VL-8B. Two config layouts exist in the wild:

1. **Flat** (Alpamayo-1.5, ``model_type="alpamayo1_5"``): a flat ``config.json``
   that carries only the Alpamayo-specific fields (``expert_cfg``,
   ``traj_tokenizer_cfg``, ``action_in_proj_cfg``, ``traj_token_ids`` ...) plus a
   ``vlm_name_or_path`` pointing at the base Qwen3-VL model. It has **no** nested
   ``text_config`` / ``vision_config``, so we materialize them from a base
   Qwen3-VL config at construction time.

2. **Merged** (Alpamayo-R1): a ``config.json`` that already contains nested
   ``text_config`` / ``vision_config`` (effectively a Qwen3-VL config with the
   Alpamayo fields overlaid).

Both layouts resolve to a :class:`~transformers.Qwen3VLConfig`-compatible object
so the base vLLM ``Qwen3VLForConditionalGeneration`` can consume it directly,
while the extra Alpamayo fields remain available as attributes for the AR-stage
model, the diffusion expert and the multimodal processor.
"""

from __future__ import annotations

import os

from transformers import AutoConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

# Upstream Qwen3-VL-8B-Instruct used as the VLM base when the Alpamayo
# checkpoint ships a flat config. Override with ``ALPAMAYO_VLM_BASE``
# (local path or HF repo id).
_DEFAULT_VLM_BASE_HF = "Qwen/Qwen3-VL-8B-Instruct"

# Alpamayo-specific top-level keys (kept as attributes; not part of Qwen3VLConfig).
_ALPAMAYO_KEYS = (
    "action_in_proj_cfg",
    "action_out_proj_cfg",
    "action_space_cfg",
    "add_special_tokens",
    "diffusion_cfg",
    "expert_cfg",
    "expert_non_causal_attention",
    "hist_traj_tokenizer_cfg",
    "include_camera_ids",
    "include_frame_nums",
    "keep_same_dtype",
    "max_pixels",
    "min_pixels",
    "model_dtype",
    "padding_side",
    "tokens_per_future_traj",
    "tokens_per_history_traj",
    "traj_token_ids",
    "traj_token_start_idx",
    "traj_tokenizer_cfg",
    "traj_vocab_size",
    "vlm_backend",
    "vlm_name_or_path",
)


def _resolve_vlm_base() -> str:
    return os.environ.get("ALPAMAYO_VLM_BASE") or _DEFAULT_VLM_BASE_HF


class _AlpamayoConfigBase(Qwen3VLConfig):
    """Shared Alpamayo config logic. Subclasses set the concrete ``model_type``."""

    # The base Qwen3-VL fields that, when missing from a flat Alpamayo config,
    # must be pulled from the VLM base config so vLLM builds the right modules.
    _BASE_TOKEN_FIELDS = (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    )

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        # Stash Alpamayo-specific fields so they survive super().__init__ and
        # round-trip through to_dict()/from_dict().
        alpamayo_fields = {k: kwargs.pop(k) for k in _ALPAMAYO_KEYS if k in kwargs}

        # Flat config (1.5): no nested sub-configs -> materialize from base VLM.
        if text_config is None and vision_config is None:
            base = self._load_base_vlm_config(
                kwargs.pop("vlm_name_or_path", None) or alpamayo_fields.get("vlm_name_or_path")
            )
            if base is not None:
                text_config = base.text_config.to_dict() if base.text_config is not None else None
                vision_config = base.vision_config.to_dict() if base.vision_config is not None else None
                for f in self._BASE_TOKEN_FIELDS:
                    kwargs.setdefault(f, getattr(base, f, None))

        super().__init__(text_config=text_config, vision_config=vision_config, **kwargs)

        for k, v in alpamayo_fields.items():
            setattr(self, k, v)

        # Alpamayo extends the tokenizer vocab; keep text_config in sync so the
        # embedding / lm_head are sized correctly.
        vocab_size = alpamayo_fields.get("vocab_size", kwargs.get("vocab_size"))
        if vocab_size is None:
            vocab_size = getattr(self, "vocab_size", None)
        if vocab_size is not None and self.text_config is not None:
            self.text_config.vocab_size = vocab_size
            self.vocab_size = vocab_size

    @staticmethod
    def _load_base_vlm_config(vlm_name_or_path: str | None):
        """Load a base Qwen3-VL config to source text_config / vision_config."""
        candidates = []
        if vlm_name_or_path and (os.path.isdir(vlm_name_or_path) or "/" in vlm_name_or_path):
            candidates.append(vlm_name_or_path)
        candidates.append(_resolve_vlm_base())
        last_err: Exception | None = None
        for cand in candidates:
            try:
                cfg = AutoConfig.from_pretrained(cand, trust_remote_code=True)
                if getattr(cfg, "text_config", None) is not None:
                    return cfg
            except Exception as e:  # noqa: BLE001 - best effort base resolution
                last_err = e
                continue
        if last_err is not None:
            raise RuntimeError(
                "Alpamayo config requires a base Qwen3-VL config but none could be "
                f"loaded (tried {candidates}). Set ALPAMAYO_VLM_BASE to a valid "
                f"Qwen3-VL model path. Last error: {last_err}"
            )
        return None


class Alpamayo15Config(_AlpamayoConfigBase):
    """Config for Alpamayo-1.5 (architectures: ['Alpamayo1_5'])."""

    model_type = "alpamayo1_5"


class AlpamayoR1Config(_AlpamayoConfigBase):
    """Config for Alpamayo-R1 (architectures: ['AlpamayoR1'])."""

    model_type = "alpamayo_r1"


__all__ = ["Alpamayo15Config", "AlpamayoR1Config"]
