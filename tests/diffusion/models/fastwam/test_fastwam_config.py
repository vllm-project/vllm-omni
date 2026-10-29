# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastWAM presets and the checkpoint/preset consistency checks."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.fastwam.config import (
    DEFAULT_VARIANT,
    PRESETS,
    ActionCodecConfig,
    resolve_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CAE_ACTION_VAE_CONFIG = {
    "action_dim": 7,
    "action_horizon": 32,
    "num_tokens": 8,
    "latent_dim": 16,
    "encoder_hidden": 128,
    "decoder_hidden": 128,
    "video_cond_dim": 0,
    "variational": True,
    "beta_kl": 1e-4,
    "free_bits": 0.1,
    "recon_loss": "mse",
    "num_suites": 4,
    "sigreg_lambda": 0.0,
}


def _action_expert_state(hidden: int, token_dim: int, ffn: int) -> dict[str, torch.Tensor]:
    return {
        "mixtures.action.action_encoder.weight": torch.empty(hidden, token_dim),
        "mixtures.action.blocks.0.ffn.0.weight": torch.empty(ffn, hidden),
    }


def test_default_is_validated_cae_preset():
    cfg = resolve_config({})
    assert DEFAULT_VARIANT == "cae_fastwam_libero" and cfg.variant == DEFAULT_VARIANT
    assert cfg.validated
    assert cfg.num_action_tokens == 8 and cfg.action_token_dim == 16 and cfg.action_shift == 5.0
    assert cfg.image_size == (224, 448)
    assert cfg.stats_file("libero_goal") == "stats/libero_goal.json"
    assert dict(cfg.suite_ids) == {"libero_spatial": 0, "libero_object": 1, "libero_goal": 2, "libero_10": 3}


def test_original_fastwam_preset_is_marked_unvalidated():
    cfg = resolve_config({"fastwam_variant": "fastwam_libero_uncond"})
    assert not cfg.validated
    assert cfg.action_codec is None and cfg.num_action_tokens == 32 and cfg.action_shift == 1.0


def test_unknown_variant_and_asset_overrides():
    with pytest.raises(ValueError, match="Unknown fastwam_variant"):
        resolve_config({"fastwam_variant": "nope"})
    cfg = resolve_config({"policy_checkpoint": "other.pt", "stats_path": "s/{suite}.json"})
    assert cfg.policy_checkpoint == "other.pt" and cfg.stats_file("x") == "s/x.json"


def test_checkpoint_shape_check_accepts_matching_preset():
    PRESETS["cae_fastwam_libero"].check_policy_state_dict(_action_expert_state(32, 16, 256))
    PRESETS["fastwam_libero_uncond"].check_policy_state_dict(_action_expert_state(1024, 7, 4096))


def test_checkpoint_shape_check_rejects_the_other_preset():
    # A FastWAM checkpoint served with the CAE preset would run the wrong shift silently.
    with pytest.raises(ValueError, match="fastwam_variant"):
        PRESETS["cae_fastwam_libero"].check_policy_state_dict(_action_expert_state(1024, 7, 4096))
    with pytest.raises(ValueError, match="fastwam_variant"):
        PRESETS["fastwam_libero_uncond"].check_policy_state_dict(_action_expert_state(32, 16, 256))
    with pytest.raises(ValueError, match="no FastWAM action expert"):
        PRESETS["cae_fastwam_libero"].check_policy_state_dict({})


def test_codec_config_from_checkpoint_and_validation():
    codec = ActionCodecConfig.from_checkpoint_config(CAE_ACTION_VAE_CONFIG)
    assert codec == ActionCodecConfig()
    assert PRESETS["cae_fastwam_libero"].with_codec(codec).action_codec == codec
    with pytest.raises(ValueError, match="latent_dim"):
        PRESETS["cae_fastwam_libero"].with_codec(ActionCodecConfig(latent_dim=8))
    with pytest.raises(ValueError, match="chunks"):
        PRESETS["cae_fastwam_libero"].with_codec(ActionCodecConfig(action_horizon=16, num_tokens=4))
    with pytest.raises(ValueError, match="video_cond_dim"):
        ActionCodecConfig.from_checkpoint_config({**CAE_ACTION_VAE_CONFIG, "video_cond_dim": 12288})
    with pytest.raises(ValueError, match="divisible"):
        ActionCodecConfig.from_checkpoint_config({**CAE_ACTION_VAE_CONFIG, "num_tokens": 5})
