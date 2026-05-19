# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for Gr00tN1d7Config (F1).

Validates the HF config registration and that AutoConfig.from_pretrained
resolves the GR00T-N1.7-3B local checkpoint to a Gr00tN1d7Config with the
pinned field values from config.json.
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# Local checkpoint path is environment-specific.  Set
# ``GR00T_CHECKPOINT_DIR`` to a directory containing config.json for the
# `test_config_loads_from_local_checkpoint` end-to-end check; otherwise
# that single test is skipped.
GR00T_LOCAL_PATH = os.environ.get("GR00T_CHECKPOINT_DIR", "")


def test_config_class_is_registered():
    from transformers import CONFIG_MAPPING

    import vllm_omni.transformers_utils.configs  # noqa: F401  (side-effect: register)
    from vllm_omni.transformers_utils.configs.gr00t import Gr00tN1d7Config

    assert "Gr00tN1d7" in CONFIG_MAPPING
    assert CONFIG_MAPPING["Gr00tN1d7"] is Gr00tN1d7Config


def test_config_default_values():
    from vllm_omni.transformers_utils.configs.gr00t import Gr00tN1d7Config

    cfg = Gr00tN1d7Config(model_name="")  # empty name skips backbone overlay
    assert cfg.action_horizon == 40
    assert cfg.max_action_dim == 132
    assert cfg.max_state_dim == 132
    assert cfg.max_num_embodiments == 32
    assert cfg.select_layer == 16
    assert cfg.num_inference_timesteps == 4
    assert cfg.backbone_embedding_dim == 2048
    assert cfg.hidden_size == 1024
    assert cfg.model_type == "Gr00tN1d7"


def test_config_roundtrip_preserves_fields():
    from vllm_omni.transformers_utils.configs.gr00t import Gr00tN1d7Config

    cfg = Gr00tN1d7Config(model_name="")
    payload = cfg.to_dict()
    cfg2 = Gr00tN1d7Config(**payload)
    assert cfg2.action_horizon == cfg.action_horizon
    assert cfg2.diffusion_model_cfg == cfg.diffusion_model_cfg
    assert cfg2.vl_self_attention_cfg == cfg.vl_self_attention_cfg


@pytest.mark.skipif(
    not GR00T_LOCAL_PATH or not os.path.isdir(GR00T_LOCAL_PATH),
    reason="Set GR00T_CHECKPOINT_DIR to a local GR00T-N1.7-3B checkout to enable this test",
)
def test_config_loads_from_local_checkpoint():
    """AutoConfig must resolve the local checkpoint to Gr00tN1d7Config."""
    from transformers import AutoConfig

    import vllm_omni.transformers_utils.configs  # noqa: F401
    from vllm_omni.transformers_utils.configs.gr00t import Gr00tN1d7Config

    cfg = AutoConfig.from_pretrained(GR00T_LOCAL_PATH, trust_remote_code=True)
    assert isinstance(cfg, Gr00tN1d7Config), type(cfg)

    with open(os.path.join(GR00T_LOCAL_PATH, "config.json")) as f:
        raw = json.load(f)

    assert cfg.action_horizon == raw["action_horizon"]
    assert cfg.max_action_dim == raw["max_action_dim"]
    assert cfg.max_state_dim == raw["max_state_dim"]
    assert cfg.max_num_embodiments == raw["max_num_embodiments"]
    assert cfg.select_layer == raw["select_layer"]
    assert cfg.num_inference_timesteps == raw["num_inference_timesteps"]
    assert cfg.backbone_embedding_dim == raw["backbone_embedding_dim"]
    assert cfg.hidden_size == raw["hidden_size"]
    assert cfg.model_name == raw["model_name"]
    assert cfg.diffusion_model_cfg == raw["diffusion_model_cfg"]
    assert cfg.vl_self_attention_cfg == raw["vl_self_attention_cfg"]
