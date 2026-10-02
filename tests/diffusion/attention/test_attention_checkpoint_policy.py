# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint defaults, complete deployment overrides and late metadata loading."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig, TransformerConfig

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def policy():
    return {
        "schema_version": 1,
        "config": {
            "presets": {"dense": "TORCH_SDPA"},
            "layouts": {"first": {"default": "dense"}, "last": {"default": "dense"}},
            "schedule": {
                "coordinate": "step_index",
                "phases": [
                    {"until": 2, "layout": "first"},
                    {"until": None, "layout": "last"},
                ],
            },
        },
    }


def metadata(value):
    return TransformerConfig.from_dict({"runtime": {"attention_strategy": value}})


@pytest.mark.parametrize("late", [False, True])
def test_checkpoint_policy_is_loaded_before_execution(late):
    raw = policy()
    saved = deepcopy(raw)
    cfg = OmniDiffusionConfig(
        model="unused", enforce_eager=True, tf_model_config=TransformerConfig() if late else metadata(raw)
    )
    if late:
        cfg.set_tf_model_config(metadata(raw))
    assert cfg.attention_policy_source == "checkpoint"
    assert cfg.diffusion_attention_config.strategy.phases == ((2, "first"), (None, "last"))
    assert raw == saved
    # Re-reading the metadata preserves its source and does not become a runtime override.
    cfg.set_tf_model_config(metadata(raw))
    assert cfg.attention_policy_source == "checkpoint"
    cfg.set_tf_model_config(TransformerConfig())
    assert cfg.attention_policy_source == "default"
    assert cfg.diffusion_attention_config.strategy is None


@pytest.mark.parametrize(
    "override",
    [
        {"default": "TORCH_SDPA"},
        {"per_role": {"self": "TORCH_SDPA"}},
        {"presets": {"base": "TORCH_SDPA"}, "layout": {"default": "base"}},
        {"checkpoint_policy": "ignore"},
    ],
)
@pytest.mark.parametrize("checkpoint", [{"schema_version": 999}, policy()])
def test_runtime_override_replaces_checkpoint_even_if_invalid(override, checkpoint):
    cfg = OmniDiffusionConfig(model="unused", enforce_eager=True, diffusion_attention_config=override)
    cfg.set_tf_model_config(metadata(checkpoint))
    assert asdict(cfg.diffusion_attention_config) == asdict(AttentionConfig(**override))
    assert cfg.attention_policy_source == ("runtime_disabled" if "checkpoint_policy" in override else "runtime")


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"schema_version": 2, "config": {}},
        {"schema_version": True, "config": {}},
        {"schema_version": 1, "config": {}},
        {"schema_version": 1, "config": {"default": "TORCH_SDPA"}},
        {"schema_version": 1, "config": {"presets": {"base": "TORCH_SDPA"}, "layout": {"default": "unknown"}}},
    ],
)
def test_invalid_checkpoint_policy_fails_closed(raw):
    cfg = OmniDiffusionConfig(model="unused", enforce_eager=True)
    with pytest.raises(ValueError, match="[Cc]heckpoint|preset"):
        cfg.set_tf_model_config(metadata(raw))


def test_checkpoint_policy_cannot_bypass_execution_validation():
    cfg = OmniDiffusionConfig(model="unused", cache_backend="cache_dit", enforce_eager=True)
    with pytest.raises(ValueError, match="cache acceleration"):
        cfg.set_tf_model_config(metadata(policy()))


def test_environment_backend_overrides_checkpoint(monkeypatch):
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    cfg = OmniDiffusionConfig(model="unused", enforce_eager=True)
    cfg.set_tf_model_config(metadata(policy()))
    assert cfg.attention_policy_source == "runtime"
    assert cfg.diffusion_attention_config.strategy is None
    assert cfg.diffusion_attention_config.default.backend == "TORCH_SDPA"


def test_checkpoint_policy_errors_do_not_trigger_discovery_fallback(monkeypatch):
    from vllm_omni.diffusion.attention.checkpoint import CheckpointAttentionPolicyError
    from vllm_omni.diffusion.utils import hf_utils

    monkeypatch.setattr(
        hf_utils, "get_diffusion_model_index", lambda *args, **kwargs: {"_class_name": "Cosmos3OmniDiffusersPipeline"}
    )

    def load_component(cfg):
        cfg.set_tf_model_config(metadata({"schema_version": 999}))

    monkeypatch.setattr(OmniDiffusionConfig, "_load_component_transformer_config", load_component)
    cfg = OmniDiffusionConfig(model="unused", enforce_eager=True)
    with pytest.raises(CheckpointAttentionPolicyError, match="schema_version"):
        cfg.enrich_config()


def test_absent_policy_preserves_existing_configuration():
    cfg = OmniDiffusionConfig(model="unused", enforce_eager=True)
    original = cfg.diffusion_attention_config
    # Changes made after startup must survive unrelated metadata reloads.
    original.per_role["self"] = AttentionConfig(default="TORCH_SDPA").default
    cfg.set_tf_model_config(TransformerConfig.from_dict({"runtime": {"unrelated": True}}))
    assert cfg.diffusion_attention_config is original
    assert cfg.diffusion_attention_config.per_role["self"].backend == "TORCH_SDPA"
    assert cfg.attention_policy_source == "runtime"


@pytest.mark.parametrize("value", [None, True, "typo"])
def test_invalid_checkpoint_policy_option(value):
    with pytest.raises(ValueError, match="checkpoint_policy"):
        AttentionConfig(checkpoint_policy=value)
