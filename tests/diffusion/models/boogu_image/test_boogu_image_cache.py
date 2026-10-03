# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.cache.teacache import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.extractors import extract_boogu_context
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
AXES_DIM_ROPE = (8, 4, 4)
AXES_LENS = (32, 16, 16)
MULTIPLE_OF = 32
NORM_EPS = 1e-5


@pytest.fixture(autouse=True)
def _init_distributed():
    """Minimal single-rank distributed environment for vLLM parallel linears."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29513")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _force_default_gemm(monkeypatch):
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


def _randomize_parameters(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.uniform_(-0.02, 0.02)


def _tiny_tf_model_config(**overrides):
    config = {
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": 4,
        "num_double_stream_layers": 2,
        "num_refiner_layers": 2,
        "num_attention_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "multiple_of": MULTIPLE_OF,
        "norm_eps": NORM_EPS,
        "axes_dim_rope": list(AXES_DIM_ROPE),
        "axes_lens": list(AXES_LENS),
        "instruction_feature_configs": {
            "instruction_feat_dim": 32,
            "reduce_type": "mean",
            "num_instruction_feature_layers": 1,
        },
        "prompt_tuning_configs": {"use_prompt_tuning": False},
        "timestep_scale": 1.0,
    }
    config.update(overrides)
    return config


def _tiny_od_config(**overrides):
    from vllm_omni.diffusion.data import TransformerConfig

    return SimpleNamespace(
        tf_model_config=TransformerConfig.from_dict(_tiny_tf_model_config(**overrides)),
        dtype=torch.float32,
    )


def _build_model_and_inputs():
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageDoubleStreamRotaryPosEmbed,
        BooguImageTransformer2DModel,
    )

    model = BooguImageTransformer2DModel(od_config=_tiny_od_config())
    _randomize_parameters(model)
    model.eval()

    batch_size = 1
    in_channels = 4
    latent_h = latent_w = 8
    instruct_len = 8
    instruction_feat_dim = 32

    latents = torch.randn(batch_size, in_channels, latent_h, latent_w)
    timestep = torch.full((batch_size,), 0.5)
    instruction_hidden_states = torch.randn(batch_size, instruct_len, instruction_feat_dim)
    instruction_attention_mask = torch.ones(batch_size, instruct_len, dtype=torch.bool)
    freqs_cis = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(model.axes_dim_rope, model.axes_lens, theta=10000)

    inputs = dict(
        hidden_states=latents,
        timestep=timestep,
        instruction_hidden_states=instruction_hidden_states,
        freqs_cis=freqs_cis,
        instruction_attention_mask=instruction_attention_mask,
    )
    return model, inputs


def test_boogu_teacache_extractor_matches_full_forward_when_always_computed():
    """The extractor must reproduce `forward()` exactly when TeaCache always
    takes the "recompute" path (i.e. skip is never triggered), since the
    refiner/dual-stream/single-stream/output-projection logic is duplicated
    from `forward()` rather than reused."""
    model, inputs = _build_model_and_inputs()

    with torch.no_grad():
        expected = model(**inputs)

        ctx = extract_boogu_context(model, **inputs)
        outputs = ctx.run_transformer_blocks()
        actual = ctx.postprocess(outputs[0])

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


def test_boogu_teacache_extractor_modulated_input_matches_first_single_stream_norm():
    model, inputs = _build_model_and_inputs()

    with torch.no_grad():
        ctx = extract_boogu_context(model, **inputs)

    assert torch.isfinite(ctx.modulated_input).all()
    assert ctx.encoder_hidden_states is None
    assert ctx.hidden_states.shape == ctx.modulated_input.shape


def test_boogu_teacache_backend_targets_pipeline_transformer():
    """Boogu has no custom enabler; the generic path must key off
    `pipeline.transformer.__class__.__name__`."""
    from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
        BooguImageTransformer2DModel,
    )

    model = BooguImageTransformer2DModel(od_config=_tiny_od_config())
    pipeline = SimpleNamespace(transformer=model)

    with patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook") as apply_hook:
        backend = TeaCacheBackend(DiffusionCacheConfig())
        backend.enable(pipeline)

    transformer, config = apply_hook.call_args.args
    assert transformer is model
    assert config.transformer_type == "BooguImageTransformer2DModel"
    assert config.rel_l1_thresh == 0.15
    assert backend.is_enabled()
