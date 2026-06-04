# SPDX-License-Identifier: Apache-2.0
"""Unit tests for step-execution interface (no model weights required).

These tests verify the structural correctness of the step-execution
implementation without loading the full model. They can run quickly
in CI environments without GPU access to model weights.

Usage:
    python -m pytest tests/diffusion/models/sensenova_u1/test_sensenova_u1_step_execution_unit.py -v
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

PATCH_SIZE = 16
MERGE_SIZE = 1
HIDDEN_DIM = 32
IMAGE_SIZE = (64, 64)


def _make_pipeline():
    """Create a SenseNovaU1Pipeline stub without loading weights."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        SenseNovaU1Pipeline,
    )

    pipeline = object.__new__(SenseNovaU1Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.patch_size = PATCH_SIZE
    pipeline.merge_size = MERGE_SIZE
    pipeline._step_states = {}
    pipeline.top_cfg = SimpleNamespace(
        add_noise_scale_embedding=False,
        noise_scale_max_value=1.0,
        use_pixel_head=False,
        t_eps=0.02,
    )
    mock_lm = MagicMock()
    mock_lm.config = SimpleNamespace(hidden_size=HIDDEN_DIM)
    pipeline.language_model = mock_lm
    return pipeline


def _make_fake_state(request_id="test_req", **extra_args_overrides):
    """Create a SimpleNamespace mimicking DiffusionRequestState."""
    sampling = SimpleNamespace(
        height=IMAGE_SIZE[1],
        width=IMAGE_SIZE[0],
        num_inference_steps=4,
        seed=42,
        extra_args=extra_args_overrides,
    )
    return SimpleNamespace(
        request_id=request_id,
        prompts=["test prompt"],
        sampling=sampling,
        extra={},
        step_index=0,
        latents=None,
    )


# ============================================================
# Structural tests
# ============================================================


def test_class_declares_step_execution():
    """SenseNovaU1Pipeline declares supports_step_execution = True."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    assert hasattr(SenseNovaU1Pipeline, "supports_step_execution")
    assert SenseNovaU1Pipeline.supports_step_execution is True


def test_class_has_step_methods():
    """All four SupportsStepExecution methods are defined."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    for name in ("prepare_encode", "denoise_step", "step_scheduler", "post_decode"):
        method = getattr(SenseNovaU1Pipeline, name, None)
        assert method is not None, f"Missing method: {name}"
        assert callable(method), f"{name} is not callable"


def test_class_has_step_states_dict():
    """Pipeline __init__ creates _step_states dict."""
    import inspect

    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    source = inspect.getsource(SenseNovaU1Pipeline.__init__)
    assert "_step_states" in source


def test_protocol_isinstance_check():
    """Pipeline class satisfies SupportsStepExecution protocol structurally."""
    from vllm_omni.diffusion.models.interface import SupportsStepExecution
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    assert SenseNovaU1Pipeline.supports_step_execution is True
    for method_name in ("prepare_encode", "denoise_step", "step_scheduler", "post_decode"):
        assert callable(getattr(SenseNovaU1Pipeline, method_name, None))


def test_helper_methods_exist():
    """Helper methods for step execution are defined."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    helpers = [
        "_parse_request_from_state",
        "_build_t2i_caches",
        "_build_it2i_caches",
        "_step_denoise_single",
    ]
    for name in helpers:
        assert hasattr(SenseNovaU1Pipeline, name), f"Missing helper: {name}"


# ============================================================
# _parse_request_from_state tests
# ============================================================


def test_parse_request_default_values():
    """Default parameter values match original forward() defaults."""
    pipeline = _make_pipeline()
    state = _make_fake_state()
    p = pipeline._parse_request_from_state(state)

    assert p.cfg_scale == 4.0
    assert p.img_cfg_scale == 1.0
    assert p.cfg_norm == "none"
    assert p.timestep_shift == 3.0
    assert p.cfg_interval == (0.0, 1.0)
    assert p.num_steps == 4
    assert p.think_mode is False
    assert p.t_eps == 0.02
    assert p.batch_size == 1
    assert p.seed == 42


def test_parse_request_grid_factor_rounding():
    """Height/width not divisible by patch_size*merge_size gets rounded."""
    pipeline = _make_pipeline()
    state = _make_fake_state()
    state.sampling.height = 100
    state.sampling.width = 100

    p = pipeline._parse_request_from_state(state)

    grid_factor = PATCH_SIZE * MERGE_SIZE
    assert p.image_size[0] % grid_factor == 0
    assert p.image_size[1] % grid_factor == 0
    assert p.image_size == (96, 96)


# ============================================================
# step_scheduler tests
# ============================================================


def test_step_scheduler_advances_step_index():
    """step_scheduler increments both state.step_index and extra._current_step_index."""
    pipeline = _make_pipeline()

    image_pred = torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    timesteps = torch.linspace(1.0, 0.0, 5)
    extra = {
        "p": SimpleNamespace(image_size=IMAGE_SIZE),
        "ns": SimpleNamespace(
            timesteps=timesteps,
            merge_size=MERGE_SIZE,
        ),
        "_image_prediction": image_pred.clone(),
        "_current_step_index": 0,
    }
    pipeline._step_states["req1"] = extra

    state = SimpleNamespace(
        request_id="req1",
        step_index=0,
        latents=image_pred.clone(),
    )
    noise_pred = torch.randn(1, (IMAGE_SIZE[1] // PATCH_SIZE) * (IMAGE_SIZE[0] // PATCH_SIZE), PATCH_SIZE**2 * 3)

    pipeline.step_scheduler(state, noise_pred)

    assert state.step_index == 1
    assert extra["_current_step_index"] == 1


def test_step_scheduler_updates_image_prediction():
    """step_scheduler updates extra['_image_prediction'] and state.latents."""
    pipeline = _make_pipeline()

    image_pred = torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    original_pred = image_pred.clone()
    timesteps = torch.linspace(1.0, 0.0, 5)
    extra = {
        "p": SimpleNamespace(image_size=IMAGE_SIZE),
        "ns": SimpleNamespace(
            timesteps=timesteps,
            merge_size=MERGE_SIZE,
        ),
        "_image_prediction": image_pred,
        "_current_step_index": 0,
    }
    pipeline._step_states["req1"] = extra

    state = SimpleNamespace(
        request_id="req1",
        step_index=0,
        latents=image_pred.clone(),
    )
    noise_pred = torch.ones(1, (IMAGE_SIZE[1] // PATCH_SIZE) * (IMAGE_SIZE[0] // PATCH_SIZE), PATCH_SIZE**2 * 3)

    pipeline.step_scheduler(state, noise_pred)

    assert not torch.equal(extra["_image_prediction"], original_pred)
    assert state.latents is extra["_image_prediction"]


# ============================================================
# post_decode tests
# ============================================================


def test_post_decode_cleans_up_step_states():
    """post_decode removes the request from _step_states."""
    pipeline = _make_pipeline()
    pipeline._step_states["req1"] = {"caches": {}}

    image_pred = torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    state = SimpleNamespace(
        request_id="req1",
        latents=image_pred,
        extra={"caches": {}, "think_text": ""},
    )

    pipeline.post_decode(state)

    assert "req1" not in pipeline._step_states


def test_post_decode_returns_diffusion_output():
    """post_decode returns a DiffusionOutput instance."""
    from vllm_omni.diffusion.data import DiffusionOutput

    pipeline = _make_pipeline()
    pipeline._step_states["req1"] = {"caches": {}}

    image_pred = torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    state = SimpleNamespace(
        request_id="req1",
        latents=image_pred,
        extra={"caches": {}, "think_text": ""},
    )

    result = pipeline.post_decode(state)

    assert isinstance(result, DiffusionOutput)
    assert result.output is not None


def test_post_decode_passes_think_text():
    """post_decode includes think_text in custom_output when present."""
    pipeline = _make_pipeline()
    pipeline._step_states["req1"] = {"caches": {}}

    image_pred = torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    state = SimpleNamespace(
        request_id="req1",
        latents=image_pred,
        extra={"caches": {}, "think_text": "I should draw a cat."},
    )

    result = pipeline.post_decode(state)

    assert result.custom_output is not None
    assert result.custom_output["think_text"] == "I should draw a cat."


if __name__ == "__main__":
    tests = [
        ("Class declaration", test_class_declares_step_execution),
        ("Step methods exist", test_class_has_step_methods),
        ("_step_states in __init__", test_class_has_step_states_dict),
        ("Protocol conformance", test_protocol_isinstance_check),
        ("Helper methods exist", test_helper_methods_exist),
        ("Parse request defaults", test_parse_request_default_values),
        ("Parse request rounding", test_parse_request_grid_factor_rounding),
        ("Step scheduler advances index", test_step_scheduler_advances_step_index),
        ("Step scheduler updates prediction", test_step_scheduler_updates_image_prediction),
        ("Post decode cleanup", test_post_decode_cleans_up_step_states),
        ("Post decode returns output", test_post_decode_returns_diffusion_output),
        ("Post decode think text", test_post_decode_passes_think_text),
    ]

    for i, (label, fn) in enumerate(tests, 1):
        fn()
        print(f"[{i}/{len(tests)}] {label}: PASS")

    print(f"\nAll {len(tests)} unit tests passed.")
