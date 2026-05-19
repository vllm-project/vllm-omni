# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for F5: Gr00tN1d7Pipeline + weight-prefix routing + registry.

We don't instantiate the real Gr00tN1d7Pipeline.__init__ here (it requires
a full Cosmos-Reason2-2B-shaped Qwen3VL and a checkpoint dir).  Instead we
exercise the load_weights routing logic directly against an instance where
``backbone`` and ``action_head`` are stand-in modules whose state_dict keys
match the GR00T checkpoint prefix layout.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_pipeline_is_registered():
    from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

    assert "Gr00tN1d7Pipeline" in _DIFFUSION_MODELS
    folder, mod, cls = _DIFFUSION_MODELS["Gr00tN1d7Pipeline"]
    assert (folder, mod, cls) == ("gr00t", "pipeline_gr00t", "Gr00tN1d7Pipeline")


def test_registry_can_load_pipeline_class():
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    cls = DiffusionModelRegistry._try_load_model_cls("Gr00tN1d7Pipeline")
    assert cls is not None
    assert cls.__name__ == "Gr00tN1d7Pipeline"


# ---------------------------------------------------------------------------
# load_weights routing
# ---------------------------------------------------------------------------


class _BackboneStub(nn.Module):
    """Mimics the attribute path Gr00tN1d7Pipeline.load_weights expects:
    ``self.backbone.<key>`` for keys starting with ``backbone.``."""

    def __init__(self):
        super().__init__()
        # Stand-in for backbone.model.lm_head.weight
        self.model = nn.Module()
        self.model.lm_head = nn.Linear(4, 4, bias=False)
        # Stand-in for backbone.model.model.language_model.layers.0.input_layernorm.weight
        self.model.model = nn.Module()
        self.model.model.language_model = nn.Module()
        self.model.model.language_model.layers = nn.ModuleList(
            [nn.LayerNorm(4)]
        )


class _ActionHeadStub(nn.Module):
    """Mimics action_head.action_decoder.layer1.W / .b layout."""

    def __init__(self):
        super().__init__()
        self.action_decoder = nn.Module()
        self.action_decoder.layer1 = nn.Module()
        self.action_decoder.layer1.W = nn.Parameter(torch.zeros(3, 4, 4))
        self.action_decoder.layer1.b = nn.Parameter(torch.zeros(3, 4))


def _build_pipeline_with_stubs():
    """Build a Gr00tN1d7Pipeline instance with stubbed backbone + action_head
    so we can exercise load_weights without standing up Qwen3-VL."""
    from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline

    pipeline = Gr00tN1d7Pipeline.__new__(Gr00tN1d7Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.backbone = _BackboneStub()
    pipeline.action_head = _ActionHeadStub()
    pipeline.embodiment_name_to_id = {}
    return pipeline


def test_load_weights_routes_backbone_prefix():
    pipeline = _build_pipeline_with_stubs()

    weights = [
        (
            "backbone.model.lm_head.weight",
            torch.full((4, 4), 7.0),
        ),
        (
            "backbone.model.model.language_model.layers.0.weight",
            torch.full((4,), 3.0),
        ),
    ]
    loaded = pipeline.load_weights(weights)
    assert "backbone.model.lm_head.weight" in loaded
    assert "backbone.model.model.language_model.layers.0.weight" in loaded

    # Tensor actually landed on the stub
    actual = pipeline.backbone.model.lm_head.weight.detach()
    torch.testing.assert_close(actual, torch.full((4, 4), 7.0))


def test_load_weights_routes_action_head_prefix():
    pipeline = _build_pipeline_with_stubs()

    W = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
    b = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    loaded = pipeline.load_weights(
        [
            ("action_head.action_decoder.layer1.W", W),
            ("action_head.action_decoder.layer1.b", b),
        ]
    )
    assert "action_head.action_decoder.layer1.W" in loaded
    assert "action_head.action_decoder.layer1.b" in loaded
    torch.testing.assert_close(
        pipeline.action_head.action_decoder.layer1.W.detach(), W
    )
    torch.testing.assert_close(
        pipeline.action_head.action_decoder.layer1.b.detach(), b
    )


def test_load_weights_ignores_unknown_prefix(caplog):
    pipeline = _build_pipeline_with_stubs()
    weights = [
        ("foo.bar", torch.zeros(2, 2)),
        ("backbone.model.lm_head.weight", torch.zeros(4, 4)),
    ]
    loaded = pipeline.load_weights(weights)
    # Unknown keys are silently dropped (logged, not raised)
    assert "foo.bar" not in loaded
    assert "backbone.model.lm_head.weight" in loaded


# ---------------------------------------------------------------------------
# weights_sources
# ---------------------------------------------------------------------------


def test_weights_sources_format():
    """The ComponentSource list must be shape-compatible with
    DiffusersPipelineLoader (matches DreamZero's pattern)."""
    from vllm_omni.diffusion.model_loader.diffusers_loader import (
        DiffusersPipelineLoader,
    )
    from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline

    pipeline = Gr00tN1d7Pipeline.__new__(Gr00tN1d7Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._weights_sources = [
        DiffusersPipelineLoader.ComponentSource(
            model_or_path="/tmp/fake",
            subfolder=None,
            revision=None,
            prefix="",
            fall_back_to_pt=False,
            allow_patterns_overrides=["model-*.safetensors", "model.safetensors"],
        ),
    ]
    sources = pipeline.weights_sources
    assert len(sources) == 1
    s = sources[0]
    assert s.model_or_path == "/tmp/fake"
    assert s.allow_patterns_overrides == [
        "model-*.safetensors",
        "model.safetensors",
    ]


# Embodiment-id resolution is exercised inside `transform.encode` against
# the verbatim Isaac table; covered by tests in test_transform.py
# (`test_embodiment_id_decode_known_tag`,
# `test_embodiment_id_decode_unknown_tag_raises`).
