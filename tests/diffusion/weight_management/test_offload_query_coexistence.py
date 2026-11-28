# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Coexistence tests for unified-mode offload query functions (RFC 01-3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.offloader import config as offload_cfg
from vllm_omni.diffusion.offloader.config import (
    DEFAULT_OFFLOAD_COMPONENTS,
    DIT_COMPONENT,
    TEXT_ENCODER_COMPONENT,
    DLOTransfer,
    OffloadStrategy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _unified_cfg(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "diffusion_weight_stack": "unified",
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "enable_distributed_layerwise_offload": False,
        "diffusion_offload_config": None,
        "dlo_use_allgather": True,
        "dlo_resident_layers": 0,
        "pin_cpu_memory": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_cpu_offload": True},
        {"enable_layerwise_offload": True},
        {"enable_distributed_layerwise_offload": True},
        {"enable_distributed_layerwise_offload": True, "dlo_use_allgather": False, "dlo_resident_layers": 4},
        {"diffusion_offload_config": {"mode": "module", "components": ["dit", "text_encoder"]}},
        {
            "diffusion_offload_config": {
                "mode": "layer",
                "components": ["dit", "text_encoder"],
                "layer_options": {
                    "dit": {"weight_transfer": "allgather"},
                    "text_encoder": {"weight_transfer": "allgather"},
                },
            }
        },
        # Invalid legacy offload config must also be ignored without validation in unified mode.
        {"diffusion_offload_config": {"invalid_key": True}},
        {"pin_cpu_memory": False, "enable_cpu_offload": True},
    ],
)
def test_unified_queries_match_legacy_no_offload(overrides):
    legacy_none = SimpleNamespace(
        diffusion_weight_stack="legacy",
        pin_cpu_memory=overrides.get("pin_cpu_memory", True),
    )
    unified = _unified_cfg(**overrides)

    expected = offload_cfg.resolve_offload(legacy_none)
    actual = offload_cfg.resolve_offload(unified)

    assert actual == expected
    assert actual.strategy is OffloadStrategy.NONE
    assert actual.public is None
    assert actual.components == DEFAULT_OFFLOAD_COMPONENTS
    assert actual.transfers[DIT_COMPONENT] is DLOTransfer.ALLGATHER
    assert actual.transfers[TEXT_ENCODER_COMPONENT] is DLOTransfer.RANK_LOCAL
    assert actual.resident_layers == 0
    assert not hasattr(unified, "_resolved_diffusion_offload")

    assert offload_cfg.resolve_offload_strategy(unified) is OffloadStrategy.NONE
    assert offload_cfg.offload_enabled(unified) is False
    assert offload_cfg.offload_streams_blocks(unified) is False
    assert offload_cfg.should_offload_component(unified, DIT_COMPONENT) is False
    assert offload_cfg.should_offload_component(unified, TEXT_ENCODER_COMPONENT) is False
    assert offload_cfg.any_selected_component_uses_allgather(unified) is False
    assert offload_cfg.selected_offload_components(unified) == offload_cfg.selected_offload_components(legacy_none)
    assert offload_cfg.component_uses_allgather(unified, DIT_COMPONENT) == offload_cfg.component_uses_allgather(
        legacy_none, DIT_COMPONENT
    )
    with pytest.raises(ValueError, match="not selected"):
        offload_cfg.component_uses_allgather(unified, TEXT_ENCODER_COMPONENT)


def test_unified_materialize_writes_no_fields():
    unified = _unified_cfg(enable_cpu_offload=True, pin_cpu_memory=False)
    before = dict(vars(unified))
    assert offload_cfg.materialize_legacy_offload_flags(unified) is OffloadStrategy.NONE
    assert vars(unified) == before
