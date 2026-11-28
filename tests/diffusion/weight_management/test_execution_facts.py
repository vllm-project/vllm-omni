# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for ExecutionOffloadFacts and engine/executor fact readers (RFC 01-3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.offloader.config import (
    TEXT_ENCODER_COMPONENT,
    OffloadStrategy,
    any_selected_component_uses_allgather,
    resolve_offload,
    resolve_offload_strategy,
)
from vllm_omni.diffusion.weight_management.interfaces import ExecutionOffloadFacts
from vllm_omni.diffusion.weight_management.startup import execution_offload_facts

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"enable_cpu_offload": True},
        {"enable_layerwise_offload": True},
        {"enable_distributed_layerwise_offload": True},
        {"enable_distributed_layerwise_offload": True, "dlo_use_allgather": False},
        {"diffusion_offload_config": {"mode": "module", "components": ["dit", "text_encoder"]}},
        {"diffusion_offload_config": {"mode": "layer", "components": ["dit"]}},
        {
            "diffusion_offload_config": {
                "mode": "layer",
                "components": ["dit", "text_encoder"],
                "layer_options": {
                    "dit": {"weight_transfer": "allgather"},
                    "text_encoder": {"weight_transfer": "rank-local"},
                },
            }
        },
        {
            "diffusion_offload_config": {
                "mode": "layer",
                "components": ["dit", "text_encoder"],
                "layer_options": {
                    "dit": {"weight_transfer": "rank-local"},
                    "text_encoder": {"weight_transfer": "allgather"},
                },
            }
        },
    ],
)
def test_legacy_execution_facts_match_original_expressions(kwargs):
    cfg = OmniDiffusionConfig(**kwargs)
    resolved = resolve_offload(cfg)
    expected = ExecutionOffloadFacts(
        weight_collectives=any_selected_component_uses_allgather(cfg),
        text_encoder_allgather=resolved.offloads(TEXT_ENCODER_COMPONENT)
        and resolved.uses_allgather(TEXT_ENCODER_COMPONENT),
        whole_component_offload=resolve_offload_strategy(cfg) is OffloadStrategy.MODEL_LEVEL,
    )
    assert execution_offload_facts(cfg) == expected


def test_unified_execution_facts_are_false_before_rfc05():
    cfg = OmniDiffusionConfig(diffusion_weight_stack="unified")
    assert execution_offload_facts(cfg) == ExecutionOffloadFacts(
        weight_collectives=False,
        text_encoder_allgather=False,
        whole_component_offload=False,
    )


def test_engine_and_executor_helpers_read_execution_facts():
    from vllm_omni.diffusion.data import uses_rank_local_dp_concurrency
    from vllm_omni.diffusion.sched.request_scheduler import uses_text_encoder_allgather

    dp2 = SimpleNamespace(data_parallel_size=2)
    dlo_te = OmniDiffusionConfig(
        diffusion_offload_config={
            "mode": "layer",
            "components": ["dit", "text_encoder"],
            "layer_options": {
                "dit": {"weight_transfer": "allgather"},
                "text_encoder": {"weight_transfer": "allgather"},
            },
        },
    )
    dlo_te.parallel_config = dp2  # type: ignore[assignment]
    unified = OmniDiffusionConfig(diffusion_weight_stack="unified")
    unified.parallel_config = dp2  # type: ignore[assignment]

    assert uses_rank_local_dp_concurrency(dlo_te) is True
    assert uses_text_encoder_allgather(dlo_te) is True

    assert uses_rank_local_dp_concurrency(unified) is False
    assert uses_text_encoder_allgather(unified) is False
