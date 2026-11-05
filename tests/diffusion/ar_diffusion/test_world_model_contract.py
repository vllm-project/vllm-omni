# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Every AR-Diffusion world model keeps the shared session contract.

The unified data layers are ``Session`` (cross-request state) and
``ARDiffusionBatch`` (one execution envelope). Paged world models keep their own
batching and storage; only the Noisy cell strategy is a continuous single-request
execution. These checks pin which side each model is on, so adding a model cannot
silently drop the session lifecycle or drift into the wrong batch semantics.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

SESSION_LIFECYCLE = (
    "ar_diffusion_kv_cache_spec",
    "bind_ar_diffusion_state",
    "reset_ar_diffusion_session",
    "close_ar_diffusion_session",
)
NOISY_CELL_CAPABILITY = ("ar_diffusion_noisy_kv_spec", "bind_ar_diffusion_cell_context")


def _paged_pipelines():
    from vllm_omni.diffusion.models.abot_world.pipeline_abot_world import ABotWorldCausalPipeline
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
    from vllm_omni.diffusion.models.lingbot_world.pipeline import LingBotWorldCausalDMDPipeline
    from vllm_omni.diffusion.models.taomate_h3.pipeline import TaoMateH3Pipeline

    # Cosmos3 is a regular diffusion pipeline on the default engine, so it has no
    # AR-Diffusion session lifecycle to keep.
    return [
        ABotWorldCausalPipeline,
        DreamZeroPipeline,
        LingBotWorldCausalDMDPipeline,
        TaoMateH3Pipeline,
    ]


@pytest.mark.parametrize("pipeline_cls", _paged_pipelines(), ids=lambda cls: cls.__name__)
def test_paged_world_models_expose_the_session_lifecycle(pipeline_cls):
    for name in SESSION_LIFECYCLE:
        assert callable(getattr(pipeline_cls, name, None)), f"{pipeline_cls.__name__} is missing {name}()"


@pytest.mark.parametrize("pipeline_cls", _paged_pipelines(), ids=lambda cls: cls.__name__)
def test_paged_world_models_stay_out_of_the_noisy_cell_strategy(pipeline_cls):
    """A paged model keeps its own batching; it is never a continuous cell request."""
    for name in NOISY_CELL_CAPABILITY:
        assert not hasattr(pipeline_cls, name), f"{pipeline_cls.__name__} must not declare {name}()"


def test_causvid_is_the_noisy_cell_strategy_with_strict_outer_batch():
    """CausVid is the cell strategy: its session lives in the Noisy registry.

    The pipeline exposes the cell capability only. Cross-request state is held by
    ``NoisyKVSession`` (retained versions plus ``model_state``), not by pipeline
    lifecycle hooks, and one Noisy request runs its whole ``CellPlan``
    continuously, so it cannot be an independently schedulable batch member.
    """
    from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import CausVidWanPipeline

    for name in NOISY_CELL_CAPABILITY:
        assert callable(getattr(CausVidWanPipeline, name, None)), f"CausVidWanPipeline is missing {name}()"
    for name in SESSION_LIFECYCLE:
        assert not hasattr(CausVidWanPipeline, name), f"CausVidWanPipeline must not declare paged {name}()"
