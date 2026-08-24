# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the SenseNova-Vision think two-stage pipeline config.

Covers the ``sensenova_vision_think`` structural-topology invariants that the
``BAGEL_THINK_PIPELINE`` mirror must satisfy:

* Stage 0 wires ``expand_cfg_prompts_think`` (not ``expand_cfg_prompts``).
* Stage 0's ``omni_kv_config`` has NO ``kv_transfer_criteria`` (so KV transfer
  happens after EOS, not after prefill) and still ``need_send_cache=True``.
* Stage 1 has ``need_recv_cache=True`` and a ``custom_process_input_func``
  wired to the sensenova prompt-utils text bridge.
* The registry resolves ``sensenova_vision_think`` to the same config instance.
"""

from __future__ import annotations

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType
from vllm_omni.model_executor.models.sensenova_vision.pipeline import (
    SENSENOVA_VISION_PIPELINE,
    SENSENOVA_VISION_THINK_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_think_pipeline_stage0_uses_think_expander() -> None:
    """Stage 0 must expand with ``expand_cfg_prompts_think``."""
    stage0 = SENSENOVA_VISION_THINK_PIPELINE.get_stage(0)
    assert stage0 is not None
    assert stage0.prompt_expand_func == (
        "vllm_omni.model_executor.stage_input_processors.bagel.expand_cfg_prompts_think"
    )


def test_think_pipeline_stage0_omits_kv_transfer_criteria() -> None:
    """Stage 0 transfers after EOS: no ``kv_transfer_criteria`` in omni_kv_config."""
    stage0 = SENSENOVA_VISION_THINK_PIPELINE.get_stage(0)
    assert stage0 is not None
    kv = stage0.omni_kv_config or {}
    assert kv.get("need_send_cache") is True
    assert "kv_transfer_criteria" not in kv


def test_base_pipeline_keeps_kv_transfer_criteria() -> None:
    """The plain sensenova_vision pipeline must be left untouched (prefill transfer)."""
    stage0 = SENSENOVA_VISION_PIPELINE.get_stage(0)
    assert stage0 is not None
    kv = stage0.omni_kv_config or {}
    assert kv.get("kv_transfer_criteria") == {"type": "prefill_finished"}


def test_think_pipeline_stage1_recvs_cache_and_wires_text_bridge() -> None:
    """Stage 1 must receive KV and lift the AR think text via the bridge."""
    stage1 = SENSENOVA_VISION_THINK_PIPELINE.get_stage(1)
    assert stage1 is not None
    assert stage1.execution_type == StageExecutionType.DIFFUSION
    assert stage1.input_sources == (0,)
    assert (stage1.omni_kv_config or {}).get("need_recv_cache") is True
    assert stage1.custom_process_input_func == (
        "vllm_omni.model_executor.models.sensenova_vision.prompt_utils.bridge_think_text_to_image"
    )


def test_think_pipeline_validate() -> None:
    """Topology must be structurally valid."""
    assert SENSENOVA_VISION_THINK_PIPELINE.validate() == []


def test_registry_resolves_sensenova_vision_think() -> None:
    """The registry maps ``sensenova_vision_think`` to the new config and it round-trips."""
    assert OMNI_PIPELINES["sensenova_vision_think"] is SENSENOVA_VISION_THINK_PIPELINE
    resolved = resolve_pipeline_config("sensenova_vision_think")
    assert isinstance(resolved, PipelineConfig)
    assert resolved is SENSENOVA_VISION_THINK_PIPELINE
    assert resolved.default_deploy_config_name == "sensenova_vision_think.yaml"
