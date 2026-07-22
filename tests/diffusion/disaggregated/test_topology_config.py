# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config/topology resolution tests for the DreamZero disaggregated pipeline.

These import the real config + registry (vllm/transformers), so they are marked
``needs_runtime`` and skipped where those deps are unavailable. Run on the node:

    pytest tests/diffusion/disaggregated/test_topology_config.py -m needs_runtime

The torch-free topology *validation rules* are covered separately in
test_stage_roles.py.
"""

from __future__ import annotations

import pytest

try:
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
    from vllm_omni.config.stage_config import StageExecutionType
    from vllm_omni.model_executor.models.dreamzero.pipeline import (
        DREAMZERO_DISAGGREGATED_PIPELINE,
        DREAMZERO_PIPELINE,
        GENERIC_DIFFUSION_PROCESSOR,
    )
except Exception as exc:  # pragma: no cover - import-environment dependent
    pytest.skip(f"vllm_omni config runtime unavailable: {exc}", allow_module_level=True)

pytestmark = pytest.mark.needs_runtime


def test_single_stage_dreamzero_still_registered():
    assert OMNI_PIPELINES["dreamzero"] is DREAMZERO_PIPELINE
    assert len(DREAMZERO_PIPELINE.stages) == 1
    assert DREAMZERO_PIPELINE.stages[0].model_stage == "diffusion"
    assert DREAMZERO_PIPELINE.stages[0].final_output


def test_disaggregated_pipeline_registered():
    assert OMNI_PIPELINES["dreamzero_disaggregated"] is DREAMZERO_DISAGGREGATED_PIPELINE


def test_disaggregated_topology_roles_and_sources():
    stages = DREAMZERO_DISAGGREGATED_PIPELINE.stages
    assert [s.model_stage for s in stages] == ["encode", "denoise", "decode"]
    assert [tuple(s.input_sources) for s in stages] == [(), (0,), (1,)]
    assert [s.final_output for s in stages] == [False, False, True]
    assert stages[2].final_output_type == "image"
    for s in stages:
        assert s.execution_type == StageExecutionType.DIFFUSION


def test_disaggregated_uses_generic_processor():
    encode, denoise, decode = DREAMZERO_DISAGGREGATED_PIPELINE.stages
    assert encode.custom_process_next_stage_input_func == GENERIC_DIFFUSION_PROCESSOR
    assert denoise.custom_process_input_func == GENERIC_DIFFUSION_PROCESSOR
    assert denoise.custom_process_next_stage_input_func == GENERIC_DIFFUSION_PROCESSOR
    assert decode.custom_process_input_func == GENERIC_DIFFUSION_PROCESSOR
    # encode has no consumer hook; decode has no producer hook
    assert encode.custom_process_input_func is None
    assert decode.custom_process_next_stage_input_func is None


def test_disaggregated_topology_validates_clean():
    assert DREAMZERO_DISAGGREGATED_PIPELINE.validate() == []


def test_single_stage_topology_validates_clean():
    assert DREAMZERO_PIPELINE.validate() == []


def test_invalid_disaggregated_topology_fails_validation():
    from vllm_omni.config.stage_config import PipelineConfig, StagePipelineConfig

    # decode points at the encode stage (no denoise source) -> must error
    bad = PipelineConfig(
        model_type="dreamzero_bad",
        model_arch="DreamZeroPipeline",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="encode",
                execution_type=StageExecutionType.DIFFUSION,
                input_sources=(),
            ),
            StagePipelineConfig(
                stage_id=2,
                model_stage="decode",
                execution_type=StageExecutionType.DIFFUSION,
                input_sources=(0,),
                final_output=True,
            ),
        ),
    )
    errors = bad.validate()
    assert any("no upstream denoise source" in e for e in errors)
