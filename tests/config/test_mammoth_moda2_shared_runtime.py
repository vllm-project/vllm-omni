# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS
from vllm_omni.model_executor.models.mammoth_moda2.pipeline import (
    MAMMOTH_MODA2_AR_PIPELINE,
    MAMMOTH_MODA2_PIPELINE,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS


def test_mammothmoda2_generation_stage_uses_shared_diffusion_runtime() -> None:
    ar_stage, dit_stage = MAMMOTH_MODA2_PIPELINE.stages
    assert ar_stage.execution_type is StageExecutionType.LLM_AR
    assert dit_stage.execution_type is StageExecutionType.DIFFUSION
    assert dit_stage.model_arch == "MammothModa2DiTPipeline"
    assert dit_stage.custom_process_input_func.endswith(".ar2diffusion")
    assert dit_stage.omni_kv_config == {"need_recv_cache": False}
    assert dit_stage.input_sources == (0,)
    assert dit_stage.final_output is True
    assert dit_stage.final_output_type == "image"


def test_mammothmoda2_dit_is_registered_only_with_diffusion_runtime() -> None:
    assert _DIFFUSION_MODELS["MammothModa2DiTPipeline"] == (
        "mammoth_moda2",
        "pipeline_mammothmoda2_dit",
        "MammothModa2DiTPipeline",
    )
    assert "MammothModa2DiTPipeline" not in _OMNI_MODELS


def test_mammothmoda2_ar_only_topology_is_unchanged() -> None:
    assert len(MAMMOTH_MODA2_AR_PIPELINE.stages) == 1
    stage = MAMMOTH_MODA2_AR_PIPELINE.stages[0]
    assert stage.execution_type is StageExecutionType.LLM_AR
    assert stage.final_output_type == "text"
