# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Qwen3-ASR pipeline registration and deploy config.

Covers the single-stage text pipeline in the central registry and the deploy
yaml's 310P availability knobs (fp16, audio input bound, decode-only graphs).
"""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import StageExecutionType, resolve_deploy_yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ASR_ARCH = "Qwen3ASRForConditionalGeneration"


def test_qwen3_asr_is_a_single_stage_text_pipeline():
    pipeline = OMNI_PIPELINES["qwen3_asr"]
    assert pipeline.model_type == "qwen3_asr"
    assert pipeline.model_arch == _ASR_ARCH
    assert pipeline.hf_architectures == (_ASR_ARCH,)
    assert pipeline.default_deploy_config_name == "qwen3_asr.yaml"
    assert len(pipeline.stages) == 1

    stage = pipeline.stages[0]
    assert stage.stage_id == 0
    assert stage.model_stage == "asr"
    assert stage.execution_type is StageExecutionType.LLM_AR
    assert stage.input_sources == ()
    assert stage.final_output is True
    assert stage.final_output_type == "text"
    assert stage.engine_output_type == "text"
    assert stage.owns_tokenizer is True
    assert stage.requires_multimodal_data is True
    assert stage.sampling_constraints == {"detokenize": True}


def test_qwen3_asr_deploy_config():
    deploy = resolve_deploy_yaml(get_deploy_config_path("qwen3_asr.yaml"))
    assert deploy["pipeline"] == "qwen3_asr"
    assert deploy["async_chunk"] is False
    # No pipeline-wide dtype: the checkpoint ships no torch_dtype, so `auto`
    # keeps the plain-vLLM default on non-NPU platforms.
    assert "dtype" not in deploy

    stage = deploy["stages"][0]
    assert stage["limit_mm_per_prompt"] == {"audio": 1}
    assert stage["skip_mm_profiling"] is True
    assert stage["default_sampling_params"]["detokenize"] is True

    # 310P has no bf16 and the checkpoint ships no torch_dtype: fp16 is pinned
    # only for the NPU platform, next to a graph mode 310P can capture.
    npu_stage = deploy["platforms"]["npu"]["stages"][0]
    assert npu_stage["dtype"] == "float16"
    assert npu_stage["compilation_config"] == {"cudagraph_mode": "FULL_DECODE_ONLY"}
