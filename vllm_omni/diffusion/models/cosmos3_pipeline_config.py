# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3 disaggregated topology: one stage per Mixture-of-Transformers tower.

    stage 0 (reasoner, UND tower) --per-layer text K/V--> stage 1 (generator, GEN tower + VAE)

The UND tower runs once per request and contributes only its per-layer K/V, so a
GEN-only stage replaying K/V computed by a UND-only stage reproduces the
co-located result. See ``cosmos3/pipeline_cosmos3_disagg.py``.

This module lives outside the ``cosmos3`` package so the orchestrator can read
the topology and payload keys without importing the diffusion pipeline.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

#: Payload keys on the reasoner -> generator edge, shared by the tower pipelines
#: and the stage input processor.
COSMOS3_UND_KV_KEY = "cosmos3_und_kv"
COSMOS3_UND_META_KEY = "cosmos3_und_meta"

#: Contract identifier of the payload (``Cosmos3TextConditioning`` in
#: ``cosmos3/pipeline_cosmos3_disagg.py``); the generator rejects any other value.
COSMOS3_UND_SCHEMA = "cosmos3.text_conditioning/v1"

#: The reasoner warns when one request's K/V payload exceeds this size (~256 KiB
#: per token per branch in bf16). Not a hard limit.
COSMOS3_UND_PAYLOAD_WARN_MIB = 512.0

COSMOS3_ARCH = "Cosmos3OmniDiffusersPipeline"
COSMOS3_REASONER_ARCH = "Cosmos3ReasonerPipeline"
COSMOS3_GENERATOR_ARCH = "Cosmos3GeneratorPipeline"

_COSMOS3_INPUT_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.cosmos3"


COSMOS3_DISAGG_PIPELINE = PipelineConfig(
    model_type="cosmos3_omni_disagg",
    default_deploy_config_name="cosmos3_super_t2i_disagg.yaml",
    model_arch="Cosmos3ForConditionalGeneration",
    # No auto-detect hooks (and no diffusers_class_name): this topology shares its
    # checkpoint with the co-located deployment and is selected only by an explicit
    # ``pipeline: cosmos3_omni_disagg`` in the deploy YAML.
    hf_architectures=(),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="reasoner",
            # The UND tower is a prologue inside the diffusion pipeline, not an AR engine.
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            owns_tokenizer=True,
            requires_multimodal_data=False,
            final_output=False,
            model_arch=COSMOS3_REASONER_ARCH,
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="generator",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            requires_multimodal_data=False,
            final_output=True,
            final_output_type="image",
            model_arch=COSMOS3_GENERATOR_ARCH,
            custom_process_input_func=f"{_COSMOS3_INPUT_PROCESSOR}.reasoner2generator",
            # The K/V travels in the stage payload, not the AR KV-transfer path.
            omni_kv_config={"need_recv_cache": False},
        ),
    ),
)
