# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK pipeline topology (frozen).

Stage 0: Encoder — frozen Qwen2.5-Omni thinker, prefill only, emits the
         learned layer fusion as ``multimodal_outputs["hidden_states"]["output"]``
Stage 1: DiT     — rectified-flow transformer plus the BigVGAN-flow VAE,
         emits a 24 kHz waveform
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.auk"

AUK_PIPELINE = PipelineConfig(
    model_type="auk",
    default_deploy_config_name="auk.yaml",
    model_arch="AuKForConditionalGeneration",
    hf_architectures=("AuKForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="encoder",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            model_arch="AuKForConditionalGeneration",
            # No model_subdir: tools/prepare_auk_checkpoint.py assembles one
            # flat directory, so the encoder finds the thinker shards, the
            # tokenizer, the audio processor and the layer-fusion tensors in
            # auk.safetensors all under the pipeline root.
            # One prefill step produces the text condition; the sampled token
            # is discarded, and detokenizing it would only cost latency.
            sampling_constraints={"max_tokens": 1, "temperature": 0.0, "detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            requires_multimodal_data=True,
            final_output=True,
            final_output_type="audio",
            model_arch="AuKPipeline",
            custom_process_input_func=f"{_PROC}.encoder2dit",
            omni_kv_config={"need_recv_cache": False},
            # Single replica, and the whole ODE runs inside one forward, so
            # the stage stays in the orchestrator process.
            inline_diffusion=True,
        ),
    ),
)
