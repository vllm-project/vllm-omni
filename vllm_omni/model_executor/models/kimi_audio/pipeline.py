# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-Audio pipeline topology (frozen).

Stage 0: Fused thinker — Whisper + VQ-Adaptor + Qwen2 LLM + 6-layer MIMO
  branch; emits text and audio semantic tokens jointly.
Stage 1: Code2Wav — audio detokenizer (flow-matching DiT + BigVGAN) turns
  semantic tokens into a 24 kHz waveform.

Sync vs. async-chunk streaming is selected by the YAML's ``async_chunk``
flag; both processors are declared here side-by-side so one PipelineConfig
covers both deployments (mirrors cosyvoice3/pipeline.py).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.kimi_audio"

KIMI_AUDIO_PIPELINE = PipelineConfig(
    model_type="kimi_audio",
    model_arch="KimiAudioForConditionalGeneration",
    # HF config has model_type=null; route by architecture name instead.
    hf_architectures=("MoonshotKimiaForCausalLM",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="fused_thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.kimi2code2wav_async_chunk"),
            # Model config has `eos_token_ids: [151644, 151645]` (plural,
            # non-standard) so vLLM parses `eos_token_id=None` and has no
            # stop token by default. For audio-out (output_type="both")
            # upstream only stops on audio-head EOD — compute_logits boosts
            # 151644 once the MIMO head emits msg_end/media_end so vLLM
            # terminates cleanly.
            #   151644 - [EOS], boosted by compute_logits on audio-head EOD
            #   151645 - <|im_msg_end|>
            # 151667 (kimia_text_eos) is intentionally NOT listed — it
            # fires on the text stream well before the audio stream has
            # produced usable codec tokens and would cut audio generation
            # short. Callers doing text-only output should override.
            sampling_constraints={"detokenize": True, "stop_token_ids": [151644, 151645]},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.kimi2code2wav",
            sampling_constraints={"detokenize": False},
        ),
    ),
)
