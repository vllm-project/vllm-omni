# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax Music 3 pipeline topology: AR talker -> acoustic decoder.

Stage 0 ``minimax_music3_ar``: a Qwen3 backbone (the repo's ``language_model/``
subfolder is a plain ``Qwen3ForCausalLM``) plus a depth decoder, wrapped by a
class that owns its sampler. It emits 200-frame windows of frame hidden states
and the matching 8-codebook RVQ codes.

Stage 1 ``minimax_music3_acoustic``: flow-matching DiT plus DAV vocoder. Pure
compute, no KV cache, weights loaded from the repo's ``condition_encoder/``,
``transformer/`` and ``vocoder/`` subfolders. Output is 32 kHz stereo.

Classifier-free guidance is mandatory for this checkpoint: every request
decodes as a cond/uncond pair, and the uncond row is produced by
``prompt_expand_func``. The deploy YAMLs therefore carry
``default_sampling_params.extra_args`` with ``cfg_role`` and ``cfg_scale`` so
that ``has_sampling_extra_args`` is on for the stage and per-row extra args
reach the model's forward.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.minimax_music3"

# ``<|audio_end|>`` in the checkpoint tokenizer (pinned by
# ``models/minimax_music3/prompt.SPECIAL_TOKEN_IDS``). The talker's own sampler
# forces this at the frame budget; the stop id is the engine-side safety net.
MINIMAX_MUSIC3_AUDIO_END_TOKEN_ID = 151670

# MiniMax captions carry the full style description and run to roughly 5000
# characters, an order of magnitude past the 500-character serving default.
# ``tts_args`` is a pipeline-level stage extra, not a deploy-YAML field: it
# reaches the serving layer through ``StageConfig.yaml_extras``, which is where
# ``ServingSpeech._compute_max_instructions_length`` reads it from.
_MAX_INSTRUCTIONS_LENGTH = 20_000

MINIMAX_MUSIC3_PIPELINE = PipelineConfig(
    model_type="minimax_music3",
    model_arch="MiniMaxMusic3TalkerForConditionalGeneration",
    hf_architectures=("MiniMaxMusic3ForConditionalGeneration",),
    default_deploy_config_name="minimax_music3.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="minimax_music3_ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            # The repo root config.json only names the composite architecture;
            # the AR weights and the tokenizer live in their own subfolders.
            model_subdir="language_model",
            tokenizer_subdir="tokenizer",
            # detokenize: the sampled stream is audio codes, never text.
            # stop_token_ids: <|audio_end|> closes the song.
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [MINIMAX_MUSIC3_AUDIO_END_TOKEN_ID],
            },
            # CFG is mandatory here, so this expands on essentially every
            # request rather than only on an opt-in flag.
            prompt_expand_func=f"{_PROC}.expand_cfg_prompts",
            async_chunk_process_next_stage_input_func=f"{_PROC}.ar2acoustic_async_chunk",
            # retains_state_across_chunks stays False. It is read only by
            # OmniGenerationScheduler, so it is a no-op on an LLM_AR stage;
            # more to the point, stage 0 produces chunks rather than parking on
            # them, so it never holds a runner slot while waiting for one.
            extras={"tts_args": {"max_instructions_length": _MAX_INSTRUCTIONS_LENGTH}},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="minimax_music3_acoustic",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            # "audio" (not "latent"): the accumulator concatenates audio over
            # time, whereas the latent path would stack the two stereo channels.
            engine_output_type="audio",
            model_arch="MiniMaxMusic3AcousticForConditionalGeneration",
            # This stage reads the repo ROOT so it can reach condition_encoder/,
            # transformer/ and vocoder/, and the root ships no tokenizer files.
            tokenizer_subdir="tokenizer",
            # The acoustic stage carries the previous window's latent forward as
            # the next window's flow-matching prompt, so a request parked between
            # chunks still owns its model state and must keep counting against
            # max_num_seqs.
            retains_state_across_chunks=True,
            sync_process_input_func=f"{_PROC}.ar2acoustic",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

__all__ = [
    "MINIMAX_MUSIC3_AUDIO_END_TOKEN_ID",
    "MINIMAX_MUSIC3_PIPELINE",
]
