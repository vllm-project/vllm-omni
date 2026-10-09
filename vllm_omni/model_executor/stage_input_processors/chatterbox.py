# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox stage handoff when ``async_chunk`` is off.

Streaming uses CosyVoice3's ``talker2code2wav_async_chunk`` unchanged: S3
tokens are the same 6561-way codec, and it reads the reference from the
request's ``additional_information`` under the names
``conditioning.build_prompt`` writes. This module is the other mode, where
stage 1 receives the finished utterance once.
"""

import torch

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

SPEECH_TOKEN_LIMIT = ChatterboxConfig().speech_token_limit


def t3_to_s3gen(
    source_outputs: list,
    prompt: dict | list[dict],
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build stage 1's input from stage 0's finished output.

    The payload is shaped as a stream of one final chunk, so stage 1 has a
    single input contract in both modes.

    Args:
        source_outputs: Stage 0's request outputs.
        prompt: The request's own prompt, or one per source output, each
            from ``conditioning.build_prompt``.
        _requires_multimodal_data: Unused; part of the processor signature.

    Returns:
        One prompt per finished output: the valid speech tokens, with the
        request's reference and the stream metadata.

    Raises:
        RuntimeError: If stage 0 produced no speech tokens for a request.
    """
    prompts = prompt if isinstance(prompt, list) else [prompt] * len(source_outputs)
    stage_inputs: list[OmniTokensPrompt] = []
    for source_output, request_prompt in zip(source_outputs, prompts, strict=True):
        if not source_output.finished:
            continue
        # Drops the stop token, and the placeholder prompt ids should an
        # engine version report the prompt in the output history.
        codes = [token for token in source_output.outputs[0].cumulative_token_ids if token < SPEECH_TOKEN_LIMIT]
        if not codes:
            raise RuntimeError(f"Chatterbox T3 produced no speech tokens for request {source_output.request_id}")
        reference = request_prompt["additional_information"]["embed"]
        stage_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codes,
                additional_information={
                    "embed": {
                        "speech_token": reference["speech_token"],
                        "speech_feat": reference["speech_feat"],
                        "embedding": reference["embedding"],
                    },
                    "meta": {"stream_finished": torch.tensor(True), "left_context_size": 0},
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return stage_inputs
