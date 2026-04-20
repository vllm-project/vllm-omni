# SPDX-License-Identifier: Apache-2.0
"""Stage input processor: Fun-Audio-Chat-8B Stage 0 → Stage 1 (token2wav).

Collects accumulated crq_tokens from Stage 0's engine outputs and packages
them as OmniTokensPrompt for the FunAudioChatToken2Wav stage.
"""

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Collect crq_tokens from Stage 0 engine outputs → OmniTokensPrompt for Stage 1.

    Each output's multimodal_output["crq_tokens"] is a 1D tensor of CRQ token IDs
    accumulated across all decode steps of Stage 0.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    source_stage_id = engine_input_source[0]
    stage0_outputs = stage_list[source_stage_id].engine_outputs

    result: list[OmniTokensPrompt] = []
    for request_output in stage0_outputs:
        out = request_output.outputs[0]
        mm = out.multimodal_output if out.multimodal_output else {}
        crq_tokens = mm.get("crq_tokens")

        if isinstance(crq_tokens, torch.Tensor) and crq_tokens.numel() > 0:
            token_ids = crq_tokens.flatten().to(torch.long).tolist()
        else:
            logger.warning("funaudiochat talker2code2wav: no crq_tokens in stage 0 output")
            token_ids = []

        result.append(
            OmniTokensPrompt(
                prompt_token_ids=token_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )
    return result
