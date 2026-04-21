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

    # Deduplicate: Stage-0 emits one engine output per decode step, and each
    # output's multimodal_output["crq_tokens"] carries the FULL cumulative
    # speech_ids so far (our postprocess returns a snapshot, not a delta).
    # Concatenating every output's tensor multiplies the token stream by the
    # number of steps (observed: 512 steps × 2560 final tokens = 222955-ish
    # tokens vs 2560 expected). Deduplicate by request id, keep latest.
    last_per_req: dict[str, tuple[Any, Any]] = {}
    for request_output in stage0_outputs:
        out = request_output.outputs[0]
        req_id = str(getattr(request_output, "request_id", None) or id(request_output))
        mm = out.multimodal_output if out.multimodal_output else {}
        crq_tokens = mm.get("crq_tokens")
        last_per_req[req_id] = (request_output, crq_tokens)

    STAGE1_MAX = 65535  # Stage-1 max_model_len
    result: list[OmniTokensPrompt] = []
    for req_id, (_request_output, crq_tokens) in last_per_req.items():
        if isinstance(crq_tokens, torch.Tensor) and crq_tokens.numel() > 0:
            token_ids = crq_tokens.flatten().to(torch.long).tolist()
            logger.info(
                "funaudiochat talker2code2wav: req=%s crq_tokens len=%d",
                req_id, len(token_ids),
            )
            if len(token_ids) > STAGE1_MAX:
                logger.warning(
                    "talker2code2wav: crq_tokens length %d > Stage-1 max %d — truncating",
                    len(token_ids), STAGE1_MAX,
                )
                token_ids = token_ids[:STAGE1_MAX]
        else:
            logger.warning(
                "funaudiochat talker2code2wav: no crq_tokens in stage 0 output "
                "for req=%s — emitting sentinel token", req_id,
            )
            token_ids = [0]

        result.append(
            OmniTokensPrompt(
                prompt_token_ids=token_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )
    return result
