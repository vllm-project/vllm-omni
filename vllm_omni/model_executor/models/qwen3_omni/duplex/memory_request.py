# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Private Thinker request adapters; no live policy or budget allocation.

The caller supplies an allowance and the deployed stage's context length.
These helpers neither reserve that allowance nor enable Qwen memory work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.memory_types import DuplexContextAuxiliaryInput, DuplexMemoryProposalBatch
from vllm_omni.engine.duplex.plugin import DuplexAuxiliaryRequest, duplex_history_digest
from vllm_omni.model_executor.models.qwen3_omni.duplex.memory import build_memory_messages, parse_memory_output
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from transformers import Qwen3OmniMoeProcessor


def build_memory_request(
    context: DuplexContextAuxiliaryInput,
    *,
    processor: Qwen3OmniMoeProcessor,
    input_token_limit: int,
    output_token_limit: int,
    max_model_len: int,
) -> DuplexAuxiliaryRequest | None:
    """Fit a complete capture or defer it; never truncate acknowledged scope."""
    if (
        type(input_token_limit) is not int
        or not 0 <= input_token_limit <= 2048
        or type(output_token_limit) is not int
        or not 0 <= output_token_limit <= 512
        or type(max_model_len) is not int
        or max_model_len <= 0
    ):
        raise ValueError("invalid_memory_request_limits")
    if input_token_limit == 0 or output_token_limit == 0:
        return None
    messages = build_memory_messages(context)
    if messages is None:
        return None
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids = processor.tokenizer.encode(rendered, add_special_tokens=False)
    if len(token_ids) > input_token_limit or len(token_ids) + output_token_limit > max_model_len:
        return None
    return DuplexAuxiliaryRequest(
        stage_id=0,
        prompt={"prompt_token_ids": token_ids},
        sampling_params=SamplingParams(
            n=1,
            temperature=0,
            max_tokens=output_token_limit,
            detokenize=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
        ),
        # The context path validates the retained, versioned capture instead
        # of the legacy whole-history digest. The runtime binds context_work.
        history_length=0,
        history_digest=duplex_history_digest(()),
        model_source=context,
    )


def decode_memory_result(*, request: DuplexAuxiliaryRequest, output: object) -> DuplexMemoryProposalBatch | None:
    """Decode one final text completion; session admission remains authoritative."""
    context = request.model_source
    if (
        request.stage_id != 0
        or not isinstance(context, DuplexContextAuxiliaryInput)
        or request.context_work != context.work
        or not isinstance(output, RequestOutput)
        or output.finished is not True
        or len(output.outputs) != 1
    ):
        return None
    if isinstance(output, OmniRequestOutput) and (
        output.stage_id != 0 or output.error is not None or output.final_output_type != "text"
    ):
        return None
    completion = output.outputs[0]
    if not isinstance(completion, CompletionOutput) or completion.index != 0 or completion.finish_reason != "stop":
        return None
    return parse_memory_output(completion.text, context=context)
