# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transfer Spark's generated codec IDs to its full-sequence BiCodec stage."""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.spark_tts.tokenizer import (
    END_GLOBAL,
    GLOBAL_OFFSET,
    GLOBAL_SIZE,
    SEMANTIC_OFFSET,
    SEMANTIC_SIZE,
    START_GLOBAL,
)

logger = init_logger(__name__)


def parse_codes(output_ids: list[int], prompt_ids: list[int]) -> tuple[list[int], list[int]]:
    """Return semantic and global code indices, using prompt globals for clone mode."""
    semantic = [t - SEMANTIC_OFFSET for t in output_ids if SEMANTIC_OFFSET <= t < SEMANTIC_OFFSET + SEMANTIC_SIZE]
    generated_global = [t - GLOBAL_OFFSET for t in output_ids if GLOBAL_OFFSET <= t < GLOBAL_OFFSET + GLOBAL_SIZE]
    prompt_global: list[int] = []
    if START_GLOBAL in prompt_ids:
        start = prompt_ids.index(START_GLOBAL) + 1
        end = prompt_ids.index(END_GLOBAL, start) if END_GLOBAL in prompt_ids[start:] else len(prompt_ids)
        prompt_global = [
            t - GLOBAL_OFFSET for t in prompt_ids[start:end] if GLOBAL_OFFSET <= t < GLOBAL_OFFSET + GLOBAL_SIZE
        ]
    return semantic, prompt_global or generated_global


def llm_to_bicodec_full_payload(transfer_manager: Any, pooling_output: Any, request: Any) -> OmniPayloadStruct:
    """Send a terminal payload even for malformed output so stage 1 cannot wait forever."""
    del transfer_manager, pooling_output
    output_ids = list(getattr(request, "output_token_ids", None) or [])
    prompt_ids = list(getattr(request, "prompt_token_ids", None) or [])
    semantic, global_codes = parse_codes(output_ids, prompt_ids)
    request_id = (
        getattr(request, "external_req_id", None)
        or getattr(request, "request_id", None)
        or getattr(request, "req_id", None)
    )
    if request_id is None:
        raise ValueError("Spark-TTS transfer is missing a request id")
    logger.info(
        "Spark payload req=%s prompt=%d output=%d semantic=%d global=%d",
        request_id,
        len(prompt_ids),
        len(output_ids),
        len(semantic),
        len(global_codes),
    )
    return OmniPayloadStruct(
        codes=CodesStruct(
            audio=torch.tensor(semantic, dtype=torch.long).reshape(-1, 1),
            ref=torch.tensor(global_codes, dtype=torch.long).reshape(-1, 1),
        ),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool), req_id=[str(request_id)]),
    )


def bicodec_token_only(
    source_outputs: list[Any], prompt: Any = None, _requires_multimodal_data: bool = False
) -> list[OmniTokensPrompt]:
    """The connector carries actual codes; this input only schedules stage 1."""
    del prompt, _requires_multimodal_data
    inputs = []
    for source in source_outputs:
        if not getattr(source, "finished", False):
            continue
        # The full codec payload travels through the connector. One token makes
        # this a single generation input unit regardless of the source length.
        inputs.append(OmniTokensPrompt(prompt_token_ids=[0]))
    return inputs
