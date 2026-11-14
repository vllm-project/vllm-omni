# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# NOTE: Vendored by vLLM-Omni from upstream vllm/v1/sample/ops/penalties.py
# (4ba787ec9959e566d3b587a796dff07673324da5); removed or restructured at the
# pin by aac1b2a4f3 (#63373). Import lines rewritten; other deviations: none.

import torch
from vllm.model_executor.layers.utils import apply_penalties
from vllm.utils.torch_utils import PIN_MEMORY

from vllm_omni.worker.v1_compat._util import make_tensor_with_pad


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """Applies presence, frequency and repetition penalties to the logits."""
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)

    # In the async scheduling case, rows that won't have penalties applied may contain
    # -1 placeholder token ids. We must replace these with valid token ids so that the
    # scatter done in apply_penalties is valid.
    # NOTE(nick): The penalties implementation is currently quite inefficient and
    # will be reworked anyhow.
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    return apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


def _convert_to_tensors(output_token_ids: list[list[int]], vocab_size: int, device: torch.device) -> torch.Tensor:
    """Convert the different list data structures to tensors."""
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=PIN_MEMORY,
    )
    return output_tokens_tensor.to(device, non_blocking=True)
