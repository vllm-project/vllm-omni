# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling-state helpers shared by the GPU and NPU AR model runners."""

from collections.abc import Mapping, Sequence

import torch
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import LogitsProcessors, MinTokensLogitsProcessor

logger = init_logger(__name__)

__all__ = ["accepted_hidden_rows", "sanitize_min_tokens_stop_ids"]


def accepted_hidden_rows(
    *,
    req_id: str,
    req_index: int,
    num_scheduled_tokens: int,
    scheduled_spec_decode_tokens: Mapping[str, Sequence[int]],
    valid_sampled_token_ids: Sequence[Sequence[int]],
) -> int:
    """Rows of this step's hidden states that ``req_id`` actually emitted.

    A speculative step forwards one real position plus the drafts, and
    rejection sampling keeps a *prefix* of them. Slicing an Omni payload by
    ``num_scheduled_tokens`` therefore also ships the rows past that prefix:
    hidden states for draft tokens the model never emitted. A downstream AR
    stage that conditions on them -- a codec talker, say -- renders text that
    does not exist.

    ``num_scheduled_tokens`` is returned unchanged whenever the step cannot
    carry a rejection remainder for this request: no drafts were scheduled for
    it (a prefill step schedules the prompt, and the downstream stage needs all
    of it), or its accepted-token bookkeeping is not available yet (async
    scheduling fills ``valid_sampled_token_ids`` in after this point).
    """
    if not scheduled_spec_decode_tokens.get(req_id):
        return num_scheduled_tokens
    if req_index >= len(valid_sampled_token_ids):
        return num_scheduled_tokens
    num_accepted = len(valid_sampled_token_ids[req_index])
    if num_accepted <= 0:
        return num_scheduled_tokens
    return min(num_scheduled_tokens, num_accepted)


def sanitize_min_tokens_stop_ids(logitsprocs: LogitsProcessors, logits_vocab: int) -> None:
    """Drop stop ids the model head cannot emit from min-tokens masking state.

    vLLM's input processor unconditionally folds the stage tokenizer's EOS id
    into ``SamplingParams.all_stop_token_ids``. AR stages whose lm_head is
    narrower than the tokenizer vocabulary (codec talkers such as Qwen3-TTS:
    3072 logits vs text EOS 151645) then crash on any ``min_tokens >= 1``:
    ``MinTokensLogitsProcessor.apply`` writes ``-inf`` at an out-of-range
    index and ``index_put_`` triggers a CUDA device-side assert (#4962).

    Out-of-range ids are unreachable for the head, so dropping them never
    changes sampling or stopping behavior. The per-request stop-id set is
    mutated in place (it is shared with the request's ``SamplingParams``),
    so each request is sanitized at most once; the processor's device-side
    mask slice is rebuilt only when an out-of-range id was found.
    """
    for proc in logitsprocs.non_argmax_invariant:
        if not isinstance(proc, MinTokensLogitsProcessor):
            continue
        min_toks = getattr(proc, "min_toks", None)
        if not min_toks:
            continue
        needs_rebuild = False
        for _, _, stop_tok_ids in min_toks.values():
            oob = [tok for tok in stop_tok_ids if tok >= logits_vocab]
            if not oob:
                continue
            stop_tok_ids.difference_update(oob)
            needs_rebuild = True
            logger.warning_once(
                "min_tokens: dropped stop token ids %s that exceed the logits vocabulary (%d); "
                "the model head cannot emit them.",
                str(sorted(oob)),
                logits_vocab,
            )
        if needs_rebuild:
            reqs: list[int] = []
            tok_ids: list[int] = []
            for index, (_, _, stop_tok_ids) in min_toks.items():
                reqs.extend([index] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)
            proc.logits_slice = (
                proc._device_tensor(reqs, torch.int32),
                proc._device_tensor(tok_ids, torch.int32),
            )
