# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sampling helpers shared by the GPU and NPU AR model runners."""

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import LogitsProcessors, MinTokensLogitsProcessor

logger = init_logger(__name__)

__all__ = [
    "build_model_sampler_extra_args",
    "call_model_sampler",
    "clamp_prompt_ids_to_penalty_padding",
    "sanitize_min_tokens_stop_ids",
]


def build_model_sampler_extra_args(input_batch: Any, requests: Any) -> list[dict | None]:
    """Return per-request ``SamplingParams.extra_args`` in batch-row order."""
    request_states = requests or {}
    per_req_extra_args: list[dict | None] = []
    for req_id in getattr(input_batch, "req_ids", []):
        state = request_states.get(req_id)
        params = getattr(state, "sampling_params", None)
        per_req_extra_args.append(getattr(params, "extra_args", None))
    return per_req_extra_args


def call_model_sampler(
    model: Any,
    model_sample: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    *,
    input_batch: Any,
    requests: Any,
) -> Any:
    """Call an opted-in model sampler with per-request extra arguments.

    The opt-in keeps the existing two-argument sampler contract unchanged for
    every other model while allowing custom samplers to make row-local choices.
    """
    if not getattr(model, "model_sampler_wants_extra_args", False):
        return model_sample(logits, sampling_metadata)
    return model_sample(
        logits,
        sampling_metadata,
        per_req_extra_args=build_model_sampler_extra_args(input_batch, requests),
    )


def clamp_prompt_ids_to_penalty_padding(prompt_token_ids: torch.Tensor, logits_vocab: int) -> torch.Tensor:
    """Clamp batch-level pad ids down to ``logits_vocab`` — upstream's
    designed penalty padding value.

    ``max=logits_vocab`` (NOT ``logits_vocab - 1``) is deliberate: upstream
    penalty computation allocates ``vocab_size + 1`` bins and drops the last
    column, so ``vocab_size`` is the padding value that never affects
    penalties (vllm/model_executor/layers/utils.py::
    get_token_bin_counts_and_mask). Clamping one lower would count padding
    as real occurrences of the last vocab token.
    """
    return prompt_token_ids.clamp(max=logits_vocab)


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
        for _, _, stop_tok_ids, _ in min_toks.values():
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
            restore_reqs: list[int] = []
            restore_tok_ids: list[int] = []
            for index, (_, _, stop_tok_ids, uses_structured_output) in min_toks.items():
                reqs.extend([index] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)
                if uses_structured_output:
                    restore_reqs.extend([index] * len(stop_tok_ids))
                    restore_tok_ids.extend(stop_tok_ids)
            proc.logits_slice = (
                proc._device_tensor(reqs, torch.int32),
                proc._device_tensor(tok_ids, torch.int32),
            )
            proc.restore_logits_slice = (
                proc._device_tensor(restore_reqs, torch.int32),
                proc._device_tensor(restore_tok_ids, torch.int32),
            )
