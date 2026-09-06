# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling-state guards shared by the GPU and NPU AR model runners."""

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors, MinTokensLogitsProcessor

logger = init_logger(__name__)

__all__ = [
    "apply_fixed_seed_to_sampling_params",
    "get_tts_local_seed",
    "clamp_prompt_ids_to_penalty_padding",
    "sanitize_min_tokens_stop_ids",
    "sanitize_sampling_params_min_tokens_stop_ids",
]


def get_tts_local_seed(sampling_params: Any) -> int | None:
    """Return the explicit per-request Talker MTP seed, if configured."""
    extra_args = getattr(sampling_params, "extra_args", None) if sampling_params is not None else None
    seed = extra_args.get("tts_local_seed") if isinstance(extra_args, dict) else None
    return int(seed) if seed is not None else None


def apply_fixed_seed_to_sampling_params(
    sampling_params: Any,
    seed: int,
    *,
    seed_talker_mtp: bool,
) -> None:
    """Apply one request seed to a stage sampler and, for Talkers, its MTP RNG."""
    seed = int(seed)
    sampling_params.seed = seed
    if not seed_talker_mtp:
        return
    extra_args = dict(getattr(sampling_params, "extra_args", None) or {})
    extra_args["tts_local_seed"] = seed
    sampling_params.extra_args = extra_args


def sanitize_sampling_params_min_tokens_stop_ids(
    sampling_params: SamplingParams,
    logits_vocab: int,
) -> None:
    """Remove unreachable ids before MRv2 builds persistent min-token state.

    MRv2 copies ``SamplingParams.all_stop_token_ids`` into GPU-resident
    ``LogitBiasState`` when a request is registered. Narrow codec heads cannot
    safely carry the text tokenizer EOS in that state. The engine-facing EOS
    and explicit ``stop_token_ids`` fields remain unchanged; only the set used
    by the sampler's min-token mask is normalized.
    """
    if sampling_params.min_tokens <= 0:
        return
    stop_token_ids = sampling_params.all_stop_token_ids
    unreachable = {token_id for token_id in stop_token_ids if token_id < 0 or token_id >= logits_vocab}
    if not unreachable:
        return
    stop_token_ids.difference_update(unreachable)
    logger.warning_once(
        "min_tokens: dropped stop token ids %s that exceed the logits vocabulary (%d); "
        "the model head cannot emit them.",
        str(sorted(unreachable)),
        logits_vocab,
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
