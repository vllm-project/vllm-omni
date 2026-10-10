# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Single-kernel MRv2 sampler for the Qwen3-TTS Talker.

The upstream sampler runs the Talker's settings (codec mask, ``min_tokens``,
repetition penalty, temperature, top-k) as a chain of about ten small kernel
launches and tensor allocations, which costs ~0.5 ms of host time per step
regardless of the batch size. The codec vocabulary is small (3072), so one
program per request applies them all to its row in registers and draws the
token with the upstream seeded Gumbel noise (per request seed and position).

Requests using anything else (logprobs, allowed tokens, logit bias, bad words,
min_p, top_p, a top_k above ``MAX_TOP_K``, structured outputs, thinking
budgets) take the upstream sampler for the whole step.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.worker_v2.omni_sampler import OmniSampler

logger = init_logger(__name__)

MAX_TOP_K = 64
MAX_VOCAB = 8192


@triton.jit
def _talker_sample_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    sampled_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    idx_mapping_ptr,
    logits_indices_ptr,
    positions_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    disallowed_ptr,
    temperature_ptr,
    top_k_ptr,
    seeds_ptr,
    min_lens_ptr,
    num_stop_ptr,
    stop_ids_ptr,
    stop_ids_stride,
    rep_ptr,
    freq_ptr,
    pres_ptr,
    prompt_mask_ptr,
    prompt_mask_stride,
    out_counts_ptr,
    out_counts_stride,
    BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_mapping_ptr + row).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    in_vocab = cols < vocab_size
    logits = tl.load(logits_ptr + row * logits_stride + cols, mask=in_vocab, other=float("-inf")).to(tl.float32)
    disallowed = tl.load(disallowed_ptr + cols, mask=in_vocab, other=1) != 0
    logits = tl.where(disallowed, float("-inf"), logits)

    pos = tl.load(positions_ptr + tl.load(logits_indices_ptr + row))

    # min_tokens: stop tokens are banned before the request reaches its minimum length.
    num_stop = tl.load(num_stop_ptr + req)
    if num_stop > 0 and pos + 1 < tl.load(min_lens_ptr + req):
        for i in range(num_stop):
            stop_id = tl.load(stop_ids_ptr + req * stop_ids_stride + i)
            logits = tl.where(cols == stop_id, float("-inf"), logits)

    # Penalties, as in the upstream _penalties_kernel.
    rep = tl.load(rep_ptr + req)
    freq = tl.load(freq_ptr + req)
    pres = tl.load(pres_ptr + req)
    if rep != 1.0 or freq != 0.0 or pres != 0.0:
        counts = tl.load(out_counts_ptr + req * out_counts_stride + cols, mask=in_vocab, other=0)
        seen = counts > 0
        if rep != 1.0:
            packed = tl.load(prompt_mask_ptr + req * prompt_mask_stride + cols // 32, mask=in_vocab, other=0)
            in_prompt = ((packed >> (cols % 32)) & 1) != 0
            scale = tl.where(in_prompt | seen, rep, 1.0)
            logits *= tl.where(logits > 0, 1.0 / scale, scale)
        logits -= freq * counts
        logits -= pres * seen.to(tl.float32)

    temp = tl.load(temperature_ptr + req).to(tl.float32)
    if temp != 0.0 and temp != 1.0:
        logits = logits / temp

    top_k = tl.load(top_k_ptr + req)
    if top_k < vocab_size:
        ordered = tl.topk(logits, TOPK_BLOCK)
        threshold = tl.sum(tl.where(tl.arange(0, TOPK_BLOCK) == top_k - 1, ordered, 0.0))
        logits = tl.where(logits < threshold, float("-inf"), logits)

    seed = tl.load(seeds_ptr + req)
    _, token = gumbel_noised_argmax(
        logits,
        cols,
        in_vocab,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=False,
        APPLY_TEMPERATURE=False,
    )
    tl.store(sampled_ptr + row, token.to(tl.int64))
    # One logit per request: chunked-prefill rows sample nothing, and nothing is rejected.
    chunked = tl.load(seq_lens_ptr + row) < tl.load(prefill_len_ptr + req)
    tl.store(num_sampled_ptr + row, tl.where(chunked, 0, 1).to(tl.int32))
    tl.store(num_rejected_ptr + row, tl.zeros((), tl.int32))


class Qwen3TTSTalkerSampler(OmniSampler):
    """Fused Talker sampling for the settings the Talker uses, upstream sampling otherwise."""

    def __init__(self, base_sampler: Sampler, talker: Any):
        super().__init__(base_sampler)
        self.vocab_size = int(base_sampler.sampling_states.vocab_size)
        self._fast = np.zeros(base_sampler.req_states.max_num_reqs, dtype=bool)
        # The Talker's codec mask buffer, read on the first call (once the model is on its device).
        self._talker = talker
        self._disallowed: torch.Tensor | None = None
        self._enabled = self._supported(base_sampler)
        self._warmed = False
        if self._enabled:
            logger.info("Qwen3-TTS Talker: single-kernel MRv2 sampler (top_k <= %d)", MAX_TOP_K)

    def _disallowed_on(self, device: torch.device) -> torch.Tensor:
        if self._disallowed is None:
            mask = self._talker._codec_disallowed_mask
            self._disallowed = mask.to(device=device, dtype=torch.int8).contiguous()
        return self._disallowed

    def _supported(self, base: Sampler) -> bool:
        from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
        from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
        from vllm.v1.worker.gpu.sample.penalties import PenaltiesState

        processors = getattr(base, "logits_processors", None)
        return (
            type(base) is Sampler
            and torch.cuda.is_available()
            and self.vocab_size <= MAX_VOCAB
            and not base.compute_nans
            # The kernel draws fp32 Gumbel noise.
            and not base.use_fp64_gumbel
            and base.trace_replay_state is None
            and not base.return_sampling_mask
            and processors is not None
            and [type(p) for p in processors] == [LogitBiasState, PenaltiesState, BadWordsState]
        )

    @property
    def fused_disallowed_mask(self) -> bool:
        """True when the codec mask is applied here, so compute_logits may skip it."""
        return self._enabled

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        base = self.base_sampler
        base.add_request(req_idx, sampling_params)
        if not self._enabled:
            return
        states = base.sampling_states
        bias = base.logits_processors[0]
        bad_words = base.logits_processors[2]
        thinking = base.thinking_budget_state
        top_k = int(states.top_k.np[req_idx])
        self._fast[req_idx] = bool(
            base.get_logprobs_dims(np.array([req_idx])) is None
            and bias.num_allowed_token_ids.np[req_idx] == 0
            and bias.num_logit_bias.np[req_idx] == 0
            and bias.restore_when_all_masked.np[req_idx] == 0
            and bad_words.num_bad_words.np[req_idx] == 0
            and not (getattr(thinking, "enabled", False) and thinking.use_thinking_budget[req_idx])
            and states.min_p.np[req_idx] == 0.0
            and states.top_p.np[req_idx] == 1.0
            and (top_k <= MAX_TOP_K or top_k >= self.vocab_size)
        )

    def __call__(self, logits: torch.Tensor, input_batch: Any) -> SamplerOutput:
        n = input_batch.num_reqs
        if self._enabled and logits.shape[0] == n and self._fast[input_batch.idx_mapping_np[:n]].all():
            return self._sample(logits, input_batch, n)
        if self._enabled and not self._warmed and n > 0 and logits.shape[0] == n:
            # Compile the kernel on the runner's dummy sampler run, not on the first request.
            self._sample(logits, input_batch, n)
            self._warmed = True
        if self.fused_disallowed_mask:
            logits = logits.masked_fill(self._talker._codec_disallowed_mask, float("-inf"))
        return self.base_sampler(logits, input_batch)

    def _sample(self, logits: torch.Tensor, input_batch: Any, n: int) -> SamplerOutput:
        base = self.base_sampler
        states = base.sampling_states
        bias = base.logits_processors[0]
        penalties = base.penalties_state
        sampled = torch.empty((n, 1), dtype=torch.int64, device=logits.device)
        counts = torch.empty((2, n), dtype=torch.int32, device=logits.device)
        disallowed = self._disallowed_on(logits.device)
        _talker_sample_kernel[(n,)](
            logits,
            logits.stride(0),
            logits.shape[1],
            sampled,
            counts[0],
            counts[1],
            input_batch.idx_mapping,
            input_batch.logits_indices,
            input_batch.positions,
            input_batch.seq_lens,
            base.req_states.prefill_len.gpu,
            disallowed,
            states.temperature.gpu,
            states.top_k.gpu,
            states.seeds.gpu,
            bias.min_lens.gpu,
            bias.num_stop_token_ids.gpu,
            bias.stop_token_ids.gpu,
            bias.stop_token_ids.gpu.stride(0),
            penalties.repetition_penalty.gpu,
            penalties.frequency_penalty.gpu,
            penalties.presence_penalty.gpu,
            penalties.prompt_bin_mask,
            penalties.prompt_bin_mask.stride(0),
            penalties.output_bin_counts,
            penalties.output_bin_counts.stride(0),
            BLOCK=triton.next_power_of_2(logits.shape[1]),
            TOPK_BLOCK=MAX_TOP_K,
            num_warps=8,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=counts[0],
            num_rejected=counts[1],
        )
