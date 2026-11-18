# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One sampler registration contract for token and post-sampling audio output."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.worker_v2.output_snapshot import RequestOutputSnapshot

StandardSample = Callable[
    [torch.Tensor, InputBatch, GrammarOutput | None], tuple[SamplerOutput, torch.Tensor, torch.Tensor]
]


def patch_resampled_token_insertion() -> None:
    """Backport vLLM #66048's launch configuration to the pinned 0.33 runtime."""
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _insert_resampled_kernel

    # Every warp loads num_sampled, but warp 0 increments it before the others
    # necessarily read it. A late warp can then use an out-of-range logits index
    # or skip the bonus-token reduction. Keep the upstream kernel and algorithm;
    # remove this adapter when the minimum vLLM includes #66048.
    run = _insert_resampled_kernel.run
    if isinstance(run, partial) and run.keywords.get("num_warps") == 1:
        return
    _insert_resampled_kernel.run = partial(run, num_warps=1)


@dataclass
class OmniSamplingOutput:
    sampler_output: SamplerOutput
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    # None preserves forward's payload. An empty dict explicitly replaces it.
    # Tensors must remain valid until the runner finishes its output copy.
    multimodal_outputs: dict[str, Any] | None = None
    include_hidden_states: bool = True
    owns_multimodal_outputs: bool = False
    # Invoked on CPU-owned snapshots after D2H, with effective per-request counts.
    finalize_multimodal: Callable[[dict[str, Any], list[int]], dict[str, Any] | RequestOutputSnapshot] | None = None


class OmniSampler:
    """Register through ModelState.custom_sampler, just like an upstream sampler.

    Delegate request lifecycle/state to the original sampler without copying
    its attributes. Models can override __call__ for logits-only adjustments,
    or sample_step when sampling also produces audio. Both use one registered
    sampler and one result contract; there is no separate ModelState step hook.
    """

    def __init__(self, base_sampler: Sampler):
        self.base_sampler = base_sampler

    @property
    def omni_static_staged_writes(self) -> bool:
        # Preserve the runner's no-new-requests fast path for the stock sampler.
        return type(self.base_sampler) is Sampler or bool(
            getattr(self.base_sampler, "omni_static_staged_writes", False)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_sampler, name)

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        return self.base_sampler(logits, input_batch)

    def sample_step(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
        grammar_output: GrammarOutput | None,
        standard_sample: StandardSample,
    ) -> OmniSamplingOutput:
        # standard_sample calls this adapter's __call__ through the runner.
        return OmniSamplingOutput(*standard_sample(hidden_states, input_batch, grammar_output))


def sample_with_output(
    sampler: Sampler | OmniSampler | None,
    standard_sample: StandardSample,
    hidden_states: torch.Tensor,
    input_batch: InputBatch,
    req_states: RequestState,
    grammar_output: GrammarOutput | None,
) -> OmniSamplingOutput:
    if isinstance(sampler, OmniSampler):
        return sampler.sample_step(hidden_states, input_batch, req_states, grammar_output, standard_sample)
    return OmniSamplingOutput(*standard_sample(hidden_states, input_batch, grammar_output))
