# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HunyuanImage3 AR constraints for the device-resident MRv2 sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.worker_v2.omni_sampler import OmniSampler

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import HunyuanImage3ForConditionalGeneration


class HunyuanImage3ARSampler(OmniSampler):
    """Apply the model's transition, ratio and comprehension rules before sampling.

    Read only the bounded suffix needed by the forced sequences. MRv2 owns
    the full token history, including restored history after preemption.
    """

    def __init__(self, base_sampler: Sampler, model: HunyuanImage3ForConditionalGeneration):
        super().__init__(base_sampler)
        self.model = model
        device = self.req_states.prompt_len.gpu.device
        self._comprehension = torch.full_like(self.req_states.prompt_len.gpu, model._is_comprehension, dtype=torch.bool)
        self._blocked_ids = torch.tensor(sorted(model._blocked_token_ids), device=device, dtype=torch.long)
        self._ratio_ids = torch.tensor(sorted(model._all_ratio_ids), device=device, dtype=torch.long)
        self._transition_prefixes = [
            (torch.tensor([trigger, *sequence[:offset]], device=device), next_token)
            for trigger, sequence in model._stage_transitions.items()
            for offset, next_token in enumerate(sequence)
        ]
        history_size = max((prefix.numel() for prefix, _ in self._transition_prefixes), default=1)
        self._history_offsets = torch.arange(history_size, 0, -1, device=device)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        self.base_sampler.add_request(req_idx, sampling_params)
        self._comprehension[req_idx] = self.model._is_comprehension or self.model._row_is_comprehension(
            [sampling_params.extra_args], 0
        )

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        comprehension = self._comprehension[input_batch.idx_mapping]
        min_score = torch.finfo(logits.dtype).min
        logits[:, self._blocked_ids] = torch.where(comprehension[:, None], min_score, logits[:, self._blocked_ids])
        if self.model._is_comprehension:
            return self.base_sampler(logits, input_batch)

        # seq_lens includes the current forward's inputs and excludes its
        # next sampled token. Keep restored output tokens separate from the
        # original prompt, just as V1's output_token_ids does.
        slots = input_batch.idx_mapping[:, None]
        positions = input_batch.seq_lens[: input_batch.num_reqs, None] - self._history_offsets
        history = self.req_states.all_token_ids.gpu[slots, positions.clamp_min(0)]
        history = history.masked_fill(positions < self.req_states.prompt_len.gpu[slots], -1)
        last_token = history[:, -1]

        ratio_scores = logits[:, self._ratio_ids]
        best_ratio = self._ratio_ids[ratio_scores.argmax(dim=-1)]
        forced_token = torch.full_like(last_token, -1)
        forced_token = torch.where(last_token == self.model._size_token_id, best_ratio, forced_token)
        forced_token = torch.where(
            (last_token[:, None] == self._ratio_ids).any(dim=-1), self.model._eos_token_id, forced_token
        )
        for prefix, next_token in self._transition_prefixes:
            matches = (history[:, -prefix.numel() :] == prefix).all(dim=-1)
            forced_token = torch.where(matches, next_token, forced_token)

        force = (forced_token >= 0) & ~comprehension
        logits.masked_fill_(force[:, None], min_score)
        selected = forced_token.clamp_min(0)[:, None]
        logits.scatter_(1, selected, torch.where(force[:, None], 0.0, logits.gather(1, selected)))
        return self.base_sampler(logits, input_batch)
