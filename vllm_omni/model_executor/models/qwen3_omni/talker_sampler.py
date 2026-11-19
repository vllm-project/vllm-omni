# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Talker codebook-0 sampling that follows the V1 runner's seeded stream.

MRv2's default sampler draws Gumbel noise with murmur3(seed, position, token).
The V1 runner instead keeps one CUDA generator per seeded request and draws an
exponential row from it on every step. Those streams are both unbiased, but on
Qwen3-Omni they are not interchangeable: with the deploy seed the murmur3
stream raises Seed-TTS word error by about 0.4 percentage points. Seeded
Talker requests therefore sample here the way V1 does. Greedy requests
(temperature 0) keep the upstream argmax path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.worker_v2.omni_sampler import OmniSampler

logger = init_logger(__name__)


class Qwen3OmniTalkerSampler(OmniSampler):
    """Route seeded Talker rows through V1's per-request exponential sampler."""

    def __init__(self, base_sampler: Sampler):
        super().__init__(base_sampler)
        self._generators: dict[int, torch.Generator] = {}
        self._sample_random = base_sampler._sample_random
        base_sampler._sample_random = self._sample_like_v1
        self._logged_v1 = False
        self._logged_mixed = False
        self._logged_fallback = False
        logger.info("Qwen3-Omni Talker: codebook 0 uses the V1 per-request generator")

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        self.base_sampler.add_request(req_idx, sampling_params)
        seed = sampling_params.seed
        temperature = float(sampling_params.temperature or 0.0)
        if seed is None or temperature == 0.0:
            self._generators.pop(req_idx, None)
            return
        generator = torch.Generator(device=self.base_sampler.penalties_state.device)
        generator.manual_seed(int(seed))
        self._generators[req_idx] = generator

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        # Upstream samples a row for every request, then drops the token of an
        # unfinished chunked prefill. Like V1, that draw must not advance the
        # request's stream, or the token budget would shift every later draw.
        held = [(generator, generator.get_state()) for generator in self._unfinished_prefill_generators(input_batch)]
        output = self.base_sampler(logits, input_batch)
        for generator, state in held:
            generator.set_state(state)
        return output

    def _unfinished_prefill_generators(self, input_batch: InputBatch) -> list[torch.Generator]:
        if not self._generators or not input_batch.has_prefill:
            return []
        num_reqs = input_batch.num_reqs
        seq_lens = input_batch.num_computed_prefill_tokens_np[:num_reqs] + input_batch.num_scheduled_tokens[:num_reqs]
        unfinished = seq_lens < input_batch.prefill_len_np[:num_reqs]
        slots = {int(slot) for slot in input_batch.idx_mapping_np[:num_reqs][unfinished]}
        return [self._generators[slot] for slot in slots if slot in self._generators]

    def _sample_like_v1(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        use_fused_sampler: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original: Callable[..., tuple[torch.Tensor, torch.Tensor]] = self._sample_random
        n_rows = int(processed_logits.shape[0])
        if n_rows == len(idx_mapping_np):
            slots = idx_mapping_np
        else:
            # Padded logit rows are not 1:1 with the request list. The expanded
            # mapping is what the upstream sampler uses for those rows.
            slots = expanded_idx_mapping.detach().to(device="cpu").numpy()
        generators: dict[int, torch.Generator] = {}
        for row, slot in enumerate(slots[:n_rows]):
            generator = self._generators.get(int(slot))
            if generator is not None:
                generators[row] = generator
        mappable = expanded_idx_mapping.shape[0] >= n_rows and pos.shape[0] >= n_rows
        if not generators or (len(generators) != n_rows and not mappable):
            if generators and not self._logged_fallback:
                logger.warning(
                    "Qwen3-Omni Talker kept the MRv2 sampler: seeded %d/%d rows, %d mapped rows",
                    len(generators),
                    n_rows,
                    int(expanded_idx_mapping.shape[0]),
                )
                self._logged_fallback = True
            return original(
                processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_fused_sampler
            )
        masked = apply_top_k_top_p(processed_logits, top_k, top_p)
        if len(generators) == n_rows:
            if not self._logged_v1:
                logger.info("Qwen3-Omni Talker sampled %d rows with the V1 generator", n_rows)
                self._logged_v1 = True
            probs = masked.softmax(dim=-1, dtype=torch.float32)
            return random_sample(probs, generators), masked
        # Seeded rows keep their own V1 stream whatever they are batched with.
        # Greedy and unseeded rows stay on the upstream sampler, which also
        # owns the temperature-0 argmax.
        if not self._logged_mixed:
            logger.info("Qwen3-Omni Talker split a step: %d/%d rows on the V1 generator", len(generators), n_rows)
            self._logged_mixed = True
        seeded_rows = list(generators)
        seeded_set = set(seeded_rows)
        other_rows = [row for row in range(n_rows) if row not in seeded_set]
        device = processed_logits.device
        seeded_idx = torch.tensor(seeded_rows, dtype=torch.long, device=device)
        other_idx = torch.tensor(other_rows, dtype=torch.long, device=device)
        sampled = torch.empty(n_rows, dtype=torch.int64, device=device)
        probs = masked.index_select(0, seeded_idx).softmax(dim=-1, dtype=torch.float32)
        sampled[seeded_idx] = random_sample(probs, {i: generators[row] for i, row in enumerate(seeded_rows)})
        # Rows are already top-k/top-p masked, and a fused sampler is never
        # eligible once any row has an explicit seed.
        other_sampled, _ = original(
            masked.index_select(0, other_idx),
            expanded_idx_mapping[:n_rows].index_select(0, other_idx),
            idx_mapping_np,
            pos[:n_rows].index_select(0, other_idx),
            None,
            None,
            False,
        )
        sampled[other_idx] = other_sampled.to(torch.int64)
        return sampled, masked


def mrv2_talker_sampler(sampler: Any) -> tuple[Qwen3OmniTalkerSampler, None]:
    return Qwen3OmniTalkerSampler(sampler), None
