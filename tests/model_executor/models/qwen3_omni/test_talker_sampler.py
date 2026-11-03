# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.model_executor.models.qwen3_omni.talker_sampler import Qwen3OmniTalkerSampler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_VOCAB = 64


class _BaseSampler:
    def __init__(self) -> None:
        self.penalties_state = SimpleNamespace(device=torch.device("cpu"))
        self.upstream_rows: list[int] = []

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        pass

    def _sample_random(self, logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_fused_sampler):
        self.upstream_rows.append(int(logits.shape[0]))
        return logits.argmax(dim=-1), logits

    def __call__(self, logits, input_batch):
        # Like upstream: every row is sampled, unfinished prefills are dropped later.
        slots = input_batch.idx_mapping_np
        sampled, _ = self._sample_random(
            logits,
            torch.from_numpy(slots).to(torch.int64),
            slots,
            torch.zeros(len(slots), dtype=torch.int64),
            None,
            None,
            False,
        )
        return sampled


def _sampler(requests: dict[int, SamplingParams]) -> tuple[Qwen3OmniTalkerSampler, _BaseSampler]:
    base = _BaseSampler()
    sampler = Qwen3OmniTalkerSampler(base)
    for req_idx, params in requests.items():
        sampler.add_request(req_idx, params)
    return sampler, base


def _step(sampler: Qwen3OmniTalkerSampler, slots: list[int], logits: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(slots, dtype=torch.int64)
    sampled, _ = sampler.base_sampler._sample_random(
        logits.clone(), mapping, np.array(slots), torch.zeros(len(slots), dtype=torch.int64), None, None, False
    )
    return sampled


def test_seeded_row_stream_does_not_depend_on_batch_mates():
    seeded = SamplingParams(temperature=0.9, seed=42)
    greedy = SamplingParams(temperature=0.0)
    alone, alone_base = _sampler({0: seeded})
    mixed, mixed_base = _sampler({0: seeded, 1: greedy})
    torch.manual_seed(0)
    for _ in range(8):
        seeded_logits = torch.randn(1, _VOCAB)
        greedy_logits = torch.randn(1, _VOCAB)
        expected = _step(alone, [0], seeded_logits)
        sampled = _step(mixed, [1, 0], torch.cat([greedy_logits, seeded_logits]))
        assert sampled[1] == expected[0]
        assert sampled[0] == greedy_logits.argmax(dim=-1)[0]
    assert alone_base.upstream_rows == []
    # Only the greedy row reaches the upstream sampler.
    assert mixed_base.upstream_rows == [1] * 8


def _batch(slots: list[int], computed: list[int], scheduled: list[int], prefill: list[int]) -> SimpleNamespace:
    computed_np = np.array(computed, dtype=np.int32)
    prefill_np = np.array(prefill, dtype=np.int32)
    return SimpleNamespace(
        num_reqs=len(slots),
        idx_mapping_np=np.array(slots, dtype=np.intp),
        num_scheduled_tokens=np.array(scheduled, dtype=np.int32),
        num_computed_prefill_tokens_np=computed_np,
        prefill_len_np=prefill_np,
        has_prefill=bool((computed_np < prefill_np).any()),
    )


def _accepted_codes(other: dict[int, SamplingParams], discarded_steps: int) -> list[int]:
    """Slot 0 is seeded with a 12-token prefill, sliced into discarded steps first."""
    sampler, _ = _sampler({0: SamplingParams(temperature=0.9, seed=42), **other})
    slots = [0, *other]
    rest = [0] * len(other)
    torch.manual_seed(1)
    for step in range(discarded_steps):
        state = sampler._generators[0].get_state()
        sampler(torch.randn(len(slots), _VOCAB), _batch(slots, [step, *rest], [1, *rest], [12, *rest]))
        assert torch.equal(sampler._generators[0].get_state(), state)
    computed = discarded_steps
    first = sampler(
        torch.full((len(slots), _VOCAB), 0.0), _batch(slots, [computed, *rest], [12 - computed, *rest], [12, *rest])
    )
    codes = [int(first[0])]
    torch.manual_seed(2)
    for _ in range(8):
        sampled = sampler(torch.randn(len(slots), _VOCAB), _batch(slots, [12, *rest], [1, *rest], [12, *rest]))
        codes.append(int(sampled[0]))
    return codes


@pytest.mark.parametrize(
    "other",
    [{}, {1: SamplingParams(temperature=0.0)}, {1: SamplingParams(temperature=0.9, seed=7)}],
    ids=["seeded-only", "mixed-greedy", "two-seeded"],
)
def test_discarded_prefill_steps_do_not_advance_the_seeded_stream(other):
    expected = _accepted_codes(other, discarded_steps=0)
    assert _accepted_codes(other, discarded_steps=1) == expected
    assert _accepted_codes(other, discarded_steps=3) == expected


def test_unseeded_step_stays_on_upstream_sampler():
    sampler, base = _sampler({0: SamplingParams(temperature=0.9), 1: SamplingParams(temperature=0.0)})
    logits = torch.randn(2, _VOCAB)
    sampled = _step(sampler, [0, 1], logits)
    assert torch.equal(sampled, logits.argmax(dim=-1))
    assert base.upstream_rows == [2]
