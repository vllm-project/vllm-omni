# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock, create_autospec, patch

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.model_executor.models.qwen3_omni.talker_sampler import Qwen3OmniTalkerSampler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_VOCAB = 64


class _BaseSampler:
    def __init__(self) -> None:
        self.penalties_state = SimpleNamespace(device=torch.device("cpu"))
        self.upstream_rows: list[int] = []

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        pass

    def _sample_random(
        self, logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, has_topp_only_rows, use_fused_sampler
    ):
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
            False,
        )
        return sampled


def test_custom_sampler_registers_native_masking_kernels(monkeypatch):
    """Seeded Talker sampling uses native masking even with FlashInfer available."""
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    # The talker sampler registers the vendored kernels the module it reads
    # imports (upstream moved v1.sample.ops at the pin; omni vendored it).
    from vllm_omni.worker.v1_compat.sample_ops import topk_topp_sampler

    monkeypatch.setattr(
        topk_topp_sampler,
        "current_platform",
        SimpleNamespace(is_cpu=lambda: False, is_cuda_alike=lambda: True),
    )
    registry = JitWarmupRegistry(SimpleNamespace())
    with registry.activate():
        Qwen3OmniTalkerSampler(_BaseSampler())

    assert set(registry._registrations) == {
        topk_topp_sampler._topk_topp,
        topk_topp_sampler._topp_split_stats,
        topk_topp_sampler._topp_split_step,
        topk_topp_sampler._topp_split_mask,
    }
    assert len(registry) == 4


def _sampler(requests: dict[int, SamplingParams]) -> tuple[Qwen3OmniTalkerSampler, _BaseSampler]:
    base = _BaseSampler()
    sampler = Qwen3OmniTalkerSampler(base)
    for req_idx, params in requests.items():
        sampler.add_request(req_idx, params)
    return sampler, base


def _step(sampler: Qwen3OmniTalkerSampler, slots: list[int], logits: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(slots, dtype=torch.int64)
    sampled, _ = sampler.base_sampler._sample_random(
        logits.clone(), mapping, np.array(slots), torch.zeros(len(slots), dtype=torch.int64), None, None, False, False
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


@pytest.mark.parametrize("has_topp_only_rows", [False, True])
@pytest.mark.parametrize("seeded_rows", [0, 1, 2])
def test_target_sampler_dispatch_preserves_top_p_only_rows(has_topp_only_rows, seeded_rows):
    """Exercise the target runtime dispatcher, rather than an older hook stub."""
    base = Mock(spec=Sampler)
    logits = torch.zeros(2, _VOCAB)
    mapping = torch.arange(2, dtype=torch.int64)
    slots = np.arange(2)
    pos = torch.zeros(2, dtype=torch.int64)
    base._sample_random = create_autospec(
        Sampler._sample_random.__get__(base, Sampler),
        side_effect=lambda rows, *args: (rows.argmax(dim=-1), rows),
    )
    original = base._sample_random
    base.apply_sampling_params.side_effect = lambda logits, *args, **kwargs: logits
    base.sampling_states = Mock()
    base.sampling_states.get_top_k_top_p.return_value = (None, None, has_topp_only_rows)
    base.sampling_states.any_greedy.return_value = False
    base.sampling_states.any_explicit_seed.return_value = False
    base.use_flashinfer = False
    base.use_xpu_sampler = False
    base.penalties_state = Mock(device=torch.device("cpu"))
    sampler = Qwen3OmniTalkerSampler(base)
    for row in range(seeded_rows):
        sampler.add_request(row, SamplingParams(temperature=0.9, seed=row))
    with patch(
        "vllm_omni.model_executor.models.qwen3_omni.talker_sampler.apply_top_k_top_p", return_value=logits
    ) as mask:
        sampled, processed = Sampler.sample(base, logits, mapping, mapping, slots, pos, mapping, mapping, slots)

    assert sampled.shape == (2,)
    assert processed is logits
    if seeded_rows:
        mask.assert_called_once_with(logits, None, None, has_topp_only_rows=has_topp_only_rows)
    else:
        mask.assert_not_called()
    if seeded_rows == 0:
        original.assert_called_once_with(logits, mapping, slots, pos, None, None, has_topp_only_rows, False)
    elif seeded_rows == 1:
        original.assert_called_once()
        # The other row was already masked, so its upstream call must not mask again.
        assert original.call_args.args[-4:] == (None, None, False, False)
        assert original.call_args.args[0].shape == (1, _VOCAB)
        assert original.call_args.args[1].tolist() == [1]
        assert sampled[1] == 0
    else:
        original.assert_not_called()
