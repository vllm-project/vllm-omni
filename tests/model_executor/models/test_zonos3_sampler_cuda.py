# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA filtering of the ten RVQ heads, including single-request broadcasts."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.test_zonos3_sampler import add_sampling, codes, make_batch, make_state, run
from vllm_omni.model_executor.models.zonos3.model_state import Zonos3ARModelState
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}),
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="Requires CUDA top-k/top-p kernels"),
]


@pytest.mark.parametrize("order", [[0], [1, 0]])
@torch.inference_mode()
def test_top_k_filters_every_codebook_with_single_and_reordered_requests(order):
    device = torch.device("cuda")
    settings = {
        "top_k": torch.tensor([5, 9], device=device, dtype=torch.int32),
        "top_p": torch.ones(2, device=device),
        "temperature": torch.ones(2, device=device),
        "min_p": torch.zeros(2, device=device),
        "min_frames": torch.zeros(2, device=device, dtype=torch.int32),
    }
    owner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(codebook_size=64, audio_vocab=66, eoa_id=64)),
        sampling_state=SimpleNamespace(
            position=torch.zeros(2, device=device, dtype=torch.int64),
            columns=torch.arange(10, device=device),
        ),
        _setting=lambda name, slots: settings[name].index_select(0, slots),
    )
    slots = torch.tensor(order, device=device)
    logits = torch.arange(66, device=device, dtype=torch.float32).expand(len(order), 10, -1).clone()
    probs = Zonos3ARModelState.probabilities(owner, logits, slots, 0)
    assert torch.isfinite(probs).all()
    for row, slot in enumerate(order):
        expected = torch.arange(66, device=device) >= 65 - (5 if slot == 0 else 9)
        expected[65] = False  # Audio padding must never be sampled.
        assert torch.equal(probs[row] > 0, expected.expand(10, -1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_gpu_warmup_determined_tokens_and_owned_snapshots(dtype):
    from vllm import SamplingParams

    state = make_state(4, "cuda", dtype)
    _, requests = make_batch([0, 1, 2, 3], device="cuda")
    state.warmup_sampling(torch.randn(4, 16, device="cuda", dtype=dtype), requests)
    assert state.options == [None] * 4 and not state.sampling_state.position.any()
    singles = [make_state(1, "cuda", dtype) for _ in range(4)]
    for slot, single in enumerate(singles):
        params = SamplingParams(temperature=0.7, top_k=5, top_p=0.85, seed=100 + slot)
        add_sampling(state, slot, params)
        add_sampling(single, 0, params)
        single.apply_staged_writes()
    state.apply_staged_writes()
    previous, saved = None, None
    for step in range(4):
        hidden = torch.randn(4, 16, device="cuda", dtype=dtype)
        order = [3, 2, 1, 0] if step % 2 else [0, 1, 2, 3]
        output, snapshot = run(state, order, hidden[order])
        assert output.num_sampled.tolist() == [1] * 4
        for row, slot in enumerate(order):
            _, expected = run(singles[slot], [0], hidden[slot : slot + 1])
            assert torch.equal(codes(snapshot)[row], codes(expected)[0])
        if previous is not None:
            assert torch.equal(codes(previous), saved)
        previous, saved = snapshot, codes(snapshot).clone()
