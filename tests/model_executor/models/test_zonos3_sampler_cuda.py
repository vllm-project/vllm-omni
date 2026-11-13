# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA filtering of the ten RVQ heads, including single-request broadcasts."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.zonos3.sampler import Zonos3Sampler
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
        config=SimpleNamespace(codebook_size=64, audio_vocab=66, eoa_id=64),
        state=SimpleNamespace(
            position=torch.zeros(2, device=device, dtype=torch.int64),
            columns=torch.arange(10, device=device),
        ),
        _setting=lambda name, slots: settings[name].index_select(0, slots),
    )
    slots = torch.tensor(order, device=device)
    logits = torch.arange(66, device=device, dtype=torch.float32).expand(len(order), 10, -1).clone()
    probs = Zonos3Sampler.probabilities(owner, logits, slots, 0)
    assert torch.isfinite(probs).all()
    for row, slot in enumerate(order):
        expected = torch.arange(66, device=device) >= 65 - (5 if slot == 0 else 9)
        expected[65] = False  # Audio padding must never be sampled.
        assert torch.equal(probs[row] > 0, expected.expand(10, -1))
