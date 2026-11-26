# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.platforms import current_platform

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.higgs_audio_v2.delay_pattern_mask import mask_audio_logits_

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": ["L4", "H100", "B200"]}, num_cards=1),
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA mask kernel"),
]


def _reference(logits, states, bos, eos):
    """The original per-codebook masking, executed on CPU for comparison."""
    for i, (delay, remaining) in enumerate(states):
        for q in range(logits.shape[1]):
            forced = None
            if remaining >= 0 and q < logits.shape[1] - remaining:
                forced = eos
            elif remaining < 0 and q > delay:
                forced = bos
            if forced is not None:
                keep = logits[i, q, forced].clone()
                logits[i, q].fill_(float("-inf"))
                logits[i, q, forced] = keep
            else:
                logits[i, q, bos] = float("-inf")
                if remaining >= 0 or q != 0:
                    logits[i, q, eos] = float("-inf")
    return logits


@pytest.mark.parametrize("q,v", [(2, 8), (8, 1026)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_delay_mask_matches_original(q, v, dtype, strided):
    states = [(delay, remaining) for delay in range(q) for remaining in range(-1, q + 1)]
    raw = torch.randn((len(states), q, v * (2 if strided else 1)), dtype=dtype, device="cuda")
    logits = raw[..., ::2] if strided else raw
    logits[0, 0, 0] = float("nan")
    logits[0, 0, 1] = float("inf")
    expected = _reference(logits.cpu(), states, v - 2, v - 1)
    padding = raw[..., 1::2].cpu() if strided else None
    pointer = logits.data_ptr()
    mask_audio_logits_(logits, states, v - 2, v - 1)
    assert logits.data_ptr() == pointer
    torch.testing.assert_close(logits.cpu(), expected, atol=0, rtol=0, equal_nan=True)
    if padding is not None:
        torch.testing.assert_close(raw[..., 1::2].cpu(), padding, atol=0, rtol=0)


def test_empty_audio_batch():
    logits = torch.empty((0, 8, 1026), device="cuda", dtype=torch.bfloat16)
    mask_audio_logits_(logits, [], 1024, 1025)
    assert logits.shape == (0, 8, 1026)
