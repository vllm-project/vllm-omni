# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-known Delay mask boundaries must not trigger CUDA shape discovery."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


@pytest.mark.parametrize("n_vq", [16, 32])
def test_delay_active_interval_matches_mask(n_vq: int) -> None:
    idx = torch.arange(n_vq)
    for audio_lengths in range(-2, n_vq + 3):
        for delayed_lengths in range(-2, n_vq + 3):
            post = torch.ones(n_vq, dtype=torch.bool) if delayed_lengths == -1 else idx > delayed_lengths - 1
            expected = idx[(idx < audio_lengths) & post].tolist()
            actual = list(range(max(0, delayed_lengths), min(n_vq, audio_lengths)))
            assert actual == expected


@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize(
    "audio_lengths,delayed_lengths",
    [(0, -1), (1, -1), (4, -1), (4, 2), (4, 4), (-2, -2), (4, -2), (7, -1), (4, 7), (2, 3)],
)
@torch.inference_mode()
def test_delay_audio_sampling_avoids_mask_discovery(
    monkeypatch: pytest.MonkeyPatch, stacked: bool, audio_lengths: int, delayed_lengths: int
) -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSDelayTalkerForGeneration,
    )

    cls = MossTTSDelayTalkerForGeneration
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.n_vq = 4
    model.audio_pad_code = 32
    model.audio_heads = torch.nn.ModuleList([torch.nn.Linear(8, 33, bias=False) for _ in range(4)])
    model._stacked_audio_head_w = torch.stack([head.weight for head in model.audio_heads]) if stacked else None

    def forbidden(*args, **kwargs):
        raise AssertionError("Delay sampling must not inspect a tensor mask on the host")

    with monkeypatch.context() as patch:
        for method in ("__bool__", "nonzero", "item", "__int__", "tolist", "__index__", "__float__"):
            patch.setattr(torch.Tensor, method, forbidden)
        result = model._sample_audio_codes(
            torch.randn(1, 8), dict(audio_lengths=audio_lengths, delayed_lengths=delayed_lengths)
        )
    idx = torch.arange(model.n_vq)
    post_mask = torch.ones_like(idx, dtype=torch.bool) if delayed_lengths == -1 else idx >= delayed_lengths
    expected_active = (idx < audio_lengths) & post_mask
    assert result.shape == (4,)
    assert result.dtype == torch.long
    assert torch.equal(result.ne(32), expected_active)
