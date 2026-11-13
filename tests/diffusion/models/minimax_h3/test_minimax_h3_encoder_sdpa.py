# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterator

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def restore_cudnn_sdp_state() -> Iterator[None]:
    original = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        yield
    finally:
        torch.backends.cuda.enable_cudnn_sdp(original)


def _encoder_without_weights() -> MiniMaxH3Qwen3VLEncoder:
    encoder = MiniMaxH3Qwen3VLEncoder("unused", device=torch.device("cpu"), load_model=False)
    encoder.text_model = nn.Linear(1, 1)
    return encoder


@pytest.mark.parametrize("initial_state", [False, True])
def test_encode_ids_restores_cudnn_sdp_after_success(monkeypatch: pytest.MonkeyPatch, initial_state: bool) -> None:
    encoder = _encoder_without_weights()
    expected = torch.tensor([1.0])

    def encode(*args, **kwargs):
        assert torch.backends.cuda.cudnn_sdp_enabled()
        return expected

    monkeypatch.setattr(encoder, "_encode", encode)
    torch.backends.cuda.enable_cudnn_sdp(initial_state)

    actual = encoder.encode_ids(torch.tensor([1]))

    torch.testing.assert_close(actual, expected)
    assert torch.backends.cuda.cudnn_sdp_enabled() is initial_state


@pytest.mark.parametrize("initial_state", [False, True])
def test_encode_ids_restores_cudnn_sdp_after_failure(monkeypatch: pytest.MonkeyPatch, initial_state: bool) -> None:
    encoder = _encoder_without_weights()

    def fail(*args, **kwargs):
        assert torch.backends.cuda.cudnn_sdp_enabled()
        raise RuntimeError("encode failed")

    monkeypatch.setattr(encoder, "_encode", fail)
    torch.backends.cuda.enable_cudnn_sdp(initial_state)

    with pytest.raises(RuntimeError, match="encode failed"):
        encoder.encode_ids(torch.tensor([1]))

    assert torch.backends.cuda.cudnn_sdp_enabled() is initial_state
