# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import WhisperConfig

from vllm_omni.model_executor.models.common.whisper_vq import WhisperVQEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def _make_encoder() -> WhisperVQEncoder:
    encoder = WhisperVQEncoder.__new__(WhisperVQEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.config = WhisperConfig(pooling_kernel_size=2)
    encoder.pooling_layer = torch.nn.MaxPool1d(kernel_size=2)
    encoder.codebook = torch.nn.Embedding(8, 2)
    encoder.embed_positions2 = None
    return encoder


def test_variable_length_helpers_batch_length_transfer(monkeypatch: pytest.MonkeyPatch):
    encoder = _make_encoder()
    hidden_states = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2)
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
            [False, False, False, False, False],
        ]
    )

    def fail_item(*_args, **_kwargs):
        raise AssertionError("per-sample Tensor.item() synchronizes the device")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    pooled, pooled_mask = encoder._apply_pooling(hidden_states, valid_mask)
    quantized, token_ids = encoder._apply_vq(hidden_states, valid_mask)

    assert pooled.shape == (3, 3, 2)
    assert pooled_mask.sum(dim=1).tolist() == [3, 2, 0]
    assert token_ids is not None
    assert token_ids.ne(-1).sum(dim=1).tolist() == [5, 3, 0]
    assert torch.count_nonzero(quantized[2]) == 0
