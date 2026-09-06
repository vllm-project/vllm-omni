# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.mimo_audio.mimo_audio_code2wav import MiMoAudioTokenizerWorker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_group_by_length_batches_length_transfer(monkeypatch: pytest.MonkeyPatch):
    worker = MiMoAudioTokenizerWorker.__new__(MiMoAudioTokenizerWorker)
    features = torch.arange(30, dtype=torch.float32).reshape(15, 2)
    lengths = torch.tensor([4, 3, 6, 2])

    def fail_item(*_args, **_kwargs):
        raise AssertionError("per-length Tensor.item() synchronizes the device")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    feature_groups, length_groups = worker.group_by_length(features, lengths, max_length=7)

    assert [group.shape[0] for group in feature_groups] == [7, 6, 2]
    assert [group.tolist() for group in length_groups] == [[4, 3], [6], [2]]
