# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Only multi-rank Conformer length updates require a collective."""

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.asr.conformer_encoder import (
    ConformerEncoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _EncoderState:
    sync_max_audio_length: bool = True
    max_audio_length: int = 16

    def set_max_audio_length(self, length: int) -> None:
        self.max_audio_length = length


@pytest.mark.parametrize("length, expected", [(12, 16), (20, 20)])
def test_single_rank_uses_local_length_without_device_tensor(monkeypatch, length, expected) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    def unexpected_tensor(*args, **kwargs):
        raise AssertionError("a single rank does not need a length tensor or collective")

    monkeypatch.setattr(torch, "tensor", unexpected_tensor)
    encoder = _EncoderState()
    ConformerEncoder.update_max_seq_length(encoder, length, torch.device("cpu"))
    assert encoder.max_audio_length == expected


def test_multiple_ranks_retain_global_maximum(monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    calls = []

    def global_max(tensor: torch.Tensor, op) -> None:
        assert op == torch.distributed.ReduceOp.MAX
        calls.append(tensor.item())
        tensor.fill_(32)

    monkeypatch.setattr(torch.distributed, "all_reduce", global_max)
    encoder = _EncoderState()
    ConformerEncoder.update_max_seq_length(encoder, 20, torch.device("cpu"))
    assert calls == [20]
    assert encoder.max_audio_length == 32


@pytest.mark.parametrize("sync, initialized", [(False, True), (True, False)])
def test_disabled_sync_or_uninitialized_distributed_stays_local(monkeypatch, sync, initialized) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)

    def unexpected_world_size():
        raise AssertionError("inactive distributed synchronization must not query the process group")

    monkeypatch.setattr(torch.distributed, "get_world_size", unexpected_world_size)
    encoder = _EncoderState(sync_max_audio_length=sync)
    ConformerEncoder.update_max_seq_length(encoder, 20, torch.device("cpu"))
    assert encoder.max_audio_length == 20
