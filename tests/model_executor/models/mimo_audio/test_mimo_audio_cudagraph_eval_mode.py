# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import contextlib
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _mimo_llm_module():
    """Defer the mimo_audio_llm import (pulls vLLM model_executor) until use."""
    from vllm_omni.model_executor.models.mimo_audio import mimo_audio_llm

    return mimo_audio_llm


class _RecordingTransformer(torch.nn.Module):
    """Minimal transformer stand-in that records the training flag seen during
    the captured forward, the property that decides whether attention dropout
    is active."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_training: list[bool] = []

    def forward(self, *args, **kwargs):
        self.seen_training.append(self.training)
        return SimpleNamespace(last_hidden_state=torch.zeros(1))


class _LocalDecodeModel:
    def __init__(self) -> None:
        self.local_transformer = _RecordingTransformer()

    def base_local_forward(self, input_tensor, local_sampler=None):
        self.local_transformer()
        return torch.zeros(1)


class _InputLocalModel:
    def __init__(self) -> None:
        self.input_local_transformer = _RecordingTransformer()


def _mock_cuda_capture(mocker: MockerFixture, module):
    """Replace CUDA graph machinery so capture() runs on a CPU-only runner."""
    fake_graph = mocker.patch.object(torch.cuda, "CUDAGraph")
    fake_graph.return_value = SimpleNamespace(replay=lambda: None)
    mocker.patch.object(torch.cuda, "graph", lambda *args, **kwargs: contextlib.nullcontext())
    mocker.patch.object(module.current_platform, "get_global_graph_pool", return_value=None)


def test_local_decode_capture_forces_eval_mode(mocker: MockerFixture):
    module = _mimo_llm_module()
    _mock_cuda_capture(mocker, module)
    buffer = mocker.Mock()
    buffer.inputs.return_value = (torch.zeros(1), None)
    model = _LocalDecodeModel()
    assert model.local_transformer.training

    module.MiMoLocalDecodeCudaGraph.capture(model, buffer, batch_size=1)

    assert model.local_transformer.training is False
    # The recorded (graph) forward must observe eval mode, otherwise the
    # attention_dropout from the configs is baked into the graph (#8601).
    assert model.local_transformer.seen_training == [False, False]


def test_input_local_transformer_capture_forces_eval_mode(mocker: MockerFixture):
    module = _mimo_llm_module()
    _mock_cuda_capture(mocker, module)
    buffer = mocker.Mock()
    buffer.inputs.return_value = torch.zeros(1)
    model = _InputLocalModel()
    assert model.input_local_transformer.training

    module.MiMoInputLocalTransformerCudaGraph.capture(model, buffer, batch_size=1)

    assert model.input_local_transformer.training is False
    assert model.input_local_transformer.seen_training == [False, False]
