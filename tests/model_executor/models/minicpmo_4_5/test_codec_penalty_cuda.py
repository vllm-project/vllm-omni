# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA parity and capture checks for the windowed codec penalty."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _apply_batched_repetition_penalty


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("batch_size", [1, 4, 65])
def test_codec_penalty_cuda_graph_matches_eager_reference(batch_size):
    generator = torch.Generator().manual_seed(42)
    cpu_logits = torch.randn(batch_size, 64, generator=generator)
    cpu_histories = [torch.randint(64, (16,), generator=generator) for _ in range(batch_size)]
    penalties = torch.linspace(1.0, 1.2, batch_size)
    expected = cpu_logits.clone()
    expected_negated = cpu_logits.clone()
    for row, history in enumerate(cpu_histories):
        frequency = torch.bincount(history, minlength=64).float()
        alpha = penalties[row].pow(frequency)
        expected[row] = torch.where(cpu_logits[row] < 0, cpu_logits[row] * alpha, cpu_logits[row] / alpha)
        negative = -cpu_logits[row]
        expected_negated[row] = torch.where(negative < 0, negative * alpha, negative / alpha)
    logits = cpu_logits.cuda()
    histories = [history.cuda() for history in cpu_histories]
    device_penalties = penalties.cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _apply_batched_repetition_penalty(logits, histories, penalty=device_penalties, window_size=16)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _apply_batched_repetition_penalty(logits, histories, penalty=device_penalties, window_size=16)
    graph.replay()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=1e-6)
    # Replay must consume updated logits without modifying the source tensor.
    logits.neg_()
    graph.replay()
    torch.testing.assert_close(actual.cpu(), expected_negated, rtol=1e-6, atol=1e-6)
