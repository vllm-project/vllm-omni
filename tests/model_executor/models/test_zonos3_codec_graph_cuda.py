# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Actual CUDA capture/replay of the complete ZAC decode and synthesis path."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.test_zonos3_codec import graph_network
from vllm_omni.model_executor.models.zonos3.cuda_graph_wrapper import ZACCodecGraphWrapper
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}),
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="CUDA capture/replay requires an NVIDIA GPU"),
]


@torch.inference_mode()
def test_full_codec_graph_padding_final_and_reused_slots():
    network = graph_network("cuda")
    state = network.allocate_state(8)
    eager_state = network.allocate_state(8)
    graph = ZACCodecGraphWrapper(network, state, 4, (4,), (4,))
    eager = ZACCodecGraphWrapper(network, eager_state, 4, (4,), (4,))
    graph.capture()
    assert set(graph.buckets) == {(4, 4)}
    assert not state.frames.any() and not graph.synthesis.columns.any()
    slots = torch.tensor([2, 0], device="cuda")
    for lengths, final in (([1, 3], [False, False]), ([3, 2], [True, False]), ([0, 0], [False, True])):
        codes = torch.randint(0, 4096, (2, 3, 10), device="cuda")
        valid = torch.tensor(lengths, device="cuda")
        last = torch.tensor(final, device="cuda")
        if lengths[0] == 0:
            # Finished rows are absent from a real cohort. Address a scratch
            # slot rather than reopening the completed stream as a padding row.
            slots = torch.tensor([5, 0], device="cuda")
        actual, actual_n = graph(codes, slots, valid, last)
        expected, expected_n = eager(codes, slots, valid, last)
        assert actual_n.tolist() == expected_n.tolist()
        for row, n in enumerate(actual_n.tolist()):
            # BF16 transformer GEMMs may round differently with padded B/T.
            torch.testing.assert_close(actual[row, :n], expected[row, :n], atol=3e-5, rtol=1e-2)
    assert graph.replay_count == 3 and graph.eager_count == 0
    assert state.frames[:4].tolist() == eager_state.frames[:4].tolist()
    # Capture and zero-length padding never commit recurrent state.
    assert not state.frames[4:].any() and not graph.synthesis.columns[4:].any()
    slots = torch.tensor([0], device="cuda")
    for wrapper in (graph, eager):
        wrapper.state.reset(slots)
        wrapper.reset(slots)
    # An out-of-bucket chunk uses the identical state layout eagerly, then
    # a padded final chunk resumes replay without losing history.
    long_codes = torch.randint(0, 4096, (1, 5, 10), device="cuda")
    long_lengths = torch.tensor([5], device="cuda")
    continuing = torch.zeros(1, dtype=torch.bool, device="cuda")
    actual, actual_n = graph(long_codes, slots, long_lengths, continuing)
    expected, expected_n = eager(long_codes, slots, long_lengths, continuing)
    assert actual_n.tolist() == expected_n.tolist() == [13230]
    torch.testing.assert_close(actual[0, :13230], expected[0, :13230], atol=3e-5, rtol=1e-2)
    codes = torch.randint(0, 4096, (1, 1, 10), device="cuda")
    lengths = torch.ones(1, dtype=torch.long, device="cuda")
    final = torch.ones(1, dtype=torch.bool, device="cuda")
    actual, actual_n = graph(codes, slots, lengths, final)
    expected, expected_n = eager(codes, slots, lengths, final)
    assert actual_n.tolist() == expected_n.tolist() == [4410]
    torch.testing.assert_close(actual[0, :4410], expected[0, :4410], atol=3e-5, rtol=1e-2)
    assert graph.replay_count == 4 and graph.eager_count == 1
