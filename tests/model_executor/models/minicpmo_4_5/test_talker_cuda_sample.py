# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.triton_utils import triton

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.minicpmo_4_5.talker_cuda_sample import _validate_argmax, sample

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
def test_cpu_fallback_preserves_rng():
    probabilities = torch.tensor([[0.1, 0.2, 0.7]])
    expected_rng = torch.Generator().manual_seed(27)
    actual_rng = torch.Generator().manual_seed(27)
    for _ in range(20):
        expected = torch.multinomial(probabilities, 1, generator=expected_rng).reshape(1)
        torch.testing.assert_close(sample(probabilities, actual_rng), expected)
    assert torch.equal(expected_rng.get_state(), actual_rng.get_state())


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("width", [25, 626, 6562])
@pytest.mark.parametrize("sparse", [False, True])
def test_cuda_sample_and_graph_preserve_tokens_and_rng(width, sparse):
    logits = torch.randn(1, width, device="cuda")
    if sparse:
        logits[:, 20:] = -torch.inf
    probabilities = logits.softmax(-1)
    expected_rng = torch.Generator("cuda").manual_seed(987)
    actual_rng = torch.Generator("cuda").manual_seed(987)
    for _ in range(100):
        expected = torch.multinomial(probabilities, 1, generator=expected_rng).reshape(1)
        assert torch.equal(expected, sample(probabilities, actual_rng))
    assert torch.equal(expected_rng.get_state(), actual_rng.get_state())
    graphs, outputs = [], []
    for fn in (
        lambda: torch.multinomial(probabilities, 1, generator=expected_rng).reshape(1),
        lambda: sample(probabilities, actual_rng),
    ):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn()
        graphs.append(graph)
        outputs.append(output)
    expected_rng.manual_seed(12345)
    actual_rng.manual_seed(12345)
    for _ in range(100):
        for graph in graphs:
            graph.replay()
        assert torch.equal(*outputs)
    assert torch.equal(expected_rng.get_state(), actual_rng.get_state())


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 0.0])
def test_invalid_probability_flag(bad):
    # Inspect the flag without poisoning the process with a device assertion.
    probabilities = torch.zeros((1, 6562), device="cuda")
    probabilities[0, 5] = bad
    output = torch.empty(1, device="cuda", dtype=torch.long)
    valid = torch.empty(1, device="cuda", dtype=torch.bool)
    _validate_argmax[(1,)](probabilities, probabilities, output, valid, 6562, triton.next_power_of_2(6562))
    assert not valid.item()
