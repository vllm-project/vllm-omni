# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HiFT graphs consume live inputs and return independently owned waveforms."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_incremental_hift import _make_hift

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("finalize", [False, True])
@torch.inference_mode()
def test_hift_graph_preserves_current_and_previous_request(finalize):
    hift = _make_hift().cuda().eval()
    hift.remove_weight_norm()
    hift.f0_predictor.remove_weight_norm()
    hift.enable_decode_graphs()
    generator = torch.Generator(device="cuda").manual_seed(17)
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(5):
        mel = torch.randn(1, 80, 30, device="cuda", generator=generator)
        source = torch.randn(1, 1, 30 * 480, device="cuda", generator=generator)
        expected = hift._decode_eager(mel, source, finalize=finalize)
        actual = hift.decode(mel, source, finalize=finalize)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for previous, snapshot in saved:
            assert torch.equal(previous, snapshot)
        if hift._decode_graphs.graphs:
            saved.append((actual, actual.clone()))
    assert len(hift._decode_graphs.graphs) == 1
    hift._decode_graphs.max_graphs = 1
    mel = torch.randn(1, 80, 31, device="cuda", generator=generator)
    source = torch.randn(1, 1, 31 * 480, device="cuda", generator=generator)
    actual = hift.decode(mel, source, finalize=finalize)
    expected = hift._decode_eager(mel, source, finalize=finalize)
    assert torch.equal(actual, expected)
    assert len(hift._decode_graphs.graphs) == 1
