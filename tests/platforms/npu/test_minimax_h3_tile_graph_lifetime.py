# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device coverage for outputs that outlive their graph replica's next replay."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms.npu.models.minimax_h3_image import _NPUTileGraphs

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"npu": "A2"}, num_cards=1)]


def test_two_streams_preserve_three_distinct_tiles():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    generator = torch.Generator().manual_seed(73)
    tiles = [torch.randn(1, 24, 1, 16, 16, generator=generator).to("npu") for _ in range(3)]

    def decode(value):
        return value.square() + value * 3

    runner = _NPUTileGraphs(decode, autocast_dtype=torch.bfloat16)
    runner.replicas = 2
    with torch.inference_mode():
        expected = [decode(tile) for tile in tiles]
        outputs = runner.map(tiles)
        # Replica zero is reused for tile two. Reuse both replicas again before
        # checking any earlier result, so aliases of static outputs cannot pass.
        runner.map([tile + 1 for tile in tiles])
        torch.npu.synchronize()
        assert not torch.equal(expected[0], expected[2])
        for actual, reference in zip(outputs, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
