# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.cache.seacache.sea_filter import apply_sea_filter

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("dimensions", [(0, 1, 2), (2, 1, 0)])
def test_real_gain_scales_complex_spectrum_on_npu(dimensions):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an available NPU")

    # Non-contiguous THWC input with odd/even axis lengths and nonzero phase.
    x = torch.randn(7, 3, 4, 5, generator=torch.Generator().manual_seed(696)).movedim(0, -1)
    expected = apply_sea_filter(x, 0.93, dimensions=dimensions)
    actual = apply_sea_filter(x.to("npu"), 0.93, dimensions=dimensions)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
