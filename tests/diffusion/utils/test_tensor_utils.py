# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.utils.tensor_utils import expand_scalar_to_batch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_expand_scalar_to_batch_avoids_scalar_item(monkeypatch):
    scalar = torch.tensor(3.5, dtype=torch.float32)

    def fail_item(_self):
        raise AssertionError("scalar expansion must stay tensor-native")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    actual = expand_scalar_to_batch(scalar, 4, dtype=torch.float64)

    assert actual.shape == (4,)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, torch.full((4,), 3.5, dtype=torch.float64))
