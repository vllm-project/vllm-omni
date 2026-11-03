# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU parity for frame-local K/V prefixes; no weights or RNG are reused."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.easymagpie.local_transformer import EasyMagpieLocalTransformer, EasyMagpieLTLayer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def transformer(prefix_list, dtype):
    model = EasyMagpieLocalTransformer.__new__(EasyMagpieLocalTransformer)
    nn.Module.__init__(model)
    model.position_embeddings = nn.Embedding(14, 16, dtype=dtype)
    model.register_buffer("_positions", torch.arange(14), persistent=False)
    model.layers = nn.ModuleList([EasyMagpieLTLayer(16, 64, 4).to(dtype) for _ in range(2)])
    model.norm_out = nn.Identity()
    model.prefix_list_enabled = prefix_list
    return model.eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("batch", [1, 3, 8])
def test_all_twelve_positions_match_and_each_frame_starts_empty(dtype, batch):
    torch.manual_seed(7)
    native = transformer(False, dtype)
    candidate = deepcopy(native)
    candidate.prefix_list_enabled = True
    # A second frame with reversed rows must not inherit the preceding prefix.
    with torch.inference_mode():
        for _ in range(2):
            inputs = torch.randn(batch, 12, 16, dtype=dtype).flip(0)
            left = native.allocate_frame_cache(inputs, 12)
            right = candidate.allocate_frame_cache(inputs, 12)
            assert all(not keys and not values for keys, values in right)
            for position in range(12):
                x = inputs[:, position : position + 1]
                expected = native.forward_step(x, left, position)
                actual = candidate.forward_step(x, right, position)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert all(len(keys) == len(values) == position + 1 for keys, values in right)
    assert not torch.cuda.is_initialized()
