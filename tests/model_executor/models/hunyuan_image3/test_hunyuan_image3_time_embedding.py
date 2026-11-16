# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR time conditioning must use the same full-weight MLP as DiT."""

import pytest
import torch
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    TimestepEmbedder as DiTEmbedder,
)
from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import TimestepEmbedder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ar_time_embedding_loads_full_weights_and_matches_dit(dtype):
    reference = DiTEmbedder(hidden_size=16, frequency_embedding_size=8, out_size=12, dtype=dtype)
    actual = TimestepEmbedder(hidden_size=16, frequency_embedding_size=8, out_size=12, dtype=dtype)
    weights = reference.state_dict()
    loaded = AutoWeightsLoader(actual).load_weights(weights.items())
    assert loaded == set(weights)
    for name, parameter in actual.named_parameters():
        assert parameter.shape == weights[name].shape
        assert parameter.dtype == dtype
    for timestep in (torch.tensor(0.0), torch.tensor([0.0, 1.0, 500.0])):
        with torch.inference_mode():
            expected = reference(timestep.reshape(-1))
            torch.testing.assert_close(actual(timestep), expected, rtol=0, atol=0)
