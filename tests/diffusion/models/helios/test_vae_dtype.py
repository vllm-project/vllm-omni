# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.helios.pipeline_helios import resolve_vae_dtype

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_none_keeps_pipeline_default():
    assert resolve_vae_dtype(None) is torch.float32


def test_valid_dtype_names_resolve():
    assert resolve_vae_dtype("float32") is torch.float32
    assert resolve_vae_dtype("bfloat16") is torch.bfloat16
    assert resolve_vae_dtype("float16") is torch.float16


@pytest.mark.parametrize("name", ["fp8_nope", "cuda", "int8", "bool"])
def test_invalid_dtype_name_raises(name):
    with pytest.raises(ValueError, match="vae_dtype"):
        resolve_vae_dtype(name)
