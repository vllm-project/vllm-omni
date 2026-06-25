# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import nn

from vllm_omni.diffusion.models.ltx2.ltx2_transformer import _make_rms_norm


def test_ltx_rms_norm_no_affine_identity_weight_is_non_persistent_buffer():
    norm = _make_rms_norm(8, eps=1e-6, elementwise_affine=False)

    assert "weight" not in dict(norm.named_parameters())
    assert "weight" in dict(norm.named_buffers())
    assert "weight" not in norm.state_dict()


def test_ltx_rms_norm_affine_weight_remains_parameter():
    norm = _make_rms_norm(8, eps=1e-6, elementwise_affine=True)

    assert isinstance(dict(norm.named_parameters())["weight"], nn.Parameter)
    assert "weight" not in dict(norm.named_buffers())
    assert "weight" in norm.state_dict()
