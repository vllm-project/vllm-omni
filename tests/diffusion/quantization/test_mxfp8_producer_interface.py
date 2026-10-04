# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.layers.mxfp8 import mxfp8_linear, mxfp8_quantize_project, silu_mxfp8_linear

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("rows,hidden,out_features", [(1, 32, 128), (137, 96, 256), (256, 128, 384)])
def test_export_keeps_quantized_input_and_scale_as_explicit_outputs(rows, hidden, out_features):
    class Projection(torch.nn.Module):
        def forward(self, x, weight, scale):
            return mxfp8_quantize_project(x, weight, scale)

    x = torch.empty(rows, hidden, dtype=torch.bfloat16, device="meta")
    weight = torch.empty(out_features, hidden, dtype=torch.float8_e4m3fn, device="meta")
    scale = torch.empty(4096, dtype=torch.float8_e8m0fnu, device="meta")
    exported = torch.export.export(Projection(), (x, weight, scale))
    output, payload, activation_scale = exported.module()(x, weight, scale)
    assert output.shape == (rows, out_features) and output.dtype == torch.bfloat16
    assert payload.shape == x.shape and payload.dtype == torch.float8_e4m3fn
    assert activation_scale.shape == (((rows + 127) // 128) * 128 * ((hidden // 32 + 3) // 4) * 4,)
    assert activation_scale.dtype == torch.uint8
    assert len(exported.graph_signature.output_specs) == 3


@pytest.mark.parametrize("operation", [mxfp8_linear, silu_mxfp8_linear])
def test_projection_fake_preserves_leading_dimensions(operation):
    x = torch.empty(2, 3, 64, dtype=torch.bfloat16, device="meta")
    weight = torch.empty(128, 64 if operation is mxfp8_linear else 32, dtype=torch.float8_e4m3fn, device="meta")
    scale = torch.empty(512, dtype=torch.float8_e8m0fnu, device="meta")
    output = operation(x, weight, scale)
    assert output.shape == (2, 3, 128) and output.dtype == torch.bfloat16
