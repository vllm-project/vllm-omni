# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-repository CPU dispatch contracts, requiring the matching MindIE-SD source.

Only native operators are replaced. Public adapters, rotations, layouts, padding,
scale preparation, sequence metadata and output cropping execute their real code.
Operator substitutes carry floating tensors; these are not numerical quantization tests.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends import flash_attn
from vllm_omni.platforms.npu.quant.kv_quant_npu import get_quant_attention_rotation

mindiesd = pytest.importorskip("mindiesd")
torch_npu = importlib.import_module("torch_npu")

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_omni_calls_real_dense_public_api(monkeypatch, precision, layout):
    assert callable(getattr(mindiesd, "quant_attention", None)), "MindIE-SD must export quant_attention"
    monkeypatch.setattr(flash_attn, "current_omni_platform", SimpleNamespace(is_npu=lambda: True, device_name="npu"))
    monkeypatch.setattr(flash_attn, "get_current_diffusion_config_or_none", lambda: None)
    quantized_inputs = []

    def dynamic_quant(tensor, **kwargs):
        quantized_inputs.append((tensor.clone(), kwargs))
        shape = list(tensor.shape)
        axis = kwargs.get("axis", -1)
        shape[axis] = max(1, shape[axis] // 32)
        return tensor, torch.ones(shape, dtype=torch.uint8)

    monkeypatch.setattr(torch_npu, "npu_dynamic_block_quant", dynamic_quant, raising=False)
    monkeypatch.setattr(torch_npu, "npu_dynamic_mx_quant", dynamic_quant, raising=False)
    execute = Mock(side_effect=lambda query, *args, **kwargs: (query, None))
    metadata = Mock(return_value=torch.zeros(1, dtype=torch.int32))
    if precision == "fp8":
        monkeypatch.setattr(torch.ops.mindiesd, "fused_infer_attention_score_v2", execute, raising=False)
    elif precision == "mxfp8":
        monkeypatch.setattr(torch_npu, "npu_fused_infer_attention_score_v2", execute, raising=False)
    else:
        monkeypatch.setattr(torch.ops.mindiesd, "quant_flash_attn", execute, raising=False)
        monkeypatch.setattr(torch.ops.mindiesd, "quant_flash_attn_metadata", metadata, raising=False)

    query = torch.randn(1, 130, 2, 64, dtype=torch.bfloat16)
    if layout == "BNSD":
        query = query.transpose(1, 2)
    impl = flash_attn.FlashAttentionImpl(
        num_heads=2,
        head_size=64,
        softmax_scale=0.37,
        qkv_layout=layout,
        backend_kwargs={"quant": {"method": precision, "fallback": []}},
    )
    output = impl.forward_fa_quant_npu(query, query, query)
    expected = query
    if precision != "mxfp4":
        rotation = get_quant_attention_rotation(query.device, query.dtype, 64, 425500)
        expected = query @ rotation
    torch.testing.assert_close(output, expected)
    assert output.shape == query.shape and len(quantized_inputs) == 3
    execute.assert_called_once()
    kwargs = execute.call_args.kwargs
    assert kwargs["softmax_scale"] == 0.37
    if precision == "fp8":
        assert kwargs["input_layout"] == "BNSD"
        assert quantized_inputs[0][0].shape == (2, 130, 64)
        assert quantized_inputs[0][1]["row_block_size"] == 128
    elif precision == "mxfp8":
        assert kwargs["input_layout"] == "TND"
        assert kwargs["actual_seq_qlen"] == kwargs["actual_seq_kvlen"] == [130]
        assert [entry[1]["axis"] for entry in quantized_inputs] == [-1, -1, 0]
    else:
        seq_axis = 1 if layout == "BSND" else 2
        assert quantized_inputs[0][0].shape[seq_axis] == 512
        assert kwargs["seqused_q"].tolist() == kwargs["seqused_kv"].tolist() == [130]
        assert kwargs["layout_q"] == kwargs["layout_out"] == layout
        assert kwargs["q_dtype"] == torch_npu.float4_e2m1fn_x2
        assert kwargs["q_descale_dtype"] == torch_npu.float8_e8m0fnu
        assert execute.call_args.args[3].shape[:2] == (1, 2)
        assert [entry[1]["axis"] for entry in quantized_inputs] == [-1, -1, seq_axis]
        assert kwargs["metadata"] is metadata.return_value
        metadata.assert_called_once()


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("q_len,kv_len", [(17, 8), (8, 17)])
def test_omni_mxfp8_preserves_separate_lengths_and_heads(monkeypatch, layout, q_len, kv_len):
    monkeypatch.setattr(flash_attn, "current_omni_platform", SimpleNamespace(is_npu=lambda: True, device_name="npu"))
    monkeypatch.setattr(flash_attn, "get_current_diffusion_config_or_none", lambda: None)

    def quantize(tensor, **kwargs):
        return tensor, torch.ones_like(tensor, dtype=torch.uint8)

    monkeypatch.setattr(torch_npu, "npu_dynamic_mx_quant", quantize, raising=False)
    execute = Mock(side_effect=lambda q, k, v, **kwargs: (q, None))
    monkeypatch.setattr(torch_npu, "npu_fused_infer_attention_score_v2", execute, raising=False)
    q = torch.randn(2, q_len, 4, 64, dtype=torch.bfloat16)
    k = torch.randn(2, kv_len, 2, 64, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    if layout == "BNSD":
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    impl = flash_attn.FlashAttentionImpl(
        num_heads=4,
        head_size=64,
        softmax_scale=0.37,
        qkv_layout=layout,
        backend_kwargs={"quant": {"method": "mxfp8", "fallback": ["fp8"]}},
    )
    output = impl.forward_fa_quant_npu(q, k, v)
    rotation = get_quant_attention_rotation(q.device, q.dtype, 64, 425500)
    torch.testing.assert_close(output, q @ rotation)
    execute.assert_called_once()
    args, kwargs = execute.call_args
    assert args[0].shape == (2 * q_len, 4, 64)
    assert args[1].shape == args[2].shape == (2 * kv_len, 2, 64)
    assert kwargs["input_layout"] == "TND"
    assert kwargs["num_query_heads"] == 4 and kwargs["num_key_value_heads"] == 2
    assert kwargs["actual_seq_qlen"] == [q_len, 2 * q_len]
    assert kwargs["actual_seq_kvlen"] == [kv_len, 2 * kv_len]
