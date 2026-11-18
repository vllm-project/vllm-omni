# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from transformers import T5Config, T5EncoderModel

from vllm_omni.diffusion.models.sd3.pipeline_sd3 import _get_t5_quantization_config
from vllm_omni.quantization import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("spec", [None, "fp8", {"transformer": {"method": "fp8"}}])
def test_t5_quantization_requires_component_config(spec):
    assert _get_t5_quantization_config(build_quant_config(spec)) is None


@pytest.mark.parametrize(
    "spec",
    [
        {"method": "fp8"},
        {"method": "int8", "is_checkpoint_int8_serialized": True},
    ],
)
def test_t5_rejects_unsupported_quantization(spec):
    config = build_quant_config({"text_encoder_3": spec})
    with pytest.raises(ValueError, match="text_encoder_3"):
        _get_t5_quantization_config(config)


def test_t5_rejects_cpu_offload():
    config = build_quant_config({"text_encoder_3": {"method": "int8"}})
    with pytest.raises(ValueError, match="CPU offload"):
        _get_t5_quantization_config(config, cpu_offload=True)


def test_t5_quantizes_during_cpu_checkpoint_load(tmp_path):
    pytest.importorskip("torchao")
    from torchao.dtypes import AffineQuantizedTensor

    torch.manual_seed(42)
    model = (
        T5EncoderModel(
            T5Config(vocab_size=32, d_model=64, d_kv=16, d_ff=128, num_layers=1, num_heads=4, dropout_rate=0.0)
        )
        .to(torch.bfloat16)
        .eval()
    )
    model.save_pretrained(tmp_path)
    config = build_quant_config({"transformer": {"method": "fp8"}, "text_encoder_3": {"method": "int8"}})
    matmul_precision = torch.get_float32_matmul_precision()
    quantized = T5EncoderModel.from_pretrained(
        tmp_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        quantization_config=_get_t5_quantization_config(config),
    ).eval()
    assert torch.get_float32_matmul_precision() == matmul_precision

    weight = quantized.encoder.block[0].layer[0].SelfAttention.q.weight
    assert isinstance(weight, AffineQuantizedTensor)
    assert weight.tensor_impl.get_plain()[0].dtype == torch.int8
    assert weight.device.type == "cpu"
    assert not isinstance(quantized.shared.weight, AffineQuantizedTensor)
    tokens = torch.tensor([[0, 1, 2, 3]])
    with torch.inference_mode():
        reference = model(tokens).last_hidden_state
        actual = quantized(tokens).last_hidden_state
    torch.testing.assert_close(actual, reference, rtol=0.05, atol=0.05)
