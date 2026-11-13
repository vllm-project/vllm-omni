# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.diffusion.models.minimax_h3 import vae_mxfp8
from vllm_omni.quantization.component_config import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "npu"])
def test_video_vae_mxfp8_rejects_unsupported_devices_without_import(monkeypatch, mocker, device_type):
    monkeypatch.setitem(sys.modules, "vllm_omni.quantization.mxfp8_config", None)
    is_a5 = mocker.patch.object(vae_mxfp8, "is_a5", return_value=False)
    config = mocker.Mock(spec=QuantizationConfig)
    config.get_name.return_value = "mxfp8"
    device = mocker.Mock(spec=torch.device, type=device_type)

    with pytest.raises(ValueError, match="requires an Ascend A5"):
        vae_mxfp8.resolve_h3_vae_mxfp8(ComponentQuantizationConfig({"video_vae": config}), device)
    assert is_a5.call_count == (1 if device_type == "npu" else 0)


def test_video_vae_mxfp8_only_uses_explicit_component(mocker):
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    mocker.patch.object(vae_mxfp8, "is_a5", return_value=True)
    device = mocker.Mock(spec=torch.device, type="npu")
    config = DiffusionMXFP8Config()

    assert vae_mxfp8.resolve_h3_vae_mxfp8(config, device) is None
    assert vae_mxfp8.resolve_h3_vae_mxfp8(ComponentQuantizationConfig({}, default_config=config), device) is None
    assert (
        vae_mxfp8.resolve_h3_vae_mxfp8(ComponentQuantizationConfig({"video_vae.decoder": config}), device) is None
    )
    assert vae_mxfp8.resolve_h3_vae_mxfp8(ComponentQuantizationConfig({"video_vae": config}), device) is config


@pytest.mark.parametrize(("batch", "length", "calls"), [(1, 195, 1), (2, 192, 1), (2, 195, 2)])
def test_mxfp8_fa_preserves_unaligned_tile_boundaries(mocker, batch, length, calls):
    q = torch.arange(batch * length * 64, dtype=torch.float32).reshape(batch, length, 1, 64)
    backend = mocker.Mock()
    backend.forward_fa_quant_npu.side_effect = lambda query, key, value, metadata: query + key + value

    result = vae_mxfp8._run_mxfp8_fa(backend, q, q + 1, q + 2)

    torch.testing.assert_close(result, 3 * q + 3)
    assert backend.forward_fa_quant_npu.call_count == calls
    for index, call in enumerate(backend.forward_fa_quant_npu.call_args_list):
        query, _, _, metadata = call.args
        torch.testing.assert_close(query, q if calls == 1 else q[index : index + 1])
        assert metadata.extra["kv_cache_dtype"] == "mxfp8"


@pytest.mark.parametrize("metadata_key", [None, "mask_mod", "cu_seqlens", "block_sparse"])
def test_mxfp8_fa_keeps_native_masked_or_packed_attention(mocker, metadata_key):
    class NativeAttention(nn.Module):
        def perform_attention(self, query, key, value, pack_info):
            return key

    attention = NativeAttention().eval()
    backend = mocker.Mock()
    attention._omni_mxfp8_fa = backend
    device = mocker.Mock(spec=torch.device, type="npu")
    query, key, value = [
        mocker.Mock(spec=torch.Tensor, device=device, shape=(1, 195, 1, 64), ndim=4, dtype=torch.float16)
        for _ in range(3)
    ]
    query.numel.return_value = 195 * 64
    pack_info = {} if metadata_key is None else {metadata_key: object()}

    result = vae_mxfp8._perform_mxfp8_attention_npu(attention, query, key, value, pack_info)

    if metadata_key is None:
        assert result is backend.forward_fa_quant_npu.return_value
        backend.forward_fa_quant_npu.assert_called_once()
    else:
        assert result is key
        backend.forward_fa_quant_npu.assert_not_called()
