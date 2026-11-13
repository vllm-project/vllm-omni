# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import weakref
from types import ModuleType

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.diffusion.models.minimax_h3 import vae_mxfp8
from vllm_omni.quantization.component_config import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("offload_flags", "quantized", "stage_on_cpu"),
    [
        ({"enable_layerwise_offload": True}, True, True),
        ({"enable_distributed_layerwise_offload": True}, True, True),
        ({"enable_cpu_offload": True}, True, False),
        ({}, True, False),
        ({"enable_layerwise_offload": True}, False, False),
    ],
)
def test_standalone_decoder_stages_mxfp8_for_layerwise_offload(
    tmp_path, mocker, offload_flags, quantized, stage_on_cpu
):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3_decoder as decoder
    from vllm_omni.diffusion.offloader import OffloadConfig, resolve_offload_plan
    from vllm_omni.quantization import build_quant_config

    config = OmniDiffusionConfig(
        model=str(tmp_path),
        task_type="fl2va",
        quantization_config=build_quant_config({"video_vae": {"method": "mxfp8"}}) if quantized else None,
        **offload_flags,
    )
    device = mocker.Mock(spec=torch.device, type="npu")
    mocker.patch.object(decoder, "get_local_device", return_value=device)
    video_vae = mocker.Mock(spec=decoder.MiniMaxH3VideoVAE)
    create_video_vae = mocker.patch.object(decoder, "MiniMaxH3VideoVAE", return_value=video_vae)
    create_audio_vae = mocker.patch.object(decoder, "MiniMaxH3AudioVAE", return_value=nn.Module())
    mocker.patch.object(decoder.MiniMaxH3DecoderPipeline, "setup_diffusion_pipeline_profiler")

    pipeline = decoder.MiniMaxH3DecoderPipeline(od_config=config)

    video_args = create_video_vae.call_args.kwargs
    assert video_args["device"] is device
    assert (video_args.get("load_device") or video_args["device"]) == (torch.device("cpu") if stage_on_cpu else device)
    assert video_args["quantization_config"] is config.quantization_config
    assert create_audio_vae.call_args.kwargs["device"] is device
    plan = resolve_offload_plan(pipeline, OffloadConfig.from_od_config(config))
    assert [component.path for component in plan.vaes if component.on_demand] == (["video_vae"] if stage_on_cpu else [])


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


def test_video_vae_mxfp8_only_uses_explicit_component(monkeypatch, mocker):
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    mocker.patch.object(vae_mxfp8, "is_a5", return_value=True)
    device = mocker.Mock(spec=torch.device, type="npu")
    config = DiffusionMXFP8Config()
    mindiesd = ModuleType("mindiesd")
    monkeypatch.setitem(sys.modules, "mindiesd", mindiesd)

    assert vae_mxfp8.resolve_h3_vae_mxfp8(config, device) is None
    assert vae_mxfp8.resolve_h3_vae_mxfp8(ComponentQuantizationConfig({}, default_config=config), device) is None
    assert vae_mxfp8.resolve_h3_vae_mxfp8(ComponentQuantizationConfig({"video_vae.decoder": None}), device) is None
    assert (
        vae_mxfp8.resolve_h3_vae_mxfp8(ComponentQuantizationConfig({"video_vae": None}, default_config=config), device)
        is None
    )
    monkeypatch.setattr(mindiesd, "quant_attention", mocker.Mock(), raising=False)
    assert vae_mxfp8.resolve_h3_vae_mxfp8(ComponentQuantizationConfig({"video_vae": config}), device) is config


@pytest.mark.parametrize(
    "exports",
    [None, {}, {"quant_attention": None}],
    ids=["missing-package", "missing-symbol", "non-callable-symbol"],
)
def test_video_vae_mxfp8_rejects_unavailable_dense_attention_at_init(monkeypatch, mocker, exports):
    from vllm_omni.diffusion.models.minimax_h3 import vae
    from vllm_omni.quantization import build_quant_config

    mindiesd = None
    if exports is not None:
        mindiesd = ModuleType("mindiesd")
        vars(mindiesd).update(exports)
    monkeypatch.setitem(sys.modules, "mindiesd", mindiesd)
    mocker.patch.object(vae_mxfp8, "is_a5", return_value=True)
    read_checkpoint = mocker.patch.object(
        vae, "_load_component_config", side_effect=AssertionError("checkpoint configuration read reached")
    )
    config = build_quant_config({"video_vae": {"method": "mxfp8"}})
    device = mocker.Mock(spec=torch.device, type="npu")

    with pytest.raises(ImportError, match=r"MindIE-SD.*mindiesd\.quant_attention.*video_vae to null"):
        vae.MiniMaxH3VideoVAE("unused-model-dir", device=device, quantization_config=config)
    read_checkpoint.assert_not_called()


@pytest.mark.parametrize("parent", [{}, {"video_vae": None}], ids=["missing-parent", "null-parent"])
def test_video_vae_rejects_non_null_child_without_enabled_parent(parent):
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        {
            **parent,
            "video_vae.remote.model.decoder.transformer_blocks.0": {"method": "mxfp8"},
        }
    )

    with pytest.raises(ValueError, match="video_vae does not support quantization via child overrides"):
        vae_mxfp8.resolve_h3_vae_mxfp8(config, torch.device("cpu"))


@pytest.mark.parametrize("override", [None, {"method": "mxfp8"}])
def test_video_vae_mxfp8_rejects_child_overrides(mocker, override):
    from vllm_omni.quantization import build_quant_config

    mocker.patch.object(vae_mxfp8, "is_a5", return_value=True)
    device = mocker.Mock(spec=torch.device, type="npu")
    config = build_quant_config(
        {
            "video_vae": {"method": "mxfp8"},
            "video_vae.remote.model.decoder.transformer_blocks.0": override,
        }
    )

    with pytest.raises(ValueError, match="video_vae MXFP8 does not support child overrides"):
        vae_mxfp8.resolve_h3_vae_mxfp8(config, device)


def test_video_vae_rejects_explicit_fp8():
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config({"video_vae": {"method": "fp8"}})
    with pytest.raises(ValueError, match="video_vae.*fp8"):
        vae_mxfp8.resolve_h3_vae_mxfp8(config, torch.device("cpu"))


def test_video_vae_rejects_incompatible_mxfp8_config(mocker):
    mocker.patch.object(vae_mxfp8, "is_a5", return_value=True)
    config = mocker.Mock(spec=QuantizationConfig)
    config.get_name.return_value = "mxfp8"
    device = mocker.Mock(spec=torch.device, type="npu")

    with pytest.raises(ValueError, match="video_vae MXFP8 requires DiffusionMXFP8Config"):
        vae_mxfp8.resolve_h3_vae_mxfp8(ComponentQuantizationConfig({"video_vae": config}), device)


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
    backend.forward_fa_quant_npu.return_value = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 1.0], dtype=torch.float16
    )
    device = mocker.Mock(spec=torch.device, type="npu")
    query, key, value = [
        mocker.Mock(spec=torch.Tensor, device=device, shape=(1, 195, 1, 64), ndim=4, dtype=torch.float16)
        for _ in range(3)
    ]
    query.numel.return_value = 195 * 64
    pack_info = {} if metadata_key is None else {metadata_key: object()}

    result = vae_mxfp8._perform_mxfp8_attention_npu(attention, query, key, value, pack_info)

    if metadata_key is None:
        finite_limit = torch.finfo(torch.float16).max
        expected = torch.tensor([0.0, finite_limit, -finite_limit, 1.0], dtype=torch.float16)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        backend.forward_fa_quant_npu.assert_called_once()
    else:
        assert result is key
        backend.forward_fa_quant_npu.assert_not_called()


def test_video_vae_mxfp8_releases_replaced_linear_before_next_conversion(monkeypatch, mocker):
    from vllm_omni.diffusion.attention.backends import flash_attn
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    class NativeAttention(nn.Module):
        def perform_attention(self, query, key, value, pack_info):
            return query

    block = nn.Module()
    block.attn = NativeAttention()
    block.attn.heads = 1
    block.attn.dim_head = 32
    block.attn.to_qkv = nn.Linear(32, 96)
    block.attn.to_out = nn.Linear(32, 32)
    block.ff = nn.Module()
    block.ff.w1 = nn.Linear(32, 64)
    block.ff.w2 = nn.Linear(64, 32)
    decoder = nn.Module()
    decoder.transformer_blocks = nn.ModuleList([block])
    original_qkv = weakref.ref(block.attn.to_qkv)

    def quantize(linear, _config, _device, _load_device):
        if linear is block.attn.to_out:
            assert original_qkv() is None
        return nn.Identity()

    mocker.patch.object(flash_attn, "FlashAttentionImpl")
    monkeypatch.setattr(vae_mxfp8, "_NPUMxfp8VideoVAELinear", quantize)
    cpu = torch.device("cpu")

    vae_mxfp8.install_h3_vae_mxfp8(decoder, DiffusionMXFP8Config(), device=cpu, load_device=cpu)

    assert original_qkv() is None
    assert isinstance(block.attn.to_qkv, nn.Identity)
