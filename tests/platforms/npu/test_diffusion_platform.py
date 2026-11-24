# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import json
import runpy
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("available", [False, True])
def test_dense_flash_capability_tracks_mindiesd(monkeypatch, available):
    pytest.importorskip("vllm_ascend")
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "mindiesd" and available else None,
    )

    assert NPUOmniPlatform.supports_diffusion_dense_flash_attention() is available


def test_paged_config_uses_ascend_kernel_block_size() -> None:
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

    from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    NPUOmniPlatform.configure_diffusion_vllm_config(
        vllm_config,
        SimpleNamespace(diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER),
    )

    assert vllm_config.cache_config.block_size == AscendAttentionBackend.get_supported_kernel_block_sizes()[0]


def test_strict_ulysses_paged_backend_bypasses_ascend_pcp() -> None:
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackend,
        AscendAttentionBackendImpl,
        AscendAttentionMetadataBuilder,
    )

    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    assert (
        NPUOmniPlatform.get_diffusion_paged_kv_attn_backend(
            AscendAttentionBackend,
            ulysses_degree=1,
        )
        is AscendAttentionBackend
    )
    backend = NPUOmniPlatform.get_diffusion_paged_kv_attn_backend(
        AscendAttentionBackend,
        ulysses_degree=2,
    )
    assert backend.get_impl_cls() is AscendAttentionBackendImpl
    assert backend.get_builder_cls() is AscendAttentionMetadataBuilder


def _write_eagle_kernel(kernels, soc, op, inputs, *, binary=True):
    descriptor = kernels / soc / op / "kernel.json"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text(json.dumps({"supportInfo": {"inputs": inputs, "outputs": [{"dtype": "bfloat16"}]}}))
    if binary:
        descriptor.with_suffix(".o").touch()
    path = descriptor.relative_to(kernels)
    return {"binaryList": [{"jsonPath": path.as_posix(), "binPath": path.with_suffix(".o").as_posix()}]}


def _qkv_inputs(*, head_dim=128, rot_dim=96, eps=1e-6):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention

    attention = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
    nn.Module.__init__(attention)
    attention.num_heads = attention.num_kv_heads = 2
    attention.head_dim = head_dim
    attention.rot_dim = rot_dim
    attention.vdn = None
    attention.qkv_proj = lambda x: (x, None)
    for name in ("q_norm", "k_norm"):
        norm = nn.RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16)
        norm.variance_epsilon = norm.eps
        setattr(attention, name, norm)
    packed = torch.randn(7, 3 * attention.num_heads * head_dim, dtype=torch.bfloat16)
    table = torch.arange(7 * rot_dim).reshape(7, rot_dim).to(packed.dtype)
    return attention, packed, table


def _install_npu_qkv(attention):
    from vllm_omni.platforms.npu.models.minimax_h3 import _npu_project_qkv

    def project(x):
        return Mock(wraps=x, shape=x.shape, device=SimpleNamespace(type="npu")), None

    attention.qkv_proj = Mock(side_effect=project)
    attention._project_qkv = MethodType(_npu_project_qkv, attention)


@pytest.fixture
def eagle_qkv(monkeypatch):
    op = Mock()
    monkeypatch.setitem(sys.modules, "mindiesd.fused_ops", SimpleNamespace(eagle_split_norm_rope_concat=op))
    return op


@pytest.mark.diffusion
class TestMiniMaxH3Fusions:
    @pytest.mark.parametrize(
        ("device", "soc", "registered", "late", "expected"),
        [
            ("A2", "ascend910b", "qkv", False, (True, False)),
            ("A2", "ascend910b", "swiglu", False, (False, True)),
            ("A2", "ascend910b", "both", True, (False, False)),
            ("A5", "ascend950", "both", False, (True, False)),
            ("A5", "ascend910b", "both", False, (False, False)),
        ],
    )
    def test_h3_platform_installs_registered_fusions(
        self, tmp_path, monkeypatch, device, soc, registered, late, expected
    ):
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention, MiniMaxH3MLP
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
        from vllm_omni.platforms.npu.models import minimax_h3 as h3

        sdk = tmp_path / "mindiesd"
        vendor = sdk / "ops/vendors/aie_ascendc"
        kernels = vendor / "op_impl/ai_core/tbe/kernel"
        int8_inputs = [{"dtype": "int8"}, {"dtype": "int8", "format": "ND"}, {"dtype": "float32"}]
        index = {
            h3._QKV: _write_eagle_kernel(kernels, soc, h3._QKV, [{"dtype": "bfloat16"}], binary=registered != "swiglu"),
            h3._SWIGLU: _write_eagle_kernel(kernels, soc, h3._SWIGLU, int8_inputs),
        }
        config = kernels / "config" / soc
        config.mkdir(parents=True)
        (config / "binary_info_config.json").write_text(json.dumps(index))
        op_info = {h3._QKV: {}} if registered == "qkv" else dict.fromkeys(index, {})
        config = kernels.parent / "config" / soc
        config.mkdir(parents=True)
        (config / f"aic-{soc}-ops-info.json").write_text(json.dumps(op_info))
        (sdk / "__init__.py").write_text(f"import os\nos.environ['ASCEND_CUSTOM_OPP_PATH'] = {str(vendor)!r}\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setitem(sys.modules, "mindiesd", None)
        del sys.modules["mindiesd"]
        monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "")

        monkeypatch.setitem(sys.modules, "vllm_ascend.platform", SimpleNamespace(NPUPlatform=object))
        platform = runpy.run_module("vllm_omni.platforms.npu.platform")["NPUOmniPlatform"]
        platform.get_diffusion_attn_backend_cls("TORCH_SDPA" if late else "FLASH_ATTN", 128)
        if late:
            importlib.import_module("mindiesd")
            platform.get_diffusion_attn_backend_cls("FLASH_ATTN", 128)
        utils = SimpleNamespace(get_ascend_device_type=lambda: SimpleNamespace(name=device))
        monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        nn.Module.__init__(pipeline)
        attention, _, _ = _qkv_inputs()
        unsupported = [_qkv_inputs(**config)[0] for config in ({"head_dim": 96}, {"rot_dim": 48}, {"eps": 0.0})]
        mismatched_eps, _, _ = _qkv_inputs()
        mismatched_eps.k_norm.variance_epsilon = 1e-5
        mlp = MiniMaxH3MLP.__new__(MiniMaxH3MLP)
        nn.Module.__init__(mlp)
        pipeline.transformer = nn.ModuleList([attention, mlp, *unsupported, mismatched_eps])

        platform.apply_diffusion_model_patches(pipeline, SimpleNamespace(model_class_name="MiniMaxH3Pipeline"))

        use_qkv, use_swiglu = expected
        assert attention._project_qkv.__func__ is (h3._npu_project_qkv if use_qkv else MiniMaxH3Attention._project_qkv)
        for block in (*unsupported, mismatched_eps):
            assert block._project_qkv.__func__ is MiniMaxH3Attention._project_qkv
        assert mlp.forward.__func__ is (h3.forward_h3_mlp if use_swiglu else MiniMaxH3MLP.forward)

    def test_project_qkv_calls_eagle_with_local_rope_halves(self, eagle_qkv):
        attention, packed, table = _qkv_inputs()
        attention.vdn = nn.Identity()
        q, k, _ = packed.view(7, 6, 128).split(2, dim=1)
        outputs = (object(), object(), object())
        eagle_qkv.return_value = outputs
        _install_npu_qkv(attention)

        actual_q, actual_k, actual_v, raw_q, raw_k = attention._project_qkv(packed, table, vdn_window=object())

        eagle_qkv.assert_called_once()
        qkv, qw, kw, cos, sin, heads = eagle_qkv.call_args.args
        assert qkv.is_contiguous() and list(heads) == [2, 2, 2]
        torch.testing.assert_close(qkv, packed.view(7, 6, 128), atol=0, rtol=0)
        assert qw is attention.q_norm.weight and kw is attention.k_norm.weight
        assert cos.is_contiguous() and sin.is_contiguous()
        torch.testing.assert_close(cos, table[:, :48], atol=0, rtol=0)
        torch.testing.assert_close(sin, table[:, 48:], atol=0, rtol=0)
        assert eagle_qkv.call_args.kwargs == dict(
            norm_type=3, rope_mode=1, rotary_dim=96, epsilon=1e-6, output_bnsd=False
        )
        assert (actual_q, actual_k, actual_v) == outputs
        for actual, expected in ((raw_q, q), (raw_k, k)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize(("rows", "use_eagle"), [(99_864, True), (99_865, False)])
    def test_project_qkv_eagle_numel_limit(self, eagle_qkv, rows, use_eagle):
        attention, _, _ = _qkv_inputs()
        attention.to("meta")
        attention.num_heads = attention.num_kv_heads = 56
        packed = torch.empty(rows, 3 * attention.num_heads * attention.head_dim, device="meta", dtype=torch.bfloat16)
        table = packed.new_empty((rows, attention.rot_dim))
        eagle_qkv.return_value = (None, None, None)
        _install_npu_qkv(attention)

        q, *_ = attention._project_qkv(packed, table)

        assert eagle_qkv.called == use_eagle
        if not use_eagle:
            assert q.shape == (rows, attention.num_heads, attention.head_dim)

    def test_qkv_missing_native_uses_projected_values(self, eagle_qkv):
        attention, packed, table = _qkv_inputs()
        expected = attention._project_qkv(packed, table)
        eagle_qkv.side_effect = ImportError("native QKV unavailable")
        _install_npu_qkv(attention)

        actual = attention._project_qkv(packed, table)

        attention.qkv_proj.assert_called_once_with(packed)
        eagle_qkv.assert_called_once()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_int8_swiglu_feeds_fc2(self, monkeypatch):
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        from vllm_omni.platforms.npu.models.minimax_h3 import forward_h3_mlp
        from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

        fc1 = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
        nn.Module.__init__(fc1)
        fc1.quant_method = Mock(spec=NPUInt8LinearMethod)
        fc1.weight, fc1.weight_scale = object(), object()
        hidden = torch.arange(30).reshape(6, 5)
        quantize = Mock(return_value=(object(), object()))
        eagle = Mock(return_value=hidden)
        monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_dynamic_quant=quantize))
        monkeypatch.setitem(sys.modules, "mindiesd.fused_ops", SimpleNamespace(eagle_quant_batch_matmul_swiglu=eagle))
        x = Mock(shape=(2, 3, 4), device=SimpleNamespace(type="npu"), dtype=torch.bfloat16)
        mlp = SimpleNamespace(fc1=fc1, fc2=Mock(return_value=(hidden, None)))

        assert forward_h3_mlp(mlp, x) is hidden
        quantized, scale = quantize.return_value
        eagle.assert_called_once_with(
            quantized, fc1.weight, fc1.weight_scale, pertoken_scale=scale, output_dtype=x.dtype
        )
        torch.testing.assert_close(mlp.fc2.call_args.args[0], hidden.reshape(2, 3, 5))

    def test_swiglu_preserves_wrapped_fc1(self):
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3MLP
        from vllm_omni.platforms.npu.models.minimax_h3 import forward_h3_mlp
        from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

        mlp = MiniMaxH3MLP.__new__(MiniMaxH3MLP)
        nn.Module.__init__(mlp)
        mlp.fc1 = Mock(return_value=(torch.ones(2, 4), None), quant_method=Mock(spec=NPUInt8LinearMethod))
        mlp.act_fn = nn.Identity()
        mlp.fc2 = Mock(return_value=(torch.ones(2, 2), None))
        x = Mock(device=SimpleNamespace(type="npu"), dtype=torch.bfloat16)

        assert forward_h3_mlp(mlp, x) is mlp.fc2.return_value[0]
        mlp.fc1.assert_called_once_with(x)
