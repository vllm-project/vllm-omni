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


def _write_eagle_kernel(kernels, soc, op, *, binary=True):
    path = kernels / soc / op / "kernel.o"
    if binary:
        path.parent.mkdir(parents=True)
        path.touch()
    return {"binaryList": [{"binPath": path.relative_to(kernels).as_posix()}]}


def _qkv_inputs():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention

    attention = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
    nn.Module.__init__(attention)
    attention.num_heads = attention.num_kv_heads = 2
    attention.head_dim = 128
    attention.rot_dim = 96
    attention.vdn = None
    attention.qkv_proj = lambda x: (x, None)
    for name in ("q_norm", "k_norm"):
        norm = nn.RMSNorm(attention.head_dim, eps=1e-5, dtype=torch.bfloat16)
        norm.variance_epsilon = norm.eps
        setattr(attention, name, norm)
    packed = torch.randn(7, 3 * attention.num_heads * attention.head_dim, dtype=torch.bfloat16)
    table = torch.arange(7 * attention.rot_dim).reshape(7, attention.rot_dim).to(packed.dtype)
    return attention, packed, table


@pytest.fixture(scope="module")
def h3():
    with pytest.MonkeyPatch.context() as patch:
        if importlib.util.find_spec("torch_npu") is None:
            patch.setitem(sys.modules, "torch_npu", SimpleNamespace())
        return importlib.import_module("vllm_omni.platforms.npu.models.minimax_h3")


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
            ("A2", "ascend910b", "both", False, (True, True)),
            ("A2", "ascend910b", "qkv", False, (True, False)),
            ("A2", "ascend910b", "swiglu", False, (False, True)),
            ("A2", "ascend910b", "both", True, (False, False)),
            ("A5", "ascend950", "both", False, (True, False)),
            ("A5", "ascend910b", "both", False, (False, False)),
        ],
    )
    def test_h3_platform_installs_registered_fusions(
        self, tmp_path, monkeypatch, h3, device, soc, registered, late, expected
    ):
        from vllm_omni.diffusion.models.minimax_h3.controlnet import MiniMaxH3ControlBlock
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3DiTBlock, MiniMaxH3MLP
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3TokenRefinerBlock
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

        sdk = tmp_path / "mindiesd"
        vendor = sdk / "ops/vendors/aie_ascendc"
        kernels = vendor / "op_impl/ai_core/tbe/kernel"
        index = {
            h3._QKV: _write_eagle_kernel(kernels, soc, h3._QKV, binary=registered != "swiglu"),
            h3._SWIGLU: _write_eagle_kernel(kernels, soc, h3._SWIGLU),
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

        def make_block(cls):
            block = cls.__new__(cls)
            nn.Module.__init__(block)
            block.attn, _, _ = _qkv_inputs()
            block.mlp = MiniMaxH3MLP.__new__(MiniMaxH3MLP)
            nn.Module.__init__(block.mlp)
            return block

        main, short_rope, zero_eps, unequal_eps = [make_block(MiniMaxH3DiTBlock) for _ in range(4)]
        short_rope.attn.rot_dim = 48
        zero_eps.attn.q_norm.variance_epsilon = zero_eps.attn.k_norm.variance_epsilon = 0
        unequal_eps.attn.k_norm.variance_epsilon = 1e-4
        refiner = make_block(MiniMaxH3TokenRefinerBlock)
        control = make_block(MiniMaxH3ControlBlock)
        pipeline.transformer = nn.Module()
        pipeline.transformer.blocks = nn.ModuleList([main, short_rope, zero_eps, unequal_eps])
        pipeline.transformer.token_refiner = nn.Module()
        pipeline.transformer.token_refiner.blocks = nn.ModuleList([refiner])
        pipeline.transformer.controlnet = nn.Module()
        pipeline.transformer.controlnet.control_blocks = nn.ModuleList([control])

        platform.apply_diffusion_model_patches(pipeline, SimpleNamespace(model_class_name="MiniMaxH3Pipeline"))

        use_qkv, use_swiglu = expected
        for block in (main, control):
            assert (block.attn._project_qkv.__func__ is h3._npu_project_qkv) == use_qkv
        for block in (short_rope, zero_eps, unequal_eps, refiner):
            assert "_project_qkv" not in vars(block.attn)
        for block in (main, refiner):
            assert block.mlp.forward.__func__ is (h3.forward_h3_mlp if use_swiglu else MiniMaxH3MLP.forward)

    def test_project_qkv_calls_eagle_with_local_rope_halves(self, h3, eagle_qkv):
        attention, packed, table = _qkv_inputs()
        attention.vdn = nn.Identity()
        q, k, _ = packed.view(7, 6, 128).split(2, dim=1)
        outputs = (object(), object(), object())
        eagle_qkv.return_value = outputs
        attention._project_qkv = MethodType(h3._npu_project_qkv, attention)

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
            norm_type=3, rope_mode=1, rotary_dim=96, epsilon=1e-5, output_bnsd=False
        )
        assert (actual_q, actual_k, actual_v) == outputs
        for actual, expected in ((raw_q, q), (raw_k, k)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize(("rows", "use_eagle"), [(99_864, True), (99_865, False)])
    def test_project_qkv_eagle_numel_limit(self, h3, eagle_qkv, rows, use_eagle):
        attention, _, _ = _qkv_inputs()
        attention.to("meta")
        attention.num_heads = attention.num_kv_heads = 56
        packed = torch.empty(rows, 3 * attention.num_heads * attention.head_dim, device="meta", dtype=torch.bfloat16)
        table = packed.new_empty((rows, attention.rot_dim))
        eagle_qkv.return_value = (None, None, None)
        attention._project_qkv = MethodType(h3._npu_project_qkv, attention)

        q, *_ = attention._project_qkv(packed, table)

        assert eagle_qkv.called == use_eagle
        if not use_eagle:
            assert q.shape == (rows, attention.num_heads, attention.head_dim)

    def test_qkv_missing_native_uses_projected_values(self, h3, eagle_qkv):
        attention, packed, table = _qkv_inputs()
        expected = attention._project_qkv(packed, table)
        attention.qkv_proj = Mock(wraps=attention.qkv_proj)
        eagle_qkv.side_effect = ImportError("native QKV unavailable")
        attention._project_qkv = MethodType(h3._npu_project_qkv, attention)

        actual = attention._project_qkv(packed, table)

        attention.qkv_proj.assert_called_once_with(packed)
        eagle_qkv.assert_called_once()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_int8_swiglu_feeds_fc2(self, monkeypatch, h3):
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

        fc1 = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
        nn.Module.__init__(fc1)
        fc1.quant_method = Mock(spec=NPUInt8LinearMethod)
        fc1.weight, fc1.weight_scale = object(), object()
        hidden = torch.arange(30).reshape(6, 5)
        quantize = Mock(return_value=(object(), object()))
        eagle = Mock(return_value=hidden)
        monkeypatch.setattr(h3, "torch_npu", SimpleNamespace(npu_dynamic_quant=quantize))
        monkeypatch.setitem(sys.modules, "mindiesd.fused_ops", SimpleNamespace(eagle_quant_batch_matmul_swiglu=eagle))
        x = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
        mlp = SimpleNamespace(fc1=fc1, fc2=Mock(return_value=(hidden, None)))

        assert h3.forward_h3_mlp(mlp, x) is hidden
        quantized, scale = quantize.return_value
        eagle.assert_called_once_with(
            quantized, fc1.weight, fc1.weight_scale, pertoken_scale=scale, output_dtype=x.dtype
        )
        torch.testing.assert_close(mlp.fc2.call_args.args[0], hidden.reshape(2, 3, 5))
