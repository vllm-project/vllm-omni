# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import json
import runpy
import sys
from types import SimpleNamespace
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


@pytest.fixture(scope="module")
def h3():
    with pytest.MonkeyPatch.context() as patch:
        if importlib.util.find_spec("torch_npu") is None:
            patch.setitem(sys.modules, "torch_npu", SimpleNamespace())
        return importlib.import_module("vllm_omni.platforms.npu.models.minimax_h3")


@pytest.fixture
def eagle_qkv(monkeypatch):
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fused

    op = Mock()
    monkeypatch.setitem(sys.modules, "mindiesd.fused_ops", SimpleNamespace(eagle_split_norm_rope_concat=op))
    monkeypatch.setattr(fused, "_fused_npu_supported", lambda *args: True)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_rms_norm=lambda x, weight, epsilon: (x,), npu_rotary_mul=lambda x, cos, sin: x),
    )
    return op


@pytest.mark.diffusion
class TestEagleSplitNormRopeConcat:
    @pytest.mark.parametrize(
        ("device", "soc", "kernel", "late", "expected"),
        [
            ("A2", "ascend910b", "bfloat16", False, True),
            ("A2", "ascend910b", "float16", False, False),
            ("A2", "ascend910b", None, False, False),
            ("A2", "ascend910b", "bfloat16", True, False),
            ("A5", "ascend950", "bfloat16", False, True),
            ("A5", "ascend910b", "bfloat16", False, False),
        ],
    )
    def test_h3_platform_enables_registered_qkv(self, tmp_path, monkeypatch, h3, device, soc, kernel, late, expected):
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

        sdk = tmp_path / "mindiesd"
        vendor = sdk / "ops/vendors/aie_ascendc"
        tbe = vendor / "op_impl/ai_core/tbe"
        kernels = tbe / "kernel"
        config = kernels / "config" / soc
        config.mkdir(parents=True)
        entry = {"binPath": "qkv.o", "jsonPath": "qkv.json"}
        if kernel is not None:
            (kernels / "qkv.o").touch()
            support = {"inputs": [{"dtype": kernel}], "outputs": [{"dtype": kernel}]}
            (kernels / "qkv.json").write_text(json.dumps({"supportInfo": support}))
        (config / "binary_info_config.json").write_text(json.dumps({h3._QKV: {"binaryList": [entry]}}))
        config = tbe / "config" / soc
        config.mkdir(parents=True)
        (config / f"aic-{soc}-ops-info.json").write_text(json.dumps({h3._QKV: {}}))
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
        od_config = SimpleNamespace(model_class_name="MiniMaxH3Pipeline")

        platform.apply_diffusion_model_patches(pipeline, od_config)

        assert od_config._minimax_h3_eagle_qkv is expected

    def test_qkv_reuses_k_for_value_and_packs_rope(self, eagle_qkv):
        from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
        from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

        q = torch.randn(7, 4, 128, dtype=torch.bfloat16)
        k = torch.randn(7, 2, 128, dtype=q.dtype)
        qw, kw = torch.randn(2, 128, dtype=q.dtype).unbind()
        table = torch.arange(7 * 96).reshape(7, 96).to(q.dtype)
        outputs = (object(), object(), object())
        eagle_qkv.return_value = outputs
        ctx = ForwardContext(omni_diffusion_config=SimpleNamespace(_minimax_h3_eagle_qkv=True))
        with override_forward_context(ctx):
            assert fused_qk_norm_rope(q, k, qw, kw, table, 1e-5) == outputs[:2]

        eagle_qkv.assert_called_once()
        packed, q_weight, k_weight, cos, sin, heads = eagle_qkv.call_args.args
        assert packed.is_contiguous() and list(heads) == [4, 2, 2]
        torch.testing.assert_close(packed, torch.cat((q, k, k), dim=1), atol=0, rtol=0)
        assert qw is q_weight and kw is k_weight
        assert cos.is_contiguous() and sin.is_contiguous()
        torch.testing.assert_close(cos, table[:, :48], atol=0, rtol=0)
        torch.testing.assert_close(sin, table[:, 48:], atol=0, rtol=0)
        assert eagle_qkv.call_args.kwargs == dict(
            norm_type=3, rope_mode=1, rotary_dim=96, epsilon=1e-5, output_bnsd=False
        )

    @pytest.mark.parametrize("mode", ["other_pipeline", "fp32_rope", "missing_native"])
    def test_qkv_falls_back_to_npu_primitives(self, eagle_qkv, mode):
        from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
        from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

        q, k = torch.randn(2, 7, 2, 128, dtype=torch.bfloat16).unbind()
        weight = torch.ones(128, dtype=q.dtype)
        table = torch.ones(7, 96, dtype=torch.float32 if mode == "fp32_rope" else q.dtype)
        config = SimpleNamespace() if mode == "other_pipeline" else SimpleNamespace(_minimax_h3_eagle_qkv=True)
        eagle_qkv.side_effect = ImportError("native QKV unavailable")

        with override_forward_context(ForwardContext(omni_diffusion_config=config)):
            actual_q, actual_k = fused_qk_norm_rope(q, k, weight, weight, table, 1e-5)

        assert eagle_qkv.called == (mode == "missing_native")
        torch.testing.assert_close(actual_q, q, atol=0, rtol=0)
        torch.testing.assert_close(actual_k, k, atol=0, rtol=0)

    @pytest.mark.parametrize(("rows", "use_eagle"), [(99_864, True), (99_865, False)])
    def test_packed_qkv_numel_limit(self, eagle_qkv, rows, use_eagle):
        from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
        from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

        q = torch.empty(rows, 56, 128, device="meta", dtype=torch.bfloat16)
        k = torch.empty_like(q)
        weight = q.new_empty((128,))
        table = q.new_empty((rows, 96))
        eagle_qkv.return_value = (q, k, None)

        ctx = ForwardContext(omni_diffusion_config=SimpleNamespace(_minimax_h3_eagle_qkv=True))
        with override_forward_context(ctx):
            actual_q, actual_k = fused_qk_norm_rope(q, k, weight, weight, table, 1e-5)

        assert eagle_qkv.called == use_eagle
        assert actual_q.shape == q.shape and actual_k.shape == k.shape
