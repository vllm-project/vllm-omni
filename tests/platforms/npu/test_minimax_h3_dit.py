# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
import json
import runpy
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _write_eagle_kernel(kernels, soc, op, inputs, output="bfloat16", *, binary=True):
    descriptor = kernels / soc / op / "kernel.json"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text(json.dumps({"supportInfo": {"inputs": inputs, "outputs": [{"dtype": output}]}}))
    if binary:
        descriptor.with_suffix(".o").touch()
    path = descriptor.relative_to(kernels)
    return {"binaryList": [{"jsonPath": path.as_posix(), "binPath": path.with_suffix(".o").as_posix()}]}


@pytest.mark.parametrize(
    ("device", "soc", "qkv_binary", "swiglu_output", "registration", "expected"),
    [
        pytest.param("A2", "ascend910b", True, "bfloat16", "eager", (True, True), id="a2-both"),
        pytest.param("A2", "ascend910b", True, "float16", "startup", (True, False), id="a2-qkv-only"),
        pytest.param("A2", "ascend910b", False, "bfloat16", "startup", (False, True), id="a2-swiglu-only"),
        pytest.param("A2", "ascend910b", True, "bfloat16", "late", (False, False), id="a2-late-import"),
        pytest.param("A2", "ascend910b", True, "bfloat16", "unlisted", (True, False), id="swiglu-not-registered"),
        pytest.param("A5", "ascend950", True, "bfloat16", "startup", (True, False), id="a5-qkv-only"),
        pytest.param("A5", "ascend910b", True, "bfloat16", "startup", (False, False), id="wrong-soc"),
        pytest.param("A3", "ascend910b", True, "bfloat16", "startup", (False, False), id="unsupported-device"),
    ],
)
def test_h3_platform_installs_registered_fusions(
    tmp_path, monkeypatch, device, soc, qkv_binary, swiglu_output, registration, expected
):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention, MiniMaxH3MLP
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.platforms.npu.models import minimax_h3 as h3

    sdk = tmp_path / "mindiesd"
    vendor = sdk / "ops/vendors/aie_ascendc"
    kernels = vendor / "op_impl/ai_core/tbe/kernel"
    int8_inputs = [{"dtype": "int8"}, {"dtype": "int8", "format": "ND"}, {"dtype": "float32"}]
    index = {
        h3._QKV: _write_eagle_kernel(kernels, soc, h3._QKV, [{"dtype": "bfloat16"}], binary=qkv_binary),
        h3._SWIGLU: _write_eagle_kernel(kernels, soc, h3._SWIGLU, int8_inputs, swiglu_output),
    }
    config = kernels / "config" / soc
    config.mkdir(parents=True)
    (config / "binary_info_config.json").write_text(json.dumps(index))
    op_info = {h3._QKV: {}} if registration == "unlisted" else dict.fromkeys(index, {})
    config = kernels.parent / "config" / soc
    config.mkdir(parents=True)
    (config / f"aic-{soc}-ops-info.json").write_text(json.dumps(op_info))
    (sdk / "__init__.py").write_text(f"import os\nos.environ['ASCEND_CUSTOM_OPP_PATH'] = {str(vendor)!r}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "mindiesd", None)
    del sys.modules["mindiesd"]
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "" if registration in ("late", "eager") else str(vendor))

    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", SimpleNamespace(NPUPlatform=object))
    platform = runpy.run_module("vllm_omni.platforms.npu.platform")["NPUOmniPlatform"]
    platform.get_diffusion_attn_backend_cls("FLASH_ATTN" if registration == "eager" else "TORCH_SDPA", 128)
    if registration == "late":
        assert "mindiesd" not in sys.modules
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
    if registration == "startup":
        assert "mindiesd" not in sys.modules


def _qkv_inputs(*, head_dim=128, rot_dim=96, eps=1e-6):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention

    attention = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
    nn.Module.__init__(attention)
    attention.num_heads = attention.num_kv_heads = 2
    attention.head_dim = head_dim
    attention.rot_dim = rot_dim
    attention.vdn = None
    attention.qkv_proj = lambda x: (x, None)
    generator = torch.Generator().manual_seed(37)
    for name in ("q_norm", "k_norm"):
        norm = nn.RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16)
        norm.variance_epsilon = norm.eps
        setattr(attention, name, norm)
    packed = torch.randn(7, 3 * attention.num_heads * head_dim, generator=generator, dtype=torch.bfloat16)
    angles = torch.randn(7, rot_dim // 2, generator=generator)
    table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(packed.dtype)
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


def test_project_qkv_calls_eagle_with_local_rope_halves(eagle_qkv):
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
    assert eagle_qkv.call_args.kwargs == dict(norm_type=3, rope_mode=1, rotary_dim=96, epsilon=1e-6, output_bnsd=False)
    assert (actual_q, actual_k, actual_v) == outputs
    for actual, expected in ((raw_q, q), (raw_k, k)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(("rows", "use_eagle"), [(99_864, True), (99_865, False)])
def test_project_qkv_eagle_numel_limit(eagle_qkv, rows, use_eagle):
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


@pytest.mark.parametrize("has_rope", [True, False], ids=["missing-native", "no-rope"])
def test_project_qkv_preserves_fallback(eagle_qkv, has_rope):
    attention, packed, table = _qkv_inputs()
    table = table if has_rope else None
    expected = attention._project_qkv(packed, table)
    eagle_qkv.side_effect = ImportError("native QKV unavailable")
    _install_npu_qkv(attention)

    actual = attention._project_qkv(packed, table)

    attention.qkv_proj.assert_called_once_with(packed)
    assert eagle_qkv.call_count == int(has_rope)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _swiglu_inputs(mocker, quant_method):
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
    from vllm_omni.platforms.npu.models.minimax_h3 import forward_h3_mlp

    x = Mock(device=SimpleNamespace(type="npu"), shape=(2, 3, 4), dtype=torch.bfloat16)
    fc1 = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    nn.Module.__init__(fc1)
    fc1.quant_method = quant_method
    fc1.weight = mocker.sentinel.weight
    fc1.weight_scale = mocker.sentinel.weight_scale
    fc1.forward = Mock(return_value=(mocker.sentinel.projected, None))

    mlp = blocks.MiniMaxH3MLP.__new__(blocks.MiniMaxH3MLP)
    nn.Module.__init__(mlp)
    mlp.fc1 = fc1
    mlp.act_fn = Mock()
    mlp.fc2 = Mock(return_value=(mocker.sentinel.output, None))

    quantize = Mock(return_value=(mocker.sentinel.quantized, mocker.sentinel.scale))
    fused = Mock(return_value=torch.arange(30, dtype=x.dtype).reshape(6, 5))
    mocker.patch.dict(
        sys.modules,
        {
            "torch_npu": SimpleNamespace(npu_dynamic_quant=quantize),
            "mindiesd.fused_ops": SimpleNamespace(eagle_quant_batch_matmul_swiglu=fused),
        },
    )
    mlp.forward = MethodType(forward_h3_mlp, mlp)
    return mlp, x, quantize, fused


def test_mlp_calls_mindiesd_int8_swiglu(mocker):
    from vllm_omni.quantization.int8_config import DiffusionInt8Config, NPUInt8OnlineLinearMethod

    mlp, x, quantize, fused = _swiglu_inputs(mocker, NPUInt8OnlineLinearMethod(DiffusionInt8Config()))
    assert mlp(x) is mocker.sentinel.output

    x.reshape.assert_called_once_with(-1, 4)
    quantize.assert_called_once_with(x.reshape.return_value)
    fused.assert_called_once_with(
        mocker.sentinel.quantized,
        mlp.fc1.weight,
        mlp.fc1.weight_scale,
        pertoken_scale=mocker.sentinel.scale,
        output_dtype=x.dtype,
    )
    mlp.fc2.assert_called_once()
    torch.testing.assert_close(mlp.fc2.call_args.args[0], fused.return_value.reshape(2, 3, 5))
    mlp.fc1.forward.assert_not_called()
    mlp.act_fn.assert_not_called()


@pytest.mark.parametrize("case", ["missing_sdk", "missing_native", "non_int8", "wrapped"])
def test_mlp_preserves_swiglu_fallback(mocker, case):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_omni.quantization.int8_config import DiffusionInt8Config, NPUInt8LinearMethod

    method = NPUInt8LinearMethod(DiffusionInt8Config(is_checkpoint_int8_serialized=True))
    mlp, x, quantize, fused = _swiglu_inputs(mocker, method)
    if case == "missing_sdk":
        mocker.patch.dict(sys.modules, {"mindiesd.fused_ops": None})
    elif case == "missing_native":
        fused.side_effect = ImportError("native SwiGLU unavailable")
    elif case == "non_int8":
        mlp.fc1.quant_method = UnquantizedLinearMethod()
    elif case == "wrapped":
        wrapper = nn.Module()
        wrapper.base_layer = mlp.fc1
        wrapper.quant_method = mlp.fc1.quant_method
        wrapper.forward = Mock(wraps=wrapper.base_layer)
        mlp.fc1 = wrapper

    assert mlp(x) is mocker.sentinel.output

    expected_calls = 1 if case == "missing_native" else 0
    assert quantize.call_count == fused.call_count == expected_calls
    mlp.fc1.forward.assert_called_once_with(x)
    mlp.act_fn.assert_called_once_with(mocker.sentinel.projected)
    mlp.fc2.assert_called_once_with(mlp.act_fn.return_value)
