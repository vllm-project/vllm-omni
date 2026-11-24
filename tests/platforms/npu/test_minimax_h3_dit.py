# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("soc", "dtype", "registered", "binary", "expected"),
    [
        pytest.param("ascend950", "bfloat16", True, True, True, id="active-950-bf16"),
        pytest.param("ascend910b", "bfloat16", True, True, False, id="910b-only"),
        pytest.param("ascend950", "float16", True, True, False, id="fp16-only"),
        pytest.param("ascend950", "bfloat16", True, False, False, id="missing-binary"),
        pytest.param("ascend950", "bfloat16", False, True, False, id="inactive-vendor"),
    ],
)
def test_a5_qkv_requires_registered_bf16_kernel(tmp_path, monkeypatch, soc, dtype, registered, binary, expected):
    from vllm_omni.platforms.npu.models.minimax_h3_dit import _supports_eagle_qkv_on_a5

    sdk = tmp_path / "mindiesd"
    sdk.mkdir()
    (sdk / "__init__.py").touch()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "mindiesd", raising=False)

    vendor = sdk / "ops/vendors/aie_ascendc"
    kernels = vendor / "op_impl/ai_core/tbe/kernel"
    descriptor = kernels / soc / "eagle_split_norm_rope_concat/EagleSplitNormRopeConcat_0.json"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text("{}", encoding="utf-8")
    if binary:
        descriptor.with_suffix(".o").touch()
    manifest = kernels / "config" / soc / "eagle_split_norm_rope_concat.json"
    manifest.parent.mkdir(parents=True)
    entry = {"inputs": [{"dtype": dtype}], "binInfo": {"jsonFilePath": descriptor.relative_to(kernels).as_posix()}}
    manifest.write_text(json.dumps({"binList": [entry]}), encoding="utf-8")
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(vendor if registered else tmp_path / "other_vendor"))

    assert _supports_eagle_qkv_on_a5() is expected


def test_npu_platform_installs_h3_dit_fusion(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.platforms.npu.models import minimax_h3_dit

    # Load the real hook without importing the vllm-ascend plugin on CPU.
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", SimpleNamespace(NPUPlatform=object))
    platform = runpy.run_module("vllm_omni.platforms.npu.platform")["NPUOmniPlatform"]
    monkeypatch.setattr(minimax_h3_dit, "_ascend_device_name", lambda: "A2")
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    attention = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
    nn.Module.__init__(attention)
    pipeline.transformer = nn.Sequential(attention)

    platform.apply_diffusion_model_patches(pipeline, SimpleNamespace(model_class_name="MiniMaxH3Pipeline"))

    assert attention._project_qkv.__func__ is minimax_h3_dit._npu_project_qkv


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


def _install_npu_qkv(monkeypatch, attention, device_name, *, a5_native=True):
    from vllm_omni.platforms.npu.models import minimax_h3_dit

    def project(x):
        return Mock(wraps=x, shape=x.shape, device=SimpleNamespace(type="npu")), None

    attention.qkv_proj = Mock(side_effect=project)
    monkeypatch.setattr(minimax_h3_dit, "_ascend_device_name", lambda: device_name)
    monkeypatch.setattr(minimax_h3_dit, "_supports_eagle_qkv_on_a5", lambda: a5_native)
    minimax_h3_dit.apply_minimax_h3_dit_fusions(nn.ModuleList([attention]))


@pytest.fixture
def eagle_qkv(monkeypatch):
    op = Mock()
    monkeypatch.setitem(sys.modules, "mindiesd.fused_ops", SimpleNamespace(eagle_split_norm_rope_concat=op))
    return op


def test_missing_device_detection_disables_eagle(monkeypatch):
    from vllm_omni.platforms.npu.models.minimax_h3_dit import _ascend_device_name

    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", None)
    assert _ascend_device_name() is None


def test_npu_qkv_patch_is_local_to_the_model(monkeypatch, eagle_qkv):
    from vllm_omni.platforms.npu.models import minimax_h3_dit

    attention, packed, table = _qkv_inputs()
    expected = attention._project_qkv(packed, table)
    common_method = type(attention)._project_qkv
    monkeypatch.setattr(minimax_h3_dit, "_ascend_device_name", lambda: "A2")
    minimax_h3_dit.apply_minimax_h3_dit_fusions(attention)

    assert type(attention)._project_qkv is common_method
    torch.testing.assert_close(attention._project_qkv(packed, table), expected, atol=0, rtol=0)
    eagle_qkv.assert_not_called()


@pytest.mark.parametrize("device_name", ["A2", "A5"])
def test_project_qkv_calls_eagle_with_local_rope_halves(monkeypatch, eagle_qkv, device_name):
    attention, packed, table = _qkv_inputs()
    attention.vdn = nn.Identity()
    q, k, _ = packed.view(7, 6, 128).split(2, dim=1)
    outputs = (object(), object(), object())
    eagle_qkv.return_value = outputs
    _install_npu_qkv(monkeypatch, attention, device_name)

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
def test_project_qkv_eagle_numel_limit(monkeypatch, eagle_qkv, rows, use_eagle):
    attention, _, _ = _qkv_inputs()
    attention.to("meta")
    attention.num_heads = attention.num_kv_heads = 56
    packed = torch.empty(rows, 3 * attention.num_heads * attention.head_dim, device="meta", dtype=torch.bfloat16)
    table = packed.new_empty((rows, attention.rot_dim))
    eagle_qkv.return_value = (None, None, None)
    _install_npu_qkv(monkeypatch, attention, "A2")

    q, *_ = attention._project_qkv(packed, table)

    assert eagle_qkv.called == use_eagle
    if not use_eagle:
        assert q.shape == (rows, attention.num_heads, attention.head_dim)


@pytest.mark.parametrize(("head_dim", "rot_dim", "eps"), [(128, 48, 1e-6), (96, 96, 1e-6), (128, 96, 0.0)])
def test_project_qkv_falls_back_for_unsupported_eagle_config(monkeypatch, eagle_qkv, head_dim, rot_dim, eps):
    attention, packed, table = _qkv_inputs(head_dim=head_dim, rot_dim=rot_dim, eps=eps)
    expected = attention._project_qkv(packed, table)
    _install_npu_qkv(monkeypatch, attention, "A2")

    actual = attention._project_qkv(packed, table)

    eagle_qkv.assert_not_called()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("device_name", "has_rope", "expected_calls"),
    [
        pytest.param("A2", True, 1, id="missing-native"),
        pytest.param("A2", False, 0, id="no-rope"),
        pytest.param("A3", True, 0, id="unsupported-device"),
    ],
)
def test_project_qkv_preserves_fallback(monkeypatch, eagle_qkv, device_name, has_rope, expected_calls):
    attention, packed, table = _qkv_inputs()
    table = table if has_rope else None
    expected = attention._project_qkv(packed, table)

    eagle_qkv.side_effect = ImportError("native QKV unavailable")
    _install_npu_qkv(monkeypatch, attention, device_name)
    actual = attention._project_qkv(packed, table)

    attention.qkv_proj.assert_called_once_with(packed)
    assert eagle_qkv.call_count == expected_calls
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_a5_without_native_qkv_uses_fallback_with_public_op_present(monkeypatch, eagle_qkv):
    attention, packed, table = _qkv_inputs()
    expected = attention._project_qkv(packed, table)
    _install_npu_qkv(monkeypatch, attention, "A5", a5_native=False)

    actual = attention._project_qkv(packed, table)

    eagle_qkv.assert_not_called()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _swiglu_inputs(mocker, quant_method, *, device_name="A2"):
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
    from vllm_omni.platforms.npu.models import minimax_h3_dit

    mocker.patch.object(minimax_h3_dit, "_ascend_device_name", return_value=device_name)
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
    minimax_h3_dit.apply_minimax_h3_dit_fusions(mlp)
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


@pytest.mark.parametrize("case", ["missing_sdk", "missing_native", "non_int8", "wrapped", "A5"])
def test_mlp_preserves_swiglu_fallback(mocker, case):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_omni.quantization.int8_config import DiffusionInt8Config, NPUInt8LinearMethod

    method = NPUInt8LinearMethod(DiffusionInt8Config(is_checkpoint_int8_serialized=True))
    mlp, x, quantize, fused = _swiglu_inputs(mocker, method, device_name="A5" if case == "A5" else "A2")
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
