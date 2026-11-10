# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# Load the actual adapter without importing Omni's GPU runtime/registry. This
# makes guard and lifecycle tests runnable in CPU CI with no AITER installation.
_PATH = Path(__file__).parents[4] / "vllm_omni/diffusion/models/minimax_h3/aiter_gemm.py"
_SPEC = importlib.util.spec_from_file_location("minimax_h3_aiter_under_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
gemm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gemm)


def _tensor(shape=(4, 8), *, dtype=torch.bfloat16, device="cuda:0", strided=False):
    if strided:
        elem = torch.empty_strided(shape, (shape[1] * 2, 2), dtype=dtype, device="meta")
    else:
        elem = torch.empty(shape, dtype=dtype, device="meta")
    return FakeTensor(FakeTensorMode(), elem, torch.device(device))


def _linear(mocker, cls=QKVParallelLinear, *, weight=None):
    # Exercise real vLLM types/forward methods without initializing distributed
    # state or allocating the model's multi-GiB projections in CPU tests.
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.weight = _tensor((16, 8)) if weight is None else weight
    layer.bias = None
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.skip_bias_add = False
    layer.return_bias = True
    layer.gather_output = False
    layer.input_is_parallel = True
    layer.reduce_results = True
    original = UnquantizedLinearMethod.__new__(UnquantizedLinearMethod)
    original.apply = mocker.Mock(return_value=object())
    layer.quant_method = original
    return layer


def _block(mocker):
    return types.SimpleNamespace(
        attn=types.SimpleNamespace(qkv_proj=_linear(mocker)),
        mlp=types.SimpleNamespace(
            fc1=_linear(mocker, MergedColumnParallelLinear),
            fc2=_linear(mocker, RowParallelLinear),
        ),
    )


@pytest.fixture(autouse=True)
def runtime(monkeypatch, mocker):
    monkeypatch.setattr(gemm, "current_platform", types.SimpleNamespace(is_rocm=lambda: True))
    monkeypatch.setattr(gemm.vllm_envs, "VLLM_BATCH_INVARIANT", False)
    gemm._has_tuned_gemm.cache_clear()
    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    tuned = types.ModuleType("aiter.tuned_gemm")
    setattr(tuned, "get_GEMM_A16W16_tuned_config", mocker.Mock(return_value={"libtype": "hipblaslt"}))
    setattr(tuned, "gemm_a16w16", mocker.Mock(return_value=object()))
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.tuned_gemm", tuned)
    yield tuned
    gemm._has_tuned_gemm.cache_clear()


def test_tuned_dispatch_uses_existing_aiter_interface_and_caches_lookup(runtime, mocker):
    layer = _linear(mocker)
    original = layer.quant_method
    layer.quant_method = gemm.MiniMaxH3AiterLinearMethod(original)
    x = _tensor()
    with torch.inference_mode():
        for _ in range(2):
            result = layer.quant_method.apply(layer, x)
    assert result is runtime.gemm_a16w16.return_value
    runtime.get_GEMM_A16W16_tuned_config.assert_called_once_with(
        4, 16, 8, False, "torch.bfloat16", "torch.bfloat16", False, False
    )
    assert runtime.gemm_a16w16.call_count == 2
    args, kwargs = runtime.gemm_a16w16.call_args
    assert args[0] is x and args[1] is layer.weight
    assert kwargs == {"otype": torch.bfloat16}
    original.apply.assert_not_called()


def test_untuned_shape_delegates_without_initializing_native_gemm(runtime, mocker):
    runtime.get_GEMM_A16W16_tuned_config.return_value = None
    layer = _linear(mocker)
    original = layer.quant_method
    x = _tensor()
    method = gemm.MiniMaxH3AiterLinearMethod(original)
    with torch.inference_mode():
        result = method.apply(layer, x)
    assert result is original.apply.return_value
    original.apply.assert_called_once_with(layer, x, None)
    runtime.gemm_a16w16.assert_not_called()


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "fp32",
        "rank3",
        "strided_input",
        "empty",
        "bias",
        "layer_bias",
        "tp2",
        "weight_dtype",
        "weight_rank",
        "weight_device",
        "strided_weight",
        "shuffled",
        "k_mismatch",
        "training",
        "compile",
        "batch_invariant",
        "non_rocm",
    ],
)
def test_unsupported_input_preserves_original_method(case, runtime, monkeypatch, mocker):
    layer = _linear(mocker)
    original = layer.quant_method
    x = _tensor()
    bias = None
    if case == "cpu":
        x = torch.empty((4, 8), dtype=torch.bfloat16)
    elif case == "fp32":
        x = _tensor(dtype=torch.float32)
    elif case == "rank3":
        x = _tensor((1, 4, 8))
    elif case == "strided_input":
        x = _tensor(strided=True)
    elif case == "empty":
        x = _tensor((0, 8))
    elif case == "bias":
        bias = _tensor((16,))
    elif case == "layer_bias":
        layer.bias = _tensor((16,))
    elif case == "tp2":
        layer.tp_size = 2
    elif case == "weight_dtype":
        layer.weight = _tensor((16, 8), dtype=torch.float32)
    elif case == "weight_rank":
        layer.weight = _tensor((1, 16, 8))
    elif case == "weight_device":
        layer.weight = _tensor((16, 8), device="cuda:1")
    elif case == "strided_weight":
        layer.weight = _tensor((16, 8), strided=True)
    elif case == "shuffled":
        layer.weight.is_shuffled = True
    elif case == "k_mismatch":
        x = _tensor((4, 7))
    elif case == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif case == "batch_invariant":
        monkeypatch.setattr(gemm.vllm_envs, "VLLM_BATCH_INVARIANT", True)
    elif case == "non_rocm":
        monkeypatch.setattr(gemm, "current_platform", types.SimpleNamespace(is_rocm=lambda: False))
    method = gemm.MiniMaxH3AiterLinearMethod(original)
    with torch.set_grad_enabled(case == "training"):
        result = method.apply(layer, x, bias)
    assert result is original.apply.return_value
    args = original.apply.call_args.args
    assert args[0] is layer and args[1] is x and args[2] is bias
    runtime.get_GEMM_A16W16_tuned_config.assert_not_called()
    runtime.gemm_a16w16.assert_not_called()


def test_weight_creation_and_postprocessing_delegate_to_original_state(mocker):
    layer = _linear(mocker)
    original = layer.quant_method
    original.create_weights = mocker.Mock()
    original.process_weights_after_loading = mocker.Mock()
    method = gemm.MiniMaxH3AiterLinearMethod(original)
    loader = object()
    method.create_weights(layer, 8, [16], 8, 16, torch.bfloat16, weight_loader=loader)
    method.process_weights_after_loading(layer)
    original.create_weights.assert_called_once_with(layer, 8, [16], 8, 16, torch.bfloat16, weight_loader=loader)
    original.process_weights_after_loading.assert_called_once_with(layer)


def test_installer_is_scoped_to_native_unquantized_main_block_projections(mocker):
    class SpecializedQKV(QKVParallelLinear):
        pass

    class SpecializedMethod(UnquantizedLinearMethod):
        pass

    block = _block(mocker)
    originals = [block.attn.qkv_proj.quant_method, block.mlp.fc1.quant_method, block.mlp.fc2.quant_method]
    other = _block(mocker)
    other.attn.qkv_proj = _linear(mocker, SpecializedQKV)
    other.mlp.fc1.quant_method = SpecializedMethod.__new__(SpecializedMethod)
    other.mlp.fc2.quant_method = object()  # A quantized/custom linear method.
    excluded = [other.attn.qkv_proj.quant_method, other.mlp.fc1.quant_method, other.mlp.fc2.quant_method]
    gemm.enable_minimax_h3_aiter_gemm([block, other])
    methods = [block.attn.qkv_proj.quant_method, block.mlp.fc1.quant_method, block.mlp.fc2.quant_method]
    assert all(type(method) is gemm.MiniMaxH3AiterLinearMethod for method in methods)
    assert all(method._original is original for method, original in zip(methods, originals))
    assert [other.attn.qkv_proj.quant_method, other.mlp.fc1.quant_method, other.mlp.fc2.quant_method] == excluded
    gemm.enable_minimax_h3_aiter_gemm([block])
    assert block.attn.qkv_proj.quant_method is methods[0]


def test_installer_is_disabled_on_other_platforms(monkeypatch, mocker):
    block = _block(mocker)
    original = block.attn.qkv_proj.quant_method
    monkeypatch.setattr(gemm, "current_platform", types.SimpleNamespace(is_rocm=lambda: False))
    gemm.enable_minimax_h3_aiter_gemm([block])
    assert block.attn.qkv_proj.quant_method is original


@pytest.mark.parametrize("cls", [QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear])
@pytest.mark.parametrize("return_bias", [False, True])
def test_native_forward_and_bias_return_contract_are_preserved(cls, return_bias, mocker):
    layer = _linear(mocker, cls, weight=torch.randn(16, 8))
    original = layer.quant_method
    # Use the real unquantized apply, retaining its already-created backend.
    del original.apply
    original._gemm_impl = lambda layer, x, weight, bias: F.linear(x, weight, bias)
    layer.quant_method = gemm.MiniMaxH3AiterLinearMethod(original)
    layer.bias = torch.randn(16)
    layer.skip_bias_add = True
    layer.return_bias = return_bias
    x = torch.randn(4, 8)
    with torch.inference_mode():
        result = layer(x)
    if return_bias:
        output, bias = result
        assert bias is layer.bias
    else:
        output = result
    torch.testing.assert_close(output, F.linear(x, layer.weight))
