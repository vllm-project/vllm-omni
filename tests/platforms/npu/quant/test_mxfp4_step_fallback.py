# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real A5 kernels: step selection, bias, shape, and reuse of prepared W4 weights."""

import pytest
import torch

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.npu]

torch_npu = pytest.importorskip("torch_npu")


@torch.inference_mode()
def _exercise_mxfp4_fallback(with_bias, dtype, input_shape, mode, monkeypatch, *, policy, scale_alg=2):
    from vllm_omni.diffusion.forward_context import (
        ForwardContext,
        override_forward_context,
        set_forward_context_denoise_step_idx,
    )
    from vllm_omni.quantization.mxfp4_config import (
        DiffusionMXFP4Config,
        NPUMxfp4LinearMethod,
        NPUMxfp4OnlineLinearMethod,
    )

    prefix = "blocks.10.attn1.to_qkv"
    steps = [1] if policy in ("step", "layer_and_step") else []
    layers = [prefix] if policy in ("layer", "layer_and_step") else []
    fallback = {"w4a8_fallback_steps": steps, "w4a8_fallback_layers": layers, "mxfp4_scale_alg": scale_alg}
    torch.manual_seed(42)
    layer = torch.nn.Linear(1024, 128, bias=with_bias, device="npu", dtype=dtype)
    if mode == "single":
        method = NPUMxfp4OnlineLinearMethod(DiffusionMXFP4Config(**fallback), prefix=prefix)
    elif mode == "single_offline":
        method = NPUMxfp4LinearMethod(
            DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True, **fallback), prefix=prefix
        )
        layer.orig_dtype = dtype
        values = torch.tensor([-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        layer.weight.copy_(values[torch.randint(8, (128, 1024))].to(layer.weight))
        layer.register_parameter(
            "weight_scale", torch.nn.Parameter(torch.full((128, 32), 124, dtype=torch.uint8, device="npu"), False)
        )
        layer.register_parameter("mul_scale", torch.nn.Parameter((torch.rand(1024) * 1.5 + 0.5).npu(), False))
    else:
        raise ValueError(f"Unsupported single-scale test mode: {mode}")
    method.process_weights_after_loading(layer)
    x = torch.randn(input_shape, device="npu", dtype=dtype)
    pointers = (layer.weight.data_ptr(), layer.weight_scale.data_ptr())
    method.process_weights_after_loading(layer)
    quantize = torch_npu.npu_dynamic_mx_quant
    activation_types = []

    def record_quantize(x, **kwargs):
        dst = kwargs["dst_type"]
        expected_alg = 0 if dst == torch_npu.float8_e4m3fn else scale_alg
        assert kwargs["scale_alg"] == expected_alg
        assert kwargs["axis"] == -1 and kwargs["block_size"] == 32 and kwargs["round_mode"] == "rint"
        if expected_alg == 2:
            assert kwargs["dst_type_max"] == 7.25
        else:
            assert "dst_type_max" not in kwargs
        activation_types.append(dst)
        return quantize(x, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_dynamic_mx_quant", record_quantize)
    results = []
    for _request in range(2):
        with override_forward_context(ForwardContext()):
            for step in range(3):
                set_forward_context_denoise_step_idx(step)
                output = method.apply(layer, x, layer.bias)
                torch.npu.synchronize()
                assert output.shape == (*input_shape[:-1], 128)
                assert output.dtype == dtype
                assert torch.isfinite(output).all()
                results.append(output.cpu())
    expected_types = [
        torch_npu.float8_e4m3fn if layers or step in steps else torch_npu.float4_e2m1fn_x2 for step in range(3)
    ]
    assert activation_types == expected_types * 2
    assert (layer.weight.data_ptr(), layer.weight_scale.data_ptr()) == pointers
    assert not hasattr(layer, "w4a8_weight")
    assert not hasattr(layer, "w4a8_weight_scale")
    for first, second in zip(results[:3], results[3:]):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("input_shape", [(16, 1024), (2, 8, 1024)])
@pytest.mark.parametrize("mode", ["single", "single_offline"])
def test_mxfp4_step_fallback(with_bias, dtype, input_shape, mode, monkeypatch):
    _exercise_mxfp4_fallback(with_bias, dtype, input_shape, mode, monkeypatch, policy="step")


@pytest.mark.cards_1
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_single_scale_step_fallback_matches_independent_dense_reference(dtype):
    if not torch.npu.is_available() or "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("MXFP4 kernels require an Ascend950 NPU")

    from vllm_omni.diffusion.forward_context import (
        ForwardContext,
        override_forward_context,
        set_forward_context_denoise_step_idx,
    )
    from vllm_omni.quantization.mxfp4_config import (
        DiffusionMXFP4Config,
        NPUMxfp4LinearMethod,
    )

    # Synthetic serialized single-scale values and E8M0 exponent bytes.
    # Powers of two make both activation precisions exact: this independent
    # CPU dense oracle needs no production packing, dequantization or apply.
    # Real C7 export validation remains a separate checkpoint/E2E gate.
    values = torch.tensor([1.0, -1.0, 2.0, -2.0]).repeat(32).unsqueeze(1).expand(128, 512)
    weight = torch.cat((values, values), dim=1)
    fine = torch.cat((torch.full((128, 16), 127), torch.full((128, 16), 128)), dim=1).to(torch.uint8)
    smooth = torch.cat((torch.full((512,), 0.5), torch.full((512,), 2.0)))
    bias = torch.tensor([0.25, -0.25]).repeat(64)
    x = torch.tensor([1.0, -1.0, 0.5, -0.5]).repeat(4).unsqueeze(1).expand(16, 1024) / 64
    expected = torch.nn.functional.linear(x * smooth, torch.cat((values, 2 * values), dim=1), bias)

    layer = torch.nn.Linear(1024, 128, bias=True, device="npu", dtype=dtype)
    layer.weight.copy_(weight)
    layer.bias.copy_(bias)
    layer.orig_dtype = dtype
    for name, value in {"weight_scale": fine, "mul_scale": smooth}.items():
        layer.register_parameter(name, torch.nn.Parameter(value.npu(), requires_grad=False))
    method = NPUMxfp4LinearMethod(
        DiffusionMXFP4Config(
            is_checkpoint_mxfp4_serialized=True, require_smooth_scale=True, w4a8_fallback_steps=[1], mxfp4_scale_alg=2
        )
    )
    method.process_weights_after_loading(layer)
    prepared_pointers = (layer.weight.data_ptr(), layer.weight_scale.data_ptr())
    assert set(layer.state_dict()) == {"weight", "bias", "weight_scale", "mul_scale"}
    assert not hasattr(layer, "w4a8_weight")
    assert not hasattr(layer, "w4a8_weight_scale")
    x = x.to(device="npu", dtype=dtype)

    with override_forward_context(ForwardContext()):
        for step in range(3):
            set_forward_context_denoise_step_idx(step)
            actual = method.apply(layer, x, layer.bias).cpu()
            # A8's documented GEMM output is BF16 before restoring the caller
            # dtype; the reference does not reuse quantization or unpacking.
            reference = expected.bfloat16().to(dtype) if step == 1 else expected.to(dtype)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert actual.dtype == dtype
    assert (layer.weight.data_ptr(), layer.weight_scale.data_ptr()) == prepared_pointers


@pytest.mark.cards_1
@pytest.mark.parametrize("mode", ["single", "single_offline"])
@pytest.mark.parametrize("policy", ["layer", "layer_and_step"])
def test_mxfp4_layer_fallback(mode, policy, monkeypatch):
    """Reuse the kernel/shape/cache assertions without multiplying the full dtype/bias matrix."""
    _exercise_mxfp4_fallback(False, torch.bfloat16, (16, 1024), mode, monkeypatch, policy=policy)


@pytest.mark.cards_1
@pytest.mark.parametrize("mode", ["single", "single_offline"])
def test_mxfp4_ocp_compatibility(mode, monkeypatch):
    """OCP compatibility: A4 -> A8 -> A4 with the same W4 across requests."""
    _exercise_mxfp4_fallback(False, torch.bfloat16, (16, 1024), mode, monkeypatch, policy="step", scale_alg=0)
