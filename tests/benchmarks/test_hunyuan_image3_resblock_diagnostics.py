# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import importlib
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from benchmarks.kernels import hunyuan_image3_resblock_benchmarks as benchmark

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _diagnostics():
    return importlib.import_module("benchmarks.kernels.hunyuan_image3_resblock_diagnostics")


def test_diagnostic_error_metrics_keep_original_gate() -> None:
    diagnostic = _diagnostics()
    reference = torch.tensor([0.0, 1.0, 2.0], dtype=torch.bfloat16)
    candidate = torch.tensor([0.125, 1.0, 2.0], dtype=torch.bfloat16)
    result = diagnostic.compare(candidate, reference)
    assert result["rtol"] == result["atol"] == 0.02
    assert result["mismatched"] == 1
    assert result["numel"] == 3
    assert not result["passed"]
    assert result["max_abs_diff"] == 0.125
    json.dumps(result, allow_nan=False)


def test_gate_uses_original_bf16_isclose_not_upcast_arithmetic() -> None:
    reference = torch.tensor([0.00119781494140625], dtype=torch.bfloat16)
    candidate = torch.tensor([0.021240234375], dtype=torch.bfloat16)
    torch.testing.assert_close(candidate, reference, rtol=0.02, atol=0.02)
    assert _diagnostics().compare(candidate, reference)["passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_diagnostics_fail_and_serialize(value: float) -> None:
    result = _diagnostics().compare(torch.tensor([value]), torch.tensor([value]))
    assert result["passed"] is False
    assert result["nonfinite"] == 1
    json.dumps(result, allow_nan=False)


def test_rounding_probe_is_mechanism_not_failure_cause() -> None:
    result = _diagnostics().rounding_probe()
    assert result["stepwise"] == 0.73046875
    assert result["fp32_then_store"] == 0.734375
    assert result["comparison"]["passed"]
    assert result["establishes_gpu_failure_cause"] is False


def test_diagnostic_cli_requires_original_production_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark", "--diagnose", "--case", "production_patch_embed"])
    args = benchmark._parse_args()
    assert args.diagnose
    assert args.dtype == "bfloat16"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--diagnose", "--case", "identity"])
    with pytest.raises(SystemExit):
        benchmark._parse_args()
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--diagnose", "--case", "production_final_layer", "--dtype", "float32"]
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()


def test_diagnostic_cli_writes_failure_before_exiting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "failed.json"
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--diagnose", "--case", "production_final_layer", "--output-json", str(output)]
    )
    monkeypatch.setattr(benchmark, "_run_diagnostic", lambda args: {"qualification_passed": False})
    with pytest.raises(SystemExit) as failure:
        benchmark.main()
    assert failure.value.code == 1
    assert json.loads(output.read_text()) == {"qualification_passed": False}


@pytest.fixture
def native_block():
    from vllm_omni.diffusion.models.hunyuan_image3.layers.native.transformer_blocks import ResBlock

    with torch.random.fork_rng():
        torch.manual_seed(19)
        block = ResBlock(32, 16, out_channels=64, dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.fill_(0.02)
        inputs = torch.randn(1, 32, 4, 4).bfloat16()
        embedding = torch.randn(1, 16).bfloat16()
    return block, inputs, embedding


def test_native_capture_matches_real_forward_and_removes_hooks(native_block) -> None:
    block, inputs, embedding = native_block
    diagnostic = _diagnostics()
    policy = diagnostic.backend_policy()
    with torch.inference_mode():
        expected = block(inputs, embedding)
        capture, policies = diagnostic.capture(block, inputs, embedding, fused=False)
    torch.testing.assert_close(capture["final_add"], expected, rtol=0, atol=0)
    torch.testing.assert_close(capture["skip"] + capture["conv2"], expected, rtol=0, atol=0)
    assert {
        "gn",
        "gn_silu",
        "conv1",
        "embedding",
        "adaptive_gn",
        "modulation",
        "adaptive_gn_silu",
        "conv2",
        "skip",
        "final_add",
    } == set(capture)
    assert policies["conv1"] == policy
    assert diagnostic.backend_policy() == policy
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in block.modules())


def test_capture_removes_hooks_after_failure(native_block, monkeypatch: pytest.MonkeyPatch) -> None:
    block, inputs, embedding = native_block

    def fail(*args):
        raise RuntimeError("sentinel")

    monkeypatch.setattr(block.in_layers[2], "forward", fail)
    with pytest.raises(RuntimeError, match="sentinel"):
        _diagnostics().capture(block, inputs, embedding, fused=False)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in block.modules())


def test_cudnn_policy_restores_after_failure() -> None:
    diagnostic = _diagnostics()
    before = diagnostic.backend_policy()
    with (
        pytest.raises(RuntimeError, match="sentinel"),
        diagnostic.convolution_policy({**before, "cudnn_benchmark": not before["cudnn_benchmark"]}),
    ):
        raise RuntimeError("sentinel")
    assert diagnostic.backend_policy() == before


def test_diagnostic_uses_unchanged_qualification_fixtures() -> None:
    qualification = importlib.import_module(
        "tests.diffusion.models.hunyuan_image3.test_hunyuan_image3_resblock_qualification"
    )
    assert benchmark._SEED == qualification._SEED == 20260911
    assert benchmark._TOLERANCES == qualification._TOLERANCES
    for case in qualification._PRODUCTION_BF16_CASES:
        values = asdict(case)
        name = values.pop("name")
        assert asdict(benchmark.CASES[name]) == values
    for name in ("_randomize_parameters", "_make_inputs"):
        assert ast.dump(ast.parse(inspect.getsource(getattr(benchmark, name)))) == ast.dump(
            ast.parse(inspect.getsource(getattr(qualification, name)))
        )


def test_cpu_fallback_rehearses_full_diagnostic_without_claiming_triton(
    native_block, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_omni.diffusion.models.hunyuan_image3.layers.nvidia.transformer_blocks import ResBlock

    for module in benchmark._load_fused_op_modules():
        monkeypatch.setattr(module, "HAS_TRITON", False)
    native, inputs, embedding = native_block
    nvidia = ResBlock(32, 16, out_channels=64, dtype=torch.bfloat16).eval()
    nvidia.load_state_dict(native.state_dict(), strict=True)
    diagnostic = _diagnostics()
    before = diagnostic.backend_policy()
    result = diagnostic.diagnose(native, nvidia, inputs, embedding)
    assert result["qualification_passed"]
    assert result["uninstrumented_final"]["passed"]
    assert result["capture_repeatability"] == {"native_bitwise_equal": True, "nvidia_bitwise_equal": True}
    assert "triton_fused_ops" not in result
    assert result["policies"]["nvidia"]["conv1"]["cudnn_benchmark"]
    assert result["policies"]["nvidia"]["conv1"]["cudnn_deterministic"]
    assert result["policies"]["after"] == before
    replay = result["same_native_input_replay"]
    assert replay["conv1"]["original_policies"]["passed"]
    assert replay["conv2"]["matched_nvidia_policy"]["passed"]
    assert replay["conv2"]["native_policy_only"]["passed"]
    assert replay["adaptive_gn_silu"]["passed"]
    assert replay["final_add"]["passed"]
    assert set(result["native_only_intermediates"]) == {"gn", "adaptive_gn", "modulation", "scale", "shift"}
    json.dumps(result, allow_nan=False)


def test_diagnostic_cli_refuses_fused_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in benchmark._load_fused_op_modules():
        monkeypatch.setattr(module, "HAS_TRITON", False)
    with pytest.raises(RuntimeError, match="fallback is active"):
        benchmark._require_triton_fused_ops()


@pytest.mark.parametrize("nonfinite_reference", (False, True))
def test_cpu_fallback_retains_failed_final_gate(
    native_block, monkeypatch: pytest.MonkeyPatch, nonfinite_reference: bool
) -> None:
    from vllm_omni.diffusion.models.hunyuan_image3.layers.nvidia.transformer_blocks import ResBlock

    for module in benchmark._load_fused_op_modules():
        monkeypatch.setattr(module, "HAS_TRITON", False)
    native, inputs, embedding = native_block
    nvidia = ResBlock(32, 16, out_channels=64, dtype=torch.bfloat16).eval()
    nvidia.load_state_dict(native.state_dict(), strict=True)
    with torch.no_grad():
        nvidia.out_layers[3].bias.add_(1)
        if nonfinite_reference:
            native.in_layers[0].weight.fill_(float("nan"))
    result = _diagnostics().diagnose(native, nvidia, inputs, embedding)
    assert result["qualification_passed"] is False
    assert result["uninstrumented_final"]["mismatched"] > 0
    assert result["same_native_input_replay"]["conv2"]["matched_nvidia_policy"]["passed"] is False
    json.dumps(result, allow_nan=False)


def test_source_metadata_binds_actual_files(native_block, monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.hunyuan_image3.layers.nvidia.transformer_blocks import ResBlock

    diagnostic = _diagnostics()
    monkeypatch.setattr(diagnostic, "version", lambda distribution: "cpu-test-no-triton")
    native, _, _ = native_block
    nvidia = ResBlock(32, 16, out_channels=64, dtype=torch.bfloat16).eval()
    result = diagnostic.source_metadata(native, nvidia)
    assert result["triton"] == "cpu-test-no-triton"
    assert len(result["tracked_diff_sha256"]) == 64
    assert str(Path(inspect.getfile(type(native))).resolve()) in result["source_sha256"]
    assert str(Path(inspect.getfile(type(nvidia))).resolve()) in result["source_sha256"]
    assert str(Path(diagnostic.__file__).resolve()) in result["source_sha256"]
    json.dumps(result, allow_nan=False)


def test_fp64_reference_errors_are_not_a_replacement_gate() -> None:
    candidate = torch.tensor([0, 2], dtype=torch.bfloat16)
    reference = torch.tensor([0.125, 1.5], dtype=torch.float64)
    result = _diagnostics().reference_error(candidate, reference)
    assert result["candidate_dtype"] == "torch.bfloat16"
    assert result["reference_dtype"] == "torch.float64"
    assert result["finite_pairs"] == 2
    assert result["nonfinite"] == 0
    assert result["outside_tolerance"] == 2
    assert result["max_abs_error"] == 0.5
    assert result["mean_abs_error"] == 0.3125
    assert result["rms_error"] == pytest.approx((0.1328125) ** 0.5)
    assert result["qualification_gate"] is False
    assert "passed" not in result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("reference_value,closer", [(1.001, "stepwise"), (1.007, "single_store")])
def test_fp64_errors_do_not_assume_one_rounding_route_wins(reference_value: float, closer: str) -> None:
    reference = torch.tensor([reference_value], dtype=torch.float64)
    errors = {
        name: _diagnostics().reference_error(torch.tensor([value], dtype=torch.bfloat16), reference)["mean_abs_error"]
        for name, value in (("stepwise", 1.0), ("single_store", 1.0078125))
    }
    assert min(errors, key=errors.__getitem__) == closer


@pytest.mark.parametrize("all_nonfinite", (False, True))
def test_fp64_errors_preserve_nonfinite_counts_and_json(all_nonfinite: bool) -> None:
    candidate = torch.tensor([float("nan"), float("inf"), 1.0], dtype=torch.bfloat16)
    reference = torch.tensor([0.0, float("inf"), float("nan") if all_nonfinite else 1.0], dtype=torch.float64)
    result = _diagnostics().reference_error(candidate, reference)
    assert result["nonfinite"] == (3 if all_nonfinite else 2)
    assert result["outside_tolerance"] == result["nonfinite"]
    assert result["mean_abs_error"] == (None if all_nonfinite else 0.0)
    assert result["rms_error"] == (None if all_nonfinite else 0.0)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("wrong_shape", (False, True))
def test_fp64_errors_reject_wrong_reference_contract(wrong_shape: bool) -> None:
    reference = torch.ones(2, dtype=torch.float64) if wrong_shape else torch.ones(1)
    with pytest.raises(ValueError, match="FP64 reference"):
        _diagnostics().reference_error(torch.ones(1, dtype=torch.bfloat16), reference)


@pytest.mark.parametrize("adaptive", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_fp64_group_norm_promotes_existing_values(dtype: torch.dtype, adaptive: bool) -> None:
    generator = torch.Generator().manual_seed(27)
    inputs = torch.randn(1, 4, 2, 2, generator=generator).to(dtype)
    norm = torch.nn.GroupNorm(2, 4, eps=1e-6, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(4, generator=generator).to(dtype))
        norm.bias.copy_(torch.randn(4, generator=generator).to(dtype))
    scale = torch.randn(1, 4, 1, 1, generator=generator).to(dtype) if adaptive else None
    shift = torch.randn(1, 4, 1, 1, generator=generator).to(dtype) if adaptive else None
    grouped = inputs.double().reshape(1, 2, -1)
    variance, mean = torch.var_mean(grouped, dim=-1, correction=0, keepdim=True)
    expected = ((grouped - mean) * torch.rsqrt(variance + norm.eps)).reshape_as(inputs)
    expected = expected * norm.weight.double()[None, :, None, None] + norm.bias.double()[None, :, None, None]
    if adaptive:
        expected = expected * (1 + scale.double()) + shift.double()
    expected = torch.nn.functional.silu(expected)
    result = _diagnostics().group_norm_reference(inputs, norm, scale=scale, shift=shift)
    assert result.dtype == torch.float64
    assert result.device == inputs.device
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    assert inputs.dtype == norm.weight.dtype == dtype


def test_fp64_group_norm_requires_both_conditioning_operands() -> None:
    with pytest.raises(ValueError, match="scale and shift"):
        _diagnostics().group_norm_reference(
            torch.ones(1, 4, 2, 2), torch.nn.GroupNorm(2, 4), scale=torch.ones(1, 4, 1, 1)
        )


def test_fp64_block_preserves_equation_quantized_state_and_rng(native_block) -> None:
    native, inputs, embedding = native_block
    state = {name: tensor.clone() for name, tensor in native.state_dict().items()}
    random_state = torch.get_rng_state().clone()
    policy = _diagnostics().backend_policy()
    with torch.inference_mode():
        hidden = torch.nn.functional.group_norm(
            inputs.double(),
            32,
            state["in_layers.0.weight"].double(),
            state["in_layers.0.bias"].double(),
            native.in_layers[0].eps,
        )
        hidden = torch.nn.functional.conv2d(
            torch.nn.functional.silu(hidden),
            state["in_layers.2.weight"].double(),
            state["in_layers.2.bias"].double(),
            padding=1,
        )
        projected = torch.nn.functional.linear(
            torch.nn.functional.silu(embedding.double()),
            state["emb_layers.1.weight"].double(),
            state["emb_layers.1.bias"].double(),
        )[..., None, None]
        scale, shift = projected.chunk(2, dim=1)
        hidden = torch.nn.functional.group_norm(
            hidden,
            32,
            state["out_layers.0.weight"].double(),
            state["out_layers.0.bias"].double(),
            native.out_layers[0].eps,
        )
        hidden = torch.nn.functional.conv2d(
            torch.nn.functional.silu(hidden * (1 + scale) + shift),
            state["out_layers.3.weight"].double(),
            state["out_layers.3.bias"].double(),
            padding=1,
        )
        skip = torch.nn.functional.conv2d(
            inputs.double(),
            state["skip_connection.weight"].double(),
            state["skip_connection.bias"].double(),
        )
    result = _diagnostics().block_precision_reference(native, inputs, embedding)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, hidden + skip, rtol=1e-12, atol=1e-12)
    assert torch.equal(random_state, torch.get_rng_state())
    assert _diagnostics().backend_policy() == policy
    for name, tensor in native.state_dict().items():
        assert tensor.dtype == state[name].dtype
        assert torch.equal(tensor, state[name])


def test_precision_replay_uses_actual_supplied_fused_output(native_block, monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.hunyuan_image3.layers.nvidia import transformer_blocks

    for module in benchmark._load_fused_op_modules():
        monkeypatch.setattr(module, "HAS_TRITON", False)
    native, inputs, embedding = native_block
    nvidia = transformer_blocks.ResBlock(32, 16, out_channels=64, dtype=torch.bfloat16).eval()
    nvidia.load_state_dict(native.state_dict())
    supplied = torch.full_like(inputs, 4.0)
    supplied_adaptive = torch.full((1, 64, 4, 4), -0.5, dtype=inputs.dtype)
    calls = []
    adaptive_calls = []
    original_forward = type(native).forward

    def supplied_fused_output(*args, **kwargs):
        calls.append(args[0].clone())
        return supplied.clone()

    def supplied_adaptive_output(*args, **kwargs):
        adaptive_calls.append(args[0].clone())
        return supplied_adaptive.clone()

    def policy_sensitive_forward(block, hidden, conditioning):
        output = original_forward(block, hidden, conditioning)
        if hidden.dtype == torch.bfloat16 and torch.backends.cudnn.benchmark:
            output = output + 1
        return output

    monkeypatch.setattr(transformer_blocks, "fused_group_norm_silu", supplied_fused_output)
    monkeypatch.setattr(transformer_blocks, "fused_adaptive_group_norm_silu", supplied_adaptive_output)
    monkeypatch.setattr(type(native), "forward", policy_sensitive_forward)
    diagnostic = _diagnostics()
    before = diagnostic.backend_policy()
    default_policy = {**before, "cudnn_benchmark": False, "cudnn_deterministic": False}
    with diagnostic.convolution_policy(default_policy), torch.inference_mode():
        captured, _ = diagnostic.capture(native, inputs, embedding, fused=False)
        high_precision = diagnostic.block_precision_reference(native, inputs, embedding)
        native_output = native(inputs, embedding)
        nvidia_output = nvidia(inputs, embedding)
        with diagnostic.convolution_policy({**default_policy, "cudnn_benchmark": True, "cudnn_deterministic": True}):
            native_matched = native(inputs, embedding)
        result = diagnostic.diagnose(native, nvidia, inputs, embedding)
    assert diagnostic.backend_policy() == before
    assert not torch.equal(native_output, native_matched)
    assert calls and all(torch.equal(value, inputs) for value in calls)
    expected = diagnostic.group_norm_reference(inputs, native.in_layers[0])
    precision = result["same_native_input_replay"]["gn_silu"]["fp64_reference"]
    assert precision["native"] == diagnostic.reference_error(captured["gn_silu"], expected)
    assert precision["fused"] == diagnostic.reference_error(supplied, expected)
    assert precision["reference_device"] == "cpu"
    scale, shift = captured["embedding"][..., None, None].chunk(2, dim=1)
    adaptive_reference = diagnostic.group_norm_reference(
        captured["conv1"], native.out_layers[0], scale=scale, shift=shift
    )
    assert adaptive_calls and torch.equal(adaptive_calls[-1], captured["conv1"])
    adaptive_precision = result["same_native_input_replay"]["adaptive_gn_silu"]["fp64_reference"]
    assert adaptive_precision["native"] == diagnostic.reference_error(captured["adaptive_gn_silu"], adaptive_reference)
    assert adaptive_precision["fused"] == diagnostic.reference_error(supplied_adaptive, adaptive_reference)
    whole = result["whole_block_fp64_reference"]
    assert whole["reference_dtype"] == "torch.float64"
    assert whole["native_default"] == diagnostic.reference_error(native_output, high_precision)
    assert whole["native_matched_policy"] == diagnostic.reference_error(native_matched, high_precision)
    assert whole["nvidia"] == diagnostic.reference_error(nvidia_output, high_precision)
    assert whole["native_policy_only"] == diagnostic.compare(native_matched, native_output)
    assert whole["nvidia_vs_native_matched_policy"] == diagnostic.compare(nvidia_output, native_matched)
    assert whole["native_default_policy"] == result["policies"]["native"]["conv1"]
    assert whole["native_matched_policy_settings"] == result["policies"]["nvidia"]["conv1"]
    assert whole["reference_policy"] == default_policy
    assert result["qualification_passed"] == diagnostic.compare(nvidia_output, native_output)["passed"]
    assert "triton_fused_ops" not in result
    json.dumps(result, allow_nan=False)
