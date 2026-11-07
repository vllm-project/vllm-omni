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
