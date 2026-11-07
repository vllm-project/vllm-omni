# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Untimed ResBlock attribution; never substitutes for the qualification suite."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import platform
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import torch
from torch.nn import functional as functional

from benchmarks.kernels.hunyuan_image3_resblock_benchmarks import _TOLERANCES


def compare(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, object]:
    rtol, atol = _TOLERANCES[reference.dtype]
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise ValueError("Diagnostic comparisons require matching shapes and dtypes")
    comparison_reference = reference.to(candidate.device)
    matches = (
        torch.isfinite(candidate)
        & torch.isfinite(comparison_reference)
        & torch.isclose(candidate, comparison_reference, rtol=rtol, atol=atol)
    ).cpu()
    candidate_float = candidate.detach().double().cpu()
    reference_float = reference.detach().double().cpu()
    finite = torch.isfinite(candidate_float) & torch.isfinite(reference_float)
    difference = (candidate_float - reference_float).abs()
    finite_difference = difference[finite]
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "comparison_device": str(candidate.device),
        "rtol": rtol,
        "atol": atol,
        "numel": reference.numel(),
        "nonfinite": (~finite).sum().item(),
        "mismatched": (~matches).sum().item(),
        "passed": bool(matches.all().item()),
        "max_abs_diff": finite_difference.max().item() if finite_difference.numel() else None,
        "mean_abs_diff": finite_difference.mean().item() if finite_difference.numel() else None,
    }


def backend_policy() -> dict[str, object]:
    return {
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


@contextmanager
def convolution_policy(policy: dict[str, object]) -> Iterator[None]:
    before = backend_policy()
    try:
        torch.backends.cudnn.benchmark = bool(policy["cudnn_benchmark"])
        torch.backends.cudnn.deterministic = bool(policy["cudnn_deterministic"])
        yield
    finally:
        torch.backends.cudnn.benchmark = bool(before["cudnn_benchmark"])
        torch.backends.cudnn.deterministic = bool(before["cudnn_deterministic"])


def capture(
    block: torch.nn.Module, inputs: torch.Tensor, embedding: torch.Tensor, *, fused: bool
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, object]]]:
    """Observe real forwards without replacing either fused operation."""
    if block.updown or block.training or block.dropout != 0:
        raise ValueError("Intermediate diagnostics require plain eval blocks with dropout=0")
    tensors: dict[str, torch.Tensor] = {}
    policies: dict[str, dict[str, object]] = {}
    handles = []

    def record(name: str, tensor: torch.Tensor) -> None:
        tensors[name] = tensor.detach().to(device="cpu", copy=True)
        policies[name] = backend_policy()

    def output_hook(name: str):
        def hook(module, args, output):
            record(name, output)

        return hook

    def input_hook(name: str):
        def hook(module, args):
            record(name, args[0])

        return hook

    outputs = {
        "conv1": block.in_layers[2],
        "embedding": block.emb_layers,
        "conv2": block.out_layers[3],
        "skip": block.skip_connection,
    }
    if not fused:
        outputs.update(
            gn=block.in_layers[0],
            gn_silu=block.in_layers[1],
            adaptive_gn=block.out_layers[0],
            adaptive_gn_silu=block.out_layers[1],
        )
    try:
        for name, module in outputs.items():
            handles.append(module.register_forward_hook(output_hook(name)))
        if fused:
            handles.append(block.in_layers[2].register_forward_pre_hook(input_hook("gn_silu")))
            handles.append(block.out_layers[2].register_forward_pre_hook(input_hook("adaptive_gn_silu")))
        else:
            handles.append(block.out_layers[1].register_forward_pre_hook(input_hook("modulation")))
        with torch.inference_mode():
            record("final_add", block(inputs, embedding))
    finally:
        for handle in handles:
            handle.remove()
    return tensors, policies


def rounding_probe() -> dict[str, object]:
    normalized = torch.ones(1, dtype=torch.bfloat16)
    scale = torch.full_like(normalized, 1 / 256)
    shift = torch.zeros_like(normalized)
    stepwise = functional.silu(normalized * (1 + scale) + shift)
    final_store = functional.silu(normalized.float() * (1 + scale.float()) + shift.float()).bfloat16()
    return {
        "scope": "CPU arithmetic only; no GroupNorm, convolution or Triton execution",
        "stepwise": stepwise.item(),
        "fp32_then_store": final_store.item(),
        "comparison": compare(final_store, stepwise),
        "establishes_gpu_failure_cause": False,
    }


def replay(
    native: torch.nn.Module,
    nvidia: torch.nn.Module,
    inputs: torch.Tensor,
    embedding: torch.Tensor,
    reference: dict[str, torch.Tensor],
    native_policy: dict[str, object],
    nvidia_policy: dict[str, object],
) -> dict[str, object]:
    """Reset each boundary to the captured native input, avoiding error propagation."""
    fused_module = importlib.import_module(type(nvidia).__module__)
    device = inputs.device
    results: dict[str, object] = {}
    in_norm = native.in_layers[0]
    out_norm = native.out_layers[0]
    with torch.inference_mode():
        fused_gn = fused_module.fused_group_norm_silu(
            inputs, in_norm.weight, in_norm.bias, num_groups=in_norm.num_groups, eps=in_norm.eps
        )
        results["gn_silu"] = compare(fused_gn, reference["gn_silu"])
        del fused_gn
        for name, source, native_layer, nvidia_layer in (
            ("conv1", reference["gn_silu"], native.in_layers[2], nvidia.in_layers[2]),
            ("embedding", embedding, native.emb_layers, nvidia.emb_layers),
            ("conv2", reference["adaptive_gn_silu"], native.out_layers[3], nvidia.out_layers[3]),
            ("skip", inputs, native.skip_connection, nvidia.skip_connection),
        ):
            shared_input = source.to(device)
            with convolution_policy(native_policy):
                native_output = native_layer(shared_input).cpu()
            with convolution_policy(nvidia_policy):
                nvidia_output = nvidia_layer(shared_input).cpu()
                native_matched_policy = native_layer(shared_input).cpu()
            results[name] = {
                "original_policies": compare(nvidia_output, native_output),
                "matched_nvidia_policy": compare(nvidia_output, native_matched_policy),
                "native_policy_only": compare(native_matched_policy, native_output),
            }
            del shared_input, native_output, nvidia_output, native_matched_policy

        hidden = reference["conv1"].to(device)
        projected = reference["embedding"].to(device)[..., None, None]
        scale, shift = projected.chunk(2, dim=1)
        adaptive = fused_module.fused_adaptive_group_norm_silu(
            hidden, out_norm.weight, out_norm.bias, scale, shift, num_groups=out_norm.num_groups, eps=out_norm.eps
        )
        results["adaptive_gn_silu"] = compare(adaptive, reference["adaptive_gn_silu"])
        del adaptive, hidden
        normalized = reference["adaptive_gn"].to(device)
        stepwise_modulation = normalized * (1 + scale) + shift
        fp32_modulation = normalized.float() * (1 + scale.float()) + shift.float()
        results["modulation_rounding_only"] = {
            "scope": "Captured native GN output; FP32 expression is not the fused kernel's internal GN",
            "modulation": compare(fp32_modulation.to(normalized.dtype), stepwise_modulation),
            "silu": compare(
                functional.silu(fp32_modulation).to(normalized.dtype), functional.silu(stepwise_modulation)
            ),
            "establishes_gpu_failure_cause": False,
        }
        shared_skip = reference["skip"].to(device)
        shared_residual = reference["conv2"].to(device)
        results["final_add"] = compare(shared_skip + shared_residual, reference["final_add"])
    return results


def source_metadata(native: torch.nn.Module, nvidia: torch.nn.Module) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__), Path(inspect.getfile(type(native))), Path(inspect.getfile(type(nvidia)))]
    fused_module = importlib.import_module(type(nvidia).__module__)
    source_paths.extend(
        Path(inspect.getfile(operation))
        for operation in (fused_module.fused_group_norm_silu, fused_module.fused_adaptive_group_norm_silu)
    )
    source_paths.append(root / "benchmarks/kernels/hunyuan_image3_resblock_benchmarks.py")
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, capture_output=True, check=True)
    merge = subprocess.run(["git", "rev-parse", "--verify", "MERGE_HEAD"], cwd=root, capture_output=True, text=True)
    return {
        "merge_head": merge.stdout.strip() if merge.returncode == 0 else None,
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": version("triton"),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "NVIDIA_TF32_OVERRIDE",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDNN_CONV_WSCAP_DBG",
                "TORCH_CUDNN_V8_API_DISABLED",
            )
        },
    }


def diagnose(
    native: torch.nn.Module, nvidia: torch.nn.Module, inputs: torch.Tensor, embedding: torch.Tensor
) -> dict[str, object]:
    before = backend_policy()
    with torch.inference_mode(), torch.autocast(inputs.device.type, enabled=False):
        native_output = native(inputs, embedding)
        nvidia_output = nvidia(inputs, embedding)
        uninstrumented = compare(nvidia_output, native_output)
        native_output = native_output.cpu()
        nvidia_output = nvidia_output.cpu()
        reference, native_policies = capture(native, inputs, embedding, fused=False)
        candidate, nvidia_policies = capture(nvidia, inputs, embedding, fused=True)
        stages = {name: compare(candidate[name], reference[name]) for name in candidate}
        local = replay(native, nvidia, inputs, embedding, reference, native_policies["conv1"], nvidia_policies["conv1"])
        reference["scale"], reference["shift"] = reference["embedding"].chunk(2, dim=1)
    if backend_policy() != before:
        raise RuntimeError("Diagnostic did not restore backend policy")
    return {
        "scope": "Untimed eager block diagnostic; not model qualification or performance",
        "qualification_passed": uninstrumented["passed"],
        "uninstrumented_final": uninstrumented,
        "capture_repeatability": {
            "native_bitwise_equal": torch.equal(native_output, reference["final_add"]),
            "nvidia_bitwise_equal": torch.equal(nvidia_output, candidate["final_add"]),
        },
        "autocast_enabled": False,
        "propagated_stages": stages,
        "native_only_intermediates": {
            name: {
                "shape": list(reference[name].shape),
                "dtype": str(reference[name].dtype),
                "finite": bool(torch.isfinite(reference[name]).all().item()),
                "max_abs": (
                    reference[name].float().abs().max().item() if torch.isfinite(reference[name]).all().item() else None
                ),
            }
            for name in ("gn", "adaptive_gn", "modulation", "scale", "shift")
        },
        "fused_internal_values": "GN and modulation are not separately materialized by fused kernels",
        "same_native_input_replay": local,
        "policies": {"before": before, "native": native_policies, "nvidia": nvidia_policies, "after": backend_policy()},
        "rounding_mechanism_probe": rounding_probe(),
        "failure_cause": (
            "Unassigned: inspect propagated, same-input and policy-only comparisons; no automatic attribution"
        ),
    }
