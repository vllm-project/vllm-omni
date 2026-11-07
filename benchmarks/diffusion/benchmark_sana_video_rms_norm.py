# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark SANA-Video's TP=1 distributed RMSNorm fast path.

This benchmark deliberately measures the sum/count expression used by
``SanaDistributedRMSNorm``.  The older ``exact_sana_rms_norm`` helper keeps a
mean reduction for ``SanaRMSNorm`` and is not the video-token production path.

The operator presets are the two released SANA-Video 2B profiles with standard
classifier-free guidance:

* 480p: 81 output frames -> 21x60x104 latents -> 32,760 patched tokens;
* 720p: 81 output frames -> 11x22x40 latents -> 9,680 patched tokens.

Examples:

.. code-block:: bash

    python benchmarks/diffusion/benchmark_sana_video_rms_norm.py operator
    python benchmarks/diffusion/benchmark_sana_video_rms_norm.py operator \
        --preset 720p --expected-head <commit>
    python benchmarks/diffusion/benchmark_sana_video_rms_norm.py transformer \
        --preset 480p --samples 5 --warmups 1 --json-output result.json

The transformer mode downloads the checkpoint's transformer component by
default, loads its weights into vLLM-Omni's native
``SanaVideoTransformer3DModel``, and runs the full native forward without an
engine.  Pass ``--random-weights`` only for a routing smoke test.

Every qualification compares raw BF16 storage.  A signature that silently
falls back, becomes disabled, or produces different output terminates the run
instead of reporting a performance number.

The reported dispatch count observes calls to
``_launch_exact_sana_rms_norm_sum``.  One such fast-op dispatch currently
contains two Triton kernels plus ATen reduction/arithmetic operations, so it is
not a count of GPU kernel launches.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import vllm_omni.diffusion.layers.sana_rms_norm as sana_rms
from vllm_omni.platforms import current_omni_platform

_PRESETS: dict[str, dict[str, Any]] = {
    "480p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        "operator_shape": (2, 32760, 2240),
        "latent_shape": (2, 16, 21, 60, 104),
        "caption_shape": (2, 300, 2304),
        "expected_video_tokens": 32760,
    },
    "720p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        "operator_shape": (2, 9680, 2240),
        "latent_shape": (2, 128, 11, 22, 40),
        "caption_shape": (2, 300, 2304),
        "expected_video_tokens": 9680,
    },
}


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    samples: int


def _raw_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype is torch.bfloat16:
        return torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16))
    if left.dtype is torch.float32:
        return torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))
    raise TypeError(f"Raw comparison is not implemented for {left.dtype}")


def _sum_count_reference(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Current TP=1 ``SanaDistributedRMSNorm.forward`` expression."""
    x_float = hidden_states.float()
    sum_sq = x_float.pow(2).sum(dim=-1, keepdim=True)
    normalized = x_float * torch.rsqrt(sum_sq / hidden_states.shape[-1] + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        normalized = normalized.to(weight.dtype)
    return normalized * weight


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _changed_python_hashes() -> tuple[list[str], dict[str, str]]:
    tracked = _command_output(["git", "diff", "--name-only", "--diff-filter=ACMRTUXB"]) or ""
    staged = _command_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB"]) or ""
    untracked = _command_output(["git", "ls-files", "--others", "--exclude-standard"]) or ""
    paths = sorted(
        {
            path
            for output in (tracked, staged, untracked)
            for path in output.splitlines()
            if path.endswith(".py") and Path(path).is_file()
        }
    )
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}
    return paths, hashes


def _metadata(expected_head: str | None, device_index: int) -> dict[str, Any]:
    head = _command_output(["git", "rev-parse", "HEAD"])
    if expected_head is not None and head != expected_head:
        raise RuntimeError(f"Expected git head {expected_head}, found {head}")

    status = _command_output(["git", "status", "--short", "--untracked-files=all"]) or ""
    changed_python, changed_python_sha256 = _changed_python_hashes()
    capability = current_omni_platform.get_device_capability(device_index)
    driver = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            f"--id={device_index}",
        ]
    )
    return {
        "git_head": head,
        "expected_head": expected_head,
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "changed_python_files": changed_python,
        "changed_python_sha256": changed_python_sha256,
        "command": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": current_omni_platform.get_device_name(device_index),
        "compute_capability": str(capability) if capability is not None else None,
        "gpu_total_memory_bytes": current_omni_platform.get_device_total_memory(device_index),
        "driver": driver,
        "triton": _version("triton"),
        "vllm": _version("vllm"),
        "vllm_omni": _version("vllm_omni"),
        "diffusers": _version("diffusers"),
        "transformers": _version("transformers"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _require_cuda_fast_path() -> None:
    if not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() < 1:
        raise RuntimeError("NVIDIA CUDA is required")
    if not getattr(sana_rms, "HAS_TRITON", False):
        raise RuntimeError("Triton is required")
    required = (
        "exact_sana_rms_norm_sum",
        "_launch_exact_sana_rms_norm_sum",
        "_VERIFIED_SUM_SIGNATURES",
        "_DISABLED_SUM_SIGNATURES",
    )
    missing = [name for name in required if not hasattr(sana_rms, name)]
    if missing:
        raise RuntimeError(
            f"This benchmark requires the TP=1 sum/count fast-path patch; missing symbols: {', '.join(missing)}"
        )


def _summary(samples: list[float]) -> TimingSummary:
    if not samples:
        raise ValueError("No timing samples")
    return TimingSummary(
        median_ms=statistics.median(samples),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
        samples=len(samples),
    )


def _cuda_time(call: Callable[[], torch.Tensor]) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), output


def _balanced_times(
    baseline: Callable[[], torch.Tensor],
    fast: Callable[[], torch.Tensor],
    *,
    warmups: int,
    samples: int,
) -> tuple[TimingSummary, TimingSummary]:
    for index in range(warmups):
        calls = (baseline, fast) if index % 2 == 0 else (fast, baseline)
        for call in calls:
            call()
    current_omni_platform.synchronize()

    baseline_ms: list[float] = []
    fast_ms: list[float] = []
    for index in range(samples):
        calls = (("baseline", baseline), ("fast", fast))
        if index % 2:
            calls = tuple(reversed(calls))
        for name, call in calls:
            elapsed, _ = _cuda_time(call)
            (baseline_ms if name == "baseline" else fast_ms).append(elapsed)
    return _summary(baseline_ms), _summary(fast_ms)


def _balanced_transformer_times(
    forward: Callable[[], torch.Tensor],
    *,
    warmups: int,
    samples: int,
) -> tuple[TimingSummary, TimingSummary]:
    """Time full forwards while changing the A/B route outside CUDA events."""
    for index in range(warmups):
        arms = ("baseline", "fast") if index % 2 == 0 else ("fast", "baseline")
        for arm in arms:
            route = _baseline_distributed_route() if arm == "baseline" else contextlib.nullcontext()
            with route:
                forward()
    current_omni_platform.synchronize()

    baseline_ms: list[float] = []
    fast_ms: list[float] = []
    for index in range(samples):
        arms = ("baseline", "fast") if index % 2 == 0 else ("fast", "baseline")
        for arm in arms:
            route = _baseline_distributed_route() if arm == "baseline" else contextlib.nullcontext()
            with route:
                elapsed, _ = _cuda_time(forward)
            (baseline_ms if arm == "baseline" else fast_ms).append(elapsed)
    return _summary(baseline_ms), _summary(fast_ms)


def _peak_allocated(call: Callable[[], torch.Tensor], device: torch.device) -> int:
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats(device)
    output = call()
    current_omni_platform.synchronize()
    peak = torch.accelerator.max_memory_allocated(device)
    del output
    return peak


@contextlib.contextmanager
def _count_sum_launches() -> Iterator[Counter[tuple[int, ...]]]:
    original = sana_rms._launch_exact_sana_rms_norm_sum
    counts: Counter[tuple[int, ...]] = Counter()

    def counted(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        counts[tuple(hidden_states.shape)] += 1
        return original(hidden_states, weight, eps)

    sana_rms._launch_exact_sana_rms_norm_sum = counted
    try:
        yield counts
    finally:
        sana_rms._launch_exact_sana_rms_norm_sum = original


def _sum_signature(shape: tuple[int, ...], device: torch.device) -> Any:
    return device, torch.bfloat16, math.prod(shape[:-1]), shape[-1]


def _assert_fast_signature(shape: tuple[int, ...], device: torch.device) -> None:
    signature = _sum_signature(shape, device)
    verified = sana_rms._VERIFIED_SUM_SIGNATURES
    disabled = sana_rms._DISABLED_SUM_SIGNATURES
    if signature not in verified or signature in disabled:
        raise RuntimeError(
            "The sum/count signature did not qualify for the fast path: "
            f"signature={signature!r}, verified={signature in verified}, disabled={signature in disabled}"
        )


def _assert_no_disabled_sum_signatures() -> None:
    if sana_rms._DISABLED_SUM_SIGNATURES:
        raise RuntimeError(f"Sum/count signatures were disabled: {sana_rms._DISABLED_SUM_SIGNATURES!r}")


@torch.no_grad()
def _operator_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    preset = _PRESETS[args.preset]
    shape = tuple(args.shape) if args.shape is not None else preset["operator_shape"]
    torch.manual_seed(args.seed)
    hidden_states = torch.randn(shape, device=args.device_obj, dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], device=args.device_obj, dtype=torch.bfloat16)
    eps = args.eps

    def baseline() -> torch.Tensor:
        return _sum_count_reference(hidden_states, weight, eps)

    def fast() -> torch.Tensor:
        return sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, eps)

    sana_rms._VERIFIED_SUM_SIGNATURES.clear()
    sana_rms._DISABLED_SUM_SIGNATURES.clear()
    expected = baseline()
    with _count_sum_launches() as qualification_launches:
        actual = fast()
    if not _raw_equal(actual, expected):
        raise RuntimeError("TP=1 sum/count fast path is not raw-BF16 identical to the production expression")
    _assert_fast_signature(shape, args.device_obj)
    if sum(qualification_launches.values()) != 1:
        raise RuntimeError(f"Qualification expected one sum/count fast-op dispatch, got {dict(qualification_launches)}")
    del actual, expected

    with _count_sum_launches() as timed_launches:
        baseline_timing, fast_timing = _balanced_times(
            baseline,
            fast,
            warmups=args.warmups,
            samples=args.samples,
        )
    expected_fast_launches = args.warmups + args.samples
    if sum(timed_launches.values()) != expected_fast_launches:
        raise RuntimeError(
            f"Expected {expected_fast_launches} timed/warmup sum/count fast-op dispatches, got {dict(timed_launches)}"
        )
    _assert_fast_signature(shape, args.device_obj)

    final_expected = baseline()
    final_fast = fast()
    if not _raw_equal(final_fast, final_expected):
        raise RuntimeError("Steady-state TP=1 fast output changed after timing")
    del final_expected, final_fast
    _assert_no_disabled_sum_signatures()

    baseline_peak = _peak_allocated(baseline, args.device_obj)
    fast_peak = _peak_allocated(fast, args.device_obj)
    _assert_fast_signature(shape, args.device_obj)
    _assert_no_disabled_sum_signatures()

    baseline_data = asdict(baseline_timing)
    baseline_data["peak_allocated_bytes"] = baseline_peak
    fast_data = asdict(fast_timing)
    fast_data["peak_allocated_bytes"] = fast_peak
    return {
        "mode": "operator",
        "preset": args.preset,
        "shape": shape,
        "eps": eps,
        "baseline": baseline_data,
        "fast": fast_data,
        "speedup": baseline_timing.median_ms / fast_timing.median_ms,
        "dispatch_definition": "Calls to _launch_exact_sana_rms_norm_sum; not GPU kernel launches",
        "qualification_sum_op_dispatches": {str(key): value for key, value in qualification_launches.items()},
        "timed_and_warmup_sum_op_dispatches": {str(key): value for key, value in timed_launches.items()},
        "raw_bf16_equal": True,
    }


def _load_native_transformer(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        SanaVideoTransformer3DModel,
        SanaVideoTransformerConfig,
    )

    preset = _PRESETS[args.preset]
    model_name = args.model or preset["model"]
    if args.random_weights:
        config = {
            "in_channels": preset["latent_shape"][1],
            "out_channels": preset["latent_shape"][1],
            "patch_size": (1, 2, 2) if args.preset == "480p" else (1, 1, 1),
            "sample_size": 30 if args.preset == "480p" else 22,
            "mlp_ratio": 3.0,
        }
        torch.manual_seed(args.seed)
        model = SanaVideoTransformer3DModel(**config).to(device=args.device_obj, dtype=torch.bfloat16).eval()
        return model, {
            "source": "random_weights",
            "requested_model": model_name,
            "requested_revision": None,
            "resolved_commit": None,
        }

    try:
        from diffusers import SanaVideoTransformer3DModel as DiffusersSanaVideoTransformer3DModel
    except ImportError as error:
        raise RuntimeError("Transformer mode needs Diffusers to load the released checkpoint") from error

    local_path = Path(model_name).expanduser()
    subfolder = None if local_path.is_dir() and (local_path / "config.json").is_file() else "transformer"
    reference = DiffusersSanaVideoTransformer3DModel.from_pretrained(
        str(local_path) if local_path.exists() else model_name,
        subfolder=subfolder,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    checkpoint = {
        "source": "pretrained",
        "requested_model": model_name,
        "requested_revision": args.revision,
        "resolved_commit": getattr(reference.config, "_commit_hash", None),
        "resolved_name_or_path": getattr(reference.config, "_name_or_path", None),
        "subfolder": subfolder,
    }
    config_keys = SanaVideoTransformerConfig.__dataclass_fields__
    config = {key: getattr(reference.config, key) for key in config_keys}
    model = SanaVideoTransformer3DModel.from_config(config).to(device=args.device_obj, dtype=torch.bfloat16).eval()
    incompatible = model.load_state_dict(reference.state_dict(), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/native transformer state dictionaries differ: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    del reference
    return model, checkpoint


def _validate_transformer_preset(model: torch.nn.Module, preset_name: str) -> tuple[int, ...]:
    preset = _PRESETS[preset_name]
    latent_shape = preset["latent_shape"]
    expected_patch_size = (1, 2, 2) if preset_name == "480p" else (1, 1, 1)
    config = model.config
    actual_patch_size = tuple(config.patch_size)
    hidden_size = config.num_attention_heads * config.attention_head_dim
    mismatches = []
    if (
        len(actual_patch_size) != 3
        or any(size <= 0 for size in actual_patch_size)
        or any(latent_shape[dimension] % actual_patch_size[dimension - 2] for dimension in range(2, 5))
    ):
        mismatches.append(f"patch_size={actual_patch_size} does not divide latent shape {latent_shape[2:]}")
        patched_tokens = -1
    else:
        patched_tokens = math.prod(
            latent_shape[dimension] // actual_patch_size[dimension - 2] for dimension in range(2, 5)
        )
    if config.in_channels != latent_shape[1]:
        mismatches.append(f"in_channels={config.in_channels}, expected {latent_shape[1]}")
    if actual_patch_size != expected_patch_size:
        mismatches.append(f"patch_size={actual_patch_size}, expected {expected_patch_size}")
    if hidden_size != 2240:
        mismatches.append(f"hidden_size={hidden_size}, expected 2240")
    if config.num_layers != 20:
        mismatches.append(f"num_layers={config.num_layers}, expected 20")
    if config.caption_channels != preset["caption_shape"][-1]:
        mismatches.append(f"caption_channels={config.caption_channels}, expected {preset['caption_shape'][-1]}")
    if patched_tokens != preset["expected_video_tokens"]:
        mismatches.append(f"patched_tokens={patched_tokens}, expected {preset['expected_video_tokens']}")
    if mismatches:
        raise RuntimeError(f"Checkpoint does not match the {preset_name} production preset: " + "; ".join(mismatches))
    return latent_shape[0], patched_tokens, hidden_size


def _capture_norm_inputs(model: torch.nn.Module) -> tuple[list[Any], Counter[tuple[str, tuple[int, ...]]]]:
    from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
        SanaDistributedRMSNorm,
        SanaRMSNorm,
    )

    handles: list[Any] = []
    calls: Counter[tuple[str, tuple[int, ...]]] = Counter()
    for name, module in model.named_modules():
        if not isinstance(module, (SanaDistributedRMSNorm, SanaRMSNorm)):
            continue

        def record(_module: torch.nn.Module, inputs: tuple[Any, ...], *, qualified_name: str = name) -> None:
            calls[(qualified_name, tuple(inputs[0].shape))] += 1

        handles.append(module.register_forward_pre_hook(record))
    return handles, calls


@contextlib.contextmanager
def _baseline_distributed_route() -> Iterator[None]:
    """Replace only the transformer's imported sum helper for an A/B arm."""
    import vllm_omni.diffusion.models.sana_video.transformer_sana_video as sana_transformer

    if not hasattr(sana_transformer, "exact_sana_rms_norm_sum"):
        raise RuntimeError("SanaDistributedRMSNorm is not routed to exact_sana_rms_norm_sum")
    original = sana_transformer.exact_sana_rms_norm_sum
    sana_transformer.exact_sana_rms_norm_sum = _sum_count_reference
    try:
        yield
    finally:
        sana_transformer.exact_sana_rms_norm_sum = original


@contextlib.contextmanager
def _single_rank_transformer_runtime(args: argparse.Namespace) -> Iterator[None]:
    from vllm.utils.network_utils import get_open_port

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        DiffusionParallelConfig,
        OmniDiffusionConfig,
    )
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if model_parallel_is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError("Transformer benchmark must start before a distributed environment is initialized")

    saved_env = {
        name: os.environ.get(name) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    }
    port = get_open_port()
    os.environ.update(
        {
            "RANK": "0",
            "LOCAL_RANK": str(args.device),
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark
    try:
        current_omni_platform.set_device(args.device_obj)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=args.device,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        initialize_model_parallel(tensor_parallel_size=1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        parallel_config = DiffusionParallelConfig(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            tensor_parallel_size=1,
            sequence_parallel_size=1,
            ulysses_degree=1,
            ring_degree=1,
            cfg_parallel_size=1,
        )
        od_config = OmniDiffusionConfig(
            model=args.model or _PRESETS[args.preset]["model"],
            dtype=torch.bfloat16,
            parallel_config=parallel_config,
            diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
        )
        with set_current_diffusion_config(od_config):
            yield
    finally:
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.benchmark = old_benchmark
        destroy_distributed_env()
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _transformer_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from vllm.distributed import get_tensor_model_parallel_world_size

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size != 1:
        raise RuntimeError(f"Transformer benchmark covers only the TP=1 fast path, got TP={tp_size}")

    preset = _PRESETS[args.preset]
    model, checkpoint = _load_native_transformer(args)
    latent_shape = preset["latent_shape"]
    caption_shape = preset["caption_shape"]
    expected_shape = _validate_transformer_preset(model, args.preset)
    torch.manual_seed(args.seed)
    hidden_states = torch.randn(latent_shape, device=args.device_obj, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(caption_shape, device=args.device_obj, dtype=torch.bfloat16)
    encoder_attention_mask = torch.ones(caption_shape[:2], device=args.device_obj, dtype=torch.bool)
    timestep = torch.full((latent_shape[0],), 500.0, device=args.device_obj)

    def forward() -> torch.Tensor:
        return model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=encoder_attention_mask,
        ).sample

    sana_rms._VERIFIED_SUM_SIGNATURES.clear()
    sana_rms._DISABLED_SUM_SIGNATURES.clear()
    handles, norm_calls = _capture_norm_inputs(model)
    try:
        with torch.no_grad(), _count_sum_launches() as qualification_launches:
            fast_output = forward()
    finally:
        for handle in handles:
            handle.remove()

    with torch.no_grad(), _baseline_distributed_route():
        baseline_output = forward()
    if not _raw_equal(fast_output, baseline_output):
        raise RuntimeError("Full native transformer output differs in raw BF16 storage")
    del fast_output, baseline_output
    if sum(qualification_launches.values()) != 60:
        raise RuntimeError(
            "One 20-block SANA transformer forward must dispatch 60 video-token RMSNorm sum/count fast ops; "
            f"got {dict(qualification_launches)}"
        )
    if qualification_launches[expected_shape] != 60:
        raise RuntimeError(
            f"Expected 60 sum/count fast-op dispatches at real video shape {expected_shape}, "
            f"got {dict(qualification_launches)}"
        )
    for shape in qualification_launches:
        _assert_fast_signature(shape, args.device_obj)

    with torch.no_grad(), _count_sum_launches() as timed_launches:
        baseline_timing, fast_timing = _balanced_transformer_times(
            forward,
            warmups=args.warmups,
            samples=args.samples,
        )
    expected_launches = 60 * (args.warmups + args.samples)
    if sum(timed_launches.values()) != expected_launches:
        raise RuntimeError(
            f"Expected {expected_launches} full-model sum/count fast-op dispatches, got {dict(timed_launches)}"
        )

    with torch.no_grad():
        final_fast = forward()
        with _baseline_distributed_route():
            final_baseline = forward()
        if not _raw_equal(final_fast, final_baseline):
            raise RuntimeError("Full native transformer output changed after timing")
        del final_fast, final_baseline
        for shape in timed_launches:
            _assert_fast_signature(shape, args.device_obj)
        _assert_no_disabled_sum_signatures()
        with _baseline_distributed_route():
            baseline_peak = _peak_allocated(forward, args.device_obj)
        fast_peak = _peak_allocated(forward, args.device_obj)
        for shape in timed_launches:
            _assert_fast_signature(shape, args.device_obj)
        _assert_no_disabled_sum_signatures()

    aggregated_norm_calls: Counter[tuple[str, tuple[int, ...]]] = Counter()
    for (name, shape), count in norm_calls.items():
        module_kind = "caption" if name == "caption_norm" else "distributed"
        aggregated_norm_calls[(module_kind, shape)] += count

    return {
        "mode": "transformer",
        "preset": args.preset,
        "model": args.model or preset["model"],
        "pretrained": not args.random_weights,
        "checkpoint": checkpoint,
        "latent_shape": latent_shape,
        "caption_shape": caption_shape,
        "baseline": {**asdict(baseline_timing), "peak_allocated_bytes": baseline_peak},
        "fast": {**asdict(fast_timing), "peak_allocated_bytes": fast_peak},
        "speedup": baseline_timing.median_ms / fast_timing.median_ms,
        "dispatch_definition": "Calls to _launch_exact_sana_rms_norm_sum; not GPU kernel launches",
        "qualification_sum_op_dispatches": {str(key): value for key, value in qualification_launches.items()},
        "timed_and_warmup_sum_op_dispatches": {str(key): value for key, value in timed_launches.items()},
        "norm_calls_in_one_forward": {
            f"{kind} {shape}": count for (kind, shape), count in sorted(aggregated_norm_calls.items(), key=str)
        },
        "raw_bf16_equal": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("operator", "transformer"))
    parser.add_argument("--preset", choices=tuple(_PRESETS), default="480p")
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index")
    parser.add_argument("--expected-head", help="Fail unless git HEAD is exactly this commit")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--shape", type=int, nargs=3, metavar=("BATCH", "TOKENS", "HIDDEN"))
    parser.add_argument("--model", help="Hugging Face model id or local checkpoint directory")
    parser.add_argument("--revision", help="Hugging Face checkpoint revision (commit SHA recommended)")
    parser.add_argument("--random-weights", action="store_true", help="Skip checkpoint loading (routing smoke only)")
    args = parser.parse_args()
    if args.shape is not None and args.mode != "operator":
        parser.error("--shape is valid only in operator mode")
    if args.random_weights and args.mode != "transformer":
        parser.error("--random-weights is valid only in transformer mode")
    if args.random_weights and args.revision is not None:
        parser.error("--revision cannot be combined with --random-weights")
    if args.mode == "operator" and (args.model is not None or args.revision is not None):
        parser.error("--model and --revision are valid only in transformer mode")
    if args.warmups is None:
        args.warmups = 20 if args.mode == "operator" else 1
    if args.samples is None:
        args.samples = 32 if args.mode == "operator" else 5
    if args.warmups < 0 or args.samples < 1:
        parser.error("--warmups must be >= 0 and --samples must be >= 1")
    return args


def main() -> None:
    args = _parse_args()
    _require_cuda_fast_path()
    if args.device < 0 or args.device >= current_omni_platform.get_device_count():
        visible_devices = current_omni_platform.get_device_count()
        raise RuntimeError(f"CUDA device {args.device} is unavailable; visible device count is {visible_devices}")
    if args.mode == "transformer" and args.device != 0:
        raise RuntimeError(
            "Single-rank transformer mode requires visible CUDA device index 0; select a physical GPU with "
            "CUDA_VISIBLE_DEVICES"
        )
    args.device_obj = current_omni_platform.get_torch_device(args.device)
    current_omni_platform.set_device(args.device_obj)
    metadata = _metadata(args.expected_head, args.device)
    if args.mode == "operator":
        result = _operator_benchmark(args)
    else:
        with _single_rank_transformer_runtime(args):
            result = _transformer_benchmark(args)
    report = {
        "metadata": metadata,
        "result": result,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
