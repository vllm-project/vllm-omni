# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark repeated diffusion LoRA updates.

L1 isolates the precomputed arithmetic tail. L2 calls the real manager
activation entry point with cached synthetic adapters and includes binding,
host-to-device copies, delta GEMM, cast, target planning, and commit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from vllm import __version__ as vllm_version
from vllm.lora.lora_weights import LoRALayerWeights

from vllm_omni import __version__ as vllm_omni_version
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.platforms import current_omni_platform


class _BenchLayer(torch.nn.Module):
    def __init__(self, base: torch.Tensor, rank: int):
        super().__init__()
        self.base_layer = torch.nn.Module()
        self.base_layer.weight = torch.nn.Parameter(base.clone())
        self.base_layer.quant_method = None
        out_dim, in_dim = base.shape
        self.n_slices = 1
        self.output_slices = (out_dim,)
        self.lora_a_stacked = (torch.zeros(1, 1, rank, in_dim, dtype=base.dtype, device=base.device),)
        self.lora_b_stacked = (torch.zeros(1, 1, out_dim, rank, dtype=base.dtype, device=base.device),)
        self._diffusion_lora_active_slices = (False,)

    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor) -> None:
        assert index == 0
        with torch.no_grad():
            self.lora_a_stacked[0].zero_()
            self.lora_b_stacked[0].zero_()
            self.lora_a_stacked[0][0, 0, : lora_a.shape[0]].copy_(lora_a)
            self.lora_b_stacked[0][0, 0, :, : lora_b.shape[1]].copy_(lora_b)
        self._diffusion_lora_active_slices = (True,)

    def reset_lora(self, index: int) -> None:
        assert index == 0
        with torch.no_grad():
            self.lora_a_stacked[0].zero_()
            self.lora_b_stacked[0].zero_()
        self._diffusion_lora_active_slices = (False,)


class _BenchLoRAModel:
    def __init__(self, adapter_id: int, lora: LoRALayerWeights):
        self.id = adapter_id
        self.loras = {"transformer.bench": lora}

    def get_lora(self, name: str) -> LoRALayerWeights | None:
        return self.loras.get(name)


class _ReferenceManager(DiffusionLoRAManager):
    """Benchmark-only B0: restore the old weight, then add the new delta."""

    def _merge_active_adapter(self) -> None:
        if self._merged_layer_names:
            self._unmerge_active_adapter()

        plan: list[tuple[str, _BenchLayer, torch.Tensor, torch.Tensor]] = []
        for name, layer in self._lora_modules.items():
            delta_fp32 = self._compute_layer_delta(layer)
            if delta_fp32 is None:
                continue
            weight = layer.base_layer.weight
            pristine = self._pristine_for(name, weight)
            plan.append((name, layer, pristine, delta_fp32.to(weight.dtype)))
        if not plan:
            raise ValueError("reference update produced no delta")

        with torch.no_grad():
            for _, layer, pristine, delta_low in plan:
                layer.base_layer.weight.copy_(pristine)
                layer.base_layer.weight.add_(delta_low)
        self._reset_lora_layers()
        self._merged_layer_names = {name for name, _, _, _ in plan}
        self._merged_weight_versions = {name: layer.base_layer.weight._version for name, layer, _, _ in plan}
        self._merged = True


def _make_lora(adapter_id: int, out_dim: int, in_dim: int, rank: int) -> _BenchLoRAModel:
    generator = torch.Generator(device="cpu").manual_seed(1000 + adapter_id)
    lora = LoRALayerWeights(
        module_name="bench",
        rank=rank,
        lora_alpha=rank,
        lora_a=torch.randn(rank, in_dim, generator=generator),
        lora_b=torch.randn(out_dim, rank, generator=generator),
    )
    return _BenchLoRAModel(adapter_id, lora)


def _make_manager(
    manager_type: type[DiffusionLoRAManager],
    base: torch.Tensor,
    rank: int,
    adapters: tuple[_BenchLoRAModel, _BenchLoRAModel],
) -> tuple[DiffusionLoRAManager, _BenchLayer]:
    manager = manager_type.__new__(manager_type)
    manager.pipeline = torch.nn.Module()
    manager.device = base.device
    manager.dtype = base.dtype
    manager.max_cached_adapters = 2
    manager._registered_adapters = {adapter.id: adapter for adapter in adapters}
    manager._active_adapter_id = None
    manager._adapter_scales = {}
    manager._adapter_access_order = OrderedDict()
    manager._pinned_adapters = set()
    layer = _BenchLayer(base, rank)
    manager._lora_modules = {"transformer.bench": layer}
    manager._max_lora_rank = rank
    manager._supported_lora_modules = ["bench"]
    manager._packed_modules_mapping = {}
    manager._expected_lora_modules = {"bench"}
    manager._resident_lora_device = None
    manager.merge_on_load = True
    manager._merge_enabled = True
    manager._wrappers_installed = False
    manager._suspended_adapter_id = None
    manager._merged = False
    manager._merged_layer_names = set()
    manager._pristine_weights = {}
    manager._pristine_weight_versions = {}
    manager._merged_weight_versions = {}
    return manager, layer


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def _measure(operation: Callable[[], None], device: torch.device) -> tuple[float, float]:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.accelerator.synchronize(device)
        wall_start = time.perf_counter_ns()
        start.record()
        operation()
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        return float(start.elapsed_time(end)), wall_ms

    wall_start = time.perf_counter_ns()
    operation()
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
    return wall_ms, wall_ms


def _paired_samples(
    operations: dict[str, Callable[[], None]],
    device: torch.device,
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    names = tuple(operations)
    assert len(names) == 2
    for index in range(warmup):
        order = names if index % 2 == 0 else tuple(reversed(names))
        for name in order:
            _measure(operations[name], device)

    raw: dict[str, list[dict[str, float | int]]] = {name: [] for name in names}
    for index in range(samples):
        order = names if index % 2 == 0 else tuple(reversed(names))
        for position, name in enumerate(order):
            gpu_ms, wall_ms = _measure(operations[name], device)
            raw[name].append(
                {
                    "block_id": index,
                    "position": position,
                    "gpu_ms": gpu_ms,
                    "wall_ms": wall_ms,
                    "valid": True,
                    "error": None,
                }
            )

    return {
        "warmup_per_variant": warmup,
        "samples_per_variant": samples,
        "order": "alternating paired B0/B1",
        "raw": raw,
        "summary": {
            name: {
                "gpu_ms": _summary([float(sample["gpu_ms"]) for sample in variant]),
                "wall_ms": _summary([float(sample["wall_ms"]) for sample in variant]),
            }
            for name, variant in raw.items()
        },
    }


def _run_l1(
    shape: tuple[int, int],
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    out_dim, in_dim = shape
    generator = torch.Generator(device="cpu").manual_seed(7)
    base = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    lora_a = torch.randn(rank, in_dim, generator=generator, device="cpu").to(device)
    lora_b = torch.randn(out_dim, rank, generator=generator, device="cpu").to(device)
    delta_low = (lora_b.float() @ lora_a.float()).to(dtype)
    b0_weight = torch.empty_like(base)
    b1_weight = torch.empty_like(base)
    memory_before = torch.accelerator.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.accelerator.reset_peak_memory_stats(device)

    def b0() -> None:
        with torch.no_grad():
            b0_weight.copy_(base)
            b0_weight.add_(delta_low)

    def b1() -> None:
        with torch.no_grad():
            torch.add(base, delta_low, out=b1_weight)

    measured = _paired_samples({"B0_copy_add": b0, "B1_add_out": b1}, device, warmup, samples)
    b0()
    b1()
    if device.type == "cuda":
        torch.accelerator.synchronize(device)
    measured.update(
        {
            "level": "L1_arithmetic_tail",
            "shape": list(shape),
            "rank": rank,
            "target_count": 1,
            "target_weight_bytes": base.numel() * base.element_size(),
            "byte_equal": torch.equal(b0_weight, b1_weight),
            "memory": {
                "allocated_before_bytes": memory_before,
                "peak_allocated_bytes": torch.accelerator.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "allocated_after_bytes": torch.accelerator.memory_allocated(device) if device.type == "cuda" else None,
                "scope": "shared-process L1 pair; not per-arm isolation",
            },
        }
    )
    return measured


def _run_l2(
    shape: tuple[int, int],
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    out_dim, in_dim = shape
    generator = torch.Generator(device="cpu").manual_seed(11)
    base = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    adapters = (_make_lora(1, out_dim, in_dim, rank), _make_lora(2, out_dim, in_dim, rank))
    b0_manager, b0_layer = _make_manager(_ReferenceManager, base, rank, adapters)
    b1_manager, b1_layer = _make_manager(DiffusionLoRAManager, base, rank, adapters)
    memory_before = torch.accelerator.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.accelerator.reset_peak_memory_stats(device)

    def switch(manager: DiffusionLoRAManager) -> None:
        adapter_id = 2 if manager._active_adapter_id == 1 else 1
        manager._activate_adapter(adapter_id, scale=1.0)

    measured = _paired_samples(
        {
            "B0_restore_merge_manager": lambda: switch(b0_manager),
            "B1_direct_manager": lambda: switch(b1_manager),
        },
        device,
        warmup,
        samples,
    )
    if device.type == "cuda":
        torch.accelerator.synchronize(device)
    same_active = b0_manager._active_adapter_id == b1_manager._active_adapter_id
    byte_equal = same_active and torch.equal(b0_layer.base_layer.weight, b1_layer.base_layer.weight)
    b0_manager._deactivate_all_adapters()
    b1_manager._deactivate_all_adapters()
    if device.type == "cuda":
        torch.accelerator.synchronize(device)
    measured.update(
        {
            "level": "L2_synthetic_manager_update",
            "shape": list(shape),
            "rank": rank,
            "target_count": 1,
            "target_weight_bytes": base.numel() * base.element_size(),
            "old_new_target_overlap": 1.0,
            "cache_capacity": 2,
            "update_sequence": "add old, add new, alternate immutable IDs, remove after switch",
            "byte_equal": byte_equal,
            "base_restore_byte_equal": torch.equal(b0_layer.base_layer.weight, base)
            and torch.equal(b1_layer.base_layer.weight, base),
            "memory": {
                "allocated_before_bytes": memory_before,
                "peak_allocated_bytes": torch.accelerator.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "allocated_after_bytes": torch.accelerator.memory_allocated(device) if device.type == "cuda" else None,
                "scope": "shared-process L2 pair; not per-arm isolation",
            },
        }
    )
    return measured


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes = []
    for item in value.split(","):
        out_dim, separator, in_dim = item.lower().partition("x")
        if not separator:
            raise argparse.ArgumentTypeError(f"invalid shape {item!r}; expected OUTxIN")
        shapes.append((int(out_dim), int(in_dim)))
    return shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--shapes", type=_parse_shapes, default=_parse_shapes("1024x1024,3584x3584"))
    parser.add_argument("--ranks", type=lambda value: [int(item) for item in value.split(",")], default=[16, 64])
    parser.add_argument("--level", choices=("l1", "l2", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment-id", default="lora-direct-remerge")
    parser.add_argument("--source-sha", default="UNRESOLVED")
    parser.add_argument("--dependency-sha", default="UNRESOLVED")
    parser.add_argument("--upstream-main-sha", default="UNRESOLVED")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", 0)
        current_omni_platform.set_device(device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("CPU float16 GEMM is not a supported benchmark configuration")
    if args.warmup < 0 or args.samples < 2:
        raise ValueError("warmup must be non-negative and samples must be at least 2")

    results = []
    for shape in args.shapes:
        for rank in args.ranks:
            if args.level in ("l1", "both"):
                results.append(_run_l1(shape, rank, dtype, device, args.warmup, args.samples))
            if args.level in ("l2", "both"):
                results.append(_run_l2(shape, rank, dtype, device, args.warmup, args.samples))

    timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "session_id": f"{args.experiment_id}-{timestamp_utc}",
        "timestamp_utc": timestamp_utc,
        "source": {
            "patch_sha": args.source_sha,
            "dependency_sha": args.dependency_sha,
            "upstream_main_sha": args.upstream_main_sha,
        },
        "environment": {
            "controller_pid": os.getpid(),
            "worker_pids": [],
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "vllm": vllm_version,
            "vllm_omni": vllm_omni_version,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": args.dtype,
            "tp": 1,
        },
        "notes": [
            "L1 excludes delta GEMM and transfer.",
            "L2 is a synthetic one-layer cached-adapter manager update, not model E2E.",
            "GPU event time covers the default stream; wall time includes synchronization.",
        ],
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
