# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Cosmos3's Wan encoder against the reference on H100/GB200.

Examples::

    python benchmarks/diffusion/bench_wan_vae_encode.py --frames 1 --profile
    python benchmarks/diffusion/bench_wan_vae_encode.py --frames 93 --check-reconstruction --json results.json
    python benchmarks/diffusion/bench_wan_vae_encode.py --input pixels.pt --check-reconstruction
    torchrun --nproc-per-node 2 benchmarks/diffusion/bench_wan_vae_encode.py --vae-patch-parallel-size 2

``--input`` accepts preprocessed natural images/videos saved as a floating-point
RGB tensor [B, 3, T, H, W] in [-1, 1]. Input loading/transfers, JIT compilation,
quality checks and optional reference-decoder reconstruction are outside timing.
The default input is seeded uniform noise. The reference is always set up and
validated first. Console output uses timing and quality tables; ``--json``
saves full metrics and per-level failure reasons.

GPU clocks depend on the thermal and power history, and power-limited boards
(H100 NVL, RTX 5090, RTX PRO 6000) clock the same kernels down by more than 2x
under sustained load. By default (``--schedule interleaved``) every level's VAE
stays resident and one encode per level runs in each round, with the first
level rotating from round to round, so all levels are timed under the same
conditions. ``--schedule sequential`` times one level after another, loading
one VAE at a time. NVML samples SM clock, power, temperature and clock-event
(throttle) reasons during timed encodes; ``--lock-sm-clock-mhz`` pins the SM
clock (requires administrator rights) and restores the default on exit.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
import json
import math
import os
import platform
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
    OmniAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    VAE_FAST_PATH_LEVELS,
    install_wan_vae_encoder_fastpath,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath._utils import encoder_nrmse_limit
from vllm_omni.platforms import current_omni_platform

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
SCHEDULES = ("interleaved", "sequential")
MIN_RECONSTRUCTION_PSNR_DB = 50.0
# NVML clock-event ("throttle") reason bits, named as in ``nvidia-smi -q``.
CLOCK_EVENT_REASONS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
    0x100: "display_clocks",
}
# Reasons that lower the clock below what the application requested.
THROTTLE_REASONS = ("sw_power_cap", "hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake")
# Relative spread of median SM clocks across levels above which speedups are flagged.
CLOCK_SPREAD_WARNING = 0.03
TINY_CONFIG = dict(
    base_dim=20,
    decoder_base_dim=32,
    z_dim=48,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    temperal_downsample=[False, True, True],
    is_residual=True,
    patch_size=2,
    in_channels=12,
    out_channels=12,
    scale_factor_temporal=4,
    scale_factor_spatial=16,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--subfolder", default="vae")
    parser.add_argument("--revision", default=None, help="Pin the HF checkpoint revision")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--size", default="1280x720", help="RGB input WxH")
    parser.add_argument("--frames", type=int, default=93)
    parser.add_argument("--input", type=Path, help="Preprocessed RGB .pt tensor; overrides size/frames")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--fast-path", default=",".join(VAE_FAST_PATH_LEVELS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tf32", choices=("on", "off"), default="on")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-rows", type=int, default=30)
    parser.add_argument("--check-reconstruction", action="store_true", help="Require reference-decode PSNR >= 50 dB")
    parser.add_argument("--json", type=Path, help="Write timings, environment and quality metrics")
    parser.add_argument("--tiling", action="store_true")
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1)
    parser.add_argument(
        "--schedule",
        choices=SCHEDULES,
        default="interleaved",
        help="interleaved: all levels resident, timed round-robin (default); sequential: one level at a time",
    )
    parser.add_argument(
        "--telemetry-interval-ms",
        type=float,
        default=50.0,
        help="NVML sampling period during timed encodes; 0 disables telemetry",
    )
    parser.add_argument(
        "--lock-sm-clock-mhz",
        type=int,
        default=None,
        help="Lock the SM clock for the run and reset it on exit (requires administrator rights)",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1 or args.vae_patch_parallel_size < 1:
        parser.error("warmup must be nonnegative; iters and parallel size must be positive")
    if args.telemetry_interval_ms < 0 or (args.lock_sm_clock_mhz is not None and args.lock_sm_clock_mhz < 1):
        parser.error("telemetry interval must be nonnegative and the locked SM clock positive")
    levels = list(dict.fromkeys(args.fast_path.split(",")))
    if any(level not in VAE_FAST_PATH_LEVELS for level in levels):
        parser.error(f"fast-path levels must be in {VAE_FAST_PATH_LEVELS}")
    args.levels = ["off", *(level for level in levels if level != "off")]
    return args


def make_pixels(args):
    if args.input:
        pixels = torch.load(args.input, map_location="cpu", weights_only=True)
    else:
        width, height = map(int, args.size.lower().split("x"))
        generator = torch.Generator().manual_seed(args.seed)
        pixels = torch.rand((1, 3, args.frames, height, width), generator=generator) * 2 - 1
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 5 or pixels.shape[1] != 3:
        raise ValueError("input must be a [B, 3, T, H, W] RGB tensor")
    if not pixels.is_floating_point() or pixels.numel() == 0 or not torch.isfinite(pixels).all():
        raise ValueError("input must contain finite floating-point values")
    if pixels.min() < -1 or pixels.max() > 1:
        raise ValueError("input pixels must be normalized to [-1, 1]")
    if (pixels.shape[2] - 1) % 4 or pixels.shape[3] % 16 or pixels.shape[4] % 16:
        raise ValueError("Cosmos3 inputs require 1+4k frames and spatial dimensions divisible by 16")
    return pixels


def distributed_setup(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != args.vae_patch_parallel_size:
        raise ValueError("torchrun world size must equal --vae-patch-parallel-size")
    if world == 1:
        current_omni_platform.set_device(current_omni_platform.get_torch_device(0))
        return 0
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    current_omni_platform.set_device(current_omni_platform.get_torch_device(local_rank))
    init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank)
    initialize_model_parallel(sequence_parallel_size=world, ulysses_degree=world)
    return rank


def sync():
    torch.accelerator.synchronize()
    if dist.is_initialized():
        dist.barrier()


def max_across_ranks(value):
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def _nvml_handle(nvml, device_index):
    """NVML handle of a torch CUDA device; PCI ids are immune to CUDA_VISIBLE_DEVICES remapping."""
    props = torch.cuda.get_device_properties(device_index)
    if hasattr(props, "pci_bus_id"):
        bus_id = f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:{props.pci_device_id:02X}.0"
        return nvml.nvmlDeviceGetHandleByPciBusId(bus_id.encode())
    return nvml.nvmlDeviceGetHandleByIndex(torch.cuda._get_nvml_device_index(device_index))


class GpuTelemetry:
    """NVML clocks, power, temperature and clock-event reasons, sampled in a background thread.

    Samples are attributed to the ``(level, timed iteration)`` running at the
    time; warmup, installation and validation are never sampled. A query the
    device does not support is recorded as missing. Telemetry is optional:
    without NVML, ``available`` is false and ``error`` says why.
    """

    def __init__(self, interval_s: float, device_index: int | None = None, nvml: Any = None):
        self.interval_s = interval_s
        self.error: str | None = None
        self.locked_sm_clock_mhz: int | None = None
        self._nvml = nvml
        self._handle = None
        self._label: tuple[str, int] | None = None
        self._samples: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            if self._nvml is None:
                from vllm.utils.import_utils import import_pynvml

                self._nvml = import_pynvml()
            self._nvml.nvmlInit()
            index = torch.accelerator.current_device_index() if device_index is None else device_index
            self._handle = _nvml_handle(self._nvml, index)
        except Exception as exc:  # NVML is optional; report why it is missing.
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._handle is not None

    @staticmethod
    def _query(fn):
        try:
            return fn()
        except Exception:
            return None

    def _read(self) -> dict[str, Any]:
        nvml, handle = self._nvml, self._handle
        reasons = getattr(nvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or getattr(
            nvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None
        )
        return dict(
            sm_clock_mhz=self._query(lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)),
            memory_clock_mhz=self._query(lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)),
            power_w=self._query(lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000),
            temperature_c=self._query(lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)),
            clock_event_reasons=self._query(lambda: reasons(handle)),
        )

    def device_info(self) -> dict[str, Any]:
        if not self.available:
            return dict(available=False, error=self.error)
        nvml, handle = self._nvml, self._handle
        return dict(
            available=True,
            interval_ms=self.interval_s * 1000,
            locked_sm_clock_mhz=self.locked_sm_clock_mhz,
            power_limit_w=self._query(lambda: nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000),
            max_sm_clock_mhz=self._query(lambda: nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_SM)),
            max_memory_clock_mhz=self._query(lambda: nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_MEM)),
        )

    def lock_sm_clock(self, mhz: int) -> None:
        if not self.available:
            raise RuntimeError(f"NVML is unavailable ({self.error})")
        self._nvml.nvmlDeviceSetGpuLockedClocks(self._handle, mhz, mhz)
        self.locked_sm_clock_mhz = mhz

    def start(self) -> None:
        if self.available and self.interval_s > 0 and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def sample(self) -> None:
        """Record one sample for the timed encode in progress; a no-op between timed encodes."""
        label = self._label
        if label is None:
            return
        sample = self._read()
        with self._lock:
            # Drop a sample that straddled the end of the encode it started in.
            if self._label == label:
                self._samples[label].append(sample)

    @contextlib.contextmanager
    def measure(self, level: str, iteration: int):
        self._label = (level, iteration)
        try:
            yield
        finally:
            self._label = None

    def summary(self, level: str) -> dict[str, Any]:
        """Aggregate one level's samples; ``sm_clock_mhz_per_iter`` is ``None`` for unsampled iterations."""
        with self._lock:
            groups = {iteration: list(s) for (name, iteration), s in self._samples.items() if name == level}
        samples = [sample for iteration in sorted(groups) for sample in groups[iteration]]

        def values(key, group=samples):
            return [sample[key] for sample in group if sample[key] is not None]

        result: dict[str, Any] = dict(samples=len(samples))
        if clocks := values("sm_clock_mhz"):
            result["sm_clock_mhz"] = dict(median=statistics.median(clocks), min=min(clocks), max=max(clocks))
        if clocks := values("memory_clock_mhz"):
            result["memory_clock_mhz"] = dict(median=statistics.median(clocks), min=min(clocks))
        if power := values("power_w"):
            result["power_w"] = dict(mean=statistics.fmean(power), max=max(power))
        if temperature := values("temperature_c"):
            result["temperature_c"] = dict(mean=statistics.fmean(temperature), max=max(temperature))
        if reasons := values("clock_event_reasons"):
            result["clock_event_reasons"] = {
                name: sum(1 for bits in reasons if bits & bit) / len(reasons)
                for bit, name in CLOCK_EVENT_REASONS.items()
                if any(bits & bit for bits in reasons)
            }
        per_iter = []
        for iteration in range(max(groups, default=-1) + 1):
            clocks = values("sm_clock_mhz", groups.get(iteration, []))
            per_iter.append(statistics.median(clocks) if clocks else None)
        result["sm_clock_mhz_per_iter"] = per_iter
        return result

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.locked_sm_clock_mhz is not None:
            try:
                self._nvml.nvmlDeviceResetGpuLockedClocks(self._handle)
                self.locked_sm_clock_mhz = None
            except Exception as exc:
                print(f"Warning: could not reset the locked SM clock ({exc}); run `nvidia-smi --reset-gpu-clocks`.")
        if self.available:
            self._query(self._nvml.nvmlShutdown)


def load_vae(args):
    parallel = args.vae_patch_parallel_size > 1
    cls = DistributedAutoencoderKLWan if parallel else OmniAutoencoderKLWan
    if args.tiny:
        torch.manual_seed(args.seed)
        vae = cls(**TINY_CONFIG).to(device="cuda", dtype=DTYPES[args.dtype]).eval()
        if parallel:
            vae.init_distributed()
    else:
        vae = (
            cls.from_pretrained(
                args.model,
                subfolder=args.subfolder,
                revision=args.revision,
                torch_dtype=DTYPES[args.dtype],
            )
            .to("cuda")
            .eval()
        )
    if args.tiling or parallel:
        vae.enable_tiling()
    if parallel:
        vae.set_parallel_size(args.vae_patch_parallel_size, mode="tile")
    return vae


def differences(actual, reference):
    integer = torch.int16 if actual.element_size() == 2 else torch.int32
    equal = torch.equal(actual.contiguous().view(integer), reference.contiguous().view(integer))
    a, b = actual.float(), reference.float()
    error = a - b
    rmse = error.square().mean().sqrt().item()
    scale = max(b.square().mean().sqrt().item(), 1e-8)
    return {"bitwise_equal": equal, "max_abs_diff": error.abs().max().item(), "normalized_rmse": rmse / scale}


def validate_result(result, reference):
    """Attach quality metrics and explicit failure reasons to one timed result."""
    stats, moments, mean, logvar, reconstruction = result
    level = stats["level"]
    errors = stats["errors"] = []
    if reference is None:
        stats["status"] = "unvalidated"
        errors.append(f"[{level}] reference level 'off' is unavailable; cannot compute speedup or validate outputs.")
        return

    stats["speedup"] = reference[0]["median_s"] / stats["median_s"]
    for name, actual, expected in (
        ("moments", moments, reference[1]),
        ("mean", mean, reference[2]),
        ("logvar", logvar, reference[3]),
    ):
        metrics = stats[name] = differences(actual, expected)
        nrmse = metrics["normalized_rmse"]
        max_abs = metrics["max_abs_diff"]
        if not math.isfinite(nrmse) or not math.isfinite(max_abs):
            errors.append(
                f"[{level}] {name}: nonfinite comparison metrics "
                f"(normalized_rmse={nrmse!r}, max_abs_diff={max_abs!r}); "
                "check candidate/reference tensors for NaN, Inf or numerical overflow."
            )
        if level == "lossless" and not metrics["bitwise_equal"]:
            errors.append(
                f"[{level}] {name}: bitwise_equal=False, required True "
                f"(max_abs_diff={max_abs:.6g}, normalized_rmse={nrmse!r})."
            )
        limit = encoder_nrmse_limit(actual.dtype)
        if level == "channels_last" and math.isfinite(nrmse) and nrmse > limit:
            errors.append(
                f"[{level}] {name}: normalized_rmse={nrmse!r} exceeds "
                f"{limit:g} ({limit:.0%}); max_abs_diff={max_abs:.6g}."
            )
    if reconstruction is not None:
        mse = (reconstruction.float() - reference[4].float()).square().mean().item()
        if not math.isfinite(mse):
            psnr = math.nan if math.isnan(mse) else -math.inf
        else:
            psnr = math.inf if mse == 0 else 10 * math.log10(4 / mse)
        stats["reconstruction_psnr_db"] = "inf" if psnr == math.inf else psnr
        if math.isnan(psnr) or psnr < MIN_RECONSTRUCTION_PSNR_DB:
            errors.append(
                f"[{level}] reconstruction: PSNR={psnr!r} dB, "
                f"required >= {MIN_RECONSTRUCTION_PSNR_DB:g} dB (MSE={mse!r})."
            )
    stats["status"] = "validation_failed" if errors else "ok"


def _print_table(headers, rows, text_columns=2, trailing_text_columns=0):
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    first_trailing = len(headers) - trailing_text_columns
    for index, row in enumerate([headers, *rows]):
        print(
            "  ".join(
                value.ljust(width) if i < text_columns or i >= first_trailing else value.rjust(width)
                for i, (value, width) in enumerate(zip(row, widths))
            ).rstrip()
        )
        if index == 0:
            print("  ".join("-" * width for width in widths))


def _print_telemetry(results, environment):
    """Per-level clocks during timed encodes, and a warning when they differ or the GPU throttled."""
    rows, medians, throttled = [], {}, set()
    for stats in results:
        telemetry = stats.get("telemetry") or {}
        if not telemetry.get("samples"):
            continue
        clock = telemetry.get("sm_clock_mhz")
        if clock:
            medians[stats["level"]] = clock["median"]
        reasons = telemetry.get("clock_event_reasons", {})
        throttled.update(name for name in THROTTLE_REASONS if reasons.get(name, 0) > 0)
        shown = sorted(((f, n) for n, f in reasons.items() if n != "gpu_idle"), reverse=True)
        rows.append(
            [
                stats["level"],
                str(telemetry["samples"]),
                f"{clock['median']:.0f}" if clock else "-",
                f"{clock['min']:.0f}" if clock else "-",
                f"{telemetry['power_w']['mean']:.0f}" if "power_w" in telemetry else "-",
                f"{telemetry['temperature_c']['max']:.0f}" if "temperature_c" in telemetry else "-",
                ", ".join(f"{name} {fraction:.0%}" for fraction, name in shown) or "none",
            ]
        )
    if not rows:
        return
    info = environment.get("telemetry") or {}
    details = [f"sampled every {info['interval_ms']:g} ms" if info.get("interval_ms") else None]
    if info.get("power_limit_w"):
        details.append(f"power limit {info['power_limit_w']:.0f} W")
    if info.get("max_sm_clock_mhz"):
        details.append(f"max SM clock {info['max_sm_clock_mhz']} MHz")
    if info.get("locked_sm_clock_mhz"):
        details.append(f"SM clock locked at {info['locked_sm_clock_mhz']} MHz")
    print(f"\nGPU telemetry during timed encodes (NVML; {', '.join(d for d in details if d)})")
    _print_table(
        ["Level", "Samples", "SM MHz (median)", "SM MHz (min)", "Power (W)", "Temp max (C)", "Clock-event reasons"],
        rows,
        text_columns=1,
        trailing_text_columns=1,
    )
    if len(medians) > 1:
        spread = (max(medians.values()) - min(medians.values())) / max(medians.values())
        if spread > CLOCK_SPREAD_WARNING:
            print(
                f"Warning: median SM clocks differ by {spread:.0%} across levels, so the speedups include "
                "clock differences. Use --schedule interleaved or --lock-sm-clock-mhz for comparable timings."
            )
    if throttled:
        print(
            f"Note: the GPU throttled during timed encodes ({', '.join(sorted(throttled))}); absolute times "
            "depend on cooling and the power limit."
        )


def print_results(results, environment):
    """Human-readable report; retain the full machine-readable data in --json."""
    shape = "x".join(map(str, environment["input_shape"]))
    print(f"\nWan VAE encoder | {environment['gpu']} | {environment['dtype']} | RGB {shape} (B,C,T,H,W)")
    line = f"Model: {environment['model']} | GPUs: {environment['world_size']} | tiling: {environment['tiling']}"
    if environment.get("schedule"):
        line += f" | schedule: {environment['schedule']}"
    print(line)
    print("\nTiming (validation status is separate from speedup)")
    labels = {"ok": "PASS", "validation_failed": "FAIL", "oom": "OOM", "unvalidated": "NO REF"}
    rows = []
    for stats in results:
        rows.append(
            [
                stats["level"],
                labels[stats["status"]],
                f"{stats['median_s']:.4f}" if "median_s" in stats else "-",
                f"{stats['speedup']:.2f}x" if "speedup" in stats else "-",
                f"{stats['frames_per_s']:.2f}" if "frames_per_s" in stats else "-",
                f"{stats['peak_gib']:.2f}" if "peak_gib" in stats else "-",
                f"{stats['install_s']:.4f}" if "install_s" in stats else "-",
                f"{stats['first_call_s']:.4f}" if "first_call_s" in stats else "-",
            ]
        )
    _print_table(
        ["Level", "Status", "Median (s)", "Speedup", "Frames/s", "Peak (GiB)", "Install (s)", "First (s)"], rows
    )
    _print_telemetry(results, environment)

    rows = []
    for stats in results:
        for name in ("moments", "mean", "logvar"):
            if name in stats:
                metrics = stats[name]
                rows.append(
                    [
                        stats["level"],
                        name,
                        "yes" if metrics["bitwise_equal"] else "no",
                        f"{metrics['max_abs_diff']:.6g}",
                        f"{metrics['normalized_rmse']:.4%}",
                    ]
                )
    if rows:
        print("\nPosterior quality vs. off (NRMSE is normalized RMSE; bitwise 'no' is allowed for channels_last)")
        _print_table(["Level", "Tensor", "Bitwise", "Max abs diff", "NRMSE"], rows)
    rows = [
        [stats["level"], f"{float(stats['reconstruction_psnr_db']):.2f}"]
        for stats in results
        if "reconstruction_psnr_db" in stats
    ]
    if rows:
        print("\nReference-decoder reconstruction quality")
        _print_table(["Level", "PSNR (dB)"], rows, text_columns=1)
    print(
        "\nGates: lossless must be bitwise equal; "
        f"channels_last NRMSE <= {encoder_nrmse_limit(torch.bfloat16):.0%} for BF16, "
        f"{encoder_nrmse_limit(torch.float32):.0%} for FP16/FP32; "
        "posterior comparison metrics must be finite."
    )
    if rows:
        print(f"Reconstruction gate: PSNR >= {MIN_RECONSTRUCTION_PSNR_DB:g} dB.")
    print(flush=True)


def failure_summary(results):
    failed = [stats for stats in results if stats["status"] != "ok"]
    lines = [f"Encoder benchmark failed for {len(failed)} level(s):"]
    lines.extend(f"  - {error}" for stats in failed for error in stats["errors"])
    return "\n".join(lines)


@dataclass
class LevelRun:
    """One level's state between setup, round-robin timing and validation."""

    level: str
    input_shape: list[int]
    vae: Any = None
    inputs: torch.Tensor | None = None
    phase: str = "model loading"
    report: Any = None
    install_s: float = 0.0
    first_call_s: float = 0.0
    times: list[float] = field(default_factory=list)
    peak_bytes: int = 0
    # Weights of the other levels resident during this level's encodes (interleaved schedule).
    other_resident_bytes: int = 0
    # Posterior (moments, mean, logvar) of the last timed encode, on the CPU.
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


def _release(run: LevelRun) -> None:
    run.vae = None
    run.inputs = None


def _oom_error(args, run: LevelRun, exc: BaseException) -> torch.OutOfMemoryError:
    if run.phase == "model loading":
        hint = "Free GPU memory before loading the VAE, or use --tiny for a development-only run."
        if getattr(args, "schedule", "sequential") == "interleaved":
            hint += " The interleaved schedule keeps every level's VAE resident; --schedule sequential does not."
    elif run.phase == "reference-decoder reconstruction":
        hint = "Try a smaller input or omit --check-reconstruction to benchmark encoding alone."
    else:
        hint = "Try a smaller --size/--frames workload or enable --tiling."
    return torch.OutOfMemoryError(
        f"[{run.level}] out of memory during {run.phase} "
        f"(input={run.input_shape}, dtype={args.dtype}). {hint}\n"
        f"    Original error: {exc}"
    )


def resident_bytes(module: torch.nn.Module) -> int:
    """Bytes of the module's CUDA parameters and buffers, counting shared storage once."""
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in itertools.chain(module.parameters(), module.buffers())
        if tensor.is_cuda
    }
    return sum(storages.values())


def _update_other_resident(runs: list[LevelRun]) -> None:
    """Record, per run, the other levels' resident weights, subtracted from its peak memory."""
    if len(runs) < 2:
        for run in runs:
            run.other_resident_bytes = 0
        return
    sizes = [resident_bytes(run.vae) for run in runs]
    for run, size in zip(runs, sizes):
        run.other_resident_bytes = sum(sizes) - size


def prepare_level(args, level, pixels, inputs=None) -> LevelRun:
    """Load and install one level, then run its untimed first encode (JIT, cuDNN plans, verdict probes)."""
    run = LevelRun(level=level, input_shape=list(pixels.shape))
    try:
        run.vae = load_vae(args)
        run.phase = "input transfer"
        run.inputs = pixels.to(device="cuda", dtype=DTYPES[args.dtype]) if inputs is None else inputs
        sync()
        start = time.perf_counter()
        run.phase = "fast-path installation"
        run.report = install_wan_vae_encoder_fastpath(run.vae, level=level)
        sync()
        run.install_s = max_across_ranks(time.perf_counter() - start)
        if level != "off" and not run.report.installed:
            raise RuntimeError(f"[{level}] requested encoder fast path was not installed: {run.report.reason}")
        sync()
        start = time.perf_counter()
        run.phase = "first encode"
        run.vae.encode(run.inputs)
        sync()
        run.first_call_s = max_across_ranks(time.perf_counter() - start)
        return run
    except torch.OutOfMemoryError as exc:
        _release(run)
        raise _oom_error(args, run, exc) from exc


def _timed_encode(args, run: LevelRun, telemetry: GpuTelemetry | None, iteration: int) -> None:
    run.phase = "timed encoding"
    sync()
    torch.accelerator.reset_peak_memory_stats()
    with telemetry.measure(run.level, iteration) if telemetry is not None else contextlib.nullcontext():
        start = time.perf_counter()
        posterior = run.vae.encode(run.inputs).latent_dist
        torch.accelerator.synchronize()
        elapsed = time.perf_counter() - start
    run.times.append(max_across_ranks(elapsed))
    run.peak_bytes = max(run.peak_bytes, torch.accelerator.max_memory_allocated() - run.other_resident_bytes)
    if iteration == args.iters - 1:
        run.phase = "posterior transfer"
        run.outputs = (
            posterior.parameters.detach().cpu(),
            posterior.mode().detach().cpu(),
            posterior.logvar.detach().cpu(),
        )


def run_rounds(args, runs: list[LevelRun], telemetry: GpuTelemetry | None = None, *, fail_fast: bool = True):
    """Warm up, then time, every level round-robin, rotating the first level each round.

    Rotation puts every level in every position of the order, so no level
    systematically follows a heavier one or runs later on a hotter GPU.
    Returns ``{level: OutOfMemoryError}`` for levels dropped after running out
    of memory; with ``fail_fast`` the first such error is raised instead.
    """
    failures: dict[str, torch.OutOfMemoryError] = {}
    active = list(runs)
    _update_other_resident(active)
    for index in range(args.warmup + args.iters):
        if not active:
            break
        shift = index % len(active)
        for run in active[shift:] + active[:shift]:
            try:
                if index < args.warmup:
                    run.phase = "encode warmup"
                    run.vae.encode(run.inputs)
                else:
                    _timed_encode(args, run, telemetry, index - args.warmup)
            except torch.OutOfMemoryError as exc:
                error = _oom_error(args, run, exc)
                if fail_fast:
                    raise error from exc
                # Keep the message only: a traceback would pin the failed model's frames.
                exc.__traceback__ = None
                error.__cause__ = exc
                failures[run.level] = error
                _release(run)
        if failures:
            active = [run for run in active if run.level not in failures]
            gc.collect()
            torch.accelerator.empty_cache()
            _update_other_resident(active)
    return failures


def finish_level(args, run: LevelRun, rank, telemetry: GpuTelemetry | None = None):
    """Summarize the timings, run the untimed reconstruction/profile, then release the model."""
    level, vae, inputs = run.level, run.vae, run.inputs
    try:
        peak = max_across_ranks(run.peak_bytes) / 2**30
        median = statistics.median(run.times)
        stats = dict(
            status="ok",
            level=level,
            install_s=run.install_s,
            first_call_s=run.first_call_s,
            median_s=median,
            min_s=min(run.times),
            times_s=run.times,
            peak_gib=peak,
            frames_per_s=inputs.shape[0] * inputs.shape[2] / median,
            patched=dict(run.report.patched),
            fused_silu=run.report.fused_silu_dtypes,
            checkpoint_revision=getattr(vae.config, "_commit_hash", None),
        )
        if telemetry is not None and telemetry.available:
            stats["telemetry"] = telemetry.summary(level)
        moments, mean, logvar = run.outputs
        reconstruction = None
        if args.check_reconstruction:
            run.phase = "reference-decoder reconstruction"
            # No decoder fast path is installed in this benchmark. Every candidate
            # is reconstructed by the same reference decoder weights and settings.
            decoded = vae.decode(mean.to(device="cuda", dtype=DTYPES[args.dtype])).sample
            if rank == 0:
                reconstruction = decoded.cpu()
            del decoded
        if args.profile:
            run.phase = "profiling"
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                vae.encode(inputs)
                sync()
            if rank == 0:
                kernels = {}
                for event in prof.events():
                    if event.device_type.name == "CUDA":
                        entry = kernels.setdefault(event.name, {"name": event.name, "ms": 0.0, "calls": 0})
                        entry["ms"] += event.device_time_total / 1000
                        entry["calls"] += 1
                ordered = sorted(kernels.values(), key=lambda item: item["ms"], reverse=True)
                total = sum(item["ms"] for item in ordered)
                conv = sum(
                    item["ms"]
                    for item in ordered
                    if any(tag in item["name"].lower() for tag in ("conv", "cudnn", "gemm", "cutlass", "xmma"))
                )
                stats["profile"] = {
                    "kernel_time_ms": total,
                    "conv_like_percent": 100 * conv / total if total else 0,
                    "kernels": ordered[: args.profile_rows],
                }
                print(f"[{level}] CUDA operator profile")
                print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=args.profile_rows))
        run.phase = "cleanup"
        _release(run)
        del vae, inputs
        # Instance-bound forwards form cycles; collect them before the next level
        # so its peak memory excludes the previous model's parameters.
        gc.collect()
        torch.accelerator.empty_cache()
        return stats, moments, mean, logvar, reconstruction
    except torch.OutOfMemoryError as exc:
        _release(run)
        raise _oom_error(args, run, exc) from exc


@torch.inference_mode()
def run_level(args, level, pixels, rank, telemetry=None):
    """Sequential schedule: set up, time and validate one level before loading the next."""
    run = prepare_level(args, level, pixels)
    run_rounds(args, [run], telemetry)
    return finish_level(args, run, rank, telemetry)


@torch.inference_mode()
def run_interleaved(args, pixels, rank, telemetry=None):
    """Set up every level, time them round-robin, then validate each.

    Returns ``{level: result or OutOfMemoryError}``. A single-process run drops
    only the level that ran out of memory; distributed runs re-raise, because a
    failed collective cannot be retried by one rank.
    """
    fail_fast = dist.is_initialized()
    outcomes: dict[str, Any] = {}
    runs: list[LevelRun] = []
    inputs = None
    for level in args.levels:
        if rank == 0:
            print(f"Setting up {level}...", flush=True)
        try:
            run = prepare_level(args, level, pixels, inputs)
        except torch.OutOfMemoryError as exc:
            if fail_fast:
                raise
            exc.__traceback__ = None
            outcomes[level] = exc
            gc.collect()
            torch.accelerator.empty_cache()
            continue
        inputs = run.inputs
        runs.append(run)
    if rank == 0 and runs:
        names = ", ".join(run.level for run in runs)
        print(f"Timing {names} round-robin ({args.warmup} warmup + {args.iters} timed rounds)...", flush=True)
    outcomes.update(run_rounds(args, runs, telemetry, fail_fast=fail_fast))
    for run in runs:
        if run.level in outcomes:
            continue
        try:
            outcomes[run.level] = finish_level(args, run, rank, telemetry)
        except torch.OutOfMemoryError as exc:
            if fail_fast:
                raise
            exc.__traceback__ = None
            outcomes[run.level] = exc
    return outcomes


def level_outcomes(args, pixels, rank, telemetry=None):
    """Yield ``(level, result or OutOfMemoryError)`` in ``args.levels`` order."""
    if args.schedule == "interleaved":
        outcomes = run_interleaved(args, pixels, rank, telemetry)
        for level in args.levels:
            yield level, outcomes[level]
        return
    for level in args.levels:
        if rank == 0:
            print(f"Running {level}...", flush=True)
        try:
            outcome = run_level(args, level, pixels, rank, telemetry=telemetry)
        except torch.OutOfMemoryError as exc:
            if dist.is_initialized():
                raise  # A failed collective cannot be safely retried by one rank.
            outcome = exc
        yield level, outcome


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; use H100 or GB200 for target performance validation")
    rank = distributed_setup(args)
    torch.backends.cudnn.allow_tf32 = args.tf32 == "on"
    pixels = make_pixels(args)
    results = []
    reference = None
    environment = dict(
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        diffusers=package_version("diffusers"),
        triton=package_version("triton"),
        vllm=package_version("vllm"),
        vllm_omni=package_version("vllm-omni"),
        platform=platform.platform(),
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        tiny=args.tiny,
        tiling=args.tiling or args.vae_patch_parallel_size > 1,
        warmup=args.warmup,
        iters=args.iters,
        input_shape=list(pixels.shape),
        input=str(args.input) if args.input else "seeded_uniform",
        seed=args.seed,
        tf32=args.tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        world_size=args.vae_patch_parallel_size,
        schedule=args.schedule,
    )
    telemetry = None
    if args.telemetry_interval_ms > 0 or args.lock_sm_clock_mhz is not None:
        # Every rank samples (and locks) its own device; rank 0 reports its own.
        telemetry = GpuTelemetry(args.telemetry_interval_ms / 1000)
    try:
        if args.lock_sm_clock_mhz is not None:
            clock = args.lock_sm_clock_mhz
            try:
                telemetry.lock_sm_clock(clock)
            except Exception as exc:
                raise SystemExit(
                    f"Could not lock the SM clock to {clock} MHz ({exc}). Locking needs administrator rights; "
                    f"alternatively run `sudo nvidia-smi --lock-gpu-clocks={clock},{clock}` before the benchmark "
                    "and `sudo nvidia-smi --reset-gpu-clocks` after it."
                ) from exc
        if telemetry is not None:
            environment["telemetry"] = telemetry.device_info()
            telemetry.start()
        for level, outcome in level_outcomes(args, pixels, rank, telemetry):
            if isinstance(outcome, torch.OutOfMemoryError):
                results.append(dict(level=level, status="oom", errors=[str(outcome)]))
                # Release the exception traceback before collecting failed models.
                del outcome
                gc.collect()
                torch.accelerator.empty_cache()
                continue
            result = outcome
            stats = result[0]
            if level == "off":
                reference = result
            if rank == 0:
                validate_result(result, reference)
            results.append(stats)
        if rank == 0:
            print_results(results, environment)
            if args.json:
                args.json.write_text(json.dumps(dict(environment=environment, results=results), indent=2) + "\n")
                print(f"Detailed results saved to {args.json}", flush=True)
        failed = bool(max_across_ranks(int(any(stats["status"] != "ok" for stats in results))))
    finally:
        if telemetry is not None:
            telemetry.close()
        if dist.is_initialized():
            dist.destroy_process_group()
    if failed:
        raise SystemExit(failure_summary(results) if rank == 0 else "Encoder benchmark failed; see rank 0's report.")
    if rank == 0:
        print("All requested validation checks passed.")


if __name__ == "__main__":
    main()
