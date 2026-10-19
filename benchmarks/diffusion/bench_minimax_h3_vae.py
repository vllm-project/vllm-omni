# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure one H3 decoder per process and compare unquantized outputs.

Use --variant original/lite for native decode, official for a pinned LightVAE
reference environment, and compare to summarize their report.json files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def compare_outputs(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool | None]:
    if reference.shape != candidate.shape or reference.ndim != 5 or reference.shape[1] != 3:
        raise ValueError("video shape must match in BCTHW RGB layout")
    for video in (reference, candidate):
        if not video.is_floating_point() or not torch.isfinite(video).all():
            raise ValueError("quality input must be finite floating-point video")
        if video.min() < 0 or video.max() > 1:
            raise ValueError("quality input must have range [0,1]")
    from torchmetrics.functional.image import structural_similarity_index_measure

    reference, candidate = reference.cpu().float(), candidate.cpu().float()
    difference = (reference.double() - candidate.double()).abs()
    mse = float(difference.square().mean())
    scores = []
    for frame in range(reference.shape[2]):
        scores.append(
            float(
                structural_similarity_index_measure(
                    candidate[:, :, frame],
                    reference[:, :, frame],
                    data_range=1.0,
                    kernel_size=11,
                    sigma=1.5,
                )
            )
        )
    return {
        "mae": float(difference.mean()),
        "max_abs": float(difference.max()),
        "psnr": None if mse == 0 else -10 * math.log10(mse),
        "ssim": statistics.mean(scores),
        "exact": mse == 0,
    }


def _device_api(device: torch.device):
    return torch.get_device_module(device)


def measure_decode(
    decode: Callable[[], torch.Tensor],
    device: torch.device,
    *,
    warmup: int = 5,
    repeats: int = 20,
    rounds: int = 3,
) -> dict[str, object]:
    if warmup < 0 or repeats < 1 or rounds < 1:
        raise ValueError("warmup must be nonnegative; repeats and rounds must be positive")
    api = _device_api(device)
    sync = getattr(api, "synchronize", lambda: None)
    reset = getattr(api, "reset_peak_memory_stats", None)
    allocated = getattr(api, "max_memory_allocated", None)
    reserved = getattr(api, "max_memory_reserved", None)
    samples, allocated_peaks, reserved_peaks = [], [], []
    with torch.inference_mode():
        for _ in range(rounds):
            for _ in range(warmup):
                output = decode()
                del output
            sync()
            if reset:
                reset()
            group = []
            for _ in range(repeats):
                sync()
                start = time.perf_counter()
                output = decode()
                sync()
                group.append((time.perf_counter() - start) * 1000)
                del output
            samples.append(group)
            if allocated:
                allocated_peaks.append(int(allocated()))
            if reserved:
                reserved_peaks.append(int(reserved()))
    return {
        "samples_ms": samples,
        "median_ms_by_round": [statistics.median(group) for group in samples],
        "peak_allocated_bytes": max(allocated_peaks, default=None),
        "peak_reserved_bytes": max(reserved_peaks, default=None),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*.safetensors")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"no safetensors weights found under {path}")
    for item in paths:
        if path.is_dir():
            digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _environment(device: torch.device, *, component_path: Path | None = None) -> dict[str, object]:
    versions = {}
    for package in (
        "torch",
        "torch-npu",
        "vllm",
        "vllm-omni",
        "mindiesd",
        "torchmetrics",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    api = _device_api(device)
    root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__).resolve()]
    source_paths.extend(sorted((root / "vllm_omni/diffusion/models/minimax_h3").rglob("*.py")))
    sources = {str(path.relative_to(root)): _sha256(path) for path in source_paths}
    if component_path is not None:
        sources.update(
            {
                f"remote/{path.relative_to(component_path).as_posix()}": _sha256(path)
                for path in sorted(component_path.rglob("*.py"))
            }
        )
    return {
        "device": str(device),
        "device_name": api.get_device_name(device) if hasattr(api, "get_device_name") else "CPU",
        "versions": versions,
        "code_revision": _revision(root),
        "source_sha256": sources,
    }


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not args.device or not args.dtype or not args.latents:
        raise ValueError("run requires --device, --dtype and --latents")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    device = torch.device(args.device)
    api = _device_api(device)
    if device.type != "cpu":
        api.set_device(device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    latent_path = Path(args.latents)
    latent = load_file(str(latent_path), device="cpu")["latent"].to(device=device, dtype=dtype)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    started = time.perf_counter()
    reference_revision = None
    if args.variant == "official":
        if checkpoint is None:
            raise ValueError("official requires --checkpoint")
        import lightvae.h3_lightvae_decoder as official

        reference_revision = _revision(Path(official.__file__).resolve().parents[1])
        model = official.Decoder(checkpoint=checkpoint).eval().to(device)
        component_source = Path(official.__file__).resolve().parent
        config = model.config
        tile_size, tile_overlap = model.tile_size, model.tile_overlap

        def decode_model():
            # Canonicalize the official RGB [-1,1] output exactly once.
            return ((model(latent).float() + 1) / 2).clamp(0, 1)
    else:
        if not args.model_path:
            raise ValueError("native run requires --model-path")
        if args.variant == "lite" and checkpoint is None:
            raise ValueError("lite requires --checkpoint")
        # The default streaming path quantizes to uint8; numerical comparisons
        # need the native floating-point temporal path on both variants.
        os.environ["VLLM_OMNI_VAE_LEGACY_TEMPORAL"] = "1"
        from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

        expected_source = Path(__file__).resolve().parents[2] / "vllm_omni/diffusion/models/minimax_h3/vae.py"
        if Path(inspect.getfile(MiniMaxH3VideoVAE)).resolve() != expected_source:
            raise ValueError("Omni was imported from another checkout; set PYTHONPATH to this repository")
        model = MiniMaxH3VideoVAE(
            args.model_path,
            device=device,
            decode_only=True,
            trust_remote_code=args.trust_remote_code,
            lightvae_decoder_path=str(checkpoint) if args.variant == "lite" else None,
        )
        component_source = Path(inspect.getfile(type(model.remote))).resolve().parent
        config = model.config_dict
        tile_size = config["vae_tile_size"]
        tile_overlap = config["vae_tile_overlap_min"]

        def decode_model():
            output = model.decode_latent(latent)
            if not output.is_floating_point():
                raise ValueError("native decoder quantized output; cannot validate pre-quantization quality")
            return output

    def decode():
        context = nullcontext() if dtype == torch.float32 else torch.autocast(device.type, dtype=dtype)
        with torch.inference_mode(), context:
            return decode_model()

    startup_ms = (time.perf_counter() - started) * 1000
    result = (
        {}
        if args.variant == "official"
        else measure_decode(
            decode,
            device,
            warmup=args.warmup,
            repeats=args.repeats,
            rounds=args.rounds,
        )
    )
    output = decode().cpu().float()
    if output.ndim != 5 or output.shape[1] != 3 or not torch.isfinite(output).all():
        raise ValueError("decoder output must be finite BCTHW RGB video")
    if output.min() < 0 or output.max() > 1:
        raise ValueError("decoder output must have range [0,1]")
    save_file({"video": output.contiguous()}, Path(args.output_dir) / "decoded.safetensors")
    weights_path = checkpoint if args.variant in ("lite", "official") else Path(args.model_path)
    result.update(
        {
            "variant": args.variant,
            "status": "measured",
            "output_file": "decoded.safetensors",
            "model_load_ms": startup_ms,
            "weights_sha256": _sha256(weights_path),
            "environment": _environment(device, component_path=component_source),
            "reference_revision": reference_revision,
            "signature": {
                "input_sha256": _sha256(latent_path),
                "shape": list(latent.shape),
                "dtype": args.dtype,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "parallel_size": 1,
                "offload": "resident",
                "output_range": [0, 1],
                "base_model_sha256": _sha256(Path(args.model_path)) if args.variant != "official" else None,
            },
            "temporal_mode": "official-float" if args.variant == "official" else "legacy-float",
            "latent_channels": config["latent_channels"],
        }
    )
    return result


def _timing_medians(report: dict) -> list[float]:
    samples = report.get("samples_ms", [])
    declared = report.get("median_ms_by_round", [])
    if len(samples) < 3 or len(samples) != len(declared):
        raise ValueError("timing requires at least three complete rounds")
    result = []
    for group, median in zip(samples, declared):
        if len(group) < 20 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in group):
            raise ValueError("timing requires at least 20 positive finite samples per round")
        computed = statistics.median(group)
        if type(median) not in (int, float) or not math.isclose(computed, median, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("reported median does not match raw timing samples")
        result.append(computed)
    return result


def _compare(paths: Sequence[str], ground_truth: str | None) -> dict[str, object]:
    reports, directories = {}, {}
    for path in map(Path, paths):
        report = json.loads(path.read_text())
        variant = report["variant"]
        if variant in reports:
            raise ValueError(f"duplicate variant {variant}")
        if report.get("status") != "measured":
            raise ValueError(f"{variant} did not finish: {report.get('error', report.get('status'))}")
        reports[variant], directories[variant] = report, path.parent
    result: dict[str, object] = {
        "variant": "compare",
        "status": "not_validated",
        "speedup_verified": False,
    }
    if not {"original", "lite", "official"} <= reports.keys():
        result["reason"] = "original, lite and official reports are all required"
        return result
    original, lite, official = (reports[name] for name in ("original", "lite", "official"))
    if original["signature"] != lite["signature"]:
        raise ValueError("original and lite experimental configuration mismatch")
    for key in ("device", "device_name", "versions", "code_revision", "source_sha256"):
        value = original.get("environment", {}).get(key)
        if not value or value != lite.get("environment", {}).get(key):
            raise ValueError(f"original and lite runtime {key} mismatch or missing")
    environment = original["environment"]
    versions = environment["versions"]
    required_packages = ["torch"]
    if environment["device"].startswith("npu"):
        required_packages.append("torch-npu")
    if not isinstance(versions, dict) or any(
        not isinstance(versions.get(name), str) or not versions[name] for name in required_packages
    ):
        raise ValueError("original and lite runtime versions are unknown")
    sources = environment["source_sha256"]
    if not isinstance(sources, dict) or any(not isinstance(value, str) or not value for value in sources.values()):
        raise ValueError("original and lite runtime source_sha256 is unknown")
    for key in ("input_sha256", "shape", "tile_size", "tile_overlap", "output_range"):
        if lite["signature"][key] != official["signature"][key]:
            raise ValueError(f"official reference mismatch on {key}")
    if lite["weights_sha256"] != official["weights_sha256"]:
        raise ValueError("official reference uses different Lite weights")
    videos = {
        name: load_file(str(directories[name] / report["output_file"]), device="cpu")["video"]
        for name, report in reports.items()
    }
    reference_metrics = compare_outputs(videos["official"], videos["lite"])
    passed = reference_metrics["exact"] or reference_metrics["psnr"] >= 40
    result.update(
        {
            "status": "passed" if passed else "failed",
            "lite_reference": reference_metrics,
            "original_vs_lite": compare_outputs(videos["original"], videos["lite"]),
            "sources": list(paths),
        }
    )
    base_medians, lite_medians = _timing_medians(original), _timing_medians(lite)
    result["decode_speedup"] = statistics.median(base_medians) / statistics.median(lite_medians)
    result["speedup_verified"] = bool(passed and max(lite_medians) < min(base_medians))
    if ground_truth:
        target = load_file(ground_truth, device="cpu")["video"]
        result["original_reconstruction"] = compare_outputs(target, videos["original"])
        result["lite_reconstruction"] = compare_outputs(target, videos["lite"])
        result["ground_truth_sha256"] = _sha256(Path(ground_truth))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("original", "lite", "official", "compare"), required=True)
    for name in (
        "model-path",
        "checkpoint",
        "latents",
        "dtype",
        "device",
        "ground-truth",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument("--reports", nargs="+", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = _compare(args.reports, args.ground_truth) if args.variant == "compare" else _run(args)
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        ImportError,
        subprocess.CalledProcessError,
    ) as exc:
        result = {"variant": args.variant, "status": "failed", "error": str(exc)}
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return {"failed": 1, "not_validated": 2}.get(result["status"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
