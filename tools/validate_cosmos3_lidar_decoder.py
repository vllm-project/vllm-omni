#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-process parity and performance validation for the Cosmos3 LiDAR VAE.

The reference and candidate implementations intentionally run in separate
Python environments:

1. ``--mode imaginaire4`` loads the exported VAE with imaginaire4's original
   NATTEN implementation and writes inputs, reference outputs, and measurements
   to one safetensors artifact.
2. ``--mode vllm-omni`` loads that artifact, runs vLLM-Omni's FlexAttention
   implementation on the saved inputs, and writes a JSON comparison report.

The model argument is the root of an exported checkpoint containing
``lidar_vae/config.json`` and ``lidar_vae/diffusion_pytorch_model.safetensors``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vllm_omni.platforms import current_omni_platform

ARTIFACT_METADATA_KEY = "cosmos3_lidar_validation"
SCHEMA_VERSION = 1
LIDAR_VAE_SUBFOLDER = "lidar_vae"
LIDAR_VAE_CONFIG = "config.json"
LIDAR_VAE_WEIGHTS = "diffusion_pytorch_model.safetensors"


@dataclass(frozen=True)
class Measurement:
    first_call_ms: float
    warmup_calls: int
    timed_calls: int
    timed_ms: list[float]
    mean_ms: float
    min_ms: float
    max_ms: float
    resident_bytes: int | None
    first_call_peak_increment_bytes: int | None
    warmed_peak_increment_bytes: int | None
    rng_preserved: bool
    repeated_output_identical: bool


def _torch():
    import torch

    return torch


def _json_dump(value: Any) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_component(model: str) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve a local exported checkpoint and return root, component, config."""
    root = Path(model).expanduser().resolve()
    if root.name == LIDAR_VAE_SUBFOLDER and (root / LIDAR_VAE_CONFIG).is_file():
        component = root
        root = root.parent
    else:
        component = root / LIDAR_VAE_SUBFOLDER
    config_path = component / LIDAR_VAE_CONFIG
    weights_path = component / LIDAR_VAE_WEIGHTS
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Expected {config_path} and {weights_path}. This validator requires a local exported checkpoint."
        )
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    return root, component, config


def _model_identity(component: Path, config: Mapping[str, Any]) -> dict[str, str]:
    return {
        "config_sha256": hashlib.sha256(_canonical_json(config)).hexdigest(),
        "weights_sha256": _sha256_file(component / LIDAR_VAE_WEIGHTS),
    }


def _load_tensor_file(path: Path, key: str):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if key not in keys:
            raise ValueError(f"{path} must contain tensor {key!r}; found {keys}.")
        return handle.get_tensor(key).float().contiguous()


def _normalize_latents(latents, config: Mapping[str, Any]):
    torch = _torch()
    network = config["network_config"]
    spatial = config["spatial_compression"]
    expected_hw = tuple(size // factor for size, factor in zip(network["resolution"], spatial, strict=True))
    expected_channels = config["latent_channels"]
    if (
        latents.ndim != 5
        or latents.shape[0] < 1
        or latents.shape[1] != expected_channels
        or latents.shape[2] < 1
        or tuple(latents.shape[-2:]) != expected_hw
    ):
        raise ValueError(
            f"Expected nonempty latents [B,{expected_channels},T,{expected_hw[0]},{expected_hw[1]}], "
            f"got {tuple(latents.shape)}."
        )
    if not torch.isfinite(latents).all():
        raise ValueError("LiDAR latents must be finite.")
    return latents.float().contiguous()


def _normalize_encoder_frames(frames, config: Mapping[str, Any]):
    torch = _torch()
    projection = config["range_projection"]
    if frames.ndim == 4:
        frames = frames.unsqueeze(0)
    expected_hw = (projection["native_height"], projection["semantic_width"])
    if (
        frames.ndim != 5
        or frames.shape[0] < 1
        or frames.shape[1] != 3
        or frames.shape[2] < 1
        or tuple(frames.shape[-2:]) != expected_hw
    ):
        raise ValueError(
            f"Expected encoder frames [B,3,T,{expected_hw[0]},{expected_hw[1]}] or the unbatched form, "
            f"got {tuple(frames.shape)}."
        )
    if not torch.isfinite(frames).all():
        raise ValueError("LiDAR encoder frames must be finite.")
    return frames.float().contiguous()


def _synthetic_latents(config: Mapping[str, Any], *, batch_size: int, frames: int, seed: int):
    torch = _torch()
    network = config["network_config"]
    height, width = (
        size // factor for size, factor in zip(network["resolution"], config["spatial_compression"], strict=True)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch_size, config["latent_channels"], frames, height, width, generator=generator)


def _prepare_encoder_input(frames, config: Mapping[str, Any]):
    """Apply the shared V1.2 metric normalization and circular width padding."""
    torch = _torch()
    projection = config["range_projection"]
    padding = projection["model_width"] - projection["semantic_width"]
    if padding < 0 or padding % 2:
        raise ValueError("LiDAR model width must permit symmetric padding of the semantic width.")
    half = padding // 2
    if half:
        frames = torch.cat((frames[..., -half:], frames, frames[..., :half]), dim=-1)
    minimum, maximum = projection["min_range_m"], projection["max_range_m"]
    ranges, intensities, validity = frames.float().split(1, dim=1)
    valid = (validity >= 0.5) & (ranges >= minimum) & (ranges <= maximum)
    ranges = (ranges.clamp(minimum, maximum) - minimum) / (maximum - minimum) * 2 - 1
    intensities = intensities.clamp(0, 1) * 2 - 1
    return torch.cat((ranges.masked_fill(~valid, -1), intensities.masked_fill(~valid, -1), valid.float()), dim=1)


def _postprocess(raw, config: Mapping[str, Any]) -> dict[str, Any]:
    """Produce validity probabilities and the public, cropped metric tensor."""
    torch = _torch()
    projection = config["range_projection"]
    expected_hw = (projection["native_height"], projection["model_width"])
    if raw.ndim != 5 or raw.shape[1] != 3 or tuple(raw.shape[-2:]) != expected_hw:
        raise ValueError(f"Unexpected raw decoder output shape {tuple(raw.shape)}; expected [B,3,T,{expected_hw}].")
    raw = raw.float()
    minimum, maximum = projection["min_range_m"], projection["max_range_m"]
    ranges = (raw[:, :1].clamp(-1, 1) + 1) * 0.5 * (maximum - minimum) + minimum
    intensity = (raw[:, 1:2].clamp(-1, 1) + 1) * 0.5
    validity = raw[:, 2:3].sigmoid()
    if config["apply_validity_mask"]:
        binary = validity >= projection.get("validity_threshold", 0.5)
        ranges = ranges.masked_fill(~binary, 0)
        intensity = intensity.masked_fill(~binary, 0)
        validity_channel = binary.float()
    else:
        binary = validity >= projection.get("validity_threshold", 0.5)
        validity_channel = validity
    metric = torch.cat((ranges, intensity, validity_channel), dim=1)
    offset = (projection["model_width"] - projection["semantic_width"]) // 2
    crop = slice(offset, offset + projection["semantic_width"])
    return {
        "decoder_raw": raw.contiguous(),
        "decoder_validity": validity[..., crop].contiguous(),
        "decoder_binary_validity": binary[..., crop].contiguous(),
        "decoder_metric": metric[..., crop].contiguous(),
    }


def _sync(device) -> None:
    if device.type == "cuda":
        current_omni_platform.synchronize()


def _rng_state(device) -> tuple[Any, Any | None]:
    torch = _torch()
    cpu = torch.random.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    return cpu, cuda


def _rng_equal(left: tuple[Any, Any | None], right: tuple[Any, Any | None]) -> bool:
    torch = _torch()
    return bool(torch.equal(left[0], right[0]) and (left[1] is None or torch.equal(left[1], right[1])))


def _outputs_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    torch = _torch()
    return left.keys() == right.keys() and all(torch.equal(left[name], right[name]) for name in left)


def _benchmark(
    function: Callable[[], Mapping[str, Any]],
    *,
    device,
    warmups: int,
    runs: int,
    after_first: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], Measurement]:
    """Measure compilation separately from warmed calls and verify determinism."""
    gc.collect()
    if device.type == "cuda":
        current_omni_platform.empty_cache()
        _sync(device)
        resident = current_omni_platform.memory_allocated()
        current_omni_platform.reset_peak_memory_stats()
    else:
        resident = None
    before_rng = _rng_state(device)
    started = time.perf_counter()
    first_device = dict(function())
    _sync(device)
    first_ms = (time.perf_counter() - started) * 1000
    first_peak = current_omni_platform.max_memory_allocated() - resident if resident is not None else None
    first = _cpu_tensors(first_device)
    del first_device
    if after_first is not None:
        after_first()

    identical = True
    for _ in range(warmups):
        current = function()
        _sync(device)
        identical = identical and _outputs_equal(first, _cpu_tensors(current))
        del current

    if device.type == "cuda":
        warmed_resident = current_omni_platform.memory_allocated()
        current_omni_platform.reset_peak_memory_stats()
    else:
        warmed_resident = None
    durations = []
    for _ in range(runs):
        started = time.perf_counter()
        current = function()
        _sync(device)
        durations.append((time.perf_counter() - started) * 1000)
        identical = identical and _outputs_equal(first, _cpu_tensors(current))
        del current
    warmed_peak = (
        current_omni_platform.max_memory_allocated() - warmed_resident if warmed_resident is not None else None
    )
    after_rng = _rng_state(device)
    measurement = Measurement(
        first_call_ms=first_ms,
        warmup_calls=warmups,
        timed_calls=runs,
        timed_ms=durations,
        mean_ms=sum(durations) / len(durations),
        min_ms=min(durations),
        max_ms=max(durations),
        resident_bytes=resident,
        first_call_peak_increment_bytes=first_peak,
        warmed_peak_increment_bytes=warmed_peak,
        rng_preserved=_rng_equal(before_rng, after_rng),
        repeated_output_identical=identical,
    )
    if not measurement.rng_preserved:
        raise RuntimeError("LiDAR validation changed the process RNG state.")
    if not measurement.repeated_output_identical:
        raise RuntimeError("Repeated LiDAR validation calls did not produce identical tensors.")
    return first, measurement


def _environment(mode: str, device) -> dict[str, Any]:
    torch = _torch()

    def version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    result = {
        "mode": mode,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "natten": version("natten"),
        "vllm_omni": version("vllm-omni"),
    }
    if device.type == "cuda":
        result.update(
            cuda_device_count=current_omni_platform.get_device_count(),
            cuda_device_name=torch.cuda.get_device_name(device),
            cuda_capability=list(torch.cuda.get_device_capability(device)),
        )
    return result


def _load_reference_model(component: Path, config: Mapping[str, Any], device):
    torch = _torch()
    from projects.cosmos3.tokenizer.lidar_tokenizer.network.transformer_vae import TransformerVAE
    from safetensors.torch import load_file

    model = TransformerVAE(**config["network_config"]).float()
    state = load_file(component / LIDAR_VAE_WEIGHTS, device="cpu")
    latent_mean = state.pop("latent_mean")
    latent_std = state.pop("latent_std")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    stats_shape = (1, config["latent_channels"], 1, 1, 1)
    return (
        model,
        latent_mean.reshape(stats_shape).to(device),
        latent_std.reshape(stats_shape).to(device),
    )


def _reference_decoder(model, latents, mean, std, config: Mapping[str, Any]) -> dict[str, Any]:
    with _torch().inference_mode():
        raw = model.decode_streaming(
            latents * std + mean,
            chunk_frames=config["streaming_chunk_frames"],
            context_frames=config["streaming_context_frames"],
            return_validity=False,
        )
        if raw.ndim == 4:
            raw = raw.unsqueeze(2)
        return _postprocess(raw, config)


def _reference_encoder(model, frames, mean, std, config: Mapping[str, Any]) -> dict[str, Any]:
    with _torch().inference_mode():
        normalized = _prepare_encoder_input(frames, config)
        latent = model.encode_streaming(
            normalized,
            chunk_frames=config["streaming_chunk_frames"],
            context_frames=config["streaming_context_frames"],
            sample_posterior=False,
        )
        return {"encoder_latents": ((latent - mean) / std).contiguous()}


def _cache_metadata() -> dict[str, Any]:
    from vllm_omni.diffusion.models.cosmos3.lidar_encoder import neighborhood_attention

    masks = neighborhood_attention._get_block_mask.cache_info()
    runners = neighborhood_attention._get_compiled_runner.cache_info()
    return {
        "block_masks": masks._asdict(),
        "compiled_runners": runners._asdict(),
    }


def _cache_reuse_check(first: Mapping[str, Any], final: Mapping[str, Any]) -> dict[str, Any]:
    details = {}
    for name in first:
        initial = first[name]
        current = final[name]
        details[name] = {
            "passed": (
                current["misses"] == initial["misses"]
                and current["currsize"] == initial["currsize"]
                and current["hits"] > initial["hits"]
            ),
            "after_first": initial,
            "after_repeats": current,
        }
    return {"passed": all(value["passed"] for value in details.values()), "details": details}


def _local_attention_hooks(model) -> tuple[list[Any], list[dict[str, Any]]]:
    from vllm_omni.diffusion.models.cosmos3.lidar_encoder.transformer_vae import (
        CircularNeighborhoodSelfAttentionBlock,
    )

    shapes: list[dict[str, Any]] = []

    def record(module, inputs, output) -> None:
        del output
        tensor = inputs[0]
        shape = {
            "batch_frames": tensor.shape[0],
            "height": tensor.shape[1],
            "width": tensor.shape[2],
            "channels": tensor.shape[3],
            "heads": module.num_heads,
            "kernel": list(module.kernel_size),
            "dilation": list(module.dilation),
        }
        if shape not in shapes:
            shapes.append(shape)

    handles = [
        module.register_forward_hook(record)
        for module in model.modules()
        if isinstance(module, CircularNeighborhoodSelfAttentionBlock)
    ]
    return handles, shapes


def _vllm_decoder_raw(model, latents, config: Mapping[str, Any]) -> dict[str, Any]:
    torch = _torch()
    with torch.inference_mode(), torch.autocast(device_type=model.coords.device.type, enabled=False):
        latents = latents.to(device=model.coords.device, dtype=torch.float32)
        latents = latents * model.latent_std + model.latent_mean
        chunk_size = config["streaming_chunk_frames"]
        context = config["streaming_context_frames"]
        outputs = []
        cache = None
        for start in range(0, latents.shape[2], chunk_size):
            latent = latents[:, :, start : start + chunk_size]
            batch, _, frames, height, width = latent.shape
            if cache is not None and context is not None:
                keep = context - frames
                cache = (
                    {
                        key: (key_cache[:, :, -keep:], value_cache[:, :, -keep:])
                        for key, (key_cache, value_cache) in cache.items()
                    }
                    if keep > 0
                    else None
                )
            latent = latent.permute(0, 2, 1, 3, 4).flatten(0, 1)
            latent = model.post_quant_conv(latent)
            latent = latent.reshape(batch, frames, -1, height, width).permute(0, 2, 1, 3, 4)
            output, cache = model.decoder(
                latent,
                model.coords,
                temporal_kv_cache=cache,
                return_temporal_kv_cache=True,
            )
            outputs.append(output)
        return _postprocess(torch.cat(outputs, dim=2), config)


def _vllm_encoder(model, frames) -> dict[str, Any]:
    # The runtime wrapper accepts one request at a time; preserve an optional
    # batch in the portable artifact by executing each request independently.
    outputs = [model(sample) for sample in frames]
    return {"encoder_latents": _torch().cat(outputs, dim=0)}


def _cpu_tensors(values: Mapping[str, Any]) -> dict[str, Any]:
    return {name: tensor.detach().to(device="cpu").contiguous() for name, tensor in values.items()}


def _write_artifact(path: Path, tensors: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(_cpu_tensors(tensors), temporary, metadata={ARTIFACT_METADATA_KEY: json.dumps(metadata, sort_keys=True)})
    os.replace(temporary, path)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(_json_dump(report))
    os.replace(temporary, path)


def _read_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        encoded = handle.metadata().get(ARTIFACT_METADATA_KEY)
        if encoded is None:
            raise ValueError(f"{path} is not a Cosmos3 LiDAR validation artifact.")
        metadata = json.loads(encoded)
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported validation artifact schema {metadata.get('schema_version')}; expected {SCHEMA_VERSION}."
        )
    return tensors, metadata


def _comparison(actual, expected, *, rtol: float, atol: float, exact: bool = False) -> dict[str, Any]:
    torch = _torch()
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": list(actual.shape), "reference_shape": list(expected.shape)}
    actual = actual.float()
    expected = expected.float()
    difference = (actual - expected).abs()
    relative = difference.double() / expected.double().abs().clamp_min(torch.finfo(torch.float32).eps)
    passed = torch.equal(actual, expected) if exact else torch.allclose(actual, expected, rtol=rtol, atol=atol)

    def finite_stat(value) -> float | None:
        result = value.item()
        return float(result) if torch.isfinite(torch.as_tensor(result)) else None

    return {
        "passed": bool(passed),
        "exact": exact,
        "shape": list(actual.shape),
        "nonfinite_actual": int((~torch.isfinite(actual)).sum().item()),
        "nonfinite_reference": int((~torch.isfinite(expected)).sum().item()),
        "max_abs_error": finite_stat(difference.max()) if difference.numel() else 0.0,
        "mean_abs_error": finite_stat(difference.mean()) if difference.numel() else 0.0,
        "max_rel_error": finite_stat(relative.max()) if relative.numel() else 0.0,
    }


def _dense_score_limit(local_shapes: Sequence[Mapping[str, Any]]) -> int | None:
    if not local_shapes:
        return None
    tokens = max(shape["height"] * shape["width"] for shape in local_shapes)
    return tokens * tokens * 4


def run_imaginaire4(args: argparse.Namespace) -> None:
    torch = _torch()
    imaginaire_root = args.imaginaire_root.resolve()
    if not imaginaire_root.is_dir():
        raise FileNotFoundError(f"imaginaire4 checkout not found at {imaginaire_root}.")
    sys.path.insert(0, str(imaginaire_root))
    root, component, config = _resolve_component(args.model)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--mode imaginaire4 requested CUDA, but torch.cuda.is_available() is false.")
    import natten

    fused_natten = False
    if device.type == "cuda" and callable(getattr(natten, "use_fused_na", None)):
        natten.use_fused_na(True)
        fused_natten = True

    if args.latents is None:
        latents = _synthetic_latents(config, batch_size=args.batch_size, frames=args.frames, seed=args.seed)
        input_kind = "synthetic"
    else:
        latents = _load_tensor_file(args.latents, "latents")
        input_kind = "saved"
    latents = _normalize_latents(latents, config).to(device)
    encoder_frames = None
    if args.encoder_input is not None:
        encoder_frames = _normalize_encoder_frames(_load_tensor_file(args.encoder_input, "frames"), config).to(device)
    if args.production_inputs and (args.latents is None or encoder_frames is None):
        raise ValueError("--production-inputs requires both --latents and --encoder-input.")

    model, mean, std = _load_reference_model(component, config, device)
    decoder_outputs, decoder_measurement = _benchmark(
        lambda: _reference_decoder(model, latents, mean, std, config),
        device=device,
        warmups=args.warmups,
        runs=args.runs,
    )
    tensors = {"latents": latents, **decoder_outputs}
    measurements: dict[str, Any] = {"decoder": asdict(decoder_measurement)}
    if encoder_frames is not None:
        encoder_outputs, encoder_measurement = _benchmark(
            lambda: _reference_encoder(model, encoder_frames, mean, std, config),
            device=device,
            warmups=args.warmups,
            runs=args.runs,
        )
        tensors.update(encoder_frames=encoder_frames, **encoder_outputs)
        measurements["encoder"] = asdict(encoder_measurement)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "producer": "imaginaire4",
        "model_identity": _model_identity(component, config),
        "model_path": str(root),
        "input_kind": input_kind,
        "production_inputs": bool(args.production_inputs),
        "seed": args.seed if input_kind == "synthetic" else None,
        "config": config,
        "environment": _environment("imaginaire4", device),
        "natten_fused": fused_natten,
        "measurements": measurements,
        "limitations": ["Persistent compiler caches were not cleared before first-call timing."],
    }
    _write_artifact(args.artifact, tensors, metadata)
    print(f"Wrote imaginaire4 reference artifact: {args.artifact}")


def run_vllm_omni(args: argparse.Namespace) -> None:
    torch = _torch()
    if args.report.resolve() == args.artifact.resolve():
        raise ValueError("--report must not overwrite the reference --artifact.")
    root, component, config = _resolve_component(args.model)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--mode vllm-omni requested CUDA, but torch.cuda.is_available() is false.")
    reference, reference_metadata = _read_artifact(args.artifact)
    identity = _model_identity(component, config)
    if reference_metadata["model_identity"] != identity:
        raise ValueError(
            "Reference artifact and candidate checkpoint differ: "
            f"reference={reference_metadata['model_identity']}, candidate={identity}."
        )
    required = {"latents", "decoder_raw", "decoder_validity", "decoder_binary_validity", "decoder_metric"}
    if missing := required - reference.keys():
        raise ValueError(f"Reference artifact is missing tensors: {sorted(missing)}.")

    from vllm_omni.diffusion.models.cosmos3.lidar import Cosmos3LidarDecoder, Cosmos3LidarEncoder
    from vllm_omni.diffusion.models.cosmos3.lidar_encoder import neighborhood_attention

    neighborhood_attention._get_block_mask.cache_clear()
    neighborhood_attention._get_compiled_runner.cache_clear()
    latents = _normalize_latents(reference["latents"], config).to(device)
    decoder = Cosmos3LidarDecoder.from_pretrained(str(root), config, device)
    handles, decoder_shapes = _local_attention_hooks(decoder)
    decoder_cache_first: dict[str, Any] = {}

    def remove_decoder_hooks() -> None:
        for handle in handles:
            handle.remove()
        decoder_cache_first.update(_cache_metadata())

    decoder_outputs, decoder_measurement = _benchmark(
        lambda: _vllm_decoder_raw(decoder, latents, config),
        device=device,
        warmups=args.warmups,
        runs=args.runs,
        after_first=remove_decoder_hooks,
    )
    actual = _cpu_tensors(decoder_outputs)
    measurements: dict[str, Any] = {"decoder": asdict(decoder_measurement)}
    local_shapes: dict[str, Any] = {"decoder": decoder_shapes}
    decoder_cache_final = _cache_metadata()
    cache = {
        "decoder": _cache_reuse_check(decoder_cache_first, decoder_cache_final),
        "after_decoder": decoder_cache_final,
    }
    decoder = None
    decoder_outputs = None
    latents = None
    gc.collect()
    if device.type == "cuda":
        current_omni_platform.empty_cache()

    if "encoder_frames" in reference:
        encoder = Cosmos3LidarEncoder.from_pretrained(str(root), config, device)
        handles, encoder_shapes = _local_attention_hooks(encoder)
        encoder_cache_first: dict[str, Any] = {}

        def remove_encoder_hooks() -> None:
            for handle in handles:
                handle.remove()
            encoder_cache_first.update(_cache_metadata())

        encoder_frames = _normalize_encoder_frames(reference["encoder_frames"], config).to(device)
        encoder_outputs, encoder_measurement = _benchmark(
            lambda: _vllm_encoder(encoder, encoder_frames),
            device=device,
            warmups=args.warmups,
            runs=args.runs,
            after_first=remove_encoder_hooks,
        )
        actual.update(_cpu_tensors(encoder_outputs))
        measurements["encoder"] = asdict(encoder_measurement)
        local_shapes["encoder"] = encoder_shapes
        encoder_cache_final = _cache_metadata()
        cache["encoder"] = _cache_reuse_check(encoder_cache_first, encoder_cache_final)
        cache["after_encoder"] = encoder_cache_final

    comparisons = {
        "decoder_raw_range_intensity": _comparison(
            actual["decoder_raw"][:, :2], reference["decoder_raw"][:, :2], rtol=args.rtol, atol=args.atol
        ),
        "decoder_raw_validity_logits": _comparison(
            actual["decoder_raw"][:, 2:], reference["decoder_raw"][:, 2:], rtol=args.rtol, atol=args.atol
        ),
        "decoder_validity_probability": _comparison(
            actual["decoder_validity"], reference["decoder_validity"], rtol=args.rtol, atol=args.atol
        ),
        "decoder_metric_range_intensity": _comparison(
            actual["decoder_metric"][:, :2], reference["decoder_metric"][:, :2], rtol=args.rtol, atol=args.atol
        ),
        "decoder_binary_validity": _comparison(
            actual["decoder_binary_validity"],
            reference["decoder_binary_validity"],
            rtol=0,
            atol=0,
            exact=True,
        ),
    }
    if "encoder_latents" in reference:
        if "encoder_latents" not in actual:
            raise ValueError("Reference artifact has encoder output but no encoder input.")
        comparisons["encoder_normalized_latents"] = _comparison(
            actual["encoder_latents"], reference["encoder_latents"], rtol=args.rtol, atol=args.atol
        )

    memory_checks = {}
    for component_name, shapes in local_shapes.items():
        dense_limit = _dense_score_limit(shapes)
        warmed_peak = measurements[component_name]["warmed_peak_increment_bytes"]
        memory_checks[component_name] = {
            "passed": dense_limit is not None and warmed_peak is not None and warmed_peak < dense_limit,
            "warmed_peak_increment_bytes": warmed_peak,
            "dense_fp32_score_matrix_bytes": dense_limit,
        }
    numeric_passed = all(result["passed"] for result in comparisons.values())
    memory_passed = bool(memory_checks) and all(result["passed"] for result in memory_checks.values())
    cache_passed = all(value["passed"] for key, value in cache.items() if not key.startswith("after_"))
    state_passed = all(
        result["rng_preserved"] and result["repeated_output_identical"] for result in measurements.values()
    )
    passed = numeric_passed and memory_passed and cache_passed and state_passed
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "production_inputs": reference_metadata["production_inputs"],
        "production_qualified": passed and reference_metadata["production_inputs"],
        "tolerances": {"rtol": args.rtol, "atol": args.atol},
        "model_identity": identity,
        "model_path": str(root),
        "config": config,
        "reference_artifact": str(args.artifact.resolve()),
        "comparisons": comparisons,
        "memory_checks": memory_checks,
        "local_attention_shapes": local_shapes,
        "cache": cache,
        "reference": {
            "environment": reference_metadata["environment"],
            "measurements": reference_metadata["measurements"],
        },
        "candidate": {
            "environment": _environment("vllm-omni", device),
            "measurements": measurements,
        },
        "limitations": ["Persistent compiler caches were not cleared before first-call timing."],
    }
    _write_report(args.report, report)
    print(f"Wrote vLLM-Omni comparison report: {args.report}")
    print(f"passed={passed} production_qualified={report['production_qualified']}")
    if not passed:
        raise SystemExit(1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("imaginaire4", "vllm-omni"))
    parser.add_argument("--model", required=True, help="Local exported checkpoint root (or its lidar_vae directory).")
    parser.add_argument(
        "--artifact",
        required=True,
        type=Path,
        help="Reference safetensors file to create in imaginaire4 mode or consume in vllm-omni mode.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device; CUDA is required for production qualification.")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)

    reference = parser.add_argument_group("imaginaire4 mode")
    reference.add_argument("--imaginaire-root", type=Path, default=Path("../imaginaire4"))
    reference.add_argument("--latents", type=Path, help="Optional safetensors input containing normalized `latents`.")
    reference.add_argument(
        "--encoder-input", type=Path, help="Optional safetensors input containing metric FP32 `frames`."
    )
    reference.add_argument("--frames", type=int, default=19, help="Synthetic latent frame count.")
    reference.add_argument("--batch-size", type=int, default=1, help="Synthetic latent batch size.")
    reference.add_argument("--seed", type=int, default=42, help="Synthetic latent seed.")
    reference.add_argument(
        "--production-inputs",
        action="store_true",
        help="Mark supplied --latents and --encoder-input as real production pipeline inputs.",
    )

    candidate = parser.add_argument_group("vllm-omni mode")
    candidate.add_argument("--report", type=Path, help="JSON report path (required in vllm-omni mode).")
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.runs < 1:
        parser.error("--warmups must be nonnegative and --runs must be positive.")
    if args.frames < 1 or args.batch_size < 1:
        parser.error("--frames and --batch-size must be positive.")
    if args.rtol < 0 or args.atol < 0:
        parser.error("--rtol and --atol must be nonnegative.")
    if args.mode == "vllm-omni":
        if args.report is None:
            parser.error("--report is required in vllm-omni mode.")
        if args.latents is not None or args.encoder_input is not None or args.production_inputs:
            parser.error("vllm-omni mode takes all inputs from --artifact; input flags belong to imaginaire4 mode.")
    elif args.report is not None:
        parser.error("--report belongs to vllm-omni mode; imaginaire4 writes metadata into --artifact.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "imaginaire4":
        run_imaginaire4(args)
    else:
        run_vllm_omni(args)


if __name__ == "__main__":
    main()
