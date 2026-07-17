# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile MiniCPM-o 4.5 vision attention with real image/video inputs.

The benchmark loads the official VPM weights, preprocesses media through the
MiniCPM-o image processor, and compares the production attention path against a
prototype that computes unpadding metadata once per vision-model forward.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_llm as minicpmo_model

try:
    from benchmarks.minicpmo import profile_vision_encoder as encoder_profiler
except ModuleNotFoundError:
    import profile_vision_encoder as encoder_profiler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencyStats:
    samples_ms: tuple[float, ...]
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float


@dataclass(frozen=True)
class MediaWorkload:
    name: str
    description: str
    source_paths: tuple[str, ...]
    source_sizes: tuple[tuple[int, int], ...]
    preprocess_ms: float
    pixel_values: torch.Tensor
    tgt_sizes: torch.Tensor
    patch_attention_mask: torch.Tensor
    valid_lengths: tuple[int, ...]

    @property
    def padding_fraction(self) -> float:
        total_tokens = len(self.valid_lengths) * max(self.valid_lengths)
        return 1.0 - sum(self.valid_lengths) / total_tokens

    @property
    def uses_varlen_attention(self) -> bool:
        return self.padding_fraction > 0.0


class MediaSiglipVisionModel(nn.Module):
    """Production SigLIP modules without the Transformers dispatch wrapper."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.embeddings = minicpmo_model.SiglipVisionEmbeddings(config)
        self.encoder = minicpmo_model.SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor,
        tgt_sizes: torch.Tensor,
        force_varlen: bool = False,
        reuse_unpadding_metadata: bool = True,
    ) -> torch.Tensor:
        batch_size = pixel_values.size(0)
        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            tgt_sizes=tgt_sizes,
        )
        flat_attention_mask = patch_attention_mask.view(batch_size, -1)
        attention_mask = flat_attention_mask if force_varlen or torch.any(~flat_attention_mask) else None
        if reuse_unpadding_metadata:
            hidden_states = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )[0]
        else:
            for encoder_layer in self.encoder.layers:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    unpadding_metadata=None,
                )[0]
        return self.post_layernorm(hidden_states)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_stats(samples_ms: Sequence[float]) -> LatencyStats:
    samples = tuple(samples_ms)
    if not samples:
        raise ValueError("At least one latency sample is required")
    return LatencyStats(
        samples_ms=samples,
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        p95_ms=_percentile(samples, 0.95),
        min_ms=min(samples),
        max_ms=max(samples),
        stdev_ms=statistics.pstdev(samples),
    )


def _stats_to_dict(stats: LatencyStats) -> dict[str, Any]:
    return {
        "samples_ms": list(stats.samples_ms),
        "mean_ms": stats.mean_ms,
        "median_ms": stats.median_ms,
        "p95_ms": stats.p95_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
        "stdev_ms": stats.stdev_ms,
    }


def _parse_dtype(value: str) -> torch.dtype:
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    try:
        return dtypes[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {value}") from exc


def _load_model_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _build_image_processor(config: dict[str, Any]) -> Any:
    slice_config = config["slice_config"]
    vision_config = config["vision_config"]
    return minicpmo_model.MiniCPMVImageProcessor(
        max_slice_nums=slice_config["max_slice_nums"],
        scale_resolution=config["image_size"],
        patch_size=vision_config["patch_size"],
        use_image_id=config["use_image_id"],
        image_feature_size=config["query_num"],
    )


def _load_vpm_weights(
    model: nn.Module,
    model_path: Path,
    device: str,
    dtype: torch.dtype,
) -> int:
    index_path = model_path / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    vpm_keys = {key: shard for key, shard in weight_map.items() if key.startswith("vpm.")}
    if not vpm_keys:
        raise RuntimeError("No vpm.* weights found in the model index")

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(vpm_keys.values())):
        shard_keys = [key for key, shard in vpm_keys.items() if shard == shard_name]
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for full_key in shard_keys:
                tensor = handle.get_tensor(full_key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                state_dict[full_key.removeprefix("vpm.")] = tensor.to(device=device)

    model.load_state_dict(state_dict, strict=True)
    return len(state_dict)


def _build_vision_model(
    config: dict[str, Any],
    model_path: Path,
    device: str,
    dtype: torch.dtype,
) -> tuple[nn.Module, dict[str, Any]]:
    vision_config = minicpmo_model.SiglipVisionConfig(**config["vision_config"])
    vision_config._attn_implementation = "flash_attention_2"

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = MediaSiglipVisionModel(vision_config)
    finally:
        torch.set_default_dtype(previous_dtype)

    loaded_tensor_count = _load_vpm_weights(model, model_path, device, dtype)
    configured_layers = len(model.encoder.layers)
    if config.get("drop_vision_last_layer", True):
        model.encoder.layers = model.encoder.layers[:-1]
    model.eval()
    metadata = {
        "configured_layers": configured_layers,
        "active_layers": len(model.encoder.layers),
        "loaded_tensor_count": loaded_tensor_count,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, metadata


def _decode_video(video_path: Path, frame_count: int) -> tuple[list[Image.Image], dict[str, Any]]:
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    indices = np.linspace(0, total_frames - 1, num=min(frame_count, total_frames), dtype=np.int64)
    frames: list[Image.Image] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Unable to decode frame {index} from {video_path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    capture.release()
    return frames, {
        "total_frames": total_frames,
        "sampled_indices": indices.tolist(),
        "fps": fps,
        "decode_ms": (time.perf_counter() - started) * 1000.0,
    }


def _prepare_workload(
    name: str,
    description: str,
    source_paths: Sequence[Path],
    image_groups: list[list[Image.Image]],
    processor: Any,
    max_slice_nums: int,
    patch_size: int,
    device: str,
    dtype: torch.dtype,
) -> MediaWorkload:
    started = time.perf_counter()
    processed = processor.preprocess(
        images=image_groups,
        max_slice_nums=max_slice_nums,
        return_tensors="pt",
    )

    flat_pixels: list[torch.Tensor] = []
    target_sizes: list[torch.Tensor] = []
    for pixel_group, size_group in zip(processed["pixel_values"], processed["tgt_sizes"], strict=True):
        for pixel_values in pixel_group:
            tensor = pixel_values if isinstance(pixel_values, torch.Tensor) else torch.from_numpy(pixel_values)
            flat_pixels.append(tensor.flatten(end_dim=1).permute(1, 0))
        target_sizes.append(size_group if isinstance(size_group, torch.Tensor) else torch.from_numpy(size_group))

    tgt_sizes = torch.vstack(target_sizes).to(dtype=torch.int32)
    if len(flat_pixels) != tgt_sizes.shape[0]:
        raise RuntimeError(
            f"Preprocessor returned {len(flat_pixels)} pixel tensors but {tgt_sizes.shape[0]} target sizes"
        )
    valid_lengths = tuple(int(value) for value in tgt_sizes.prod(dim=-1).tolist())
    padded_pixels = torch.nn.utils.rnn.pad_sequence(flat_pixels, batch_first=True, padding_value=0.0)
    batch_size, flattened_length, flattened_channels = padded_pixels.shape
    if flattened_channels != 3 * patch_size:
        raise RuntimeError(f"Unexpected flattened channel count: {flattened_channels}")
    pixel_values = padded_pixels.permute(0, 2, 1).reshape(
        batch_size,
        3,
        patch_size,
        flattened_length,
    )
    max_patches = max(valid_lengths)
    attention_mask = torch.zeros((batch_size, 1, max_patches), dtype=torch.bool)
    for index, valid_length in enumerate(valid_lengths):
        attention_mask[index, 0, :valid_length] = True

    preprocess_ms = (time.perf_counter() - started) * 1000.0
    source_sizes = tuple(image.size for group in image_groups for image in group)
    return MediaWorkload(
        name=name,
        description=description,
        source_paths=tuple(str(path) for path in source_paths),
        source_sizes=source_sizes,
        preprocess_ms=preprocess_ms,
        pixel_values=pixel_values.to(device=device, dtype=dtype),
        tgt_sizes=tgt_sizes.to(device=device),
        patch_attention_mask=attention_mask.to(device=device),
        valid_lengths=valid_lengths,
    )


def _run_model(
    model: nn.Module,
    workload: MediaWorkload,
    *,
    reuse_unpadding_metadata: bool = True,
) -> torch.Tensor:
    return model(
        pixel_values=workload.pixel_values,
        patch_attention_mask=workload.patch_attention_mask,
        tgt_sizes=workload.tgt_sizes,
        reuse_unpadding_metadata=reuse_unpadding_metadata,
    )


def _benchmark_runner(
    runner: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
) -> LatencyStats:
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    with torch.inference_mode():
        for _ in range(warmup):
            output = runner()
            del output
        torch.accelerator.synchronize()
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = runner()
            end.record()
            event_pairs.append((start, end))
            del output
        torch.accelerator.synchronize()
    return _latency_stats([start.elapsed_time(end) for start, end in event_pairs])


def _combine_stats(*stats: LatencyStats) -> LatencyStats:
    samples = [sample for item in stats for sample in item.samples_ms]
    return _latency_stats(samples)


def _benchmark_paired_runners(
    baseline_runner: Callable[[], torch.Tensor],
    metadata_reuse_runner: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
) -> tuple[LatencyStats, LatencyStats]:
    event_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        "baseline": [],
        "metadata_reuse": [],
    }
    runners = {
        "baseline": baseline_runner,
        "metadata_reuse": metadata_reuse_runner,
    }
    with torch.inference_mode():
        for index in range(warmup):
            order = ("baseline", "metadata_reuse") if index % 2 == 0 else ("metadata_reuse", "baseline")
            for name in order:
                output = runners[name]()
                del output
        torch.accelerator.synchronize()
        for index in range(iterations):
            order = ("baseline", "metadata_reuse") if index % 2 == 0 else ("metadata_reuse", "baseline")
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = runners[name]()
                end.record()
                event_pairs[name].append((start, end))
                del output
        torch.accelerator.synchronize()
    return (
        _latency_stats([start.elapsed_time(end) for start, end in event_pairs["baseline"]]),
        _latency_stats([start.elapsed_time(end) for start, end in event_pairs["metadata_reuse"]]),
    )


def _parity_against_unpadded_samples(
    model: nn.Module,
    workload: MediaWorkload,
    patch_size: int,
    batched_output: torch.Tensor,
) -> dict[str, Any]:
    sample_metrics = []
    with torch.inference_mode():
        batched_embeddings = model.embeddings(
            pixel_values=workload.pixel_values,
            patch_attention_mask=workload.patch_attention_mask,
            tgt_sizes=workload.tgt_sizes,
        )
        for index, valid_length in enumerate(workload.valid_lengths):
            pixel_values = workload.pixel_values[index : index + 1, :, :, : valid_length * patch_size]
            tgt_sizes = workload.tgt_sizes[index : index + 1]
            attention_mask = torch.ones(
                (1, 1, valid_length),
                dtype=torch.bool,
                device=workload.pixel_values.device,
            )
            unpadded_embeddings = model.embeddings(
                pixel_values=pixel_values,
                patch_attention_mask=attention_mask,
                tgt_sizes=tgt_sizes,
            )
            output = model(
                pixel_values=pixel_values,
                patch_attention_mask=attention_mask,
                tgt_sizes=tgt_sizes,
                force_varlen=workload.uses_varlen_attention,
            )
            actual = batched_output[index : index + 1, :valid_length]
            expected = output[:, :valid_length]
            diff = actual.float() - expected.float()
            embedding_diff = (
                batched_embeddings[index : index + 1, :valid_length].float()
                - unpadded_embeddings[:, :valid_length].float()
            )
            reference_rms = float(expected.float().square().mean().sqrt().item())
            rmse = float(diff.square().mean().sqrt().item())
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    actual.float().flatten(),
                    expected.float().flatten(),
                    dim=0,
                ).item()
            )
            sample_metrics.append(
                {
                    "sample": index,
                    "embedding_max_abs_diff": float(embedding_diff.abs().max().item()),
                    "max_abs_diff": float(diff.abs().max().item()),
                    "mean_abs_diff": float(diff.abs().mean().item()),
                    "rmse": rmse,
                    "relative_rmse": rmse / reference_rms if reference_rms else 0.0,
                    "cosine_similarity": cosine,
                }
            )
    return {
        "per_sample": sample_metrics,
        "embedding_max_abs_diff": max(
            (item["embedding_max_abs_diff"] for item in sample_metrics),
            default=0.0,
        ),
        "max_abs_diff": max((item["max_abs_diff"] for item in sample_metrics), default=0.0),
        "max_relative_rmse": max(
            (item["relative_rmse"] for item in sample_metrics),
            default=0.0,
        ),
        "min_cosine_similarity": min(
            (item["cosine_similarity"] for item in sample_metrics),
            default=1.0,
        ),
    }


def _profile_workload(
    model: nn.Module,
    workload: MediaWorkload,
    output_dir: Path,
    *,
    reuse_unpadding_metadata: bool,
    variant: str,
) -> dict[str, Any]:
    state = encoder_profiler.InstrumentationState(enabled=True, record_ranges=True)
    originals = encoder_profiler._install_function_ranges(state)
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
        ) as profiler:
            with torch.inference_mode(), record_function(f"minicpmo.media.{workload.name}.{variant}"):
                output = _run_model(
                    model,
                    workload,
                    reuse_unpadding_metadata=reuse_unpadding_metadata,
                )
                del output
                torch.accelerator.synchronize()
    finally:
        encoder_profiler._restore_functions(originals)

    trace_path = output_dir / f"trace_{workload.name}_{variant}.json"
    profiler.export_chrome_trace(str(trace_path))
    events = list(profiler.key_averages())
    return {
        "trace_path": str(trace_path),
        "categories": encoder_profiler._aggregate_profile_categories(events),
    }


def _run_workload(
    model: nn.Module,
    workload: MediaWorkload,
    patch_size: int,
    warmup: int,
    iterations: int,
    output_dir: Path,
) -> dict[str, Any]:
    def baseline_runner() -> torch.Tensor:
        return _run_model(
            model,
            workload,
            reuse_unpadding_metadata=False,
        )

    def optimized_runner() -> torch.Tensor:
        return _run_model(
            model,
            workload,
            reuse_unpadding_metadata=True,
        )

    baseline_before = _benchmark_runner(baseline_runner, warmup, iterations)
    with torch.inference_mode():
        baseline_output = baseline_runner()
        torch.accelerator.synchronize()

    unpadded_parity = _parity_against_unpadded_samples(
        model,
        workload,
        patch_size,
        baseline_output,
    )
    baseline_profiler = _profile_workload(
        model,
        workload,
        output_dir,
        reuse_unpadding_metadata=False,
        variant="baseline",
    )
    optimized_profiler = _profile_workload(
        model,
        workload,
        output_dir,
        reuse_unpadding_metadata=True,
        variant="optimized",
    )

    optimized_stats = _benchmark_runner(optimized_runner, warmup, iterations)
    with torch.inference_mode():
        optimized_output = optimized_runner()
        torch.accelerator.synchronize()
    optimized_max_abs = float((baseline_output - optimized_output).abs().max().item())
    torch.testing.assert_close(baseline_output, optimized_output, atol=0.0, rtol=0.0)
    paired_baseline, paired_optimized = _benchmark_paired_runners(
        baseline_runner,
        optimized_runner,
        warmup,
        iterations,
    )

    baseline_after = _benchmark_runner(baseline_runner, warmup, iterations)
    baseline_combined = _combine_stats(baseline_before, baseline_after)
    delta_ms = paired_optimized.mean_ms - paired_baseline.mean_ms
    return {
        "description": workload.description,
        "source_paths": list(workload.source_paths),
        "source_sizes": [list(size) for size in workload.source_sizes],
        "preprocess_ms": workload.preprocess_ms,
        "batch_size": len(workload.valid_lengths),
        "valid_lengths": list(workload.valid_lengths),
        "max_seq_len": max(workload.valid_lengths),
        "total_valid_tokens": sum(workload.valid_lengths),
        "padding_fraction": workload.padding_fraction,
        "attention_path": "varlen" if workload.uses_varlen_attention else "dense",
        "baseline_before": _stats_to_dict(baseline_before),
        "baseline_after": _stats_to_dict(baseline_after),
        "baseline_combined": _stats_to_dict(baseline_combined),
        "metadata_reuse": _stats_to_dict(optimized_stats),
        "paired_baseline": _stats_to_dict(paired_baseline),
        "paired_metadata_reuse": _stats_to_dict(paired_optimized),
        "metadata_reuse_delta_ms": delta_ms,
        "metadata_reuse_delta_percent": delta_ms / paired_baseline.mean_ms * 100.0,
        "metadata_reuse_speedup": paired_baseline.mean_ms / paired_optimized.mean_ms,
        "metadata_reuse_max_abs_diff": optimized_max_abs,
        "padded_vs_per_sample_unpadded": unpadded_parity,
        "profiler_baseline": baseline_profiler,
        "profiler_optimized": optimized_profiler,
    }


def _write_csv(path: Path, workloads: dict[str, dict[str, Any]]) -> None:
    fields = (
        "workload",
        "attention_path",
        "batch_size",
        "padding_fraction",
        "baseline_mean_ms",
        "baseline_p95_ms",
        "metadata_reuse_mean_ms",
        "metadata_reuse_p95_ms",
        "delta_ms",
        "delta_percent",
        "max_abs_diff",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in workloads.items():
            writer.writerow(
                {
                    "workload": name,
                    "attention_path": result["attention_path"],
                    "batch_size": result["batch_size"],
                    "padding_fraction": result["padding_fraction"],
                    "baseline_mean_ms": result["paired_baseline"]["mean_ms"],
                    "baseline_p95_ms": result["paired_baseline"]["p95_ms"],
                    "metadata_reuse_mean_ms": result["paired_metadata_reuse"]["mean_ms"],
                    "metadata_reuse_p95_ms": result["paired_metadata_reuse"]["p95_ms"],
                    "delta_ms": result["metadata_reuse_delta_ms"],
                    "delta_percent": result["metadata_reuse_delta_percent"],
                    "max_abs_diff": result["metadata_reuse_max_abs_diff"],
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/autodl-tmp/models/OpenBMB/MiniCPM-o-4_5"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--video-frames", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/minicpmo_vision_media"),
    )
    args = parser.parse_args()
    for name in ("warmup", "iterations", "video_frames"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(torch.device(args.device).index or 0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    config = _load_model_config(args.model_path)
    processor = _build_image_processor(config)
    patch_size = int(config["vision_config"]["patch_size"])
    max_slice_nums = int(config["slice_config"]["max_slice_nums"])
    asset_dir = args.model_path / "assets"
    fossil_path = asset_dir / "fossil.png"
    highway_path = asset_dir / "highway.png"
    video_path = asset_dir / "Skiing.mp4"
    for path in (fossil_path, highway_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing model asset: {path}")

    fossil = Image.open(fossil_path).convert("RGB")
    highway = Image.open(highway_path).convert("RGB")
    frames, video_metadata = _decode_video(video_path, args.video_frames)

    workloads = [
        _prepare_workload(
            "single_image",
            "Official fossil.png processed as one image.",
            [fossil_path],
            [[fossil]],
            processor,
            max_slice_nums,
            patch_size,
            args.device,
            args.dtype,
        ),
        _prepare_workload(
            "mixed_image_batch",
            "Official fossil.png and highway.png processed as a heterogeneous image batch.",
            [fossil_path, highway_path],
            [[fossil], [highway]],
            processor,
            max_slice_nums,
            patch_size,
            args.device,
            args.dtype,
        ),
        _prepare_workload(
            "short_video",
            "Uniformly sampled frames decoded from the official Skiing.mp4 asset.",
            [video_path],
            [frames],
            processor,
            1,
            patch_size,
            args.device,
            args.dtype,
        ),
    ]

    logger.info("Loading official VPM weights from %s", args.model_path)
    model, model_metadata = _build_vision_model(
        config,
        args.model_path,
        args.device,
        args.dtype,
    )
    logger.info(
        "Loaded %d tensors; active layers=%d; parameters=%d",
        model_metadata["loaded_tensor_count"],
        model_metadata["active_layers"],
        model_metadata["parameter_count"],
    )

    summary: dict[str, Any] = {
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.accelerator.current_device_index()),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "config": {
            "model_path": str(args.model_path),
            "dtype": str(args.dtype),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "video_frames": args.video_frames,
            "max_slice_nums": max_slice_nums,
            "image_size": config["image_size"],
            "vision_config": config["vision_config"],
            "drop_vision_last_layer": config.get("drop_vision_last_layer", True),
        },
        "model": model_metadata,
        "video_decode": video_metadata,
        "workloads": {},
    }

    for workload in workloads:
        logger.info(
            "Running %s: batch=%d valid_lengths=%s padding=%.2f%% path=%s",
            workload.name,
            len(workload.valid_lengths),
            workload.valid_lengths,
            workload.padding_fraction * 100.0,
            "varlen" if workload.uses_varlen_attention else "dense",
        )
        result = _run_workload(
            model,
            workload,
            patch_size,
            args.warmup,
            args.iterations,
            args.output_dir,
        )
        summary["workloads"][workload.name] = result
        logger.info(
            "%s baseline=%.3f ms metadata_reuse=%.3f ms delta=%+.3f ms (%+.2f%%) parity=%g",
            workload.name,
            result["paired_baseline"]["mean_ms"],
            result["paired_metadata_reuse"]["mean_ms"],
            result["metadata_reuse_delta_ms"],
            result["metadata_reuse_delta_percent"],
            result["metadata_reuse_max_abs_diff"],
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "workloads.csv"
    _write_csv(csv_path, summary["workloads"])
    logger.info("Summary: %s", summary_path)
    logger.info("CSV: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
