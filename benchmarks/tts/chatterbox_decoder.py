#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Profile real-weight S3Gen and sweep acoustic decoder batch sizes.

Replays real speech codes from the built-in voice, not newly generated T3
outputs. Isolated decoder capacity estimates are not serving measurements.
"""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from vllm_omni.config.stage_config import resolve_deploy_yaml
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import (
    REFERENCE_ENCODER_PREFIXES,
    Chunk,
    Reference,
    S3GenDecoder,
)
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioning
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "vllm_omni" / "deploy"
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
REPEATS = 5
WARMUPS = 2


def main():
    """Load one variant, measure stages, and export a CPU/CUDA trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("variant", choices=("turbo", "original"))
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--fp32-flow", action="store_true", help="Run the flow in FP32 instead of production's BF16 autocast"
    )
    parser.add_argument("--tf32", action="store_true", help="Use PyTorch high-precision FP32 matmul policy")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    args = parser.parse_args()
    if any(size not in BATCH_SIZES for size in args.batch_sizes):
        parser.error("batch sizes must be powers of two from 1 through 64")
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    config = ChatterboxConfig(
        args.variant,
        flow_dtype="float32" if args.fp32_flow else "bfloat16",
    )
    # Stage 1 as the variant's deploy file sizes it.
    deploy = resolve_deploy_yaml(
        DEPLOY_DIR / ("chatterbox_turbo.yaml" if args.variant == "turbo" else "chatterbox.yaml")
    )
    decoder = S3GenDecoder(
        config,
        deploy["stages"][1]["max_model_len"],
        deploy["connectors"]["connector_of_shared_memory"]["extra"]["codec_left_context_frames"],
        deploy["stages"][1]["seed"],
    ).eval()
    weights = load_file(str(args.model / config.s3gen_weights))
    decoder.load_state_dict(
        {name: value for name, value in weights.items() if not name.startswith(REFERENCE_ENCODER_PREFIXES)},
        strict=True,
    )
    del weights
    decoder.to("cuda")
    voice = VoiceConditioning.from_builtin(str(args.model))
    reference = Reference(voice.prompt_token.cuda(), voice.prompt_feat.cuda(), voice.embedding.cuda())
    # Use actual quantized speech codes at different positions for each row.
    codes = voice.cond_tokens[0].cuda()
    if codes.numel() < 113:
        raise ValueError("built-in speech needs at least 113 codes for the steady chunk probe")
    report = {
        "variant": args.variant,
        "model": str(args.model),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "flow_dtype": config.flow_dtype,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "workload": "isolated real-weight decoder with built-in speech-code replay",
        "measurements": [],
        "batch_sizes": args.batch_sizes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.fp32_flow or args.tf32:
        probe = make_chunks(decoder, reference, codes, 1, "steady")
        torch.set_float32_matmul_precision("highest")
        config.flow_dtype = "float32"
        expected = decoder.chunk_mels(probe)[0]
        config.flow_dtype = "float32" if args.fp32_flow else "bfloat16"
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
        actual = decoder.chunk_mels(probe)[0]
        torch.manual_seed(2718)
        expected_audio = decoder.vocode([expected], expected.new_zeros(1, 1, 0))[0][0]
        torch.manual_seed(2718)
        actual_audio = decoder.vocode([actual], actual.new_zeros(1, 1, 0))[0][0]
        for label, waveform in (("fp32", expected_audio), ("selected", actual_audio)):
            sf.write(
                str(args.output.with_suffix(f".{label}.wav")),
                waveform.flatten().numpy(force=True),
                config.sample_rate,
                subtype="FLOAT",
            )
        report["precision_probe"] = {
            "mel_max_absolute_error": float((expected - actual).abs().max()),
            "mel_rmse": float((expected - actual).square().mean().sqrt()),
            "audio_cosine_similarity": float(
                torch.nn.functional.cosine_similarity(expected_audio.flatten(), actual_audio.flatten(), dim=0)
            ),
            "audio_finite": bool(torch.isfinite(actual_audio).all()),
            "scope": "one real-code replay chunk; not an ASR or speaker-quality evaluation",
        }
    for scenario in tqdm(("first", "steady", "final"), desc=args.variant):
        for batch_size in args.batch_sizes:
            try:
                chunks = make_chunks(decoder, reference, codes, batch_size, scenario)
                mels = decoder.chunk_mels(chunks)
                # This isolates HiFT; full streaming below includes cache and overlap handling.
                cache = mels[0].new_zeros(batch_size, 1, 0)
                operations = {
                    "flow": lambda: decoder.chunk_mels(chunks),
                    "vocoder": lambda: decoder.vocode(mels, cache),
                    "streaming": lambda: decoder.chunked_decode_streaming(chunks),
                }
                row = {"scenario": scenario, "batch_size": batch_size, "stages": {}}
                for name, operation in operations.items():
                    row["stages"][name] = measure(operation)
                audio = decoder.chunked_decode_streaming(chunks)[0][0]
                seconds = audio.shape[-1] / config.sample_rate
                elapsed = row["stages"]["streaming"]["wall_seconds_median"]
                row.update(
                    audio_seconds_per_row=seconds,
                    isolated_rtf=elapsed / seconds,
                    estimated_sequential_64_stream_rtf=elapsed * (64 // batch_size) / seconds,
                )
                report["measurements"].append(row)
                logging.info(
                    "%s B%s streaming %.4fs, isolated 64-stream RTF %.3f",
                    scenario,
                    batch_size,
                    elapsed,
                    row["estimated_sequential_64_stream_rtf"],
                )
            except torch.cuda.OutOfMemoryError:
                report["measurements"].append({"scenario": scenario, "batch_size": batch_size, "oom": True})
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                torch.accelerator.empty_cache()
                break
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["profile_batch_size"] = max(
        (
            row["batch_size"]
            for row in report["measurements"]
            if row["scenario"] == "steady" and not row.get("oom", False)
        ),
        default=None,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["profile_batch_size"] is None:
        logging.info("No steady decoder batch fits; saved OOM results without a profile")
        return
    chunks = make_chunks(decoder, reference, codes, report["profile_batch_size"], "steady")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profile:
        with torch.profiler.record_function("chatterbox_streaming_decode"):
            decoder.chunked_decode_streaming(chunks)
        torch.accelerator.synchronize()
    profile.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
    args.output.with_suffix(".profile.txt").write_text(
        profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=60)
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


@torch.inference_mode()
def make_chunks(decoder, reference, codes, batch_size, scenario):
    """Construct first, continuing, or final chunks with real cached source."""
    rows = [codes.roll(index) for index in range(batch_size)]
    if scenario == "first":
        return [Chunk(row[:23], 0, reference, None, False) for row in rows]
    first = [Chunk(row[:53], 0, reference, None, False) for row in rows]
    states = decoder.chunked_decode_streaming(first)
    return [
        Chunk(row[:113], 50, reference, state, scenario == "final")
        for row, (_, state) in zip(rows, states, strict=True)
    ]


@torch.inference_mode()
def measure(operation):
    """Measure warmed wall time and CUDA events without profiler overhead."""
    for _ in range(WARMUPS):
        operation()
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    walls, devices = [], []
    for _ in range(REPEATS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        operation()
        end.record()
        end.synchronize()
        walls.append(time.perf_counter() - started)
        devices.append(start.elapsed_time(end) / 1000)
    return {
        "wall_seconds_median": statistics.median(walls),
        "cuda_seconds_median": statistics.median(devices),
        "wall_seconds": walls,
        "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
        "peak_reserved_bytes": torch.accelerator.max_memory_reserved(),
    }


if __name__ == "__main__":
    with torch.inference_mode():
        main()
