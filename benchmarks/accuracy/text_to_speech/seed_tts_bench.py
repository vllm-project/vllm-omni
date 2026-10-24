#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standalone accuracy evaluation for Seed-TTS models.

Generates TTS outputs from prompts, evaluates speech quality using WER/SIM/UTMOS,
and produces a summary report. Works with any Seed-TTS compatible model.

Workflow:
1. Generate TTS outputs for test prompts via vLLM API
2. Evaluate speech quality (WER via Whisper/Paraformer, SIM via WavLM, UTMOS)
3. Summarize results with per-prompt and aggregate metrics

Usage:
    # Generate and evaluate
    python benchmarks/accuracy/text_to_speech/seed_tts_bench.py \\
        --model Qwen/Qwen3-TTS \\
        --dataset-path ./seed-tts-eval \\
        --locale en \\
        --output-dir ./tts_accuracy_results

    # With different device
    python benchmarks/accuracy/text_to_speech/seed_tts_bench.py \\
        --model Qwen/Qwen3-TTS \\
        --locale zh \\
        --eval-device cuda:1 \\
        --output-dir ./tts_results_zh

Prerequisites:
- vLLM server running with Seed-TTS model
- vllm-omni[dev] installed for WER/SIM/UTMOS evaluation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from vllm_omni.benchmarks.data_modules.seed_tts_dataset import SeedTTSDataset

# Import from seed_tts_eval for reusable logic
from vllm_omni.benchmarks.data_modules.seed_tts_eval import (
    compute_seed_tts_wer_metrics,
    pcm_s16le_mono_to_wav_bytes,
    print_seed_tts_wer_summary,
)


def _utc_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%SZ")


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


async def generate_tts_outputs(
    model_id: str,
    prompts: list[str],
    base_url: str = "http://localhost:8000",
    locale: str = "en",
) -> dict[str, Any]:
    """Generate TTS outputs for prompts via vLLM API.

    Args:
        model_id: Model identifier
        prompts: List of text prompts to synthesize
        base_url: vLLM server base URL
        locale: Language locale (en or zh)

    Returns:
        dict with generated audio data per prompt
    """
    endpoint = f"{base_url}/v1/audio/speech"
    results = {"generated": [], "failed": []}

    async with aiohttp.ClientSession() as session:
        for i, prompt in enumerate(prompts):
            try:
                payload = {
                    "model": model_id,
                    "input": prompt,
                    "voice": "default",
                    "response_format": "pcm",
                }

                async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status != 200:
                        results["failed"].append({
                            "index": i,
                            "prompt": prompt,
                            "error": f"HTTP {resp.status}",
                        })
                        continue

                    audio_bytes = await resp.read()
                    results["generated"].append({
                        "index": i,
                        "prompt": prompt,
                        "audio_bytes": audio_bytes,
                        "audio_path": None,  # Will be set when saved
                    })
            except Exception as e:
                results["failed"].append({
                    "index": i,
                    "prompt": prompt,
                    "error": str(e),
                })

    return results


def save_generated_audio(
    outputs: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Save generated audio files to disk.

    Args:
        outputs: Dict from generate_tts_outputs()
        output_dir: Directory to save audio files

    Returns:
        Updated outputs dict with audio_path filled in
    """
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for item in outputs.get("generated", []):
        audio_path = audio_dir / f"prompt_{item['index']:03d}.wav"
        wav_bytes = pcm_s16le_mono_to_wav_bytes(item["audio_bytes"])
        audio_path.write_bytes(wav_bytes)
        item["audio_path"] = str(audio_path)

    return outputs


def evaluate_generated_audio(
    outputs: dict[str, Any],
    locale: str = "en",
) -> dict[str, Any]:
    """Evaluate generated audio using WER/SIM/UTMOS.

    Args:
        outputs: Dict from save_generated_audio()
        locale: Language locale (en or zh)

    Returns:
        Dict with evaluation results per item
    """
    # Create SampleRequest-like objects for compute_seed_tts_wer_metrics
    eval_items = []
    reference_paths = {}  # Would come from dataset in real scenario

    for item in outputs.get("generated", []):
        # In standalone script, we only have synthesized audio
        # In real scenario, would have reference_audio_path from dataset
        eval_items.append({
            "prompt": item["prompt"],
            "audio_path": item["audio_path"],
            "reference_audio_path": reference_paths.get(item["index"]),
        })

    # Call core evaluation logic from seed_tts_eval.py
    metrics = compute_seed_tts_wer_metrics(
        eval_items,
        [{"audio_data": Path(item["audio_path"]).read_bytes()} for item in eval_items],
    )

    return metrics


def summarize_results(
    metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Summarize evaluation results.

    Args:
        metrics: Dict from evaluate_generated_audio()
        output_dir: Directory to save summary

    Returns:
        Summary statistics
    """
    summary = {
        "timestamp": _utc_timestamp(),
        "num_evaluated": metrics.get("num_evaluated", 0),
        "num_failed": metrics.get("num_failed", 0),
    }

    # Extract aggregate metrics
    if metrics.get("seed_tts_mean_wer") is not None:
        summary["mean_wer"] = metrics["seed_tts_mean_wer"]
        summary["mean_sim"] = metrics.get("seed_tts_mean_sim")
        summary["mean_utmos"] = metrics.get("seed_tts_mean_utmos")

    # Save full metrics
    metrics_path = output_dir / f"metrics_{_utc_timestamp()}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    summary["metrics_file"] = str(metrics_path)

    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description="TTS accuracy evaluation")
    parser.add_argument("--model", required=True, help="Model ID (e.g., Qwen/Qwen3-TTS)")
    parser.add_argument("--host", default="localhost", help="vLLM server host")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to Seed-TTS dataset root (default: from env or download)",
    )
    parser.add_argument("--locale", choices=["en", "zh"], default="en", help="Language locale")
    parser.add_argument("--num-prompts", type=int, default=None, help="Max prompts to evaluate")
    parser.add_argument("--output-dir", type=Path, default="./tts_accuracy_results")
    parser.add_argument(
        "--eval-device",
        type=str,
        default=None,
        help="Device for WER/SIM/UTMOS evaluation (e.g. cuda:0, cpu)",
    )

    args = parser.parse_args()

    output_dir = _ensure_output_dir(args.output_dir)
    base_url = f"http://{args.host}:{args.port}"

    # Set evaluation device if specified
    if args.eval_device:
        os.environ["SEED_TTS_EVAL_DEVICE"] = args.eval_device

    print("Generating TTS outputs...")
    print(f"  Model: {args.model}")
    print(f"  Server: {base_url}")
    print(f"  Locale: {args.locale}")
    print(f"  Output: {output_dir}")

    # Load prompts from dataset
    try:
        from vllm_omni.benchmarks.data_modules.seed_tts_dataset import resolve_seed_tts_root
        dataset_root = resolve_seed_tts_root(args.dataset_path, locale=args.locale)
        dataset = SeedTTSDataset(
            dataset_root=dataset_root,
            locale=args.locale,
            num_requests=args.num_prompts,
        )
        prompts = [req.prompt for req in dataset.sample(None, num_requests=args.num_prompts or 10)]
    except Exception as e:
        print(f"  Warning: Could not load real Seed-TTS dataset: {e}")
        print("  Using random prompts instead")
        prompts = [
            "Hello, this is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Please tell me a story about a magical forest.",
        ] * ((args.num_prompts or 10) // 3 + 1)
        prompts = prompts[: args.num_prompts or 10]

    print(f"  Prompts: {len(prompts)}")

    # Generate outputs
    outputs = await generate_tts_outputs(args.model, prompts, base_url, args.locale)
    print(f"  Generated: {len(outputs['generated'])}, Failed: {len(outputs['failed'])}")

    # Save to disk
    outputs = save_generated_audio(outputs, output_dir)
    print(f"  Saved audio to {output_dir / 'audio'}")

    # Evaluate
    print("Evaluating with WER/SIM/UTMOS...")
    metrics = evaluate_generated_audio(outputs, args.locale)

    # Summarize
    summary = summarize_results(metrics, output_dir)

    # Print results
    print("\nResults:")
    print_seed_tts_wer_summary(metrics)

    # Save summary
    summary_path = output_dir / f"summary_{_utc_timestamp()}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
