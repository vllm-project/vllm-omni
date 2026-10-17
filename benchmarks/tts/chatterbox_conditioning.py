#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight CPU conditioning parity and 64-reference sustained batch benchmark."""

import argparse
import hashlib
import json
import logging
import resource
import sys
import time
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioner
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

COUNT = 64
DURATIONS = (6.1, 8.3, 10.1, 16.0)
CYCLES = 3
TOLERANCE = 1e-4


def main():
    """Write reproducible measurements and exit nonzero on any parity or mutation failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Local checkpoint snapshot directory")
    parser.add_argument("reference", type=Path, help="Actual speech recording, at least five seconds")
    parser.add_argument("output", type=Path)
    parser.add_argument("variant", nargs="?", choices=("turbo", "original"))
    parser.add_argument("--device", default="cpu", help="Explicit conditioning device, e.g. cuda:0")
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("model must be a local snapshot directory")
    variant = args.variant or ("turbo" if "turbo" in str(args.model).lower() else "original")
    wav, rate = sf.read(args.reference, dtype="float32", always_2d=True)
    source = torch.from_numpy(wav).mean(dim=1)
    if len(source) < rate * 5 or not torch.isfinite(source).all() or not torch.any(source):
        parser.error("reference must contain at least five seconds of finite, nonzero speech")
    references = []
    hashes = []
    for index in range(COUNT):
        length = round(DURATIONS[index % len(DURATIONS)] * rate)
        samples = source.tile((length + len(source) - 1) // len(source))[:length]
        envelope = 0.8 + 0.1 * torch.sin(torch.arange(length) / rate * (index + 1) / COUNT)
        samples = (samples * envelope).numpy(force=True)
        references.append((samples, rate))
        hashes.append(hashlib.sha256(samples.tobytes()).hexdigest())
    device = torch.device(args.device)
    conditioner = VoiceConditioner(str(args.model), ChatterboxConfig(variant), device)
    modules = {
        "voice_encoder": conditioner.voice_encoder,
        "tokenizer": conditioner.tokenizer,
        "campplus": conditioner.campplus,
    }
    # Warm lazy libraries before measuring, but snapshot buffers before any inference.
    buffers = {
        f"{name}.{key}": value.detach().clone()
        for name, module in modules.items()
        for key, value in module.named_buffers()
    }
    calls = defaultdict(list)
    handles = []
    for name, module in modules.items():
        for child_name, child in [("", module), *module.named_children()]:
            label = f"{name}.{child_name}".rstrip(".")
            handles.append(
                child.register_forward_hook(
                    lambda module, inputs, output, label=label: calls[label].append(
                        int(inputs[0].shape[0]) if inputs and isinstance(inputs[0], torch.Tensor) else None
                    )
                )
            )
    conditioner.prepare(*references[0])
    calls.clear()
    report = {
        "model": str(args.model.resolve()),
        "variant": variant,
        "reference": str(args.reference.resolve()),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "reference_mode": "repeated real speech with deterministic amplitude envelopes",
        "durations_seconds": [DURATIONS[i % 4] for i in range(COUNT)],
        "waveform_sha256": hashes,
        "references": COUNT,
        # Every reference is encoded in one batch; the stage's admission sets the size.
        "batch_size": COUNT,
        "cycles": CYCLES,
        "torch_version": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": str(device),
        "tolerance": {"atol": TOLERANCE, "rtol": TOLERANCE},
        "measurements": [],
        "parity": [],
    }
    for cycle in tqdm(range(CYCLES), desc="conditioning serial/batch parity"):
        outputs = {}
        for mode in ("serial", "batch"):
            calls.clear()
            started = time.perf_counter()
            outputs[mode] = (
                [conditioner.prepare(*reference) for reference in references]
                if mode == "serial"
                else conditioner.prepare_batch(references)
            )
            elapsed = time.perf_counter() - started
            report["measurements"].append(
                {
                    "cycle": cycle,
                    "mode": mode,
                    "seconds": elapsed,
                    "references_per_second": COUNT / elapsed,
                    "module_forward_batch_sizes": dict(calls),
                }
            )
        report["parity"].append(compare(outputs["serial"], outputs["batch"], cycle))
    after = {f"{name}.{key}": value for name, module in modules.items() for key, value in module.named_buffers()}
    report["mutated_buffers"] = [
        key for key, value in buffers.items() if key not in after or not torch.equal(value, after[key])
    ]
    report["new_buffers"] = sorted(after.keys() - buffers.keys())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
    if device.type == "cuda":
        report["peak_cuda_allocated_bytes"] = torch.accelerator.max_memory_allocated(device)
        report["peak_cuda_reserved_bytes"] = torch.accelerator.max_memory_reserved(device)
    report["passed"] = (
        all(p["passed"] for p in report["parity"]) and not report["mutated_buffers"] and not report["new_buffers"]
    )
    for handle in handles:
        handle.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logging.basicConfig(level=logging.INFO)
    logging.info("Conditioning report %s: passed=%s", args.output, report["passed"])
    raise SystemExit(0 if report["passed"] else 1)


def compare(serial, batched, cycle):
    """Check all five tensors: tokens must match exactly, features within TOLERANCE.

    The batched STFT behind the prompt mel is not bit-identical to the
    clip's own on every BLAS, so mels are compared like the embeddings.
    """
    report = {"cycle": cycle, "passed": len(serial) == len(batched) == COUNT, "tensors": {}}
    for field in fields(serial[0]):
        exact = field.name in {"cond_tokens", "prompt_token"}
        failures = []
        maximum = 0.0
        for index, (left, right) in enumerate(zip(serial, batched)):
            expected, actual = getattr(left, field.name), getattr(right, field.name)
            valid_shape = expected.shape == actual.shape
            finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
            same = (
                valid_shape
                and finite
                and (
                    torch.equal(expected, actual)
                    if exact
                    else torch.allclose(expected, actual, atol=TOLERANCE, rtol=TOLERANCE)
                )
            )
            if not same:
                failures.append(index)
            if valid_shape:
                maximum = max(maximum, float((expected.double() - actual.double()).abs().max()))
        report["tensors"][field.name] = {
            "exact_required": exact,
            "failed_indices": failures,
            "max_absolute_error": maximum,
        }
        report["passed"] = report["passed"] and not failures
    return report


if __name__ == "__main__":
    main()
