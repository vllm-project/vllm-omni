# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare Fish DAC encoding paths with local weights and reference audio.
Run with --model-path /models/fish-speech-s2-pro --ref-audio ref.wav other.wav.
Times DAC.encode only: audio loading, downmixing, and resampling are excluded.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import soundfile as sf
import torch

from vllm_omni.model_executor.models.fish_speech.dac_encoder import (
    _load_dac_codec,
    _prepare_reference_audio_tensor,
)
from vllm_omni.model_executor.models.fish_speech.dac_utils import DAC_SAMPLE_RATE
from vllm_omni.platforms import current_omni_platform


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="Local directory containing codec.pth")
    parser.add_argument("--ref-audio", required=True, nargs="+", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10, help="Measured pairs, alternating AB and BA")
    parser.add_argument("--n-quantizers", type=int)
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "codec.pth").is_file():
        parser.error("--model-path must contain an existing codec.pth; this benchmark does not download weights")
    if args.warmup < 1 or args.iterations < 2 or (args.n_quantizers is not None and args.n_quantizers < 1):
        parser.error("require warmup >= 1, iterations >= 2, and n-quantizers >= 1 when supplied")

    device = current_omni_platform.get_torch_device()
    accelerator = device.type != "cpu"

    def synchronize():
        if accelerator:
            current_omni_platform.synchronize()

    codec = _load_dac_codec(str(model_path), device=device, dtype=torch.float32)
    candidate = codec.quantizer.encode

    def baseline(z, *args, **kwargs):
        return codec.quantizer(z, *args, **kwargs).codes

    modes = {"baseline": baseline, "candidate": candidate}
    checkout = str(Path(__file__).resolve().parents[2])
    revision = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", checkout, "status", "--porcelain"], text=True).strip())
    environment = {
        "source_commit": revision,
        "source_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "vllm": importlib.metadata.version("vllm"),
        "vllm_omni": importlib.metadata.version("vllm-omni"),
        "device": str(device),
        "device_name": current_omni_platform.get_device_name() if accelerator else platform.processor(),
        "device_runtime": current_omni_platform.get_device_version() if accelerator else None,
        "dtype": "torch.float32",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "model_path": str(model_path),
        "codec_sha256": sha256(model_path / "codec.pth"),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "n_quantizers": args.n_quantizers,
    }
    print(json.dumps({"environment": environment}), flush=True)
    try:
        for ref_path in args.ref_audio:
            ref_path = ref_path.expanduser().resolve()
            audio_sha256 = sha256(ref_path)
            samples, sample_rate = sf.read(ref_path, dtype="float32", always_2d=True)
            audio = _prepare_reference_audio_tensor(samples.T, sample_rate, device=device, dtype=torch.float32)
            if not audio.numel():
                raise ValueError(f"Reference audio is empty: {ref_path}")
            lengths = torch.tensor([audio.numel()], device=device, dtype=torch.long)
            audio = audio[None, None]
            codec.quantizer.encode = baseline
            expected = codec.encode(audio, lengths, n_quantizers=args.n_quantizers)
            codec.quantizer.encode = candidate
            actual = codec.encode(audio, lengths, n_quantizers=args.n_quantizers)
            if not all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)):
                raise AssertionError(f"Code or length mismatch: {ref_path}")
            code_shape = list(actual[0].shape)
            del actual, expected

            timings = {name: [] for name in modes}
            peaks = {name: [] for name in modes}
            extra_peaks = {name: [] for name in modes}
            for iteration in range(-args.warmup, args.iterations):
                order = ("baseline", "candidate") if iteration % 2 == 0 else ("candidate", "baseline")
                for name in order:
                    codec.quantizer.encode = modes[name]
                    synchronize()
                    if iteration < 0:
                        codec.encode(audio, lengths, n_quantizers=args.n_quantizers)
                        continue
                    if accelerator:
                        torch.accelerator.reset_peak_memory_stats()
                        allocated = torch.accelerator.memory_allocated()
                    start = time.perf_counter()
                    output = codec.encode(audio, lengths, n_quantizers=args.n_quantizers)
                    synchronize()
                    timings[name].append((time.perf_counter() - start) * 1000)
                    if accelerator:
                        peak = torch.accelerator.max_memory_allocated()
                        peaks[name].append(peak)
                        extra_peaks[name].append(peak - allocated)
                    del output
            result = {
                "reference_audio": str(ref_path),
                "reference_audio_sha256": audio_sha256,
                "prepared_seconds": audio.shape[-1] / DAC_SAMPLE_RATE,
                "code_shape": code_shape,
                "codes_and_lengths_exact": True,
                "scope": "DAC.encode; synchronized wall clock; alternating AB/BA; no profiler",
                "results": {
                    name: {
                        "raw_ms": values,
                        "median_ms": statistics.median(values),
                        "min_ms": min(values),
                        "max_ms": max(values),
                        "peak_allocated_bytes": max(peaks[name], default=None),
                        "extra_peak_allocated_bytes": max(extra_peaks[name], default=None),
                    }
                    for name, values in timings.items()
                },
            }
            print(json.dumps(result), flush=True)
    finally:
        codec.quantizer.encode = candidate


if __name__ == "__main__":
    main()
