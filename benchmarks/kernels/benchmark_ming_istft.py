# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure Ming ISTFT and optionally its pretrained streaming AudioVAE decoder.

Example (no model weights needed for the operator benchmark):
    python benchmarks/kernels/benchmark_ming_istft.py --compile --iterations 100

Add --decoder to measure the published decoder with fixed synthetic latents.
This is a decoder-stage benchmark, not text-to-speech serving latency.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from functools import partial

import torch

from vllm_omni.model_executor.models.common.ming.audio_dsp import ISTFT
from vllm_omni.model_executor.models.common.ming.fused_istft import fused_istft


def measure(operation: Callable, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        operation()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        torch.accelerator.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    torch.accelerator.reset_peak_memory_stats()
    before = torch.accelerator.memory_allocated()
    operation()
    torch.accelerator.synchronize()
    return {
        "p50_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples),
        "extra_peak_mib": (torch.accelerator.max_memory_allocated() - before) / 1024**2,
    }


def native_post_fft(module, frames, audio, window):
    """The native middle-chunk post-FFT sequence, isolated from cuFFT."""
    count = frames.shape[-1]
    size = (count - 1) * module.hop_length + module.win_length

    def fold(value):
        return torch.nn.functional.fold(
            value, output_size=(1, size), kernel_size=(1, module.win_length), stride=(1, module.hop_length)
        )[:, 0, 0, :]

    samples = fold(frames * module.window[None, :, None])
    samples, audio = module._buffer_process(samples, audio, 0, streaming=True)
    envelope = fold(module.window.square().expand(1, count, -1).transpose(1, 2))
    envelope, window = module._buffer_process(envelope, window, 0, streaming=True)
    if not (envelope > 1e-11).all():
        raise RuntimeError("ISTFT window envelope underflowed; invalid overlap-add state.")
    return samples / envelope, audio, window


def assert_outputs(actual, reference):
    maximum = 0.0
    for value, expected in zip(actual, reference, strict=True):
        if expected is None:
            assert value is None
        else:
            torch.testing.assert_close(value, expected, atol=5e-7, rtol=1e-5)
            maximum = max(maximum, (value - expected).abs().max().item())
    return maximum


def benchmark_istft(args):
    for dtype in (torch.float32, torch.bfloat16):
        module = ISTFT(3528, 882, 3528).to(device="cuda", dtype=dtype)
        for count in (16, 100, 400):
            spec = torch.randn(1, 1765, count, device="cuda", dtype=torch.complex64)
            _, audio, window = module.forward_native(spec, streaming=True)
            frames = torch.fft.irfft(spec, 3528, dim=1)
            for scope in ("post_fft", "istft"):
                if scope == "post_fft":
                    native = partial(native_post_fft, module, frames, audio, window)
                    fused = partial(fused_istft, frames, module.window, 882, audio, window, True, False)
                else:
                    native = partial(module.forward_native, spec, audio, window, streaming=True)
                    fused = partial(module, spec, audio, window, streaming=True)
                error = assert_outputs(fused(), native())
                results = {"native": measure(native, args.warmup, args.iterations)}
                if args.compile:
                    compiled = torch.compile(native)
                    compiled_output, reference = compiled(), native()
                    # Inductor can change BF16 envelope rounding. Report that
                    # difference without weakening the fused-kernel parity gate.
                    compiled_matches = all(
                        torch.allclose(value, expected, atol=5e-7, rtol=1e-5)
                        for value, expected in zip(compiled_output, reference, strict=True)
                    )
                    results["compiled"] = measure(compiled, args.warmup, args.iterations)
                    results["compiled"]["matches_native_tolerance"] = compiled_matches
                    results["compiled"]["waveform_max_abs_error"] = (
                        (compiled_output[0] - reference[0]).abs().max().item()
                    )
                results["fused"] = measure(fused, args.warmup, args.iterations)
                print(
                    json.dumps(
                        {
                            "scope": scope,
                            "window_dtype": str(dtype),
                            "frames": count,
                            "max_abs_error": error,
                            "results": results,
                        }
                    ),
                    flush=True,
                )


def benchmark_decoder(args):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from vllm_omni.model_executor.models.common.ming.audio_vae import Decoder

    model = "inclusionAI/Ming-omni-tts-0.5B"
    revision = "9154772e7fbc585907b6237e3190790676f28975"
    config_path = hf_hub_download(model, "config.json", revision=revision)
    with open(config_path) as source:
        config = json.load(source)["audio_tokenizer_config"]
    kwargs = config["dec_kwargs"]
    backbone = kwargs["backbone"]
    backbone["_attn_implementation"] = "sdpa"
    decoder = Decoder(backbone, kwargs["output_dim"], kwargs["latent_dim"], config["patch_size"])
    checkpoint = hf_hub_download(model, "model.safetensors", revision=revision)
    prefix = "audio.decoder."
    with safe_open(checkpoint, framework="pt") as weights:
        state = {key.removeprefix(prefix): weights.get_tensor(key) for key in weights.keys() if key.startswith(prefix)}
    decoder.load_state_dict(state, strict=True)
    decoder = decoder.eval().to(device="cuda", dtype=torch.bfloat16)
    chunks = [
        torch.randn(1, size, kwargs["latent_dim"], device="cuda", dtype=torch.bfloat16) for size in (4, 25, 25, 5)
    ]
    istft = decoder.head.istft
    fused_forward = istft.forward

    def run():
        cache = state = None
        outputs = []
        for index, chunk in enumerate(chunks):
            output, state, cache = decoder.low_level_reconstruct(
                chunk, past_key_values=cache, use_cache=True, stream_state=state, last_chunk=index == len(chunks) - 1
            )
            outputs.append(output)
        return torch.cat(outputs, dim=-1)

    istft.forward = istft.forward_native
    expected = run()
    istft.forward = fused_forward
    actual = run()
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=1e-5)
    # Alternate the order to expose drift when the decoder-level gain is small.
    results = {"native": [], "fused": []}
    for order in (("native", "fused"), ("fused", "native"), ("native", "fused")):
        for name in order:
            istft.forward = istft.forward_native if name == "native" else fused_forward
            results[name].append(measure(run, args.warmup, args.decoder_iterations))
    print(
        json.dumps(
            {
                "scope": "pretrained_decoder",
                "model": model,
                "revision": revision,
                "backend": "sdpa",
                "dtype": "bfloat16",
                "latent_chunks": [4, 25, 25, 5],
                "max_abs_error": (actual - expected).abs().max().item(),
                "results": results,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--decoder", action="store_true")
    parser.add_argument("--decoder-iterations", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or min(args.iterations, args.decoder_iterations) < 1:
        parser.error("warmup must be nonnegative and iteration counts must be positive")
    torch.manual_seed(0)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "warmup": args.warmup,
                "iterations": args.iterations,
                "decoder_iterations": args.decoder_iterations,
            }
        )
    )
    with torch.inference_mode():
        benchmark_istft(args)
        if args.decoder:
            benchmark_decoder(args)


if __name__ == "__main__":
    main()
