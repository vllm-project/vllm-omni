# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare Lance VAE output assembly. Decoder compute is not timed."""

import argparse
import json
import statistics
import time
from functools import partial

import torch

from vllm_omni.diffusion.models.lance.vae_output import write_unpatchified
from vllm_omni.diffusion.models.lance.wan_vae import _unpatchify


def original(chunks: list[torch.Tensor]) -> torch.Tensor:
    output = chunks[0]
    for chunk in chunks[1:]:
        output = torch.cat([output, chunk], dim=2)
    return _unpatchify(output, patch_size=2).clamp_(-1, 1)


def single_cat(chunks: list[torch.Tensor]) -> torch.Tensor:
    output = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
    return _unpatchify(output, patch_size=2).clamp_(-1, 1)


def fused(chunks: list[torch.Tensor], *, fuse_clamp: bool = True) -> torch.Tensor:
    batch, channels, _, height, width = chunks[0].shape
    frames = sum(chunk.shape[2] for chunk in chunks)
    output = chunks[0].new_empty((batch, channels // 4, frames, height * 2, width * 2))
    offset = 0
    for chunk in chunks:
        write_unpatchified(chunk, output, offset, clamp=fuse_clamp)
        offset += chunk.shape[2]
    return output if fuse_clamp else output.clamp_(-1, 1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 61, 121])
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--ablation", action="store_true", help="Also measure one concatenation and separate clamp")
    parser.add_argument("--profile", action="store_true", help="Profile the original operators after timing")
    args = parser.parse_args()
    if args.height % 2 or args.width % 2 or any(n < 1 or (n - 1) % 4 for n in args.frames):
        parser.error("Use even spatial sizes and 1 + 4*n frames")
    torch.manual_seed(81)
    dtype = getattr(torch, args.dtype)
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "args": vars(args)}))
    for frames in args.frames:
        chunks = [
            torch.randn(1, 12, t, args.height // 2, args.width // 2, device="cuda", dtype=dtype)
            for t in [1] + [4] * ((frames - 1) // 4)
        ]
        methods = {"main": original, "PR": fused}
        if args.ablation:
            methods = {
                "main": original,
                "single_cat": single_cat,
                "direct_write": partial(fused, fuse_clamp=False),
                "PR": fused,
            }
        reference = original(chunks)
        for method in methods.values():
            assert torch.equal(reference, method(chunks))
        del reference
        samples: dict[str, list[float]] = {name: [] for name in methods}
        peaks = {}
        for name, method in methods.items():
            for _ in range(5):
                method(chunks)
            torch.accelerator.synchronize()
            allocated = torch.accelerator.memory.memory_allocated()
            torch.accelerator.memory.reset_peak_memory_stats()
            output = method(chunks)
            torch.accelerator.synchronize()
            peaks[name] = (torch.accelerator.memory.max_memory_allocated() - allocated) / 2**20
            del output
        for round_ in range(args.rounds):
            for name in list(methods) if round_ % 2 == 0 else list(reversed(methods)):
                torch.accelerator.synchronize()
                start = time.perf_counter()
                output = methods[name](chunks)
                torch.accelerator.synchronize()
                samples[name].append((time.perf_counter() - start) * 1000)
                del output
        print(
            json.dumps(
                {
                    "frames": frames,
                    "extra_peak_MiB": peaks,
                    "results": {
                        name: {
                            "mean_ms": statistics.mean(values),
                            "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0,
                            "median_ms": statistics.median(values),
                            "samples_ms": values,
                        }
                        for name, values in samples.items()
                    },
                }
            )
        )
        if args.profile:
            # Diagnostic time is separate from the benchmark above.
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profile:
                output = original(chunks)
                torch.accelerator.synchronize()
            del output
            print(
                json.dumps(
                    {
                        "frames": frames,
                        "profile_aten": [
                            {
                                "name": event.key,
                                "calls": event.count,
                                "self_cpu_ms": event.self_cpu_time_total / 1000,
                                "self_device_ms": event.self_device_time_total / 1000,
                            }
                            for event in profile.key_averages()
                            if event.key.startswith("aten::")
                        ],
                    }
                )
            )


if __name__ == "__main__":
    main()
