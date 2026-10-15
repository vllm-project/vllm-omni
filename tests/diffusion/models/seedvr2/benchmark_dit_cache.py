# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Paired native-VAE short benchmark with exact DiT/RGB output checks.

Run from the checkout with PYTHONPATH=. and a local SeedVR2-3B model directory.
The default profile respects the single-GPU admission limits. No VAE fusion,
compile, quantization, or admission overrides are enabled.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from vllm_omni.diffusion.models.seedvr2 import nadit, pipeline_seedvr2, vae
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


class BenchmarkRun(TypedDict):
    round: int
    seed: int
    cache: bool
    request_s: float
    stages_s: dict[str, float]
    peak_bytes: dict[str, int]
    cache_stats: dict[str, int]
    dit_sha256: str
    rgb_sha256: str
    rgb_max_abs: int
    dit_bit_exact: bool
    rgb_bit_exact: bool


def sha(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--reference-only", action="store_true", help="also runnable against unmodified main")
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("at least two paired rounds are required")
    pipeline_seedvr2.validate_clip_size(
        args.frames, args.height, args.width, pipeline_seedvr2.MAX_FRAME_PIXELS, pipeline_seedvr2.MAX_CLIP_PIXELS
    )
    with ExitStack() as cleanup:
        device_api = torch.get_device_module(current_omni_platform.device_type)
        original_env = os.environ.get("VLLM_OMNI_SEEDVR2_DIT_CACHE")

        def restore_environment():
            if original_env is None:
                os.environ.pop("VLLM_OMNI_SEEDVR2_DIT_CACHE", None)
            else:
                os.environ["VLLM_OMNI_SEEDVR2_DIT_CACHE"] = original_env

        cleanup.callback(restore_environment)
        cleanup.callback(setattr, torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
        torch.backends.cudnn.benchmark = False
        active = False
        stages: dict[str, float] = {}
        captures: dict[str, torch.Tensor] = {}
        cleanup.callback(captures.clear)
        cache_stats: dict[str, int] = {}

        def wrap(owner, method, name):
            original = getattr(owner, method)
            cleanup.callback(setattr, owner, method, original)

            @functools.wraps(original)
            def measured(self, *positional, **keywords):
                if not active:
                    return original(self, *positional, **keywords)
                current_omni_platform.synchronize()
                start = time.perf_counter()
                result = original(self, *positional, **keywords)
                current_omni_platform.synchronize()
                stages[name] = time.perf_counter() - start
                if name == "dit":
                    # Retain a reference only; hash/copy after the timed request.
                    captures["dit"] = result.vid_sample.detach()
                    runtime = keywords.get("runtime") or (positional[5] if len(positional) > 5 else None)
                    if runtime is not None:
                        contexts = list(runtime._contexts.values())
                        cache_stats.update(
                            layouts=len(contexts),
                            row_plans=sum(getattr(c, "sdpa_groups", None) is not None for c in contexts),
                            rotary_builds=sum(c.rotary_cache.builds for c in contexts if hasattr(c, "rotary_cache")),
                            rotary_hits=sum(c.rotary_cache.hits for c in contexts if hasattr(c, "rotary_cache")),
                        )
                return result

            setattr(owner, method, measured)

        wrap(vae.SeedVR2VAE, "encode", "vae_encode")
        wrap(nadit.SeedVR2NaDiT, "forward", "dit")
        wrap(vae.SeedVR2VAE, "decode", "vae_decode")
        inputs = {
            seed: torch.randint(
                0,
                256,
                (args.frames, args.height, args.width, 3),
                dtype=torch.uint8,
                generator=torch.Generator().manual_seed(seed + 100),
            )
            for seed in (1101, 1102)
        }
        runs: list[BenchmarkRun] = []
        report: dict[str, object] = dict(
            protocol=dict(
                height=args.height,
                width=args.width,
                frames=args.frames,
                fps=24,
                steps=1,
                cfg=1.0,
                dtype="float16",
                eager=True,
                vae_tiling=True,
                color="lab",
                num_gpus=1,
                warmup_per_variant=1,
                paired_rounds=args.rounds,
                seeds=[1101, 1102],
                timing="GPU-synchronized stage and request wall time; tensor hashes after timing",
                task="same-resolution restoration; synthetic uint8 source; no MP4 encoding",
            ),
            software={name: importlib.metadata.version(name) for name in ("torch", "vllm", "vllm-omni", "numpy")},
            hardware=dict(
                platform=platform.machine(),
                device=str(device_api.get_device_properties(0)),
                host_memtotal=next(
                    (r for r in Path("/proc/meminfo").read_text().splitlines() if r.startswith("MemTotal:")), "unknown"
                ),
            ),
            sources={
                name: dict(
                    path=inspect.getfile(module),
                    sha256=hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest(),
                )
                for name, module in (("nadit", nadit), ("pipeline", pipeline_seedvr2), ("vae", vae))
            },
            input_sha256={str(seed): sha(frames) for seed, frames in inputs.items()},
            runs=runs,
        )
        reference: dict[int, tuple[np.ndarray, torch.Tensor]] = {}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Write the report even if close fails; ExitStack still runs every restoration.
        cleanup.callback(lambda: args.output.write_text(json.dumps(report, indent=2) + "\n"))
        try:
            engine = Omni(
                model=args.model,
                model_class_name="SeedVR2Pipeline",
                dtype="float16",
                enforce_eager=True,
                vae_use_tiling=True,
                num_gpus=1,
            )

            cleanup.callback(engine.close)

            def request(enabled, seed):
                os.environ["VLLM_OMNI_SEEDVR2_DIT_CACHE"] = "1" if enabled else "0"
                return engine.generate(
                    {"prompt": " ", "multi_modal_data": {"video": inputs[seed]}},
                    OmniDiffusionSamplingParams(
                        height=args.height,
                        width=args.width,
                        num_frames=args.frames,
                        fps=24,
                        num_inference_steps=1,
                        guidance_scale=1.0,
                        seed=seed,
                        output_type="np",
                        extra_args={"color_correction_method": "lab"},
                    ),
                    use_tqdm=False,
                )

            for enabled in (False,) if args.reference_only else (False, True):
                request(enabled, 1101)
            for round_index in range(args.rounds):
                seed = 1101 + round_index % 2
                variants = (False, True) if round_index % 2 == 0 else (True, False)
                for enabled in (False,) if args.reference_only else variants:
                    stages.clear()
                    captures.clear()
                    cache_stats.clear()
                    device_api.reset_peak_memory_stats()
                    active = True
                    current_omni_platform.synchronize()
                    start = time.perf_counter()
                    output = request(enabled, seed)
                    current_omni_platform.synchronize()
                    elapsed = time.perf_counter() - start
                    active = False
                    assert set(stages) == {"vae_encode", "dit", "vae_decode"}, "requires single-process execution"
                    peak = dict(allocated=device_api.max_memory_allocated(), reserved=device_api.max_memory_reserved())
                    rgb = np.asarray(output[0].images[0])
                    latent = captures["dit"].cpu()
                    assert rgb.shape == (1, args.frames, args.height, args.width, 3) and rgb.dtype == np.uint8
                    if seed not in reference:
                        reference[seed] = (rgb.copy(), latent)
                    ref_rgb, ref_latent = reference[seed]
                    exact_rgb, exact_dit = np.array_equal(rgb, ref_rgb), torch.equal(latent, ref_latent)
                    row = BenchmarkRun(
                        round=round_index,
                        seed=seed,
                        cache=enabled,
                        request_s=elapsed,
                        stages_s=dict(stages),
                        peak_bytes=peak,
                        cache_stats=dict(cache_stats),
                        dit_sha256=sha(latent),
                        rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        rgb_max_abs=int(np.abs(rgb.astype(np.int16) - ref_rgb.astype(np.int16)).max()),
                        dit_bit_exact=exact_dit,
                        rgb_bit_exact=exact_rgb,
                    )
                    runs.append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                    assert exact_rgb and exact_dit, "cache changed output"
                    assert not enabled or cache_stats.get("rotary_hits", 0) > 0, "cache was not activated"
                    del output, rgb, latent
            summary = {}
            for enabled in (False, True):
                selected_runs = [r for r in runs if r["cache"] == enabled]
                if not selected_runs:
                    continue
                summary[str(enabled)] = {
                    key: dict(mean_s=statistics.mean(values), stdev_s=statistics.stdev(values))
                    for key, values in [("request", [r["request_s"] for r in selected_runs])]
                    + [
                        (stage, [r["stages_s"][stage] for r in selected_runs])
                        for stage in ("vae_encode", "dit", "vae_decode")
                    ]
                }
            report["summary"] = summary
            report["status"] = "PASS"
        finally:
            active = False
            captures.clear()


if __name__ == "__main__":
    main()
