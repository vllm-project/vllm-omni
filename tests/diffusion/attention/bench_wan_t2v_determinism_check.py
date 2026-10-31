# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Determinism control for bench_wan_t2v_ring_opt.py.

The A/B bench reports a frame difference between the baseline and optimized
videos. That number only means something once you know how much two identical
runs already differ. This probe settles it: load ONE engine (a single, fixed
ring implementation) and run the SAME seeded generation twice, then report the
frame diff. If two identical runs already diverge by a comparable amount, the
A/B difference is inherent engine nondeterminism rather than the optimization.

Run it with the same shape, steps, ring degree and model as the A/B leg you
are explaining, otherwise the two numbers are not comparable.
"""

import argparse
import glob
import os
import site
import sys


def _ensure_cu13():
    dirs = []
    compat = os.path.join(sys.prefix, "cuda13-compat")
    if glob.glob(os.path.join(compat, "libcuda.so*")):
        dirs.append(compat)
    for base in site.getsitepackages() if hasattr(site, "getsitepackages") else []:
        d = os.path.join(base, "nvidia", "cu13", "lib")
        if glob.glob(os.path.join(d, "libcudart.so.13")):
            dirs.append(d)
    if not dirs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = cur.split(":") if cur else []
    need = [d for d in dirs if d not in parts]
    if need:
        os.environ["LD_LIBRARY_PATH"] = ":".join(need + parts)
        os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    _ensure_cu13()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        required=True,
        help="Path or HF id of the Wan2.2 T2V checkpoint (e.g. Wan-AI/Wan2.2-T2V-A14B-Diffusers).",
    )
    p.add_argument("--opt", type=int, default=0)
    p.add_argument("--ring-degree", type=int, default=4)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--seed", type=int, default=142)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument(
        "--guidance-scale-high",
        type=float,
        default=None,
        help="CFG for the high-noise expert; two-expert Wan2.2 backbones (A14B) only",
    )
    p.add_argument("--flow-shift", type=float, default=None)
    p.add_argument(
        "--use-hsdp",
        action="store_true",
        default=True,
        help="Shard DiT weights across the ring group (needed for A14B).",
    )
    p.add_argument("--no-hsdp", dest="use_hsdp", action="store_false")
    p.add_argument(
        "--force-fa3",
        action="store_true",
        help="Make ring selection skip vLLM's bundled FA so it falls through to source-built FA3",
    )
    args = p.parse_args()

    import numpy as np
    import torch

    # reuse helpers from the bench
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_wan_t2v_ring_opt import _frames_to_numpy, _install_worker_injection

    # The package ships one ring implementation. For the opt=0 control, rebind
    # the ring forward to the pre-optimization baseline by injecting a
    # sitecustomize that the engine's spawned workers inherit (before Omni
    # build). Either setting is a valid control: both generations use the same
    # one, which is the whole point.
    if args.opt == 0:
        _install_worker_injection(baseline=True, force_fa3=args.force_fa3)

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # Pass the flat diffusion-parallel fields + an explicit ``devices`` list (not a
    # pre-built parallel_config object): Wan2.2 resolves through the pipeline
    # registry, which drops a passed parallel_config and defaults to a single GPU.
    # use_hsdp shards the A14B weights so they fit across the ring group; it does
    # not affect run-to-run determinism (identical in both generations).
    omni_kwargs = dict(
        model=args.model,
        ring_degree=args.ring_degree,
        ulysses_degree=1,
        use_hsdp=args.use_hsdp,
        devices=",".join(str(i) for i in range(args.ring_degree)),
        enforce_eager=True,
        vae_use_tiling=True,
        init_timeout=3600,
        stage_init_timeout=3600,
    )
    if args.flow_shift is not None:
        omni_kwargs["flow_shift"] = args.flow_shift
    omni = Omni(**omni_kwargs)
    prompt = {"prompt": "Two anthropomorphic cats in boxing gear fight intensely on a spotlighted stage."}

    def gen():
        extra = {}
        if args.guidance_scale_high is not None:
            extra["guidance_scale_2"] = args.guidance_scale_high
        sp = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_outputs_per_prompt=1,
            **extra,
        )
        return _frames_to_numpy(omni.generate(prompt, sp))

    a = gen().astype(np.float64)
    b = gen().astype(np.float64)
    diff = np.abs(a - b)
    mse = (diff**2).mean()
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    print(f"\n##### DETERMINISM CONTROL (opt={args.opt}, same seed twice) #####")
    print(f"  shape           : {tuple(int(x) for x in a.shape)}")
    print(f"  max pixel diff  : {diff.max():.0f} / 255")
    print(f"  mean pixel diff : {diff.mean():.4f}")
    print(f"  pixels differing: {(diff > 0).mean() * 100:.3f}%")
    print(f"  PSNR            : {psnr:.1f} dB")
    print(f"  -> {'DETERMINISTIC (bit-identical)' if mse == 0 else 'NONDETERMINISTIC run-to-run'}")


if __name__ == "__main__":
    main()
