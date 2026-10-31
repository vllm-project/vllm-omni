# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end validation of the Ring-Flash-Attention optimization on Wan2.2 T2V.

Text-to-*video* counterpart of ``bench_qwen_image_ring_opt.py``. The image bench
showed no clear end-to-end win because a 1024x1024 image is only ~4k latent
tokens, so ring attention (and therefore its comm/merge optimization) is a tiny
slice of the total. A video is far longer: Wan2.2 at 480x832x81f is ~32k latent
tokens (8x the image), so the self-attention — and the ring comm/merge the opt
targets — is a much bigger share of the DiT time. This bench measures whether
that makes the optimization visible end-to-end.

Runs the *same* seeded Wan2.2 text-to-video generation twice, changing only the
ring-attention implementation:

  * baseline  (--opt 0): the upstream ring forward, from
               ``tests/diffusion/attention/ring_baseline.py``, which circulates
               K and V as two separate transfers
  * optimized (--opt 1): the shipped ``ring_flash_attn_forward``, which packs
               K and V into one transfer per hop on FA/FA3

The package ships one ring implementation and reads no env var to pick one, so
the ``--opt 0`` leg injects a ``sitecustomize`` on PYTHONPATH that rebinds
``ring_flash_attn_forward`` to the baseline inside every (spawned) engine
worker; the optimized leg leaves the engine untouched (see
``_install_worker_injection``).

Then it reports:
  * CORRECTNESS - per-pixel diff between the two videos (should be near-identical;
    only bf16 round-off in the attention merge differs).
  * SPEED       - median end-to-end generation time, baseline vs optimized.

Because only the ring-attention kernel differs between the two runs, any timing
delta is attributable to the optimization. (The end-to-end speedup is diluted by
text-encode + VAE-decode + the HSDP weight all-gather, all shared and unchanged;
larger --num-frames / --height / --width / --steps raise attention's share.)

Each generation runs in its own subprocess so the distributed engine and the
baseline/optimized selection are set cleanly per run. Ring SP is exercised via
``--ring-degree N`` (N GPUs). The A14B backbone (two 14B experts, ~54 GiB in
bf16) does not fit replicated on one card, so ``--use-hsdp`` (default on) shards
the DiT weights across the ring group; the all-gather it adds is identical in
both runs, so it does not bias the comparison.

The ring path uses FA3 (a CUDA-13 build). On a box whose system driver predates
CUDA 13, drop the CUDA-13 forward-compat driver in ``<venv>/cuda13-compat`` (a
``cuda-compat-13-0`` extraction) and this script puts it on LD_LIBRARY_PATH for
itself and the Omni worker subprocesses. No-op if that directory is absent.

Usage:

    cd projs/learn/vllm-omni
    .venv/bin/python tests/diffusion/attention/bench_wan_t2v_ring_opt.py \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --ring-degree 4 --steps 15 --height 480 --width 832 --num-frames 81 \
        --runs 3 --enforce-eager

``--model`` accepts a local path or a HuggingFace id; pass it at runtime (no
hard-coded path). ``--workdir`` (default a temp dir) is where the two videos,
frame ``.npy`` dumps and timing JSON are written.
"""

import argparse
import glob
import json
import os
import site
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _cu13_lib_dirs():
    """Directories that must be on LD_LIBRARY_PATH for the FA3 (CUDA-13) ring path:

      * ``<venv>/cuda13-compat``            - forward-compat *driver* (libcuda.so.1)
                                              from a cuda-compat-13-0 extraction, so
                                              the CUDA-13 runtime loads on an older
                                              system driver. Must precede the system
                                              libcuda, hence listed first.
      * ``site-packages/nvidia/cu13/lib``   - CUDA-13 runtime (libcudart.so.13).

    Returns only the dirs that actually exist (so an FA2/torch-only or
    natively-CUDA-13 box is unaffected).
    """
    dirs = []
    compat = os.path.join(sys.prefix, "cuda13-compat")
    if glob.glob(os.path.join(compat, "libcuda.so*")):
        dirs.append(compat)
    bases = list(site.getsitepackages()) if hasattr(site, "getsitepackages") else []
    for base in bases:
        d = os.path.join(base, "nvidia", "cu13", "lib")
        if glob.glob(os.path.join(d, "libcudart.so.13")):
            dirs.append(d)
    return dirs


def _ensure_cu13_runtime_on_ld_path():
    """Prepend the CUDA-13 driver+runtime dirs to LD_LIBRARY_PATH and re-exec so
    this process — and the Omni worker subprocesses it spawns (they inherit the
    env) — can load FA3. No-op if the dirs are absent."""
    dirs = _cu13_lib_dirs()
    if not dirs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = cur.split(":") if cur else []
    need = [d for d in dirs if d not in parts]
    if need:
        os.environ["LD_LIBRARY_PATH"] = ":".join(need + parts)
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _install_worker_injection(*, baseline: bool, force_fa3: bool):
    """Patch the engine's spawned worker ranks before ``Omni`` is built.

    The diffusion engine spawns its worker ranks with ``multiprocessing``
    ``spawn`` (fresh interpreters), so an in-process monkeypatch here would
    never reach them. Instead we drop a ``sitecustomize.py`` in a temp dir and
    prepend it to ``PYTHONPATH``: every spawned worker auto-imports it at
    interpreter startup, and ``spawn`` inherits the env.

    ``baseline`` rebinds ``ring_flash_attn_forward`` to
    ``tests/diffusion/attention/ring_baseline.py``. The package ships a single
    ring implementation and reads no env var to select one, so the A/B baseline
    leg has to come from outside. The autograd wrapper looks that name up on
    the module at call time, so the rebind takes effect.

    ``force_fa3`` makes ``parallel/ring.py`` skip its first backend preference
    so selection falls through to the source-built FA3. There is no config for
    this: ``can_use_vllm_fa`` wins whenever vLLM's bundled FlashAttention is
    importable, which is every normal CUDA install. Apply it to BOTH legs so
    the comparison stays FA3 against FA3.

    Neither block is guarded by a try/except: a failed injection would
    silently measure the wrong thing and report a meaningless ratio.
    """
    if not (baseline or force_fa3):
        return None
    inject_dir = tempfile.mkdtemp(prefix="wan_ring_inject_")
    baseline_dir = str(Path(__file__).resolve().parent)
    lines = [
        "# Auto-generated by bench_wan_t2v_ring_opt.py.\n",
        "# Patches each spawned diffusion worker. No-op unless its guard is set.\n",
        "import os\n",
        "import sys\n",
        "\n",
        "if os.environ.get('WAN_BENCH_FORCE_FA3') == '1':\n",
        "    from vllm_omni.diffusion.attention.parallel import ring as _ring\n",
        "\n",
        "    _ring._can_use_vllm_flash_attn = lambda _device: False\n",
        "\n",
        "if os.environ.get('WAN_BENCH_FORCE_BASELINE_RING') == '1':\n",
        f"    sys.path.insert(0, {baseline_dir!r})\n",
        "    from ring_baseline import baseline_ring_flash_attn_forward\n",
        "\n",
        "    from vllm_omni.diffusion.attention.backends import ring_flash_attn as _r\n",
        "\n",
        "    _r.ring_flash_attn_forward = baseline_ring_flash_attn_forward\n",
    ]
    with open(os.path.join(inject_dir, "sitecustomize.py"), "w") as f:
        f.writelines(lines)
    if baseline:
        os.environ["WAN_BENCH_FORCE_BASELINE_RING"] = "1"
    if force_fa3:
        os.environ["WAN_BENCH_FORCE_FA3"] = "1"
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join([inject_dir, prev]) if prev else inject_dir
    return inject_dir


# ===========================================================================
# Frame extraction: pull a (T, H, W, 3) uint8 array out of omni.generate().
# Adapted from examples/offline_inference/text_to_video/text_to_video.py, which
# handles the several shapes Omni video pipelines can return.
# ===========================================================================
def _frames_to_numpy(result):
    import numpy as np
    import torch

    from vllm_omni.outputs import OmniRequestOutput

    frames = result
    if isinstance(frames, list):
        frames = frames[0] if frames else None

    if isinstance(frames, OmniRequestOutput):
        # Upstream flattened pipeline outputs directly onto the OmniRequestOutput
        # (the old nested ``.request_output`` attribute was removed), so read the
        # video straight from ``.images`` -- matching the current
        # examples/offline_inference/text_to_video/text_to_video.py extraction.
        if not frames.images:
            raise ValueError("No video frames in OmniRequestOutput.")
        imgs = frames.images
        if len(imgs) == 1 and isinstance(imgs[0], dict):
            frames = imgs[0].get("frames") or imgs[0].get("video")
        elif len(imgs) == 1 and isinstance(imgs[0], tuple) and len(imgs[0]) == 2:
            frames = imgs[0][0]
        else:
            frames = imgs

    if isinstance(frames, list) and frames:
        first = frames[0]
        if isinstance(first, tuple) and len(first) == 2:
            frames = first[0]
        elif isinstance(first, dict):
            frames = first.get("frames") or first.get("video")
        elif isinstance(first, list):
            frames = first

    if isinstance(frames, tuple) and len(frames) == 2:
        frames = frames[0]
    if isinstance(frames, dict):
        frames = frames.get("frames") or frames.get("video")

    if frames is None:
        raise ValueError("Could not locate video frames in generate() output.")

    def _to_thwc_float(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu()
            if t.dim() == 5:  # (B, C, T, H, W) or (B, T, C, H, W)
                t = t[0]
            if t.dim() == 4 and t.shape[0] in (3, 4):  # (C, T, H, W)
                t = t.permute(1, 2, 3, 0)
            elif t.dim() == 4 and t.shape[1] in (3, 4):  # (T, C, H, W)
                t = t.permute(0, 2, 3, 1)
            if t.is_floating_point():
                t = t.clamp(-1, 1) * 0.5 + 0.5 if t.min() < -0.01 else t.clamp(0, 1)
            return t.float().numpy()
        if isinstance(x, np.ndarray):
            a = x
            if a.ndim == 5:
                a = a[0]
            if np.issubdtype(a.dtype, np.integer):
                a = a.astype(np.float32) / 255.0
            return a
        # list of per-frame images / arrays / tensors
        out = []
        for fr in x:
            if isinstance(fr, torch.Tensor):
                ft = fr.detach().cpu()
                if ft.dim() == 3 and ft.shape[0] in (3, 4):
                    ft = ft.permute(1, 2, 0)
                if ft.is_floating_point():
                    ft = ft.clamp(-1, 1) * 0.5 + 0.5 if ft.min() < -0.01 else ft.clamp(0, 1)
                out.append(ft.float().numpy())
            elif isinstance(fr, np.ndarray):
                out.append(fr.astype(np.float32) / 255.0 if np.issubdtype(fr.dtype, np.integer) else fr)
            else:  # PIL
                out.append(np.asarray(fr).astype(np.float32) / 255.0)
        return np.stack(out, axis=0)

    arr = _to_thwc_float(frames)
    arr = np.asarray(arr)
    # Collapse any leading singleton/batch dims down to (T, H, W, C).
    while arr.ndim > 4 and arr.shape[0] == 1:
        arr = arr[0]
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


# ===========================================================================
# Worker: one generation run under a fixed ring-attention implementation.
# ===========================================================================
def run_worker(args):
    # The package ships one ring implementation and offers no runtime toggle.
    # For the baseline A/B leg, inject a sitecustomize (before Omni spawns its
    # worker ranks, so they inherit PYTHONPATH) that rebinds the ring forward
    # to the pre-optimization baseline. Each leg is its own subprocess, so this
    # environment change cannot leak into the optimized leg.
    _install_worker_injection(baseline=args.opt == 0, force_fa3=args.force_fa3)

    import numpy as np
    import torch

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # Wan2.2 resolves through the pipeline registry, whose stage-config path only
    # honors the *flat* diffusion-parallel fields (ring_degree/ulysses_degree/
    # use_hsdp/...) declared on StageDeployConfig, and takes the worker GPU set from
    # the stage ``devices`` string. A pre-built DiffusionParallelConfig object is
    # silently dropped there (build_stage_runtime_overrides drops the orchestrator-
    # owned ``parallel_config`` key) and ``devices`` defaults to a single GPU, which
    # collapses the run to ring=1 on GPU0. So pass the flat fields + an explicit
    # ``devices`` list instead of a parallel_config object.
    omni_kwargs = dict(
        model=args.model,
        ring_degree=args.ring_degree,
        ulysses_degree=1,
        use_hsdp=args.use_hsdp,
        devices=",".join(str(i) for i in range(args.ring_degree)),
        enforce_eager=args.enforce_eager,
        vae_use_tiling=True,
        # Loading the A14B backbone (two 14B experts, ~108 GiB of shards on disk)
        # plus the engine's profiling forward blows past the 600s default.
        init_timeout=args.init_timeout,
        stage_init_timeout=args.init_timeout,
    )
    if args.flow_shift is not None:
        omni_kwargs["flow_shift"] = args.flow_shift
    if args.boundary_ratio is not None:
        omni_kwargs["boundary_ratio"] = args.boundary_ratio

    omni = Omni(**omni_kwargs)

    prompt_dict = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt

    def make_sampling_params():
        # Fresh, identically-seeded params each call -> every run does the same work.
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        extra = {}
        # Second-expert CFG only exists on the two-expert Wan2.2 backbones
        # (A14B). A dense checkpoint such as TI2V-5B has no high-noise expert.
        if args.guidance_scale_high is not None:
            extra["guidance_scale_2"] = args.guidance_scale_high
        return OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            generator=generator,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_outputs_per_prompt=1,
            **extra,
        )

    for _ in range(args.warmup):  # warmup (compile / caches / autotune)
        omni.generate(prompt_dict, make_sampling_params())

    times = []
    outputs = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        outputs = omni.generate(prompt_dict, make_sampling_params())
        times.append(time.perf_counter() - t0)

    frames = _frames_to_numpy(outputs)
    Path(args.npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.npy, frames)
    if args.out:
        try:
            from diffusers.utils import export_to_video

            export_to_video([f for f in (frames.astype(np.float32) / 255.0)], args.out, fps=args.fps)
        except Exception as e:  # saving an mp4 is a convenience, not the measurement
            print(f"[worker opt={args.opt}] mp4 export skipped: {e}")

    times_sorted = sorted(times)
    json.dump(
        {
            "opt": args.opt,
            "ring_degree": args.ring_degree,
            "times": times,
            "median": times_sorted[len(times_sorted) // 2],
            "min": times_sorted[0],
            "frames_shape": list(frames.shape),
        },
        open(args.timing, "w"),
    )
    print(
        f"[worker opt={args.opt}] frames={frames.shape} "
        f"times={[f'{t:.3f}' for t in times]}s "
        f"median={times_sorted[len(times_sorted) // 2]:.3f}s -> {args.npy}"
    )


# ===========================================================================
# Driver: run baseline + optimized as subprocesses, then compare.
# ===========================================================================
def run_driver(args):
    import numpy as np

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    results = {}

    for opt in (0, 1):
        npy = workdir / f"wan_opt{opt}.npy"
        out = workdir / f"wan_opt{opt}.mp4"
        timing = workdir / f"timing_opt{opt}.json"
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--role",
            "worker",
            "--opt",
            str(opt),
            "--model",
            args.model,
            "--ring-degree",
            str(args.ring_degree),
            "--steps",
            str(args.steps),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--num-frames",
            str(args.num_frames),
            "--guidance-scale",
            str(args.guidance_scale),
            "--prompt",
            args.prompt,
            "--seed",
            str(args.seed),
            "--runs",
            str(args.runs),
            "--warmup",
            str(args.warmup),
            "--fps",
            str(args.fps),
            "--init-timeout",
            str(args.init_timeout),
            "--npy",
            str(npy),
            "--out",
            str(out),
            "--timing",
            str(timing),
        ]
        if args.negative_prompt:
            cmd += ["--negative-prompt", args.negative_prompt]
        if args.flow_shift is not None:
            cmd += ["--flow-shift", str(args.flow_shift)]
        if args.boundary_ratio is not None:
            cmd += ["--boundary-ratio", str(args.boundary_ratio)]
        if args.guidance_scale_high is not None:
            cmd += ["--guidance-scale-high", str(args.guidance_scale_high)]
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        if args.force_fa3:
            cmd.append("--force-fa3")
        cmd.append("--use-hsdp" if args.use_hsdp else "--no-hsdp")
        label = "baseline (opt=0)" if opt == 0 else "optimized (opt=1)"
        print(f"\n{'=' * 70}\n[driver] launching {label}\n{'=' * 70}")
        rc = subprocess.run(cmd, env=os.environ.copy()).returncode
        if rc != 0:
            raise SystemExit(f"[driver] worker opt={opt} failed (rc={rc})")
        results[opt] = json.load(open(timing))

    a = np.load(workdir / "wan_opt0.npy").astype(np.float64)
    b = np.load(workdir / "wan_opt1.npy").astype(np.float64)
    if a.shape != b.shape:
        raise SystemExit(f"video shape mismatch {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    max_diff, mean_diff = diff.max(), diff.mean()
    mse = (diff**2).mean()
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    frac_off = (diff > 0).mean() * 100.0

    tb, to = results[0]["median"], results[1]["median"]
    tb_min, to_min = results[0]["min"], results[1]["min"]

    print(f"\n{'#' * 70}")
    print("Wan2.2-T2V-A14B  |  Ring-Flash-Attention: baseline vs optimized")
    print(f"{'#' * 70}")
    print(
        f"config: ring_degree={args.ring_degree}  hsdp={args.use_hsdp}  steps={args.steps}  "
        f"size={args.width}x{args.height}  frames={args.num_frames}  "
        f"runs={args.runs}  seed={args.seed}"
    )
    print(f"video shape (T,H,W,C): {tuple(int(x) for x in a.shape)}")
    print("-" * 70)
    print("CORRECTNESS (optimized video vs baseline video):")
    print(f"  max pixel diff : {max_diff:.0f} / 255")
    print(f"  mean pixel diff: {mean_diff:.4f}")
    print(f"  pixels differing: {frac_off:.3f}%")
    print(f"  PSNR           : {psnr:.1f} dB")
    verdict = (
        "PASS (visually identical)"
        if (psnr >= 40.0 or max_diff <= 2)
        else "REVIEW (diff larger than bf16 round-off expectation)"
    )
    print(f"  -> {verdict}")
    print("-" * 70)
    print("SPEED (end-to-end omni.generate, lower is better):")
    print(f"  baseline : median {tb:.3f}s   (min {tb_min:.3f}s)")
    print(f"  optimized: median {to:.3f}s   (min {to_min:.3f}s)")
    print(f"  -> speedup: {tb / to:.3f}x (median), {tb_min / to_min:.3f}x (min)")
    print(f"  baseline runs : {[f'{t:.3f}' for t in results[0]['times']]}")
    print(f"  optimized runs: {[f'{t:.3f}' for t in results[1]['times']]}")
    print(f"{'#' * 70}")
    print(f"videos: {workdir / 'wan_opt0.mp4'}  vs  {workdir / 'wan_opt1.mp4'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=["driver", "worker"], default="driver")
    p.add_argument(
        "--model",
        required=True,
        help="Path or HF id of the Wan2.2 T2V checkpoint (e.g. Wan-AI/Wan2.2-T2V-A14B-Diffusers).",
    )
    p.add_argument("--ring-degree", type=int, default=4, help="GPUs for ring SP")
    p.add_argument(
        "--use-hsdp",
        action="store_true",
        default=True,
        help="Shard DiT weights across the ring group (needed for A14B).",
    )
    p.add_argument("--no-hsdp", dest="use_hsdp", action="store_false")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--guidance-scale", type=float, default=4.0, help="CFG, low-noise expert")
    p.add_argument(
        "--guidance-scale-high",
        type=float,
        default=None,
        help="CFG for the high-noise expert; two-expert Wan2.2 backbones (A14B) only",
    )
    p.add_argument("--flow-shift", type=float, default=12.0, help="Scheduler flow_shift (12.0 @480p, 5.0 @720p)")
    p.add_argument("--boundary-ratio", type=float, default=None, help="Wan2.2 low/high split (default from model)")
    p.add_argument(
        "--prompt", default="Two anthropomorphic cats in boxing gear fight intensely on a spotlighted stage."
    )
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--seed", type=int, default=142)
    p.add_argument("--runs", type=int, default=3, help="timed generations per setting")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--init-timeout", type=int, default=3600, help="Omni orchestrator/stage init timeout (s)")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument(
        "--force-fa3",
        action="store_true",
        help="Make ring selection skip vLLM's bundled FA so it falls through to source-built FA3",
    )
    p.add_argument("--workdir", default=os.path.join(tempfile.gettempdir(), "wan_t2v_ring_opt"))
    # worker-only outputs
    p.add_argument("--opt", type=int, choices=[0, 1], default=0)
    p.add_argument("--npy", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--timing", default=None)
    return p.parse_args()


def main():
    _ensure_cu13_runtime_on_ld_path()
    args = parse_args()
    if args.role == "worker":
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
