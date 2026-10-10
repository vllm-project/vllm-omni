# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    mp.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--pp", type=int, required=True)
    p.add_argument("--cfg", type=int, required=True)
    p.add_argument("--mode", choices=["none", "full", "collect", "cache"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=["smoke", "calibration", "validation"], default="smoke")
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--cache-warmup-steps", type=int, default=0)
    p.add_argument("--paired", action="store_true")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    root = Path(os.environ.get("WAN_VALIDATION_ROOT", Path(__file__).parent))
    a.out.mkdir(parents=True, exist_ok=True)
    source = Path(__import__("vllm_omni").__file__).resolve().parent.parent
    tracked = [
        "vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py",
        "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
        "vllm_omni/diffusion/cache/teacache/hook.py",
        "vllm_omni/diffusion/cache/teacache/extractors.py",
        "vllm_omni/diffusion/cache/teacache/backend.py",
        "vllm_omni/diffusion/distributed/pipeline_parallel.py",
    ]
    provenance = {
        "sha": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_hashes": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in tracked},
        "driver_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "harness_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
        },
    }
    (a.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    assert a.mode == "none" or os.environ.get("WAN_TRACE_MODE") == a.mode
    if a.mode != "none":
        import wan_instrument

        assert wan_instrument.backend.apply_teacache_hook is wan_instrument.apply
    kw = dict(
        model=str(root / "model"),
        model_class_name="WanPipeline",
        dtype="bfloat16",
        pipeline_parallel_size=a.pp,
        cfg_parallel_size=a.cfg,
        tensor_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        enforce_eager=True,
    )
    if a.mode != "none":
        kw.update(
            cache_backend="tea_cache", cache_config={"rel_l1_thresh": a.threshold, "coefficients": [0, 0, 0, 0, 1]}
        )
    manifest = json.loads((root / "prompts.json").read_text())
    prompts = manifest["calibration"][:1] if a.split == "smoke" else manifest[a.split]
    if a.limit:
        prompts = prompts[: a.limit]
    seeds = [42] if a.split == "smoke" else ([17, 29] if a.split == "calibration" else [101, 202, 303])
    steps = 6 if a.split == "smoke" else 50
    side = 256 if a.split == "smoke" else 512
    frames = 5 if a.split == "smoke" else 17
    omni = Omni(**kw)
    try:
        requests = [("warmup", "A small red ball rolls on a wooden table.", 0)] + [
            (f"{i:02d}-{seed}", prompt, seed) for i, prompt in enumerate(prompts) for seed in seeds
        ]
        records = []
        by_mode = {"none": [], "cache": []}
        if a.paired:
            assert a.mode == "cache" and os.environ.get("WAN_CONTROL")
            requests = [
                (name, prompt, seed, mode)
                for i, (name, prompt, seed) in enumerate(requests)
                for mode in (("none", "cache") if i % 2 == 0 else ("cache", "none"))
            ]
        else:
            requests = [(*row, a.mode) for row in requests]
        for name, prompt, seed, mode in requests:
            out_dir = a.out / mode if a.paired else a.out
            out_dir.mkdir(parents=True, exist_ok=True)
            if os.environ.get("WAN_CONTROL"):
                control = Path(os.environ["WAN_CONTROL"])
                tmp = control.with_suffix(".tmp")
                tmp.write_text(json.dumps({"mode": mode, "name": name, "cache_warmup_steps": a.cache_warmup_steps}))
                tmp.replace(control)
            params = OmniDiffusionSamplingParams(
                height=side,
                width=side,
                num_frames=frames,
                num_inference_steps=steps,
                guidance_scale=5.0,
                seed=seed,
                generator=torch.Generator("cuda").manual_seed(seed),
            )
            start = time.perf_counter()
            output = omni.generate(
                {"prompt": prompt, "negative_prompt": "blurry, low quality, text, watermark"}, params
            )[0]
            elapsed = time.perf_counter() - start
            images = output.images
            if isinstance(images, list) and len(images) == 1:
                images = images[0]
            if isinstance(images, torch.Tensor):
                images = images.detach().cpu().float().numpy()
            images = np.asarray(images)
            if images.ndim == 5 and images.shape[0] == 1:
                images = images[0]
            if images.ndim == 4 and images.shape[0] == 3:
                images = images.transpose(1, 2, 3, 0)
            assert images.ndim == 4 and images.shape[-1] == 3, images.shape
            assert np.isfinite(images).all()
            if images.dtype == np.uint8:
                images = images.astype(np.float32) / 255
            assert images.min() >= -0.001 and images.max() <= 1.001, (images.min(), images.max())
            if name != "warmup":
                assert images.shape[0] == frames, images.shape
                np.save(out_dir / (name + ".npy"), images)
                export_to_video(list(images), str(out_dir / (name + ".mp4")), fps=8)
                for idx in (0, frames // 2, frames - 1):
                    Image.fromarray((images[idx].clip(0, 1) * 255).astype("uint8")).save(
                        out_dir / f"{name}-frame{idx}.png"
                    )
                records = by_mode[mode] if a.paired else records
                records.append(dict(name=name, prompt=prompt, seed=seed, seconds=elapsed, shape=list(images.shape)))
                (out_dir / "results.json").write_text(
                    json.dumps({"args": {**vars(a), "out": str(a.out)}, "records": records}, indent=2)
                )
            print(json.dumps({"name": name, "seconds": elapsed, "shape": list(images.shape)}), flush=True)
        if a.mode != "none":
            assert any(Path(os.environ["WAN_TRACE_DIR"]).rglob("rank-*.jsonl")), "Missing real-hook worker traces"
    finally:
        omni.close()


if __name__ == "__main__":
    main()
