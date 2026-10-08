#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 eager-mode functional + latency sweep over batch size × camera views.

Runs the real ``lerobot/pi05_base`` weights on GPU and, for every
(batch_size, num_views) cell, checks that a full 10-step flow-matching
``sample_actions`` produces a correctly shaped finite action chunk, then times
it. Model level on purpose: the OpenPI serving path is B=1 by construction
(``pi05.yaml`` sets ``max_num_seqs: 1``), so batch behaviour can only be
exercised here.

No CUDA Graph, no Triton, no caching — this is the eager baseline that later
optimization work is measured against.

    python bench_pi05.py --out results/            # full sweep
    python bench_pi05.py --bs 1 --views 3 --iters 5

Writes ``pi05_sweep.csv`` and ``pi05_sweep.md`` into --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

ACTION_DIM = 32
STATE_DIM = 32
ACTION_HORIZON = 50
MAX_TOKEN_LEN = 200
NUM_STATE_BINS = 256
NUM_STEPS = 10


def resolve_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=path, repo_type="model")


def build_model(checkpoint_dir: str, device: str, dtype: str):
    from vllm_omni.diffusion.models.pi05 import Pi05Config, Pi05ForActionPrediction

    cfg = Pi05Config(
        max_action_dim=ACTION_DIM,
        max_state_dim=STATE_DIM,
        chunk_size=ACTION_HORIZON,
        num_inference_steps=NUM_STEPS,
        tokenizer_max_length=MAX_TOKEN_LEN,
        state_num_bins=NUM_STATE_BINS,
        dtype=dtype,
    )
    model = Pi05ForActionPrediction(cfg)
    model.to(device).eval()

    import safetensors.torch

    state = safetensors.torch.load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    filled = model.load_weights(list(state.items()))
    if not filled:
        raise RuntimeError("no weights loaded — the remap rules are broken")
    return model, cfg


def make_inputs(batch_size: int, num_views: int, device: str, seed: int = 0):
    """Synthetic but deterministic inputs, shaped exactly as the pipeline feeds
    them: one (B, 3, 224, 224) tensor per camera plus a padded token block."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    images = [torch.rand(batch_size, 3, 224, 224, generator=g).to(device) * 2 - 1 for _ in range(num_views)]
    image_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(num_views)]
    # π0.5 pads the prompt (task text + discretized state) to 200 tokens.
    lang_tokens = torch.randint(0, 1000, (batch_size, MAX_TOKEN_LEN), generator=g).to(device)
    lang_masks = torch.ones(batch_size, MAX_TOKEN_LEN, dtype=torch.bool, device=device)
    noise = torch.randn(batch_size, ACTION_HORIZON, ACTION_DIM, generator=g).to(device)
    return images, image_masks, lang_tokens, lang_masks, noise


def time_cell(model, batch_size: int, num_views: int, device: str, warmup: int, iters: int) -> dict:
    images, image_masks, lang_tokens, lang_masks, noise = make_inputs(batch_size, num_views, device)
    kwargs = dict(
        images=images,
        image_masks=image_masks,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        noise=noise,
        num_steps=NUM_STEPS,
    )

    if device.startswith("cuda"):
        torch.accelerator.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(warmup):
            actions = model.sample_actions(**kwargs)
        if device.startswith("cuda"):
            torch.accelerator.synchronize()

        latencies = []
        for _ in range(iters):
            start = time.perf_counter()
            actions = model.sample_actions(**kwargs)
            if device.startswith("cuda"):
                torch.accelerator.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)

    expected = (batch_size, ACTION_HORIZON, ACTION_DIM)
    ok_shape = tuple(actions.shape) == expected
    ok_finite = bool(torch.isfinite(actions).all())

    latencies.sort()
    return {
        "batch_size": batch_size,
        "num_views": num_views,
        "shape": "x".join(str(d) for d in actions.shape),
        "expected_shape": "x".join(str(d) for d in expected),
        "shape_ok": ok_shape,
        "finite": ok_finite,
        "status": "PASS" if (ok_shape and ok_finite) else "FAIL",
        "mean_ms": round(statistics.fmean(latencies), 2),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 2),
        "min_ms": round(latencies[0], 2),
        "max_ms": round(latencies[-1], 2),
        "ms_per_sample": round(statistics.fmean(latencies) / batch_size, 2),
        "peak_mem_gb": (
            round(torch.accelerator.max_memory_allocated() / 2**30, 2) if device.startswith("cuda") else None
        ),
    }


def environment(device: str) -> dict:
    import importlib.metadata as md

    def version(pkg):
        try:
            return md.version(pkg)
        except Exception:
            return "MISSING"

    info = {
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm": version("vllm"),
        "vllm-omni": version("vllm-omni"),
        "transformers": version("transformers"),
        "num_steps": NUM_STEPS,
        "mode": "eager (no CUDA Graph / Triton / caching)",
    }
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
        info["capability"] = "sm_{}{}".format(*torch.cuda.get_device_capability(0))
    return info


def write_reports(rows: list[dict], env: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "pi05_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# π0.5 eager sweep: batch size × camera views",
        "",
        "```",
        json.dumps(env, indent=2),
        "```",
        "",
        "| bs | views | status | shape | p50 ms | p95 ms | ms/sample | peak GB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['num_views']} | {row['status']} | {row['shape']} | "
            f"{row['p50_ms']} | {row['p95_ms']} | {row['ms_per_sample']} | {row['peak_mem_gb']} |"
        )
    (out_dir / "pi05_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {csv_path} and {out_dir / 'pi05_sweep.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("PI05_MODEL_PATH", "lerobot/pi05_base"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--views", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out", default="pi05_bench")
    args = parser.parse_args()

    checkpoint_dir = resolve_checkpoint(args.model)
    print(f"checkpoint: {checkpoint_dir}")
    env = environment(args.device)
    print(json.dumps(env, indent=2))

    model, _ = build_model(checkpoint_dir, args.device, args.dtype)
    print("model built\n")

    rows = []
    for num_views in args.views:
        for batch_size in args.bs:
            label = f"bs={batch_size} views={num_views}"
            try:
                row = time_cell(model, batch_size, num_views, args.device, args.warmup, args.iters)
            except Exception as exc:  # a cell that OOMs or errors is a result, not a crash
                row = {
                    "batch_size": batch_size,
                    "num_views": num_views,
                    "shape": "-",
                    "expected_shape": f"{batch_size}x{ACTION_HORIZON}x{ACTION_DIM}",
                    "shape_ok": False,
                    "finite": False,
                    "status": f"ERROR: {type(exc).__name__}: {exc}"[:200],
                    "mean_ms": None,
                    "p50_ms": None,
                    "p95_ms": None,
                    "min_ms": None,
                    "max_ms": None,
                    "ms_per_sample": None,
                    "peak_mem_gb": None,
                }
                if args.device.startswith("cuda"):
                    torch.accelerator.empty_cache()
            print(f"{label:22s} {row['status']:12s} p50={row['p50_ms']} ms  peak={row['peak_mem_gb']} GB")
            rows.append(row)

    write_reports(rows, env, Path(args.out))
    failed = [r for r in rows if r["status"] != "PASS"]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} cells PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
