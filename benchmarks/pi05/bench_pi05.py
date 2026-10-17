#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 model-level functional + latency sweep over batch size × camera views.

Runs the real ``lerobot/pi05_base`` weights on GPU and, for every
(batch_size, num_views) cell, checks that a full 10-step flow-matching
``sample_actions`` produces a correctly shaped finite action chunk, then times
it. Model level on purpose: the OpenPI serving path is B=1 by construction
(``pi05.yaml`` sets ``max_num_seqs: 1``), so batch behaviour can only be
exercised here.

The model is built by ``Pi05Pipeline`` as the server builds it, from the deploy
config's ``model_config``: ``--dtype`` sets the serving dtype (bfloat16 is the
mixed layout serving runs), and the optimized path (CUDA graphs over the fused
Triton kernels) runs unless ``--enforce-eager`` is passed. The optimized path
serves batch 1; other batch sizes run eagerly.

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
from dataclasses import asdict
from pathlib import Path

import torch

# The deployed shapes (``vllm_omni/deploy/pi05.yaml``), which make_inputs feeds.
ACTION_DIM = 32
ACTION_HORIZON = 50
MAX_TOKEN_LEN = 200
NUM_STEPS = 10


def resolve_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=path, repo_type="model")


def build_model(
    checkpoint_dir: str,
    deploy_config: str,
    dtype: str,
    enforce_eager: bool,
    vision_precision: str | None = None,
    batch_cameras: bool | None = None,
    *,
    lm_precision: str | None = None,
    precompute_adarms: bool | None = None,
    hoist_denoise_invariants: bool | None = None,
    prefix_kv_only: bool | None = None,
    execution_mode: str | None = None,
):
    """The model ``Pi05Pipeline`` serves: its dtype layout, KV cache, and with
    ``enforce_eager`` unset its fused kernels and CUDA graphs."""
    import yaml
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    with open(deploy_config, encoding="utf-8") as f:
        (stage,) = yaml.safe_load(f)["stages"]
    custom_args = dict(stage.get("custom_pipeline_args") or {})
    if vision_precision is not None:
        custom_args["pi05_vision_precision"] = vision_precision
    if lm_precision is not None:
        custom_args["pi05_lm_precision"] = lm_precision
    if batch_cameras is not None:
        custom_args["pi05_batch_cameras"] = batch_cameras
    for key, value in (
        ("pi05_precompute_adarms", precompute_adarms),
        ("pi05_hoist_denoise_invariants", hoist_denoise_invariants),
        ("pi05_prefix_kv_only", prefix_kv_only),
    ):
        if value is not None:
            custom_args[key] = value
    execution_mode = execution_mode or ("eager" if enforce_eager else "graph")
    if execution_mode not in ("eager", "fused", "graph"):
        raise ValueError(f"Unsupported execution_mode={execution_mode!r}.")
    if enforce_eager and execution_mode != "eager":
        raise ValueError("--enforce-eager conflicts with --execution-mode fused/graph.")
    od_config = OmniDiffusionConfig(
        model=checkpoint_dir,
        dtype=dtype,
        model_config=stage["model_config"],
        enforce_eager=execution_mode != "graph",
        custom_pipeline_args=custom_args,
    )
    # Under the serving dtype as torch's default, as the server's loader builds
    # (and captures) the pipeline.
    with set_default_torch_dtype(od_config.dtype):
        pipeline = Pi05Pipeline(od_config=od_config)
    if execution_mode == "fused":
        pipeline.model.enable_fused_kernels()
    return pipeline.model.eval()


def make_inputs(batch_size: int, num_views: int, device: str, seed: int = 0, max_cameras: int = 3):
    """Synthetic but deterministic inputs, shaped exactly as the pipeline feeds
    them: one (B, 3, 224, 224) tensor per camera plus a padded token block."""
    if not 1 <= num_views <= max_cameras:
        raise ValueError(f"num_views must be between 1 and max_cameras={max_cameras}, got {num_views}.")
    g = torch.Generator(device="cpu").manual_seed(seed)
    images = [torch.rand(batch_size, 3, 224, 224, generator=g).to(device) * 2 - 1 for _ in range(num_views)]
    image_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(num_views)]
    for _ in range(max_cameras - num_views):
        images.append(torch.full_like(images[0], -1.0))
        image_masks.append(torch.zeros_like(image_masks[0]))
    # π0.5 pads the prompt (task text + discretized state) to 200 tokens.
    lang_tokens = torch.randint(0, 1000, (batch_size, MAX_TOKEN_LEN), generator=g).to(device)
    lang_masks = torch.ones(batch_size, MAX_TOKEN_LEN, dtype=torch.bool, device=device)
    noise = torch.randn(batch_size, ACTION_HORIZON, ACTION_DIM, generator=g).to(device)
    return images, image_masks, lang_tokens, lang_masks, noise


def time_cell(model, batch_size: int, num_views: int, device: str, warmup: int, iters: int) -> dict:
    images, image_masks, lang_tokens, lang_masks, noise = make_inputs(
        batch_size, num_views, device, max_cameras=model.config.max_cameras
    )
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


def environment(model, device: str) -> dict:
    import importlib.metadata as md

    def version(pkg):
        try:
            return md.version(pkg)
        except Exception:
            return "MISSING"

    schedule = model.runtime_step_table(NUM_STEPS, batch_size=1)
    info = {
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm": version("vllm"),
        "vllm-omni": version("vllm-omni"),
        "transformers": version("transformers"),
        "num_steps": NUM_STEPS,
        "vision_precision": model.paligemma_with_expert.vision_options.precision,
        "lm_precision": model.paligemma_with_expert.lm_precision,
        "batch_cameras": model.paligemma_with_expert.vision_options.batch_cameras,
        "runtime_options": asdict(model.runtime_options),
        "precomputed_schedule_steps": len(schedule) if schedule is not None else 0,
        "projector_weight_dtype": str(
            next(model.paligemma_with_expert.paligemma.model.multi_modal_projector.parameters()).dtype
        ),
        # What the built model runs, not what was asked for.
        "path": (
            "graph"
            if model.cuda_graphs is not None
            else "fused"
            if model.paligemma_with_expert.fused_kernels
            else "eager"
        ),
        "gemm_weight_dtype": str(
            model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        ),
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
        "# π0.5 model-level sweep: batch size × camera views",
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
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="Deploy config whose model_config builds the model (default: the package's vllm_omni/deploy/pi05.yaml)",
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--vision-precision", choices=["fp32", "bf16_autocast", "selective_bf16"], default=None)
    parser.add_argument("--lm-precision", choices=["legacy", "lerobot_bf16"], default=None)
    parser.add_argument("--batch-cameras", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--precompute-adarms", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hoist-denoise-invariants", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefix-kv-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--execution-mode", choices=["eager", "fused", "graph"], default=None)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Run the eager baseline instead of the optimized path (CUDA graphs + fused kernels)",
    )
    parser.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--views", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out", default="pi05_bench")
    args = parser.parse_args()
    if args.enforce_eager and args.execution_mode not in (None, "eager"):
        parser.error("--enforce-eager conflicts with --execution-mode fused/graph")

    if args.deploy_config is None:
        import vllm_omni

        args.deploy_config = str(Path(vllm_omni.__file__).parent / "deploy" / "pi05.yaml")
    checkpoint_dir = resolve_checkpoint(args.model)
    print(f"checkpoint: {checkpoint_dir}")

    model = build_model(
        checkpoint_dir,
        args.deploy_config,
        args.dtype,
        args.enforce_eager,
        args.vision_precision,
        args.batch_cameras,
        lm_precision=args.lm_precision,
        precompute_adarms=args.precompute_adarms,
        hoist_denoise_invariants=args.hoist_denoise_invariants,
        prefix_kv_only=args.prefix_kv_only,
        execution_mode=args.execution_mode,
    )
    device = str(next(model.parameters()).device)
    env = environment(model, device)
    print(json.dumps(env, indent=2))
    print("model built\n")

    rows = []
    for num_views in args.views:
        for batch_size in args.bs:
            label = f"bs={batch_size} views={num_views}"
            try:
                row = time_cell(model, batch_size, num_views, device, args.warmup, args.iters)
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
                if device.startswith("cuda"):
                    torch.accelerator.empty_cache()
            print(f"{label:22s} {row['status']:12s} p50={row['p50_ms']} ms  peak={row['peak_mem_gb']} GB")
            rows.append(row)

    if model.cuda_graphs is not None:
        env["graph_replays_total"] = dict(model.cuda_graphs.num_replays)
    write_reports(rows, env, Path(args.out))
    failed = [r for r in rows if r["status"] != "PASS"]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} cells PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
