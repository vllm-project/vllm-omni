"""
HunyuanImage-3.0 KV async prefetch benchmark — offline example.

Runs multiple text2img requests with the async-KV deploy config and reports
per-request KV receive timing, prefetch hit/miss, and end-to-end latency.

Two submission modes:
  --mode sequential  (default): One request at a time.  Every request shows
      MISS because the next request hasn't been submitted when the DiT
      scheduler builds the prefetch stub.  Use this as the sync baseline.
  --mode pipeline:  All requests are submitted to the orchestrator at once.
      While the DiT denoises request N (~25 s), the AR stage has already
      finished request N+1 and enqueued it to the DiT waiting queue.  The
      scheduler sees the next request and populates prefetch_stub, so
      start_load_kv() fires — subsequent requests should show HIT.

Usage:
    python -m examples.offline_inference.hunyuan_image3.kv_prefetch_bench \
        --model tencent/HunyuanImage-3.0-Instruct \
        --num-requests 4 --steps 50

    # Compare sequential (always MISS) vs pipeline (prefetch HIT):
    python -m examples.offline_inference.hunyuan_image3.kv_prefetch_bench --mode sequential
    python -m examples.offline_inference.hunyuan_image3.kv_prefetch_bench --mode pipeline

    # Sync baseline (no async KV at all):
    python -m examples.offline_inference.hunyuan_image3.kv_prefetch_bench --no-prefetch

Requirements:
    - GPUs with CUDA (2 for AR + 2 for DiT by default)
    - The async-KV deploy YAML at vllm_omni/deploy/hunyuan_image_3_moe_async_kv.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ASYNC_KV_YAML = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image_3_moe_async_kv.yaml")
_BASE_YAML = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image_3_moe.yaml")


def _save_images(outputs: list, output_dir: str, request_idx: int) -> bool:
    """Save images from generation outputs. Returns True if any image was saved."""
    img_saved = False
    for req_output in outputs:
        images = getattr(req_output, "images", None)
        if not images:
            ro = getattr(req_output, "request_output", None)
            if ro and hasattr(ro, "images"):
                images = ro.images
        if images:
            for j, img in enumerate(images):
                save_path = os.path.join(output_dir, f"bench_{request_idx}_{j}.png")
                img.save(save_path)
                img_saved = True
    return img_saved


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KV async prefetch benchmark for HunyuanImage-3.0")
    p.add_argument("--model", default="tencent/HunyuanImage-3.0-Instruct")
    p.add_argument("--num-requests", type=int, default=4, help="Number of sequential text2img requests")
    p.add_argument("--steps", type=int, default=50, help="Denoising steps per request")
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--output", type=str, default=".", help="Directory to save images")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--init-timeout", type=int, default=300)
    p.add_argument("--no-prefetch", action="store_true", help="Disable async prefetch (use sync KV receive)")
    p.add_argument(
        "--mode",
        choices=["sequential", "pipeline"],
        default="sequential",
        help=(
            "Submission mode. 'sequential' submits one request at a time (always MISS). "
            "'pipeline' submits all requests at once so the DiT scheduler can prefetch "
            "the next request's KV while denoising the current one (HIT on req 1+)."
        ),
    )
    p.add_argument("--warmup", type=int, default=1, help="Warmup requests (excluded from stats)")
    p.add_argument("--quantization", type=str, default=None, help="Quantization method (e.g. fp8)")
    return p.parse_args()


def _main() -> None:
    args = _parse_args()
    os.makedirs(args.output, exist_ok=True)

    deploy_config = _BASE_YAML if args.no_prefetch else _ASYNC_KV_YAML
    mode_label = "sync" if args.no_prefetch else f"async-prefetch/{args.mode}"
    print(f"\n{'=' * 70}")
    print(f"  HunyuanImage-3.0 KV Prefetch Benchmark  [{mode_label}]")
    print(f"  Deploy config: {os.path.basename(deploy_config)}")
    print(f"  Mode: {args.mode}")
    print(f"  Requests: {args.num_requests} (+ {args.warmup} warmup)")
    print(f"  Steps: {args.steps}, Guidance: {args.guidance_scale}")
    print(f"{'=' * 70}\n")

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    omni_kwargs = dict(
        model=args.model,
        deploy_config=deploy_config,
        enforce_eager=args.enforce_eager,
        init_timeout=args.init_timeout,
        mode="text-to-image",
    )
    if args.quantization:
        omni_kwargs["quantization"] = args.quantization
    omni = Omni(**omni_kwargs)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        build_prompt_tokens,
        resolve_stop_token_ids,
        resolve_sys_type,
    )

    prompts = [f"A cute cat number {i}" for i in range(args.num_requests + args.warmup)]
    total = len(prompts)

    params_list = list(omni.default_sampling_params_list)
    task, bot_task = "t2i", "think"
    sys_type = resolve_sys_type(bot_task)
    stop_token_ids = resolve_stop_token_ids(task=task, bot_task=bot_task, tokenizer=tokenizer)

    for sp in params_list:
        if isinstance(sp, OmniDiffusionSamplingParams):
            sp.num_inference_steps = args.steps
            sp.guidance_scale = args.guidance_scale
            sp.guidance_scale_provided = True
            if args.seed is not None:
                sp.seed = args.seed
            if args.height is not None:
                sp.height = args.height
            if args.width is not None:
                sp.width = args.width
        elif hasattr(sp, "stop_token_ids"):
            sp.stop_token_ids = stop_token_ids

    formatted_prompts = []
    for prompt_text in prompts:
        result = build_prompt_tokens(prompt_text, tokenizer, task=task, bot_task=bot_task)
        formatted_prompts.append(
            {
                "prompt_token_ids": result.token_ids,
                "prompt": prompt_text,
                "use_system_prompt": sys_type,
                "modalities": ["image"],
            }
        )

    # --- Run requests ---
    per_request_times: list[dict] = []

    if args.mode == "sequential":
        # Sequential: one request at a time — prefetch stub is always None
        # because the next request hasn't been submitted yet.
        print(f"Running {total} requests sequentially ({args.warmup} warmup + {args.num_requests} measured)...\n")

        for i, fp in enumerate(formatted_prompts):
            is_warmup = i < args.warmup
            label = f"{'warmup' if is_warmup else 'req'}-{i}"
            t0 = time.perf_counter()

            outputs = list(omni.generate(prompts=fp, sampling_params_list=params_list))

            elapsed = time.perf_counter() - t0

            img_saved = _save_images(outputs, args.output, i)
            per_request_times.append(
                {"index": i, "label": label, "elapsed_s": elapsed, "warmup": is_warmup, "img_saved": img_saved}
            )
            tag = "(warmup)" if is_warmup else ""
            print(f"  [{label}] {tag} e2e={elapsed:.2f}s  img={'ok' if img_saved else 'none'}")

    else:
        # Pipeline: submit ALL requests at once — the orchestrator enqueues
        # them all, and while DiT denoises request N, the AR stage has already
        # finished request N+1 and queued it in the DiT scheduler's waiting
        # list.  The scheduler then builds a prefetch_stub for N+1 so
        # start_load_kv() fires during request N's step 0.
        print(f"Running {total} requests in pipeline mode ({args.warmup} warmup + {args.num_requests} measured)...\n")
        print(f"  Submitting all {total} requests to the orchestrator at once...")
        t_wall_start = time.perf_counter()

        # omni.generate accepts a list of prompts — it submits them all
        # before polling outputs, creating pipeline overlap.
        all_outputs = list(omni.generate(prompts=formatted_prompts, sampling_params_list=params_list, use_tqdm=True))
        t_wall_end = time.perf_counter()

        # Map outputs back to request indices.  Outputs arrive in
        # completion order, but each OmniRequestOutput carries request_id.
        for idx, req_output in enumerate(all_outputs):
            i = idx  # sequential output order matches input order
            is_warmup = i < args.warmup
            label = f"{'warmup' if is_warmup else 'req'}-{i}"
            img_saved = _save_images([req_output], args.output, i)
            per_request_times.append(
                {"index": i, "label": label, "elapsed_s": 0.0, "warmup": is_warmup, "img_saved": img_saved}
            )

        print(f"\n  Pipeline wall time: {t_wall_end - t_wall_start:.2f}s for {total} requests")
        print("  Per-request e2e is not individually measurable in pipeline mode.")
        print("  Check process logs for KV prefetch HIT/MISS lines.")

    omni.shutdown()

    # --- Report results ---
    print(f"\n{'=' * 70}")
    print(f"  Results [{mode_label}]")
    print(f"{'=' * 70}")

    measured = [e for e in per_request_times if not e["warmup"]]
    if not measured:
        print("  No measured requests.")
        return

    if args.mode == "pipeline":
        # In pipeline mode we don't have per-request e2e timing (requests
        # overlap), but we report which images were generated and remind the
        # user to check logs for HIT/MISS.
        for e in measured:
            print(f"    {e['label']}: img={'ok' if e['img_saved'] else 'none'}")
        print("\n  In pipeline mode, per-request e2e is not individually measurable.")
        print("  Check the process log output for lines matching 'KV prefetch HIT/MISS'.")
        print("  Expected: request 0 = MISS, requests 1+ = HIT (when prefetch enabled).")
    else:
        e2e_times = [e["elapsed_s"] for e in measured]
        print("\n  Per-request end-to-end latency:")
        for e in measured:
            print(f"    {e['label']}: {e['elapsed_s']:.2f}s")

        avg_e2e = sum(e2e_times) / len(e2e_times)
        first_e2e = e2e_times[0]
        rest_e2e = e2e_times[1:] if len(e2e_times) > 1 else []
        avg_rest = sum(rest_e2e) / len(rest_e2e) if rest_e2e else 0

        print("\n  Summary:")
        print(f"    First request (always sync): {first_e2e:.2f}s")
        if rest_e2e:
            print(f"    Subsequent requests avg:     {avg_rest:.2f}s")
            if not args.no_prefetch and avg_rest > 0:
                delta = first_e2e - avg_rest
                pct = delta / first_e2e * 100 if first_e2e > 0 else 0
                print(f"    Estimated prefetch speedup:  {delta:.2f}s ({pct:+.1f}%)")
                print(f"    (Prefetch overlap saves ~{delta:.1f}s of KV-receive latency per request)")
        print(f"    Overall avg:                 {avg_e2e:.2f}s")
        print(f"    Total measured:              {sum(e2e_times):.2f}s")

    print("\n  Note: For precise per-request KV hit/miss timing, check the")
    print("  process log output for lines matching 'KV prefetch HIT/MISS'.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    _main()
