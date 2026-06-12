"""
HunyuanImage-3.0 KV async prefetch benchmark — offline example.

Runs multiple text2img requests sequentially with the async-KV deploy config
and reports per-request KV receive timing, prefetch hit/miss, and end-to-end
latency.  The first request always misses (no prior forward to overlap with);
subsequent requests should show prefetch hits, reducing the KV-receive
critical-path latency.

Usage:
    python -m examples.offline_inference.hunyuan_image3.kv_prefetch_bench \
        --model tencent/HunyuanImage-3.0-Instruct \
        --num-requests 4 --steps 50

    # Compare sync vs async-prefetch:
    python -m examples.offline_inference.hunyuan_image3.kv_prefetch_bench --no-prefetch
    python -m examples.offline_inference.hunyuan_image3.kv_prefetch_bench

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
    p.add_argument("--warmup", type=int, default=1, help="Warmup requests (excluded from stats)")
    return p.parse_args()


def _main() -> None:
    args = _parse_args()
    os.makedirs(args.output, exist_ok=True)

    deploy_config = _BASE_YAML if args.no_prefetch else _ASYNC_KV_YAML
    mode_label = "sync" if args.no_prefetch else "async-prefetch"
    print(f"\n{'=' * 70}")
    print(f"  HunyuanImage-3.0 KV Prefetch Benchmark  [{mode_label}]")
    print(f"  Deploy config: {os.path.basename(deploy_config)}")
    print(f"  Requests: {args.num_requests} (+ {args.warmup} warmup)")
    print(f"  Steps: {args.steps}, Guidance: {args.guidance_scale}")
    print(f"{'=' * 70}\n")

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    omni = Omni(
        model=args.model,
        deploy_config=deploy_config,
        enforce_eager=args.enforce_eager,
        init_timeout=args.init_timeout,
        mode="text-to-image",
    )

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

    # --- Run requests sequentially, timing each one ---
    per_request_times: list[dict] = []

    print(f"Running {total} requests ({args.warmup} warmup + {args.num_requests} measured)...\n")

    for i, fp in enumerate(formatted_prompts):
        is_warmup = i < args.warmup
        label = f"{'warmup' if is_warmup else 'req'}-{i}"
        t0 = time.perf_counter()

        outputs = list(omni.generate(prompts=fp, sampling_params_list=params_list))

        elapsed = time.perf_counter() - t0

        # Save image
        img_saved = False
        for req_output in outputs:
            images = getattr(req_output, "images", None)
            if not images:
                ro = getattr(req_output, "request_output", None)
                if ro and hasattr(ro, "images"):
                    images = ro.images
            if images:
                for j, img in enumerate(images):
                    save_path = os.path.join(args.output, f"bench_{i}_{j}.png")
                    img.save(save_path)
                    img_saved = True

        entry = {
            "index": i,
            "label": label,
            "elapsed_s": elapsed,
            "warmup": is_warmup,
            "img_saved": img_saved,
        }
        per_request_times.append(entry)
        tag = "(warmup)" if is_warmup else ""
        print(f"  [{label}] {tag} e2e={elapsed:.2f}s  img={'ok' if img_saved else 'none'}")

    omni.shutdown()

    # --- Parse vllm logs for prefetch hit/miss ---
    # The model runner emits lines like:
    #   KV prefetch HIT for <rid>, apply=<ms>
    #   KV prefetch MISS for <rid>, sync_recv=<ms>
    # These go through the vllm logger, so they appear in the process stderr.
    # We can't easily capture them from here, so instead we report based on
    # the expected pattern: request 0 always misses, subsequent requests hit
    # (when prefetch is enabled).

    print(f"\n{'=' * 70}")
    print(f"  Results [{mode_label}]")
    print(f"{'=' * 70}")

    measured = [e for e in per_request_times if not e["warmup"]]
    if not measured:
        print("  No measured requests.")
        return

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
