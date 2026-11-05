# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 BS=1 latency of the layers between the model and the OpenPI client.

Workload: ``lerobot/pi05_base``, batch size 1, action horizon 50, 10 flow
steps, noise seed 42, 200-token padded language input, 1/2/3 camera views (the
observation ``e2e_latency_pi05.py`` sends, fixed per view count). The path is
the deploy default (CUDA graphs over the fused kernels) unless
``--enforce-eager`` is passed.

Modes
-----
``standalone``
    One clean process that only builds ``Pi05Pipeline``: no engine, no
    orchestrator threads. Times ``sample_actions`` (model-only) and
    ``Pi05Pipeline.forward`` (in-process pipeline).
``engine``
    ``Omni.generate`` with the ``uni`` or ``mp`` diffusion executor. With
    ``--trace-dir`` the per-request latency trace is enabled, so the model and
    pipeline spans of the very same engine requests are recorded as well.

The OpenPI websocket layer is measured by ``e2e_latency_pi05.py`` against
``vllm serve``; ``report_engine_overhead_pi05.py`` turns the results into tables.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

PROMPT = "pick up the red block and place it in the bin"
SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("standalone", "engine"), required=True)
    parser.add_argument("--backend", choices=("uni", "mp"), default="mp", help="Diffusion executor for --mode engine.")
    parser.add_argument("--model", default="lerobot/pi05_base")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/pi05.yaml")
    parser.add_argument("--views", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--enforce-eager", action="store_true", help="Run the eager baseline path.")
    parser.add_argument("--trace-dir", help="Enable the latency trace and write it here (engine mode).")
    parser.add_argument("--out", required=True, help="Result JSON.")
    parser.add_argument("--save-actions", help="Optional .npz with one action chunk per view count and layer.")
    return parser.parse_args()


ARGS = _parse_args()
if ARGS.trace_dir:
    # Must be set before vllm_omni is imported and inherited by worker processes.
    os.environ["VLLM_OMNI_LATENCY_TRACE_DIR"] = ARGS.trace_dir

import numpy as np  # noqa: E402
import torch  # noqa: E402
from e2e_latency_pi05 import make_obs  # noqa: E402


def robot_obs(views: int) -> dict:
    """The observation the server hands the pipeline for one client request."""
    return make_obs(views=views, prompt=PROMPT, session_id="bench", rng=np.random.default_rng(0))


def summary(samples_ms: list[float]) -> dict:
    values = sorted(samples_ms)
    n = len(values)

    def quantile(q: float) -> float:
        position = q * (n - 1)
        low = int(position)
        high = min(low + 1, n - 1)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        "n": n,
        "p50_ms": quantile(0.5),
        "p95_ms": quantile(0.95),
        "min_ms": values[0],
        "max_ms": values[-1],
        "mean_ms": sum(values) / n,
    }


def environment() -> dict:
    import transformers
    import vllm

    import vllm_omni

    info = {
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": vllm.__version__,
        "vllm_omni": getattr(vllm_omni, "__version__", "unknown"),
        "vllm_omni_path": os.path.dirname(os.path.abspath(vllm_omni.__file__)),
        "transformers": transformers.__version__,
        "torch_threads": torch.get_num_threads(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "ATTENTION_BACKEND",
                "DIFFUSION_ATTENTION_BACKEND",
                "VLLM_DISABLE_COMPILE_CACHE",
                "VLLM_OMNI_LATENCY_TRACE_DIR",
                "VLLM_OMNI_EVENT_DRIVEN_ORCH",
                "OMP_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return info


def timed(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(iters):
        started = time.perf_counter_ns()
        fn()
        torch.accelerator.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return samples


def make_layer_fns(pipeline, views: int, steps: int, captured: dict):
    """Model-only and in-process pipeline callables for one view count."""
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    obs = robot_obs(views)
    device = pipeline._device
    with torch.inference_mode():
        images, image_masks, lang_tokens, lang_masks = pipeline.processor.build_model_inputs(obs)

    def model_only() -> None:
        with torch.inference_mode():
            actions = pipeline.model.sample_actions(
                images=images,
                image_masks=image_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                num_steps=steps,
                generator=torch.Generator(device=device).manual_seed(SEED),
            )
        captured["model"] = actions

    def in_process_pipeline() -> None:
        params = OmniDiffusionSamplingParams(
            num_inference_steps=steps,
            generator=torch.Generator(device=device).manual_seed(SEED),
            extra_args={"robot_obs": obs},
        )
        request = OmniDiffusionRequest(prompt=PROMPT, sampling_params=params, request_id="bench-pipeline")
        captured["pipeline"] = pipeline.forward(request)

    return model_only, in_process_pipeline


def measure_layers(pipeline, args: argparse.Namespace, saved: dict, prefix: str) -> dict:
    results = {}
    for views in args.views:
        captured: dict = {}
        model_only, in_process_pipeline = make_layer_fns(pipeline, views, args.steps, captured)
        torch.accelerator.reset_peak_memory_stats()
        layers = {
            "model_only_ms": summary(timed(model_only, args.iters, args.warmup)),
            "pipeline_ms": summary(timed(in_process_pipeline, args.iters, args.warmup)),
        }
        pipeline_actions = np.asarray(captured["pipeline"].output["actions"], dtype=np.float32)
        model_actions = pipeline.processor.build_model_outputs(captured["model"], robot_obs(views))
        saved[f"{prefix}_pipeline_views{views}"] = pipeline_actions
        saved[f"{prefix}_model_views{views}"] = np.asarray(model_actions, dtype=np.float32)
        results[str(views)] = {
            "views": views,
            "layers": layers,
            "model_equals_pipeline": bool(np.array_equal(pipeline_actions, model_actions)),
            "peak_memory_allocated_mib": torch.accelerator.max_memory_allocated() / 2**20,
            "peak_memory_reserved_mib": torch.accelerator.max_memory_reserved() / 2**20,
        }
        print(
            f"[{prefix}] views={views} " + json.dumps({k: round(v["p50_ms"], 3) for k, v in layers.items()}), flush=True
        )
    return results


def run_standalone(args: argparse.Namespace) -> dict:
    import yaml

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    with open(args.deploy_config, encoding="utf-8") as handle:
        deploy = yaml.safe_load(handle)
    stage = deploy["stages"][0]
    load_started = time.perf_counter()
    od_config = OmniDiffusionConfig.from_kwargs(
        model=args.model,
        dtype=deploy.get("dtype", "float32"),
        model_class_name=stage["model_class_name"],
        model_config=stage["model_config"],
        enforce_eager=args.enforce_eager,
    )
    # Under the serving dtype as torch's default, as the server's loader builds
    # (and captures) the pipeline.
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_default_torch_dtype(od_config.dtype):
        pipeline = Pi05Pipeline(od_config=od_config)
    load_s = time.perf_counter() - load_started
    saved: dict = {}
    result = {
        "mode": "standalone",
        "enforce_eager": args.enforce_eager,
        "dtype": str(pipeline._torch_dtype),
        "load_s": load_s,
        "views": measure_layers(pipeline, args, saved, "standalone"),
    }
    return result, saved


def run_engine(args: argparse.Namespace) -> dict:
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.utils import latency_trace

    kwargs = {"model": args.model, "deploy_config": args.deploy_config, "log_stats": True}
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.backend == "uni":
        kwargs["distributed_executor_backend"] = "uni"
    load_started = time.perf_counter()
    omni = Omni(**kwargs)
    load_s = time.perf_counter() - load_started
    saved: dict = {}
    try:
        result: dict = {
            "mode": "engine",
            "backend": args.backend,
            "enforce_eager": args.enforce_eager,
            "load_s": load_s,
            "views": {},
        }
        raw_rows = []
        for views in args.views:
            obs = robot_obs(views)

            def engine_call():
                params = OmniDiffusionSamplingParams(
                    num_inference_steps=args.steps, seed=SEED, extra_args={"robot_obs": obs}
                )
                return omni.generate(PROMPT, sampling_params_list=[params], use_tqdm=False)

            cold_started = time.perf_counter_ns()
            outputs = engine_call()
            first_ms = (time.perf_counter_ns() - cold_started) / 1e6
            for _ in range(args.warmup - 1):
                outputs = engine_call()
            samples = []
            for index in range(args.iters):
                started = time.perf_counter_ns()
                outputs = engine_call()
                ended = time.perf_counter_ns()
                samples.append((ended - started) / 1e6)
                final = outputs[-1]
                raw_rows.append(
                    {
                        "views": views,
                        "iteration": index,
                        "request_id": getattr(final, "request_id", None),
                        "call_start_ns": started,
                        "call_end_ns": ended,
                        "latency_ms": samples[-1],
                    }
                )
            final = outputs[-1]
            actions = np.asarray((final.multimodal_output or {}).get("actions"), dtype=np.float32)
            saved[f"engine_{args.backend}_views{views}"] = actions
            result["views"][str(views)] = {
                "views": views,
                "layers": {f"engine_{args.backend}_ms": summary(samples)},
                "first_request_ms": first_ms,
                "peak_memory_mb": getattr(final, "peak_memory_mb", None),
            }
            print(
                f"[engine/{args.backend}] views={views} p50={summary(samples)['p50_ms']:.3f} ms "
                f"first={first_ms:.1f} ms",
                flush=True,
            )
        result["requests"] = raw_rows
        latency_trace.flush()
    finally:
        omni.close()
    return result, saved


def main() -> None:
    args = ARGS
    result, saved = run_standalone(args) if args.mode == "standalone" else run_engine(args)
    result["workload"] = {
        "batch_size": 1,
        "action_horizon": 50,
        "steps": args.steps,
        "seed": SEED,
        "views": args.views,
        "warmup": args.warmup,
        "iters": args.iters,
        "prompt": PROMPT,
    }
    result["environment"] = environment()
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    if args.save_actions:
        np.savez(args.save_actions, **saved)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
