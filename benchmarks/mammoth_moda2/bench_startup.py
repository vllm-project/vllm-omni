# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup / loading benchmark for multi-stage omni models (MammothModa2 first).

Measures, from a single fresh process:
  * python import time (torch + vllm_omni)
  * Omni() construction wall time (all stages: weight load, profiling, KV cache, warmup)
  * first request latency (includes JIT / lazy init that only happens on first run)
  * steady-state request latency (repeat N-1 more requests)
  * optional host RSS / device memory samples (--sample-memory, once a second)

Per-stage breakdown (weight loading, model-load memory, engine init) is emitted by the
stage engine-core subprocesses into the log; use parse_startup_log.py on the captured
stdout/stderr to merge them with the JSON written here.

Usage (keep the log: the per-stage breakdown is parsed from it afterwards):
  python benchmarks/mammoth_moda2/bench_startup.py --model /path/MammothModa2-Preview \
      --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
      --height 1024 --width 1024 --seed 42 \
      --extra-body '{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": 50}' \
      --repeat 2 --label nfs-warm --output-json out.json 2>&1 | tee run.log
  python benchmarks/mammoth_moda2/parse_startup_log.py run.log --markdown
"""

import argparse
import json
import os
import platform
import subprocess
import threading
import time

T_PROCESS_START = time.perf_counter()

import torch  # noqa: E402

from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args  # noqa: E402
from vllm_omni.entrypoints.omni import Omni  # noqa: E402
from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402
from vllm_omni.model_extras import (  # noqa: E402
    build_text_to_image_prompt as build_model_text_to_image_prompt,
)
from vllm_omni.model_extras import (  # noqa: E402
    get_extra_body_params,
    get_model_class_name,
    should_init_extra_args_for_non_diffusion_stages,
)
from vllm_omni.platforms import current_omni_platform  # noqa: E402

T_IMPORTS_DONE = time.perf_counter()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--deploy-config", default=None)
    p.add_argument("--prompt", default="A stylish woman riding a motorcycle in NYC, movie poster style")
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra-body", type=json.loads, default=None, help="JSON object forwarded as model extra_body")
    p.add_argument(
        "--repeat", type=int, default=2, help="number of generate() calls; 1st = first-request, rest = steady"
    )
    p.add_argument("--label", default="run", help="free-form label, e.g. nfs-cold / nfs-warm / local-nvme")
    p.add_argument("--output-json", default=None)
    p.add_argument("--save-image", default=None, help="optional path to save the first request's image")
    p.add_argument("--sample-memory", action="store_true", help="sample memory once a second (adds overhead)")
    p.add_argument(
        "--parallel-stage-init",
        action="store_true",
        help="initialize stages that share a GPU concurrently (VllmOmniOrchestratorConfig.parallel_stage_init)",
    )
    args = p.parse_args()
    if args.repeat < 1:
        p.error("--repeat must be at least 1")
    return args


def gpu_info() -> dict:
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version()}
    try:
        import vllm

        info["vllm"] = vllm.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import vllm_omni

        info["vllm_omni"] = getattr(vllm_omni, "__version__", "src")
    except Exception:  # pragma: no cover
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info["gpu"] = out.stdout.strip()
    except Exception:
        info["gpu"] = "n/a"
    return info


class PeakSampler:
    """Poll host RSS of this process tree and GPU memory in use once a second (benchmark-side only)."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.host_rss_peak_gib = 0.0
        self.gpu_used_peak_gib = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            import psutil

            root = psutil.Process()
            rss = root.memory_info().rss + sum(
                c.memory_info().rss for c in root.children(recursive=True) if c.is_running()
            )
            self.host_rss_peak_gib = max(self.host_rss_peak_gib, rss / 2**30)
        except Exception:
            pass
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            used = max(float(x) for x in out.stdout.split() if x.strip())
            self.gpu_used_peak_gib = max(self.gpu_used_peak_gib, used / 1024)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def start(self) -> "PeakSampler":
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=10)
        self._sample()
        return {
            "host_rss_peak_gib": round(self.host_rss_peak_gib, 2),
            "gpu_used_peak_gib": round(self.gpu_used_peak_gib, 2),
        }


def fs_type(path: str) -> str:
    try:
        out = subprocess.run(["stat", "-f", "-c", "%T", path], capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_request(omni: Omni, args: argparse.Namespace, generator: torch.Generator):
    model_class_name = get_model_class_name(omni)
    declared = get_extra_body_params(model_class_name)

    prompt_dict = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt
    prompt_dict = build_model_text_to_image_prompt(
        model_class_name=model_class_name, prompt=prompt_dict, height=args.height, width=args.width
    )

    diffusion_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        generator=generator,
    )
    user_extra = dict(args.extra_body or {})
    if declared:
        apply_declared_extra_args(diffusion_params, declared, user_extra)
    else:
        diffusion_params.extra_args.update({k: v for k, v in user_extra.items() if v is not None})

    init_non_diffusion = should_init_extra_args_for_non_diffusion_stages(model_class_name)
    sampling_params_list = [clone_sampling_params(p) for p in (omni.default_sampling_params_list or [])]
    if not sampling_params_list:
        sampling_params_list = [diffusion_params]

    replaced = False
    for idx, params in enumerate(sampling_params_list):
        if isinstance(params, OmniDiffusionSamplingParams):
            sampling_params_list[idx] = diffusion_params
            replaced = True
        elif init_non_diffusion and hasattr(params, "extra_args"):
            params.extra_args = {**(params.extra_args or {}), **(diffusion_params.extra_args or {})}
            if hasattr(params, "seed"):
                params.seed = args.seed
            # MammothModa2 AR stage: one visual token per grid cell + one EOL per row + 1 look-ahead.
            info = prompt_dict.get("additional_information", {})
            if idx == 0 and info.get("omni_task") == ["t2i"]:
                ar_w = int(info.get("ar_width", [0])[0])
                ar_h = int(info.get("ar_height", [0])[0])
                if ar_w > 0 and ar_h > 0:
                    params.max_tokens = ar_h * (ar_w + 1) + 1
    if not replaced and len(sampling_params_list) == 1:
        sampling_params_list = [diffusion_params]
    return prompt_dict, sampling_params_list


def stage_metrics(outputs) -> dict:
    """Pull vllm-omni's per-stage timing out of the request output when present."""
    res = {}
    for out in outputs or []:
        metrics = getattr(out, "metrics", None)
        if isinstance(metrics, dict) and "stage_metrics" in metrics:
            for sid, m in metrics["stage_metrics"].items():
                if not isinstance(m, dict):
                    continue
                res[str(sid)] = {
                    k: m.get(k)
                    for k in ("stage_gen_time_ms", "num_tokens_in", "num_tokens_out", "vllm_ttft_ms", "vllm_tpot_ms")
                    if k in m
                }
    return res


def main() -> None:
    args = parse_args()
    result = {
        "label": args.label,
        "model": args.model,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "sample_memory": args.sample_memory,
        "model_fs": fs_type(args.model),
        "deploy_config": args.deploy_config,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "extra_body": args.extra_body,
        "parallel_stage_init": args.parallel_stage_init,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "env": gpu_info(),
        "phases_s": {"imports": round(T_IMPORTS_DONE - T_PROCESS_START, 3)},
        "requests": [],
    }

    omni_kwargs = {"model": args.model, "mode": "text-to-image", "log_stats": True}
    if args.deploy_config:
        omni_kwargs["deploy_config"] = args.deploy_config
    if args.parallel_stage_init:
        omni_kwargs["parallel_stage_init"] = True

    sampler = PeakSampler().start() if args.sample_memory else None
    t0 = time.perf_counter()
    omni = None
    try:
        omni = Omni(**omni_kwargs)
        t1 = time.perf_counter()
        result["peak_mem_startup"] = sampler.stop() if sampler else {}
        sampler = None
        result["phases_s"]["engine_init"] = round(t1 - t0, 3)
        result["phases_s"]["process_to_engine_ready"] = round(t1 - T_PROCESS_START, 3)
        print(
            f"BENCH engine_init={t1 - t0:.2f}s process_to_engine_ready={t1 - T_PROCESS_START:.2f}s "
            f"peak_mem_startup={result['peak_mem_startup']}",
            flush=True,
        )
        sampler = PeakSampler().start() if args.sample_memory else None

        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
        outputs = None
        first_outputs = None
        first_image_elapsed = 0.0
        for i in range(args.repeat):
            prompt_dict, sampling_params_list = build_request(omni, args, generator)
            tr0 = time.perf_counter()
            outputs = omni.generate(prompt_dict, sampling_params_list=sampling_params_list)
            tr1 = time.perf_counter()
            if first_outputs is None:
                first_outputs = outputs
                first_image_elapsed = tr1 - T_PROCESS_START
            rec = {
                "index": i,
                "kind": "first" if i == 0 else "steady",
                "latency_s": round(tr1 - tr0, 3),
                "stages": stage_metrics(outputs),
            }
            result["requests"].append(rec)
            print(f"BENCH request[{i}] {rec['kind']} latency={tr1 - tr0:.2f}s stages={rec['stages']}", flush=True)

        result["peak_mem_requests"] = sampler.stop() if sampler else {}
        sampler = None
        steady = [r["latency_s"] for r in result["requests"] if r["kind"] == "steady"]
        result["summary_s"] = {
            "imports": result["phases_s"]["imports"],
            "engine_init": result["phases_s"]["engine_init"],
            "first_request": result["requests"][0]["latency_s"],
            "steady_request_avg": round(sum(steady) / len(steady), 3) if steady else None,
            "time_to_first_image_from_process_start": round(first_image_elapsed, 3),
        }

        for key in ("host_rss_peak_gib", "gpu_used_peak_gib"):
            result["summary_s"][key] = (
                max(result["peak_mem_startup"][key], result["peak_mem_requests"][key]) if args.sample_memory else None
            )

        from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs

        images = extract_images_from_outputs(first_outputs)
        if not images:
            raise RuntimeError("First request returned no image")
        if args.save_image:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_image)), exist_ok=True)
            images[0].save(args.save_image)

        print("BENCH_JSON " + json.dumps(result), flush=True)
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"BENCH wrote {args.output_json}", flush=True)
    finally:
        if sampler:
            sampler.stop()
        if omni is not None:
            omni.close()


if __name__ == "__main__":
    main()
