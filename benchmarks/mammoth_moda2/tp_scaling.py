# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Latency, memory and capacity of the MammothModa2 AR stage across TP degrees.

Produces the table recorded in ``recipes/MammothModa2/MammothModa2.md``. One
engine per degree; only stage 0's ``tensor_parallel_size`` and device list
change between columns, so stage 1 stays fixed and acts as a control -- a column
where it moves is measuring something other than AR sharding.

Reporting follows the repository's perf-verification guidance:

* the cold first request is reported on its own row and never folded into the
  warm statistics, because they answer different questions;
* P50 and P100 rather than p95, since at these repetition counts a p95 is the
  second-largest sample wearing a percentile's name;
* sanity rows (generated tokens, stage-output rows, inference steps) travel with
  the timings, so a column that silently did less work cannot read as faster;
* a stage-handoff row, because AR->DiT transfer and scheduling gaps sit between
  the per-stage numbers and are invisible in their deltas.

Device memory is sampled through NVML rather than read from this process's
allocator counters: the engine runs in worker subprocesses, so the benchmark
process never allocates and its own counters stay at zero.

Example::

    python benchmarks/mammoth_moda2/tp_scaling.py \
        --model bytedance-research/MammothModa2-Preview \
        --tp 1,2,4 --height 1024 --width 1024 --steps 50 \
        --warmup 1 --measure 8 --output-dir /tmp/mammoth_tp
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import statistics
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_extras.mammothmodal2_preview import build_text_to_image_prompt

DEFAULT_DEPLOY_CONFIG = Path(__file__).resolve().parents[2] / "vllm_omni" / "deploy" / "mammoth_moda2.yaml"
DEFAULT_PROMPT = "A cat sitting on a laptop keyboard"


class DeviceMemorySampler:
    """Poll whole-device memory while a request runs.

    The engine executes in worker subprocesses, so the benchmark process never
    allocates and its own allocator counters read zero -- the first smoke run
    reported 0.00 GiB per device for exactly that reason. NVML reports the
    device, not the process, which also matches how the recipe's existing figure
    ("largest one second whole device memory sample") was taken.
    """

    def __init__(self, interval_s: float = 0.25):
        import pynvml

        self._nvml = pynvml
        self._nvml.nvmlInit()
        self._handles = [self._nvml.nvmlDeviceGetHandleByIndex(i) for i in range(self._nvml.nvmlDeviceGetCount())]
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_gib = [0.0] * len(self._handles)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            for i, h in enumerate(self._handles):
                used = self._nvml.nvmlDeviceGetMemoryInfo(h).used / 2**30
                if used > self.peak_gib[i]:
                    self.peak_gib[i] = used

    def reset(self) -> None:
        self.peak_gib = [0.0] * len(self._handles)

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def deploy_config(base_config: Path, tp_size: int, seed: int) -> str:
    """Write a deploy yaml with stage 0 sharded and stage 1 held fixed.

    Stage 1 stays at TP=1 on device 0 in every column so it acts as a control.
    ``max_num_seqs`` is pinned to 1 on both stages: this measures single-request
    latency, and leaving stage 0 at the shipped 100 would let the scheduler batch
    and turn the comparison into something else. That also means the numbers here
    say nothing about throughput under concurrency.
    """
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    stages = config["stages"]
    stages[0].update(
        {
            "tensor_parallel_size": tp_size,
            "devices": ",".join(str(i) for i in range(tp_size)),
            "seed": seed,
            "max_num_seqs": 1,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
        }
    )
    stages[1].update(
        {
            "tensor_parallel_size": 1,
            "devices": "0",
            "seed": seed,
            "max_num_seqs": 1,
            "enforce_eager": True,
        }
    )
    fd, path = tempfile.mkstemp(prefix=f"{base_config.stem}_tp{tp_size}_", suffix=".yaml")
    atexit.register(Path(path).unlink, missing_ok=True)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=None, sort_keys=False, allow_unicode=True, indent=2)
    return path


def build_prompt(text: str, height: int, width: int, steps: int, guidance: float) -> dict:
    prompt = build_text_to_image_prompt(text, None, height=height, width=width)
    info = prompt["additional_information"]
    info.update(
        {
            "num_inference_steps": [steps],
            "text_guidance_scale": [guidance],
            "cfg_range": [0.0, 1.0],
            "visual_ids": [151655, 151656, 151652, 151653],
        }
    )
    return prompt


def stage_timings(sink: list[dict]) -> None:
    """Record when the AR stage handed off, so the gap to completion is visible."""
    import vllm_omni.model_executor.stage_input_processors.mammoth_moda2 as proc

    original = proc.ar2dit

    def _timed(source_outputs, prompts=None, _requires_multimodal_data=False):
        result = original(source_outputs, prompts, _requires_multimodal_data)
        for dit_prompt in result:
            info = dit_prompt["additional_information"]
            sink.append(
                {
                    "handoff_monotonic": time.monotonic(),
                    "generated_tokens": len(info["full_token_ids"]) - int(info["answer_start_index"][0]),
                    "hidden_rows": int(info["full_hidden_states"].shape[0]),
                }
            )
        return result

    proc.ar2dit = _timed
    return original


def run_column(tp_size: int, args) -> dict:
    """One engine, one warmup-and-measure population, torn down before the next."""
    import vllm_omni.model_executor.stage_input_processors.mammoth_moda2 as proc

    prompt = build_prompt(args.prompt, args.height, args.width, args.steps, args.guidance)
    ar_grid = (args.height // 16) * ((args.width // 16) + 1)
    ar_sampling = SamplingParams(temperature=0.0, top_k=1, seed=args.seed, max_tokens=ar_grid + 1, detokenize=False)
    dit_sampling = SamplingParams(
        temperature=0.0,
        seed=args.seed,
        max_tokens=1,
        detokenize=False,
        extra_args={
            "num_inference_steps": args.steps,
            "text_guidance_scale": args.guidance,
            "cfg_range": [0.0, 1.0],
        },
    )

    handoffs: list[dict] = []
    original_ar2dit = stage_timings(handoffs)
    samples: list[dict] = []
    startup_s = None
    sampler = DeviceMemorySampler()
    try:
        t0 = time.monotonic()
        with sampler:
            omni = Omni(
                model=args.model,
                deploy_config=deploy_config(Path(args.deploy_config), tp_size, args.seed),
            )
            startup_s = time.monotonic() - t0
            for i in range(args.warmup + args.measure):
                sampler.reset()
                before = len(handoffs)
                t_req = time.monotonic()
                outputs = list(omni.generate([prompt], [ar_sampling, dit_sampling]))
                t_end = time.monotonic()
                new = handoffs[before:]
                handoff = new[0] if new else None
                image = None
                for out in outputs:
                    for ro in out if isinstance(out, list) else [out]:
                        mm = getattr(ro, "multimodal_output", None)
                        if isinstance(mm, Mapping) and "image" in mm:
                            img = mm["image"]
                            image = img[0] if hasattr(img, "ndim") and img.ndim == 4 else img
                samples.append(
                    {
                        "index": i,
                        "phase": "warmup" if i < args.warmup else "measured",
                        "e2e_s": t_end - t_req,
                        "ar_s": (handoff["handoff_monotonic"] - t_req) if handoff else None,
                        "post_handoff_s": (t_end - handoff["handoff_monotonic"]) if handoff else None,
                        "generated_tokens": handoff["generated_tokens"] if handoff else None,
                        "hidden_rows": handoff["hidden_rows"] if handoff else None,
                        "image_shape": list(image.shape) if image is not None else None,
                        "peak_device_gib": list(sampler.peak_gib),
                    }
                )
                print(
                    f"  tp{tp_size} req {i} [{samples[-1]['phase']}] "
                    f"e2e={samples[-1]['e2e_s']:.2f}s ar={samples[-1]['ar_s'] or float('nan'):.2f}s "
                    f"tokens={samples[-1]['generated_tokens']}",
                    flush=True,
                )
    finally:
        proc.ar2dit = original_ar2dit
        del omni
        gc.collect()
        torch.accelerator.empty_cache()

    measured = [s for s in samples if s["phase"] == "measured"]

    def agg(key):
        vals = [s[key] for s in measured if s[key] is not None]
        return {"p50": statistics.median(vals), "p100": max(vals), "min": min(vals), "n": len(vals)} if vals else None

    return {
        "tp_size": tp_size,
        "startup_s": startup_s,
        "cold_first_request_s": samples[0]["e2e_s"] if samples else None,
        "e2e": agg("e2e_s"),
        "ar": agg("ar_s"),
        "post_handoff": agg("post_handoff_s"),
        "peak_device_gib_per_device": [
            max(s["peak_device_gib"][d] for s in measured) for d in range(torch.accelerator.device_count())
        ]
        if measured
        else None,
        "sanity": {
            "generated_tokens": sorted({s["generated_tokens"] for s in measured}),
            "hidden_rows": sorted({s["hidden_rows"] for s in measured}),
            "image_shape": [s["image_shape"] for s in measured[:1]],
            "inference_steps": args.steps,
        },
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bytedance-research/MammothModa2-Preview")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--measure", type=int, default=10)
    ap.add_argument("--tp", default="1,2,4", help="comma-separated TP degrees for stage 0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--deploy-config", default=str(DEFAULT_DEPLOY_CONFIG))
    ap.add_argument("--output-dir", dest="out", default="/tmp/mammoth_tp_scaling")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import subprocess

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()

    results = {}
    for tp in (int(x) for x in args.tp.split(",")):
        print(
            f"\n=== TP={tp} : {args.height}x{args.width}, {args.steps} steps, "
            f"warmup {args.warmup} + measure {args.measure} ===",
            flush=True,
        )
        results[tp] = run_column(tp, args)
        (out / f"tp{tp}.json").write_text(json.dumps(results[tp], indent=2))

    report = {
        "commit": commit,
        "worktree_dirty": bool(dirty),
        "dirty_files": dirty.splitlines(),
        "config": vars(args),
        "gpu": torch.accelerator.current_accelerator().type,
        "gpu_count": torch.accelerator.device_count(),
        "torch": torch.__version__,
        "columns": results,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 78)
    print(f"commit {commit}{'  (WORKTREE DIRTY)' if dirty else ''}")
    print(f"{args.height}x{args.width}, {args.steps} steps, guidance {args.guidance}, seed {args.seed}")
    print(
        f"warmup {args.warmup}, measured {args.measure}, "
        f"{torch.accelerator.device_count()} x {torch.accelerator.current_accelerator().type}"
    )
    print("=" * 78)
    hdr = f"{'row':<34}" + "".join(f"{'TP=' + str(t):>16}" for t in results)
    print(hdr)

    def row(label, fn):
        print(f"{label:<34}" + "".join(f"{fn(r):>16}" for r in results.values()))

    row("engine startup (s)", lambda r: f"{r['startup_s']:.1f}")
    row("cold first request (s)", lambda r: f"{r['cold_first_request_s']:.2f}")
    row("e2e P50 (s)", lambda r: f"{r['e2e']['p50']:.2f}")
    row("e2e P100 (s)", lambda r: f"{r['e2e']['p100']:.2f}")
    row("AR stage P50 (s)", lambda r: f"{r['ar']['p50']:.2f}" if r["ar"] else "n/a")
    row("post-handoff P50 (s)", lambda r: f"{r['post_handoff']['p50']:.2f}" if r["post_handoff"] else "n/a")
    row("peak device mem dev0 (GiB)", lambda r: f"{r['peak_device_gib_per_device'][0]:.2f}")
    row(
        "peak device mem dev1 (GiB)",
        lambda r: f"{r['peak_device_gib_per_device'][1]:.2f}" if len(r["peak_device_gib_per_device"]) > 1 else "-",
    )
    row("GPU-seconds per image", lambda r: f"{r['e2e']['p50'] * r['tp_size']:.1f}")
    print("-" * 78)
    row("[sanity] generated tokens", lambda r: str(r["sanity"]["generated_tokens"]))
    row("[sanity] hidden rows", lambda r: str(r["sanity"]["hidden_rows"]))
    row("[sanity] inference steps", lambda r: str(r["sanity"]["inference_steps"]))
    if 1 in results and 2 in results:
        s = results[1]["e2e"]["p50"] / results[2]["e2e"]["p50"]
        a = results[1]["ar"]["p50"] / results[2]["ar"]["p50"] if results[1]["ar"] and results[2]["ar"] else float("nan")
        print("-" * 78)
        print(f"e2e speedup TP1->TP2      : {s:.3f}x  (scaling efficiency {s / 2:.1%})")
        print(f"AR-stage speedup TP1->TP2 : {a:.3f}x  (scaling efficiency {a / 2:.1%})")
        print("post-handoff is the DiT plus stage transfer; stage 1 is TP=1 on device 0 in")
        print("both columns, so it pins comparability -- a change there is not AR sharding.")
    print(f"\nwrote {out}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
