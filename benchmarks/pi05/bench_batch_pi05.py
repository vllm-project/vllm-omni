#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 stage profile: rounds of B requests, time per stage and GPU activity per stage.

One run measures one chain (an execution mode and a precision policy) at the model level:

    python bench_batch_pi05.py --mode eager --out results/                                    # R0, eager
    python bench_batch_pi05.py --mode graph --out results/                                    # R0, optimized
    python bench_batch_pi05.py --mode graph --vision-precision selective_bf16 --out results/  # OpenPI's vision cast
    python bench_batch_pi05.py --mode graph --vision-precision selective_bf16 --all4 --out results/
    python bench_batch_pi05.py --report results/                # one report over every chain in the directory

A round is one batch of B requests through ``sample_actions``; round ``i`` draws its rows from seed ``i``, so
every chain sees the same pixels, tokens and noise. The three calls ``sample_actions`` makes (prefix embedding,
prefill, every denoising step) are timed in place with a device sync after each, by wrapping the entry points
it looks up: the model's own code runs, not a copy of it. Then each stage is replayed alone for ``--seconds``
with the arguments of a real round, and the unsplit ``sample_actions`` the same way, so that the DCGM samples of
a window belong to one stage (raw 0-1 ratios and watts; those columns stay empty without ``dcgmi``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from bench_pi05 import build_model, resolve_checkpoint

STAGES = ("embed_prefix", "prefill", "denoise")
# DCGM fields: SM activity, SM occupancy, tensor-core, FP32-pipe and DRAM activity, engine active, power.
DCGM_FIELDS = {"smact": 1002, "smocc": 1003, "tenso": 1004, "fp32a": 1007, "drama": 1005, "gract": 1001, "power_w": 155}
WARMUP_SEED_BASE = 1_000_000


def seeded_inputs(model, seed: int, batch: int) -> dict:
    """The ``sample_actions`` arguments of one round: every camera live, a full prompt, fixed noise."""
    device = model.action_in_proj.weight.device
    cameras = model.config.max_cameras
    rng = np.random.default_rng(seed)
    images = rng.random((cameras, batch, 224, 224, 3), dtype=np.float32) * 2.0 - 1.0
    tokens = rng.integers(5, 2000, (batch, model.config.tokenizer_max_length), dtype=np.int32)
    noise = rng.standard_normal((batch, model.action_horizon, model.action_dim), dtype=np.float32)
    inputs = {
        "images": [torch.from_numpy(images[cam]).permute(0, 3, 1, 2).contiguous().to(device) for cam in range(cameras)],
        "image_masks": [torch.ones(batch, dtype=torch.bool, device=device) for _ in range(cameras)],
        "lang_tokens": torch.from_numpy(tokens).long().to(device),
        "lang_masks": torch.ones(tokens.shape, dtype=torch.bool, device=device),
        "noise": torch.from_numpy(noise).to(device),
    }
    torch.accelerator.synchronize()
    return inputs


class Stages:
    """The three calls ``sample_actions`` makes: timed in place, and replayable alone.

    Wraps the entry points ``sample_actions`` looks up (the CUDA graphs' when they are installed, the model's
    otherwise). Only the outermost wrapper counts, since eager denoising calls the backbone forward that
    prefill wraps.
    """

    def __init__(self, model):
        graphs = model.cuda_graphs
        if graphs is not None:
            owners = [(graphs, "embed_prefix"), (graphs, "prefix_forward"), (graphs, "denoise_step")]
        else:
            owners = [(model, "embed_prefix"), (model.paligemma_with_expert, "forward"), (model, "denoise_step")]
        self.timed = True
        self.ms = dict.fromkeys(STAGES, 0.0)
        self.recorded: dict[str, list] | None = None
        self._inside = False
        for stage, (owner, name) in zip(STAGES, owners):
            setattr(owner, name, self._wrap(stage, getattr(owner, name)))

    def _wrap(self, stage: str, fn):
        def call(*args, **kwargs):
            if self._inside or not self.timed:
                return fn(*args, **kwargs)
            self._inside = True
            start = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
                torch.accelerator.synchronize()
            finally:
                self._inside = False
            self.ms[stage] += (time.perf_counter() - start) * 1000.0
            if self.recorded is not None:
                self.recorded[stage].append((fn, args, kwargs))
            return out

        return call

    def round(self, model, inputs: dict) -> dict:
        self.ms = dict.fromkeys(STAGES, 0.0)
        start = time.perf_counter()
        model.sample_actions(**inputs)
        torch.accelerator.synchronize()
        return {**self.ms, "total": (time.perf_counter() - start) * 1000.0}

    def replay(self, stage: str) -> None:
        for fn, args, kwargs in self.recorded[stage]:
            fn(*args, **kwargs)
        torch.accelerator.synchronize()


class Dcgm:
    """Background ``dcgmi dmon`` at 100 ms; ``window`` averages the samples inside one timed loop."""

    def __init__(self, gpu: str):
        self.samples: list[tuple] = []
        fields = ",".join(str(field) for field in DCGM_FIELDS.values())
        self.proc = subprocess.Popen(
            ["dcgmi", "dmon", "-i", gpu, "-e", fields, "-d", "100"], stdout=subprocess.PIPE, text=True, bufsize=1
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            f = line.split()
            if len(f) == 2 + len(DCGM_FIELDS) and f[0] == "GPU" and "N/A" not in f:
                self.samples.append((time.time(), *(float(x) for x in f[2:])))

    def window(self, t0: float, t1: float) -> dict:
        time.sleep(0.35)  # let dmon flush the last samples of this window
        inside = [s for s in self.samples if t0 + 0.5 <= s[0] <= t1 - 0.15]
        out = {name: sum(s[i + 1] for s in inside) / len(inside) for i, name in enumerate(DCGM_FIELDS) if inside}
        return {**out, "dcgm_samples": len(inside)}


def percentiles(values) -> dict:
    v = sorted(values)
    return {"n": len(v), "p50_ms": v[len(v) // 2], "p95_ms": v[max(0, int(len(v) * 0.95) - 1)]}


def stage_loop(dcgm: Dcgm | None, name: str, call, seconds: float, warmup: int = 3) -> dict:
    """Run one stage alone, back to back, so the DCGM samples of the window belong to it. ``call`` must block."""
    for _ in range(warmup):
        call()
    latencies, t0 = [], time.time()
    while time.time() - t0 < seconds:
        start = time.perf_counter()
        call()
        latencies.append((time.perf_counter() - start) * 1000.0)
    activity = dcgm.window(t0, time.time()) if dcgm is not None else {}
    return {"stage": name, **percentiles(latencies), **activity}


def run(args) -> None:
    import vllm_omni

    deploy_config = args.deploy_config or str(Path(vllm_omni.__file__).parent / "deploy" / "pi05.yaml")
    on = True if args.all4 else None  # None leaves a switch at the deploy config's value (off by default)
    model = build_model(
        resolve_checkpoint(args.model),
        deploy_config,
        args.dtype,
        args.mode == "eager",
        args.vision_precision,
        on,
        precompute_adarms=on,
        hoist_denoise_invariants=on,
        prefix_kv_only=on,
        max_num_seqs=max(args.batches),
    )
    graphs, stages = model.cuda_graphs, Stages(model)
    dcgm = Dcgm(args.dcgm_gpu) if shutil.which("dcgmi") else None
    parts = [args.mode, args.vision_precision or "fp32"] + ["all4"] * args.all4
    chain = args.name or "_".join(parts + [args.dtype] * (args.dtype != "bfloat16"))
    meta = {
        "chain": chain,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "action_tokens": model.action_horizon,
        "num_steps": model.num_inference_steps,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "dcgm": dcgm is not None,
    }
    result = {"meta": meta, "batches": []}

    def replays() -> int:
        return sum(graphs.num_replays.values()) if graphs is not None else 0

    with torch.no_grad():
        for batch in args.batches:
            row = {"batch": batch}
            try:
                torch.accelerator.reset_peak_memory_stats()
                stages.timed, stages.recorded = True, None
                before = replays()
                for j in range(args.warmup):
                    stages.round(model, seeded_inputs(model, WARMUP_SEED_BASE + j, batch))
                    if graphs is not None and replays() == before:
                        raise RuntimeError("no CUDA graph serves this batch size (sample_actions ran eagerly)")
                rounds = [stages.round(model, seeded_inputs(model, seed, batch)) for seed in range(args.rounds)]
                row.update({key: [r[key] for r in rounds] for key in (*STAGES, "total")})
                row["peak_mem_gb"] = torch.accelerator.max_memory_allocated() / 2**30

                inputs = seeded_inputs(model, 0, batch)
                stages.recorded = {stage: [] for stage in STAGES}
                stages.round(model, inputs)
                row["loops"] = [stage_loop(dcgm, s, lambda s=s: stages.replay(s), args.seconds) for s in STAGES]
                stages.timed = False

                def unsplit():
                    model.sample_actions(**inputs)
                    torch.accelerator.synchronize()

                row["loops"].append(stage_loop(dcgm, "sample_actions", unsplit, args.seconds))
            except Exception as exc:  # a batch that cannot run is a result: report the limit, keep the rest
                row = {"batch": batch, "error": f"{type(exc).__name__}: {exc}"[:300]}
                torch.accelerator.empty_cache()
            stages.recorded = None
            result["batches"].append(row)
            summary = {key: round(percentiles(row[key])["p50_ms"], 2) for key in (*STAGES, "total") if key in row}
            print(json.dumps({"batch": batch, **summary, "error": row.get("error")}), flush=True)
    if dcgm is not None:
        dcgm.proc.terminate()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{chain}.json").write_text(json.dumps(result))
    print(f"wrote {out / f'{chain}.json'}")


def report(root: Path, out_md: Path | None) -> None:
    """One report over every chain file in ``root``; any chain written in this schema joins it (``openpi`` first)."""
    files = sorted(root.glob("*.json"), key=lambda f: (f.stem != "openpi", f.stem))
    runs = {f.stem: json.loads(f.read_text()) for f in files}
    runs = {key: run_ for key, run_ in runs.items() if "batches" in run_}
    rows = {key: {row["batch"]: row for row in run_["batches"]} for key, run_ in runs.items()}

    lines = [f"- {key}: {run_['meta']}" for key, run_ in runs.items()]
    lines += ["", "### Whole round: p50 ms (p95), p50 per request, peak memory", ""]
    lines += ["| B | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for b in sorted({b for chain_rows in rows.values() for b in chain_rows}):
        cells = []
        for key in runs:
            row = rows[key].get(b, {"error": "-"})
            if "error" in row:
                cells.append(row["error"][:70])
                continue
            total, peak = percentiles(row["total"]), row.get("peak_mem_gb", float("nan"))
            memory = f", {peak:.1f} GB" if peak == peak else ""
            cells.append(f"{total['p50_ms']:.1f} ({total['p95_ms']:.1f}), {total['p50_ms'] / b:.1f}/req{memory}")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")

    lines += ["", "### Stages: p50 ms in the staged rounds, GPU activity of the stage looped alone", ""]
    lines.append(
        "| chain | B | stage | p50 ms | ms/req | share | SM act | SM occ | tensor | FP32 | DRAM act | engine act | W |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in runs:
        for b, row in sorted(rows[key].items()):
            for loop in row.get("loops", []):
                stage = loop["stage"]  # ``sample_actions`` has no staged time: its row is the unsplit loop
                ms = percentiles(row[stage])["p50_ms"] if stage in row else loop["p50_ms"]
                share = 100 * ms / percentiles(row["total"])["p50_ms"]
                activity = " | ".join(f"{loop[n]:.2f}" if n in loop else "-" for n in list(DCGM_FIELDS)[:-1])
                watts = f"{loop['power_w']:.0f}" if "power_w" in loop else "-"
                cells = [key, b, stage, f"{ms:.1f}", f"{ms / b:.1f}", f"{share:.0f}%", activity, watts]
                lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if out_md is not None:
        out_md.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, help="Render the report of the chains in this directory and exit")
    parser.add_argument("--out-md", type=Path, help="With --report: also write the report here")
    parser.add_argument("--model", default="lerobot/pi05_base")
    parser.add_argument("--deploy-config", default=None, help="Default: the package's vllm_omni/deploy/pi05.yaml")
    parser.add_argument("--mode", default="graph", choices=["eager", "graph"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--vision-precision", default=None, help="pi05_vision_precision, e.g. selective_bf16")
    parser.add_argument("--all4", action="store_true", help="Turn on the four opt-in runtime switches")
    parser.add_argument("--name", help="Chain name (the output file's); default: derived from the options")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--rounds", type=int, default=100, help="Timed rounds per batch size")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed rounds per batch size")
    parser.add_argument("--seconds", type=float, default=10.0, help="Length of each stage loop")
    parser.add_argument("--dcgm-gpu", default="0", help="The device's DCGM GPU id")
    parser.add_argument("--out", default="pi05_batch")
    args = parser.parse_args()
    if args.report is not None:
        report(args.report, args.out_md)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
