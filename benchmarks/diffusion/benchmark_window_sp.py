# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark the SeedVR2 window-aligned sequence-parallel path.

Runs ``tests/diffusion/models/seedvr2/window_sp_worker.py --case seedvr2`` for a
list of SP degrees and merges the per-rank reports into one summary, following
the ``window_sequence_parallel`` design doc's result schema.

Usage::

    python benchmarks/diffusion/benchmark_window_sp.py \
        --ckpt /models/seedvr2_ema_3b_fp16.safetensors \
        --sp-sizes 1,2,4 --frames 4 --height 64 --width 64 --text-len 58 \
        --out /tmp/seedvr2-window-sp/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKER = Path(__file__).resolve().parents[2] / "tests/diffusion/models/seedvr2/window_sp_worker.py"


def run_degree(args: argparse.Namespace, sp_size: int) -> dict:
    report_dir = Path(args.out).parent / f"sp{sp_size}"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={sp_size}",
        str(WORKER),
        "--case",
        "seedvr2",
        "--seed",
        str(args.seed),
        "--ckpt",
        args.ckpt,
        "--frames",
        str(args.frames),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--text-len",
        str(args.text_len),
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--report-dir",
        str(report_dir),
    ]
    if args.varlen:
        command.append("--varlen")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    report_path = report_dir / "report.json"
    if not report_path.exists():
        return {
            "sp_size": sp_size,
            "status": "error",
            "error": f"worker failed (rc={result.returncode})",
            "stderr_tail": result.stderr[-2000:],
        }
    report = json.loads(report_path.read_text())
    rank_zero = next(iter(report.values()))
    statuses = sorted({entry["status"] for entry in report.values()})
    return {
        "sp_size": sp_size,
        "status": "ok" if statuses == ["ok"] else "fail",
        "statuses": statuses,
        "per_rank": {name: entry for name, entry in report.items()},
        "forward_ms_median": max(entry.get("forward_ms_spN", 0.0) for entry in report.values()),
        "forward_ms_p95": max(entry.get("forward_ms_spN_p95", 0.0) for entry in report.values()),
        "forward_ms_sp1_median": max(entry.get("forward_ms_sp1", 0.0) for entry in report.values()),
        "peak_allocated_bytes_max": max(entry.get("peak_allocated_bytes", 0) for entry in report.values()),
        "layout_transition_count": rank_zero.get("layout_transition_count"),
        "network_a2a_count": rank_zero.get("network_a2a_count"),
        "text_all_reduce_count": rank_zero.get("text_all_reduce_count"),
        "logical_remote_video_bytes": rank_zero.get("logical_remote_video_bytes"),
        "post_patch_shape": rank_zero.get("post_patch_shape"),
        "video_tokens": rank_zero.get("video_tokens"),
        "per_rank_tokens": rank_zero.get("per_rank_tokens"),
        "max_abs_error": max(entry.get("max_abs_error", 0.0) for entry in report.values()),
        "rel_l2_max": max(entry.get("rel_l2", 0.0) for entry in report.values()),
        "attention_layers_per_resolved_path": rank_zero.get("attention_layers_per_resolved_path"),
        "attention_backend_names": rank_zero.get("attention_backend_names"),
        "varlen_fallback_reasons": rank_zero.get("varlen_fallback_reasons"),
        "packed_varlen_calls": rank_zero.get("packed_varlen_calls"),
        "grouped_sdpa_calls": rank_zero.get("grouped_sdpa_calls"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--sp-sizes", default="1,2,4")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--text-len", type=int, default=58)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7723)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    results = [run_degree(args, int(size)) for size in args.sp_sizes.split(",")]
    baseline = next((entry for entry in results if entry["sp_size"] == 1 and entry["status"] == "ok"), None)
    for entry in results:
        if baseline and entry["status"] == "ok":
            entry["speedup_vs_sp1"] = entry["forward_ms_sp1_median"] / entry["forward_ms_median"]
            entry["scaling_efficiency"] = entry["speedup_vs_sp1"] / entry["sp_size"]

    payload = {
        "fixture": {
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "text_tokens": args.text_len,
            "dtype": args.dtype,
            "checkpoint": args.ckpt,
            "seed": args.seed,
            "attention_path_requested": "packed_varlen" if args.varlen else "grouped_sdpa",
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"{'SP':>3} {'fwd ms (max rank)':>18} {'speedup':>9} {'efficiency':>10} {'rel_l2':>10} {'peak GiB':>9}")
    for entry in results:
        if entry["status"] != "ok":
            print(f"{entry['sp_size']:>3} {'FAILED':>18}  {entry.get('error', '')}")
            continue
        print(
            f"{entry['sp_size']:>3} {entry['forward_ms_median']:>18.1f} "
            f"{entry.get('speedup_vs_sp1', float('nan')):>9.3f} "
            f"{entry.get('scaling_efficiency', float('nan')):>10.3f} "
            f"{entry['rel_l2_max']:>10.3e} "
            f"{entry['peak_allocated_bytes_max'] / 2**30:>9.2f}"
        )
    print(f"wrote {out_path}")
    return 0 if all(entry["status"] == "ok" for entry in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
