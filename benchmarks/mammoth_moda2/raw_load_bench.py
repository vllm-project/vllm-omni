# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Raw safetensors -> GPU micro-benchmark, without vLLM's loader.

Compares CPU and GPU dtype conversion after materializing each mmap-backed tensor.
Materialization includes page faults, CPU allocation and copying; it is not a disk
bandwidth measurement. CUDA setup is excluded; transfer includes GPU allocation.

  --mode cpu_cast         materialize -> cast to bf16 on the CPU -> copy to GPU
  --mode gpu_cast         materialize -> copy source dtype to GPU -> cast on the GPU

  python benchmarks/mammoth_moda2/raw_load_bench.py --model /path/MammothModa2-Preview --shards 6,7,8
  OMP_NUM_THREADS=4 python benchmarks/mammoth_moda2/raw_load_bench.py --model /path --shards 6,7,8 --mode cpu_cast
"""

import argparse
import json
import os
import re
import time

import torch
from safetensors import safe_open

SHARD_NO = re.compile(r"-(\d+)-of-\d+\.safetensors$")


def shard_files(model: str, shards: str) -> list[str]:
    """Resolve ``--shards`` to files, refusing empty or partial matches so a typo cannot pass as a fast load."""
    files = sorted(f for f in os.listdir(model) if f.endswith(".safetensors"))
    if not files:
        raise SystemExit(f"no .safetensors files under {model}")
    if shards != "all":
        want = {int(x) for x in shards.split(",")}
        found = {int(m.group(1)): f for f in files if (m := SHARD_NO.search(f))}
        missing = sorted(want - found.keys())
        if missing:
            raise SystemExit(f"shard(s) {missing} not found under {model}; available: {sorted(found)}")
        files = [found[n] for n in sorted(want)]
    return [os.path.join(model, f) for f in files]


def sync() -> None:
    if torch.cuda.is_available():
        torch.accelerator.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shards", default="all", help="comma list of 1-based shard numbers, or 'all'")
    ap.add_argument("--mode", choices=("cpu_cast", "gpu_cast"), default="cpu_cast")
    ap.add_argument("--label", default="raw")
    ap.add_argument("--no-cuda", action="store_true", help="read + CPU cast only (mode must be cpu_cast)")
    a = ap.parse_args()
    if a.no_cuda and a.mode != "cpu_cast":
        ap.error("--no-cuda only makes sense with --mode cpu_cast")

    paths = shard_files(a.model, a.shards)
    total_bytes = sum(os.path.getsize(p) for p in paths)

    if not a.no_cuda:
        # Initialize the context and exercise both operations outside the timer.
        warmup = torch.zeros(1).to("cuda")
        warmup = warmup.to(torch.bfloat16)
        sync()
        del warmup

    t0 = time.perf_counter()
    n_tensors, cpu_bytes = 0, 0
    dtypes: dict[str, int] = {}
    t_materialize = t_cast = t_h2d = 0.0
    for p in paths:
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                r0 = time.perf_counter()
                t = f.get_tensor(k).clone()  # touch every page and own the CPU storage
                t_materialize += time.perf_counter() - r0
                dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
                cpu_bytes += t.numel() * t.element_size()
                needs_cast = t.is_floating_point() and t.dtype != torch.bfloat16
                if a.mode == "cpu_cast":
                    c0 = time.perf_counter()
                    if needs_cast:
                        t = t.to(torch.bfloat16)
                    t_cast += time.perf_counter() - c0
                    if not a.no_cuda:
                        h0 = time.perf_counter()
                        g = t.to("cuda")
                        sync()
                        t_h2d += time.perf_counter() - h0
                        del g
                else:
                    h0 = time.perf_counter()
                    g = t.to("cuda")
                    sync()
                    t_h2d += time.perf_counter() - h0
                    c0 = time.perf_counter()
                    if needs_cast:
                        g = g.to(torch.bfloat16)
                    sync()
                    t_cast += time.perf_counter() - c0
                    del g
                n_tensors += 1
                del t
    total = time.perf_counter() - t0
    res = {
        "label": a.label,
        "mode": a.mode,
        "files": [os.path.basename(p) for p in paths],
        "file_bytes_gib": round(total_bytes / 2**30, 2),
        "tensor_bytes_gib": round(cpu_bytes / 2**30, 2),
        "n_tensors": n_tensors,
        "dtypes": dtypes,
        "torch_threads": torch.get_num_threads(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "materialize_s": round(t_materialize, 3),
        "cast_s": round(t_cast, 3),
        "h2d_s": round(t_h2d, 3),
        "total_s": round(total, 3),
        "cuda_setup_included": False,
    }
    print("RAWLOAD_JSON " + json.dumps(res))


if __name__ == "__main__":
    main()
