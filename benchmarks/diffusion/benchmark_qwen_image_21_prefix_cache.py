# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare dense and paged Qwen-Image 2.1 at matching hardware/memory limits.

Run each mode in a fresh process. Dense keeps its default decode graphs;
timing includes completed image generation but excludes engine construction.
Partial/full preparation requests are untimed. Preparations and capacity
probes request two steps; a checkpoint's fixed sigma grid can override this.
Only the stable Step-0 prefix is retained.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image


def observe_decode_graph_replays():
    # Spawned Workers import this module too. Observe the real replay path
    # rather than inferring graph use from a requested configuration flag.
    from vllm_omni.diffusion.models.qwen_image_21.decode_graph import (
        QwenImage21DecodeGraphManager,
    )

    original = QwenImage21DecodeGraphManager.try_decode

    def observed(manager, *args, **kwargs):
        result = original(manager, *args, **kwargs)
        directory = os.environ.get("QWEN_IMAGE21_BENCH_TRACE_DIR")
        if result is not None and directory and not getattr(manager, "_benchmark_replay_observed", False):
            (Path(directory) / f"graph-replay-{os.getpid()}.json").write_text('{"replayed": true}\n')
            manager._benchmark_replay_observed = True
        return result

    QwenImage21DecodeGraphManager.try_decode = observed


observe_decode_graph_replays()


def reference(seed):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (128, 128, 3), dtype=np.uint8))


def main():
    import torch
    import vllm

    from vllm_omni.diffusion.diffusion_kv.manager import DiffusionKVCacheManager
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["dense", "paged_off", "paged_on"], required=True)
    parser.add_argument("--layout", choices=["single", "hsdp8"], required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--kv-cache-gib", type=float, default=14)
    parser.add_argument("--references", type=int, default=2)
    parser.add_argument("--cfg", type=float, default=4)
    parser.add_argument("--capacity-probes", type=int, default=32)
    args = parser.parse_args()
    if args.references < 2:
        parser.error("At least two references are required for the partial-hit workload")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["QWEN_IMAGE21_BENCH_TRACE_DIR"] = str(args.output.resolve())
    config = dict(
        model=args.model,
        enforce_eager=False,
        diffusion_kv_mode="dense_legacy" if args.mode == "dense" else "paged_scheduler",
        enable_prefix_caching=args.mode == "paged_on",
        max_num_batched_tokens=65536,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        additional_config={"diffusion_kv_profile_max_sequence_length": 512},
    )
    if args.mode != "dense":
        config.update(
            diffusion_kv_max_rows_per_request=2,
            kv_cache_memory_bytes=int(args.kv_cache_gib * 1024**3),
            # Native admission validates this bound against one rank's pool.
            # 24k covers the four-image/512-text/4k-target profile while
            # fitting the benchmark's 14 GiB single-GPU pool.
            max_model_len=24576,
        )
    if args.layout == "hsdp8":
        config.update(
            num_gpus=8,
            use_hsdp=True,
            hsdp_shard_size=8,
            hsdp_replicate_size=1,
            ulysses_degree=8,
        )
    observation = {}
    original = DiffusionKVCacheManager.reserve_request

    def reserve(manager, request_id, requests):
        metadata = original(manager, request_id, requests)
        if metadata is not None:
            observation.update(
                manager=manager,
                requests=tuple(requests),
                hits=[row.cached_prefix_len for row in metadata.sequences],
                prefixes=[row.prefix_len for row in requests],
                copies=len(metadata.block_copies),
            )
        return metadata

    DiffusionKVCacheManager.reserve_request = reserve
    record = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "config": config,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(),
        "samples": [],
    }
    record["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    from vllm.third_party import pynvml

    pynvml.nvmlInit()
    visible = os.environ.get(
        "CUDA_VISIBLE_DEVICES",
        ",".join(str(i) for i in range(torch.accelerator.device_count())),
    )
    handles = [
        pynvml.nvmlDeviceGetHandleByIndex(int(device)) if device.isdigit() else pynvml.nvmlDeviceGetHandleByUUID(device)
        for device in visible.split(",")
    ]
    peak_memory = [0] * len(handles)
    stop_monitor = threading.Event()

    def monitor_memory():
        while not stop_monitor.is_set():
            for index, handle in enumerate(handles):
                peak_memory[index] = max(peak_memory[index], pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            stop_monitor.wait(0.2)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()
    (args.output / "configuration.json").write_text(json.dumps(record, indent=2) + "\n")
    omni = None

    def generate(images, *, steps=None, cache_salt=None):
        observation.clear()
        params = OmniDiffusionSamplingParams(
            height=args.size,
            width=args.size,
            num_inference_steps=steps or args.steps,
            true_cfg_scale=args.cfg,
            seed=42,
            max_sequence_length=512,
        )
        start = time.perf_counter()
        outputs = list(
            omni.generate(
                {
                    "prompt": "Combine the reference images into a watercolor painting.",
                    "negative_prompt": "blurry, low quality",
                    "cache_salt": cache_salt,
                    "multi_modal_data": {"image": [image.copy() for image in images]},
                },
                params,
                py_generator=True,
            )
        )
        # The generator has returned a CPU image, so Worker CUDA work is done.
        elapsed = time.perf_counter() - start
        return elapsed, np.asarray(outputs[0].images[0]).copy()

    try:
        omni = Omni(**config)
        for case_idx, case in enumerate(["cold", "partial", "full"]):
            for iteration in range(args.warmup + args.repeats):
                seed = 10000 + case_idx * 10000 + iteration * (args.references + 1)
                images = [reference(seed + i) for i in range(args.references)]
                if case != "cold":
                    generate(images, steps=2)
                    if case == "partial":
                        images[-1] = reference(seed + args.references)
                # Different images still share the constant text preamble.
                # Isolate cold trials so even that header cannot hit.
                elapsed, array = generate(images, cache_salt=f"benchmark-cold-{seed}" if case == "cold" else None)
                sample = {
                    "case": case,
                    "warmup": iteration < args.warmup,
                    "seconds": elapsed,
                    "cached_prefix_len": observation.get("hits"),
                    "prefix_len": observation.get("prefixes"),
                    "block_copies": observation.get("copies"),
                    "image_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
                if args.mode == "paged_on":
                    hits = sample["cached_prefix_len"]
                    assert hits is not None
                    if case == "cold":
                        assert max(hits) == 0, sample
                    elif case == "full":
                        assert min(hits) == min(sample["prefix_len"]), sample
                    else:
                        assert 0 < min(hits) < min(sample["prefix_len"]), sample
                record["samples"].append(sample)
                with (args.output / "samples.jsonl").open("a") as stream:
                    stream.write(json.dumps(sample) + "\n")
                if iteration == args.warmup:
                    np.save(args.output / f"{case}.npy", array)
                print(json.dumps(sample), flush=True)
        record["median_seconds"] = {
            case: statistics.median(s["seconds"] for s in record["samples"] if s["case"] == case and not s["warmup"])
            for case in ["cold", "partial", "full"]
        }
        if args.mode == "paged_on" and args.capacity_probes:
            probes = []
            for index in range(args.capacity_probes):
                images = [reference(50000 + index * args.references + i) for i in range(args.references)]
                generate(images, steps=2)
                probes.append(observation["requests"])
            manager = observation["manager"]
            # Lookup-only probes allocate no target pages and do not republish
            # evicted entries, unlike replaying the requests through generate.
            retained = []
            for requests in probes:
                hits = [
                    manager._safe_cached_prefix_len(r, manager.native_manager.get_computed_blocks(r)[1])
                    for r in requests
                ]
                retained.append(min(hits) == min(r.prefix_len for r in requests))
            kv_config = manager.native_manager.kv_cache_config
            record["capacity"] = {
                "pool_blocks": kv_config.num_blocks,
                "allocated_kv_bytes_per_rank": sum(tensor.size for tensor in kv_config.kv_cache_tensors),
                "probed_distinct_prefixes": len(probes),
                "retained_full_prefixes": sum(retained),
                "retained_by_probe": retained,
            }
        record["decode_graph_replay_observed"] = bool(list(args.output.glob("graph-replay-*.json")))
        record["peak_gpu_memory_bytes"] = peak_memory.copy()
        (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    finally:
        if omni is not None:
            omni.close()
        stop_monitor.set()
        monitor.join()
        pynvml.nvmlShutdown()
        DiffusionKVCacheManager.reserve_request = original


if __name__ == "__main__":
    main()
