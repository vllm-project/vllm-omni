#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception: multi-request offline inference over a whole request set.

``end2end.py`` is the usage demo — one image, one query. This script is the
throughput form: every request is handed to a **single** ``omni.generate()`` call
so vLLM's scheduler can keep up to ``max_num_seqs`` stage-0 sequences in flight
(continuous batching on a paged KV cache). Calling ``generate()`` once per
request in a Python loop admits only one sequence at a time and measures serial
latency, not overlapping prefill/decode. Stage 1 still runs once per finished
thinker request; map outputs by request id, not finish order.

Usage:
    # explicit pairs
    python batch_inference.py --model tiiuae/Falcon-Perception \\
        --image a.jpg --query "the red pepper" \\
        --image b.jpg --query "every person" --max-num-seqs 4

    # a JSONL manifest of {"image": ..., "query": ...} lines
    python batch_inference.py --manifest requests.jsonl \
        --deploy-config falcon_perception.yaml

    # a HuggingFace dataset with `image` and `expression` columns
    python batch_inference.py --dataset tiiuae/PBench --split level_1 \\
        --limit 50 --deploy-config falcon_perception.yaml

Reading the throughput numbers
------------------------------
Four things decide whether the reported ``images/s`` means anything:

* **``max_num_seqs``.** ``falcon_perception.yaml`` uses the measured A100-80GB
  optimum of 4 on both stages. Larger is not faster for this model. Use
  ``--max-num-seqs`` to retune on other hardware; this script warns if an
  override serialises the workload at 1.
* **``enforce_eager``.** The shipped YAML sets it to ``false`` on both stages.
  Stage 0 also uses ``compilation_config: {cudagraph_mode: FULL_DECODE_ONLY}``
  so autoregressive decode can use CUDA graphs. Stage 1 sets ``cudagraph_mode:
  NONE`` because it has no decode loop; AnyUp has a separate compile knob. Edit
  a copy of the deploy YAML and pass it as ``--deploy-config`` to change nested
  compilation settings.
* **Warmup.** ``--warmup`` requests run before the clock starts, absorbing
  Triton JIT and allocator growth. They reuse request 0's image, which leaves
  that image resident in the stage-1 AnyUp feature cache — so request 0's
  measured cost is a cache hit. With the shipped profile, remove the effect by
  copying the deploy YAML and setting ``hf_overrides.hr_cache_mb`` to ``0``, or
  use ``--warmup 0`` to skip warmup entirely. The
  ``FALCON_PERCEPTION_HR_CACHE_MB`` fallback works only when that model override
  is omitted.
* **Peak VRAM is not comparable to a single-process engine.** vLLM preallocates
  its KV cache to ``gpu_memory_utilization`` up front, so the number reported
  here is a reservation, not a working set.

Only the ``generate()`` call is timed. Model load, warmup, and writing overlays
sit outside the measured region.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from vllm import SamplingParams

import vllm_omni
from vllm_omni.entrypoints.omni import Omni


class _Phase:
    """NVTX range naming the phase, so a trace separates warmup from measurement.

    Uses the ``nvtx`` package rather than ``torch.cuda.nvtx``: this runs in the
    driver process, which otherwise never touches CUDA, and the torch API would
    create a context there purely to emit a marker. Enabled by
    ``FALCON_PERCEPTION_NVTX``; a no-op otherwise, so untraced timing runs are
    unaffected.

    The phase boundary matters when reading a trace. Warmup requests carry
    compilation, Triton JIT and a cold allocator, and they seed both caches — a
    trace that cannot tell them apart will attribute all of that to the measured
    work.
    """

    _push = _pop = None
    if os.environ.get("FALCON_PERCEPTION_NVTX", "0") not in ("", "0"):
        try:
            import nvtx as _nvtx

            _push, _pop = _nvtx.push_range, _nvtx.pop_range
        except ImportError:
            _push = _pop = None

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        if _Phase._push is not None:
            _Phase._push(message=self.name, domain="falcon_perception")
        return self

    def __exit__(self, *exc):
        if _Phase._pop is not None:
            _Phase._pop(domain="falcon_perception")
        return False


# The reference stops on EOS (11) and <|end_of_query|> (263).
STOP_TOKEN_IDS = [11, 263]

PALETTE = [
    (255, 59, 48),
    (52, 199, 89),
    (0, 122, 255),
    (255, 149, 0),
    (175, 82, 222),
    (255, 204, 0),
    (90, 200, 250),
    (255, 45, 85),
    (162, 132, 94),
    (48, 209, 88),
]


def build_prompt(query: str) -> str:
    """The exact string the model expects. Deviating here silently degrades output."""
    return f"<|image|>Segment these expressions in the image:<|start_of_query|>{query}<|REF_SEG|>"


def overlay(image: Image.Image, masks: np.ndarray) -> Image.Image:
    """Draw each instance mask over the original image so a reviewer can eyeball it."""
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    width, height = canvas.size
    for i, mask in enumerate(masks):
        colour = PALETTE[i % len(PALETTE)]
        binary = np.asarray(mask) > 0
        if binary.shape != (height, width):
            rows = (np.arange(height) * binary.shape[0] / height).astype(int).clip(0, binary.shape[0] - 1)
            cols = (np.arange(width) * binary.shape[1] / width).astype(int).clip(0, binary.shape[1] - 1)
            binary = binary[rows][:, cols]
        stencil = Image.fromarray((binary * 255).astype(np.uint8))
        layer = Image.composite(Image.new("RGBA", canvas.size, (*colour, 115)), layer, stencil)
    return Image.alpha_composite(canvas, layer).convert("RGB")


def load_requests(args: argparse.Namespace) -> list[tuple[str, str, Image.Image, dict[str, Any]]]:
    """Collect ``(label, query, image, extras)`` tuples from whichever source was given.

    ``extras`` carries any non-``image``/``query`` keys of a manifest line straight
    through to ``results.json``, so a workload can label its own requests (sample
    index, repeat number, ...) without this script having to know what they mean.
    """
    if args.manifest:
        root = Path(args.manifest).parent
        entries = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
        requests = []
        for i, entry in enumerate(entries):
            path = Path(entry["image"])
            if not path.is_absolute():
                path = root / path
            extras = {k: v for k, v in entry.items() if k not in ("image", "query")}
            requests.append((f"{i:04d}_{path.stem}", entry["query"], Image.open(path).convert("RGB"), extras))
        return requests

    if args.dataset:
        # Streamed, matching how the reference benchmark takes its first N samples.
        from datasets import load_dataset

        stream = load_dataset(args.dataset, split=args.split, streaming=True)
        requests = []
        for i, sample in enumerate(stream):
            if 0 < args.limit <= i:
                break
            requests.append((f"{i:04d}", sample["expression"], sample["image"].convert("RGB"), {}))
        return requests

    if len(args.image) != len(args.query):
        raise SystemExit(
            f"--image given {len(args.image)} times but --query {len(args.query)} times; they must pair up"
        )
    return [
        (f"{i:04d}_{Path(p).stem}", q, Image.open(p).convert("RGB"), {})
        for i, (p, q) in enumerate(zip(args.image, args.query))
    ]


def resolve_deploy_config(name: str) -> str:
    """Expand a packaged deploy YAML name to a full path.

    ``Omni`` resolves a bare name against ``vllm_omni/deploy/`` on one code path
    but reads it as a literal path on another, so a bare name only works when the
    CWD happens to contain the file. Resolve it here and pass a real path.
    """
    path = Path(name)
    if path.exists():
        return str(path)
    packaged = Path(vllm_omni.__file__).parent / "deploy" / name
    if packaged.exists():
        return str(packaged)
    raise SystemExit(f"deploy config not found: {name!r} (looked in ./ and {packaged.parent})")


def stage_max_num_seqs(omni: Omni) -> list[Any]:
    """Effective ``max_num_seqs`` per stage, or ``None`` where it can't be read."""
    values: list[Any] = []
    for cfg in getattr(omni, "stage_configs", None) or []:
        engine_args = getattr(cfg, "engine_args", None)
        value = None
        if engine_args is not None:
            value = (
                engine_args.get("max_num_seqs")
                if hasattr(engine_args, "get")
                else getattr(engine_args, "max_num_seqs", None)
            )
        values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="tiiuae/Falcon-Perception", help="HF id or local snapshot path")
    parser.add_argument("--deploy-config", default="falcon_perception.yaml", help="deploy YAML name or path")

    source = parser.add_argument_group("request source (pick one)")
    source.add_argument("--image", action="append", default=[], help="input image; repeat, pairing with --query")
    source.add_argument("--query", action="append", default=[], help="what to segment; repeat, pairing with --image")
    source.add_argument("--manifest", help='JSONL file of {"image": ..., "query": ...} lines')
    source.add_argument("--dataset", help="HF dataset id with `image` and `expression` columns")
    source.add_argument("--split", default="level_1", help="dataset split (with --dataset)")
    source.add_argument("--limit", type=int, default=50, help="max samples to take; -1 for all (with --dataset)")

    parser.add_argument("--max-num-seqs", type=int, help="scheduler concurrency cap per stage (stage-0 throughput)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1, help="untimed requests on image 0 before the measured run")
    parser.add_argument(
        "--passes", type=int, default=1, help="submit the set this many times, timing each; pass 2+ is cache-warm"
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--out-dir", default="/tmp/falcon_perception_batch", help="where results.json and any artifacts go"
    )
    parser.add_argument("--save-overlays", action="store_true", help="write a mask overlay PNG per request")
    parser.add_argument("--save-masks", action="store_true", help="write compressed mask arrays per request")
    args = parser.parse_args()

    if not (args.image or args.manifest or args.dataset):
        raise SystemExit("give requests via --image/--query pairs, --manifest, or --dataset")

    requests = load_requests(args)
    if not requests:
        raise SystemExit("no requests to run")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded {len(requests)} requests")

    engine_overrides: dict[str, Any] = {}
    if args.max_num_seqs is not None:
        engine_overrides["max_num_seqs"] = args.max_num_seqs
    deploy_config = resolve_deploy_config(args.deploy_config)
    omni = Omni(model=args.model, deploy_config=deploy_config, seed=args.seed, **engine_overrides)

    batch_limits = stage_max_num_seqs(omni)
    if len(requests) > 1 and any(v == 1 for v in batch_limits):
        print(
            f"\nWARNING: max_num_seqs per stage is {batch_limits} — at 1 the scheduler runs one\n"
            f"         request at a time, so throughput here matches a per-prompt generate() loop.\n"
            f"         Pass --max-num-seqs 4 or restore falcon_perception.yaml's default\n"
            f"         before reading these numbers.\n"
        )

    # One SamplingParams per stage. Greedy is required, not a preference: the
    # geometry (a bounding box per object) is decoded from the hidden state that
    # produced each <|coord|> / <|size|> token, so sampling with temperature > 0
    # would desynchronise boxes from tokens.
    params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            detokenize=True,
            stop_token_ids=STOP_TOKEN_IDS,
        ),
        SamplingParams(temperature=0.0, max_tokens=1, detokenize=False),
    ]

    prompts = [{"prompt": build_prompt(query), "multi_modal_data": {"image": image}} for _, query, image, _ in requests]

    for i in range(args.warmup):
        print(f"warmup {i + 1}/{args.warmup} ...", flush=True)
        with _Phase(f"fp/warmup[{i}]"):
            omni.generate(prompts[:1], params, use_tqdm=False)

    # One generate() admits every request; with max_num_seqs > 1 the stage-0
    # scheduler can overlap them (continuous batching). A per-prompt loop admits
    # one at a time.
    #
    # With --passes > 1 the same set is submitted again as a second batched call.
    # That is the readable way to measure the caches: pass 1 is cold, pass 2 hits
    # the KV prefix cache (identical prompts) and the stage-1 AnyUp feature cache
    # (identical image hashes). Interleaving repeats *within* one call instead
    # leaves both passes queued together, so cold and warm cost cannot be
    # separated. Only the last pass's outputs are kept.
    pass_wall_s: list[float] = []
    for p_i in range(args.passes):
        started = time.perf_counter()
        with _Phase(f"fp/pass[{p_i}][n={len(prompts)}]"):
            outputs = omni.generate(prompts, params)
        pass_wall_s.append(time.perf_counter() - started)
        if args.passes > 1:
            print(f"pass {p_i + 1}/{args.passes}: {pass_wall_s[-1]:.1f}s", flush=True)
    wall_s = pass_wall_s[-1]

    # `Omni` builds request ids as f"{index}_{uuid}", so the submission index is
    # recoverable exactly. Never map outputs back by arrival order — stages
    # finish out of order under batching.
    masks_by_idx: dict[int, np.ndarray] = {}
    boxes_by_idx: dict[int, list[list[float]]] = {}
    token_ids_by_idx: dict[int, list[int]] = {}
    tokens_out_by_idx: dict[int, int] = {}
    tokens_in_by_idx: dict[int, int] = {}
    stage_metrics_by_idx: dict[int, dict[str, Any]] = {}
    peak_memory_mb = 0.0

    for output in outputs:
        idx = int(str(output.request_id).split("_", 1)[0])
        peak_memory_mb = max(peak_memory_mb, float(getattr(output, "peak_memory_mb", 0.0) or 0.0))

        for stage in (output.metrics or {}).get("stage_metrics", {}).values():
            # Per-stage generation time / TTFT / token counts, straight from the
            # orchestrator. This is the profiling report: with everything in one
            # batch there is no per-request wall clock to measure instead.
            stage_metrics_by_idx.setdefault(idx, {})[str(stage.get("stage_id"))] = {
                k: stage.get(k)
                for k in ("stage_gen_time_ms", "vllm_ttft_ms", "vllm_tpot_ms", "num_tokens_in", "num_tokens_out")
            }
            if int(stage.get("stage_id", -1)) == 0:
                tokens_in_by_idx[idx] = int(stage.get("num_tokens_in", 0))
                tokens_out_by_idx[idx] = int(stage.get("num_tokens_out", 0))

        completion = output.request_output.outputs[0]
        if output.final_output_type == "text":
            # `.text` is empty for this model: every generated token is a special
            # token (<|coord|>, <|size|>, <|seg|>, ...) and detokenization skips
            # them. The token ids are the thing worth recording.
            token_ids_by_idx[idx] = [int(t) for t in completion.token_ids]
            tokens_out_by_idx.setdefault(idx, len(completion.token_ids))
        multimodal = getattr(completion, "multimodal_output", None)
        if not multimodal:
            continue
        if "masks" in multimodal and hasattr(multimodal["masks"], "shape"):
            masks_by_idx[idx] = np.asarray(multimodal["masks"].to(torch.uint8).cpu())
        if "boxes" in multimodal and hasattr(multimodal["boxes"], "shape"):
            boxes_by_idx[idx] = np.asarray(multimodal["boxes"].float().cpu()).tolist()

    rows = []
    for idx, (label, query, image, extras) in enumerate(requests):
        masks = masks_by_idx.get(idx, np.zeros((0, 1, 1), dtype=np.uint8))
        if args.save_masks:
            np.savez_compressed(out_dir / f"{label}_masks.npz", masks=masks)
        if args.save_overlays and masks.shape[0]:
            overlay(image, masks).save(out_dir / f"{label}_masks.png")
        rows.append(
            {
                "idx": idx,
                "label": label,
                "query": query,
                "image_size": list(image.size),
                "n_instances": int(masks.shape[0]),
                "mask_shape": list(masks.shape),
                "boxes": boxes_by_idx.get(idx, []),
                "output_token_ids": token_ids_by_idx.get(idx, []),
                "prompt_tokens": tokens_in_by_idx.get(idx),
                "output_tokens": tokens_out_by_idx.get(idx),
                "stage_metrics": stage_metrics_by_idx.get(idx, {}),
                **extras,
            }
        )

    prompt_tokens = sum(r["prompt_tokens"] or 0 for r in rows)
    output_tokens = sum(r["output_tokens"] or 0 for r in rows)
    total_instances = sum(r["n_instances"] for r in rows)
    unresolved = [r["label"] for r in rows if r["idx"] not in masks_by_idx]

    summary = {
        "model": args.model,
        "deploy_config": deploy_config,
        "max_num_seqs_per_stage": batch_limits,
        "requests": len(rows),
        "warmup_requests": args.warmup,
        "passes": args.passes,
        "pass_wall_s": pass_wall_s,
        "wall_s": wall_s,
        "images_per_s": len(rows) / wall_s if wall_s else 0.0,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens_per_s": (prompt_tokens + output_tokens) / wall_s if wall_s else 0.0,
        "output_tokens_per_s": output_tokens / wall_s if wall_s else 0.0,
        "total_instances": total_instances,
    }
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print(f"\n{'=' * 62}\nMULTI-REQUEST OFFLINE INFERENCE — one generate() call\n{'=' * 62}")
    print(f"  Requests         : {len(rows)}")
    print(f"  max_num_seqs     : {batch_limits} (per stage)")
    if args.passes > 1:
        cold, warm = pass_wall_s[0], pass_wall_s[-1]
        print(f"  Pass wall times  : {', '.join(f'{t:.1f}s' for t in pass_wall_s)}")
        print(f"  Cold -> warm     : {cold:.1f}s -> {warm:.1f}s ({cold / warm:.2f}x from caching)")
    print(f"  Wall time        : {wall_s:.1f}s  (last pass; all rates below use it)")
    print(f"  Images/s         : {summary['images_per_s']:.3f}")
    print(f"  Total tok/s      : {summary['total_tokens_per_s']:.1f}  (prompt + generated)")
    print(f"  Output tok/s     : {summary['output_tokens_per_s']:.1f}")
    print(f"  Prompt tokens    : {prompt_tokens}")
    print(f"  Output tokens    : {output_tokens}")
    print(f"  Total instances  : {total_instances}")
    counts = [r["n_instances"] for r in rows]
    print(f"  Instances/request: mean {statistics.mean(counts):.1f}  median {statistics.median(counts):.1f}")
    if peak_memory_mb:
        summary["worker_peak_memory_mb"] = peak_memory_mb
        print(f"  Worker peak VRAM : {peak_memory_mb / 1024:.2f} GiB (includes preallocated KV cache)")
    if unresolved:
        print(f"  NOTE: {len(unresolved)} request(s) produced no mask output: {unresolved[:5]}")
    print(f"\nwrote {out_dir / 'results.json'}")


# Required: the deploy YAML uses distributed_executor_backend: mp with spawn, and
# a script without this guard hangs in _check_not_importing_main.
if __name__ == "__main__":
    main()
