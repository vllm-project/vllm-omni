# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni serving benchmark for unified understanding+generation models
(e.g. HunyuanImage-3.0 AR+DiT).

Sends a configurable mix of i2t (image understanding, text output), t2i
(text-to-image) and it2i (image editing) requests to a vLLM-Omni serving
endpoint and reports overall + per-task-type + per-bot_task latency /
throughput statistics, with optional per-stage (AR / DiT) timings. This is the
workload the DTPS scheduler (ar_only vs ar_downstream) is tuned for, so
the output is shaped to support evaluating scheduling optimizations.

Three task types are dispatched over ``/v1/chat/completions`` (the AR+DiT
multi-stage path), distinguished by ``modalities`` and the presence of a
reference image — exactly how ``serving_chat._resolve_omni_task_type`` classifies
them. it2i may alternatively go through ``/v1/images/edits`` (``--it2i-endpoint
images-edits``). it2i ``bot_task`` (recaption / think / think_recaption) is
sampled per request by configurable weights to cover the three thinking
intensities.

Usage examples are in benchmarks/omni/README.md.

Quick smoke (no server needed, --dry-run just prints the plan):
    python benchmarks/omni/omni_benchmark_serving.py \
        --num-i2t 70 --num-t2i 10 --num-it2i 20 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

# Make this directory importable when run as a script (omni_backends /
# omni_dataset are sibling modules).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from omni_backends import (  # noqa: E402
    IT2I_BOT_TASKS,
    OmniRequest,
    RequestFuncOutput,
    extract_chat_outputs,
    extract_edits_outputs,
    prepare_request_images,
    select_request_func,
)
from omni_dataset import (  # noqa: E402
    GEN_RESOLUTIONS,
    OmniConfig,
    TaskCounts,
    build_requests,
    parse_bot_task_weights,
    parse_gen_resolution_weights,
)

_TASK_TYPES = ("i2t", "t2i", "it2i")
_CHAT_ENDPOINT = "/v1/chat/completions"
_EDITS_ENDPOINT = "/v1/images/edits"


def _parse_group_order(raw: str | None) -> tuple[str, ...]:
    """Parse ``--no-shuffle-order`` (e.g. ``t2i,it2i,i2t``) into a tuple.

    Must be a permutation of i2t/t2i/it2i; defaults to ``(i2t, t2i, it2i)``. Only
    consulted when ``--no-shuffle`` is set (the merge is shuffled otherwise).
    """
    if not raw:
        return _TASK_TYPES
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    if set(parts) != set(_TASK_TYPES) or len(parts) != len(_TASK_TYPES):
        raise ValueError(f"--no-shuffle-order must be a permutation of i2t,t2i,it2i, got {list(parts)!r}.")
    return parts


def _percentiles(values: list[float], ps: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    return {f"p{p}": float(np.percentile(values, p)) for p in ps}


def _latency_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": len(latencies),
        "mean": float(np.mean(latencies)),
        "median": float(np.median(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    }


def _aggregate_stage_durations(
    outputs: list[RequestFuncOutput],
) -> dict[str, dict[str, float]]:
    """Mean / p50 / p99 of each named stage duration across the given outputs."""
    buckets: dict[str, list[float]] = {}
    for o in outputs:
        for stage, val in (o.stage_durations or {}).items():
            buckets.setdefault(str(stage), []).append(float(val))
    result: dict[str, dict[str, float]] = {}
    for stage, vals in buckets.items():
        result[stage] = {
            "count": len(vals),
            "mean": float(np.mean(vals)),
            "p50": float(np.percentile(vals, 50)),
            "p99": float(np.percentile(vals, 99)),
        }
    return result


def _latency_ms_stats(values: list[float], ps: list[float]) -> dict[str, Any]:
    """mean / std / median / percentiles (ms) for a list of second-scale latencies.

    Mirrors vLLM ``serve.py``'s ttft/tpot/itl aggregation (values are converted
    to milliseconds). Returns zeroed fields when ``values`` is empty so the
    shape is stable for JSON consumers.
    """
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "median_ms": 0.0,
            **{f"p{p}_ms": 0.0 for p in ps},
        }
    arr = np.asarray(values, dtype=float) * 1000.0
    return {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "median_ms": float(np.median(arr)),
        **{f"p{p}_ms": float(np.percentile(arr, p)) for p in ps},
    }


def _text_stream_metrics(
    outputs: list[RequestFuncOutput],
    ps: list[float],
) -> dict[str, Any]:
    """TTFT / TPOT / ITL aggregation over the streaming text outputs.

    Only outputs that actually streamed text (``ttft > 0``) contribute — i.e.
    i2t AR answers and it2i AR "thinking" text from the streaming senders.
    Non-streaming runs and tasks whose text the server does not stream (chat
    t2i/it2i) are simply absent (count=0), not counted as zero-latency samples.

    Per-request TPOT follows vLLM's "time per output token excluding the first":
    ``sum(itl) / (output_tokens - 1)`` when ``output_tokens > 1``, else 0. Using
    the ITL sum (== last-text-time - ttft) instead of ``latency - ttft`` keeps
    TPOT inside the text-stream window — for i2t this matches vLLM (AR is
    terminal, latency ≈ last token); for it2i it correctly excludes the DiT
    stage that runs after the AR text finishes. ITL itself is the flattened list
    of every inter-token gap across the contributing requests.
    """
    ttfts: list[float] = []
    tpots: list[float] = []
    itls: list[float] = []
    output_tokens: list[int] = []
    for o in outputs:
        if not o.success or o.ttft <= 0.0:
            continue
        ttfts.append(o.ttft)
        output_tokens.append(o.output_tokens)
        if o.output_tokens > 1:
            tpots.append(sum(o.itl or []) / (o.output_tokens - 1))
        else:
            tpots.append(0.0)
        itls.extend(o.itl or [])
    return {
        "measured_requests": len(ttfts),
        "output_tokens": {
            "total": int(sum(output_tokens)),
            "mean": float(np.mean(output_tokens)) if output_tokens else 0.0,
            "median": float(np.median(output_tokens)) if output_tokens else 0.0,
            "max": int(max(output_tokens)) if output_tokens else 0,
        },
        "ttft": _latency_ms_stats(ttfts, ps),
        "tpot": _latency_ms_stats(tpots, ps),
        "itl": _latency_ms_stats(itls, ps),
    }


def _bucket_outputs(
    requests_list: list[OmniRequest],
    outputs: list[RequestFuncOutput],
) -> dict[str, list[tuple[OmniRequest, RequestFuncOutput]]]:
    """Group (req, out) pairs by task_type (and it2i by bot_task sub-key)."""
    buckets: dict[str, list[tuple[OmniRequest, RequestFuncOutput]]] = {t: [] for t in _TASK_TYPES}
    for req, out in zip(requests_list, outputs):
        if req.task_type in buckets:
            buckets[req.task_type].append((req, out))
    return buckets


def calculate_metrics(
    requests_list: list[OmniRequest],
    outputs: list[RequestFuncOutput],
    total_duration: float,
    selected_percentiles: list[float],
) -> dict[str, Any]:
    """Overall + per-task + per-bot_task metrics.

    Per-task throughput uses the total benchmark duration as the denominator
    (not the task's own span) so the three task throughputs sum to the overall
    request throughput — directly comparable across runs / configs.
    """
    success_all = [o for o in outputs if o.success]
    fail_all = [o for o in outputs if not o.success]
    lat_all = [o.latency for o in success_all]

    overall = {
        "duration": total_duration,
        "completed": len(success_all),
        "failed": len(fail_all),
        "throughput_qps": (len(success_all) / total_duration) if total_duration > 0 else 0.0,
        **_latency_stats(lat_all),
    }
    overall["percentiles"] = _percentiles(lat_all, selected_percentiles)
    peak_mems = [o.peak_memory_mb for o in success_all if o.peak_memory_mb > 0]
    overall["peak_memory_mb"] = {
        "max": float(max(peak_mems)) if peak_mems else 0.0,
        "mean": float(np.mean(peak_mems)) if peak_mems else 0.0,
    }
    overall["stage_durations"] = _aggregate_stage_durations(success_all)
    overall["text_stream"] = _text_stream_metrics(success_all, selected_percentiles)

    per_task: dict[str, Any] = {}
    buckets = _bucket_outputs(requests_list, outputs)
    for task in _TASK_TYPES:
        pairs = buckets[task]
        succ = [(r, o) for r, o in pairs if o.success]
        lats = [o.latency for _, o in succ]
        per_task[task] = {
            "total": len(pairs),
            "completed": len(succ),
            "failed": len(pairs) - len(succ),
            "success_rate": (len(succ) / len(pairs)) if pairs else 0.0,
            "throughput_qps": (len(succ) / total_duration) if total_duration > 0 else 0.0,
            **_latency_stats(lats),
            "percentiles": _percentiles(lats, selected_percentiles),
            "stage_durations": _aggregate_stage_durations([o for _, o in succ]),
            "text_stream": _text_stream_metrics([o for _, o in succ], selected_percentiles),
        }

    # it2i broken down by bot_task.
    it2i_by_bot: dict[str, Any] = {}
    it2i_pairs = buckets["it2i"]
    if it2i_pairs:
        by_bt: dict[str, list[tuple[OmniRequest, RequestFuncOutput]]] = {t: [] for t in IT2I_BOT_TASKS}
        for r, o in it2i_pairs:
            bt = r.bot_task or "none"
            by_bt.setdefault(bt, []).append((r, o))
        for bt, ps in by_bt.items():
            if not ps:
                continue
            succ = [(r, o) for r, o in ps if o.success]
            lats = [o.latency for _, o in succ]
            it2i_by_bot[bt] = {
                "total": len(ps),
                "completed": len(succ),
                "failed": len(ps) - len(succ),
                "success_rate": (len(succ) / len(ps)) if ps else 0.0,
                "throughput_qps": (len(succ) / total_duration) if total_duration > 0 else 0.0,
                **_latency_stats(lats),
                "percentiles": _percentiles(lats, selected_percentiles),
                "stage_durations": _aggregate_stage_durations([o for _, o in succ]),
                "text_stream": _text_stream_metrics([o for _, o in succ], selected_percentiles),
            }

    return {
        "overall": overall,
        "per_task": per_task,
        "it2i_by_bot_task": it2i_by_bot,
    }


def _print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)


def _print_latency_row(label: str, stats: dict[str, Any]) -> None:
    print(
        f"  {label:<22} n={stats.get('count', 0):<5} "
        f"mean={stats.get('mean', 0):.3f}s "
        f"p50={stats.get('p50', 0):.3f}s "
        f"p95={stats.get('p95', 0):.3f}s "
        f"p99={stats.get('p99', 0):.3f}s"
    )


def _print_text_stream(ts: dict[str, Any], indent: str = "  ") -> None:
    """Print TTFT / TPOT / ITL + output-token stats from a text_stream bucket.

    Only shown when the bucket actually measured streaming text outputs
    (``measured_requests > 0``); non-streaming runs / N/A tasks stay silent.
    """
    n = ts.get("measured_requests", 0)
    if not n:
        return
    out_tok = ts["output_tokens"]
    print(
        f"{indent}text-output (n={n}, out_tokens mean={out_tok['mean']:.1f} "
        f"median={out_tok['median']:.1f} max={out_tok['max']}):"
    )
    for key, label in (("ttft", "TTFT"), ("tpot", "TPOT"), ("itl", "ITL")):
        s = ts[key]
        print(
            f"{indent}  {label:<5} n={s['count']:<5} "
            f"mean={s['mean_ms']:.2f}ms median={s['median_ms']:.2f}ms "
            f"p50={s['p50_ms']:.2f}ms p95={s['p95_ms']:.2f}ms p99={s['p99_ms']:.2f}ms"
        )


def print_metrics(metrics: dict[str, Any], args: argparse.Namespace) -> None:
    ov = metrics["overall"]
    _print_section("Omni Serving Benchmark Result")

    print(f"{'Endpoint (i2t/t2i):':<30} {_CHAT_ENDPOINT}")
    print(f"{'Endpoint (it2i):':<30} " + (_EDITS_ENDPOINT if args.it2i_endpoint == "images-edits" else _CHAT_ENDPOINT))
    print(f"{'Model:':<30} {args.model}")
    print(f"{'Dataset:':<30} {args.dataset}")
    print(f"{'Mix (i2t:t2i:it2i):':<30} {args.num_i2t}:{args.num_t2i}:{args.num_it2i}")
    print(f"{'bot_task weights:':<30} {args.it2i_bot_task_weights}")
    print(f"{'bot_task sampling:':<30} {args.it2i_bot_task_sampling}")
    print(f"{'Shuffle:':<30} {args.shuffle}")
    if not args.shuffle:
        gorder = args.no_shuffle_order or "i2t,t2i,it2i (default)"
        print(f"{'No-shuffle group order:':<30} {gorder}")
    if args.gen_resolution_weights:
        print(f"{'Gen resolution weights:':<30} {args.gen_resolution_weights}")
        print(f"{'Gen resolution sampling:':<30} {args.gen_resolution_sampling}")
    print(f"{'Randomize input:':<30} {args.randomize_input}")
    print(f"{'Seed:':<30} {args.seed}")
    out_dir = metrics.get("config", {}).get("output_dir")
    if out_dir:
        print(f"{'Output dir:':<30} {out_dir}")

    print("-" * 60)
    print(f"{'Benchmark duration (s):':<30} {ov['duration']:.2f}")
    print(f"{'Request rate:':<30} {args.request_rate}")
    print(f"{'Max concurrency:':<30} {args.max_concurrency}")
    print(f"{'Successful requests:':<30} {ov['completed']}/{ov['completed'] + ov['failed']}")
    print(f"{'Request throughput (req/s):':<30} {ov['throughput_qps']:.2f}")
    print(f"{'Latency mean (s):':<30} {ov['mean']:.4f}")
    print(f"{'Latency p50 (s):':<30} {ov['p50']:.4f}")
    print(f"{'Latency p95 (s):':<30} {ov['p95']:.4f}")
    print(f"{'Latency p99 (s):':<30} {ov['p99']:.4f}")

    if ov["peak_memory_mb"]["max"] > 0:
        print("-" * 60)
        print(
            f"{'Peak Memory max (MB):':<30} {ov['peak_memory_mb']['max']:.2f}  mean: {ov['peak_memory_mb']['mean']:.2f}"
        )

    if ov["stage_durations"]:
        print("-" * 60)
        print("Stage Durations (overall, seconds):")
        for stage, s in ov["stage_durations"].items():
            print(f"  {stage:<22} n={s['count']:<5} mean={s['mean']:.4f} p50={s['p50']:.4f} p99={s['p99']:.4f}")

    if ov.get("text_stream", {}).get("measured_requests", 0):
        print("-" * 60)
        print("Text-Output Streaming (overall):")
        _print_text_stream(ov["text_stream"])

    _print_section("Per-Task-Type Result")
    for task in _TASK_TYPES:
        t = metrics["per_task"][task]
        print("-" * 60)
        print(
            f"  [{task}] total={t['total']} completed={t['completed']} "
            f"failed={t['failed']} success_rate={t['success_rate']:.2%} "
            f"qps={t['throughput_qps']:.2f}"
        )
        _print_latency_row(task, t)
        if t["stage_durations"]:
            for stage, s in t["stage_durations"].items():
                print(
                    f"    stage {stage:<18} n={s['count']:<5} "
                    f"mean={s['mean']:.4f} p50={s['p50']:.4f} p99={s['p99']:.4f}"
                )
        _print_text_stream(t.get("text_stream", {}), indent="    ")

    if metrics["it2i_by_bot_task"]:
        _print_section("it2i by bot_task (thinking intensity)")
        # it2i total summary first (mirrors per_task["it2i"]), then the
        # per-bot_task breakdown so the section reads total -> breakdown.
        it2i_total = metrics["per_task"]["it2i"]
        print("-" * 60)
        print(
            f"  [it2i TOTAL] total={it2i_total['total']} completed={it2i_total['completed']} "
            f"failed={it2i_total['failed']} success_rate={it2i_total['success_rate']:.2%} "
            f"qps={it2i_total['throughput_qps']:.2f}"
        )
        _print_latency_row("it2i TOTAL", it2i_total)
        if it2i_total["stage_durations"]:
            for stage, s in it2i_total["stage_durations"].items():
                print(
                    f"    stage {stage:<18} n={s['count']:<5} "
                    f"mean={s['mean']:.4f} p50={s['p50']:.4f} p99={s['p99']:.4f}"
                )
        _print_text_stream(it2i_total.get("text_stream", {}), indent="    ")
        for bt, t in metrics["it2i_by_bot_task"].items():
            print("-" * 60)
            print(
                f"  [{bt}] total={t['total']} completed={t['completed']} "
                f"failed={t['failed']} success_rate={t['success_rate']:.2%} "
                f"qps={t['throughput_qps']:.2f}"
            )
            _print_latency_row(bt, t)
            if t["stage_durations"]:
                for stage, s in t["stage_durations"].items():
                    print(
                        f"    stage {stage:<18} n={s['count']:<5} "
                        f"mean={s['mean']:.4f} p50={s['p50']:.4f} p99={s['p99']:.4f}"
                    )
            _print_text_stream(t.get("text_stream", {}), indent="    ")

    print("\n" + "=" * 60)


def _endpoint_path(api_url: str) -> str:
    """Reduce a full request URL to its endpoint path for the input record."""
    for ep in (_EDITS_ENDPOINT, _CHAT_ENDPOINT):
        if api_url.endswith(ep):
            return ep
    # Fall back to everything after the host:port.
    idx = api_url.find("/v1/")
    return api_url[idx:] if idx >= 0 else api_url


def _collect_image_urls(body: Any) -> list[str]:
    """Pull ``data:`` image URLs from either a chat or an images-edits body."""
    urls: list[str] = []
    if not isinstance(body, dict):
        return urls
    choices = body.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            content = (choice or {}).get("message", {}).get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = (item.get("image_url") or {}).get("url", "")
                        if isinstance(url, str) and url.startswith("data:"):
                            urls.append(url)
    data_items = body.get("data", [])
    if isinstance(data_items, list):
        for d in data_items:
            if isinstance(d, dict) and d.get("b64_json"):
                urls.append(f"data:image/png;base64,{d['b64_json']}")
    return urls


def _build_request_records(
    requests_list: list[OmniRequest],
    outputs: list[RequestFuncOutput],
    output_dir: str | None,
) -> list[dict[str, Any]]:
    """Build a per-request input/output correlation list for the JSON output.

    Each record pairs the *input* the benchmark sent (endpoint, prompt, input
    image path, model, sampling params, bot_task) with the *output* the server
    returned (returned text for i2t, saved image paths + AR "thinking" text for
    t2i/it2i, stage durations, peak memory, latency, error). When ``output_dir``
    is given everything lands under it in a fixed layout for easy side-by-side
    review:

        <output_dir>/
            result.json                           (written by the caller)
            inputs/req_{idx}_{task}_input{ext}    (the reference image sent)
            outputs/req_{idx}_{task}_{img}.png    (the generated image returned)

    it2i requests thus have both an ``inputs/`` entry (the reference) and an
    ``outputs/`` entry (the edit), so the before/after can be compared directly;
    i2t has only an input image, t2i only an output image. Filenames embed the
    request index so input <-> output files line up across the two folders.
    """
    import base64
    import shutil

    inputs_dir = None
    outputs_dir = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        inputs_dir = os.path.join(output_dir, "inputs")
        outputs_dir = os.path.join(output_dir, "outputs")
        os.makedirs(inputs_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)

    records: list[dict[str, Any]] = []
    saved_input_total = 0
    saved_output_total = 0
    for idx, (req, out) in enumerate(zip(requests_list, outputs)):
        extra = req.extra_body or {}

        # Save the input (reference) image(s) for i2t / it2i — copy the source
        # file verbatim into <output_dir>/inputs/ so the exact image that was
        # sent is reviewable alongside the result. t2i has no input image.
        saved_input_paths: list[str] = []
        if inputs_dir and req.image_paths:
            for k, src in enumerate(req.image_paths):
                if not src or not os.path.isfile(src):
                    continue
                ext = os.path.splitext(src)[1] or ".png"
                suffix = f"input{k}" if len(req.image_paths) > 1 else "input"
                fname = f"req_{idx:04d}_{req.task_type}_{suffix}{ext}"
                dst = os.path.join(inputs_dir, fname)
                try:
                    shutil.copyfile(src, dst)
                    saved_input_paths.append(dst)
                    saved_input_total += 1
                except Exception:
                    pass

        input_record: dict[str, Any] = {
            "endpoint": _endpoint_path(req.api_url),
            "model": req.model,
            "prompt": req.prompt,
            "prompt_word_count": len(req.prompt.split()) if req.prompt else 0,
            "image_paths": list(req.image_paths) if req.image_paths else [],
            "input_images_saved": saved_input_paths,
            "input_image_size": req.input_image_size,
            "modalities": extra.get("modalities"),
            "width": req.width,
            "height": req.height,
            "num_inference_steps": req.num_inference_steps,
            "seed": req.seed,
        }
        if req.task_type == "it2i":
            input_record["bot_task"] = req.bot_task

        body = out.response_body if out.success else None
        returned_text: str | None = None
        cot_output: str | None = None
        num_images = 0
        saved_paths: list[str] = []

        if isinstance(body, dict):
            is_edits = _endpoint_path(req.api_url) == _EDITS_ENDPOINT
            if is_edits:
                cot_output, num_images = extract_edits_outputs(body)
            else:
                returned_text, cot_output, num_images = extract_chat_outputs(body)

            # Save generated images for t2i / it2i (i2t returns text, no image).
            if outputs_dir and req.task_type != "i2t":
                urls = _collect_image_urls(body)
                for img_idx, url in enumerate(urls):
                    if "," not in url:
                        continue
                    try:
                        _, b64 = url.split(",", 1)
                        fname = f"req_{idx:04d}_{req.task_type}_{img_idx}.png"
                        path = os.path.join(outputs_dir, fname)
                        with open(path, "wb") as f:
                            f.write(base64.b64decode(b64))
                        saved_paths.append(path)
                        saved_output_total += 1
                    except Exception:
                        pass

        output_record: dict[str, Any] = {
            "success": bool(out.success),
            "latency_s": round(out.latency, 4) if out.latency else 0.0,
            "returned_text": returned_text,
            "cot_output": cot_output,
            "num_images": num_images,
            "image_paths_saved": saved_paths,
            "stage_durations": out.stage_durations or {},
            "peak_memory_mb": out.peak_memory_mb,
            "error": out.error or None,
        }
        # Streaming text-output latency (present only when --stream captured a
        # text stream for this request; zeros / empty otherwise). generated_text
        # mirrors returned_text (i2t) / cot_output (it2i edits) but is the raw
        # streamed concatenation, kept for direct TTFT/TPOT cross-check.
        if out.ttft > 0.0:
            output_record["text_stream"] = {
                "ttft_s": round(out.ttft, 4),
                "output_tokens": out.output_tokens,
                "tpot_s": round(sum(out.itl or []) / (out.output_tokens - 1), 4) if out.output_tokens > 1 else 0.0,
                "itl_s": [round(x, 4) for x in (out.itl or [])],
                "generated_text": out.generated_text or None,
            }

        records.append(
            {
                "index": idx,
                "task_type": req.task_type,
                "input": input_record,
                "output": output_record,
            }
        )

    if output_dir:
        print(f"Saved {saved_input_total} input image(s) to {inputs_dir}.")
        print(f"Saved {saved_output_total} generated image(s) to {outputs_dir}.")
    return records


async def iter_requests(
    requests_list: list[OmniRequest],
    request_rate: float,
) -> AsyncGenerator[OmniRequest, None]:
    """Yield requests with Poisson inter-arrival when rate is finite; else burst."""
    if request_rate != float("inf") and request_rate <= 0:
        raise ValueError(f"request_rate must be positive or inf, got {request_rate}.")
    for i, req in enumerate(requests_list):
        if request_rate != float("inf") and i > 0:
            await asyncio.sleep(random.expovariate(request_rate))
        yield req


async def _run_warmups(
    requests_list: list[OmniRequest],
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
) -> None:
    if not args.warmup_requests or not requests_list:
        return
    n = min(args.warmup_requests, len(requests_list))
    warm = requests_list[:n]
    sem = asyncio.Semaphore(min(args.warmup_concurrency, n))

    async def one(req: OmniRequest) -> RequestFuncOutput:
        func = select_request_func(req, args.it2i_endpoint, args.stream)
        async with sem:
            return await func(req, session, None)

    print(f"Running {n} warmup request(s) (concurrency={args.warmup_concurrency})...")
    await asyncio.gather(*[one(r) for r in warm])


async def benchmark(args: argparse.Namespace) -> None:
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    chat_url = f"{base_url}{_CHAT_ENDPOINT}"
    images_edits_url = f"{base_url}{_EDITS_ENDPOINT}"

    counts = TaskCounts(i2t=args.num_i2t, t2i=args.num_t2i, it2i=args.num_it2i)
    if counts.total == 0:
        raise ValueError("At least one of --num-i2t/--num-t2i/--num-it2i must be > 0.")

    bot_task_weights = parse_bot_task_weights(args.it2i_bot_task_weights)
    group_order = _parse_group_order(args.no_shuffle_order)
    gen_resolution_weights = parse_gen_resolution_weights(args.gen_resolution_weights)

    config = OmniConfig(
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        model=args.model,
    )
    prompts = {
        "i2t": args.i2t_prompt,
        "t2i": args.t2i_prompt,
        "it2i": args.it2i_prompt,
    }

    requests_list = build_requests(
        counts=counts,
        config=config,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        bot_task_weights=bot_task_weights,
        input_image=args.input_image,
        prompts=prompts,
        chat_url=chat_url,
        images_edits_url=images_edits_url,
        return_stage_metrics=args.return_stage_metrics,
        it2i_endpoint=args.it2i_endpoint,
        seed=args.seed,
        shuffle=args.shuffle,
        bot_task_sampling=args.it2i_bot_task_sampling,
        group_order=group_order,
        gen_resolution_weights=gen_resolution_weights,
        gen_resolution_sampling=args.gen_resolution_sampling,
        randomize_input=args.randomize_input,
    )

    # Unified on-disk output directory (new --output-dir, or the deprecated
    # --save-dir / --output-file aliases). Resolved once here so both the
    # dry-run preview and the post-run save share the same target.
    output_dir = _resolve_output_dir(args)

    # Tally the actual mix (after merge/shuffle) for the dry-run preview / log.
    tally = {t: 0 for t in _TASK_TYPES}
    bt_tally: dict[str, int] = {}
    res_tally: dict[str, int] = {}
    in_res_tally: dict[str, int] = {}
    prompt_words: dict[str, list[int]] = {t: [] for t in _TASK_TYPES}
    for r in requests_list:
        tally[r.task_type] += 1
        if r.task_type == "it2i" and r.bot_task:
            bt_tally[r.bot_task] = bt_tally.get(r.bot_task, 0) + 1
        if r.task_type in ("t2i", "it2i") and r.width and r.height:
            key = f"{r.width}x{r.height}"
            res_tally[key] = res_tally.get(key, 0) + 1
        if r.input_image_size:
            in_res_tally[r.input_image_size] = in_res_tally.get(r.input_image_size, 0) + 1
        prompt_words[r.task_type].append(len(r.prompt.split()) if r.prompt else 0)

    print(f"Prepared {len(requests_list)} omni requests: " + ", ".join(f"{t}={tally[t]}" for t in _TASK_TYPES))
    if bt_tally:
        print("it2i bot_task used: " + ", ".join(f"{k}={v}" for k, v in sorted(bt_tally.items())))
    if res_tally:
        print("t2i/it2i resolution used: " + ", ".join(f"{k}={v}" for k, v in sorted(res_tally.items())))
    if args.randomize_input:
        pw_summary = ", ".join(
            f"{t}=[{min(prompt_words[t])}-{max(prompt_words[t])}, avg="
            f"{(sum(prompt_words[t]) / len(prompt_words[t])):.1f}]"
            for t in _TASK_TYPES
            if prompt_words[t]
        )
        print(f"prompt word count ({'randomized' if args.randomize_input else 'fixed'}): {pw_summary}")
        if in_res_tally:
            print(
                "i2t/it2i input-image resolution used: "
                + ", ".join(f"{k}={v}" for k, v in sorted(in_res_tally.items()))
            )

    if args.dry_run:
        preview = args.dry_run_preview
        if preview == 0 or preview >= len(requests_list):
            shown = requests_list
            shown_label = f"all {len(requests_list)}"
        else:
            shown = requests_list[:preview]
            shown_label = f"first {preview}"
        print(
            f"\n[dry-run] {shown_label} requests in send order "
            f"(idx, task, bot_task, inWxH, genWxH, prompt_words, api_url):"
        )
        for i, r in enumerate(shown):
            in_wxh = r.input_image_size or "-"
            gen_wxh = f"{r.width or '-'}x{r.height or '-'}" if r.width and r.height else "-"
            pw = len(r.prompt.split()) if r.prompt else 0
            print(
                f"  {i:>3}  {r.task_type:<5}  bot_task={r.bot_task or '-'}  "
                f"{in_wxh:<10}  {gen_wxh:<10}  {pw:<4}  {r.api_url}"
            )
        footer = (
            f"\n[dry-run] No requests sent. Shuffle={args.shuffle}, seed={args.seed}, group_order={list(group_order)}."
        )
        if gen_resolution_weights:
            footer += (
                f"\n[dry-run] gen_resolution_weights={gen_resolution_weights}, sampling={args.gen_resolution_sampling}."
            )
        if args.randomize_input:
            footer += (
                f"\n[dry-run] randomize_input=True (prompt length + input-image size + "
                f"content randomized per request via seed={args.seed})."
            )
        if output_dir:
            footer += f"\n[dry-run] output_dir={output_dir} (result.json + inputs/ + outputs/ would be written here)."
        print(footer)
        return

    # Pre-read + pre-encode input images so the senders skip file I/O / base64
    # at send time. Done outside the timed window so it neither blocks the event
    # loop under concurrency nor distorts latency — and, with ordered dispatch
    # below, makes the actual send order match the dry-run order exactly instead
    # of being raced by per-task prep cost (t2i < it2i < i2t prep).
    prepped = 0
    for r in requests_list:
        if r.image_paths:
            prepare_request_images(r)
            prepped += 1
    if prepped:
        print(f"Pre-encoded {prepped} request image(s) before send.")

    semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
    # Order in which requests actually reach the sender (after semaphore
    # acquisition). With pre-encoded payloads + the ordered-dispatch loop below
    # this equals the list / dry-run order; surfacing it lets the user tell
    # client-send order apart from any server-side re-dispatch (e.g. the DTPS
    # scheduler processing ar_downstream before ar_only).
    dispatch_order: list[int] = []

    async def limited(req: OmniRequest, idx: int, pbar: tqdm) -> RequestFuncOutput:
        func = select_request_func(req, args.it2i_endpoint, args.stream)
        if semaphore:
            async with semaphore:
                dispatch_order.append(idx)
                return await func(req, session, pbar)
        dispatch_order.append(idx)
        return await func(req, session, pbar)

    pbar = tqdm(total=len(requests_list), disable=args.disable_tqdm)
    async with aiohttp.ClientSession() as session:
        await _run_warmups(requests_list, args, session)
        start = time.perf_counter()
        tasks = []
        idx = 0
        async for req in iter_requests(requests_list, args.request_rate):
            tasks.append(asyncio.create_task(limited(req, idx, pbar)))
            idx += 1
            # Yield to the loop so the just-scheduled task acquires the
            # semaphore and reaches session.post (the request is pre-encoded,
            # so there is no prep delay) before the next task is created. With
            # the FIFO semaphore this makes the server arrival order match the
            # list / dry-run order even under --request-rate inf + concurrency.
            await asyncio.sleep(0.1)
        outputs = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start
    pbar.close()

    # Audit trail: the order the client actually dispatched requests to the
    # sender. With pre-encoding + ordered dispatch this is the list / dry-run
    # order (0,1,2,...). Compare against the server-side receive / processing
    # log to attribute any reordering to the DTPS scheduler (which processes
    # ar_downstream t2i/it2i ahead of ar_only i2t) rather than the client.
    dispatch_in_order = dispatch_order == list(range(len(requests_list)))
    if not dispatch_in_order:
        # With a finite --request-rate the loop awaits between sends, so the
        # list order still holds; only a hand-edited scheduler / semaphore
        # would break it. Log the deviation so the user can investigate.
        print(f"Dispatch order deviates from list order (len={len(dispatch_order)}).")
    print(f"Client dispatch order matches dry-run list order: {dispatch_in_order}")

    metrics = calculate_metrics(requests_list, outputs, total_duration, args.percentiles)
    metrics["config"] = {
        "output_dir": output_dir,
        "endpoint_i2t_t2i": _CHAT_ENDPOINT,
        "endpoint_it2i": _EDITS_ENDPOINT if args.it2i_endpoint == "images-edits" else _CHAT_ENDPOINT,
        "model": args.model,
        "dataset": args.dataset,
        "mix_i2t_t2i_it2i": [args.num_i2t, args.num_t2i, args.num_it2i],
        "bot_task_weights": bot_task_weights,
        "bot_task_sampling": args.it2i_bot_task_sampling,
        "bot_task_actual": bt_tally,
        "group_order": list(group_order),
        "gen_resolution_weights": gen_resolution_weights,
        "gen_resolution_sampling": args.gen_resolution_sampling,
        "gen_resolution_actual": res_tally,
        "randomize_input": bool(args.randomize_input),
        "input_resolution_actual": in_res_tally,
        "prompt_word_count": {
            t: {
                "min": min(prompt_words[t]) if prompt_words[t] else 0,
                "max": max(prompt_words[t]) if prompt_words[t] else 0,
                "avg": round(sum(prompt_words[t]) / len(prompt_words[t]), 2) if prompt_words[t] else 0.0,
            }
            for t in _TASK_TYPES
            if prompt_words[t]
        },
        "shuffle": args.shuffle,
        "seed": args.seed,
        "request_rate": args.request_rate,
        "max_concurrency": args.max_concurrency,
        "stream": bool(args.stream),
        "dispatch_order": dispatch_order,
        "dispatch_matches_list_order": dispatch_order == list(range(len(requests_list))),
    }

    print_metrics(metrics, args)

    # Per-request input/output correlation, also saving input reference images
    # to <output_dir>/inputs/ and generated images to <output_dir>/outputs/.
    # Surfaced in the JSON as ``requests`` so each request's inputs (endpoint,
    # prompt, image path, sampling params, bot_task) line up with its outputs
    # (returned text / saved image paths / AR thinking text). The metrics JSON
    # is always written to <output_dir>/result.json (fixed name).
    if output_dir:
        metrics["requests"] = _build_request_records(requests_list, outputs, output_dir)
        result_path = os.path.join(output_dir, "result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to {result_path}")


def _argparse_float_inf(text: str) -> float:
    if text.strip().lower() == "inf":
        return float("inf")
    return float(text)


def _resolve_output_dir(args: argparse.Namespace) -> str | None:
    """Resolve the unified on-disk output directory.

    ``--output-dir`` is the single knob: ``result.json`` + ``inputs/`` +
    ``outputs/`` all land under it. The deprecated ``--save-dir`` maps to the
    same directory; ``--output-file`` maps to its parent directory (the filename
    is no longer honored — the JSON is fixed as ``result.json``). Precedence:
    ``--output-dir`` > ``--save-dir`` > ``--output-file``.
    """
    if args.output_dir:
        if args.save_dir or args.output_file:
            print(
                "Warning: --output-dir takes precedence; --save-dir / --output-file "
                "are ignored (deprecated, use --output-dir)."
            )
        return args.output_dir
    if args.save_dir:
        print(
            "Warning: --save-dir is deprecated; use --output-dir. Output now includes "
            "result.json + inputs/ + outputs/ under this directory."
        )
        return args.save_dir
    if args.output_file:
        parent = os.path.dirname(args.output_file) or "."
        print(
            "Warning: --output-file is deprecated; use --output-dir. The JSON is now "
            f"fixed as {os.path.join(parent, 'result.json')} (the given filename is ignored)."
        )
        return parent
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Omni (i2t/t2i/it2i) serving benchmark for unified AR+DiT models.",
    )
    parser.add_argument("--base-url", type=str, default=None, help="Server base URL (overrides --host/--port).")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8091)

    # Task mix — explicit per-type counts.
    parser.add_argument("--num-i2t", type=int, default=0, help="Number of i2t (understanding) requests.")
    parser.add_argument("--num-t2i", type=int, default=0, help="Number of t2i (generation) requests.")
    parser.add_argument("--num-it2i", type=int, default=0, help="Number of it2i (editing) requests.")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="(Legacy) total prompt count; ignored when any --num-* is set. Kept for compat.",
    )

    # it2i knobs.
    parser.add_argument(
        "--it2i-endpoint",
        choices=["chat", "images-edits"],
        default="chat",
        help="Endpoint for it2i traffic: 'chat' (/v1/chat/completions, unified) or 'images-edits'.",
    )
    parser.add_argument(
        "--it2i-bot-task-weights",
        type=str,
        default=None,
        help="Weighted bot_task mix for it2i, e.g. 'recaption=2,think=1,think_recaption=1'. "
        "Defaults to equal weight across recaption/think/think_recaption.",
    )
    parser.add_argument(
        "--it2i-bot-task-sampling",
        choices=["proportional", "random"],
        default="proportional",
        help="How to turn --it2i-bot-task-weights into per-request bot_task. 'proportional' "
        "(default) allocates exact counts matching the weights (largest-remainder, e.g. "
        "10 tasks at 2:2:1 -> 4/4/2). 'random' draws each task independently with the "
        "weights, matching them only in expectation (e.g. may yield 6/3/1).",
    )

    # Random send order.
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle the merged request list so task types interleave (true random send order). "
        "Use --no-shuffle to send grouped by task type.",
    )
    parser.add_argument(
        "--no-shuffle-order",
        type=str,
        default=None,
        metavar="ORDER",
        help="With --no-shuffle, the order to concatenate the i2t/t2i/it2i groups before "
        "sending, e.g. 't2i,it2i,i2t' sends all t2i, then it2i, then i2t. Must be a "
        "permutation of i2t,t2i,it2i. Default: i2t,t2i,it2i. Ignored when --shuffle is on "
        "(the merge is then randomized). The actual send order always matches this list "
        "order — see the send-order fix in README.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for shuffle + bot_task + resolution sampling.")

    # Dataset.
    parser.add_argument(
        "--dataset",
        choices=["random", "custom"],
        default="random",
        help="'random' uses synthetic prompts + a placeholder image; 'custom' reads a JSONL via --dataset-path.",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="JSONL file for --dataset custom.")
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Input image for i2t/it2i (random mode). Defaults to a generated placeholder.",
    )

    # Prompts (random mode).
    parser.add_argument("--i2t-prompt", type=str, default=None)
    parser.add_argument("--t2i-prompt", type=str, default=None)
    parser.add_argument("--it2i-prompt", type=str, default=None)

    # Generation knobs.
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed-gen", type=int, default=None, help="Generation seed for t2i/it2i outputs.")
    parser.add_argument(
        "--gen-resolution-weights",
        type=str,
        default=None,
        metavar="WEIGHTS",
        help="Generation-resolution ratio for t2i/it2i, e.g. "
        "'512x512=2,1024x1024=2,1280x720=1'. Each of t2i and it2i is allocated "
        "resolutions per the ratio (overrides --width/--height per request; a custom "
        "dataset row's explicit width/height still wins). Allowed: "
        f"{sorted(GEN_RESOLUTIONS)}. Default: none (use --width/--height for every request).",
    )
    parser.add_argument(
        "--gen-resolution-sampling",
        choices=["proportional", "random"],
        default="proportional",
        help="How to turn --gen-resolution-weights into per-request resolutions. 'proportional' "
        "(default) allocates exact counts matching the weights (largest-remainder, e.g. "
        "10 requests at 2:2:1 -> 4/4/2). 'random' draws each request independently, matching "
        "the weights only in expectation.",
    )
    parser.add_argument(
        "--randomize-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomize the *input* side of every request to approximate real-world request "
        "diversity (governed by --seed, so reproducible). For all three task types the prompt "
        "is drawn from a short->long pool so prompt lengths vary; for i2t / it2i a unique "
        "varied-content input image is generated at a size drawn from 512x512 / 1024x1024 / "
        "1280x720 (not one shared solid-color placeholder). Only affects the 'random' dataset "
        "(--dataset custom rows always win); --input-image and --i2t/--t2i/--it2i-prompt are "
        "ignored while this is on. Default off (single placeholder image + fixed prompts).",
    )

    # Traffic.
    parser.add_argument(
        "--request-rate",
        type=_argparse_float_inf,
        default=float("inf"),
        help="Requests per second (Poisson). 'inf' sends all at once. Default inf.",
    )
    parser.add_argument("--max-concurrency", type=int, default=1, help="Max in-flight requests.")
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Number of warmup requests to send before the timed run (default 0 = no warmup). "
        "Warmup results are discarded; set > 0 to prime caches / JIT before measuring.",
    )
    parser.add_argument("--warmup-concurrency", type=int, default=1)

    # Output / metrics.
    parser.add_argument("--return-stage-metrics", action="store_true", help="Ask the server for per-stage durations.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream responses and capture text-output TTFT / TPOT / ITL (vLLM-style). "
        "i2t measures the AR final-answer text; it2i (via --it2i-endpoint images-edits) "
        "measures the AR 'thinking' text streamed ahead of the image. chat t2i/it2i do not "
        "stream AR text (server-filtered), so their text metrics are N/A — use images-edits "
        "for it2i AR-text latency. Non-streaming runs report text_stream with count=0.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[50, 95, 99],
        help="Latency percentiles to report (default: 50 95 99).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Unified on-disk output directory. Writes result.json (metrics) at the root, "
        "saves the i2t/it2i input reference images under <DIR>/inputs/, and the t2i/it2i "
        "generated images under <DIR>/outputs/ — so an it2i request's reference and its "
        "edit can be compared side by side, and an i2t request's input image is reviewable. "
        "Replaces the old --output-file / --save-dir pair.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="(Deprecated, use --output-dir.) Legacy: wrote metrics JSON to this exact path. "
        "Now mapped to its parent directory with the JSON fixed as result.json.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="(Deprecated, use --output-dir.) Legacy: directory to save generated images. "
        "Now mapped to --output-dir (also writes result.json + inputs/).",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build the request plan and print it; send nothing.")
    parser.add_argument(
        "--dry-run-preview",
        type=int,
        default=20,
        metavar="N",
        help="With --dry-run, how many requests to print in send order (default: 20). Use 0 to print every request.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
