# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import random
import ssl
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from vllm.benchmarks.serve import (
    TaskType,
    check_goodput_args,
    compute_result_filename,
    get_first_model_from_server,
    main_async,
)
from vllm.utils.gc_utils import freeze_gc_heap
from vllm.utils.network_utils import join_host_port

# Import patch to register daily-omni dataset and omni backends
# This monkey-patches vllm.benchmarks.datasets.get_samples before it's used
# Must be imported before any vllm.benchmarks module usage
import vllm_omni.benchmarks.patch.patch  # noqa: F401
from vllm_omni.benchmarks.omniinteract import omniinteract_output_lock
from vllm_omni.benchmarks.patch.patch import (
    maybe_enable_stage_metrics,
    set_print_stage,
    should_request_stage_metrics,
)

_DIFFUSION_ENDPOINTS = frozenset({
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/videos",
})

_DIFFUSION_BACKENDS = frozenset({
    "openai-image-gen-omni",
    "openai-image-edits-omni",
    "openai-video-omni",
})


def is_diffusion_benchmark(args) -> bool:
    """Return True when the benchmark targets a diffusion/image/video endpoint."""
    endpoint = getattr(args, "endpoint", None) or ""
    backend = getattr(args, "backend", None) or ""
    return endpoint in _DIFFUSION_ENDPOINTS or backend in _DIFFUSION_BACKENDS


async def _main_async_diffusion(args: argparse.Namespace) -> dict[str, Any]:
    """Benchmark entry point for diffusion models.

    Mirrors the essential setup from upstream ``main_async`` but skips
    tokenizer loading entirely — diffusion models have no tokenizer.
    Delegates to the same patched ``get_samples`` and ``benchmark``
    functions used by non-diffusion paths.
    """
    from vllm_omni.benchmarks.patch.patch import benchmark, get_samples

    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    label = getattr(args, "label", None)

    # URL construction (mirrors upstream lines 1927-1933)
    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        host_port = join_host_port(args.host, args.port)
        api_url = f"http://{host_port}{args.endpoint}"
        base_url = f"http://{host_port}"

    # Headers (mirrors upstream lines 1936-1944)
    headers = None
    if args.header:
        headers = {}
        for item in args.header:
            if "=" in item:
                kvstring = item.split("=", 1)
                headers[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError(
                    "Invalid header format. Please use KEY=VALUE format."
                )

    # SSL context (mirrors upstream lines 1946-1953)
    ssl_context: ssl.SSLContext | bool | None = None
    if args.insecure:
        ssl_context = False
    elif "https://" in base_url:
        ssl_context = True

    # Model resolution (mirrors upstream lines 1956-1964)
    if args.model is None:
        print("Model not specified, fetching first model from server...")
        model_name, model_id = await get_first_model_from_server(
            base_url, headers, ssl_context
        )
        print(f"First model name: {model_name}, first model id: {model_id}")
    else:
        model_name = getattr(args, "served_model_name", args.model)
        model_id = args.model

    # No tokenizer for diffusion models
    tokenizer = None

    # Load diffusion prompts
    input_requests = get_samples(args, tokenizer)

    goodput_config_dict = check_goodput_args(args)
    extra_body = args.extra_body or {}

    percentile_metrics: str = (
        getattr(args, "percentile_metrics", None) or "ttft,tpot,itl"
    )

    freeze_gc_heap()

    benchmark_result = await benchmark(
        task_type=TaskType.GENERATION,
        endpoint_type=args.backend,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        tokenizer=None,
        input_requests=input_requests,
        logprobs=getattr(args, "logprobs", None),
        request_rate=args.request_rate,
        burstiness=getattr(args, "burstiness", 1.0),
        disable_tqdm=args.disable_tqdm,
        num_warmups=getattr(args, "num_warmups", 1),
        profile=False,
        selected_percentile_metrics=percentile_metrics.split(","),
        selected_percentiles=[
            float(p)
            for p in getattr(args, "metric_percentiles", "99").split(",")
        ],
        ignore_eos=False,
        goodput_config_dict=goodput_config_dict,
        max_concurrency=getattr(args, "max_concurrency", None),
        lora_modules=None,
        extra_headers=headers,
        extra_body=extra_body,
        ready_check_timeout_sec=getattr(args, "ready_check_timeout_sec", 600),
        ssl_context=ssl_context,
    )

    # Save config and results to JSON (mirrors upstream lines 2127-2280)
    result_json: dict[str, Any] = {}
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["backend"] = args.backend
    result_json["label"] = label
    result_json["model_id"] = model_id
    result_json["tokenizer_id"] = None
    result_json["num_prompts"] = args.num_prompts

    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                kvstring = item.split("=", 1)
                result_json[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError(
                    "Invalid metadata format. Please use KEY=VALUE format."
                )

    result_json["request_rate"] = (
        args.request_rate if args.request_rate < float("inf") else "inf"
    )
    result_json["burstiness"] = getattr(args, "burstiness", 1.0)
    result_json["max_concurrency"] = getattr(args, "max_concurrency", None)

    result_json = {**result_json, **benchmark_result}

    file_name = compute_result_filename(args, model_id, label, current_dt)

    if not getattr(args, "save_detailed", False):
        for field in [
            "input_lens",
            "output_lens",
            "start_times",
            "ttfts",
            "itls",
            "generated_texts",
            "errors",
        ]:
            result_json.pop(field, None)
            benchmark_result.pop(field, None)

    if getattr(args, "save_result", False) or getattr(
        args, "append_result", False
    ):
        if file_name is None:
            warnings.warn(
                "Cannot save results: file_name is None", stacklevel=2
            )
        else:
            with open(
                file_name,
                mode="a+" if args.append_result else "w",
                encoding="utf-8",
            ) as outfile:
                json.dump(result_json, outfile, indent=2, default=str)

    return result_json


def main(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "seed_tts_wer_eval", False):
        os.environ["SEED_TTS_WER_EVAL"] = "1"
    if getattr(args, "seed_tts_wer_save_items", False):
        os.environ["SEED_TTS_WER_SAVE_ITEMS"] = "1"
    if getattr(args, "daily_omni_save_eval_items", False):
        os.environ["DAILY_OMNI_SAVE_EVAL_ITEMS"] = "1"
    set_print_stage(getattr(args, "print_stage", False))
    args.extra_body = maybe_enable_stage_metrics(
        getattr(args, "extra_body", None),
        enabled=should_request_stage_metrics(args),
    )

    if is_diffusion_benchmark(args):
        return asyncio.run(_main_async_diffusion(args))

    lock = (
        omniinteract_output_lock(Path(args.omniinteract_output_dir))
        if getattr(args, "dataset_name", None) == "omniinteract"
        else contextlib.nullcontext()
    )
    with lock:
        return asyncio.run(main_async(args))
