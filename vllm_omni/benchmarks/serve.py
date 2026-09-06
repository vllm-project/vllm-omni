# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from pathlib import Path
from typing import Any

from vllm.benchmarks.serve import main_async

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

_ENDPOINT_BACKEND_KEYS = {
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/videos",
}


def _normalize_endpoint(endpoint: str | None) -> str | None:
    if endpoint is None:
        return None
    endpoint = str(endpoint).strip()
    if not endpoint:
        return None
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return endpoint


def _use_endpoint_backend_when_implicit(args: argparse.Namespace) -> None:
    explicit_keys = getattr(args, "explicit_keys", frozenset())
    if "backend" in explicit_keys:
        return
    # Upstream vLLM defaults --backend to "openai". Treat that non-explicit
    # default the same as an empty backend so endpoint-driven omni handlers work.
    backend = getattr(args, "backend", None)
    if backend not in (None, "", "openai"):
        return
    endpoint = _normalize_endpoint(getattr(args, "endpoint", None))
    if endpoint in _ENDPOINT_BACKEND_KEYS:
        args.backend = endpoint


def main(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "seed_tts_wer_eval", False):
        os.environ["SEED_TTS_WER_EVAL"] = "1"
    if getattr(args, "seed_tts_wer_save_items", False):
        os.environ["SEED_TTS_WER_SAVE_ITEMS"] = "1"
    if getattr(args, "daily_omni_save_eval_items", False):
        os.environ["DAILY_OMNI_SAVE_EVAL_ITEMS"] = "1"
    _use_endpoint_backend_when_implicit(args)
    set_print_stage(getattr(args, "print_stage", False))
    args.extra_body = maybe_enable_stage_metrics(
        getattr(args, "extra_body", None),
        enabled=should_request_stage_metrics(args),
    )
    lock = (
        omniinteract_output_lock(Path(args.omniinteract_output_dir))
        if getattr(args, "dataset_name", None) == "omniinteract"
        else contextlib.nullcontext()
    )
    with lock:
        return asyncio.run(main_async(args))
