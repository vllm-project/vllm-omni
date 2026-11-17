# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared V1 duplex reanchor dispatch; no backend runner imports."""

import inspect
from typing import Any


def _accepts_scheduler_output(hook: Any) -> bool:
    # Bind arguments without executing the hook. Internal TypeError must propagate:
    # a retry could apply a non-idempotent in-place rotation twice.
    # Do not cache callable instances or bound methods: they may be unhashable
    # and retaining them also retains the model across engine restarts.
    signature = inspect.signature(hook)
    try:
        signature.bind(object(), scheduler_output=None)
    except TypeError:
        signature.bind(object())
        return False
    return True


def _call_reanchor(hook: Any, runner: Any, scheduler_output: Any) -> None:
    if _accepts_scheduler_output(hook):
        hook(runner, scheduler_output=scheduler_output)
    else:
        hook(runner)


def validate_duplex_kv_cache(runner: Any) -> None:
    """Validate MiniCPM-o flattened cache views after allocation, before requests."""
    if "minicpmo_4_5" not in getattr(getattr(runner, "model", None), "__module__", ""):
        return
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
        MiniCPMO45DuplexWorkerHelper,
        flattened_key_cache_view,
    )

    caches = getattr(runner, "kv_caches", ())
    parameters = None
    for cache in caches:
        key = cache[0] if isinstance(cache, (tuple, list)) else cache
        if key.dim() != 3:
            continue
        if parameters is None:
            parameters = MiniCPMO45DuplexWorkerHelper.get_reanchor_parameters(runner)
        block_size, head_dim, inv_freq = parameters
        flattened_key_cache_view(key, block_size, head_dim, inv_freq.numel() * 2)


def apply_duplex_window_reanchor(runner: Any, scheduler_output: Any = None) -> None:
    """Run after scheduler/sampling updates, before forward."""
    reanchor_hook = getattr(getattr(runner, "model", None), "apply_duplex_kv_reanchor", None)
    if callable(reanchor_hook):
        _call_reanchor(reanchor_hook, runner, scheduler_output)
        return

    helper = getattr(runner, "_duplex_window_helper", None)
    if helper is not None and hasattr(helper, "maybe_apply_reanchor"):
        _call_reanchor(helper.maybe_apply_reanchor, runner, scheduler_output)
        return

    # Fallback for dynamic runner inspection without hardcoding model classes
    if not hasattr(runner, "input_batch") or runner.input_batch is None:
        return
    num_reqs = getattr(runner.input_batch, "num_reqs", 0)
    req_ids = runner.input_batch.req_ids[:num_reqs]
    has_reanchor = False
    for req_id in req_ids:
        info = runner.model_intermediate_buffer.get(req_id)
        if isinstance(info, dict) and isinstance(info.get("duplex"), dict):
            if "stage0_reanchor" in info["duplex"]:
                has_reanchor = True
                break
    if not has_reanchor:
        return

    model_module = getattr(getattr(runner, "model", None), "__module__", "")
    if "minicpmo_4_5" in model_module:
        from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
            MiniCPMO45DuplexWorkerHelper,
        )

        MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner, scheduler_output=scheduler_output)
