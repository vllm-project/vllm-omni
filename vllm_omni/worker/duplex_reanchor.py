# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared V1 duplex reanchor dispatch; no backend runner imports."""

from typing import Any


def apply_duplex_window_reanchor(runner: Any, scheduler_output: Any = None) -> None:
    """Run after scheduler/sampling updates, before forward."""
    reanchor_hook = getattr(getattr(runner, "model", None), "apply_duplex_kv_reanchor", None)
    if callable(reanchor_hook):
        try:
            reanchor_hook(runner, scheduler_output=scheduler_output)
        except TypeError:
            reanchor_hook(runner)
        return

    helper = getattr(runner, "_duplex_window_helper", None)
    if helper is not None and hasattr(helper, "maybe_apply_reanchor"):
        try:
            helper.maybe_apply_reanchor(runner, scheduler_output=scheduler_output)
        except TypeError:
            helper.maybe_apply_reanchor(runner)
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
