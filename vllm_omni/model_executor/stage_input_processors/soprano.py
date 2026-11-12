# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline continuous predictor-state handoff; token IDs are not codec codes."""

from collections.abc import Mapping
from time import perf_counter_ns

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, HiddenStatesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.soprano.contracts import select_predictors

logger = init_logger(__name__)


def ar_to_decoder_async_chunk(transfer_manager, multimodal_output, request, is_finished=False, new_token_ids=()):
    """Send bounded predictor deltas through the existing async connector.

    The AR runner copies scheduled CUDA rows to owned CPU storage before its
    tensor-only wire transport. The save thread sees request snapshots and
    immutable step token IDs. Only the final prefill row / current decode row
    predicts that step's sampled token. EOS excludes that predictor; length
    includes it. request_payload is reclaimed by the adapter's sender fence
    on finish/abort, before external request IDs can be reused.
    """
    key = request.external_req_id
    state = transfer_manager.request_payload.setdefault(key, {}).setdefault(
        "soprano", {"pending": [], "terminal": False, "error": None}
    )
    if state["terminal"]:
        return None
    config = transfer_manager.connector.config
    # Factory passes spec.extra directly to connector constructors; tests and
    # externally supplied connectors can retain the enclosing `extra` map.
    extra = config.get("extra", config)
    chunk_size = 4
    finished = bool(is_finished or request.is_finished())
    try:
        chunk_size = int(extra.get("soprano_chunk_size", 4))
        if not 1 <= chunk_size <= 64:
            raise ValueError("Soprano hidden chunk_size must be in 1..64")
        if len(new_token_ids) > 1:
            raise ValueError("Soprano streaming does not support speculative multi-token steps")
        if new_token_ids and new_token_ids[0] != 3 and state["error"] is None:
            hidden = multimodal_output.get("hidden") if isinstance(multimodal_output, Mapping) else None
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.device.type != "cpu"
                or hidden.ndim != 2
                or hidden.shape[1] != 512
                or hidden.shape[0] == 0
                or hidden.dtype != torch.float32
                or not torch.isfinite(hidden[-1]).all()
            ):
                raise ValueError("Missing owned CPU float32 Soprano predictor row")
            state["pending"].append(hidden[-1])
    except (ValueError, TypeError) as exc:
        state["error"] = state["error"] or str(exc)
        state["pending"].clear()
    if not finished and (state["error"] or len(state["pending"]) < chunk_size):
        return None
    rows = state["pending"]
    hidden = torch.stack(rows) if rows else torch.empty(0, 512, dtype=torch.float32)
    state["pending"] = []
    state["terminal"] = finished
    if finished:
        logger.debug("Soprano AR terminal req=%s monotonic_ns=%d", key, perf_counter_ns())
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor([0], dtype=torch.long)),
        hidden_states=HiddenStatesStruct(output=hidden, output_shape=list(hidden.shape)),
        meta=MetaStruct(
            stream_finished=torch.tensor(finished),
            replace_runtime_additional_information=True,
        ),
        additional_information={
            "soprano_stream": True,
            "soprano_chunk_size": chunk_size,
            "soprano_error": state["error"],
            "soprano_truncated": finished and bool(request.output_token_ids and request.output_token_ids[-1] != 3),
        },
    )


def ar_to_decoder_token_only(source_outputs, prompt=None, _requires_multimodal_data=False):
    """One control slot per complete waveform; real data comes via connector."""
    return [OmniTokensPrompt(prompt_token_ids=[0]) for output in source_outputs if output.finished]


def ar_to_decoder_full_payload(transfer_manager, pooling_output, request, is_finished=True):
    """Snapshot the complete consumed trajectory, then exclude EOS predictors.

    Validation errors cross the same full-payload path explicitly, rather than
    becoming a missing payload that leaves the decoder waiting forever.
    """
    try:
        if not isinstance(pooling_output, Mapping) or not isinstance(pooling_output.get("hidden"), torch.Tensor):
            raise ValueError("Missing Soprano hidden trajectory")
        sampled = list(request.output_token_ids)
        hidden = select_predictors(pooling_output["hidden"], len(request.prompt_token_ids), sampled)
        logger.info(
            "Soprano handoff req=%s prompt=%d sampled=%d retained=%d eos=%s",
            request.req_id,
            len(request.prompt_token_ids),
            len(sampled),
            hidden.shape[0],
            sampled[-1] == 3,
        )
        return {
            "hidden_states": {"output": hidden, "output_shape": list(hidden.shape)},
            "ids": {"output": sampled},
            "meta": {"finished": True},
        }
    except (ValueError, AttributeError) as exc:
        return {"additional_information": {"soprano_error": str(exc)}, "meta": {"finished": True}}
