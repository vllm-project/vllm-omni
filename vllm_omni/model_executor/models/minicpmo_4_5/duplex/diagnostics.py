# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Stage-0 diagnostics; deferred copies share the sampler's commit event."""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
from vllm_omni.utils.device_copy import index_to_device

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def trace_start_unit() -> int | None:
    """Unset disables tracing; 0 traces all units. Read once per worker."""
    value = os.environ.get("VLLM_OMNI_MINICPMO_DUPLEX_TRACE_START_UNIT")
    return max(0, int(value)) if value is not None else None


def trace_enabled(state: object) -> bool:
    start = trace_start_unit()
    return start is not None and getattr(state, "audio_chunk_idx", 0) >= start


def turn_state(state: object) -> dict[str, object]:
    return {
        name: getattr(state, name, None)
        for name in (
            "current_turn_ended",
            "pending_terminator_token",
            "last_terminator_token",
        )
    }


def emit_trace(record: dict[str, object]) -> None:
    # INFO is deliberate: explicitly requested diagnostics must be visible in
    # captured worker logs without enabling unrelated debug output.
    logger.info("MINICPMO_DUPLEX_TRACE %s", json.dumps(record, ensure_ascii=False, allow_nan=False))


class SamplingTrace:
    """Small host snapshots plus top-5 logits copied before the existing commit event.

    Deferred callers must wait for their normal pending-sample event before
    emitting. Synchronous callers copy at emit, after their normal draws.tolist().
    """

    def __init__(self, token_ids: dict[str, int]):
        self.token_ids = token_ids
        self.records: dict[int, dict[str, object]] = {}
        self.logits: list[tuple[str, list[int], torch.Tensor, int, list[str]]] = []

    def capture_logits(self, name: str, logits: torch.Tensor, rows: list[int]) -> None:
        selected = [(local, row) for local, row in enumerate(rows) if row in self.records]
        if not selected:
            return
        local, traced_rows = zip(*selected, strict=True)
        values = logits[index_to_device(local, logits.device)]
        k = min(5, values.shape[-1])
        top = values.topk(k, dim=-1)
        controls = [key for key, token in self.token_ids.items() if 0 <= token < values.shape[-1]]
        control_ids = index_to_device([self.token_ids[key] for key in controls], logits.device)
        packed = torch.cat((top.indices.float(), top.values.float(), values[:, control_ids].float()), dim=-1)
        self.logits.append((name, list(traced_rows), packed, k, controls))

    def copy_to_host(self) -> None:
        """Queue copies before the deferred sampler records its existing event."""
        for i, (name, rows, packed, k, controls) in enumerate(self.logits):
            host = torch.empty(packed.shape, dtype=packed.dtype, pin_memory=packed.device.type == "cuda")
            host.copy_(packed, non_blocking=True)
            self.logits[i] = (name, rows, host, k, controls)

    def decision(self, row: int, final: int, raw: int | None, hit: bool, stage2: bool) -> None:
        if row not in self.records:
            return
        if not stage2:
            reason = "chunk_cap"
        elif hit:
            reason = "chunk_boundary"
        elif raw == self.token_ids.get("turn_eos_token_id"):
            reason = "accept_turn_eos"
        elif final != raw:
            reason = "listen_to_tts_bos"
        else:
            reason = "sample"
        self.records[row].update(
            final_token_id=final,
            # Deferred stage-2 draws on a boundary hit are speculative and
            # rewound, not decisions. Match the synchronous trace's null.
            stage2_token_id=raw if stage2 and not hit else None,
            boundary_hit=hit,
            reason=reason,
        )

    def emit(self, model: MiniCPMO45OmniForConditionalGeneration) -> None:
        for name, rows, host, k, controls in self.logits:
            for row, values in zip(rows, host.cpu().tolist(), strict=True):
                if name == "stage2_logits" and self.records[row].get("boundary_hit"):
                    continue  # No committed stage-2 decision on this row.
                scores = [value if math.isfinite(value) else None for value in values[k:]]
                self.records[row][name] = {
                    "top_ids": [int(value) for value in values[:k]],
                    "top_scores": scores[:k],
                    "control_scores": dict(zip(controls, scores[k:], strict=True)),
                }
        for row, record in self.records.items():
            record["after"] = turn_state(model._minicpmo45_duplex_state_for_row(row))
            emit_trace(record)


def start_sampling_trace(
    model: MiniCPMO45OmniForConditionalGeneration,
    logits: torch.Tensor,
    rows: list[int],
    recent: dict[int, list[int]],
    token_ids: dict[str, int],
    row_params: list[tuple[float, float, float]],
    all_greedy: bool,
) -> SamplingTrace | None:
    if trace_start_unit() is None:
        return None
    trace = SamplingTrace(token_ids)
    for row in rows:
        state = model._minicpmo45_duplex_state_for_row(row)
        if not trace_enabled(state):
            continue
        payload = model._minicpmo45_duplex_payload_for_row(row) or {}
        trace.records[row] = {
            "event": "sample",
            "session_id": getattr(state, "session_id", None),
            "append_identity": getattr(state, "prepared_append_identity", None),
            "unit_count": getattr(state, "audio_chunk_idx", 0),
            "row": row,
            "chunk_token_ids": list(recent[row]),
            "repetition_token_ids": list(getattr(state, "generated_tokens", None) or recent[row])[
                -MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE :
            ],
            "force_listen": payload.get("force_listen") is True,
            "all_greedy": all_greedy,
            "sampling_params": dict(zip(("temperature", "top_k", "top_p"), row_params[row], strict=True)),
            "chunk_limits": {
                "max_tokens": model._minicpmo45_policy_limit(
                    "max_new_speak_tokens_per_chunk", MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK
                ),
                "max_chars": model._minicpmo45_policy_limit(
                    "max_speak_chars_per_chunk", MiniCPMO45DuplexPolicy.DEFAULT_MAX_SPEAK_CHARS_PER_CHUNK
                ),
                "request_max_tokens": model._minicpmo45_duplex_row_request_max_tokens(row),
            },
            "special_token_ids": token_ids,
            "before": turn_state(state),
        }
    if not trace.records:
        return None
    trace.capture_logits("model_logits", logits, list(range(logits.shape[0])))
    return trace
