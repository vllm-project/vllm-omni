# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model Runner V2 binding of Mooncake Thinker capture publication.

The connector owns restored rows and save buffers. This bridge snapshots the
step's token positions before sampling advances them, and rewrites the two
Thinker capture fields before the ordinary Thinker-to-Talker handoff. It stays
on the worker side because it reads ``req_states``, writes per-row output
channels and needs ``is_warmup_request``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm_omni.worker_v2.warmup import is_warmup_request

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from vllm_omni.distributed.kv_transfer.mooncake.connector import OmniMooncakeStoreConnector
    from vllm_omni.outputs import OmniModelRunnerOutput
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

OutputFinalizer = Callable[["OmniModelRunnerOutput"], "OmniModelRunnerOutput"]


@dataclass(frozen=True)
class CaptureRequestContext:
    """Request identity and token positions for one pending output step."""

    row: int
    req_id: str
    attempt_id: str
    start_token: int
    prompt_len: int


class MooncakeOutputRewrite:
    """One step's captures; consume once even when payload channels alias."""

    def __init__(self, connector: OmniMooncakeStoreConnector, rows: tuple[CaptureRequestContext, ...]):
        self.connector = connector
        self.rows = rows
        self._consumed = False

    @staticmethod
    def _discard_row(output: OmniModelRunnerOutput, row: int) -> None:
        output.sampled_token_ids[row] = []
        for channel in (output.inter_stage_outputs, output.pooler_output, output.multimodal_outputs):
            if channel and row < len(channel):
                channel[row] = {}

    def apply(self, output: OmniModelRunnerOutput) -> None:
        if self._consumed:
            raise RuntimeError("Mooncake output step was already consumed")
        self._consumed = True
        keys = {"embedding": "hidden_states.layer_0", "hidden": f"hidden_states.layer_{self.connector.hidden_layer}"}
        channels = (output.inter_stage_outputs, output.pooler_output, output.multimodal_outputs)
        for entry in self.rows:
            if output.request_errors and entry.req_id in output.request_errors:
                continue
            payloads: list[dict[str, Any]] = []
            for channel in channels:
                payload = channel[entry.row] if channel and entry.row < len(channel) else None
                if isinstance(payload, dict) and payload:
                    payloads.append(payload)
            source = next((payload for payload in payloads if all(key in payload for key in keys.values())), None)
            try:
                if source is None:
                    raise ValueError("Missing Thinker captures for Mooncake")
                values = {name: source[key] for name, key in keys.items()}
                if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in values.values()):
                    raise ValueError("Mooncake output rewrite requires CPU Thinker captures")
                captures = cast(dict[str, torch.Tensor], values)
                merged = self.connector.capture_step(
                    entry.req_id, entry.attempt_id, entry.start_token, entry.prompt_len, captures
                )
            except ValueError as exc:
                # Restored objects are validated before the forward. An
                # invalid fresh capture is a request error, never a reason
                # to send incomplete conditioning or terminate healthy rows.
                if output.request_errors is None:
                    output.request_errors = {}
                output.request_errors[entry.req_id] = str(exc)
                self._discard_row(output, entry.row)
                continue
            if merged is None:
                # D2H can finish after cancellation or request-ID reuse.
                self._discard_row(output, entry.row)
                continue
            for payload in payloads:
                for name, key in keys.items():
                    if key in payload:
                        payload[key] = merged[name]

    def finalize(self, base: OutputFinalizer | None) -> OutputFinalizer:
        def finish(output: OmniModelRunnerOutput) -> OmniModelRunnerOutput:
            self.apply(output)
            return output if base is None else base(output)

        return finish


def snapshot_mooncake_output(runner: OmniARModelRunner, input_batch: InputBatch) -> MooncakeOutputRewrite | None:
    """Snapshot before sampling mutates counts; other deployments are a no-op."""
    config = runner.vllm_config.kv_transfer_config
    if config is None or config.kv_connector != "OmniMooncakeStoreConnector":
        return None
    connector = cast("OmniMooncakeStoreConnector", runner.kv_connector.kv_connector)
    rows = []
    for row, req_id in enumerate(input_batch.req_ids[: input_batch.num_reqs]):
        if is_warmup_request(req_id):
            continue
        attempt = connector.current_attempt(req_id)
        if attempt is None:
            raise RuntimeError(f"Missing Mooncake load attempt for request {req_id}")
        req_index = input_batch.idx_mapping_np[row]
        rows.append(
            CaptureRequestContext(
                row,
                req_id,
                attempt,
                int(input_batch.num_computed_tokens_np[row]),
                int(runner.req_states.prompt_len.np[req_index]),
            )
        )
    return MooncakeOutputRewrite(connector, tuple(rows)) if rows else None
