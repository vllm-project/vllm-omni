# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Data-plane output projector for InternLM Sage-Interaction full-duplex serving."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping

from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import (
    EncodeAudio,
    MiniCPMO45DataPlaneSession,
    _output_stage_metrics,
    _runtime_result,
    output_epoch_from_metadata,
    output_turn_id_from_metadata,
)
from vllm_omni.model_executor.models.sage_interaction.duplex.delegation import (
    DELEGATION_TOOL_NAME,
    extract_delegate_query,
)
from vllm_omni.outputs.duplex import get_duplex_output_decision


class SageInteractionDataPlane(MiniCPMO45DataPlaneSession):
    """Project Sage-Interaction Stage 0/2 outputs into Realtime events and delegate calls.

    Extends :class:`MiniCPMO45DataPlaneSession` with two responsibilities:
    1. Turn ``DIRECT_RESPONSE`` outputs carrying ``duplex_native_decision="delegate"``
       into structured ``function_call`` results for ``ModelChannel``;
    2. Track per-request foreground idle state (``backend_ready``) so queued
       ``<|im_start|>tool\\n...<|im_end|>`` results are injected only when the
       foreground turn is idle (after listen, delegate, or end-of-turn).
    """

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self._backend_ready: dict[str, bool] = {}

    def begin_request(self, request_id: str) -> None:
        super().begin_request(request_id)
        self._backend_ready.setdefault(request_id, True)

    def close_request(self, request_id: str) -> None:
        super().close_request(request_id)
        self._backend_ready.pop(request_id, None)

    def backend_ready(self, request_id: str) -> bool:
        """Whether queued tool results may be prepended to the next unit for ``request_id``."""
        return self._backend_ready.get(request_id, True)

    def project_output(
        self,
        output: object,
        *,
        context: DuplexDataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]:
        ctx = context or DuplexDataPlaneContext()
        request_id = getattr(output, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            request_id = None

        direct_decision = get_duplex_output_decision(output)
        direct_metadata = getattr(direct_decision, "metadata", None)
        mm_output = dict(direct_metadata) if isinstance(direct_metadata, Mapping) else {}
        if not mm_output:
            raw_mm = getattr(output, "multimodal_output", None)
            if isinstance(raw_mm, Mapping):
                mm_output = dict(raw_mm)

        if mm_output.get("duplex_native_decision") == "delegate":
            output_turn_id = output_turn_id_from_metadata(mm_output)
            output_epoch = output_epoch_from_metadata(mm_output)
            expected_turn_id = ctx.active_response_turn_id
            stale_turn = (
                expected_turn_id is not None and output_turn_id is not None and output_turn_id < expected_turn_id
            )
            if expected_turn_id is None and output_turn_id is not None:
                stale_turn = output_turn_id < ctx.turn_id
            stale_epoch = output_epoch is not None and output_epoch != ctx.epoch
            if ctx.auto_responds and (stale_turn or stale_epoch):
                return

            if request_id is not None:
                self._backend_ready[request_id] = True

            raw_delegate_text = mm_output.get("delegate_text")
            if not isinstance(raw_delegate_text, str) or not raw_delegate_text:
                outputs = getattr(output, "outputs", None)
                completion = outputs[0] if isinstance(outputs, list) and outputs else None
                raw_delegate_text = getattr(completion, "text", "") if completion is not None else ""

            query = extract_delegate_query(str(raw_delegate_text or ""))
            stage_metrics = _output_stage_metrics(output)
            if query:
                result = _runtime_result(
                    stage_role="llm",
                    function_call=True,
                    call_id=f"call_{uuid.uuid4().hex}",
                    name=DELEGATION_TOOL_NAME,
                    arguments=json.dumps({"query": query}, ensure_ascii=False),
                    text="",
                    data_plane_request_id=request_id,
                    end_of_turn=True,
                )
            else:
                result = _runtime_result(
                    stage_role="llm",
                    is_listen=True,
                    model_listen=True,
                    listen_source="delegate_empty",
                    data_plane_request_id=request_id,
                    end_of_turn=False,
                )
            if output_turn_id is not None:
                result["model_turn_id"] = output_turn_id
            if stage_metrics is not None:
                result["stage_metrics"] = stage_metrics
            yield result
            return

        for item in super().project_output(output, context=ctx):
            if request_id is not None:
                if item.get("is_listen") is True or item.get("end_of_turn") is True:
                    self._backend_ready[request_id] = True
                elif item.get("text") or item.get("audio_data") or item.get("audio"):
                    self._backend_ready[request_id] = False
            yield item


__all__ = ["SageInteractionDataPlane"]
