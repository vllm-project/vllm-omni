# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Sequence

import msgspec
import pybase64 as base64
from vllm.outputs import RequestOutput

from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackDecoder, OmniMsgpackEncoder
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.messages import NextStageInputMessage, OutputMessage
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.protocol.run import RunRequest, RunResponse
from vllm_omni.entrypoints.openai.stage_params import to_sampling_params_list
from vllm_omni.entrypoints.openai.utils import get_stage_type
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniSamplingParams


def is_diffusion(stage_type: str) -> bool:
    return stage_type == "diffusion"


def encode_payload(payload: NextStageInputMessage | RequestOutput) -> str:
    """Encode a next stage input or a final output as base64 text for a JSON response."""
    return base64.b64encode(OmniMsgpackEncoder().encode(payload)).decode("ascii")


def decode_stage_input(data: str, stage_types: Sequence[str]) -> NextStageInputMessage:
    """Decode a next stage input encoded by `encode_payload`.

    The codec returns structs as plain containers, which we rebuild based on stage type.
    """
    fields = OmniMsgpackDecoder().decode(base64.b64decode(data))
    if not is_diffusion(stage_types[fields["receiver_stage_id"]]):
        fields["requests"] = [msgspec.convert(request, OmniEngineCoreRequest) for request in fields["requests"]]
    return NextStageInputMessage(**fields)


def decode_output(data: str) -> RequestOutput:
    """Decode a final output encoded by `encode_payload`."""
    # NOTE: This is temporarily how we get the response back from the last stage,
    # but will change as we integrate into route specific post processing.
    output = OmniMsgpackDecoder().decode(base64.b64decode(data))
    if not isinstance(output, RequestOutput):
        raise ValueError(f"output decodes to {type(output).__name__}, not a RequestOutput")
    return output


class ServingRun:
    """Runs one stage per call and returns its raw encoded result."""

    def __init__(self, engine_client: AsyncOmni) -> None:
        self.engine_client = engine_client
        self.stage_types = [get_stage_type(stage_config) for stage_config in engine_client.stage_configs]

    async def run(self, request: RunRequest, *, request_id: str) -> RunResponse:
        """Run the stage named by `request.stage_id` and return its encoded result."""
        sampling_params_list = coerce_param_message_types(
            to_sampling_params_list(self.engine_client, request.sampling_params or []), is_streaming=False
        )
        if request.stage_id == 0:
            response = await self._run_entry_stage(request, sampling_params_list, request_id=request_id)
        else:
            response = await self._run_downstream_stage(request, sampling_params_list, request_id=request_id)
        return self._build_stage_response(response)

    async def _run_entry_stage(
        self, request: RunRequest, sampling_params_list: list[OmniSamplingParams], *, request_id: str
    ) -> NextStageInputMessage | OutputMessage:
        """Run stage 0 on the prompt in request.stage_input."""
        if not isinstance(request.stage_input, dict):
            raise OmniClientError("Stage 0 takes a prompt dict as stage_input")
        return await self.engine_client.run_entry_stage(
            request.stage_input, sampling_params_list, request_id=request_id
        )

    async def _run_downstream_stage(
        self, request: RunRequest, sampling_params_list: list[OmniSamplingParams], *, request_id: str
    ) -> NextStageInputMessage | OutputMessage:
        """Run the stage that request.stage_input was built for, once it matches request.stage_id."""
        if not isinstance(request.stage_input, str):
            raise OmniClientError(f"Stage {request.stage_id} takes the stage_input returned by the previous call")
        stage_input = decode_stage_input(request.stage_input, self.stage_types)
        if stage_input.receiver_stage_id != request.stage_id:
            raise OmniClientError(
                f"stage_input is for stage {stage_input.receiver_stage_id}, not stage {request.stage_id}"
            )
        return await self.engine_client.run_downstream_stage(stage_input, sampling_params_list, request_id=request_id)

    @staticmethod
    def _build_stage_response(response: NextStageInputMessage | OutputMessage) -> RunResponse:
        """Encode the next stage input, or the final stage's output after the last stage."""
        if isinstance(response, NextStageInputMessage):
            return RunResponse(stage_id=response.receiver_stage_id, stage_input=encode_payload(response))
        return RunResponse(output=encode_payload(response.engine_outputs))
