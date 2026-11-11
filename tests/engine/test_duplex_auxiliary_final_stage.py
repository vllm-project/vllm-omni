# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ephemeral duplex requests must carry their worker-side final-stage boundary."""

from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from tests.engine.test_duplex_auxiliary_stop import native_processor as _native_processor_fixture
from tests.engine.test_duplex_orchestrator import _build
from tests.helpers.omni_scheduler import bind_omits_transfer_helpers
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    duplex_resource_request_id,
)
from vllm_omni.engine.serialization import deserialize_additional_information

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
native_processor = _native_processor_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("final_stage_id,omits_downstream", [(0, True), (1, False)])
@pytest.mark.parametrize("existing_info", [None, {"speaker": ["Ryan"]}])
async def test_ephemeral_stage_boundary_reaches_the_worker(
    native_processor, final_stage_id, omits_downstream, existing_info
):
    orchestrator, clients, _, _ = _build(stages=2)
    orchestrator._stage_input_processors[0] = native_processor
    fence = DuplexFence("auxiliary-final-stage")
    request_id = duplex_resource_request_id(fence, "auxiliary-memory-probe")
    params = SamplingParams(max_tokens=32, output_kind=RequestOutputKind.FINAL_ONLY)
    context = DuplexStageRequestContext(
        request_id=request_id,
        session_id=fence.session_id,
        fence=fence,
        stage_id=0,
        final_stage_id=final_stage_id,
        config_generation=0,
        sampling_params=(params, params),
    )
    orchestrator.ensure_request(context)
    try:
        await orchestrator.submit(
            DuplexStageSubmission(
                context=context,
                prompt={
                    "prompt_token_ids": [1, 2, 3],
                    **({"additional_information": existing_info} if existing_info else {}),
                },
                already_submitted=False,
                resumable=False,
            )
        )
        sent = clients[0].add_request_calls[0][0]
        assert sent.resumable is False
        assert sent.sampling_params.max_tokens == 32
        assert sent.sampling_params.output_kind is RequestOutputKind.FINAL_ONLY
        if existing_info is not None:
            info = deserialize_additional_information(sent.additional_information)
            assert info["speaker"] == ["Ryan"]
        # Use the actual scheduler decision, rather than inspecting a mock's
        # attributes: a private stage-0 result has no downstream receiver.
        scheduler = SimpleNamespace(_omits_kv_transfer_cache={})
        bind_omits_transfer_helpers(scheduler)
        assert scheduler._request_omits_chunk_transfer_to_next_stage(sent) is omits_downstream
        assert scheduler._request_omits_kv_transfer_to_next_stage(sent) is omits_downstream
    finally:
        await orchestrator.cleanup([request_id], abort=True)
