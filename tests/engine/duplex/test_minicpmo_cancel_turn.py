# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniCPM-o closes a cancelled response's turn inside its running Stage-0 request."""

import base64

import pytest

from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import plugin as module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PAYLOAD = {
    "audio": base64.b64encode(bytes(16000 * 4)).decode(),
    "format": "pcm_f32le",
    "sample_rate_hz": 16000,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("seq", [1, 2])
@pytest.mark.parametrize("gander", [False, True])
async def test_close_model_turn_marks_only_the_next_append(seq, gander):
    plugin = module.MiniCPMO45DuplexPlugin(lambda *args: None)
    state = plugin.create_session_state()

    async def plan(index):
        config = plugin.prepare_prompt_config({"instructions": "Be brief."}, state=state, payload=dict(_PAYLOAD))
        return (
            await plugin.prepare_append_plan(
                request_id="request",
                fence=DuplexFence("sid"),
                session_config=config,
                runtime_config={
                    "duplex_first_append_context_tokens": 5,
                    "gander_enabled": gander,
                    "gander_context_version": 7,
                },
                seq=index,
                turn_seq=index,
                payload=dict(_PAYLOAD),
                final=False,
                sampling_params=None,
            )
        ).prompt

    plain = await plan(seq)
    assert plugin.close_model_turn(state)
    closing = await plan(seq)
    after = await plan(seq + 1)
    duplex = closing["model_intermediate_buffer"]["duplex"]
    assert duplex["payload"].get("close_turn", False) is (seq > 1)
    assert duplex["payload"].get("force_listen", False) is (seq > 1)
    assert "_minicpmo45_close_turn" not in duplex["session_config"]
    assert duplex["runtime_config"]["gander_context_version"] == 7
    assert len(closing["prompt_token_ids"]) == len(plain["prompt_token_ids"])
    assert "close_turn" not in after["model_intermediate_buffer"]["duplex"]["payload"]
    assert not state.close_turn_pending
