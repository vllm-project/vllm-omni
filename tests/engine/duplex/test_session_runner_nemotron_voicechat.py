# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Essential Nemotron regressions through the real session runner and plugin."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio
import torch

from tests.engine.duplex.test_session_runner import (
    SESSION_ID,
    Harness,
    RecordingStagePort,
    _fake_encode_audio,
    append_audio,
    types,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.messages import OpenDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import load_duplex_plugin
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.model_executor.models.nemotron_voicechat.duplex import plugin as plugin_module
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    NemotronVoiceChatThinkerForConditionalGeneration as Thinker,
)
from vllm_omni.model_executor.models.nemotron_voicechat.pipeline import NEMOTRON_VOICECHAT_PIPELINE
from vllm_omni.protocol.realtime.items import normalize_conversation_item
from vllm_omni.utils.mm_outputs import build_mm_cpu
from vllm_omni.worker.output.payload_build import build_omni_mm_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest_asyncio.fixture
async def harness(monkeypatch):
    monkeypatch.setenv("NEMOTRON_VOICECHAT_LLM_PATH", "test-tokenizer")
    special_ids = dict(
        nvc_text_bos_id=0,
        nvc_text_eos_id=1,
        nvc_text_pad_id=12,
        nvc_function_sotc_id=20,
        nvc_function_eotc_id=21,
        nvc_function_eotr_id=22,
        nvc_tokenizer_ref="test-tokenizer",
    )
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [42],
        decode=lambda ids, **kwargs: '[{"name":"lookup","arguments":{"x":1}}]',
    )
    monkeypatch.setattr(plugin_module, "_load_tokenizer_runtime", lambda _: (special_ids, tokenizer))
    plugin = load_duplex_plugin(NEMOTRON_VOICECHAT_PIPELINE.duplex_plugin, _fake_encode_audio)
    port = RecordingStagePort(stage_count=3)
    output: asyncio.Queue = asyncio.Queue()
    results: asyncio.Queue = asyncio.Queue()
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=DuplexSessionRuntimeConfig(),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(stt_cfg={}), max_model_len=8192),
    )
    try:
        config = DuplexSessionConfig(model="nemotron", instructions="hi", extra_body={"auto_response": True})
        await manager.handle(OpenDuplexSessionMessage(control_id="open", session_id=SESSION_ID, session_config=config))
        result = await asyncio.wait_for(results.get(), timeout=2)
        assert result.ok, result
        h = Harness(manager, port, output, results, manager.runners[SESSION_ID])
        await h.settle()
        yield h
    finally:
        await manager.shutdown()


async def deliver(h, token_id=12, **metadata):
    output = SimpleNamespace(
        request_id=h.stage0_request_id(),
        finished=False,
        multimodal_output={},
        outputs=[SimpleNamespace(text="", token_ids=[token_id], multimodal_output={})],
    )
    return await h.deliver_and_settle(
        output,
        stage_id=0,
        segment_finished=True,
        segment_token_ids=[token_id],
        segment_output_metadata=metadata,
    )


@pytest.mark.asyncio
async def test_frame_buffering_prefill_and_resumable_append(harness):
    h = harness
    await h.run(append_audio(samples=1000))
    assert h.port.submissions == []
    await h.run(append_audio(samples=280))
    await h.run(append_audio(samples=1280))
    first, second = h.port.submissions
    assert first.prompt["prompt_token_ids"] == [0, 42, 1, 12]
    assert second.prompt["prompt_token_ids"] == [12]
    assert first.prompt["model_intermediate_buffer"]["duplex"]["source_input_seq"] == 1
    assert second.prompt["model_intermediate_buffer"]["duplex"]["source_input_seq"] == 2
    assert not first.already_submitted and second.already_submitted
    assert first.context.request_id == second.context.request_id == h.stage0_request_id()
    assert "error" not in types(h.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_ids", [("active",), ("other", "active"), ("active", "other")])
async def test_tool_batches_reach_worker_and_drain_before_ninth_result(harness, request_ids):
    h = harness
    await h.run(append_audio(samples=1280))
    worker_state = {"function_response_generation": 0, "forced_function_tokens": []}
    thinker = Thinker.__new__(Thinker)
    torch.nn.Module.__init__(thinker)
    thinker._sessions = {"active": worker_state, "other": {"function_response_generation": 2}}
    thinker._use_function_head, thinker._dtype = True, torch.float32
    thinker.function_head = torch.nn.Identity()
    key = "nvc_function_response_consumed_generation"
    assert key in thinker.omni_client_multimodal_output_keys
    infos = [{"request_id": rid} for rid in request_ids]
    hidden = torch.zeros((len(infos) + 1, 2))
    for generation in range(1, 10):
        for token_id in (20, 7, 21):
            events = await deliver(h, nvc_function_token=[token_id])
        done = next(event for event in events if event.type == "response.function_call_arguments.done")
        item = {"type": "function_call_output", "call_id": done.call_id, "output": "20"}
        events = await h.run(commands.CreateItem(item=normalize_conversation_item(item)))
        assert "error" not in types(events)
        runtime = h.session.runtime_config
        assert runtime["nvc_function_response_generation"] == generation
        assert runtime["nvc_function_response_batches"][-1] == {"generation": generation, "token_ids": [42]}
        if generation < 8:
            continue
        if generation == 8:
            with pytest.raises(plugin_module.NemotronVoiceChatClientRuntimeConfigError, match="faster than"):
                h.runner.plugin.runtime_config_for_function_output(h.session.config, runtime, item)
        # Copy into model state exactly once, even if the same runtime snapshot is replayed.
        Thinker._sync_forced_function_response(worker_state, runtime)
        Thinker._sync_forced_function_response(worker_state, runtime)
        assert worker_state["forced_function_tokens"] == [42] * generation
        assert worker_state["function_response_generation"] == generation
        output = thinker.make_omni_output(hidden, model_intermediate_buffer=infos)
        for idx, rid in enumerate(request_ids):
            payload = build_omni_mm_payload(
                combined_multimodal_outputs=None,
                mm_cpu=build_mm_cpu(output.multimodal_outputs),
                rid=rid,
                idx=idx,
                start=0 if idx == 0 else 2,
                end=2 if idx == 0 else 3,
                audio_sparse_output=False,
                sparse_mm_index={},
                hidden_seq_len=len(hidden),
                scheduled_seq_len=len(hidden),
            )
            assert int(payload[key]) == (generation if rid == "active" else 2)
            if rid == "active":
                await deliver(h, **payload)
        assert h.session.runtime_config["nvc_function_response_batches"] == []
        assert h.session.runtime_config["nvc_function_response_generation"] == generation
    assert "error" not in types(h.events)
