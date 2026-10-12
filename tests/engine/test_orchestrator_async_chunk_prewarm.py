# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""When the orchestrator attaches an async-chunk prewarm payload to a placeholder.

Only the initial add of a non-session, non-resumable request asks a stage's
``async_chunk_prewarm_payload_func``; the payload rides as flat
``f"{ASYNC_CHUNK_PREWARM_NS}.<name>"`` keys in the placeholder's
additional_information. Every other path submits the plain placeholder.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.data_entry_keys import ASYNC_CHUNK_PREWARM_NS
from vllm_omni.engine.duplex.contracts import DuplexFence, DuplexStageRequestContext, DuplexStageSubmission
from vllm_omni.engine.duplex_orchestrator import DuplexOrchestrator, DuplexOrchestratorRequestState
from vllm_omni.engine.messages import StageSubmissionMessage
from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    _attach_async_chunk_prewarm_payload,
)
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import code2wav_prewarm_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REF_KEY = f"{ASYNC_CHUNK_PREWARM_NS}.ref_audio"
_SR_KEY = f"{ASYNC_CHUNK_PREWARM_NS}.ref_audio_sr"
_PLAIN_KEYS = {"global_request_id"}
_TOKENS = [1, 2, 3]


def _waveform() -> torch.Tensor:
    return torch.randn(1600, generator=torch.Generator().manual_seed(0), dtype=torch.float32)


def _client(payload_func: Any) -> SimpleNamespace:
    return SimpleNamespace(async_chunk_prewarm_payload_func=payload_func)


class _FakePool:
    """Single-replica LLM pool that records the requests it is sent."""

    stage_type = "llm"

    def __init__(self, role: str, stage_client: Any = None) -> None:
        self.stage_client = stage_client if stage_client is not None else SimpleNamespace()
        self.stage_vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64, stage_connector_config={"extra": {"role": role}})
        )
        self.submitted: list[Any] = []

    def live_replica_ids(self) -> list[int]:
        return [0]

    async def submit_initial(self, request_id, _req_state, request, prompt_text=None):
        self.submitted.append(request)
        return 0

    async def submit_update(self, request_id, _req_state, request, prompt_text=None):
        return 0

    def get_bound_replica_id(self, request_id):
        return 0

    def get_bound_client(self, request_id):
        return self.stage_client


def _orchestrator(payload_func: Any = None, cls: type[Orchestrator] = Orchestrator) -> Orchestrator:
    """Thinker -> Talker (no payload hook) -> Code2Wav (``payload_func``, if any)."""
    code2wav_client = None if payload_func is None else _client(payload_func)
    orchestrator = object.__new__(cls)
    orchestrator.stage_pools = [_FakePool("sender"), _FakePool("receiver"), _FakePool("receiver", code2wav_client)]
    orchestrator.async_chunk = True
    orchestrator.request_states = {}
    orchestrator._emit_tx_edge = lambda **_kwargs: None
    return orchestrator


def _prompt(waveform: torch.Tensor | None) -> dict[str, Any]:
    prompt: dict[str, Any] = {
        "prompt_token_ids": list(_TOKENS),
        "additional_information": {"global_request_id": ["req"]},
    }
    if waveform is not None:
        prompt["multi_modal_data"] = {"audio": (waveform, 16000)}
    return prompt


def _stage0(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(prompt_token_ids=list(_TOKENS), **kwargs)


def _sampling_params() -> list[SamplingParams]:
    return [SamplingParams(max_tokens=1) for _ in range(3)]


def _message(msg_type: str, original_prompt: Any) -> StageSubmissionMessage:
    return StageSubmissionMessage(
        type=msg_type,
        request_id="req",
        prompt=_stage0(resumable=False),
        original_prompt=original_prompt,
        output_prompt_text=None,
        sampling_params_list=_sampling_params(),
        final_stage_id=2,
        preprocess_ms=0.0,
        request_timestamp=time.time(),
        enqueue_ts=0.0,
    )


def _entry_keys(request: Any) -> set[str]:
    info = request.additional_information
    return set() if info is None else set(info.entries)


@pytest.mark.parametrize("existing", ["dict", "none", "missing"])
def test_attach_adds_flat_keys(existing: str, mocker) -> None:
    waveform = _waveform()
    info = {"global_request_id": ["req"]}
    base_input: dict[str, Any] = {"prompt_token_ids": [0]}
    if existing != "missing":
        base_input["additional_information"] = info if existing == "dict" else None
    prompt = {"multi_modal_data": {"audio": (waveform, 16000)}}
    payload_func = mocker.Mock(return_value={"ref_audio": waveform, "ref_audio_sr": 16000})

    _attach_async_chunk_prewarm_payload(base_input, prompt, _client(payload_func), "req", 2)

    expected = {_REF_KEY: waveform, _SR_KEY: 16000}
    if existing == "dict":
        expected["global_request_id"] = ["req"]
    assert base_input["additional_information"] == expected
    assert info == {"global_request_id": ["req"]}
    assert payload_func.call_args.args[0] is prompt


@pytest.mark.parametrize(
    "make_client",
    [
        pytest.param(lambda m: SimpleNamespace(), id="no-func"),
        pytest.param(lambda m: _client(None), id="func-none"),
        # Other orchestrator tests drive MagicMock pools; their "payload" is never a dict.
        pytest.param(lambda m: m.MagicMock(), id="magicmock-client"),
        pytest.param(lambda m: _client(m.Mock(side_effect=RuntimeError("decode failed"))), id="func-raises"),
        pytest.param(lambda m: _client(m.Mock(return_value=None)), id="payload-none"),
        pytest.param(lambda m: _client(m.Mock(return_value={})), id="payload-empty"),
        pytest.param(lambda m: _client(m.Mock(return_value=[("ref_audio", 1)])), id="payload-not-dict"),
    ],
)
def test_attach_leaves_input_untouched(make_client: Any, mocker) -> None:
    info = {"global_request_id": ["req"]}
    base_input = {"prompt_token_ids": [0], "additional_information": info}

    _attach_async_chunk_prewarm_payload(base_input, {}, make_client(mocker), "req", 2)

    assert base_input == {"prompt_token_ids": [0], "additional_information": {"global_request_id": ["req"]}}
    assert base_input["additional_information"] is info


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_calls"),
    [
        pytest.param({"attach": False}, 0, id="flag-off"),
        pytest.param({"session_owned": True}, 0, id="session-owned"),
        pytest.param({"stage0": {"resumable": True}}, 0, id="resumable-request"),
        # Without an explicit flag the request inherits streaming input.
        pytest.param({"stage0": {}, "streaming": True}, 0, id="streaming-input"),
        pytest.param({"waveform": None}, 1, id="no-reference-audio"),
        pytest.param({"raises": True}, 1, id="payload-func-raises"),
        pytest.param({"no_func": True}, 0, id="client-without-func"),
    ],
)
async def test_prewarm_submits_plain_placeholder(case: dict[str, Any], expected_calls: int, mocker) -> None:
    if case.get("raises"):
        payload_func = mocker.Mock(side_effect=ValueError("bad reference"))
    else:
        payload_func = mocker.Mock(wraps=code2wav_prewarm_payload)
    orchestrator = _orchestrator(None if case.get("no_func") else payload_func)
    req_state = OrchestratorRequestState(
        request_id="req",
        prompt=_prompt(case.get("waveform", _waveform())),
        sampling_params_list=_sampling_params(),
        final_stage_id=2,
        session_owned=case.get("session_owned", False),
    )
    streaming = case.get("streaming", False)
    req_state.streaming.enabled = streaming
    stage0_kwargs = case.get("stage0", {"resumable": False})

    prewarmed = await orchestrator._prewarm_async_chunk_stages(
        "req", _stage0(**stage0_kwargs), req_state, attach_prewarm_payload=case.get("attach", True)
    )

    assert prewarmed is True
    assert payload_func.call_count == expected_calls
    (placeholder,) = orchestrator.stage_pools[2].submitted
    assert _entry_keys(placeholder) == _PLAIN_KEYS
    assert placeholder.resumable is stage0_kwargs.get("resumable", streaming)


@pytest.mark.asyncio
async def test_add_request_attaches_payload_once_and_streaming_update_does_not(mocker) -> None:
    waveform = _waveform()
    payload_func = mocker.Mock(wraps=code2wav_prewarm_payload)
    orchestrator = _orchestrator(payload_func)
    _, talker, code2wav = orchestrator.stage_pools
    prompt = _prompt(waveform)

    await orchestrator._handle_add_request(_message("add_request", prompt))

    # Only Code2Wav declares a payload func; the Talker placeholder stays plain.
    assert _entry_keys(talker.submitted[0]) == _PLAIN_KEYS
    (placeholder,) = code2wav.submitted
    assert _entry_keys(placeholder) == _PLAIN_KEYS | {_REF_KEY, _SR_KEY}
    prewarm = deserialize_additional_information(placeholder.additional_information)[ASYNC_CHUNK_PREWARM_NS]
    assert prewarm["ref_audio"].dtype == torch.float32
    assert torch.equal(prewarm["ref_audio"], waveform)
    assert prewarm["ref_audio_sr"] == 16000
    # The hook reads the original prompt (the placeholder copy has dropped
    # multi_modal_data), and that prompt is left unmodified.
    assert payload_func.call_count == 1
    assert payload_func.call_args.args[0] is prompt
    assert prompt == _prompt(waveform)

    # The re-prewarm must not resend the payload to a stage that already holds
    # the request, even though stage 0 is explicitly not resumable.
    await orchestrator._handle_streaming_update(_message("streaming_update", {"prompt": "segment-2"}))

    assert len(code2wav.submitted) == 2
    assert _entry_keys(code2wav.submitted[1]) == _PLAIN_KEYS
    assert payload_func.call_count == 1


@pytest.mark.asyncio
async def test_duplex_session_submit_does_not_request_payload(mocker) -> None:
    orchestrator = _orchestrator(cls=DuplexOrchestrator)
    prewarm = mocker.AsyncMock(return_value=True)
    orchestrator._prewarm_async_chunk_stages = prewarm  # type: ignore[method-assign]
    sampling_params = tuple(_sampling_params())
    orchestrator.request_states["sess-req"] = DuplexOrchestratorRequestState(
        request_id="sess-req", final_stage_id=2, session_owned=True, sampling_params_list=list(sampling_params)
    )
    context = DuplexStageRequestContext(
        request_id="sess-req",
        session_id="sess",
        fence=DuplexFence(session_id="sess"),
        stage_id=0,
        final_stage_id=2,
        config_generation=0,
        sampling_params=sampling_params,
    )

    await orchestrator.submit(
        DuplexStageSubmission(
            context=context, prompt={"prompt_token_ids": list(_TOKENS)}, already_submitted=False, resumable=True
        )
    )

    prewarm.assert_awaited_once()
    assert prewarm.await_args.kwargs.get("attach_prewarm_payload", False) is False
