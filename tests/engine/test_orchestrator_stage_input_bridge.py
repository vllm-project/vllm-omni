# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import janus
import pytest
import torch
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex_orchestrator import DuplexOrchestrator
from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    StreamingSegmentState,
)
from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeStageClient:
    def __init__(
        self,
        *,
        next_inputs: list[dict[str, Any]] | None = None,
        final_output: bool = False,
    ) -> None:
        self.stage_id = 0
        self.replica_id = 0
        self.stage_type = "llm"
        self.final_output = final_output
        self.final_output_type = "text"
        self.default_sampling_params = SamplingParams(max_tokens=1)
        self.requires_multimodal_data = False
        self.engine_input_source = [0]
        self.is_comprehension = False
        self.model_stage = None
        self.custom_process_input_func = None
        self.next_inputs = list(next_inputs or [])
        self.add_request_calls: list[tuple[Any, ...]] = []
        self.decoded_source_tokens: str | None = None
        self._engine_core_outputs: queue.Queue[Any] = queue.Queue()

    async def add_request_async(self, *args, **_kwargs) -> None:
        self.add_request_calls.append(args)

    async def get_output_async(self):
        try:
            return self._engine_core_outputs.get_nowait()
        except queue.Empty:
            return SimpleNamespace(outputs=[])

    def process_engine_inputs(self, _source_outputs, prompt=None, streaming_context=None):
        decoder = getattr(streaming_context, "source_token_decoder", None)
        if callable(decoder):
            self.decoded_source_tokens = decoder([11, 12], skip_special_tokens=True)
        return list(self.next_inputs)

    async def abort_requests_async(self, _request_ids: list[str]) -> None:
        return None

    def set_engine_outputs(self, _outputs) -> None:
        return None

    def check_health(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class FakeOutputProcessor:
    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer

    def add_request(self, *args, **kwargs) -> None:
        return None


class FakeInputProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.renderer = SimpleNamespace(mm_processor_cache=None)

    def process_inputs(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            request_id=kwargs["request_id"],
            prompt_token_ids=[101, 102],
            prompt_embeds=None,
            external_req_id=None,
        )


class FakePrewarmPool:
    stage_type = "llm"

    def __init__(self, role: str) -> None:
        self.stage_vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=64,
                stage_connector_config={"extra": {"role": role}},
            )
        )
        self.submitted: list[Any] = []
        self.stage_client: Any = None  # _build_payload_sender_info fallback (returns None → OK)

    async def submit_initial(self, _request_id, _req_state, request, prompt_text=None):
        self.submitted.append(request)
        return 0

    def get_bound_replica_id(self, _request_id):
        return 0

    def get_bound_client(self, _request_id: str) -> None:
        """No bound client; _build_payload_sender_info falls back to stage_client (None)."""
        return None


def _request_output(request_id: str) -> RequestOutput:
    completion = CompletionOutput(
        index=0,
        text="transcript",
        token_ids=[11, 12],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    return RequestOutput(
        request_id=request_id,
        prompt="prompt",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
        metrics=None,
        lora_request=None,
    )


@pytest.mark.asyncio
async def test_forward_text_prompt_uses_target_stage_input_processor() -> None:
    class SourceTokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return ":".join(str(token_id) for token_id in token_ids)

    stage0 = FakeStageClient(final_output=True)
    stage1 = FakeStageClient(
        final_output=True,
        next_inputs=[{"prompt": "hello", "multi_modal_data": {"video": ["frame"]}}],
    )
    stage_pools = [
        StagePool(
            0,
            [stage0],
            output_processor=FakeOutputProcessor(tokenizer=SourceTokenizer()),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
        StagePool(
            1,
            [stage1],
            output_processor=FakeOutputProcessor(),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
    ]
    request_q = janus.Queue()
    output_q = janus.Queue()
    rpc_q = janus.Queue()
    orchestrator = Orchestrator(
        request_async_queue=request_q.async_q,
        output_async_queue=output_q.async_q,
        rpc_async_queue=rpc_q.async_q,
        stage_pools=stage_pools,
        async_chunk=False,
    )
    input_processor = FakeInputProcessor()
    orchestrator._stage_input_processors[1] = input_processor
    req_state = OrchestratorRequestState(
        request_id="req-text",
        prompt={"prompt": "original"},
        sampling_params_list=[SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)],
        final_stage_id=1,
    )
    orchestrator.request_states[req_state.request_id] = req_state

    await orchestrator._forward_to_next_stage("req-text", 0, _request_output("req-text"), req_state)

    assert input_processor.calls
    assert input_processor.calls[0]["prompt"] == {"prompt": "hello", "multi_modal_data": {"video": ["frame"]}}
    assert stage1.decoded_source_tokens == "11:12"
    assert req_state.streaming.source_token_decoder is None
    assert stage1.add_request_calls
    submitted_request = stage1.add_request_calls[0][0]
    assert submitted_request.prompt_token_ids == [101, 102]
    assert submitted_request.external_req_id == "req-text"


@pytest.mark.asyncio
async def test_async_prewarm_skips_outgoing_only_stage() -> None:
    orchestrator = object.__new__(Orchestrator)
    stage0 = FakePrewarmPool("sender")
    stage1 = FakePrewarmPool("sender")
    stage2 = FakePrewarmPool("receiver")
    orchestrator.stage_pools = [stage0, stage1, stage2]
    orchestrator._emit_tx_edge = lambda **_kwargs: None
    orchestrator._on_stage_submitted = MagicMock()
    req_state = OrchestratorRequestState(
        request_id="req-prewarm",
        prompt={"prompt_token_ids": [1, 2]},
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(3)],
        final_stage_id=2,
    )

    prewarmed = await orchestrator._prewarm_async_chunk_stages(
        "req-prewarm",
        SimpleNamespace(prompt_token_ids=[1, 2], resumable=True),
        req_state,
    )

    assert prewarmed is True
    assert stage1.submitted == []
    assert len(stage2.submitted) == 1
    assert 1 not in req_state.stage_submit_ts
    assert 2 in req_state.stage_submit_ts
    orchestrator._on_stage_submitted.assert_called_once_with(
        2,
        "req-prewarm",
        0,
        req_state,
    )


@pytest.mark.asyncio
async def test_async_prewarm_skips_stage_with_custom_process_input_func() -> None:
    """AURA Stage1 has asr2aura: must not be zero-prewarmed under async_chunk.

    The custom-process-input gate lives on DuplexOrchestrator (AURA path), not
    the turn-based Orchestrator base.
    """
    orchestrator = object.__new__(DuplexOrchestrator)
    stage0 = FakePrewarmPool("sender")
    stage1 = FakePrewarmPool("receiver")  # would receive chunks if role alone decided
    stage1.stage_client = SimpleNamespace(custom_process_input_func=lambda *a, **k: None)
    stage2 = FakePrewarmPool("receiver")
    orchestrator.stage_pools = [stage0, stage1, stage2]
    orchestrator._emit_tx_edge = lambda **_kwargs: None
    orchestrator._on_stage_submitted = MagicMock()
    req_state = OrchestratorRequestState(
        request_id="req-prewarm-custom",
        prompt={"prompt_token_ids": [1, 2]},
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(3)],
        final_stage_id=2,
    )

    prewarmed = await orchestrator._prewarm_async_chunk_stages(
        "req-prewarm-custom",
        SimpleNamespace(prompt_token_ids=[1, 2], resumable=True),
        req_state,
    )

    assert prewarmed is True
    assert stage1.submitted == []
    assert len(stage2.submitted) == 1
    assert 1 not in req_state.stage_submit_ts
    assert 2 in req_state.stage_submit_ts


async def test_prewarm_uses_registered_build_prewarm_placeholder() -> None:
    """Async-chunk prewarm uses the metadata-resolved ``build_prewarm_placeholder``.

    The builder is resolved into stage metadata at init time (see
    ``_resolve_prewarm_placeholder_builder``) and read off the stage client —
    the orchestrator never re-resolves.  The placeholder length is
    ``stage0_only`` = the Qwen chat-template scan on the stage-0 prompt.  The
    synthetic ``[1, 2]`` prompt has no chat-template marker, so the scan yields
    0 and the builder floors it to a 1-token placeholder (best-effort; the
    connector fixup path replaces it later).
    """
    from vllm_omni.engine.stage_init_utils import _resolve_prewarm_placeholder_builder

    orchestrator = object.__new__(Orchestrator)
    stage0 = FakePrewarmPool("sender")
    stage1 = FakePrewarmPool("receiver")
    # The real qwen3_omni sync placeholder builder exposes build_prewarm_placeholder
    # off the resolved fn (see test_placeholder_parity.py).
    stage1.stage_client = SimpleNamespace(  # type: ignore[attr-defined]
        sync_process_input_func=(
            "vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_token_only"
        ),
        prewarm_placeholder_builder=_resolve_prewarm_placeholder_builder(
            sync_path=("vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_token_only"),
            custom_fn=None,
            stage_config=None,
        ),
    )
    orchestrator.stage_pools = [stage0, stage1]
    orchestrator._emit_tx_edge = lambda **_kwargs: None
    orchestrator._on_stage_submitted = MagicMock()
    req_state = OrchestratorRequestState(
        request_id="req-prewarm-resolved",
        prompt={"prompt_token_ids": [1, 2]},
        sampling_params_list=[SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)],
        final_stage_id=1,
    )

    prewarmed = await orchestrator._prewarm_async_chunk_stages(
        "req-prewarm-resolved",
        SimpleNamespace(prompt_token_ids=[1, 2], resumable=True),
        req_state,
    )

    assert prewarmed is True
    assert len(stage1.submitted) == 1
    # build_prewarm_placeholder uses stage0_only = the chat-template scan on the
    # stage-0 prompt.  The synthetic [1, 2] prompt has no chat-template marker,
    # so the scan yields 0 and the builder floors it to a 1-token placeholder.
    assert stage1.submitted[0].prompt_token_ids == [0]
    orchestrator._on_stage_submitted.assert_called_once_with(
        1,
        "req-prewarm-resolved",
        0,
        req_state,
    )


@pytest.mark.asyncio
async def test_async_route_forwards_to_outgoing_only_stage() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = True
    orchestrator._pd_pair = None
    orchestrator._cfg_tracker = SimpleNamespace(
        is_companion=lambda _request_id: False,
        has_companions=lambda _request_id: False,
    )
    stage0 = SimpleNamespace(final_output=False)
    stage1 = FakePrewarmPool("sender")
    orchestrator.stage_pools = [stage0, stage1]
    orchestrator._forward_to_next_stage = AsyncMock()
    req_state = OrchestratorRequestState(
        request_id="req-route",
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(2)],
        final_stage_id=1,
    )
    req_state.stage_submit_ts[0] = 1.0
    output = SimpleNamespace(request_id="req-route", finished=True)

    await orchestrator._route_output(0, 0, output, req_state, None)

    orchestrator._forward_to_next_stage.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_segment_does_not_complete_final_output_stage() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = True
    orchestrator._pd_pair = None
    orchestrator._cfg_tracker = SimpleNamespace(
        is_companion=lambda _request_id: False,
        has_companions=lambda _request_id: False,
        cleanup_parent=lambda _request_id: [],
    )
    orchestrator.stage_pools = [SimpleNamespace(final_output=True)]
    orchestrator.output_async_queue = asyncio.Queue()
    orchestrator._cleanup_request_ids = AsyncMock()

    req_state = OrchestratorRequestState(
        request_id="req-segment-final-output",
        sampling_params_list=[SamplingParams(max_tokens=1)],
        final_stage_id=0,
        final_output_stage_ids={0},
    )
    req_state.streaming.enabled = True
    req_state.streaming.segments[0] = StreamingSegmentState(finished=True)
    output = SimpleNamespace(
        request_id=req_state.request_id,
        finished=True,
    )

    await orchestrator._route_output(0, 0, output, req_state, None)

    assert req_state.finished_final_output_stage_ids == set()
    orchestrator._cleanup_request_ids.assert_not_awaited()
    routed = orchestrator.output_async_queue.get_nowait()
    assert routed.finished is False


# ---------------------------------------------------------------------------
# Dead-processor hint must not flag the selected sync hook.
# ---------------------------------------------------------------------------


def _prewarm_orchestrator_with_client(client: Any) -> Orchestrator:
    orch = object.__new__(Orchestrator)
    orch.stage_pools = [
        SimpleNamespace(
            stage_client=client,
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        )
    ]
    return orch


def test_warn_dead_input_processors_skips_selected_sync_hook() -> None:
    """The selected sync hook used as the custom hook is NOT dead.

    In non-async mode a stage wiring the same ``*_token_only`` processor as
    both ``custom_process_input_func`` and ``sync_process_input_func`` must not
    be reported as never-invoked (it is the active forward processor).
    """
    from vllm_omni.engine import orchestrator as orch_module

    def _sync_hook(source_outputs, prompt=None, requires_multimodal_data=False):  # type: ignore[no-untyped-def]
        return []

    path = f"{_sync_hook.__module__}.{_sync_hook.__qualname__}"

    class _Client:
        custom_process_input_func = _sync_hook
        sync_process_input_func = path

    orch = _prewarm_orchestrator_with_client(_Client())
    orch.async_chunk = False
    with patch.object(orch_module.logger, "warning") as mock_warn:
        orch._warn_dead_input_processors()
    dead_warnings = [call for call in mock_warn.call_args_list if call.args and "never invoked" in call.args[0]]
    assert dead_warnings == []


def test_warn_dead_input_processors_flags_distinct_sync_hook() -> None:
    """A custom hook overridden by a *different* sync hook is still reported dead."""
    from vllm_omni.engine import orchestrator as orch_module

    def _custom_hook(source_outputs, prompt=None, requires_multimodal_data=False):  # type: ignore[no-untyped-def]
        return []

    def _sync_hook(source_outputs, prompt=None, requires_multimodal_data=False):  # type: ignore[no-untyped-def]
        return []

    path = f"{_sync_hook.__module__}.{_sync_hook.__qualname__}"

    class _Client:
        custom_process_input_func = _custom_hook
        sync_process_input_func = path

    orch = _prewarm_orchestrator_with_client(_Client())
    orch.async_chunk = False
    with patch.object(orch_module.logger, "warning") as mock_warn:
        orch._warn_dead_input_processors()
    dead_warnings = [call for call in mock_warn.call_args_list if call.args and "never invoked" in call.args[0]]
    assert len(dead_warnings) == 1


# ---------------------------------------------------------------------------
# Prewarm placeholder builder is resolved into stage metadata at init time.
# ---------------------------------------------------------------------------


class _FakeProcessorSpec:
    """Minimal stand-in for ``ProcessorSpec`` with a controllable ``fn``."""

    def __init__(self, fn: Any) -> None:
        self.path = "fake"
        self.kind = "placeholder_prompt_builder"
        self.fn = fn


def test_prewarm_builder_resolved_once_into_stage_metadata() -> None:
    """The dual-entry builder is resolved exactly once, at metadata time.

    ``_resolve_prewarm_placeholder_builder`` reads ``build_prewarm_placeholder``
    off the resolved ``sync_process_input_func`` fn (with the module-level
    sibling fallback) so the orchestrator never re-resolves per async request.
    """
    from vllm_omni.engine.stage_init_utils import _resolve_prewarm_placeholder_builder

    def _build_prewarm_placeholder(**kwargs):  # type: ignore[no-untyped-def]
        return {"prompt_token_ids": [0]}

    def _sync_hook(source_outputs, prompt=None, requires_multimodal_data=False):  # type: ignore[no-untyped-def]
        return []

    _sync_hook.build_prewarm_placeholder = _build_prewarm_placeholder  # type: ignore[attr-defined]

    with patch(
        "vllm_omni.engine.stage_init_utils.resolve_processor",
        return_value=_FakeProcessorSpec(_sync_hook),
    ) as mock_resolve:
        builder = _resolve_prewarm_placeholder_builder(
            sync_path="pkg.mod.fake_token_only",
            custom_fn=None,
            stage_config=None,
        )
    assert builder is _build_prewarm_placeholder
    assert mock_resolve.call_count == 1


def test_prewarm_builder_miss_warns_once_then_keeps_inline_estimate() -> None:
    """A stage WITHOUT a builder warns once at init and the prewarm keeps the
    inline estimate — the orchestrator never re-resolves or re-warns."""
    from vllm_omni.engine import stage_init_utils as siu
    from vllm_omni.engine.stage_init_utils import _resolve_prewarm_placeholder_builder

    def _sync_hook(source_outputs, prompt=None, requires_multimodal_data=False):  # type: ignore[no-untyped-def]
        return []

    fake_module = SimpleNamespace()
    with (
        patch(
            "vllm_omni.engine.stage_init_utils.resolve_processor",
            return_value=_FakeProcessorSpec(_sync_hook),
        ) as mock_resolve,
        patch("importlib.import_module", return_value=fake_module) as mock_import,
        patch.object(siu.logger, "warning") as mock_warn,
    ):
        builder = _resolve_prewarm_placeholder_builder(
            sync_path="pkg.mod.fake_token_only",
            custom_fn=None,
            stage_config=None,
        )
    assert builder is None
    assert mock_resolve.call_count == 1
    assert mock_import.call_count == 1
    inline_estimates = [call for call in mock_warn.call_args_list if call.args and "inline estimate" in call.args[0]]
    assert len(inline_estimates) == 1


# ---------------------------------------------------------------------------
# Regression: configured async_chunk_prewarm_prompt_len must survive the
# inline-estimate fallback (amy-why-3459 review of HEAD 22e3a258).
# ---------------------------------------------------------------------------


def _override_orchestrator(client: Any, prewarm_len: int = 37) -> Orchestrator:
    """Orchestrator whose downstream stage declares a token prefill boundary."""
    orch = object.__new__(Orchestrator)
    orch.stage_pools = [
        SimpleNamespace(
            stage_client=client,
            stage_vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(
                    max_model_len=64,
                    hf_config=SimpleNamespace(async_chunk_prewarm_prompt_len=prewarm_len),
                ),
            ),
        )
    ]
    return orch


#: Builder variants shared by the override table below (a None builder is the
#: "no dual-entry builder" miss case; a raising builder exercises the fallback).
def _build_prewarm_placeholder_3(**kwargs):  # type: ignore[no-untyped-def]
    return [{"prompt_token_ids": [0, 0, 0]}]


def _build_prewarm_placeholder_789(**kwargs):  # type: ignore[no-untyped-def]
    return [{"prompt_token_ids": [7, 8, 9]}]


def _broken_build_prewarm_placeholder(**kwargs):  # type: ignore[no-untyped-def]
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    "builder,prewarm_len,expected_ids",
    [
        # A stage WITHOUT a builder keeps the configured override through the
        # inline-estimate fallback (Nemotron's 37-token speaker-prompt prefill).
        pytest.param(None, 37, [0] * 37, id="builder-miss-keeps-override"),
        # A builder that RAISES falls back to the inline estimate AND keeps it.
        pytest.param(_broken_build_prewarm_placeholder, 37, [0] * 37, id="builder-failure-keeps-override"),
        # Builder hit must still honor the override (amy-why-3459 review of HEAD ad81ac3a).
        pytest.param(_build_prewarm_placeholder_3, 37, [0] * 37, id="builder-success-keeps-override"),
        # Without a configured override, a builder hit keeps its own output.
        pytest.param(_build_prewarm_placeholder_789, None, [7, 8, 9], id="no-override-keeps-builder-output"),
        # A configured length of 1 is not a prefill override: builder output wins.
        pytest.param(_build_prewarm_placeholder_3, 1, [0, 0, 0], id="len-one-not-a-prefill-override"),
    ],
)
def test_prewarm_prompt_len_override(builder, prewarm_len, expected_ids) -> None:
    from vllm_omni.engine import orchestrator as orch_module

    class _Client:
        sync_process_input_func = "pkg.mod.fake_token_only"

        def __init__(self) -> None:
            # Instance attribute: class-level functions bind self on access.
            self.prewarm_placeholder_builder = builder

    if prewarm_len is None:
        orch = _prewarm_orchestrator_with_client(_Client())  # no async_chunk_prewarm_prompt_len
    else:
        orch = _override_orchestrator(_Client(), prewarm_len=prewarm_len)
    req_state = OrchestratorRequestState(
        request_id="req-override",
        prompt={"prompt": "hi"},
        sampling_params_list=[SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)],
        final_stage_id=0,
    )

    with patch.object(orch_module.logger, "warning"):
        base_input = orch._build_prewarm_placeholder_input(0, "req-override", [1, 2], req_state)

    assert base_input["prompt_token_ids"] == expected_ids
    assert base_input["multi_modal_data"] is None
    assert base_input["mm_processor_kwargs"] is None


class _DeepcopySpy:
    """Sentinel media object that records any attempted deep-copy."""

    def __init__(self) -> None:
        self.deepcopy_calls = 0

    def __deepcopy__(self, memo: dict[int, Any]) -> _DeepcopySpy:  # type: ignore[type-arg]
        self.deepcopy_calls += 1
        return _DeepcopySpy()


def test_prewarm_placeholder_input_filters_discarded_media() -> None:
    """Discarded prompt fields (media) must never be deep-copied for a prewarm."""
    media = _DeepcopySpy()
    orch = _prewarm_orchestrator_with_client(SimpleNamespace(prewarm_placeholder_builder=None))
    req_state = OrchestratorRequestState(
        request_id="req-prewarm-media",
        prompt={
            "prompt_token_ids": [1, 2],
            "multi_modal_data": media,
            "mm_processor_kwargs": {"foo": "bar"},
        },
        sampling_params_list=[SamplingParams(max_tokens=1)],
        final_stage_id=0,
    )

    base_input = orch._build_prewarm_placeholder_input(0, "req-prewarm-media", [1, 2], req_state)

    assert base_input["multi_modal_data"] is None
    assert base_input["mm_processor_kwargs"] is None
    assert media.deepcopy_calls == 0


def test_prewarm_placeholder_input_shares_additional_information_tensors() -> None:
    """``additional_information`` containers are isolated but tensors stay shared."""
    tensor = torch.arange(4, dtype=torch.long)
    original = {"t": tensor, "l": [1, 2]}
    orch = _prewarm_orchestrator_with_client(SimpleNamespace(prewarm_placeholder_builder=None))
    req_state = OrchestratorRequestState(
        request_id="req-prewarm-additional",
        prompt={"prompt_token_ids": [1, 2], "additional_information": original},
        sampling_params_list=[SamplingParams(max_tokens=1)],
        final_stage_id=0,
    )

    base_input = orch._build_prewarm_placeholder_input(0, "req-prewarm-additional", [1, 2], req_state)

    copied = base_input["additional_information"]
    assert copied is not original
    assert copied["l"] is not original["l"]
    assert copied["t"] is tensor
    assert copied["t"].data_ptr() == tensor.data_ptr()
