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
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

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
async def test_prewarm_uses_registered_build_prewarm_placeholder() -> None:
    """Async-chunk prewarm resolves ``sync_process_input_func`` (the
    ``*_token_only`` path) to ``build_prewarm_placeholder`` and uses its estimate.

    The placeholder length is ``stage0_only`` = the Qwen chat-template scan on
    the stage-0 prompt.  The synthetic ``[1, 2]`` prompt has no chat-template
    marker, so the scan yields 0 and the builder floors it to a 1-token
    placeholder (best-effort; the connector fixup path replaces it later).
    """
    orchestrator = object.__new__(Orchestrator)
    stage0 = FakePrewarmPool("sender")
    stage1 = FakePrewarmPool("receiver")
    # The real qwen3_omni sync placeholder builder exposes build_prewarm_placeholder
    # off the resolved fn (see test_placeholder_parity.py).
    stage1.stage_client = SimpleNamespace(  # type: ignore[attr-defined]
        sync_process_input_func=(
            "vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_token_only"
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
# P2 deep-dive: dead-processor hint must not flag the selected sync hook.
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
# P3 deep-dive: prewarm placeholder builder resolution is cached per stage.
# ---------------------------------------------------------------------------


class _FakeProcessorSpec:
    """Minimal stand-in for ``ProcessorSpec`` with a controllable ``fn``."""

    def __init__(self, fn: Any) -> None:
        self.path = "fake"
        self.kind = "placeholder_prompt_builder"
        self.fn = fn


def test_prewarm_builder_resolved_once_and_cached() -> None:
    """A stage with a builder resolves it once and caches it.

    Repeated async requests must not re-run the importlib resolution or the
    registry validation.
    """
    from vllm_omni.model_executor import stage_input_processors as sip

    def _build_prewarm_placeholder(**kwargs):  # type: ignore[no-untyped-def]
        return {"prompt_token_ids": [0]}

    def _sync_hook(source_outputs, prompt=None, requires_multimodal_data=False):  # type: ignore[no-untyped-def]
        return []

    _sync_hook.build_prewarm_placeholder = _build_prewarm_placeholder  # type: ignore[attr-defined]

    class _Client:
        sync_process_input_func = "pkg.mod.fake_token_only"

    orch = _prewarm_orchestrator_with_client(_Client())
    with patch.object(sip, "resolve_processor", return_value=_FakeProcessorSpec(_sync_hook)) as mock_resolve:
        first = orch._get_prewarm_placeholder_builder(0)
        second = orch._get_prewarm_placeholder_builder(0)
    assert first is _build_prewarm_placeholder
    assert second is _build_prewarm_placeholder
    assert mock_resolve.call_count == 1


def test_prewarm_builder_miss_warns_once_and_caches() -> None:
    """A stage WITHOUT a builder resolves once, warns once, then caches a miss.

    Qwen3-TTS and other models have no ``build_prewarm_placeholder``; without
    per-stage caching every async request re-resolved and re-warned.
    """
    from vllm_omni.engine import orchestrator as orch_module
    from vllm_omni.model_executor import stage_input_processors as sip

    def _sync_hook(source_outputs, prompt=None, requires_multimodal_data=False):  # type: ignore[no-untyped-def]
        return []

    class _Client:
        sync_process_input_func = "pkg.mod.fake_token_only"

    orch = _prewarm_orchestrator_with_client(_Client())
    fake_module = SimpleNamespace()
    with (
        patch.object(sip, "resolve_processor", return_value=_FakeProcessorSpec(_sync_hook)) as mock_resolve,
        patch("importlib.import_module", return_value=fake_module) as mock_import,
        patch.object(orch_module.logger, "warning") as mock_warn,
    ):
        first = orch._get_prewarm_placeholder_builder(0)
        second = orch._get_prewarm_placeholder_builder(0)
        third = orch._get_prewarm_placeholder_builder(0)
    assert first is None
    assert second is None
    assert third is None
    assert mock_resolve.call_count == 1
    assert mock_import.call_count == 1
    inline_estimates = [call for call in mock_warn.call_args_list if call.args and "inline estimate" in call.args[0]]
    assert len(inline_estimates) == 1


# ---------------------------------------------------------------------------
# Regression: configured async_chunk_prewarm_prompt_len must survive the
# inline-estimate fallback (amy-why-3459 review of HEAD 22e3a258).
# ---------------------------------------------------------------------------


def _override_orchestrator(client: Any) -> Orchestrator:
    """Orchestrator whose downstream stage declares a 37-token prefill boundary."""
    orch = object.__new__(Orchestrator)
    orch.stage_pools = [
        SimpleNamespace(
            stage_client=client,
            stage_vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(
                    max_model_len=64,
                    hf_config=SimpleNamespace(async_chunk_prewarm_prompt_len=37),
                ),
            ),
        )
    ]
    return orch


def test_prewarm_override_keeps_configured_length_on_builder_miss() -> None:
    """A stage WITHOUT a builder keeps ``async_chunk_prewarm_prompt_len``.

    Regression for amy-why-3459's review: the configured override (e.g. the
    Nemotron deployment's 37-token speaker-prompt prefill) must survive the
    inline-estimate fallback, not be replaced by the generic upstream estimate.
    """
    from vllm_omni.engine import orchestrator as orch_module
    from vllm_omni.model_executor import stage_input_processors as sip

    def _sync_hook(source_outputs, prompt=None, requires_multimodal_data=False):  # type: ignore[no-untyped-def]
        return []

    class _Client:
        sync_process_input_func = "pkg.mod.fake_token_only"

    orch = _override_orchestrator(_Client())
    fake_module = SimpleNamespace()
    req_state = OrchestratorRequestState(
        request_id="req-override-miss",
        prompt={"prompt": "hi"},
        sampling_params_list=[SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)],
        final_stage_id=0,
    )

    with (
        patch.object(sip, "resolve_processor", return_value=_FakeProcessorSpec(_sync_hook)),
        patch("importlib.import_module", return_value=fake_module),
        patch.object(orch_module.logger, "warning"),
    ):
        base_input = orch._build_prewarm_placeholder_input(0, "req-override-miss", [1, 2], req_state)

    # The configured 37-token prefill boundary wins over the tiny [1, 2] estimate.
    assert base_input["prompt_token_ids"] == [0] * 37
    assert base_input["multi_modal_data"] is None
    assert base_input["mm_processor_kwargs"] is None


def test_prewarm_override_keeps_configured_length_on_builder_failure() -> None:
    """A builder that RAISES falls back to the inline estimate AND keeps the override."""
    from vllm_omni.engine import orchestrator as orch_module
    from vllm_omni.model_executor import stage_input_processors as sip

    def _broken_build_prewarm_placeholder(**kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    def _sync_hook(source_outputs, prompt=None, requires_multimodal_data=False):  # type: ignore[no-untyped-def]
        return []

    _sync_hook.build_prewarm_placeholder = _broken_build_prewarm_placeholder  # type: ignore[attr-defined]

    class _Client:
        sync_process_input_func = "pkg.mod.fake_token_only"

    orch = _override_orchestrator(_Client())
    req_state = OrchestratorRequestState(
        request_id="req-override-fail",
        prompt={"prompt": "hi"},
        sampling_params_list=[SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)],
        final_stage_id=0,
    )

    with (
        patch.object(sip, "resolve_processor", return_value=_FakeProcessorSpec(_sync_hook)),
        patch.object(orch_module.logger, "warning"),
    ):
        base_input = orch._build_prewarm_placeholder_input(0, "req-override-fail", [1, 2], req_state)

    # Builder failure must not lose the configured override either.
    assert base_input["prompt_token_ids"] == [0] * 37
    assert base_input["multi_modal_data"] is None
    assert base_input["mm_processor_kwargs"] is None
