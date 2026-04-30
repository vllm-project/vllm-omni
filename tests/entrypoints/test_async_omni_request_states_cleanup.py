"""Integration tests for AsyncOmni.generate() request_states cleanup.

Verifies that request_states is properly cleaned up on ALL exit paths
(normal completion, stage error, cancellation), preventing memory leaks
in long-running servers.

These tests mock the engine layer and exercise the generate() code path
to verify request_states is always cleaned up via the finally block.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vllm import SamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_async_omni():
    """Create an AsyncOmni with a fully mocked engine layer."""
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    # Patch OmniBase.__init__ to skip heavy engine creation
    with patch.object(AsyncOmni, "__init__", lambda self, *a, **kw: None):
        omni = AsyncOmni.__new__(AsyncOmni)
        AsyncOmni.__init__(omni)

    # Set up minimal attributes that generate() needs
    omni.request_states = {}
    omni.log_stats = False
    omni.output_modalities = ["text"]
    omni.async_chunk = False
    omni.default_sampling_params_list = [SamplingParams()]
    omni._paused = False
    omni._pause_cond = asyncio.Condition()
    omni.final_output_task = MagicMock()  # pretend handler is running

    # Mock engine
    omni.engine = MagicMock()
    omni.engine.num_stages = 1
    omni.engine.add_request_async = AsyncMock()
    omni.engine.abort_async = AsyncMock()
    omni.engine.is_alive.return_value = True
    omni.engine.stage_configs = [{}]

    # Mock _stage_meta_list used by _compute_final_stage_id
    import types

    omni._stage_meta_list = [
        types.SimpleNamespace(
            final_output=True,
            final_output_type="text",
            stage_type="llm",
        )
    ]

    return omni


@pytest.fixture()
def async_omni():
    return _make_async_omni()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_states_cleaned_on_stage_error(async_omni):
    """Stage returning an error dict must raise RuntimeError AND clean up
    request_states so there is no memory leak."""

    request_id = "req-error"

    async def _fake_process_results(self, req_id, metrics, final_sid, rst, wst):
        # Simulate what _process_orchestrator_results does on error
        raise RuntimeError({"request_id": req_id, "error": "simulated stage failure"})
        # Make it a generator
        yield  # pragma: no cover

    with patch.object(
        type(async_omni),
        "_process_orchestrator_results",
        _fake_process_results,
    ):
        with pytest.raises(RuntimeError):
            async for _ in async_omni.generate(
                prompt="hello",
                request_id=request_id,
                sampling_params_list=[SamplingParams()],
            ):
                pass

    # Core assertion: request_states must be empty after error
    assert request_id not in async_omni.request_states, (
        "request_states leaked after stage error - memory leak!"
    )


@pytest.mark.asyncio
async def test_request_states_cleaned_on_normal_completion(async_omni):
    """Normal completion should also clean up request_states."""

    request_id = "req-ok"

    async def _fake_process_results(self, req_id, metrics, final_sid, rst, wst):
        yield MagicMock()  # one output

    with patch.object(
        type(async_omni),
        "_process_orchestrator_results",
        _fake_process_results,
    ):
        outputs = []
        async for out in async_omni.generate(
            prompt="hello",
            request_id=request_id,
            sampling_params_list=[SamplingParams()],
        ):
            outputs.append(out)

    assert len(outputs) == 1
    assert request_id not in async_omni.request_states, (
        "request_states leaked after normal completion!"
    )


@pytest.mark.asyncio
async def test_request_states_cleaned_on_cancellation(async_omni):
    """CancelledError (client disconnect) should clean up request_states."""

    request_id = "req-cancelled"

    async def _fake_process_results(self, req_id, metrics, final_sid, rst, wst):
        # Block indefinitely to simulate waiting for engine output
        await asyncio.sleep(999)
        yield  # pragma: no cover

    with patch.object(
        type(async_omni),
        "_process_orchestrator_results",
        _fake_process_results,
    ):
        gen = async_omni.generate(
            prompt="hello",
            request_id=request_id,
            sampling_params_list=[SamplingParams()],
        )
        task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert request_id not in async_omni.request_states, (
        "request_states leaked after cancellation!"
    )


@pytest.mark.asyncio
async def test_multiple_requests_isolated_cleanup(async_omni):
    """Error on one request must not affect another request's state."""

    async def _fake_process_results_error(self, req_id, metrics, final_sid, rst, wst):
        raise RuntimeError({"request_id": req_id, "error": f"stage error for {req_id}"})
        yield  # pragma: no cover

    with patch.object(
        type(async_omni),
        "_process_orchestrator_results",
        _fake_process_results_error,
    ):
        with pytest.raises(RuntimeError):
            async for _ in async_omni.generate(
                prompt="hello",
                request_id="req-1",
                sampling_params_list=[SamplingParams()],
            ):
                pass

    # req-1 should be cleaned up
    assert "req-1" not in async_omni.request_states

    with patch.object(
        type(async_omni),
        "_process_orchestrator_results",
        _fake_process_results_error,
    ):
        with pytest.raises(RuntimeError):
            async for _ in async_omni.generate(
                prompt="hello",
                request_id="req-2",
                sampling_params_list=[SamplingParams()],
            ):
                pass

    # Both should be cleaned up
    assert "req-2" not in async_omni.request_states
    assert len(async_omni.request_states) == 0
