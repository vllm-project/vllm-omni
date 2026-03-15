import asyncio

import pytest

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _OneShotStage:
    def __init__(self, result):
        self._result = result
        self._returned = False

    def try_collect(self):
        if self._returned:
            return None
        self._returned = True
        return self._result


@pytest.mark.asyncio
async def test_output_handler_cleans_shm_for_orphan_result(monkeypatch: pytest.MonkeyPatch):
    omni = AsyncOmni.__new__(AsyncOmni)
    omni.output_handler = None
    omni.request_states = {}
    omni._companion_to_parent = {}
    omni._rpc_results = {}
    omni.stage_list = [
        _OneShotStage(
            {
                "request_id": "orphan-req",
                "stage_id": 0,
                "engine_outputs_shm": {"name": "fake-shm", "size": 123},
            }
        )
    ]

    cleanup_calls: list[tuple[dict, str]] = []

    def _fake_cleanup(container: dict, shm_key: str = "engine_outputs_shm") -> bool:
        cleanup_calls.append((container, shm_key))
        return True

    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.cleanup_shm_from_ipc_meta",
        _fake_cleanup,
        raising=False,
    )

    omni._run_output_handler()
    await asyncio.sleep(0.02)

    assert cleanup_calls, "Expected SHM cleanup to be invoked for orphaned output"
    assert cleanup_calls[0][1] == "engine_outputs_shm"

    omni.output_handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await omni.output_handler
