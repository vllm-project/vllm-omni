"""Regression tests for issue #4384.

`AsyncOmni.generate()` defaults to ``request_id=""``. Before PR #3953 that empty
default was forwarded verbatim, which caused two independent, client-invisible
failures:

* **P1 — silent payload loss.** The stage-0 scheduler enriches each request via a
  *truthiness* guard ``self.requests.get(req_id) if req_id else None``
  (``vllm_omni/core/sched/omni_ar_scheduler.py``). An empty id is falsy, so
  ``additional_information`` / ``prompt_embeds`` were dropped before reaching the
  worker — an ASR model would transcribe padding instead of audio, with no error.
* **P2 — engine-core crash.** Two concurrent ``generate()`` calls that both used
  the empty default collided on the same id in the core scheduler, tripping a
  ``"duplicate request id"`` assert that killed the engine core.

The fix rewrites the id in ``generate()`` via ``_get_unique_request_id()`` into a
non-empty, unique value before anything downstream sees it. These tests pin that
behavior so the regression cannot return.
"""
import asyncio
from types import SimpleNamespace

import pytest

from tests.entrypoints.test_async_omni import get_async_omni_instance, get_fake_add_request
from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.cpu]


async def _drive(omni, request_id):
    async for _ in omni.generate(
        prompt={"prompt": "test"},
        request_id=request_id,
        sampling_params_list=[SimpleNamespace()],
        output_modalities=["text"],
    ):
        pass


def test_p2_concurrent_default_ids_do_not_collide():
    """P2: many concurrent default-``""`` callers must each submit a distinct,
    non-empty id to the engine core (no duplicate-id assert)."""

    async def run():
        submitted_ids = []
        omni = get_async_omni_instance(fake_add_request=get_fake_add_request(submitted_ids))
        # All callers use the unsafe default request_id="" concurrently.
        await asyncio.gather(*(_drive(omni, "") for _ in range(50)))

        assert len(submitted_ids) == 50
        assert all(rid != "" for rid in submitted_ids), "empty id leaked to engine core"
        assert len(set(submitted_ids)) == 50, "duplicate ids -> engine-core collision"

    asyncio.run(run())


def test_p1_aliased_id_passes_scheduler_payload_guard():
    """P1: the aliased id must be truthy so the stage-0 scheduler's
    ``requests.get(req_id) if req_id else None`` guard resolves the live request
    and preserves its payload (instead of silently dropping it)."""

    class _LiveRequest:
        additional_information = {"audio": "REAL_AUDIO"}

    # Exact guard expression from omni_ar_scheduler.py's schedule() rewrap.
    def scheduler_guard(req_id, requests):
        return requests.get(req_id) if req_id else None

    # Old default would drop the payload: "" is falsy -> None.
    assert scheduler_guard("", {"": _LiveRequest()}) is None

    # Aliased id is non-empty -> request found, additional_information preserved.
    aliased = AsyncOmni._get_unique_request_id("")
    assert aliased != ""
    resolved = scheduler_guard(aliased, {aliased: _LiveRequest()})
    assert resolved is not None
    assert resolved.additional_information["audio"] == "REAL_AUDIO"


def test_get_unique_request_id_is_nonempty_and_unique():
    """The fix primitive itself: empty input yields non-empty, unique ids; a
    caller-supplied prefix is preserved for traceability."""
    ids = [AsyncOmni._get_unique_request_id("") for _ in range(10_000)]
    assert all(rid != "" for rid in ids)
    assert len(set(ids)) == len(ids)

    prefixed = AsyncOmni._get_unique_request_id("my-req")
    assert prefixed.startswith("my-req-")
