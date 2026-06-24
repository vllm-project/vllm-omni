"""Unit tests for generation streaming session replacement.

These tests pin the behavior of `_update_request_as_session` against
current vLLM `Request` / `StreamingUpdate` (and Omni patches). When upgrading
vLLM, failures here should highlight incompatible changes to request state or
update payloads early.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / StreamingUpdate are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.outputs import OmniConnectorOutput

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SkippedWaitingStub:
    def __contains__(self, request: Request) -> bool:
        return False


class _ChunkTransferAdapterStub:
    def __init__(self) -> None:
        self.segment_finished_requests: set[str] = set()


class _SchedulerStub(OmniGenerationScheduler):
    """Minimal scheduler surface required by OmniGenerationScheduler."""

    def __init__(self, *, log_stats: bool = False) -> None:
        self.num_waiting_for_streaming_input = 0
        self.log_stats = log_stats
        self.chunk_transfer_adapter = _ChunkTransferAdapterStub()
        self.skipped_waiting = _SkippedWaitingStub()

    def _enqueue_waiting_request(self, session: Request) -> None:
        raise AssertionError("unexpected enqueue for skipped_waiting miss")


def _make_request(**kwargs) -> Request:
    sp = SamplingParams(max_tokens=8)
    defaults = dict(
        request_id="req-mixin-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    defaults.update(kwargs)
    return Request(**defaults)


def _make_update(**kwargs) -> StreamingUpdate:
    sp_new = SamplingParams(max_tokens=16)
    defaults = dict(
        mm_features=None,
        prompt_token_ids=[10, 20],
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=sp_new,
    )
    defaults.update(kwargs)
    return StreamingUpdate(**defaults)


class TestReplaceSessionWithStreamingUpdate:
    def test_resets_tokens_and_prompt_from_update(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        session.append_output_token_ids([7, 8])
        session.num_computed_tokens = 99
        session.status = RequestStatus.WAITING_FOR_STREAMING_REQ

        update = _make_update(prompt_token_ids=[40, 41, 42])
        sched.num_waiting_for_streaming_input = 3
        sched._update_request_as_session(session, update)

        assert session._output_token_ids == []
        assert list(session._all_token_ids) == [40, 41, 42]
        assert session.prompt_token_ids == [40, 41, 42]
        assert session.num_computed_tokens == 0
        assert session.num_prompt_tokens == 3
        assert session.arrival_time == 200.0
        assert session.sampling_params is update.sampling_params
        assert session.status == RequestStatus.WAITING
        assert sched.num_waiting_for_streaming_input == 2

    def test_none_prompt_token_ids_becomes_empty(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        session.status = RequestStatus.RUNNING
        update = _make_update(prompt_token_ids=None)
        sched._update_request_as_session(session, update)

        assert session.prompt_token_ids == ()
        assert list(session._all_token_ids) == []
        assert session.num_prompt_tokens == 0
        assert sched.num_waiting_for_streaming_input == 0

    def test_additional_information_cleared_when_update_omits_it(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        if not hasattr(session, "additional_information"):
            pytest.skip("Request has no additional_information (Omni patch inactive?)")
        session.additional_information = {"keep": True}
        session.status = RequestStatus.RUNNING

        base = _make_update()
        if not hasattr(base, "additional_information"):
            pytest.skip("StreamingUpdate has no additional_information (Omni patch inactive?)")
        update = replace(base, additional_information=None)

        sched._update_request_as_session(session, update)
        assert session.additional_information is None

    def test_does_not_decrement_waiting_when_not_streaming_status(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        session.status = RequestStatus.RUNNING
        sched.num_waiting_for_streaming_input = 5
        sched._update_request_as_session(session, _make_update())
        assert sched.num_waiting_for_streaming_input == 5

    def test_records_queued_event_when_log_stats_enabled(self) -> None:
        sched = _SchedulerStub(log_stats=True)
        session = _make_request()
        session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        sched._update_request_as_session(session, _make_update())

        assert session.events
        assert session.events[-1].type == EngineCoreEventType.QUEUED


class _RealignSchedulerStub(OmniSchedulerMixin):
    """Minimal scheduler surface for ``_realign_request_status_to_queues``.

    Pins ``self.requests`` / ``self.running`` / ``self.waiting`` to the
    same shapes the upstream ``Scheduler.finish_requests`` reads, so the
    helper sees realistic state without spinning a real scheduler.
    """

    def __init__(
        self,
        *,
        requests: dict,
        running: list,
        waiting: list,
    ) -> None:
        self.requests = requests
        self.running = running
        self.waiting = waiting


class TestRealignRequestStatusToQueues:
    """Regression for the residual hang described in
    https://github.com/vllm-project/vllm-omni/pull/3774 -- chunk-transfer
    adapter's ``requests_origin_status`` table goes stale on the
    ``waiting → running`` admit transition, and an abort that lands
    before the next deque round-trip stomps stale ``WAITING`` onto a
    request that actually lives in ``self.running``. After
    ``max_num_seqs`` such aborts every ``input_batch`` slot is leaked and
    new requests hang at ``chunks=0``.
    """

    def test_running_with_stale_waiting_status_is_realigned_to_running(
        self,
    ) -> None:
        """The exact race the hang reproduces: a request lives in
        ``self.running`` but its ``status`` is ``WAITING``. After
        realign, status must be ``RUNNING`` so upstream
        ``Scheduler.finish_requests`` removes it from ``self.running``.
        """
        req = _make_request(request_id="req-stale")
        req.status = RequestStatus.WAITING  # stale -- actually in running

        sched = _RealignSchedulerStub(
            requests={req.request_id: req},
            running=[req],
            waiting=[],
        )
        sched._realign_request_status_to_queues([req.request_id])

        assert req.status == RequestStatus.RUNNING

    def test_waiting_with_stale_running_status_is_realigned_to_waiting(
        self,
    ) -> None:
        """Symmetric case: request lives in ``self.waiting`` but status
        is ``RUNNING``. Upstream's else branch should run, so realign to
        ``WAITING``.
        """
        req = _make_request(request_id="req-stale-2")
        req.status = RequestStatus.RUNNING  # stale -- actually in waiting

        sched = _RealignSchedulerStub(
            requests={req.request_id: req},
            running=[],
            waiting=[req],
        )
        sched._realign_request_status_to_queues([req.request_id])

        assert req.status == RequestStatus.WAITING

    def test_already_aligned_status_is_left_unchanged(self) -> None:
        """Healthy case: status matches actual queue. No status mutation
        means no spurious side effects.
        """
        req_running = _make_request(request_id="req-r")
        req_running.status = RequestStatus.RUNNING
        req_waiting = _make_request(request_id="req-w")
        req_waiting.status = RequestStatus.WAITING

        sched = _RealignSchedulerStub(
            requests={
                req_running.request_id: req_running,
                req_waiting.request_id: req_waiting,
            },
            running=[req_running],
            waiting=[req_waiting],
        )
        sched._realign_request_status_to_queues([req_running.request_id, req_waiting.request_id])

        assert req_running.status == RequestStatus.RUNNING
        assert req_waiting.status == RequestStatus.WAITING

    def test_unknown_request_id_is_skipped_silently(self) -> None:
        """Aligning ids that aren't in ``self.requests`` (already freed
        upstream, or never existed) must be a no-op.
        """
        sched = _RealignSchedulerStub(requests={}, running=[], waiting=[])
        sched._realign_request_status_to_queues(["never-existed"])  # no raise

    def test_request_in_neither_queue_is_left_unchanged(self) -> None:
        """If a tracked request is in neither queue (e.g. parked in a
        chunk-transfer deque), realign must not invent a status. The
        adapter / deque purge owns that surface; the realign helper is
        only for the admit-transition staleness.
        """
        req = _make_request(request_id="req-parked")
        req.status = RequestStatus.WAITING_FOR_CHUNK

        sched = _RealignSchedulerStub(
            requests={req.request_id: req},
            running=[],
            waiting=[],
        )
        sched._realign_request_status_to_queues([req.request_id])

        assert req.status == RequestStatus.WAITING_FOR_CHUNK

    def test_request_ids_str_is_treated_as_single_id(self) -> None:
        """Match upstream ``Scheduler.finish_requests`` resolution: a
        bare string is treated as one id, not iterated as characters.
        """
        req = _make_request(request_id="req-s")
        req.status = RequestStatus.WAITING

        sched = _RealignSchedulerStub(
            requests={req.request_id: req},
            running=[req],
            waiting=[],
        )
        sched._realign_request_status_to_queues(req.request_id)

        assert req.status == RequestStatus.RUNNING

    def test_request_ids_none_aligns_every_known_request(self) -> None:
        """``request_ids=None`` matches upstream's "all requests" path.
        Realign must walk every entry in ``self.requests`` and fix any
        stale status it finds.
        """
        stale = _make_request(request_id="req-stale-none")
        stale.status = RequestStatus.WAITING  # but actually in running

        clean = _make_request(request_id="req-clean-none")
        clean.status = RequestStatus.RUNNING

        sched = _RealignSchedulerStub(
            requests={
                stale.request_id: stale,
                clean.request_id: clean,
            },
            running=[stale, clean],
            waiting=[],
        )
        sched._realign_request_status_to_queues(None)

        assert stale.status == RequestStatus.RUNNING
        assert clean.status == RequestStatus.RUNNING

    def test_finished_request_is_skipped(self) -> None:
        """Already-finished requests must not be touched -- they may
        have legitimate finished statuses (FINISHED_STOPPED etc.) that
        a status flip would corrupt.
        """
        req = _make_request(request_id="req-finished")
        req.status = RequestStatus.FINISHED_STOPPED

        sched = _RealignSchedulerStub(
            requests={req.request_id: req},
            running=[req],  # not realistic, but we want to prove the guard fires
            waiting=[],
        )
        sched._realign_request_status_to_queues([req.request_id])

        assert req.status == RequestStatus.FINISHED_STOPPED


class TestPurgeFinishedFromRunning:
    """Regression for the residual ``self.running`` slot leak surface
    paired with ``_realign_request_status_to_queues``: even after
    realign + ``super().finish_requests`` runs, corner cases can leave
    already-finished or untracked entries in ``self.running`` -- e.g.
    a connector cleanup that pops from ``self.requests`` without
    unwinding ``self.running``, or a ``status`` mid-transition when
    finish ran. The defensive post-finish purge sweeps those residues
    so the worker's ``input_batch`` slot never pins a freed request.

    See https://github.com/vllm-project/vllm-omni/pull/3774 discussion
    and the residual-hang reproduction in that PR.
    """

    def test_finished_request_is_purged_from_running(self) -> None:
        """``is_finished()`` request lingering in ``self.running`` must
        be swept so its ``input_batch`` slot is freed."""
        finished = _make_request(request_id="req-finished")
        finished.status = RequestStatus.FINISHED_STOPPED  # is_finished() True

        sched = _RealignSchedulerStub(
            requests={finished.request_id: finished},
            running=[finished],
            waiting=[],
        )
        sched._purge_finished_from_running()

        assert sched.running == []

    def test_untracked_request_is_purged_from_running(self) -> None:
        """Request lingering in ``self.running`` but no longer present
        in ``self.requests`` (already freed by upstream / connector)
        must be swept."""
        untracked = _make_request(request_id="req-untracked")
        untracked.status = RequestStatus.RUNNING

        sched = _RealignSchedulerStub(
            requests={},  # already deleted from self.requests
            running=[untracked],
            waiting=[],
        )
        sched._purge_finished_from_running()

        assert sched.running == []

    def test_healthy_running_is_left_unchanged(self) -> None:
        """Live tracked running requests must not be touched -- the
        purge is defensive, not aggressive."""
        alive = _make_request(request_id="req-alive")
        alive.status = RequestStatus.RUNNING

        sched = _RealignSchedulerStub(
            requests={alive.request_id: alive},
            running=[alive],
            waiting=[],
        )
        sched._purge_finished_from_running()

        assert sched.running == [alive]

    def test_empty_running_is_noop(self) -> None:
        """Empty ``self.running`` must short-circuit cleanly."""
        sched = _RealignSchedulerStub(requests={}, running=[], waiting=[])
        sched._purge_finished_from_running()

        assert sched.running == []

    def test_mixed_alive_and_dead_keeps_only_alive_in_order(self) -> None:
        """Mixed ``self.running`` -- some alive, some finished, some
        untracked. Sweep keeps only the alive ones and preserves their
        relative order."""
        alive_a = _make_request(request_id="req-alive-a")
        alive_a.status = RequestStatus.RUNNING
        finished_b = _make_request(request_id="req-finished-b")
        finished_b.status = RequestStatus.FINISHED_STOPPED
        untracked_c = _make_request(request_id="req-untracked-c")
        untracked_c.status = RequestStatus.RUNNING
        alive_d = _make_request(request_id="req-alive-d")
        alive_d.status = RequestStatus.RUNNING

        sched = _RealignSchedulerStub(
            requests={
                alive_a.request_id: alive_a,
                finished_b.request_id: finished_b,  # tracked but is_finished
                # untracked_c is NOT in self.requests
                alive_d.request_id: alive_d,
            },
            running=[alive_a, finished_b, untracked_c, alive_d],
            waiting=[],
        )
        sched._purge_finished_from_running()

        assert sched.running == [alive_a, alive_d]

    def test_in_place_mutation_keeps_helper_local_list_identity(self) -> None:
        """The slice-assign form (``self.running[:] = ...``) avoids an
        extra rebind inside this helper. Note that across a full
        ``finish_requests`` call upstream ``Scheduler.finish_requests``
        already rebinds ``self.running``, so global identity is not a
        contract -- this test only pins the local helper behavior so a
        future refactor away from in-place mutation is intentional.
        """
        finished = _make_request(request_id="req-finished")
        finished.status = RequestStatus.FINISHED_STOPPED
        alive = _make_request(request_id="req-alive")
        alive.status = RequestStatus.RUNNING

        sched = _RealignSchedulerStub(
            requests={
                finished.request_id: finished,
                alive.request_id: alive,
            },
            running=[finished, alive],
            waiting=[],
        )
        held_ref = sched.running
        sched._purge_finished_from_running()

        assert held_ref is sched.running
        assert held_ref == [alive]


class _FinishOnlyInputCoordinatorStub:
    def __init__(self) -> None:
        self.freed: list[str] = []
        self.finished_requests: set[str] = set()
        self.requests_with_ready_chunks: set[str] = set()
        self._completed_chunk_streams: set[str] = set()

    def free_finished_request(self, request_id: str) -> None:
        self.freed.append(request_id)

    def chunk_stream_completed(self, request_id: str) -> bool:
        return request_id in self.finished_requests or request_id in self._completed_chunk_streams


class _FinishOnlySchedulerStub(OmniGenerationScheduler):
    def __init__(self, request: Request) -> None:
        self._async_chunk_coordinator_active = True
        self._pending_finish_reqs: list[Request] = []
        self._deferred_terminal_chunk_req_ids: set[str] = set()
        self._deferred_terminal_request_metadata: dict[str, dict] = {}
        self._reqs_with_generation_output = {request.request_id}
        self.requests = {request.request_id: request}
        self.running = [request]
        self.waiting = SimpleNamespace(remove_requests=lambda _reqs: None)
        self.skipped_waiting = SimpleNamespace(remove_requests=lambda _reqs: None)
        self.input_coordinator = _FinishOnlyInputCoordinatorStub()
        self.chunk_transfer_adapter = None
        self.connector = None
        self.perf_metrics = None
        self.kv_cache_manager = SimpleNamespace(take_events=lambda: None)
        self.kv_event_publisher = SimpleNamespace(publish=lambda _batch: None)
        self.structured_output_manager = SimpleNamespace(should_advance=lambda _req: False)
        self.recompute_kv_load_failures = False
        self.finished_req_ids_dict: dict[int, set[str]] = {}
        self.log_stats = False
        self.freed: list[str] = []

    def _handle_stopped_request(self, request: Request) -> bool:
        return True

    def _free_request(self, request: Request, delay_free_blocks: bool = False):
        self.freed.append(request.request_id)
        self.requests.pop(request.request_id, None)
        return None

    def make_stats(self, *args, **kwargs):
        return None


def _empty_scheduler_output() -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _finish_only_model_output(request_id: str):
    return SimpleNamespace(
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={},
        routed_experts=None,
        req_ids=[],
        omni_connector_output=OmniConnectorOutput(
            chunk_finished_req_ids={request_id},
        ),
    )


class TestAsyncChunkFinishOnlyOutput:
    def test_multimodal_request_finishes_on_connector_only_terminal(self) -> None:
        request = _make_request(request_id="req-mm-terminal")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(request)

        outputs = sched.update_from_output(
            _empty_scheduler_output(),
            _finish_only_model_output(request.request_id),
        )

        client_outputs = outputs[request.client_index].outputs
        assert len(client_outputs) == 1
        assert client_outputs[0].request_id == request.request_id
        assert client_outputs[0].finish_reason.name == "STOP"
        assert sched.freed == [request.request_id]
        assert sched.input_coordinator.freed == [request.request_id]

    def test_multimodal_request_finishes_from_stored_coordinator_terminal(self) -> None:
        request = _make_request(request_id="req-mm-stored-terminal")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(request)
        sched.input_coordinator.finished_requests.add(request.request_id)

        outputs = sched.update_from_output(
            _empty_scheduler_output(),
            SimpleNamespace(
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=None,
                multimodal_outputs=None,
                num_nans_in_logits=None,
                kv_connector_output=None,
                cudagraph_stats=None,
                req_id_to_index={},
                routed_experts=None,
                req_ids=[],
                omni_connector_output=OmniConnectorOutput(),
            ),
        )

        client_outputs = outputs[request.client_index].outputs
        assert len(client_outputs) == 1
        assert client_outputs[0].request_id == request.request_id
        assert client_outputs[0].finish_reason.name == "STOP"
        assert sched.freed == [request.request_id]
        assert sched.input_coordinator.freed == [request.request_id]
