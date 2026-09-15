# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeARDiffusionRunner:
    def __init__(self) -> None:
        self.resets: list[str] = []
        self.closes: list[str] = []

    def reset_session(self, session_id: str) -> None:
        self.resets.append(session_id)

    def close_session(self, session_id: str) -> None:
        self.closes.append(session_id)


class FakeRegularRunner:
    pass


def make_worker(runner: object) -> DiffusionWorker:
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = runner
    return worker


def test_worker_delegates_ar_session_lifecycle_to_runner() -> None:
    runner = FakeARDiffusionRunner()
    worker = make_worker(runner)

    assert worker.reset_ar_diffusion_session("world-1") is True
    assert worker.close_ar_diffusion_session("world-1") is True
    assert runner.resets == ["world-1"]
    assert runner.closes == ["world-1"]


def test_worker_reports_unsupported_runner_without_model_assumptions() -> None:
    worker = make_worker(FakeRegularRunner())

    assert worker.reset_ar_diffusion_session("world-1") is False
    assert worker.close_ar_diffusion_session("world-1") is False


def test_worker_rejects_empty_session_id() -> None:
    worker = make_worker(FakeARDiffusionRunner())

    with pytest.raises(ValueError, match="session_id"):
        worker.close_ar_diffusion_session("")


class FakePipelineOnlyRunner:
    """A regular diffusion runner: no AR session methods, but a stateful model.

    This is the encode stage. Broadcasting the lifecycle RPC to it used to do
    nothing at all, which is how encode kept VAE history for a session whose KV
    had already been released.
    """

    def __init__(self, pipeline: object) -> None:
        self.pipeline = pipeline


class FakeStatefulPipeline:
    def __init__(self) -> None:
        self.resets: list[str] = []
        self.closes: list[str] = []
        self.states: dict[str, object] = {}

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self.resets.append(session_id)
        self.states.pop(session_id, None)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self.closes.append(session_id)
        self.states.pop(session_id, None)


class FakeStatelessPostprocessPipeline(FakeStatefulPipeline):
    """The trailing decode stage: acknowledges, allocates nothing."""

    def _get_or_create_state(self, session_id: str) -> None:
        raise AssertionError("a stateless postprocess stage must not create session state")


class FakeRaisingPipeline:
    def close_ar_diffusion_session(self, session_id: str) -> None:
        raise RuntimeError("pool cleanup failed")

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        raise RuntimeError("pool cleanup failed")


def test_worker_falls_back_to_the_model_hook_when_the_runner_has_no_ar_session():
    pipeline = FakeStatefulPipeline()
    pipeline.states["world-1"] = object()
    worker = make_worker(FakePipelineOnlyRunner(pipeline))

    assert worker.reset_ar_diffusion_session("world-1") is True
    assert worker.close_ar_diffusion_session("world-1") is True

    assert pipeline.resets == ["world-1"]
    assert pipeline.closes == ["world-1"]
    assert pipeline.states == {}


def test_worker_prefers_the_runner_so_kv_is_released_before_the_model_is_notified():
    pipeline = FakeStatefulPipeline()
    runner = FakeARDiffusionRunner()
    runner.pipeline = pipeline
    worker = make_worker(runner)

    assert worker.close_ar_diffusion_session("world-1") is True

    assert runner.closes == ["world-1"]
    # The runner owns the ordering (KV, then the model); the worker must not
    # invoke the model hook a second time.
    assert pipeline.closes == []


def test_stateless_postprocess_stage_acknowledges_an_idempotent_no_op():
    pipeline = FakeStatelessPostprocessPipeline()
    worker = make_worker(FakePipelineOnlyRunner(pipeline))

    assert worker.close_ar_diffusion_session("world-1") is True
    assert worker.close_ar_diffusion_session("world-1") is True
    assert pipeline.states == {}


def test_worker_preserves_model_hook_exceptions():
    worker = make_worker(FakePipelineOnlyRunner(FakeRaisingPipeline()))

    with pytest.raises(RuntimeError, match="pool cleanup failed"):
        worker.close_ar_diffusion_session("world-1")
    with pytest.raises(RuntimeError, match="pool cleanup failed"):
        worker.reset_ar_diffusion_session("world-1")


def test_coordinated_lifecycle_rpc_records_no_release_event():
    """The coordinator already knows about a cleanup it asked for; recording it
    would make the coordinator fan the same cleanup out again."""
    from vllm_omni.experimental.ar_diffusion.release_events import ARDiffusionReleaseEventLog

    log = ARDiffusionReleaseEventLog(stage_id=1)
    log.set_ready()

    class LoggingRunner:
        def suppress_release_events(self, session_id: str):
            return log.coordinated(session_id)

        def close_session(self, session_id: str) -> None:
            log.record(session_id, reason="close")

    worker = make_worker(LoggingRunner())

    assert worker.close_ar_diffusion_session("world-1") is True
    assert log.pending() == []

    # A release the runner decided on its own is still reported.
    log.record("world-1", reason="lru_eviction")
    assert [event["reason"] for event in log.pending()] == ["lru_eviction"]


def test_worker_release_event_rpc_passes_through_to_the_runner():
    class EventRunner:
        def __init__(self) -> None:
            self.acked: list[str] = []

        def get_ar_diffusion_release_events(self):
            return [{"event_id": "rel-1-0", "session_id": "world-1", "reason": "lru_eviction"}]

        def ack_ar_diffusion_release_events(self, event_ids):
            self.acked.extend(event_ids)
            return len(event_ids)

    runner = EventRunner()
    worker = make_worker(runner)

    assert worker.get_ar_diffusion_release_events() == [
        {"event_id": "rel-1-0", "session_id": "world-1", "reason": "lru_eviction"}
    ]
    assert worker.ack_ar_diffusion_release_events(["rel-1-0"]) == 1
    assert runner.acked == ["rel-1-0"]


def test_worker_release_event_rpc_is_empty_without_runner_support():
    worker = make_worker(FakeRegularRunner())

    assert worker.get_ar_diffusion_release_events() == []
    assert worker.ack_ar_diffusion_release_events(["rel-1-0"]) == 0
    assert worker.register_ar_diffusion_generation("world-1", 3) is False


def test_worker_generation_registration_falls_back_to_the_pipeline():
    class GenerationPipeline:
        def __init__(self) -> None:
            self.generations: dict[str, int] = {}

        def register_session_generation(self, session_id: str, generation: int) -> bool:
            self.generations[session_id] = generation
            return True

    pipeline = GenerationPipeline()
    worker = make_worker(FakePipelineOnlyRunner(pipeline))

    assert worker.register_ar_diffusion_generation("world-1", 7) is True
    assert pipeline.generations == {"world-1": 7}

    with pytest.raises(ValueError, match="session_id"):
        worker.register_ar_diffusion_generation("", 7)
