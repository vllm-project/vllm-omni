"""Integration tests for Omni.generate() error handling.

Verifies that the sync orchestrator correctly handles stage errors
without hanging indefinitely, and that the progress bar stays consistent
with completed_requests on all error/timeout paths.

These tests instantiate a real Omni with mocked stages and exercise
the actual generate() / _run_generation() code path end-to-end.
"""

import uuid
from queue import Empty, Queue

import pytest
from vllm import SamplingParams

from vllm_omni.entrypoints import utils as utils_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL = "riverclouds/qwen_image_random"


# ---------------------------------------------------------------------------
# Lightweight fakes
# ---------------------------------------------------------------------------


class _FakeQueue:
    """In-process queue replacing mp.Queue / ZmqQueue."""

    def __init__(self, maxsize=0):
        self._q: Queue = Queue(maxsize=maxsize)

    def put(self, item):
        self._q.put(item)

    def put_nowait(self, item):
        self._q.put_nowait(item)

    def get(self, timeout=None):
        return self._q.get(timeout=timeout)

    def get_nowait(self):
        return self._q.get_nowait()

    def empty(self):
        return self._q.empty()

    def close(self):
        pass


class _FakeStage:
    """Minimal OmniStage stand-in with real queues."""

    def __init__(self, stage_id: int = 0, final_output: bool = True, final_output_type: str = "text"):
        self.stage_id = stage_id
        self.final_output = final_output
        self.final_output_type = final_output_type
        self.stage_type = "llm"
        self.is_comprehension = False
        self.vllm_config = None
        self.tokenizer = None
        self.default_sampling_params = SamplingParams(temperature=1.0)
        self.engine_args = {"model_stage": None, "engine_output_type": None}
        self.engine_outputs = None
        self.prompt_expand_func = None
        self._in_q = _FakeQueue()
        self._out_q = _FakeQueue()

    def attach_queues(self, in_q, out_q):
        self._in_q = in_q
        self._out_q = out_q

    def init_stage_worker(self, model, *, is_async=False, **kwargs):
        self._out_q.put_nowait({"type": "stage_ready", "stage_id": self.stage_id})

    def stop_stage_worker(self):
        pass

    def submit(self, payload):
        self._in_q.put(payload)

    def try_collect(self):
        try:
            return self._out_q.get_nowait()
        except Empty:
            return None

    def set_engine_outputs(self, outputs):
        self.engine_outputs = outputs

    def process_engine_inputs(self, stage_list, prompts):
        return ["processed"]


class _TrackingTqdm:
    """Minimal tqdm replacement that tracks update() calls."""

    def __init__(self, iterable=None, **kwargs):
        self._iterable = iterable
        self.total = kwargs.get("total", 0)
        self.n = 0
        self.unit = "req"
        self.postfix = ""
        self.update_calls = []
        self.format_dict = {"elapsed": 1.0}

    def __iter__(self):
        return iter(self._iterable or [])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, n=1):
        self.n += n
        self.update_calls.append(n)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def omni_instance(monkeypatch):
    """Create an Omni instance with a single mocked stage."""
    from vllm_omni.entrypoints.omni import Omni

    fake_stage = _FakeStage(stage_id=0)

    monkeypatch.setattr(utils_module, "load_stage_configs_from_model", lambda model, base_engine_args=None: [])
    monkeypatch.setattr(utils_module, "resolve_model_config_path", lambda model: None)
    monkeypatch.setattr(Omni, "_start_stages", lambda self, model: None)
    monkeypatch.setattr(Omni, "_wait_for_stages_ready", lambda self, timeout=0: None)

    omni = Omni(model=MODEL, init_timeout=0)

    # Manually wire up the fake stage
    omni.stage_list = [fake_stage]
    omni._stage_out_queues = [fake_stage._out_q]
    omni._stage_in_queues = [fake_stage._in_q]
    omni.default_sampling_params_list = [fake_stage.default_sampling_params]
    omni.output_modalities = [fake_stage.final_output_type]
    omni.async_chunk = False
    omni.connectors = {}
    omni.log_stats = False
    return omni


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage_error_does_not_hang(omni_instance, monkeypatch):
    """Stage returning an error dict must cause generate() to terminate,
    not hang in an infinite polling loop.

    This is the core regression test: before the fix, a non-companion
    error never incremented completed_requests, so the while-loop
    spun forever."""
    import signal

    omni = omni_instance
    stage = omni.stage_list[0]

    # We need a predictable request_id so we can pre-populate the queue
    test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000001")
    monkeypatch.setattr(uuid, "uuid4", lambda: test_uuid)

    expected_rid = f"0_{test_uuid}"

    # Put error result into the stage output queue
    stage._out_q.put_nowait(
        {
            "request_id": expected_rid,
            "stage_id": 0,
            "error": "CUDA out of memory",
            "error_tb": "Traceback ...",
        }
    )

    # If the bug is present, generate() will hang forever.
    # Use a signal alarm as a hard timeout (5 seconds).
    timed_out = False

    def _alarm_handler(signum, frame):
        nonlocal timed_out
        timed_out = True
        raise TimeoutError("generate() hung — completed_requests not incremented")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(5)
    try:
        # generate() should complete without hanging
        omni.generate(
            prompts=["hello"],
            sampling_params_list=[SamplingParams()],
            use_tqdm=False,
        )
    except Exception:
        # generate() may raise due to close() or error propagation — that's fine,
        # what matters is it didn't hang
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    assert not timed_out, "generate() hung on stage error — infinite loop bug"


def test_stage_error_updates_progress_bar(omni_instance, monkeypatch):
    """When a stage returns an error, pbar.update(1) must be called
    so the progress bar stays consistent with completed_requests."""
    omni = omni_instance
    stage = omni.stage_list[0]

    test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000002")
    monkeypatch.setattr(uuid, "uuid4", lambda: test_uuid)

    expected_rid = f"0_{test_uuid}"

    stage._out_q.put_nowait(
        {
            "request_id": expected_rid,
            "stage_id": 0,
            "error": "simulated failure",
        }
    )

    # Inject tracking tqdm
    tracker = _TrackingTqdm()
    captured_tracker = [None]

    def _fake_tqdm(iterable=None, **kwargs):
        if iterable is not None:
            # This is the "Adding requests" tqdm — just pass through
            return iterable
        # This is the progress bar tqdm
        tracker.total = kwargs.get("total", 0)
        captured_tracker[0] = tracker
        return tracker

    try:
        omni.generate(
            prompts=["hello"],
            sampling_params_list=[SamplingParams()],
            use_tqdm=_fake_tqdm,
        )
    except Exception:
        pass

    assert captured_tracker[0] is not None, "pbar was never created"
    assert len(tracker.update_calls) == 1, (
        f"Expected exactly 1 pbar.update() call on error, got {len(tracker.update_calls)}"
    )


def test_normal_completion_updates_progress_bar(omni_instance, monkeypatch):
    """Normal completion should also call pbar.update(1)."""
    omni = omni_instance
    stage = omni.stage_list[0]

    test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000003")
    monkeypatch.setattr(uuid, "uuid4", lambda: test_uuid)

    expected_rid = f"0_{test_uuid}"

    # Fake a finished engine output
    fake_output = type(
        "FakeOutput",
        (),
        {
            "finished": True,
            "images": [],
            "prompt_token_ids": [1, 2, 3],
        },
    )()

    stage._out_q.put_nowait(
        {
            "request_id": expected_rid,
            "stage_id": 0,
            "engine_outputs": fake_output,
            "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 5.0},
        }
    )

    # Mock _load to return engine_outputs directly
    import vllm_omni.entrypoints.omni as omni_module

    monkeypatch.setattr(
        omni_module,
        "_load",
        lambda result, obj_key="", shm_key="": result.get(obj_key),
        raising=False,
    )

    tracker = _TrackingTqdm()

    def _fake_tqdm(iterable=None, **kwargs):
        if iterable is not None:
            return iterable
        tracker.total = kwargs.get("total", 0)
        return tracker

    try:
        omni.generate(
            prompts=["hello"],
            sampling_params_list=[SamplingParams()],
            use_tqdm=_fake_tqdm,
        )
    except Exception:
        pass

    assert len(tracker.update_calls) >= 1, (
        f"Expected pbar.update() on normal completion, got {len(tracker.update_calls)} calls"
    )


def test_multiple_errors_all_counted(omni_instance, monkeypatch):
    """Multiple requests all returning errors should each increment
    completed_requests and terminate without hanging."""
    import signal

    omni = omni_instance
    stage = omni.stage_list[0]

    # Use sequential UUIDs
    call_count = [0]
    uuids = [
        uuid.UUID("00000000-0000-0000-0000-000000000010"),
        uuid.UUID("00000000-0000-0000-0000-000000000011"),
        uuid.UUID("00000000-0000-0000-0000-000000000012"),
    ]

    def _next_uuid():
        idx = call_count[0]
        call_count[0] += 1
        return uuids[idx]

    monkeypatch.setattr(uuid, "uuid4", _next_uuid)

    # Pre-populate error results for all 3 requests
    for i in range(3):
        rid = f"{i}_{uuids[i]}"
        stage._out_q.put_nowait(
            {
                "request_id": rid,
                "stage_id": 0,
                "error": f"error for request {i}",
            }
        )

    timed_out = False

    def _alarm_handler(signum, frame):
        nonlocal timed_out
        timed_out = True
        raise TimeoutError("generate() hung with multiple errors")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(5)
    try:
        omni.generate(
            prompts=["a", "b", "c"],
            sampling_params_list=[SamplingParams()],
            use_tqdm=False,
        )
    except Exception:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    assert not timed_out, "generate() hung with multiple error results"
