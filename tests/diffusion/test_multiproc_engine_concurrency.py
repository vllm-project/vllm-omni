# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import multiprocessing as mp
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
import zmq
from vllm.v1.engine.exceptions import EngineDeadError

import vllm_omni.diffusion.worker.diffusion_worker as diffusion_worker_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.ipc import DIFFUSION_RPC_RESULT_ENVELOPE
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


# ───────────────────────────────────────────── helpers ─────────────────────


def _tagged_output(tag: str) -> DiffusionOutput:
    """Return a ``DiffusionOutput`` identifiable by its *error* field."""
    return DiffusionOutput(output=torch.tensor([0]), error=tag)


def _mock_request(tag: str):
    """Return a lightweight request object identifiable by *tag*."""
    return SimpleNamespace(
        request_id=tag,
        prompt=f"prompt_{tag}",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        diffusion_kv_requests=None,
    )


def _make_executor(num_gpus: int = 1):
    """Create a ``MultiprocDiffusionExecutor`` without launching workers.

    Returns ``(executor, request_queue, result_queue)``.
    """
    od_cfg = SimpleNamespace(num_gpus=num_gpus, streaming_output=False, step_execution=True)
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = od_cfg

    req_q: queue.Queue = queue.Queue()
    res_q: queue.Queue = queue.Queue()

    mock_broadcast_mq = SimpleNamespace(enqueue=req_q.put)

    mock_rmq = SimpleNamespace(dequeue=lambda timeout=None: res_q.get(timeout=timeout if timeout is not None else 10))

    executor._broadcast_mq = mock_broadcast_mq
    executor._result_mq = mock_rmq
    executor._closed = False
    executor._processes = []
    executor._is_failed = False
    executor._failure_callbacks = []
    return executor, req_q, res_q


def _make_engine(num_gpus: int = 1):
    """Create a lightweight ``DiffusionEngine`` wired to mocked executor."""
    executor, req_q, res_q = _make_executor(num_gpus)
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(streaming_output=False)
    sched = RequestScheduler()
    sched.initialize(SimpleNamespace())
    engine.scheduler = sched
    engine.executor = executor
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine._closed = False
    engine._loop_started = False
    engine._rpc_queue = queue.Queue()
    engine.abort_queue = queue.Queue()
    engine.execute_fn = executor.execute_batch
    return engine, executor, req_q, res_q


def _start_worker(req_q, res_q, count=2):
    """Simulate workers: read *count* requests from *req_q* and put
    tagged ``DiffusionOutput``s on *res_q* (FIFO order).
    """

    def _run():
        for _ in range(count):
            try:
                req = req_q.get(timeout=10)
            except queue.Empty:
                break
            method = req.get("method", "")
            args = req.get("args", ())
            if method == "execute_model_batch" and args and isinstance(args[0], DiffusionSchedulerOutput):
                sched_output = args[0]
                runner_outputs = []
                for nr in sched_output.scheduled_new_reqs:
                    tag = f"result_for_{nr.request_id}"
                    runner_outputs.append(
                        RunnerOutput(request_id=nr.request_id, finished=True, result=_tagged_output(tag))
                    )
                res_q.put(BatchRunnerOutput.from_list(runner_outputs))
            elif method in {"generate", "execute_model"} and args and hasattr(args[0], "request_id"):
                tag = f"result_for_{args[0].request_id}"
                res_q.put(_tagged_output(tag))
            elif args:
                tag = f"result_for_{args[0]}"
                res_q.put(_tagged_output(tag))
            else:
                tag = f"result_for_{method}"
                res_q.put(_tagged_output(tag))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _inject_interleave(executor):
    """Monkey-patch ``executor._broadcast_mq.enqueue`` so that:

    * The thread named **thread_a** *blocks* after its enqueue until the
      thread named **thread_b** has finished entirely.
    * All other threads pass through unblocked.

    Returns ``(a_enqueued: Event, b_complete: Event)`` for wiring.
    """
    a_enqueued = threading.Event()
    b_complete = threading.Event()
    orig_enqueue = executor._broadcast_mq.enqueue  # points to req_q.put

    def _controlled(item):
        orig_enqueue(item)
        if threading.current_thread().name == "thread_a":
            a_enqueued.set()  # tell B: "A has enqueued"
            b_complete.wait(1)  # block A until B finishes

    executor._broadcast_mq.enqueue = _controlled
    return a_enqueued, b_complete


# ───────────────── concurrent request execution ─────────────────


class TestConcurrentRequestExecution:
    """Concurrent request execution should not swap results."""

    def test_results_are_correctly_routed(self):
        engine, executor, req_q, res_q = _make_engine()
        a_enqueued, b_complete = _inject_interleave(executor)
        wt = _start_worker(req_q, res_q, count=2)

        results: dict[str, DiffusionOutput] = {}

        def _a():
            results["A"] = engine.add_req_and_wait_for_response(_mock_request("A"))

        def _b():
            a_enqueued.wait(5)  # wait for A to enqueue
            results["B"] = engine.add_req_and_wait_for_response(_mock_request("B"))
            b_complete.set()  # release A

        ta = threading.Thread(target=_a, name="thread_a", daemon=True)
        tb = threading.Thread(target=_b, name="thread_b", daemon=True)
        ta.start()
        tb.start()
        ta.join(10)
        tb.join(10)
        wt.join(5)

        # With correct (locked) implementation both assertions hold.
        # The bug causes them to be swapped.
        assert results["A"].error == "result_for_A"
        assert results["B"].error == "result_for_B"


# ───────────────── request-mode dispatch (per-request vs batch) ─────────────


def _make_sched_output(*request_ids: str) -> DiffusionSchedulerOutput:
    """Build a request-mode scheduler output with the given new requests."""
    new_reqs = [
        NewRequestData(
            request_id=rid,
            req=OmniDiffusionRequest(
                prompt=f"prompt_{rid}",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
                request_id=rid,
            ),
        )
        for rid in request_ids
    ]
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=len(new_reqs),
        num_waiting_reqs=0,
    )


class TestRequestModeDispatch:
    """Request-batch-capable dispatch uses ``execute_batch`` for request-mode cycles."""

    @pytest.mark.parametrize("request_ids", [("solo",), ("A", "B", "C")])
    def test_request_batch_capable_pipeline_uses_execute_batch(self, request_ids):
        engine, executor, _, _ = _make_engine()
        executor.execute_request = Mock(return_value="per-request")
        executor.execute_batch = Mock(return_value="batch")
        engine.execute_fn = executor.execute_batch

        out = engine.execute_fn(_make_sched_output(*request_ids))

        executor.execute_batch.assert_called_once()
        assert out == "batch"
        executor.execute_request.assert_not_called()

    @pytest.mark.parametrize("request_ids", [("solo",), ("A", "B")])
    def test_batch_path_routes_results_through_worker(self, request_ids):
        """End-to-end: a request-batch cycle goes out as one ``execute_model_batch``
        RPC and comes back as a per-request-routed ``BatchRunnerOutput``."""
        engine, executor, req_q, res_q = _make_engine()
        engine.execute_fn = executor.execute_batch
        wt = _start_worker(req_q, res_q, count=1)

        out = engine.execute_fn(_make_sched_output(*request_ids))
        wt.join(5)

        assert isinstance(out, BatchRunnerOutput)
        results = {ro.request_id: ro.result.error for ro in out.runner_outputs}
        assert results == {request_id: f"result_for_{request_id}" for request_id in request_ids}

    def test_dlo_dp_routes_multiple_requests_without_pipeline_batching(self):
        executor, _, _ = _make_executor(num_gpus=2)
        executor.od_config = SimpleNamespace(
            step_execution=False,
            parallel_config=SimpleNamespace(data_parallel_size=2),
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
        )
        executor.execute_request = Mock(return_value="dlo-dp")
        executor.collective_rpc = Mock()
        scheduler_output = _make_sched_output("A", "B")

        result = executor.execute_batch(scheduler_output)

        assert result == "dlo-dp"
        executor.execute_request.assert_called_once_with(scheduler_output)
        executor.collective_rpc.assert_not_called()

    def test_dlo_dp_multi_rank_reply_uses_synchronous_rpc_collection(self):
        executor, req_q, res_q = _make_executor(num_gpus=2)
        executor.od_config = SimpleNamespace(
            step_execution=False,
            parallel_config=SimpleNamespace(data_parallel_size=2),
        )
        executor._sync_result_buffer = res_q

        def worker():
            request = req_q.get(timeout=2)
            assert "rpc_id" not in request
            assert request["output_rank"] is None
            assert request["collect_rank_status"] is False
            wave_id = request["wave_id"]
            for dp_rank in range(2):
                res_q.put(
                    {
                        "dp_rank": dp_rank,
                        "output": _tagged_output(str(dp_rank)),
                        "wave_id": wave_id,
                    }
                )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        results = executor.collective_rpc(
            "execute_model",
            unique_reply_rank=None,
            exec_all_ranks=True,
        )
        thread.join(timeout=2)

        assert [result.error for result in results] == ["0", "1"]

    @pytest.mark.parametrize(
        ("tp_size", "sp_size", "pp_size", "cfg_size", "expected_primary_ranks"),
        [
            (2, 1, 1, 1, [0, 2]),
            (2, 2, 1, 1, [0, 4]),
            (1, 2, 2, 1, [0, 4]),
            (1, 1, 2, 2, [0, 4]),
        ],
    )
    def test_step_execution_reads_each_dp_primary_result_queue(
        self,
        tp_size,
        sp_size,
        pp_size,
        cfg_size,
        expected_primary_ranks,
    ):
        dp_size = 2
        num_gpus = dp_size * tp_size * sp_size * pp_size * cfg_size
        executor, _, _ = _make_executor(num_gpus=num_gpus)
        executor.od_config = SimpleNamespace(
            step_execution=True,
            parallel_config=SimpleNamespace(
                data_parallel_size=dp_size,
                tensor_parallel_size=tp_size,
                sequence_parallel_size=sp_size,
                pipeline_parallel_size=pp_size,
                cfg_parallel_size=cfg_size,
            ),
        )
        request: dict = {}
        executor._broadcast_mq = SimpleNamespace(enqueue=lambda value: request.update(value))

        def result_queue(global_rank):
            result_mq = MagicMock()
            result_mq.dequeue.side_effect = lambda timeout=None: {
                "dp_rank": global_rank,
                "output": _tagged_output(str(global_rank)),
                "wave_id": request["wave_id"],
            }
            return result_mq

        # Every queue returns a valid, rank-tagged result. If the executor
        # selects a non-primary queue, the returned rank exposes the mistake.
        executor._result_mqs = [result_queue(rank) for rank in range(num_gpus)]
        executor._result_mq = executor._result_mqs[0]

        results = executor.collective_rpc(
            "execute_model",
            unique_reply_rank=None,
            exec_all_ranks=True,
        )

        assert [result.error for result in results] == [str(rank) for rank in expected_primary_ranks]
        for rank, result_mq in enumerate(executor._result_mqs):
            assert result_mq.dequeue.call_count == (1 if rank in expected_primary_ranks else 0)

    @pytest.mark.parametrize("empty_prompt", ["", {"prompt": ""}])
    def test_dlo_dp_rejects_empty_prompt_before_worker_dispatch(self, empty_prompt):
        executor, _, _ = _make_executor(num_gpus=2)
        executor.od_config = SimpleNamespace(
            step_execution=False,
            parallel_config=SimpleNamespace(data_parallel_size=2),
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
        )
        executor.collective_rpc = Mock()
        scheduler_output = _make_sched_output("invalid", "valid")
        scheduler_output.scheduled_new_reqs[0].req.prompt = empty_prompt

        with pytest.raises(ValueError, match="non-empty prompt"):
            executor.execute_request(scheduler_output)

        executor.collective_rpc.assert_not_called()

    def test_dlo_dp_valid_wave_still_runs_after_rejected_wave(self):
        executor, _, _ = _make_executor(num_gpus=2)
        executor.od_config = SimpleNamespace(
            step_execution=False,
            parallel_config=SimpleNamespace(data_parallel_size=2),
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
        )
        executor.collective_rpc = Mock(return_value=[_tagged_output("A"), _tagged_output("B")])
        invalid_wave = _make_sched_output("invalid", "valid")
        invalid_wave.scheduled_new_reqs[0].req.prompt = ""

        with pytest.raises(ValueError, match="non-empty prompt"):
            executor.execute_request(invalid_wave)

        result = executor.execute_request(_make_sched_output("A", "B"))

        assert [output.result.error for output in result.runner_outputs] == ["A", "B"]
        executor.collective_rpc.assert_called_once()

    def test_dlo_dp_allows_shared_default_denoise_steps(self):
        executor, _, _ = _make_executor(num_gpus=2)
        executor.od_config = SimpleNamespace(
            step_execution=False,
            parallel_config=SimpleNamespace(data_parallel_size=2),
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
        )
        executor.collective_rpc = Mock(return_value=[_tagged_output("A"), _tagged_output("B")])
        scheduler_output = _make_sched_output("A", "B")
        for new_req in scheduler_output.scheduled_new_reqs:
            new_req.req.sampling_params.num_inference_steps = None

        result = executor.execute_request(scheduler_output)

        assert [output.result.error for output in result.runner_outputs] == ["A", "B"]
        executor.collective_rpc.assert_called_once()

    def test_dlo_dp_forwards_request_metadata_envelopes_as_one_wave(self):
        executor, _, _ = _make_executor(num_gpus=2)
        executor.od_config = SimpleNamespace(
            step_execution=False,
            parallel_config=SimpleNamespace(data_parallel_size=2),
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
        )
        executor.collective_rpc = Mock(return_value=[_tagged_output("A"), _tagged_output("B")])
        scheduler_output = _make_sched_output("A", "B")
        for new_req in scheduler_output.scheduled_new_reqs:
            new_req.diffusion_kv_metadata = SimpleNamespace(request_id=new_req.request_id)

        result = executor.execute_request(scheduler_output)

        assert [output.result.error for output in result.runner_outputs] == ["A", "B"]
        forwarded_envelopes = executor.collective_rpc.call_args.kwargs["args"][0]
        assert forwarded_envelopes is scheduler_output.scheduled_new_reqs
        assert [envelope.diffusion_kv_metadata.request_id for envelope in forwarded_envelopes] == ["A", "B"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("num_inference_steps", 2),
            ("guidance_scale", 7.5),
            ("width", 1024),
        ],
    )
    def test_dlo_dp_rejects_incompatible_collective_wave(self, field, value):
        executor, _, _ = _make_executor(num_gpus=2)
        executor.od_config = SimpleNamespace(
            step_execution=False,
            parallel_config=SimpleNamespace(data_parallel_size=2),
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
        )
        executor.collective_rpc = Mock()
        scheduler_output = _make_sched_output("A", "B")
        setattr(scheduler_output.scheduled_new_reqs[1].req.sampling_params, field, value)

        with pytest.raises(ValueError, match="compatible shape, CFG"):
            executor.execute_request(scheduler_output)

        executor.collective_rpc.assert_not_called()

    def test_dlo_dp_partial_reply_times_out_and_fails_closed(self, monkeypatch):
        from vllm_omni.diffusion.executor import multiproc_executor as executor_module

        executor, req_q, res_q = _make_executor(num_gpus=2)
        executor.od_config = SimpleNamespace(
            step_execution=False,
            parallel_config=SimpleNamespace(data_parallel_size=2),
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
        )
        executor._sync_result_buffer = res_q
        executor._fail_closed_on_dp_wave_timeout = Mock()
        monkeypatch.setattr(executor_module, "_DLO_DP_WAVE_TIMEOUT_S", 0.05)

        def reply_from_only_one_dp_rank():
            request = req_q.get(timeout=2.0)
            res_q.put(
                {
                    "dp_rank": 0,
                    "output": _tagged_output("rank-0"),
                    "wave_id": request["wave_id"],
                }
            )

        worker = threading.Thread(target=reply_from_only_one_dp_rank, daemon=True)
        worker.start()
        started = time.monotonic()

        result = executor.execute_request(_make_sched_output("A", "B"))

        elapsed = time.monotonic() - started
        worker.join(timeout=2.0)
        assert elapsed < 1.0
        assert all("timed out" in output.result.error for output in result.runner_outputs)
        executor._fail_closed_on_dp_wave_timeout.assert_called_once()
        assert isinstance(executor._fail_closed_on_dp_wave_timeout.call_args.args[0], TimeoutError)


# ───────────────── concurrent collective RPC ─────────────────


class TestConcurrentCollectiveRpc:
    """Concurrent ``collective_rpc()`` calls should not swap results."""

    def test_results_are_correctly_routed(self):
        engine, executor, req_q, res_q = _make_engine()
        a_enqueued, b_complete = _inject_interleave(executor)
        wt = _start_worker(req_q, res_q, count=2)

        results: dict[str, object] = {}

        def _a():
            results["A"] = engine.collective_rpc(
                "ping",
                args=("call_A",),
                unique_reply_rank=0,
            )

        def _b():
            a_enqueued.wait(5)
            results["B"] = engine.collective_rpc(
                "ping",
                args=("call_B",),
                unique_reply_rank=0,
            )
            b_complete.set()

        ta = threading.Thread(target=_a, name="thread_a", daemon=True)
        tb = threading.Thread(target=_b, name="thread_b", daemon=True)
        ta.start()
        tb.start()
        ta.join(10)
        tb.join(10)
        wt.join(5)

        assert results["A"].error == "result_for_call_A"
        assert results["B"].error == "result_for_call_B"


# ──────────── concurrent request execution and collective RPC ────────────


class TestConcurrentRequestExecutionAndCollectiveRpc:
    """Request execution and ``collective_rpc()`` should not swap results."""

    def test_results_are_correctly_routed(self):
        engine, executor, req_q, res_q = _make_engine()
        a_enqueued, b_complete = _inject_interleave(executor)
        wt = _start_worker(req_q, res_q, count=2)

        results: dict[str, object] = {}

        def _a():  # request execution path
            results["A"] = engine.add_req_and_wait_for_response(_mock_request("A"))

        def _b():  # collective_rpc path
            a_enqueued.wait(5)
            results["B"] = engine.collective_rpc(
                "ping",
                args=("call_B",),
                unique_reply_rank=0,
            )
            b_complete.set()

        ta = threading.Thread(target=_a, name="thread_a", daemon=True)
        tb = threading.Thread(target=_b, name="thread_b", daemon=True)
        ta.start()
        tb.start()
        ta.join(10)
        tb.join(10)
        wt.join(5)

        assert isinstance(results["A"], DiffusionOutput)
        assert results["A"].error == "result_for_A"
        assert results["B"].error == "result_for_call_B"


# ─────────────────────── serial operation coverage ───────────────────────


class TestSerialEngineOperations:
    """Verify correct behaviour for single-threaded (serial) usage.

    These tests must pass both **before** and **after** any concurrency fix
    is applied – they guard against regressions in the basic request path.
    """

    def test_serial_add_req_returns_correct_result(self):
        engine, _, req_q, res_q = _make_engine()
        wt = _start_worker(req_q, res_q, count=1)

        result = engine.add_req_and_wait_for_response(_mock_request("X"))
        wt.join(5)

        assert isinstance(result, DiffusionOutput)
        assert result.error == "result_for_X"

    def test_serial_add_req_multiple_sequential(self):
        engine, _, req_q, res_q = _make_engine()
        wt = _start_worker(req_q, res_q, count=3)

        for tag in ("one", "two", "three"):
            out = engine.add_req_and_wait_for_response(_mock_request(tag))
            assert out.error == f"result_for_{tag}"

        wt.join(5)

    def test_serial_collective_rpc_single_rank(self):
        engine, _, req_q, res_q = _make_engine()
        wt = _start_worker(req_q, res_q, count=1)

        result = engine.collective_rpc(
            "ping",
            args=("Y",),
            unique_reply_rank=0,
        )
        wt.join(5)

        assert result.error == "result_for_Y"

    def test_serial_collective_rpc_all_ranks(self):
        """``collective_rpc`` without *unique_reply_rank* returns a single
        response from rank 0 (only rank 0 has a result_mq).
        """
        engine, _, _, res_q = _make_engine(num_gpus=2)

        # Pre-populate one result (only rank 0 replies via result_mq)
        res_q.put(_tagged_output("rank0"))

        results = engine.collective_rpc("ping", args=("multi",))

        # Only 1 response expected since only rank 0 has result_mq
        assert len(results) == 1
        assert results[0].error == "rank0"

    def test_collective_rpc_all_rank_status_error_propagation(self):
        engine, _, _, res_q = _make_engine(num_gpus=2)

        res_q.put(
            {
                "type": DIFFUSION_RPC_RESULT_ENVELOPE,
                "method": "add_lora",
                "result": True,
                "rank_statuses": [
                    {"rank": 0, "ok": True, "bool_result": True},
                    {
                        "rank": 1,
                        "ok": False,
                        "error": "rank1 boom",
                        "error_type": "RuntimeError",
                        "traceback": "rank1 traceback",
                    },
                ],
            }
        )

        with pytest.raises(RuntimeError) as excinfo:
            engine.collective_rpc("add_lora")
        error = str(excinfo.value)
        assert "rank 1" in error
        assert "rank1 boom" in error
        assert "rank1 traceback" in error

    def test_collective_rpc_all_rank_bool_false_is_aggregated(self):
        engine, _, _, res_q = _make_engine(num_gpus=2)

        res_q.put(
            {
                "type": DIFFUSION_RPC_RESULT_ENVELOPE,
                "method": "remove_lora",
                "result": True,
                "rank_statuses": [
                    {"rank": 0, "ok": True, "bool_result": True},
                    {"rank": 1, "ok": True, "bool_result": False},
                ],
            }
        )

        assert engine.collective_rpc("remove_lora") == [False]

    def test_collective_rpc_collects_rank_status_only_for_control_plane_all_rank_rpc(self):
        executor, req_q, res_q = _make_executor(num_gpus=2)

        res_q.put(_tagged_output("forward"))
        result = executor.collective_rpc(
            "execute_stepwise",
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        forward_rpc = req_q.get_nowait()

        assert result.error == "forward"
        assert forward_rpc["exec_all_ranks"] is True
        assert forward_rpc["collect_rank_status"] is False

        res_q.put(
            {
                "type": DIFFUSION_RPC_RESULT_ENVELOPE,
                "method": "remove_lora",
                "result": True,
                "rank_statuses": [{"rank": 0, "ok": True, "bool_result": True}],
            }
        )
        assert executor.collective_rpc("remove_lora") == [True]
        control_rpc = req_q.get_nowait()

        assert control_rpc["exec_all_ranks"] is True
        assert control_rpc["collect_rank_status"] is True

    def test_serial_add_req_then_collective_rpc(self):
        engine, _, req_q, res_q = _make_engine()
        wt = _start_worker(req_q, res_q, count=2)

        gen_out = engine.add_req_and_wait_for_response(_mock_request("gen"))
        rpc_out = engine.collective_rpc(
            "ping",
            args=("rpc",),
            unique_reply_rank=0,
        )
        wt.join(5)

        assert gen_out.error == "result_for_gen"
        assert rpc_out.error == "result_for_rpc"

    def test_serial_add_req_error_propagation(self):
        """``add_req`` should raise when the worker reports an error."""
        engine, _, _, res_q = _make_engine()
        # Put an error response directly
        res_q.put({"status": "error", "error": "boom"})

        out = engine.add_req_and_wait_for_response(_mock_request("fail"))

        assert isinstance(out, DiffusionOutput)
        assert out.error is not None
        assert "boom" in out.error

    def test_serial_collective_rpc_error_propagation(self):
        """``collective_rpc`` should raise when the worker reports an error."""
        engine, _, _, res_q = _make_engine()
        res_q.put({"status": "error", "error": "kaboom"})

        with pytest.raises(RuntimeError, match="kaboom"):
            engine.collective_rpc("bad", unique_reply_rank=0)

    def test_collective_rpc_closed_executor_raises(self):
        engine, executor, _, _ = _make_engine()
        executor._closed = True

        with pytest.raises(RuntimeError, match="closed"):
            engine.collective_rpc("anything")


class TestWorkerProcRpcRankStatus:
    def _make_worker_proc(self, has_result_mq: bool = True):
        proc = object.__new__(WorkerProc)
        proc.gpu_id = 0
        proc.result_mq = object() if has_result_mq else None
        proc.worker = SimpleNamespace(execute_method=Mock(return_value=True))
        return proc

    def test_execute_rpc_returns_rank_status_envelope(self, monkeypatch):
        proc = self._make_worker_proc()

        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

        def _all_gather_object(out, local):
            out[0] = local
            out[1] = {
                "rank": 1,
                "ok": False,
                "error": "rank1 boom",
                "error_type": "RuntimeError",
                "traceback": "trace",
                "bool_result": None,
            }

        monkeypatch.setattr(torch.distributed, "all_gather_object", _all_gather_object)

        result, should_reply = proc._execute_rpc(
            {
                "method": "remove_lora",
                "args": (),
                "kwargs": {},
                "output_rank": 0,
                "exec_all_ranks": True,
                "collect_rank_status": True,
            }
        )

        assert should_reply is True
        assert result["type"] == DIFFUSION_RPC_RESULT_ENVELOPE
        assert result["result"] is True
        assert result["rank_statuses"][0]["rank"] == 0
        assert result["rank_statuses"][1]["rank"] == 1
        assert result["rank_statuses"][1]["ok"] is False

    def test_execute_rpc_local_exception_is_reported_in_envelope(self, monkeypatch):
        proc = self._make_worker_proc()
        proc.worker.execute_method = Mock(side_effect=RuntimeError("local boom"))
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

        result, should_reply = proc._execute_rpc(
            {
                "method": "add_lora",
                "args": (),
                "kwargs": {},
                "output_rank": 0,
                "exec_all_ranks": True,
                "collect_rank_status": True,
            }
        )

        assert should_reply is True
        assert result["type"] == DIFFUSION_RPC_RESULT_ENVELOPE
        assert len(result["rank_statuses"]) == 1
        status = result["rank_statuses"][0]
        assert status["rank"] == 0
        assert status["ok"] is False
        assert status["error"] == "local boom"
        assert status["error_type"] == "RuntimeError"
        assert status["bool_result"] is None
        assert "local boom" in status["traceback"]

    def test_execute_rpc_collect_exception_releases_traceback_and_device_cache(self, monkeypatch):
        proc = self._make_worker_proc()
        original = RuntimeError("local boom")
        proc.worker.execute_method = Mock(side_effect=original)
        gc_collect = Mock()
        mock_platform = Mock()
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
        monkeypatch.setattr(diffusion_worker_module.gc, "collect", gc_collect)
        monkeypatch.setattr(diffusion_worker_module, "current_omni_platform", mock_platform)

        proc._execute_rpc(
            {
                "method": "add_lora",
                "args": (),
                "kwargs": {},
                "output_rank": 0,
                "exec_all_ranks": True,
                "collect_rank_status": True,
            }
        )

        assert original.__traceback__ is None
        gc_collect.assert_called_once_with()
        mock_platform.empty_cache.assert_called_once_with()

    def test_execute_rpc_rejects_collect_rank_status_without_all_ranks(self):
        proc = self._make_worker_proc()

        with pytest.raises(ValueError, match="collect_rank_status requires exec_all_ranks=True"):
            proc._execute_rpc(
                {
                    "method": "ping",
                    "args": (),
                    "kwargs": {},
                    "output_rank": 0,
                    "exec_all_ranks": False,
                    "collect_rank_status": True,
                }
            )

        proc.worker.execute_method.assert_not_called()

    def test_execute_rpc_non_collect_exception_preserves_original_type(self):
        proc = self._make_worker_proc()
        original = ValueError("local boom")
        proc.worker.execute_method = Mock(side_effect=original)

        with pytest.raises(ValueError) as excinfo:
            proc._execute_rpc(
                {
                    "method": "bad",
                    "args": (),
                    "kwargs": {},
                    "output_rank": 0,
                    "exec_all_ranks": False,
                    "collect_rank_status": False,
                }
            )

        assert excinfo.value is original


# ───────── error handling: EngineDeadError propagation through layers ─────


class TestMultiprocExecutorRaisesEngineDeadError:
    """``collective_rpc`` raises ``EngineDeadError`` when the engine is failed."""

    def test_collective_rpc_raises_when_is_failed(self):
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor.od_config = SimpleNamespace(step_execution=True)
        executor._closed = False
        executor._broadcast_mq = MagicMock()
        executor._result_mq = MagicMock()
        executor._result_mq.dequeue = MagicMock(side_effect=TimeoutError)
        executor._is_failed = True

        with pytest.raises(EngineDeadError):
            executor.collective_rpc(
                "generate",
                args=(MagicMock(),),
                unique_reply_rank=0,
                exec_all_ranks=True,
            )

    def test_collective_rpc_raises_mid_dequeue_when_is_failed(self):
        """Worker dies while we are polling the dequeue loop."""
        executor, _, res_q = _make_executor()

        call_count = 0
        orig_dequeue = executor._result_mq.dequeue

        def _dying_dequeue(timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                executor._is_failed = True
                raise TimeoutError
            return orig_dequeue(timeout=timeout)

        executor._result_mq.dequeue = _dying_dequeue

        with pytest.raises(EngineDeadError):
            executor.collective_rpc(
                "generate",
                args=(MagicMock(),),
                unique_reply_rank=0,
                exec_all_ranks=True,
            )


class TestMultiprocExecutorStepStreamingOutput:
    """Streaming output uses step execution and one worker reply per step."""

    def test_execute_step_allows_streaming_output_mode(self):
        executor, req_q, res_q = _make_executor()
        executor.od_config = SimpleNamespace(streaming_output=True, step_execution=True)  # pyright: ignore[reportAttributeAccessIssue]
        runner_outputs = [
            RunnerOutput(
                request_id="sched-stream",
                step_index=1,
                finished=False,
                result=DiffusionOutput(output={"chunk": 0}, finished=False, chunk_index=0, total_chunks=2),
            ),
            RunnerOutput(
                request_id="sched-stream",
                step_index=2,
                finished=True,
                result=DiffusionOutput(output={"chunk": 1}, finished=True, chunk_index=1, total_chunks=2),
            ),
        ]
        scheduler_output = SimpleNamespace(
            scheduled_request_ids=["sched-stream"],
        )

        def _worker():
            for runner_output in runner_outputs:
                try:
                    req_q.get(timeout=10)
                except queue.Empty:
                    break
                res_q.put(runner_output)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        first: RunnerOutput = MultiprocDiffusionExecutor.execute_step(  # pyright: ignore[reportAssignmentType]
            executor,
            scheduler_output,  # pyright: ignore[reportArgumentType]
        )
        second: RunnerOutput = MultiprocDiffusionExecutor.execute_step(  # pyright: ignore[reportAssignmentType]
            executor,
            scheduler_output,  # pyright: ignore[reportArgumentType]
        )

        assert first is runner_outputs[0]
        assert first.result is not None
        assert first.result.output == {"chunk": 0}
        assert first.finished is False
        assert second is runner_outputs[1]
        assert second.result is not None
        assert second.result.output == {"chunk": 1}
        assert second.finished is True
        thread.join(timeout=2)


class TestDiffusionEngineDeadErrorPassthrough:
    """``DiffusionEngine.add_req_and_wait_for_response`` re-raises
    ``EngineDeadError`` from executor and wraps other errors."""

    def test_engine_dead_error_propagates(self):
        engine, executor, _, _ = _make_engine()
        engine.execute_fn = Mock(side_effect=EngineDeadError())

        with pytest.raises(EngineDeadError):
            engine.add_req_and_wait_for_response(_mock_request("dead"))

    def test_runtime_error_wrapped_in_output(self):
        engine, executor, _, _ = _make_engine()
        engine.execute_fn = Mock(side_effect=RuntimeError("gpu fault"))

        out = engine.add_req_and_wait_for_response(_mock_request("fault"))
        assert isinstance(out, DiffusionOutput)
        assert "gpu fault" in out.error


class TestStageDiffusionClientErrorPropagation:
    """Error surface behaviour of ``StageDiffusionClient``.

    Uses ``object.__new__`` to construct a client without spawning a real
    subprocess, then manually sets the fields needed for each test.
    """

    def _make_client(self, *, engine_dead=False, proc_alive=True):
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

        client = object.__new__(StageDiffusionClient)
        client.stage_id = 0
        client.final_output = True
        client.final_output_type = "image"
        client.default_sampling_params = None
        client.custom_process_input_func = None
        client.engine_input_source = None

        client._output_queue = asyncio.Queue()
        client._rpc_results = {}
        client._pending_rpcs = set()
        client._tasks = {}
        client._shutting_down = False
        client._engine_dead = engine_dead
        proc = MagicMock(
            is_alive=MagicMock(return_value=proc_alive),
            exitcode=1,
        )
        client._proc_manager = SimpleNamespace(proc=proc)
        client._request_socket = MagicMock()
        client._response_socket = MagicMock()
        client._encoder = MagicMock()
        client._decoder = MagicMock()

        return client

    @pytest.mark.asyncio
    async def test_add_request_raises_when_dead(self):
        client = self._make_client(engine_dead=True)

        with pytest.raises(EngineDeadError):
            await client.add_request_async("req-3", "test prompt", None)

    def test_check_health_raises_when_dead(self):
        client = self._make_client(engine_dead=True)

        with pytest.raises(EngineDeadError):
            client.check_health()

    def test_check_health_ok_when_alive(self):
        client = self._make_client()
        client.check_health()

    def test_get_output_raises_engine_dead_when_dead(self):
        """When ``_engine_dead`` is True and the output queue is empty,
        ``get_diffusion_output_nowait`` must raise ``EngineDeadError``."""
        client = self._make_client(engine_dead=True)
        # Simulate _drain_responses as a no-op (no ZMQ socket)
        client._response_socket.recv.side_effect = zmq.Again

        with pytest.raises(EngineDeadError):
            client.get_diffusion_output_nowait()

    def test_get_output_returns_none_when_alive_and_empty(self):
        """When the engine is alive and the queue is empty, return None."""
        client = self._make_client()
        client._response_socket.recv.side_effect = zmq.Again

        assert client.get_diffusion_output_nowait() is None

    def test_error_response_preserves_scheduler_metrics(self):
        from vllm_omni.metrics import definitions as metric_defs

        client = self._make_client()
        client._response_socket.recv.side_effect = [b"message", zmq.Again()]
        client._decoder.decode.return_value = {
            "type": "error",
            "request_id": "req-error",
            "error": "gpu fault",
            "metrics": {metric_defs.DIFFUSION_SCHEDULER_WAITING_KEY: 0},
        }

        output = client.get_diffusion_output_nowait()

        assert output is not None
        assert output.error == "gpu fault"
        assert output.metrics[metric_defs.DIFFUSION_SCHEDULER_WAITING_KEY] == 0

    def test_metrics_response_creates_metrics_only_output(self):
        from vllm_omni.metrics import definitions as metric_defs
        from vllm_omni.metrics.utils import DIFFUSION_METRICS_ONLY_REQUEST_ID

        client = self._make_client()
        client._response_socket.recv.side_effect = [b"message", zmq.Again()]
        client._decoder.decode.return_value = {
            "type": "metrics",
            "metrics": {metric_defs.DIFFUSION_SCHEDULER_WAITING_KEY: 0},
        }

        output = client.get_diffusion_output_nowait()

        assert output is not None
        assert output.request_id == DIFFUSION_METRICS_ONLY_REQUEST_ID
        assert output.error is None
        assert output.metrics[metric_defs.DIFFUSION_SCHEDULER_WAITING_KEY] == 0

    def test_check_health_raises_when_proc_dead(self):
        """``check_health`` detects a dead subprocess via the manager's proc
        and raises ``EngineDeadError``, setting ``_engine_dead`` as a
        side effect."""
        client = self._make_client(proc_alive=False)

        with pytest.raises(EngineDeadError, match="not alive"):
            client.check_health()

        assert client._engine_dead is True

    def test_get_output_raises_when_proc_dead(self):
        """When the subprocess has died (non-signal exit) and the output
        queue is empty, ``get_diffusion_output_nowait`` must raise
        ``EngineDeadError`` with the exit code."""
        client = self._make_client(proc_alive=False)
        client._response_socket.recv.side_effect = zmq.Again

        with pytest.raises(EngineDeadError, match="exit code"):
            client.get_diffusion_output_nowait()

        assert client._engine_dead is True

    def test_get_output_returns_none_on_signal_death(self):
        """When the subprocess was killed by a signal (exit code > 128),
        ``get_diffusion_output_nowait`` returns ``None`` and sets
        ``_shutting_down`` instead of raising."""
        client = self._make_client(proc_alive=False)
        client._proc_manager.proc.exitcode = 137  # SIGKILL (128 + 9)
        client._response_socket.recv.side_effect = zmq.Again

        result = client.get_diffusion_output_nowait()

        assert result is None
        assert client._shutting_down is True
        assert client._engine_dead is True

    def test_initialize_client_requires_replica_id(self):
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

        client = object.__new__(StageDiffusionClient)
        metadata = SimpleNamespace(
            stage_id=0,
            final_output=True,
            final_output_type="image",
            default_sampling_params=None,
            requires_multimodal_data=False,
            custom_process_input_func=None,
            engine_input_source=[],
        )

        with pytest.raises(AttributeError, match="replica_id"):
            client._initialize_client(
                metadata,
                "tcp://req",
                "tcp://resp",
                batch_size=1,
            )

    @pytest.mark.asyncio
    async def test_collective_rpc_async_returns_none_result(self, monkeypatch):
        client = self._make_client()
        client._owns_process = False
        client._proc = None
        client._encoder.encode.return_value = b"encoded-rpc"

        async def _unexpected_poll(*_, **__):
            raise AssertionError("collective_rpc_async should not keep polling after a None rpc_result arrives")

        client._response_poller = SimpleNamespace(poll=_unexpected_poll)

        rpc_id = "rpc-none"
        monkeypatch.setattr(
            "vllm_omni.diffusion.stage_diffusion_client.uuid.uuid4",
            lambda: SimpleNamespace(hex=rpc_id),
        )

        def _drain() -> None:
            client._rpc_results[rpc_id] = None

        client._drain_responses = _drain

        result = await client.collective_rpc_async(
            method="profile",
            timeout=0.01,
            args=(False, None),
        )

        assert result is None
        client._request_socket.send.assert_called_once_with(b"encoded-rpc")
        assert rpc_id not in client._pending_rpcs


class TestExecutorShutdownCleaner:
    def test_worker_joins_share_one_global_deadline(self, monkeypatch):
        from vllm_omni.diffusion.executor import multiproc_executor as executor_module

        class FakeProcess:
            def __init__(self, name):
                self.name = name
                self.alive = True
                self.terminated = False
                self.join_timeouts = []

            def is_alive(self):
                return self.alive

            def join(self, timeout):
                self.join_timeouts.append(timeout)
                if self.terminated:
                    self.alive = False

            def terminate(self):
                self.terminated = True

        monotonic = Mock(side_effect=[100.0, 100.0, 110.0, 120.0, 120.0, 124.0])
        monkeypatch.setattr(executor_module, "time", SimpleNamespace(monotonic=monotonic))
        first = FakeProcess("worker-0")
        second = FakeProcess("worker-1")
        cleaner = executor_module._ExecutorShutdownCleaner(processes=[first, second])

        cleaner()

        assert first.join_timeouts == [15.0, 5.0]
        assert second.join_timeouts == [5.0, 1.0]
        assert first.terminated and second.terminated
        assert not first.is_alive() and not second.is_alive()


# ───────── monitor thread & death sentinel integration tests ─────────


def _poll_flag(get_flag, *, timeout=5.0, interval=0.05) -> bool:
    """Poll until ``get_flag()`` returns True or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if get_flag():
            return True
        time.sleep(interval)
    return False


def _make_short_lived_process() -> mp.Process:
    """Spawn a real subprocess that exits immediately.

    The process must be started with ``"fork"`` (or the platform default)
    so that it can use a plain ``lambda`` as its target — ``"spawn"`` would
    fail to pickle it.
    """
    ctx = mp.get_context("fork")
    p = ctx.Process(target=lambda: None, name="ShortLivedWorker-0")
    p.start()
    return p


class TestMultiprocExecutorWorkerMonitor:
    """Integration tests for ``_start_worker_monitor``.

    Uses real short-lived subprocesses so that OS-level sentinel fd
    readiness is exercised end-to-end.
    """

    def test_worker_monitor_sets_is_failed_and_calls_callbacks_on_death(self):
        """When a worker process dies, the monitor thread must:
        1. Set ``_is_failed = True``
        2. Call ``shutdown()`` (which sets ``_closed = True``)
        3. Invoke all registered failure callbacks
        """
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._closed = False
        executor._is_failed = False
        executor._failure_callbacks = []
        executor._broadcast_mq = None
        executor._result_mq = None
        executor._shutdown_cleaner = None
        # Use a no-op so shutdown() doesn't crash on None resources.
        executor._finalizer = lambda: None
        # ------------------------------------------------------------------
        # Attributes added by remove_bubble_v2 (async D2H); shutdown() iterates
        # over them, so they need to exist even when constructed via __new__.
        executor._pump_stop = threading.Event()
        executor._futures_lock = threading.RLock()
        executor._rpc_futures = {}
        executor._output_futures = {}
        executor._batch_split_map = {}

        proc = _make_short_lived_process()
        executor._processes = [proc]

        callback_called = threading.Event()
        executor.register_failure_callback(callback_called.set)

        executor._start_worker_monitor()

        # Wait for the process to exit and the monitor to react.
        proc.join(5)
        assert _poll_flag(lambda: executor._is_failed), "_is_failed was not set"
        assert executor._closed, "shutdown() was not called"
        assert callback_called.wait(timeout=2), "failure callback was not invoked"
        assert executor.is_dead

    def test_worker_monitor_noop_when_already_closed(self):
        """If ``_closed`` is already True when the process dies (orderly
        shutdown), the monitor must *not* set ``_is_failed``."""
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._closed = True  # already shut down
        executor._is_failed = False
        executor._failure_callbacks = []
        executor._broadcast_mq = None
        executor._result_mq = None
        executor._shutdown_cleaner = None
        executor._finalizer = lambda: None

        proc = _make_short_lived_process()
        executor._processes = [proc]

        executor._start_worker_monitor()
        proc.join(5)

        # Give the monitor thread a chance to run (it should early-return).
        time.sleep(0.3)
        assert not executor._is_failed, "_is_failed should remain False on orderly shutdown"
        # Orderly close still reports dead via the public accessor.
        assert executor.is_dead


class TestStageDiffusionClientProcMonitor:
    """Integration test for ``StageDiffusionClient._start_proc_monitor``.

    Uses a real short-lived subprocess to verify the sentinel-based
    detection pipeline.
    """

    def test_proc_monitor_sets_engine_dead_on_process_death(self):
        """When the subprocess dies, the monitor thread must set
        ``_engine_dead = True``."""
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

        client = object.__new__(StageDiffusionClient)
        client.stage_id = 0
        client._shutting_down = False
        client._engine_dead = False

        proc = _make_short_lived_process()
        client._proc_manager = SimpleNamespace(proc=proc)

        client._start_proc_monitor()
        proc.join(5)

        assert _poll_flag(lambda: client._engine_dead), "_engine_dead was not set"


class TestDrainResponsesDeathSentinel:
    """Tests for death sentinel and error routing in
    ``StageDiffusionClient._drain_responses()``.
    """

    def _make_client(self):
        from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

        client = object.__new__(StageDiffusionClient)
        client.stage_id = 0
        client._engine_dead = False
        client._shutting_down = False
        client._output_queue = asyncio.Queue()
        client._rpc_results = {}
        client._pending_rpcs = set()
        client._response_socket = MagicMock()
        client._decoder = MagicMock()
        return client

    def test_drain_responses_sets_engine_dead_on_death_sentinel(self):
        """When ``_drain_responses`` receives the ``DIFFUSION_PROC_DEAD``
        sentinel, it must set ``_engine_dead = True`` and stop draining
        (decoder is never called)."""
        client = self._make_client()

        # First recv returns the death sentinel, second would be a normal
        # message but should never be reached.
        client._response_socket.recv.side_effect = [
            StageDiffusionProc.DIFFUSION_PROC_DEAD,
            b"should-not-be-reached",
        ]

        client._drain_responses()

        assert client._engine_dead is True
        client._decoder.decode.assert_not_called()

    def test_drain_responses_routes_error_as_omni_request_output(self):
        """When ``_drain_responses`` receives a ``{"type": "error"}`` message
        with a ``request_id``, it must place an ``OmniRequestOutput`` with
        the error on ``_output_queue``."""
        client = self._make_client()

        error_msg = {
            "type": "error",
            "request_id": "req-fail",
            "error": "gpu fault",
        }
        # First recv returns the encoded error, second raises zmq.Again.
        client._response_socket.recv.side_effect = [b"encoded-error", zmq.Again]
        client._decoder.decode.return_value = error_msg

        client._drain_responses()

        assert not client._output_queue.empty()
        output = client._output_queue.get_nowait()
        assert isinstance(output, OmniRequestOutput)
        assert output.request_id == "req-fail"
        assert output.error == "gpu fault"
        assert output.finished is True
