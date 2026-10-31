# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Observe YuE2 lifecycle contracts inside a real engine worker."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


def assert_preempted_requests_resumed(observations: dict[str, Any]) -> None:
    """Every preempted request resumed through a reconcile-checked row.

    Each detected preemption enters ``unresumed`` and only a checked resumed
    row removes it, so with ``preemptions > 0`` this also implies
    ``resumed_checks > 0``.
    """
    assert not observations["unresumed"], observations


class Yue2LifecycleWorkerExtension:
    model_runner: "GPUARModelRunner"

    def start_yue2_probe(self):
        from unittest.mock import patch

        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available

        from vllm_omni.model_executor.models.yue2.yue2 import ABC_END, HOLD_TOKEN, MUSIC_END

        model = self.model_runner.get_model()
        self.yue2_probe = {
            "preemptions": 0,
            "rollbacks": 0,
            # Rows that resumed a preempted request and passed the
            # history/engine reconciliation checks below.
            "resumed_checks": 0,
            "checked_rows": 0,
            "mixed_steps": 0,
            "synthesis_started": False,
            "ar_full_replays": 0,
            "ar_piecewise_replays": 0,
        }
        graph_call = CUDAGraphWrapper.__call__

        def count_ar_replays(wrapper, *args, **kwargs):
            if is_forward_context_available():
                context = get_forward_context()
                mode = context.cudagraph_runtime_mode
                entry = wrapper.concrete_cudagraph_entries.get(context.batch_descriptor)
                if mode == wrapper.runtime_mode and entry is not None and entry.cudagraph is not None:
                    if mode == CUDAGraphMode.FULL:
                        self.yue2_probe["ar_full_replays"] += 1
                    elif mode == CUDAGraphMode.PIECEWISE:
                        self.yue2_probe["ar_piecewise_replays"] += 1
            return graph_call(wrapper, *args, **kwargs)

        # This worker is created for this test and exits with the engine.
        # Count vLLM AR replays, separately from model-owned sampler/NAR graphs.
        self.yue2_graph_patch = patch.object(CUDAGraphWrapper, "__call__", count_ar_replays)
        self.yue2_graph_patch.start()
        self.yue2_cancel_job = None
        previous: dict[str, int] = {}
        resuming: set[str] = set()
        self.yue2_resuming = resuming
        # Bounded log of preemption/resume decisions, returned with the probe
        # so a failed assertion shows why no rollback was observed.
        self.yue2_events: list[tuple] = []
        prepare = model.prepare_runner_inputs
        reconcile = model._reconcile_history
        queue = model._queue_synthesis
        sample = model.sample
        resolve = model._resolve_pending
        emitted: dict[str, int] = {}

        def resolve_pending():
            if model._pending is not None:
                event, host, states = model._pending
                event.synchronize()
                for state, token in zip(states, host[0].tolist()):
                    end = ABC_END if state.constants.phase == "abc" else MUSIC_END
                    expected = HOLD_TOKEN if token == end and not state.constants.skip_synthesis else token
                    assert emitted[state.request_id] == expected, "engine token and model history disagree"
            resolve()

        def sample_tokens(logits, sampling_metadata):
            result = sample(logits, sampling_metadata)
            phases = {model._states[rid].constants.phase for rid, _, _ in model._step_rows if rid in model._states}
            if len(phases) == 2:
                self.yue2_probe["mixed_steps"] += 1
            for (rid, _, _), row in zip(model._step_rows, result.sampled_token_ids.tolist()):
                emitted[rid] = row[0]
            return result

        def prepare_inputs(**kwargs):
            for rid, computed in zip(kwargs["req_ids"], kwargs["num_computed_tokens"]):
                if int(computed) < previous.get(rid, 0):
                    self.yue2_probe["preemptions"] += 1
                    resuming.add(rid)
                    state = model._states.get(rid)
                    pending = model._pending is not None and any(s.request_id == rid for s in model._pending[2])
                    self._yue2_event(
                        "preempt",
                        rid,
                        previous.get(rid, 0),
                        int(computed),
                        None if state is None else len(state.history),
                        pending,
                    )
                previous[rid] = int(computed)
            return prepare(**kwargs)

        def reconcile_history(state, accepted):
            ahead = len(state.history) + int(state.end_drawn) - accepted
            if ahead > 0:
                self.yue2_probe["rollbacks"] += 1
            reconcile(state, accepted)
            assert len(state.history) + int(state.end_drawn) <= accepted
            if not state.finish_ready and state.job is None and not state.finished:
                assert len(state.history) == accepted
                assert not state.truncated and not state.end_drawn
            self.yue2_probe["checked_rows"] += 1
            if state.request_id in resuming:
                # First non-discarded row after a recompute, checked above
                # whether or not the scheduler discarded an in-flight draw.
                # Only non-discarded rows reach _reconcile_history: sample()
                # skips discarded partial recompute chunks. Do not clear the
                # id on a discarded row, or `unresumed` stops catching a
                # recompute that never completes (see
                # test_preempted_request_that_never_resumes_is_rejected in
                # tests/worker/test_yue2_lifecycle_probe.py).
                resuming.discard(state.request_id)
                self.yue2_probe["resumed_checks"] += 1
                self._yue2_event("resume", state.request_id, accepted, ahead)

        def queue_synthesis(state):
            queue(state)
            if "cancel-synthesis" in state.request_id:
                job = state.job
                assert job.started and job.work is not None and job.keep
                assert job.event is None, "test must intercept an active synthesis job"
                # Keep an actual NAR job active until the driver cancels it.
                # Its first work units ran normally; stop pumping later units
                # so cancellation cannot race against a short song finishing.
                self.yue2_cancel_advance = job.advance
                job.advance = lambda: None
                self.yue2_cancel_job = job
                self.yue2_probe["synthesis_started"] = True

        model.prepare_runner_inputs = prepare_inputs
        model._reconcile_history = reconcile_history
        model._queue_synthesis = queue_synthesis
        model.sample = sample_tokens
        model._resolve_pending = resolve_pending
        return True

    def _yue2_event(self, *event) -> None:
        if len(self.yue2_events) < 64:
            self.yue2_events.append(event)

    def get_yue2_probe(self):
        result: dict[str, Any] = dict(self.yue2_probe)
        result["events"] = list(self.yue2_events)
        result["unresumed"] = sorted(self.yue2_resuming)
        job = self.yue2_cancel_job
        if job is not None:
            if job.cancelled:
                # Exercise the original pump after cleanup: it must not submit
                # further units even though this test retains the job object.
                self.yue2_cancel_advance()
            model = self.model_runner.get_model()
            result["cancelled"] = job.cancelled
            result["released"] = job._released and job.work is None and not job.keep and job.stream is None
            result["removed"] = (
                job.request_id not in model._states
                and model._synthesis.active is not job
                and job not in model._synthesis.waiting
            )
        return result
