# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded cleanup must allow NPU runtime finalizers to finish naturally."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize(
    "npu,exit_after,expected_actions",
    [(True, 20.0, []), (False, 20.0, ["terminate"]), (True, float("inf"), ["terminate", "kill"])],
)
def test_platform_grace_and_bounded_escalation(monkeypatch, npu, exit_after, expected_actions):
    from vllm_omni.diffusion.executor import multiproc_executor as module

    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(module, "current_omni_platform", SimpleNamespace(is_npu=lambda: npu))

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.name = f"worker-{pid}"
            self.exit_at = exit_after
            self.actions = []
            self._forced_code = None

        @property
        def exitcode(self):
            return None if self.is_alive() else (self._forced_code or 0)

        def is_alive(self):
            return clock[0] < self.exit_at

        def join(self, timeout):
            clock[0] += min(timeout, max(0.0, self.exit_at - clock[0]))

        def terminate(self):
            self.actions.append("terminate")
            if exit_after != float("inf"):
                self.exit_at = clock[0]
                self._forced_code = -15

        def kill(self):
            self.actions.append("kill")
            self.exit_at = clock[0]
            self._forced_code = -9

    processes = [Process(i) for i in range(4)]
    cleaner = module._ExecutorShutdownCleaner(processes=processes)
    cleaner()
    assert all(p.actions == expected_actions for p in processes)
    assert cleaner.processes == []
    assert clock[0] <= 65.0, "The grace deadline is shared across all workers"
    if npu and exit_after == 20.0:
        assert all(p.exitcode == 0 for p in processes)
        assert clock[0] == 20.0


def test_later_rank_binding_failure_reaps_started_workers_and_closes_pipes(monkeypatch):
    from vllm_omni.diffusion.executor import multiproc_executor as module

    @contextmanager
    def affinity(config, rank, *, is_npu):
        if rank == 1:
            raise RuntimeError("no local CPUs")
        yield

    monkeypatch.setattr(module, "npu_worker_start_affinity", affinity)
    monkeypatch.setattr(module, "set_multiprocessing_worker_envs", Mock())
    monkeypatch.setattr(module.mp, "set_start_method", Mock())
    workers = [Mock(), Mock()]
    workers[0].is_alive.side_effect = [True, True, False]
    pipes = [(Mock(), Mock()), (Mock(), Mock())]
    executor = object.__new__(module.MultiprocDiffusionExecutor)
    executor._mp_context = SimpleNamespace(Pipe=Mock(side_effect=pipes), Process=Mock(side_effect=workers))
    executor.od_config = SimpleNamespace(num_gpus=2, worker_extension_cls=None)
    with pytest.raises(RuntimeError, match="no local CPUs"):
        executor._launch_workers(None, [None, None])
    workers[0].start.assert_called_once()
    workers[0].terminate.assert_called_once()
    workers[0].join.assert_called()
    workers[1].start.assert_not_called()
    for reader, writer in pipes:
        reader.close.assert_called_once()
        writer.close.assert_called_once()
