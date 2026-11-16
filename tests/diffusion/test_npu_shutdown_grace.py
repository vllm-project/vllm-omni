# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded cleanup must allow NPU runtime finalizers to finish naturally."""

from types import SimpleNamespace

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
