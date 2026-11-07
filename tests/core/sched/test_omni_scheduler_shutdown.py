# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniGenerationScheduler])
@pytest.mark.parametrize("adapter_state", ["present", "absent", "failed"])
def test_shutdown_releases_omni_and_upstream_resources(monkeypatch, scheduler_cls, adapter_state):
    events = []
    monkeypatch.setattr(Scheduler, "shutdown", lambda self: events.append("upstream"))
    scheduler = scheduler_cls.__new__(scheduler_cls)

    def release():
        events.append("omni")
        if adapter_state == "failed":
            raise RuntimeError("connector failure")

    scheduler.chunk_transfer_adapter = None if adapter_state == "absent" else SimpleNamespace(shutdown=release)
    if adapter_state == "failed":
        with pytest.raises(RuntimeError, match="connector failure"):
            scheduler.shutdown()
    else:
        scheduler.shutdown()
    assert events == (["upstream"] if adapter_state == "absent" else ["omni", "upstream"])
