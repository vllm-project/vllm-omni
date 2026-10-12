# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Omni scheduler must select the same model runner as its worker."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Scheduler(OmniSchedulerMixin):
    """Minimal carrier for the shared init hook both omni schedulers call."""

    def __init__(
        self,
        *,
        initial_runner_v2: bool,
        configured_runner_v2: bool,
    ) -> None:
        self.use_v2_model_runner = initial_runner_v2
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                async_chunk=False,
                stage_id=0,
                pooling_output_decoder=None,
                use_v2_model_runner=configured_runner_v2,
            )
        )
        self._init_omni_io_scheduling_state()


@pytest.mark.parametrize("initial", [True, False])
@pytest.mark.parametrize("configured", [True, False])
def test_scheduler_follows_explicit_model_runner_config(
    initial: bool,
    configured: bool,
) -> None:
    scheduler = _Scheduler(
        initial_runner_v2=initial,
        configured_runner_v2=configured,
    )
    assert scheduler.use_v2_model_runner is configured
