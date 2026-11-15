# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import _reject_kstep_async_scheduling

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_async_scheduler_rejects_armed_kstep_stage():
    """An async scheduler driving the K-step decode fails at construction.

    Review follow-up (PR #9733): the combination was known-broken but only
    died inside the first request, taking every in-flight request with it.
    The guard raises at startup instead.
    """

    class _Scheduler:
        def _talker_kstep_armed(self) -> bool:
            return True

    with pytest.raises(ValueError, match="async_scheduling: false"):
        _reject_kstep_async_scheduling(_Scheduler())


def test_async_scheduler_allows_unarmed_stage():
    """A scheduler without the K-step spec block keeps async scheduling."""

    class _Scheduler:
        def _talker_kstep_armed(self) -> bool:
            return False

    _reject_kstep_async_scheduling(_Scheduler())
