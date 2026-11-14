# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from vllm import SamplingParams

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler, _reject_kstep_async_scheduling
from vllm_omni.engine.serialization import serialize_additional_information

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


@pytest.mark.parametrize(
    ("scheduler_ngram", "native_duplex", "temperature", "expected"),
    [
        (True, False, 0.0, [1, 2]),
        (True, True, 0.0, []),
        (True, False, 0.7, []),
        (False, False, 0.0, [1, 2, 3]),
        (False, True, 0.0, []),
        (False, False, 0.7, [1, 2, 3]),
    ],
)
def test_thinker_ngram_limits_drafts_before_accounting(scheduler_ngram, native_duplex, temperature, expected):
    scheduler = object.__new__(OmniARScheduler)
    scheduler._scheduler_ngram = object() if scheduler_ngram else None
    scheduler.requests = {
        "chat": SimpleNamespace(resumable=False, sampling_params=SamplingParams(temperature=0.0)),
        "duplex": SimpleNamespace(
            resumable=False,
            sampling_params=SamplingParams(temperature=temperature),
            additional_information=serialize_additional_information({"duplex": {"data_plane": native_duplex}}),
        ),
    }
    output = SimpleNamespace(
        num_scheduled_tokens={"chat": 4, "duplex": 1},
        total_num_scheduled_tokens=5,
        scheduled_spec_decode_tokens={"chat": [1, 2, 3]},
        num_spec_tokens_to_schedule=2,
    )

    scheduler._limit_thinker_drafts(output)

    assert output.scheduled_spec_decode_tokens == ({"chat": expected} if expected else {})
    assert output.num_scheduled_tokens == {"chat": 1 + len(expected), "duplex": 1}
    assert output.total_num_scheduled_tokens == 2 + len(expected)
