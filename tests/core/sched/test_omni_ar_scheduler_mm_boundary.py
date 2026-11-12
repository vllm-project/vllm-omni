# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cached image embeddings must not allow a partial bidirectional prefill."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniARAsyncScheduler])
@pytest.mark.parametrize(
    "enabled,computed,count,shift,expected",
    [
        (True, 0, 3873, 0, 1227),  # Actual failing warmup request.
        (True, 1227, 2646, 0, 0),  # The previous step stopped at the image.
        (True, 1226, 2647, 1, 0),  # Shifted scheduling starts at the image.
        (True, 1227, 4982, 0, 4982),  # Next step can fit the whole image.
        (True, 0, 6251, 0, 6251),
        (True, 0, 100, 0, 100),  # Text before the image.
        (True, 6251, 1, 0, 1),  # Decode after the image.
        (False, 0, 3873, 0, 3873),
    ],
)
def test_cached_encoder_keeps_image_whole(monkeypatch, scheduler_cls, enabled, computed, count, shift, expected):
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.scheduler_config = SimpleNamespace(disable_chunked_mm_input=enabled)
    request = SimpleNamespace(
        has_encoder_inputs=True,
        mm_features=[SimpleNamespace(mm_position=SimpleNamespace(offset=1227, length=4982))],
    )
    calls = []

    def cached_inputs(self, req, start, tokens, budget, shift_computed_tokens):
        calls.append((req, start, tokens, budget, shift_computed_tokens))
        return [], tokens, budget, []

    # The upstream cache-hit branch returns without checking image boundaries.
    monkeypatch.setattr(Scheduler, "_try_schedule_encoder_inputs", cached_inputs)
    assert scheduler._try_schedule_encoder_inputs(request, computed, count, 16384, shift) == ([], expected, 16384, [])
    assert calls == [(request, computed, expected, 16384, shift)]
