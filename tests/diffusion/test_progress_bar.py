# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.progress_bar import (
    DiffusionProgress,
    ProgressBarMixin,
    progress_requests,
    progress_sink,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_request(request_id, enabled=True):
    return SimpleNamespace(request_id=request_id, sampling_params=SimpleNamespace(emit_request_lifecycle=enabled))


@pytest.mark.parametrize("disabled", [True, False])
def test_reports_steps_for_opted_in_requests(disabled):
    pipeline = ProgressBarMixin()
    pipeline.set_progress_bar_config(disable=disabled)
    events = []
    with (
        progress_sink(events.append),
        progress_requests([_make_request("a"), _make_request("b"), _make_request("offline", False)]),
    ):
        with pipeline.progress_bar(total=3) as bar:
            for _ in range(3):
                bar.update()
    assert events == [DiffusionProgress(rid, step, 3) for step in range(1, 4) for rid in ("a", "b")]


def test_request_context_does_not_leak_after_failure():
    pipeline = ProgressBarMixin()
    pipeline.set_progress_bar_config(disable=True)
    events = []
    with progress_sink(events.append):
        with pytest.raises(RuntimeError), progress_requests([_make_request("failed")]):
            with pipeline.progress_bar(total=2) as bar:
                bar.update()
                raise RuntimeError("generation failed")
        with pipeline.progress_bar(total=2) as bar:
            bar.update()
        with progress_requests([_make_request("next")]), pipeline.progress_bar(total=1) as bar:
            bar.update()
    assert events == [DiffusionProgress("failed", 1, 2), DiffusionProgress("next", 1, 1)]


def test_each_loop_reports_its_own_steps():
    pipeline = ProgressBarMixin()
    pipeline.set_progress_bar_config(disable=True)
    events = []
    with progress_sink(events.append), progress_requests([_make_request("video")]):
        for total in (2, 3):
            with pipeline.progress_bar(total=total) as bar:
                for _ in range(total):
                    bar.update()
    assert events == [
        DiffusionProgress("video", 1, 2),
        DiffusionProgress("video", 2, 2),
        DiffusionProgress("video", 1, 3),
        DiffusionProgress("video", 2, 3),
        DiffusionProgress("video", 3, 3),
    ]
