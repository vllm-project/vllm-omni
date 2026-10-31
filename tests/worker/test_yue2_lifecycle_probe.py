# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Negative controls for the YuE2 e2e lifecycle probe's preemption checks.

The probe and the e2e assertion helper run unmodified against the real YuE2
sampler and history reconciliation; only CUDA completion and the draw values
are replaced, as in test_yue2_request_state.py. Each failure case injects one
condition and expects the probe's own assertion to reject it.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.e2e.features.yue2.worker_extension import (
    Yue2LifecycleWorkerExtension,
    assert_preempted_requests_resumed,
)
from vllm_omni.model_executor.models.yue2 import yue2 as y

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LOGITS = torch.zeros((1, y.VOCAB_SIZE))


@pytest.fixture
def model(mocker):
    """Real YuE2 state/sampling control flow with CPU staging and fixed draws."""
    model = object.__new__(y.Yue2ForCausalLM)
    torch.nn.Module.__init__(model)
    model._states = {}
    model._pending = None
    model._step_rows = []
    model._step_discard = None
    model._decode_t0 = {}
    model._last_mm = None
    model._audio_queue = []
    model._max_model_len = 4096
    model._stage = torch.empty(1024, dtype=torch.long)
    model._pinned = torch.empty((2, 64), dtype=torch.long)
    model._synthesis = mocker.Mock(spec=y._SynthesisQueue)
    mocker.patch.object(model, "_finish_request_safely")
    mocker.patch.object(model, "_sampler_graph", return_value=None)
    # CPU copies complete synchronously; this replaces only CUDA completion.
    mocker.patch.object(y.torch, "Event", return_value=mocker.Mock(spec=torch.Event))
    mocker.patch.object(y, "sample_rows", return_value=(torch.tensor([y.CODEC_OFFSET + 7]), torch.tensor([False])))
    for rid in ("r", "s"):
        constants = y._RowConstants(rid, "semantic", 1.0, 0.9, 100, 1.1, 2, 1, 3, 17, False, 8)
        model._states[rid] = y._RequestState(rid, constants, prompt_len=3, prefix_ids=[1, 2, 3])
    return model


def start_probe(model) -> Yue2LifecycleWorkerExtension:
    probe = Yue2LifecycleWorkerExtension()
    probe.model_runner = SimpleNamespace(get_model=lambda: model)
    assert probe.start_yue2_probe()
    return probe


@pytest.fixture
def probe_for():
    """Start the probe on a model; stop its CUDA graph patch afterwards."""
    started = []

    def start(model):
        started.append(start_probe(model))
        return started[-1]

    yield start
    for probe in started:
        probe.yue2_graph_patch.stop()


def step(model, computed: int, scheduled: int, discard: bool = False, rid: str = "r") -> None:
    """Drive one single-row runner step through the probe-wrapped model hooks."""
    model.prepare_runner_inputs(
        req_ids=[rid],
        num_computed_tokens=[computed],
        num_scheduled_tokens=[scheduled],
        input_ids=torch.zeros(scheduled, dtype=torch.long),
        positions=torch.arange(computed, computed + scheduled),
        sampling_extra_args=None,
        discard_mask=[discard],
    )
    model.sample(LOGITS, None)


def decode_two_frames(model, rid: str = "r") -> None:
    """Prefill (3 prompt tokens), then one decode step; two draws in total."""
    step(model, 0, 3, rid=rid)
    step(model, 3, 1, rid=rid)
    assert model._states[rid].history == [y.CODEC_OFFSET + 7]


@pytest.mark.parametrize(
    ("resume_span", "rollbacks"),
    [
        # Engine retained both draws (upstream recompute preemption): nothing to trim.
        (5, 0),
        # Engine dropped the in-flight draw: the resumed row rolls it back.
        (4, 1),
    ],
    ids=["draw-retained", "draw-dropped"],
)
def test_resumed_preemption_passes(model, probe_for, resume_span: int, rollbacks: int) -> None:
    probe = probe_for(model)
    decode_two_frames(model)
    step(model, 0, resume_span)
    observations = probe.get_yue2_probe()
    assert observations["preemptions"] == 1
    assert observations["rollbacks"] == rollbacks
    assert observations["resumed_checks"] == 1
    assert observations["unresumed"] == []
    assert_preempted_requests_resumed(observations)


def test_preempted_request_that_never_resumes_is_rejected(model, probe_for) -> None:
    """Injected: the recompute never completes (only a discarded partial chunk runs)."""
    probe = probe_for(model)
    decode_two_frames(model)
    step(model, 0, 2, discard=True)
    observations = probe.get_yue2_probe()
    assert observations["preemptions"] == 1
    assert observations["resumed_checks"] == 0
    assert observations["unresumed"] == ["r"]
    with pytest.raises(AssertionError) as excinfo:
        assert_preempted_requests_resumed(observations)
    assert excinfo.traceback[-1].name == "assert_preempted_requests_resumed"


def test_one_unresumed_request_is_rejected_when_another_resumed(model, probe_for) -> None:
    """Injected: r resumes normally but s's recompute never completes."""
    probe = probe_for(model)
    decode_two_frames(model, "r")
    decode_two_frames(model, "s")
    step(model, 0, 5, rid="r")
    step(model, 0, 2, discard=True, rid="s")
    observations = probe.get_yue2_probe()
    assert observations["preemptions"] == 2
    assert observations["resumed_checks"] == 1
    assert observations["unresumed"] == ["s"]
    with pytest.raises(AssertionError) as excinfo:
        assert_preempted_requests_resumed(observations)
    assert excinfo.traceback[-1].name == "assert_preempted_requests_resumed"


def _no_trim(real):
    def reconcile(state, accepted):
        """Regression: never drops draws the engine discarded."""

    return reconcile


def _over_trim(real):
    def reconcile(state, accepted):
        """Regression: rolls back one frame too many."""
        rolled_back = len(state.history) + int(state.end_drawn) > accepted
        real(state, accepted)
        if rolled_back:
            state.history.pop()

    return reconcile


@pytest.mark.parametrize("regression", [_no_trim, _over_trim], ids=["no-trim", "over-trim"])
def test_resumed_row_with_invalid_reconciliation_is_rejected(model, probe_for, regression) -> None:
    """Injected: a broken _reconcile_history on the row that resumes a dropped draw."""
    model._reconcile_history = regression(model._reconcile_history)
    probe_for(model)
    decode_two_frames(model)
    with pytest.raises(AssertionError) as excinfo:
        step(model, 0, 4)
    # Rejected by the probe's reconcile check, not by an unrelated assert.
    assert excinfo.traceback[-1].path.name == "worker_extension.py"
    assert excinfo.traceback[-1].name == "reconcile_history"
