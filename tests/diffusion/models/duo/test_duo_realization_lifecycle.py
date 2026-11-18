# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Realization stage has to actually execute and release its turn.

Four defects made the stage unable to run, and each one hid the next, so none of
them were visible from the config layer:

1. `DuoModelRunner._prepare_batch_inputs` took two arguments while the inherited
   stepwise loop calls it with three (`use_mixfusion`), so the first step raised
   `TypeError`.
2. `DuoPipeline.post_step_output` - the only producer of the `latent_ready`
   frame - had no caller, so the plugin rejected the run with "Missing clean
   acknowledgement after final Euler" and no codec packet was ever dispatched.
3. `schedule.complete("codec"/"asr", ...)` lived only in `record_stage_timing`,
   a plugin hook the engine never calls, so `alternating`/`serial` admission
   waited forever in `DuoPhaseSchedule.admit`.
4. Session ids are epoch-qualified (`<id>/epoch/<n>`) and `finite_model_blocks`
   is unset for open-ended duplex sessions, so `final` never became true and the
   old epoch's KV pinned the single allowed session slot.

1-3 are pinned by signature and call graph because they are pure wiring; 4 is
pinned by driving the real session-admission helpers.
"""

import inspect
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.duo.model_runner import DuoModelRunner
from vllm_omni.diffusion.models.duo.pipeline import DuoPipeline
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prepare_batch_inputs_accepts_use_mixfusion():
    # The inherited stepwise loop calls this with three positional arguments.
    base = inspect.signature(DiffusionModelRunner._prepare_batch_inputs)
    override = inspect.signature(DuoModelRunner._prepare_batch_inputs)
    assert "use_mixfusion" in base.parameters
    assert "use_mixfusion" in override.parameters, (
        "DuoModelRunner._prepare_batch_inputs must accept use_mixfusion, otherwise "
        "the inherited stepwise path raises TypeError on its first step"
    )
    # The base call site passes it positionally, so the parameter order must match.
    assert list(override.parameters)[:4] == list(base.parameters)[:4]


def test_stepwise_loop_publishes_intermediate_step_output():
    # `post_step_output` is the only producer of `latent_ready`; it must be
    # reachable from the shared stepwise loop, not merely defined.
    source = inspect.getsource(DiffusionModelRunner._execute_stepwise_core)
    assert "post_step_output" in source, (
        "the stepwise loop must call pipeline.post_step_output, otherwise nothing ever "
        "emits the latent_ready frame Realization's codec handoff depends on"
    )
    # It must run after the sampling update and before the decode gate.
    assert source.index("step_scheduler") < source.index("post_step_output") < source.index("should_decode")


def test_scheduling_completions_live_on_active_paths():
    # `DuoPhaseSchedule.admit` blocks on these counters; if their only producer
    # is a hook the engine never calls, those modes deadlock.
    from vllm_omni.model_executor.models.duo.duplex.data_plane import DuoDataPlane
    from vllm_omni.model_executor.models.duo.duplex.plugin import DuoDuplexPlugin

    assert 'complete("codec"' in inspect.getsource(DuoDataPlane), (
        "codec completion must be emitted from the codec data plane"
    )
    assert 'complete("asr"' in inspect.getsource(DuoDuplexPlugin), "asr completion must be emitted from user_transcript"
    # The dead hook itself must be gone, not merely shadowed.
    assert not hasattr(DuoDuplexPlugin, "record_stage_timing")


def test_plugin_declares_no_unwired_hook_overrides():
    # Guards the class of bug behind the deadlock: overriding a hook name the
    # engine never calls. Underscore-prefixed helpers are internal, not hooks.
    from vllm_omni.engine.duplex.plugin import DuplexModelPlugin
    from vllm_omni.model_executor.models.duo.duplex.plugin import DuoDuplexPlugin

    base_hooks = {name for name in dir(DuplexModelPlugin) if not name.startswith("_")}
    overridden = {
        name for name, value in vars(DuoDuplexPlugin).items() if not name.startswith("_") and inspect.isfunction(value)
    }
    assert overridden, "expected DuoDuplexPlugin to override declared base hooks"
    assert overridden <= base_hooks, (
        f"DuoDuplexPlugin overrides names the base plugin does not declare: {sorted(overridden - base_hooks)}"
    )


def test_epoch_suffix_identifies_a_distinct_session_generation():
    # `_model_state_session_key` qualifies the Realization session id per epoch;
    # the pipeline relies on that suffix to retire a superseded epoch.
    from vllm_omni.model_executor.models.duo.duplex.plugin import DuoDuplexPlugin

    key = DuoDuplexPlugin._model_state_session_key(object(), "sess", 3)
    assert key == "sess/epoch/3"


class _StubPool:
    """Records KV releases; the real pool owns CUDA storage."""

    def __init__(self):
        self.released = []

    def empty(self):
        return SimpleNamespace()

    def release(self, cache):
        self.released.append(cache)


def _pipeline_for_admission(max_sessions=1):
    """A DuoPipeline carrying just the state session admission touches."""
    pipe = object.__new__(DuoPipeline)
    pipe.options = {"max_sessions": max_sessions, "history_blocks": 1}
    pipe.sessions = {}
    pipe.retired_fences = {}
    pipe.weight_identity = None
    pipe.pool = _StubPool()
    pipe._pool = lambda: pipe.pool
    return pipe


def test_stale_epoch_detection_is_scoped_to_the_logical_session():
    pipe = _pipeline_for_admission()
    pipe.sessions["sess/epoch/1"] = SimpleNamespace(fence=1, cache="a")
    pipe.sessions["other/epoch/1"] = SimpleNamespace(fence=1, cache="b")

    assert pipe._stale_epoch_sessions("sess/epoch/2") == ["sess/epoch/1"]
    # A bare (epoch-less) id is its own generation: nothing to retire.
    assert pipe._stale_epoch_sessions("sess") == []
    # A different logical session is never touched.
    assert pipe._stale_epoch_sessions("other/epoch/2") == ["other/epoch/1"]


def test_superseded_epoch_session_is_retired_not_leaked():
    # Two epochs of the same logical session must not both hold a slot: with
    # max_sessions=1 the second admission raised capacity exhaustion because
    # epoch-qualified ids never hit the same-id retire path.
    pipe = _pipeline_for_admission(max_sessions=1)
    pipe.sessions["sess/epoch/1"] = SimpleNamespace(fence=1, cache="old")

    stale = pipe._stale_epoch_sessions("sess/epoch/2")
    assert stale == ["sess/epoch/1"], "the superseded epoch must be identified for retirement"
    for stale_id in stale:
        pipe.close_session(stale_id, pipe.sessions[stale_id].fence)

    assert not pipe.sessions, "the retired epoch must not keep occupying a session slot"
    assert pipe.pool.released == ["old"]
    # The other logical session is untouched by the same sweep.
    assert pipe._stale_epoch_sessions("other/epoch/2") == []


def test_close_session_returns_kv_and_records_the_fence():
    pipe = _pipeline_for_admission()
    pipe.sessions["sess/epoch/1"] = SimpleNamespace(fence=1, cache="cache")

    pipe.close_session("sess/epoch/1", 1)

    assert "sess/epoch/1" not in pipe.sessions
    assert pipe.retired_fences["sess/epoch/1"] == 1
    assert pipe.pool.released == ["cache"], "the session's KV must be returned to the pool"


def test_close_session_rejects_a_stale_fence():
    # A late close must not free another epoch's KV.
    pipe = _pipeline_for_admission()
    pipe.sessions["sess/epoch/1"] = SimpleNamespace(fence=2, cache="cache")

    with pytest.raises(ValueError, match="Stale Realization session fence"):
        pipe.close_session("sess/epoch/1", 1)
