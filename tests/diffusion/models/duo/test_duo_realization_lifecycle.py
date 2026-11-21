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
import torch

from vllm_omni.diffusion.data import DiffusionOutput
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


def test_stepwise_loop_publishes_intermediate_step_output(monkeypatch):
    # `post_step_output` is the only producer of `latent_ready`. Assert on the
    # output the runner actually returns: a snapshot taken on a non-terminal
    # step has to survive into the batch, otherwise nothing ever emits the frame
    # Realization's codec handoff depends on.
    #
    # The batch indexes outputs by request id, so a second entry for the same
    # request in one step would be discarded -- the earlier revision appended
    # the snapshot as its own entry and lost it that way.
    from tests.diffusion.test_diffusion_step_pipeline import (
        _make_cached_scheduler_output,
        _make_runner,
        _make_scheduler_output,
        _make_step_request,
        _noop_forward_context,
        model_runner_module,
    )

    calls = {"post_step": 0}

    def _post_step_output(state):
        calls["post_step"] += 1
        if state.step_index != 1:
            return None
        return DiffusionOutput(
            output={"payload": {"video_latent": torch.zeros(2), "audio_latent": torch.zeros(2)}},
            finished=False,
        )

    runner = _make_runner()
    # Attach the hook to the instance: the stepwise loop resolves it with
    # getattr(pipeline, "post_step_output", None), so a plain function is
    # reached unbound and receives the state as its only argument.
    runner.pipeline.post_step_output = _post_step_output
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    first = DiffusionModelRunner.execute_stepwise(
        runner, _make_scheduler_output(_make_step_request(), step_id=0)
    ).get_request_output("req-1")
    assert calls["post_step"] == 1
    assert first.finished is False
    assert first.result is not None, (
        "the intermediate snapshot must reach the batch; the non-terminal entry "
        "has to carry it because a second entry for the same request is dropped"
    )
    assert set(first.result.output["payload"]) == {"video_latent", "audio_latent"}

    # A step without a snapshot keeps the default one-final-output contract.
    second = DiffusionModelRunner.execute_stepwise(runner, _make_cached_scheduler_output(step_id=1)).get_request_output(
        "req-1"
    )
    assert second.result is not None and second.finished is True


def test_codec_completion_covers_every_codec_mode():
    # The first version of this fix only released the delayed `duo_5_17` branch;
    # `stream_4frame` (the default) and `old` take the other branch and still
    # hung `alternating`/`serial` after block 0. Assert both branches release.
    from vllm_omni.model_executor.models.duo.duplex import data_plane

    lines = inspect.getsource(data_plane.DuoDataPlane.project).splitlines()
    # Count real calls, not comments that merely name the helper.
    calls = [
        i for i, line in enumerate(lines) if line.strip().startswith("owner.schedule.") and 'complete("codec"' in line
    ]
    # One release inside the delayed duo_5_17 branch and one in the default
    # (stream_4frame / old) branch. The first version of this fix had only the
    # former, so the default branch deadlocked serial modes after block 0.
    assert calls, "the codec slot must be released"
    assert len(calls) >= 2, (
        "both the delayed duo_5_17 branch and the default (stream_4frame / old) branch "
        f"must release the codec slot; found {len(calls)} call(s)"
    )


def test_default_codec_mode_tolerates_a_chunk_without_an_owner(monkeypatch):
    # `project()` treats a missing owner as valid on the publish path
    # (`if owner is None: ready_ids = [output.request_id]`), so the release must
    # not dereference it: a codec chunk arriving without a pipeline owner used to
    # raise AttributeError here instead of simply skipping the release.
    from vllm_omni.model_executor.models.duo.duplex import data_plane as dp

    # The encode step needs real media; this test is about the release guard.
    monkeypatch.setattr(dp, "encode_av_chunk", lambda payload, meta: {"packets": []})

    plane = object.__new__(dp.DuoDataPlane)
    plane.defer_clean = False
    # `pending` mirrors the fields the codec branch touches; `codec_received`
    # gates the duplicate-chunk check.
    plane.requests = {
        "c0": SimpleNamespace(
            emitted=False,
            decoded=None,
            identity=None,
            clean=False,
            codec_received=False,
            codec_metadata=None,
        )
    }
    plane.states = {"c0": SimpleNamespace(owner=None, session_id="s", next_block=0, fence=0)}
    plane.codec_stage_id = 2
    plane.realization_stage_id = 1

    output = SimpleNamespace(
        request_id="c0",
        stage_id=2,
        multimodal_output={
            "metadata": {"session_id": "s", "fence": 0, "block": 0, "codec_mode": "stream_4frame"},
            "payload": {"packets": []},
        },
    )

    # `project()` is a generator: it must be consumed, otherwise the body never
    # runs and this test would pass regardless of the guard.
    list(dp.DuoDataPlane.project(plane, {"data_plane_outputs": [output]}))


def test_asr_slot_is_released_before_the_empty_transcript_return():
    # A finished ASR request with no `outputs` produces no transcript, so the
    # completion must run before that early return; otherwise an empty utterance
    # holds `asr_active` and blocks every later block in serial modes.
    from vllm_omni.model_executor.models.duo.duplex.plugin import DuoDuplexPlugin

    source = inspect.getsource(DuoDuplexPlugin.user_transcript)
    release = source.index('complete("asr"')
    early_return = source.index("return None")
    assert release < early_return, "the ASR slot must be released before the empty-transcript return"
    assert "record_stage_timing" not in dir(DuoDuplexPlugin)


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


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        pytest.param(166912, 3, id="a100_fits_three_stages"),
        pytest.param(137728, 3, id="exactly_three_stage_budget"),
        pytest.param(137727, 1, id="one_byte_short"),
        pytest.param(101376, 1, id="ada_carve_out"),
        pytest.param(49152, 1, id="default_block_limit"),
        pytest.param(None, 1, id="unknown_device"),
    ],
)
def test_split_kv_stage_count_follows_the_device_shared_memory_limit(monkeypatch, limit, expected):
    # The three-stage launch needs 137,728 B and aborts on parts capped at
    # 101,376 B, so the depth has to follow the device rather than being pinned.
    # This pins the threshold: a device exactly at the budget keeps the pipelined
    # form, one byte below falls back.
    from vllm_omni.diffusion.models.duo import attention

    monkeypatch.setattr(attention, "_shared_memory_per_block", lambda _device: limit)

    assert attention._split_kv_num_stages(object()) == expected


def test_split_kv_stage_count_never_raises_on_an_unknown_backend(monkeypatch):
    # Reporting the limit can fail off CUDA; that must degrade to the single
    # stage instead of raising during planner construction.
    from vllm_omni.diffusion.models.duo import attention

    def _boom(_device):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_properties", _boom)

    assert attention._split_kv_num_stages(object()) == 1
