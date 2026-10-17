# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session end-of-life routing (RFC #4480, adapting to #5271).

The AR-Diffusion runner raises an explicit end-of-session signal on reset,
close, eviction, and failed forwards (``reset_ar_diffusion_session`` /
``close_ar_diffusion_session``). On the bespoke path those pop ``_states``; on
the opt-in manager path the session lives in the manager instead, so the hooks
must release it there or every closed session leaks its buffers.

All CPU, tiny tensors, no model.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import (
    DREAMZERO_MODEL_OWNED_STATE_BYTES_PER_SESSION,
    MAX_DREAMZERO_SESSIONS,
    MAX_RESIDENT_DREAMZERO_SESSION_STATES,
    DreamZeroPipeline,
)
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import DreamZeroState
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    AR_DIFFUSION_TICK_KEY,
    ARDiffusionTickRequest,
)
from vllm_omni.experimental.world_models.adapters.state_dreamzero_adapter import (
    DreamZeroStateAdapter,
)
from vllm_omni.experimental.world_models.session_state import (
    LatentBuffer,
    SessionAdmissionError,
    SessionStateLostError,
    SessionStateManager,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# -- SessionStateManager.drop_session ---------------------------------------


def test_drop_session_absent_returns_false() -> None:
    manager = SessionStateManager()
    assert manager.drop_session("missing") is False


def test_drop_session_removes_and_frees() -> None:
    manager = SessionStateManager()
    session = manager.get_or_create_session("s")
    buffer: LatentBuffer[torch.Tensor] = LatentBuffer()
    buffer.allocate(maxlen=None)
    buffer.append(torch.zeros(256, dtype=torch.float32))
    session.put("payload", buffer)
    assert manager.nbytes_by_device().get("cpu", 0) >= 1024

    assert manager.drop_session("s") is True
    assert "s" not in manager
    assert len(manager) == 0
    # The session is gone from the table and its buffers were reset, so the
    # manager reports no held bytes.
    assert manager.nbytes_by_device() == {}


def test_drop_session_recreates_fresh() -> None:
    manager = SessionStateManager()
    manager.get_or_create_session("s").attrs["k"] = 1
    manager.drop_session("s")
    assert manager.get_or_create_session("s").attrs == {}


# -- SessionStateManager.raise_max_sessions ----------------------------------


def test_raise_max_sessions_lifts_the_bound_and_stops_evicting() -> None:
    manager = SessionStateManager(max_sessions=2)

    assert manager.raise_max_sessions(4) is True
    assert manager.max_sessions == 4

    for index in range(4):
        manager.get_or_create_session(f"s{index}")
    assert len(manager) == 4
    assert manager.evictions == 0


def test_raise_max_sessions_refuses_to_lower() -> None:
    """Shrinking a live bound would silently strand accumulated history: the
    overflow path drops the table entry without resetting buffers."""
    manager = SessionStateManager(max_sessions=4)

    assert manager.raise_max_sessions(2) is False
    assert manager.raise_max_sessions(4) is False
    assert manager.max_sessions == 4


@pytest.mark.parametrize("capacity", [0, -1])
def test_raise_max_sessions_rejects_non_positive(capacity: int) -> None:
    manager = SessionStateManager(max_sessions=2)

    with pytest.raises(ValueError, match="max_sessions must be positive"):
        manager.raise_max_sessions(capacity)

    assert manager.max_sessions == 2


# -- SessionStateManager admission policy ------------------------------------


def test_manager_evicts_when_full_by_default() -> None:
    """Unchanged for every other model on this store (e.g. cosmos3): only an
    opt-in makes admission fail instead."""
    manager = SessionStateManager(max_sessions=2)

    for index in range(3):
        manager.get_or_create_session(f"s{index}")

    assert len(manager) == 2
    assert manager.evictions == 1
    assert "s0" not in manager


def test_manager_refuses_new_session_when_not_evicting() -> None:
    manager = SessionStateManager(max_sessions=2, evict_when_full=False)
    manager.get_or_create_session("s0")
    manager.get_or_create_session("s1")

    with pytest.raises(SessionAdmissionError, match="cannot admit session 's2'"):
        manager.get_or_create_session("s2")

    assert len(manager) == 2
    assert manager.evictions == 0
    # Both resident sessions survive the refusal.
    assert "s0" in manager
    assert "s1" in manager


def test_manager_still_serves_resident_sessions_when_full() -> None:
    manager = SessionStateManager(max_sessions=2, evict_when_full=False)
    first = manager.get_or_create_session("s0")
    manager.get_or_create_session("s1")

    assert manager.get_or_create_session("s0") is first


def test_manager_admits_again_after_drop_session() -> None:
    manager = SessionStateManager(max_sessions=1, evict_when_full=False)
    manager.get_or_create_session("s0")
    with pytest.raises(SessionAdmissionError):
        manager.get_or_create_session("s1")

    assert manager.drop_session("s0") is True

    assert manager.get_or_create_session("s1") is not None
    assert len(manager) == 1


# -- pipeline hook routing ---------------------------------------------------


def _manager_pipe(manager: SessionStateManager, session_id: str) -> DreamZeroPipeline:
    """A pipeline built via __new__ whose state is a manager-backed adapter."""
    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipe._states = OrderedDict()
    pipe._memory_manager = manager
    pipe.state = DreamZeroStateAdapter(session_id, manager, vae_encoder_window=2)
    return pipe


@pytest.mark.parametrize("hook", ["close_ar_diffusion_session", "reset_ar_diffusion_session"])
def test_hook_releases_manager_session_and_clears_alias(hook: str) -> None:
    manager = SessionStateManager()
    pipe = _manager_pipe(manager, "sess")
    assert "sess" in manager

    getattr(DreamZeroPipeline, hook)(pipe, "sess")

    assert "sess" not in manager
    # The alias viewed the released session, so it is dropped and rebuilt lazily.
    assert pipe.state is None


def test_hook_keeps_alias_for_a_different_session() -> None:
    manager = SessionStateManager()
    pipe = _manager_pipe(manager, "default")
    manager.get_or_create_session("other")

    DreamZeroPipeline.close_ar_diffusion_session(pipe, "other")

    assert "other" not in manager
    # The alias views "default", not the closed session, so it survives.
    assert isinstance(pipe.state, DreamZeroStateAdapter)
    assert pipe.state.session_id == "default"


def test_bespoke_hook_pops_states_and_clears_alias() -> None:
    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipe._states = OrderedDict()
    pipe._memory_manager = None
    state = DreamZeroState()
    pipe._states["sess"] = state
    pipe.state = state

    DreamZeroPipeline.close_ar_diffusion_session(pipe, "sess")

    assert "sess" not in pipe._states
    assert pipe.state is None


# -- admission, not eviction --------------------------------------------------
#
# The hooks above are the *only* removal path for ``_states``, and they are
# reached solely from the AR-Diffusion runner's release path. A stage that holds
# session state without hosting the engine -- the disaggregated encode stage,
# where ``engine_backend: ARDiffusionEngine`` is set on denoise only -- never
# receives that signal, so finished sessions accumulate at ~603 MiB each.
#
# The bound therefore refuses *new* sessions rather than evicting resident ones.
# Per-session state includes the Wan VAE causal-convolution cache, which has no
# recompute source, so evicting a session that is later continued would silently
# restart its rollout and return normal-looking but wrong output.


def _bespoke_pipe(max_states: int) -> DreamZeroPipeline:
    """A pipeline via __new__ on the bespoke path with a known state bound."""
    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipe._states = OrderedDict()
    pipe._memory_manager = None
    pipe._max_session_states = max_states
    pipe.state = None
    return pipe


def _manager_backed_pipe(max_sessions: int) -> tuple[DreamZeroPipeline, SessionStateManager]:
    """A pipeline on the opt-in manager path, policed the way ``__init__`` does it."""
    manager = SessionStateManager(max_sessions=max_sessions, evict_when_full=False)
    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipe._states = OrderedDict()
    pipe._memory_manager = manager
    pipe._max_session_states = max_sessions
    pipe.num_frame_per_block = 2
    pipe.state = None
    return pipe, manager


def _seeded_state(marker: int) -> DreamZeroState:
    """A state carrying a call history that must survive."""
    state = DreamZeroState()
    state.vae_encoder_out = torch.zeros(marker + 1, dtype=torch.float32)
    state.call_count = marker + 1
    return state


def _mid_rollout_state() -> DreamZeroState:
    """A state mid-rollout: VAE stream live, causal cache non-empty, prompt cached.

    ``vae_enc_feat_map`` is the field with no recompute source and the bulk of the
    ~603 MiB, so it is the one that actually has to survive capacity pressure --
    asserting on ``len(_states)`` or object identity alone would not catch its
    loss.
    """
    state = DreamZeroState()
    state.vae_stream_initialized = True
    state.vae_enc_feat_map = [torch.ones(2, 3, dtype=torch.float32)]
    state.vae_encoder_out = torch.ones(1, 1, 4, dtype=torch.float32)
    state.vae_pending_body_frames = torch.ones(3, dtype=torch.float32)
    state.prompt_embeds = torch.ones(2, dtype=torch.float32)
    state.language = torch.ones(2, dtype=torch.long)
    state.call_count = 7
    state.current_start_frame = 12
    return state


def _assert_mid_rollout_intact(state: DreamZeroState) -> None:
    """Every field ``reset()`` would have cleared is still exactly as seeded."""
    assert state.vae_stream_initialized is True
    assert state.vae_enc_feat_map is not None
    assert len(state.vae_enc_feat_map) == 1
    assert torch.equal(state.vae_enc_feat_map[0], torch.ones(2, 3, dtype=torch.float32))
    assert state.vae_encoder_out is not None
    assert torch.equal(state.vae_encoder_out, torch.ones(1, 1, 4, dtype=torch.float32))
    assert state.vae_pending_body_frames is not None
    assert state.prompt_embeds is not None
    assert state.language is not None
    assert state.call_count == 7
    assert state.current_start_frame == 12


def test_new_session_at_the_bound_is_refused() -> None:
    pipe = _bespoke_pipe(2)
    DreamZeroPipeline._get_or_create_state(pipe, "s0")
    DreamZeroPipeline._get_or_create_state(pipe, "s1")

    with pytest.raises(SessionAdmissionError, match="cannot admit DreamZero session 's2'"):
        DreamZeroPipeline._get_or_create_state(pipe, "s2")

    assert list(pipe._states) == ["s0", "s1"]


def test_refusing_a_new_session_preserves_resident_history() -> None:
    """The regression this whole change exists for: an arriving session must not
    cost an existing one its unrecoverable VAE history."""
    pipe = _bespoke_pipe(1)
    resident = _seeded_state(3)
    pipe._states["s0"] = resident
    pipe.state = resident

    with pytest.raises(SessionAdmissionError):
        DreamZeroPipeline._get_or_create_state(pipe, "s1")

    assert pipe._states["s0"] is resident
    assert resident.vae_encoder_out is not None
    assert resident.call_count == 4
    assert pipe.state is resident


def test_resident_session_is_still_served_at_the_bound() -> None:
    """The bound gates admission, not lookup: a full table must keep serving the
    sessions it already admitted, or capacity pressure would stall everyone."""
    pipe = _bespoke_pipe(2)
    first = DreamZeroPipeline._get_or_create_state(pipe, "s0")
    DreamZeroPipeline._get_or_create_state(pipe, "s1")

    assert DreamZeroPipeline._get_or_create_state(pipe, "s0") is first
    assert DreamZeroPipeline._require_session_state(pipe, "s0") is first
    # Reuse still reorders, so ordering stays meaningful for observability.
    assert list(pipe._states) == ["s1", "s0"]


def test_admission_recovers_after_a_session_is_released() -> None:
    pipe = _bespoke_pipe(1)
    DreamZeroPipeline._get_or_create_state(pipe, "s0")
    with pytest.raises(SessionAdmissionError):
        DreamZeroPipeline._get_or_create_state(pipe, "s1")

    DreamZeroPipeline.close_ar_diffusion_session(pipe, "s0")

    assert DreamZeroPipeline._get_or_create_state(pipe, "s1") is not None
    assert list(pipe._states) == ["s1"]


def test_non_positive_bound_means_unbounded() -> None:
    pipe = _bespoke_pipe(0)

    for index in range(3):
        DreamZeroPipeline._get_or_create_state(pipe, f"s{index}")

    assert len(pipe._states) == 3


def test_refusal_leaves_a_mid_rollout_session_completely_untouched() -> None:
    """The unrecoverable fields, not just the dict entry.

    ``vae_enc_feat_map`` has no recompute source, so this is the assertion that
    actually distinguishes "refused the newcomer" from "quietly reset a live
    session".
    """
    pipe = _bespoke_pipe(1)
    live = _mid_rollout_state()
    pipe._states["s0"] = live
    pipe.state = live

    with pytest.raises(SessionAdmissionError):
        DreamZeroPipeline._get_or_create_state(pipe, "s1")

    _assert_mid_rollout_intact(pipe._states["s0"])
    # ...and it is still continuable, which is the point.
    assert DreamZeroPipeline._require_session_state(pipe, "s0") is live


def test_slots_are_reclaimed_across_many_begin_close_cycles() -> None:
    """CPU stand-in for the long serial soak: admission must not leak slots.

    Holds the table at its bound and then cycles one session out and one in, many
    times over. A slot that is not truly reclaimed on close shows up as a refusal
    part-way through instead of at the end.
    """
    pipe = _bespoke_pipe(4)
    for index in range(4):
        DreamZeroPipeline._get_or_create_state(pipe, f"resident{index}")

    for cycle in range(100):
        DreamZeroPipeline.close_ar_diffusion_session(pipe, f"resident{cycle % 4}")
        DreamZeroPipeline._get_or_create_state(pipe, f"resident{cycle % 4}")
        assert len(pipe._states) == 4

    for index in range(4):
        DreamZeroPipeline.close_ar_diffusion_session(pipe, f"resident{index}")
    assert not pipe._states


# -- the same rules on the manager path --------------------------------------
#
# The two stores refuse through different mechanisms: the bespoke path via
# ``_admit_session_state()``, the manager path from inside the adapter's
# constructor, which calls ``get_or_create_session()``. Both are reachable from
# ``_get_or_create_state()``, so both need covering or a deployment's actual path
# may be the untested one.


def test_manager_path_refuses_a_new_session_at_the_bound() -> None:
    pipe, manager = _manager_backed_pipe(2)
    DreamZeroPipeline._get_or_create_state(pipe, "s0")
    DreamZeroPipeline._get_or_create_state(pipe, "s1")

    with pytest.raises(SessionAdmissionError):
        DreamZeroPipeline._get_or_create_state(pipe, "s2")

    assert len(manager) == 2
    assert manager.evictions == 0
    assert "s2" not in manager


def test_manager_path_refusal_preserves_resident_sessions() -> None:
    pipe, manager = _manager_backed_pipe(1)
    resident = DreamZeroPipeline._get_or_create_state(pipe, "s0")
    resident.call_count = 5

    with pytest.raises(SessionAdmissionError):
        DreamZeroPipeline._get_or_create_state(pipe, "s1")

    assert "s0" in manager
    # A fresh adapter is built per call, so read the session back through one.
    assert DreamZeroPipeline._require_session_state(pipe, "s0").call_count == 5


def test_manager_path_still_serves_a_resident_session_when_full() -> None:
    pipe, manager = _manager_backed_pipe(2)
    DreamZeroPipeline._get_or_create_state(pipe, "s0")
    DreamZeroPipeline._get_or_create_state(pipe, "s1")

    assert DreamZeroPipeline._get_or_create_state(pipe, "s0") is not None
    assert len(manager) == 2


def test_manager_path_admits_again_after_close() -> None:
    pipe, manager = _manager_backed_pipe(1)
    DreamZeroPipeline._get_or_create_state(pipe, "s0")
    with pytest.raises(SessionAdmissionError):
        DreamZeroPipeline._get_or_create_state(pipe, "s1")

    DreamZeroPipeline.close_ar_diffusion_session(pipe, "s0")

    assert DreamZeroPipeline._get_or_create_state(pipe, "s1") is not None
    assert len(manager) == 1


# -- continuation must find its history ---------------------------------------


def test_continuation_of_an_unknown_session_fails_loudly() -> None:
    """A continuation must never be served fresh state: ``reset_reason()`` reports
    ``"session"`` for empty state, so the pipeline would quietly restart the
    rollout while the denoise stage's KV for that session is still live."""
    pipe = _bespoke_pipe(4)

    with pytest.raises(SessionStateLostError, match="has no resident state"):
        DreamZeroPipeline._require_session_state(pipe, "never-began")

    # The failed lookup must not have created anything.
    assert not pipe._states


def test_continuation_error_says_how_to_recover() -> None:
    pipe = _bespoke_pipe(4)

    with pytest.raises(SessionStateLostError, match=r'extra_args\["reset"\]=True'):
        DreamZeroPipeline._require_session_state(pipe, "gone")


def test_continuation_returns_the_same_state_object() -> None:
    pipe = _bespoke_pipe(4)
    begun = DreamZeroPipeline._get_or_create_state(pipe, "s0")

    assert DreamZeroPipeline._require_session_state(pipe, "s0") is begun


def test_continuation_of_default_session_works() -> None:
    """``__init__`` seeds ``"default"``, so a request that sends no session_id and
    no reset still resolves."""
    pipe = _bespoke_pipe(4)
    default = DreamZeroPipeline._get_or_create_state(pipe, "default")

    assert DreamZeroPipeline._require_session_state(pipe, None) is default


def test_manager_path_continuation_does_not_create_via_the_adapter() -> None:
    """The adapter's constructor calls ``get_or_create_session()``, so membership
    has to be checked before it is built or the absent session is created by the
    very lookup meant to report it missing."""
    manager = SessionStateManager(max_sessions=4, evict_when_full=False)
    pipe = _bespoke_pipe(4)
    pipe._memory_manager = manager
    pipe.num_frame_per_block = 2

    with pytest.raises(SessionStateLostError):
        DreamZeroPipeline._require_session_state(pipe, "never-began")

    assert "never-began" not in manager
    assert len(manager) == 0


@pytest.mark.parametrize("hook", ["close_ar_diffusion_session", "reset_ar_diffusion_session"])
def test_continuation_after_release_fails_instead_of_restarting(hook: str) -> None:
    """The sequence the whole change exists to forbid.

    A released session must not be silently re-served on empty state: that is the
    path that returns normal-looking but wrong output, since ``reset_reason()``
    reports ``"session"`` for a fresh state and the rollout quietly restarts.
    """
    pipe = _bespoke_pipe(4)
    live = _mid_rollout_state()
    pipe._states["s0"] = live
    pipe.state = live

    getattr(DreamZeroPipeline, hook)(pipe, "s0")

    with pytest.raises(SessionStateLostError):
        DreamZeroPipeline._require_session_state(pipe, "s0")
    assert not pipe._states
    # A fresh begin is the documented way back, and it starts genuinely clean.
    reborn = DreamZeroPipeline._get_or_create_state(pipe, "s0")
    assert reborn is not live
    assert reborn.call_count == 0
    assert reborn.vae_enc_feat_map is None


def test_manager_path_continuation_after_release_fails() -> None:
    pipe, manager = _manager_backed_pipe(4)
    DreamZeroPipeline._get_or_create_state(pipe, "s0")

    DreamZeroPipeline.close_ar_diffusion_session(pipe, "s0")

    assert "s0" not in manager
    with pytest.raises(SessionStateLostError):
        DreamZeroPipeline._require_session_state(pipe, "s0")


@pytest.mark.parametrize("hook", ["close_ar_diffusion_session", "reset_ar_diffusion_session"])
def test_release_is_idempotent_and_unknown_release_is_a_no_op(hook: str) -> None:
    """Repeat and stray releases must not double-release or resurrect a session."""
    pipe = _bespoke_pipe(4)
    pipe._states["s0"] = _mid_rollout_state()
    survivor = DreamZeroPipeline._get_or_create_state(pipe, "s1")

    getattr(DreamZeroPipeline, hook)(pipe, "s0")
    getattr(DreamZeroPipeline, hook)(pipe, "s0")
    getattr(DreamZeroPipeline, hook)(pipe, "never-existed")

    assert list(pipe._states) == ["s1"]
    assert pipe._states["s1"] is survivor


def test_manager_path_continuation_binds_an_existing_session() -> None:
    manager = SessionStateManager(max_sessions=4, evict_when_full=False)
    manager.get_or_create_session("s0")
    pipe = _bespoke_pipe(4)
    pipe._memory_manager = manager
    pipe.num_frame_per_block = 2

    adapter = DreamZeroPipeline._require_session_state(pipe, "s0")

    assert isinstance(adapter, DreamZeroStateAdapter)
    assert adapter.session_id == "s0"


# -- begin-vs-continue is read the way the runner reads it --------------------


def test_flat_reset_marks_a_begin() -> None:
    assert DreamZeroPipeline._request_begins_session({"reset": True}) is True
    assert DreamZeroPipeline._request_begins_session({"reset": False}) is False
    assert DreamZeroPipeline._request_begins_session({}) is False


def test_tick_reset_marks_a_begin_without_a_flat_key() -> None:
    """A typed tick namespaces its fields and does not duplicate them at the top
    level, and its extra_args are merged onto a *static* per-deployment template,
    so the flat key cannot express per-request begin/continue on that path. The
    runner releases the old session on the tick's value, so reading anything else
    here would demand state the runner just dropped."""
    tick = ARDiffusionTickRequest(session_id="s0", request_id="r0", chunk_index=0, reset=True)

    assert DreamZeroPipeline._request_begins_session(tick.to_extra_args()) is True


def test_tick_continuation_is_not_a_begin() -> None:
    tick = ARDiffusionTickRequest(session_id="s0", request_id="r1", chunk_index=1, reset=False)

    assert DreamZeroPipeline._request_begins_session(tick.to_extra_args()) is False


def test_flat_reset_still_counts_alongside_a_tick() -> None:
    """The consumer merges the tick over a template that may carry flat keys;
    either source asserting a begin is enough, so a begin is never misread as a
    continuation."""
    tick = ARDiffusionTickRequest(session_id="s0", request_id="r0", chunk_index=0, reset=False)
    merged = {"reset": True, **tick.to_extra_args()}

    assert DreamZeroPipeline._request_begins_session(merged) is True


def test_malformed_tick_falls_back_to_the_flat_key() -> None:
    """The forward must not gain a new parse failure: a non-mapping tick is
    ignored here and left for the runner's own validation to reject."""
    assert DreamZeroPipeline._request_begins_session({AR_DIFFUSION_TICK_KEY: "nonsense"}) is False
    assert DreamZeroPipeline._request_begins_session({AR_DIFFUSION_TICK_KEY: "nonsense", "reset": True}) is True


# -- export consumes history, never creates it --------------------------------


@pytest.mark.parametrize(
    "method",
    ["decode_accumulated_video_latents", "clear_accumulated_video_latents"],
)
def test_export_helpers_refuse_an_unknown_session(method: str) -> None:
    """These run from the video export worker at the end of a rollout; creating a
    session there would mask a lost session rather than report it."""
    pipe = _bespoke_pipe(4)

    with pytest.raises(SessionStateLostError):
        getattr(DreamZeroPipeline, method)(pipe, "never-began")

    assert not pipe._states


@pytest.mark.parametrize(
    "method",
    ["decode_accumulated_video_latents", "clear_accumulated_video_latents"],
)
def test_export_must_run_before_the_session_is_released(method: str) -> None:
    """Pins the ordering this creates: export consumes the accumulated history, so
    it has to happen while the session is still resident. No in-tree caller closes
    a DreamZero session before exporting, but a future one that did would now get a
    clear error rather than an empty decode."""
    pipe = _bespoke_pipe(4)
    DreamZeroPipeline._get_or_create_state(pipe, "s0")
    DreamZeroPipeline.close_ar_diffusion_session(pipe, "s0")

    with pytest.raises(SessionStateLostError):
        getattr(DreamZeroPipeline, method)(pipe, "s0")


def test_clear_accumulated_latents_keeps_the_session_continuable() -> None:
    """Clearing exported latents is not an end-of-session signal: the VAE stream
    and frame history a continuation needs must survive it."""
    pipe = _bespoke_pipe(4)
    live = _mid_rollout_state()
    live.video_latents_across_time = [torch.ones(1, 1, 2, 2, 2, dtype=torch.float32)]
    pipe._states["s0"] = live

    DreamZeroPipeline.clear_accumulated_video_latents(pipe, "s0")

    assert live.video_latents_across_time == []
    _assert_mid_rollout_intact(live)
    assert DreamZeroPipeline._require_session_state(pipe, "s0") is live


# -- capacity published by the runner ----------------------------------------


def test_published_capacity_raises_the_floor_and_admits_more() -> None:
    """A runner already reserves 603 MiB/session and will keep that many live, so
    refusing a session it has room for would be our bound, not its budget."""
    pipe = _bespoke_pipe(2)

    DreamZeroPipeline.set_resident_session_state_capacity(pipe, 5)

    assert pipe._max_session_states == 5
    for index in range(5):
        DreamZeroPipeline._get_or_create_state(pipe, f"s{index}")
    assert len(pipe._states) == 5


def test_published_capacity_reaches_the_manager_too() -> None:
    """Regression: the manager is built in ``__init__`` with the configured cap,
    so raising only ``_max_session_states`` would leave the manager refusing at the
    lower bound while the bespoke path admitted."""
    manager = SessionStateManager(max_sessions=2, evict_when_full=False)
    pipe = _bespoke_pipe(2)
    pipe._memory_manager = manager

    DreamZeroPipeline.set_resident_session_state_capacity(pipe, 5)

    assert pipe._max_session_states == 5
    assert manager.max_sessions == 5

    for index in range(5):
        manager.get_or_create_session(f"s{index}")
    assert len(manager) == 5
    assert manager.evictions == 0


def test_published_capacity_never_lowers_the_bound() -> None:
    pipe = _bespoke_pipe(4)
    manager = SessionStateManager(max_sessions=4, evict_when_full=False)
    pipe._memory_manager = manager

    DreamZeroPipeline.set_resident_session_state_capacity(pipe, 1)

    assert pipe._max_session_states == 4
    assert manager.max_sessions == 4


@pytest.mark.parametrize("capacity", [0, -1])
def test_published_capacity_ignores_non_positive(capacity: int) -> None:
    pipe = _bespoke_pipe(4)

    DreamZeroPipeline.set_resident_session_state_capacity(pipe, capacity)

    assert pipe._max_session_states == 4


# -- the bound has to be able to fire ----------------------------------------


def test_resident_state_bound_is_not_the_kv_slot_count() -> None:
    """Regression guard for the trap this fix exists to close.

    ``MAX_DREAMZERO_SESSIONS`` counts KV slots. Reusing it for model-owned state
    puts the bound at 64 x 603 MiB = 38.6 GiB -- larger than any single device --
    so it can never fire before the card OOMs, and a run under it looks exactly
    like a run with no bound at all.
    """
    assert MAX_RESIDENT_DREAMZERO_SESSION_STATES < MAX_DREAMZERO_SESSIONS
    resident_bytes = MAX_RESIDENT_DREAMZERO_SESSION_STATES * DREAMZERO_MODEL_OWNED_STATE_BYTES_PER_SESSION
    assert resident_bytes < 8 * 1024**3


def test_manager_honours_the_same_small_bound_and_policy() -> None:
    """The two stores must be sized *and policed* together.

    ``_drop_ar_diffusion_session_state()`` returns early on the manager path
    without touching ``_states``, so a bound enforced on only one of them is dead
    code for whichever path a deployment actually takes -- and a bound that
    refuses on one path while evicting on the other still loses history.
    """
    manager = SessionStateManager(
        max_sessions=MAX_RESIDENT_DREAMZERO_SESSION_STATES,
        evict_when_full=False,
    )

    for index in range(MAX_RESIDENT_DREAMZERO_SESSION_STATES):
        manager.get_or_create_session(f"s{index}")

    with pytest.raises(SessionAdmissionError):
        manager.get_or_create_session("one-too-many")

    assert len(manager) == MAX_RESIDENT_DREAMZERO_SESSION_STATES
    assert manager.evictions == 0
    # Every admitted session is still there; none paid for the refused one.
    assert all(f"s{index}" in manager for index in range(MAX_RESIDENT_DREAMZERO_SESSION_STATES))
