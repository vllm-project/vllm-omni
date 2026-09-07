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

from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import DreamZeroState
from vllm_omni.experimental.world_models.adapters.state_dreamzero_adapter import (
    DreamZeroStateAdapter,
)
from vllm_omni.experimental.world_models.session_state import (
    LatentBuffer,
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
