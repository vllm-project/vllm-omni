# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Postprocess routing tests for NPUOmniModelState (diamond MRO).

NPUOmniModelState(OmniModelState, AscendModelState) routes K-step talker
outputs through ``make_omni_output`` only when the stage is actually armed
(``_npu_kstep_frames > 1``). Non-talker stages (AR, Code2Wav) must keep
delegating to ``super()`` so a future change in the ``talker_kstep_frames``
wiring cannot pull them into the K-step packing path.
"""

import pytest

pytest.importorskip("vllm_ascend")

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms.npu.models.omni_model_state import NPUOmniModelState
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


def _state(frames):
    class _FakeRunner(NPUOmniModelState):
        pass

    state = object.__new__(_FakeRunner)
    state._npu_kstep_frames = frames
    return state


def test_mro_keeps_omni_state_before_ascend_state():
    # OmniModelState must resolve first so omni output handling wins; the
    # Ascend state supplies attention metadata below it in the chain.
    mro = [c.__name__ for c in NPUOmniModelState.__mro__]
    assert mro.index("OmniModelState") < mro.index("AscendModelState")


def test_non_talker_stage_postprocess_delegates_to_super(monkeypatch):
    # A non-armed stage must not enter the K-step packing path even with a
    # non-OmniOutput payload: no make_omni_output call, straight delegation.
    calls = {}

    def fake_super_post(self, model_output, input_batch, req_states):
        calls["delegated"] = model_output
        return ("text-hidden", {"mm": True})

    monkeypatch.setattr(OmniModelState, "postprocess_model_output", fake_super_post)
    state = _state(1)
    payload = object()  # not an OmniOutput
    out = NPUOmniModelState.postprocess_model_output(state, payload, None, None)
    assert calls["delegated"] is payload
    assert out == ("text-hidden", {"mm": True})


def test_armed_stage_routes_through_make_omni_output():
    # An armed talker stage with a non-OmniOutput payload gathers the
    # intermediate buffer and calls make_omni_output with span info.
    import types

    state = _state(8)
    state.model = types.SimpleNamespace(
        make_omni_output=lambda out, **kwargs: OmniOutput(
            text_hidden_states="packed",
            multimodal_outputs={"spans": len(kwargs["request_token_spans"])},
        )
    )
    state.intermediate_buffer = types.SimpleNamespace(gather=lambda ib: ["buf"])
    state._request_token_spans = lambda ib: [(0, 8)]
    payload = object()  # not an OmniOutput
    out = NPUOmniModelState.postprocess_model_output(state, payload, None, None)
    assert out == ("packed", {"spans": 1})


def test_omni_output_payload_unwraps_without_make_omni_output():
    # An armed stage receiving an OmniOutput skips re-packing and unwraps.
    state = _state(8)
    payload = OmniOutput(text_hidden_states="t", multimodal_outputs={"x": 1})
    out = NPUOmniModelState.postprocess_model_output(state, payload, None, None)
    assert out == ("t", {"x": 1})
