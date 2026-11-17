# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The multi-frame runner contract (PR #7929 review).

The NPU runners used to probe the model with bare
``getattr(model, "supports_multi_frame_decode", False)``. A forward dropped
from the wrapper therefore read as "loop off": nothing engaged, nothing was
logged, and the failure only showed up on a device as an acl 507035
vector-core fault or a stage-1 engine that died with empty audio. The
helpers make a missing member raise, and the test below pins the wrapper's
membership so a rename cannot quietly drop it again.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.interfaces import (
    SupportsMultiFrameDecode,
    requires_request_sample_eligibility,
    supports_multi_frame_decode,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
    _request_key,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Model:
    """A model that satisfies the contract."""

    def __init__(self, armed: bool = True) -> None:
        self.supports_multi_frame_decode = armed
        self.batch_stop_logits = None

    def take_batch_stop_logits(self):
        return None

    def set_batch_stop_logits(self, logits) -> None:
        self.batch_stop_logits = logits

    def merge_frame_outputs(self, frame_outputs, frame_stop_logits):
        return frame_outputs


def _talker(k_frames=8):
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.batch_stop_logits = None
    talker._k_step_frames = k_frames
    return talker


def test_wrapper_satisfies_the_protocol():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    # The runner only ever sees the wrapper, so the wrapper is what has to
    # satisfy the contract.
    wrapper = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.model_stage = "tts"
    wrapper.talker = _talker()

    assert supports_multi_frame_decode(wrapper) is True
    assert isinstance(wrapper, SupportsMultiFrameDecode)
    logits = torch.tensor([[0.0, -torch.inf]])
    wrapper.set_batch_stop_logits(logits)
    assert wrapper.batch_stop_logits is logits
    assert wrapper.take_batch_stop_logits() is logits
    assert wrapper.batch_stop_logits is None
    assert wrapper.take_batch_stop_logits() is None

    frame = OmniOutput(text_hidden_states=torch.zeros(1, 4))
    assert wrapper.merge_frame_outputs([frame], [logits]) is frame
    assert wrapper.batch_stop_logits is logits


def test_flag_off_reads_false_without_contract():
    model = SimpleNamespace(supports_multi_frame_decode=False)
    assert supports_multi_frame_decode(model) is False
    # Absent entirely: same answer, no raise (Thinker stages have neither).
    assert supports_multi_frame_decode(SimpleNamespace()) is False


def test_gate_does_not_read_the_protocol_attributes(monkeypatch):
    """P2 on #7929: the gate read ``Protocol.__protocol_attrs__``.

    ``typing`` fills that attribute when the Protocol class is created, and
    only from Python 3.12 on: 3.10 and 3.11 leave it absent, so every model
    that armed the flag raised AttributeError instead of being checked. The
    package declares ``requires-python = ">=3.10"``. An object that answers
    nothing is what a 3.10 interpreter looks like to this gate.
    """
    from vllm_omni.model_executor.models import interfaces

    class _NoProtocolAttrs:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(interfaces, "SupportsMultiFrameDecode", _NoProtocolAttrs())

    assert interfaces.supports_multi_frame_decode(_Model(armed=True)) is True
    with pytest.raises(TypeError) as excinfo:
        interfaces.supports_multi_frame_decode(SimpleNamespace(supports_multi_frame_decode=True))
    assert "batch_stop_logits" in str(excinfo.value)


def test_required_members_covers_the_protocol():
    """The explicit member list has to track the Protocol.

    Spelled out because ``__protocol_attrs__`` is 3.12+, so a new Protocol
    member added without updating the list would silently go unverified.
    """
    from vllm_omni.model_executor.models.interfaces import _REQUIRED_MULTI_FRAME_MEMBERS

    declared = {
        name
        for name in (*vars(SupportsMultiFrameDecode), *SupportsMultiFrameDecode.__annotations__)
        if not name.startswith("_")
    }
    # The flag is the gate itself, not a member the gate verifies.
    assert declared - {"supports_multi_frame_decode"} == set(_REQUIRED_MULTI_FRAME_MEMBERS)


def test_required_mrv2_members_cover_the_protocol():
    from vllm_omni.model_executor.models.interfaces import (
        _REQUIRED_MRV2_MULTI_FRAME_MEMBERS,
        SupportsMRv2MultiFrameDecode,
    )

    declared = {
        name
        for name in (*vars(SupportsMRv2MultiFrameDecode), *SupportsMRv2MultiFrameDecode.__annotations__)
        if not name.startswith("_")
    }
    assert declared == set(_REQUIRED_MRV2_MULTI_FRAME_MEMBERS)


def test_flag_on_without_members_raises_and_names_them():
    model = SimpleNamespace(supports_multi_frame_decode=True)
    with pytest.raises(TypeError) as excinfo:
        supports_multi_frame_decode(model)
    message = str(excinfo.value)
    for member in ("batch_stop_logits", "take_batch_stop_logits", "set_batch_stop_logits", "merge_frame_outputs"):
        assert member in message


def test_sample_eligibility_flag():
    assert requires_request_sample_eligibility(SimpleNamespace(requires_request_sample_eligibility=True)) is True
    assert requires_request_sample_eligibility(SimpleNamespace()) is False


@pytest.mark.parametrize(
    ("info", "row", "expected"),
    [
        pytest.param({"req_id": "7", "request_id": "stale"}, 0, "7", id="mrv2-id-wins"),
        pytest.param({"request_id": "engine-9"}, 3, "engine-9", id="v1-id"),
        pytest.param({}, 5, "5", id="missing-id"),
        pytest.param(None, 2, "2", id="missing-info"),
        pytest.param({"req_id": 11}, 0, "11", id="integer-id"),
    ],
)
def test_request_key_matches_runner_identity(info, row, expected):
    assert _request_key(info, row) == expected


@pytest.mark.parametrize(("k_frames", "identity"), [(0, True), (1, True), (8, False)])
def test_mrv2_decode_preprocess_identity(k_frames, identity):
    # K-step rows must re-embed the previous real codec frame, not placeholders.
    assert _talker(k_frames).mrv2_decode_preprocess_is_identity is identity
