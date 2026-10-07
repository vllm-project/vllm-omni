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

from vllm_omni.model_executor.models.interfaces import (
    SupportsMultiFrameDecode,
    requires_request_sample_eligibility,
    supports_multi_frame_decode,
)

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


def test_wrapper_satisfies_the_protocol():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    # The runner only ever sees the wrapper, so the wrapper is what has to
    # satisfy the contract.
    assert isinstance(MiniCPMO45OmniForConditionalGeneration, type)  # importable
    for name in (
        "supports_multi_frame_decode",
        "batch_stop_logits",
        "take_batch_stop_logits",
        "set_batch_stop_logits",
        "merge_frame_outputs",
    ):
        assert hasattr(MiniCPMO45OmniForConditionalGeneration, name), name


def test_flag_off_reads_false_without_contract():
    model = SimpleNamespace(supports_multi_frame_decode=False)
    assert supports_multi_frame_decode(model) is False
    # Absent entirely: same answer, no raise (Thinker stages have neither).
    assert supports_multi_frame_decode(SimpleNamespace()) is False


def test_flag_on_without_members_raises_and_names_them():
    model = SimpleNamespace(supports_multi_frame_decode=True)
    with pytest.raises(TypeError) as excinfo:
        supports_multi_frame_decode(model)
    message = str(excinfo.value)
    for member in ("batch_stop_logits", "take_batch_stop_logits", "set_batch_stop_logits", "merge_frame_outputs"):
        assert member in message


def test_armed_model_passes():
    assert supports_multi_frame_decode(_Model(armed=True)) is True


def test_runtime_checkable_protocol_agrees():
    assert isinstance(_Model(), SupportsMultiFrameDecode)


def test_sample_eligibility_flag():
    assert requires_request_sample_eligibility(SimpleNamespace(requires_request_sample_eligibility=True)) is True
    assert requires_request_sample_eligibility(SimpleNamespace()) is False
