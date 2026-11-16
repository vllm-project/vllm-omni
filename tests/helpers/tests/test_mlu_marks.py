# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.helpers.mark import _skipif_not_platform, hardware_marks

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_mlu_single_card_marks():
    names = {mark.name for mark in hardware_marks(res={"mlu": "MLU580"})}
    assert names == {"mlu", "MLU580", "cards_1"}


@pytest.mark.parametrize("device_type", ["mlu", "cpu"])
@pytest.mark.parametrize("device_count", [1, 2])
def test_mlu_multicard_marks_check_only_the_mlu_backend(monkeypatch, device_type, device_count):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", device_type)
    monkeypatch.setattr(current_platform, "device_count", lambda: device_count)
    marks = hardware_marks(res={"mlu": "MLU580"}, num_cards=2)
    skips = [mark for mark in marks if mark.name == "skipif"]
    if device_type == "mlu":
        assert len(skips) == 1 and skips[0].args == (device_count < 2,)
    else:
        assert not skips


@pytest.mark.parametrize("device_type", ["mlu", "cpu"])
def test_mlu_mixed_platform_selection_uses_vendor_device_type(monkeypatch, device_type):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", device_type)
    assert _skipif_not_platform("mlu").args == (device_type != "mlu",)
