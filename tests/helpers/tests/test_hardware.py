# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.helpers import hardware

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("NVIDIA L20X", "H200"),
        ("L20X", "H200"),
        ("NVIDIA H200", "H200"),
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA L20", "L20"),
    ],
)
def test_runtime_hardware_uses_canonical_baseline_label(monkeypatch, device_name, expected):
    monkeypatch.setattr(hardware, "_read_runtime_device_name", lambda *, device_id=0: device_name)
    label = hardware.get_runtime_resource_label(refresh=True)
    assert label == expected
    assert hardware.hardware_json_value(label) == expected
