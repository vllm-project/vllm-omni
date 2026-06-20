# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.core.sched.routed_experts import RoutedExpertsLists
from vllm_omni.core.sched.utils import omni_routed_experts_for_request

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_routed_experts_fallback_shape_supports_request_filtering() -> None:
    routed = RoutedExpertsLists(
        routing_data=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        slot_mapping=np.array([10, 20, 30], dtype=np.int32),
    )
    request = SimpleNamespace(block_table=[20, 30])

    selected = omni_routed_experts_for_request(routed, request)

    assert selected is not None
    np.testing.assert_array_equal(selected, np.array([[3, 4], [5, 6]], dtype=np.int32))
