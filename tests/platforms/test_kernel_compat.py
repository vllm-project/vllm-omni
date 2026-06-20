# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_omni.platforms.kernel_compat import IrOpPriorityConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ir_op_priority_config_exposes_with_default() -> None:
    priority = IrOpPriorityConfig.with_default(
        ["vllm_c", "native"],
        rms_norm=["oink", "vllm_c", "native"],
        custom=["x"],
    )

    assert priority is not None
