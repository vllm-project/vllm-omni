# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import _should_emit_engine_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("async_chunk", "final_output", "use_v2", "stopped", "has_control", "expected"),
    [
        (True, False, True, False, False, False),  # async+v2 suppresses data
        (True, False, False, False, False, True),  # v1 continues to emit
        (True, False, True, True, False, True),  # stopped always emits
        (True, False, True, False, True, True),  # control-only still emits
    ],
)
def test_async_chunk_intermediate_stage_emits_only_control_outputs(
    async_chunk: bool,
    final_output: bool,
    use_v2: bool,
    stopped: bool,
    has_control: bool,
    expected: bool,
) -> None:
    model_config = SimpleNamespace(
        async_chunk=async_chunk,
        final_output=final_output,
        use_v2_model_runner=use_v2,
    )

    assert (
        _should_emit_engine_output(
            model_config,
            stopped=stopped,
            has_control=has_control,
        )
        is expected
    )
