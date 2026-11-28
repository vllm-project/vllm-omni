# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the native transport's real callback and request snapshot."""

from collections import defaultdict

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.zonos2 import talker2dac_async_chunk
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane, _NativeRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_native_callback_uses_current_step_tokens_and_does_not_duplicate_terminal_frame():
    manager = OmniRunnerDataPlane.__new__(OmniRunnerDataPlane)
    manager._code_prompt_token_ids = defaultdict(list)
    manager._put_req_chunk = defaultdict(int)
    manager._custom_process_func = talker2dac_async_chunk
    manager._custom_process_payload_kwarg = "multimodal_output"
    manager._custom_process_supports_is_finished = True
    request = _NativeRequestState("r", "r", [0])
    row = torch.full((1, 9), 7, dtype=torch.int32)
    payload = None
    for _ in range(24):
        request.accept_tokens([0])
        payload = manager._build_custom_process_payload(
            "r",
            request.snapshot(include_token_history=False, sampled_token_ids=[0]),
            {"codes": {"audio": row}, "meta": {"eos_frame": torch.tensor([-1])}},
            propagate_errors=True,
        )
    assert payload is not None
    assert payload.codes.audio.shape == (9, 24)
    assert len(manager.code_prompt_token_ids["r"]) == 24
    request.finished = True
    final = manager._build_custom_process_payload(
        "r",
        request.snapshot(include_token_history=False),
        {"codes": {"audio": row}},
        propagate_errors=True,
    )
    assert final.codes.audio.shape == (9, 24)
    assert bool(final.meta.finished)
    assert "r" not in manager.code_prompt_token_ids
