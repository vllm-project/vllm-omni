# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("vllm_ascend")

from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import _minicpmo_mixed_long_prefill

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("lengths", "prefilling", "enabled", "stage", "expected"),
    [
        ([39958], [True], True, "llm", False),
        ([39958, 39925], [True, True], True, "llm", False),
        ([39958, 39925], [True, False], True, "llm", True),
        ([8192, 39925], [True, False], True, "llm", False),
        ([39958, 39925], [True, False], False, "llm", False),
        ([39958, 39925], [True, False], True, "tts", False),
    ],
)
def test_only_active_long_prefill_with_a_decode_peer_uses_eager(lengths, prefilling, enabled, stage, expected):
    config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage=stage, architectures=["MiniCPMO45OmniForConditionalGeneration"]),
        additional_config={"minicpmo_long_prefill_eager": enabled},
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    batch = SimpleNamespace(
        has_prefill=any(prefilling),
        is_dummy=False,
        is_prefilling_np=np.array(prefilling),
        prefill_len_np=np.array(lengths),
    )
    assert _minicpmo_mixed_long_prefill(config, batch) is expected
