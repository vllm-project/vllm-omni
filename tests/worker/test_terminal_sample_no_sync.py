# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("enabled,capped,invalid", [(False, True, False), (True, False, False), (True, True, True)])
def test_terminal_selection_fast_path_never_reads_gpu(enabled, capped, invalid):
    class ForbiddenTensor:
        def __getattribute__(self, name):
            raise AssertionError(f"Unexpected GPU tensor access: {name}")

    runner = SimpleNamespace(
        model=SimpleNamespace(terminal_sample_drain_token_ids={151654} if enabled else None),
        requests={
            "r": SimpleNamespace(
                sampling_params=SimpleNamespace(max_tokens=1 if capped else 10),
                output_token_ids=[151654],
                num_tokens=2,
            )
        },
        max_model_len=100,
    )
    assert (
        GPUARModelRunner._terminal_sample_drain_request_ids(
            runner,
            req_ids=["r"],
            valid_sampled_token_ids=[],
            sampled_token_ids=ForbiddenTensor(),
            invalid_req_indices=[0] if invalid else [],
        )
        == []
    )
