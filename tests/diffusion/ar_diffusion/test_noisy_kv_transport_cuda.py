# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Four-rank NCCL coverage for the fixed-shape Same-Step activation path."""

import pytest
import torch

from tests.diffusion.ar_diffusion.test_noisy_kv_transport import _sink_pipeline_worker, get_open_port
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=4)
@pytest.mark.parametrize("history", [0, 1])
@pytest.mark.parametrize("chunks", [1, 6])
def test_same_step_fixed_shape_nccl_pipeline_matches_serial(history, chunks):
    torch.multiprocessing.spawn(
        _sink_pipeline_worker,
        args=(2, 2, chunks, f"tcp://127.0.0.1:{get_open_port()}", history, True, "cuda"),
        nprocs=4,
        join=True,
    )
