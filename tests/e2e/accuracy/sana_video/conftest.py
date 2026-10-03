# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest


@pytest.fixture
def sana_model_parallel(tmp_path):
    """Construct native parallel layers with real single-rank GPU groups."""
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"file://{tmp_path / 'sana-distributed'}",
    )
    try:
        initialize_model_parallel()
        yield
    finally:
        destroy_distributed_env()
