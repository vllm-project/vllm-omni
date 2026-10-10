# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sparse wrapper admission is independent of attention strategies."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.attention.block_sparse import validate_block_sparse_parallel

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("degree", [1, 2, 4])
def test_local_or_strict_ulysses(degree):
    config = SimpleNamespace(parallel_config=SimpleNamespace(ulysses_degree=degree, sequence_parallel_size=degree))
    assert validate_block_sparse_parallel(config) == degree


@pytest.mark.parametrize(
    "options",
    [
        {"ring_degree": 2},
        {"allgather_degree": 2},
        {"sequence_parallel_size": 4},
        {"ulysses_mode": "advanced_uaa"},
        {"use_hsdp": True},
        {"tensor_parallel_size": 2},
        {"pipeline_parallel_size": 2},
        {"data_parallel_size": 2},
        {"cfg_parallel_size": 2},
    ],
)
def test_unvalidated_parallel_combinations_are_rejected(options):
    parallel = SimpleNamespace(**{"ulysses_degree": 2, "sequence_parallel_size": 2, **options})
    with pytest.raises(ValueError, match="Ulysses"):
        validate_block_sparse_parallel(SimpleNamespace(parallel_config=parallel))
