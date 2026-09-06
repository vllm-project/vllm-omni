# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for model-declared AR-Diffusion capabilities."""

import pytest

from vllm_omni.experimental.ar_diffusion.capability import cfg_kv_branches

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("cfg_enabled", "cfg_parallel_world_size", "expected"),
    [
        (False, 1, (("positive", 0),)),
        (False, 2, (("positive", 0),)),
        (True, 1, (("positive", 0), ("negative", 1))),
        (True, 2, (("positive", 0), ("negative", 0))),
    ],
)
def test_cfg_kv_branches_matches_active_cfg_topology(
    cfg_enabled: bool,
    cfg_parallel_world_size: int,
    expected: tuple[tuple[str, int], ...],
) -> None:
    branches = cfg_kv_branches(
        cfg_enabled=cfg_enabled,
        cfg_parallel_world_size=cfg_parallel_world_size,
    )

    assert tuple((branch.name, branch.local_index) for branch in branches) == expected


def test_cfg_kv_branches_preserves_model_branch_names() -> None:
    branches = cfg_kv_branches(
        cfg_enabled=True,
        cfg_parallel_world_size=1,
        positive_name="conditional",
        negative_name="unconditional",
    )

    assert tuple(branch.name for branch in branches) == ("conditional", "unconditional")
