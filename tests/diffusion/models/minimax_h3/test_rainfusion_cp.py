# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_rainfusion_cp_layout_reorders_before_split_and_restores_after_gather():
    from vllm_omni.diffusion.models.minimax_h3.rainfusion_cp import RainFusionCPLayout

    # Non-divisible H/W covers rf_v2's first-frame/residual traversal too.
    layout = RainFusionCPLayout.build(
        prefix_len=5,
        latent_grid=(2, 9, 10),
        world_size=2,
        rank=1,
    )
    source = torch.arange(layout.used_len + 11).unsqueeze(-1)
    rearranged = layout.prearrange(source)

    assert layout.local_capacity % 128 == 0
    assert layout.physical_len == layout.local_capacity * 2
    assert layout.q_global_start == layout.local_capacity
    assert layout.q_valid_len == max(0, layout.used_len - layout.local_capacity)
    assert rearranged.shape[0] == layout.physical_len
    torch.testing.assert_close(layout.restore(rearranged), source[: layout.used_len])


def test_rainfusion_cp_layout_keeps_last_rank_logically_short_but_physically_aligned():
    from vllm_omni.diffusion.models.minimax_h3.rainfusion_cp import RainFusionCPLayout

    layouts = [
        RainFusionCPLayout.build(prefix_len=3, latent_grid=(2, 8, 8), world_size=3, rank=rank)
        for rank in range(3)
    ]

    assert {layout.local_capacity for layout in layouts} == {128}
    assert [layout.q_valid_len for layout in layouts] == [128, 3, 0]
    assert all(layout.physical_len == 384 for layout in layouts)
