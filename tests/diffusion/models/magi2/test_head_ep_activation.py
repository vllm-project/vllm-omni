# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Head-sharded expert parallelism for MAGI-2.

``Magi2Pipeline`` declares ``expert_parallel_style="head"``, ``get_magi2_ep_group()``
returns the head-EP subgroup when it exists, and ``validate_magi2_expert_parallel``
accepts the layout.  These are import-only checks; the collectives need real GPUs.
"""

from types import SimpleNamespace

import pytest
import torch.distributed as dist

from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.magi2.parallel import (
    Magi2ParallelGroup,
    _head_expert_parallel_group,
    get_magi2_ep_group,
    get_magi2_ep_split_indices,
    validate_magi2_expert_parallel,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def test_magi2_declares_the_head_expert_parallel_style():
    metadata = get_diffusion_model_metadata("Magi2Pipeline")

    assert metadata.expert_parallel_style == "head"


def test_other_pipelines_keep_the_default_style():
    # The default must stay "vllm" so declaring head-EP is an explicit opt-in.
    assert get_diffusion_model_metadata("WanPipeline").expert_parallel_style == "vllm"


def test_head_group_is_absent_when_ep_is_not_initialized():
    # No process group is up in this test, so the accessor must report absence
    # rather than inventing a rank-local group.
    assert _head_expert_parallel_group() is None


def test_ep_group_falls_back_without_distributed():
    assert not dist.is_initialized()
    group = get_magi2_ep_group()
    assert group.world_size == 1
    assert group.group is None


def test_expert_parallel_validation_rejects_tensor_parallelism():
    # The MoE head axis spans the full hidden size and the EP-sharded weights
    # are sliced over it, so tensor parallelism must not column-shard it too.
    with pytest.raises(ValueError, match="requires tensor_parallel_size=1"):
        validate_magi2_expert_parallel(SimpleNamespace(enable_expert_parallel=True, tensor_parallel_size=2))

    validate_magi2_expert_parallel(SimpleNamespace(enable_expert_parallel=True, tensor_parallel_size=1))
    # Without the head-EP opt-in, TP keeps being the MoE-head axis.
    validate_magi2_expert_parallel(SimpleNamespace(enable_expert_parallel=False, tensor_parallel_size=4))


def test_ep_split_indices_reindex_token_counts_into_ep_order(monkeypatch):
    # ``cp_split_sizes`` is indexed by SP rank while the head dispatch is
    # indexed by EP rank, and an EP subgroup need not start at SP rank 0.
    monkeypatch.setattr(dist, "get_process_group_ranks", list)
    sp_group = Magi2ParallelGroup(group=(0, 1, 2, 3), world_size=4, rank=2)
    cp_split_sizes = [100, 200, 300, 400]

    first = get_magi2_ep_split_indices(Magi2ParallelGroup(group=(0, 1), world_size=2, rank=0), sp_group)
    second = get_magi2_ep_split_indices(Magi2ParallelGroup(group=(2, 3), world_size=2, rank=0), sp_group)

    assert first == (0, 1)
    assert second == (2, 3)
    assert [cp_split_sizes[index] for index in first] == [100, 200]
    assert [cp_split_sizes[index] for index in second] == [300, 400]
    # Head-EP over the whole SP group is the identity permutation.
    assert get_magi2_ep_split_indices(sp_group, sp_group) == (0, 1, 2, 3)


def test_ep_split_indices_reject_an_ep_group_outside_the_sp_group(monkeypatch):
    # Re-indexing a group outside the SP group would assign token counts to the wrong ranks.
    monkeypatch.setattr(dist, "get_process_group_ranks", list)
    sp_group = Magi2ParallelGroup(group=(0, 1, 2, 3), world_size=4, rank=0)

    with pytest.raises(ValueError, match="must be contained in its SP group"):
        get_magi2_ep_split_indices(Magi2ParallelGroup(group=(0, 4), world_size=2, rank=0), sp_group)


def test_replicated_sequence_groups_have_no_split_projection():
    # The TP fallback replicates the sequence, so SP-ordered counts do not apply.
    tp_group = Magi2ParallelGroup(group=None, world_size=1, rank=0, replicated_sequence=True)

    assert get_magi2_ep_split_indices(tp_group, tp_group) is None
