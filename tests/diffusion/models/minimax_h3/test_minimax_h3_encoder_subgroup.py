# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Gloo regression for an encoder TP group smaller than the DiT world."""

import ast
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def load_builder(rank):
    root = Path(__file__).resolve().parents[4]
    path = root / "vllm_omni/diffusion/distributed/group_coordinator.py"
    coordinator = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "GroupCoordinator"
    )
    init = next(n for n in coordinator.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    cls = ast.ClassDef(name="Coordinator", bases=[], keywords=[], body=[init], decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {
        "torch": torch,
        "dist": dist,
        "envs": SimpleNamespace(LOCAL_RANK=rank),
        "current_omni_platform": SimpleNamespace(dist_backend="gloo", get_torch_device=lambda _: torch.device("cpu")),
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    namespace["init_world_group"] = lambda ranks, local_rank, backend: namespace["Coordinator"](
        [ranks], local_rank, backend
    )
    path = root / "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name in ("_SingleRankEncoderGroup", "_NonMemberEncoderGroup")
    ]
    pipeline = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3Pipeline")
    method = next(n for n in pipeline.body if isinstance(n, ast.FunctionDef) and n.name == "_build_text_encoder_group")
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *nodes, method], type_ignores=[])), str(path), "exec"
        ),
        namespace,
    )
    return namespace["_build_text_encoder_group"]


def subgroup_worker(rank, rendezvous):
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=rendezvous, timeout=timedelta(seconds=30))
    try:
        group = load_builder(rank)(SimpleNamespace(_dit_rank=rank), 2)
        assert group.world_size == 2 and group.ranks == [0, 1]
        if rank < 2:
            assert group.rank_in_group == rank
            value = torch.ones(1)
            dist.all_reduce(value, group=group.device_group)
            assert torch.equal(value, torch.tensor([2.0]))
        else:
            assert group.rank_in_group == -1 and group.device_group is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is required")
def test_text_encoder_subgroup_allows_non_encoder_ranks(tmp_path):
    mp.spawn(subgroup_worker, args=((tmp_path / "subgroup-rendezvous").as_uri(),), nprocs=4, join=True)
