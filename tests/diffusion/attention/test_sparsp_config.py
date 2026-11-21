# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Verify role-specific parsing of SparSP routing and planning."""

import pytest

from vllm_omni.diffusion.attention.sparsp.config import SparSPSpec
from vllm_omni.diffusion.data import AttentionSpec, parse_attention_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("routing", ["full_ring", "selective_direct"])
def test_sparsp_parses_routing(routing):
    attention = parse_attention_config(
        {"per_role": {"ltx2.video_self": {"backend": "SPARSP_ATTN", "sparsp": {"routing": routing}}}}
    )
    spec, _ = attention.resolve_with_source("ltx2.video_self", "self")
    assert spec is not None
    assert spec.backend_kwargs()["sparsp"] == {
        "routing": routing,
        "cdf": 0.9,
        "sim_threshold": 0.1,
        "pv_threshold": 50.0,
        "demand_planner": "torch",
        "placement": "none",
    }
    assert attention.resolve_with_source("ltx2.audio_self", "self")[0] is None


def test_sparsp_requires_its_backend():
    with pytest.raises(ValueError, match="sparsp options require backend=SPARSP_ATTN"):
        AttentionSpec(backend="TORCH_SDPA", sparsp=SparSPSpec(routing="selective_direct"))


def test_sparsp_serializes_manifest_demand():
    spec = AttentionSpec(backend="SPARSP_ATTN", sparsp={"routing": "selective_direct", "demand_planner": "manifest"})
    assert spec.backend_kwargs()["sparsp"]["demand_planner"] == "manifest"


def test_sparsp_manifest_requires_demand():
    with pytest.raises(ValueError, match="manifest demand planner requires selective routing"):
        SparSPSpec(demand_planner="manifest")


def test_sparsp_pair_relay_requires_manifest():
    with pytest.raises(ValueError, match="pair relay requires the manifest"):
        SparSPSpec(routing="selective_pair_relay")
    spec = AttentionSpec(
        backend="SPARSP_ATTN",
        sparsp={"routing": "selective_pair_relay", "demand_planner": "manifest"},
    )
    assert spec.backend_kwargs()["sparsp"]["routing"] == "selective_pair_relay"


def test_demand_plans_keep_each_peers_native_head_and_block_order(monkeypatch):
    import torch

    from vllm_omni.diffusion.attention.sparsp.executor import SparseRingExecutor

    executor = SparseRingExecutor.__new__(SparseRingExecutor)
    executor.tokens, executor.heads, executor.world, executor.rank = 128, 2, 3, 1
    executor.manifest = None
    monkeypatch.setattr(executor, "group", None, raising=False)
    selected = torch.zeros(1, 2, 2, 6, dtype=torch.bool)
    selected[0, 0, 0, 1] = True
    selected[0, 1, 1, 4] = True

    def exchange(gives, wants, *, group):
        assert wants[0, 0, 1] and wants[2, 1, 0]
        assert wants.sum() == 2
        gives.zero_()
        gives[2, 0, 0] = True
        gives[2, 1, 1] = True

    monkeypatch.setattr(torch.distributed, "all_to_all_single", exchange)
    send, receive = executor._demands(selected)
    assert send[0].tolist() == [[0, 1], [0, 1]]
    assert send[1].numel() == 0
    assert receive[0].tolist() == [[0], [1]]
    assert receive[1].tolist() == [[1], [0]]
