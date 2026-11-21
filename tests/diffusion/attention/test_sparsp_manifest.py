# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.sparsp.manifest import DemandManifest, PairRelayManifest, peer_hops

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": ["L4", "H100"]})]


@pytest.mark.parametrize(("world", "blocks"), [(2, 360), (3, 185), (4, 180)])
def test_manifest_preserves_packed_order_on_reuse(world, blocks):
    device = torch.device("cuda")
    heads = 32
    wants = torch.arange(world * heads * blocks, device=device).reshape(world, heads, blocks) % 11 != 0
    gives = wants.roll(1, 0)
    for rank in range(world):
        hops = peer_hops(rank, world, min(2, world - 1))
        manifest = DemandManifest(rank, world, heads, blocks, device)
        empty, full = torch.zeros_like(gives), torch.ones_like(wants)
        for outgoing, incoming in ((gives, wants), (empty, full), (gives, wants)):
            manifest.compact(outgoing, incoming)
            torch.cuda.current_stream(device).synchronize()
            send, receive = manifest.plan()
            for route, sent, received in zip(hops, send, receive, strict=True):
                assert torch.equal(sent, outgoing[route.destination].nonzero().T)
                assert torch.equal(received, incoming[route.source].nonzero().T)


def check_pair_relay_layout(device):
    heads, blocks = 7, 13
    manifests = [PairRelayManifest(rank, heads, blocks, device) for rank in range(4)]
    indices = torch.arange(16 * heads * blocks, device=device).reshape(4, 4, heads, blocks)
    for demand in (indices % 11 < 7, indices % 11 < 3, indices < 0, indices >= 0):
        for manifest in manifests:
            manifest.wants.copy_(demand)
            manifest.compact()
        torch.cuda.current_stream(device).synchronize()
        plans = [manifest.plan() for manifest in manifests]
        for rank, plan in enumerate(plans):
            within, across, forwarded = rank ^ 1, rank ^ 2, rank ^ 3
            assert torch.equal(plan.local_send, demand[within, rank].nonzero().T)
            assert torch.equal(plan.local_receive, plans[within].local_send)
            assert torch.equal(plan.cross_receive, plans[across].cross_send)
            prefix = demand[within, across]
            suffix = demand[rank, across] & ~prefix
            expected_cross = torch.cat((prefix.nonzero(), suffix.nonzero())).T
            assert torch.equal(plan.cross_receive, expected_cross)
            assert plan.forward_count == prefix.sum().item()
            assert torch.equal(plan.forward_receive, demand[rank, forwarded].nonzero().T)
            assert torch.equal(
                plan.forward_receive,
                plans[within].cross_receive[:, : plans[within].forward_count],
            )


def test_pair_relay_keeps_forwarded_prefix_contiguous_on_reuse():
    check_pair_relay_layout(torch.device("cuda"))
