# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for mask-keyed MoE CUDA-graph descriptor expansion (#9865)."""

from dataclasses import replace

import pytest

from vllm_omni.cudagraph.moe_routing_batch_descriptor import (
    BatchDescriptor,
    patch_batch_descriptor_moe_routing_key,
)
from vllm_omni.cudagraph.moe_routing_dispatcher import (
    expand_batch_descriptors_for_moe_routing,
    with_moe_routing_key,
)
from vllm_omni.model_executor.models.mammoth_moda2.moe_cudagraph import MoERoutingKey

patch_batch_descriptor_moe_routing_key()

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_expand_batch_descriptors_disabled_keeps_blend_only() -> None:
    base = BatchDescriptor(num_tokens=8, num_reqs=8, uniform=True)
    out = expand_batch_descriptors_for_moe_routing([base], enabled=False)
    assert len(out) == 1
    assert out[0].moe_routing_key == int(MoERoutingKey.BLEND)


def test_expand_batch_descriptors_enabled_triples_keys() -> None:
    base = BatchDescriptor(num_tokens=8, num_reqs=8, uniform=True, num_active_loras=0)
    out = expand_batch_descriptors_for_moe_routing([base], enabled=True)
    keys = sorted(d.moe_routing_key for d in out)
    assert keys == [
        int(MoERoutingKey.BLEND),
        int(MoERoutingKey.ALL_UND),
        int(MoERoutingKey.ALL_GEN),
    ]
    assert all(d.num_tokens == 8 for d in out)


def test_with_moe_routing_key_replace() -> None:
    desc = BatchDescriptor(num_tokens=4)
    keyed = with_moe_routing_key(desc, MoERoutingKey.ALL_UND)
    assert keyed.moe_routing_key == int(MoERoutingKey.ALL_UND)
    assert keyed.num_tokens == 4
    # Distinct graph keys.
    assert keyed != replace(desc, moe_routing_key=int(MoERoutingKey.BLEND))
