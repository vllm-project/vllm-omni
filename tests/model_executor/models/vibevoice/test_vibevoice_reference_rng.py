# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Seeded reference encoding must not touch other accelerators' RNG (stub tower, no weights)."""

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("index", [0, 1])
def test_seeded_reference_keeps_other_device_rng(reference_encoder, index):
    device_module = torch.get_device_module("cuda")
    device = torch.device("cuda", index)
    encoder = reference_encoder(device)
    row, count, seed = torch.zeros(1, 8), torch.tensor(4), 5
    torch.manual_seed(1)
    before = [torch.get_rng_state()] + [device_module.get_rng_state(i) for i in range(2)]
    (embedding,) = encoder([row], [count], [torch.tensor([1, seed])])
    after = [torch.get_rng_state()] + [device_module.get_rng_state(i) for i in range(2)]
    # CPU, the encoding device and the other device are all restored, even when
    # the encoding device is not the current device.
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
    # Same draw as the reference implementation after torch.manual_seed(seed).
    torch.manual_seed(seed)
    values = row[None].to(device)
    expected = (values + torch.randn_like(values)).transpose(1, 2)[0, :4]
    assert torch.equal(embedding, expected)
