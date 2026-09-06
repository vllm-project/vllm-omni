# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm_omni.diffusion.models.magi2.mh_moe import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
)
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup


def make_module():
    module = Magi2MultiHeadMoE(
        Magi2MultiHeadMoEConfig(32, 2, 7, 2, 24, torch.bfloat16),
        ep_group=Magi2ParallelGroup(None, 1, 0),
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-1, 1)
    return module


def test_values_storage_identity_and_checkpoint():
    module = make_module()
    gate, up = module.W_gate.clone(), module.W_up.clone()
    identities = id(module.W_gate), id(module.W_up)
    keys = set(module.state_dict())
    module.prepare_owned_w13()
    assert identities == (id(module.W_gate), id(module.W_up))
    assert keys == set(module.state_dict())
    assert torch.equal(module.W_gate, gate)
    assert torch.equal(module.W_up, up)
    packed = module._get_owned_w13()
    expected = torch.stack((gate.transpose(1, 2), up.transpose(1, 2)), dim=2).flatten(
        1, 2
    )
    assert torch.equal(packed, expected)
    assert (
        module.W_gate.untyped_storage().data_ptr()
        == module.W_up.untyped_storage().data_ptr()
    )
    assert (
        packed.untyped_storage().nbytes()
        == (gate.numel() + up.numel()) * gate.element_size()
    )
    module.prepare_owned_w13()
    assert module._get_owned_w13() is packed


def test_inplace_update_and_replacement():
    module = make_module()
    module.prepare_owned_w13()
    with torch.no_grad():
        module.W_gate.fill_(3)
    assert torch.all(module._get_owned_w13()[:, 0::2] == 3)
    module.W_gate.data = module.W_gate.clone().contiguous()
    assert module._get_owned_w13() is None
    module.prepare_owned_w13()
    assert torch.all(module._get_owned_w13()[:, 0::2] == 3)


def test_device_dtype_conversion_invalidates_alias():
    module = make_module()
    module.prepare_owned_w13()
    module.float()
    assert module._get_owned_w13() is None
    with pytest.raises(ValueError, match="BF16"):
        module.prepare_owned_w13()


def test_checkpoint_reload_updates_views():
    module = make_module()
    other = make_module()
    module.prepare_owned_w13()
    module.load_state_dict(other.state_dict())
    assert module._get_owned_w13() is not None
    assert torch.equal(module.W_gate, other.W_gate)
    assert torch.equal(module.W_up, other.W_up)


def test_owned_w2_layout_and_values():
    module = make_module()
    down = module.W_down.clone()
    module.prepare_owned_w2()
    assert torch.equal(module._get_owned_w2(), down.transpose(1, 2))
    assert torch.equal(module.W_down, down)
    assert module._get_owned_w2().is_contiguous()
    module.prepare_owned_w2()
