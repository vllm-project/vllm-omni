# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The owned W13 bank keeps the fused layout without a second weight bank.

``_pack_bf16_w13`` builds the grouped-GEMM layout by copying, so a module that
uses only the packed form still holds gate/up *and* the packed copy.  The owned
path instead allocates one bank and rebinds gate/up as strided views inside it.
These tests pin the contract that makes that swap safe: values, dtypes and
state_dict keys are untouched, and the bank is dropped as soon as the parameters
stop pointing into it.
"""

import copy

import pytest
import torch

from tests.diffusion.models.magi2.test_bf16_moe_wiring import _gpu_device
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.magi2 import mh_moe
from vllm_omni.diffusion.models.magi2.mh_moe import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
)
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def make_module() -> Magi2MultiHeadMoE:
    module = Magi2MultiHeadMoE(
        Magi2MultiHeadMoEConfig(32, 2, 7, 2, 24, torch.bfloat16),
        ep_group=Magi2ParallelGroup(None, 1, 0),
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-1, 1)
    return module


def make_host_module() -> Magi2MultiHeadMoE:
    """Build the module on the host whatever the session default device is."""

    with torch.device("cpu"):
        return make_module()


@pytest.mark.cpu
def test_values_storage_identity_and_checkpoint():
    module = make_module()
    gate, up = module.W_gate.clone(), module.W_up.clone()
    identities = id(module.W_gate), id(module.W_up)
    keys = set(module.state_dict())

    module._rebind_w13_to_owned_bank()

    # Parameter objects and checkpoint keys must survive: this is a storage
    # rebind, not a module rewrite.
    assert identities == (id(module.W_gate), id(module.W_up))
    assert keys == set(module.state_dict())
    assert torch.equal(module.W_gate, gate)
    assert torch.equal(module.W_up, up)

    packed = module._get_owned_w13()
    expected = torch.stack((gate.transpose(1, 2), up.transpose(1, 2)), dim=2).flatten(1, 2)
    assert torch.equal(packed, expected)
    # One bank, not three: gate/up must alias the packed storage.
    assert module.W_gate.untyped_storage().data_ptr() == module.W_up.untyped_storage().data_ptr()
    assert packed.untyped_storage().nbytes() == (gate.numel() + up.numel()) * gate.element_size()
    module._rebind_w13_to_owned_bank()
    assert module._get_owned_w13() is packed


@pytest.mark.cpu
def test_inplace_update_and_replacement():
    module = make_module()
    module._rebind_w13_to_owned_bank()

    # In-place writes must be visible through the packed bank (the views and
    # the bank are the same memory).
    with torch.no_grad():
        module.W_gate.fill_(3)
    assert torch.all(module._get_owned_w13()[:, 0::2] == 3)

    # ``.data = ...`` rebinds to foreign storage; the stale bank must be dropped
    # rather than returned with the old contents.
    module.W_gate.data = module.W_gate.clone().contiguous()
    assert module._get_owned_w13() is None

    module._rebind_w13_to_owned_bank()
    assert torch.all(module._get_owned_w13()[:, 0::2] == 3)


@pytest.mark.cpu
def test_device_dtype_conversion_invalidates_alias():
    module = make_module()
    module._rebind_w13_to_owned_bank()
    module.float()
    assert module._get_owned_w13() is None
    with pytest.raises(ValueError, match="BF16"):
        module._rebind_w13_to_owned_bank()


@pytest.mark.cpu
def test_checkpoint_reload_updates_views():
    module = make_module()
    other = make_module()
    module._rebind_w13_to_owned_bank()

    # The loader copies in place, so the owned views stay valid and must carry
    # the new checkpoint values.
    module.load_state_dict(other.state_dict())
    assert module._get_owned_w13() is not None
    assert torch.equal(module.W_gate, other.W_gate)
    assert torch.equal(module.W_up, other.W_up)


@pytest.mark.cpu
def test_owned_bank_is_not_extra_state():
    module = make_module()
    before = set(module.state_dict())
    module._rebind_w13_to_owned_bank()
    # A plain attribute, not a registered buffer: no checkpoint entry, and ``.to()``
    # does not migrate it as a second bank beside the parameters.
    assert set(module.state_dict()) == before
    assert all(buffer is not module._owned_w13 for buffer in module.buffers())


@pytest.mark.cpu
def test_packed_w13_returns_the_owned_bank_when_present():
    module = make_module()
    module._rebind_w13_to_owned_bank()
    owned = module._get_owned_w13()
    # The fused path must reuse the owned bank instead of copying beside it.
    assert module._get_bf16_packed_w13() is owned


@pytest.mark.cpu
def test_packed_w13_falls_back_to_copy_when_not_owned():
    module = make_module()
    assert module._get_owned_w13() is None
    packed = module._get_bf16_packed_w13()
    assert packed.shape == (
        module.local_flatten_num_experts,
        2 * module.d_expert,
        module.d_head,
    )
    assert module._get_owned_w13() is None


@pytest.mark.cpu
def test_no_op_move_keeps_the_owned_bank():
    module = make_module()
    module._rebind_w13_to_owned_bank()
    packed13 = module._get_owned_w13()

    # ``to`` routes through ``_apply`` even when nothing has to move.  The
    # aliases survive such a call, so the bank must survive with them.
    module.to("cpu")

    assert module._get_owned_w13() is packed13


@pytest.mark.cpu
def test_unowned_module_stays_unowned_across_a_move(monkeypatch):
    """Only a module that asked for the layout may own it.

    The direct-mmap and staged loaders move modules to the device after
    loading, and they bind checkpoint views that must not be copied behind
    their back.
    """

    # Allow host weights, so that only the ownership request decides.
    monkeypatch.setattr(mh_moe, "_can_own_fused_w13", lambda weight: weight.dtype == torch.bfloat16)
    module = make_host_module()
    module.to("cpu")
    module._apply(lambda tensor: tensor.clone())

    assert module._get_owned_w13() is None
    assert module.W_gate.untyped_storage().data_ptr() != module.W_up.untyped_storage().data_ptr()


@pytest.mark.cpu
def test_host_weights_are_left_alone():
    module = make_host_module()
    module.prepare_bf16_weights()

    # Host and meta weights never reach the fused grouped GEMM, so neither the
    # owned bank nor the copying pack belongs on them.
    assert module._owns_fused_layout is False
    assert module._get_owned_w13() is None
    assert module._bf16_packed_w13 is None


@pytest.mark.cpu
def test_move_rebuilds_the_owned_bank(monkeypatch):
    """A move must not leave the fused path packing a copy beside two banks.

    The resident pipeline stages the transformer to the host and back around
    prompt encoding, so an unhandled move would silently drop the fused layout
    for the rest of the process.
    """

    module = make_module()
    gate, up = module.W_gate.clone(), module.W_up.clone()
    # This file runs on host tensors, so relax the residency half of the guard
    # and let a storage rebuild stand in for a device move.  ``_apply``, the
    # ownership guards and the rebuild are the real implementations.
    monkeypatch.setattr(mh_moe, "_can_own_fused_w13", lambda weight: weight.dtype == torch.bfloat16)

    module.prepare_bf16_weights()
    assert module._get_owned_w13() is not None

    module._apply(lambda tensor: tensor.clone())

    packed13 = module._get_owned_w13()
    assert packed13 is not None
    # One bank, not three: the rebuilt parameters alias each other again, and
    # the fused layout is that same bank rather than a copy beside it.
    assert module.W_gate.untyped_storage().data_ptr() == module.W_up.untyped_storage().data_ptr()
    assert module._get_bf16_packed_w13() is packed13
    assert torch.equal(module.W_gate, gate)
    assert torch.equal(module.W_up, up)


@pytest.mark.cpu
def test_replacing_only_up_drops_the_owned_bank():
    module = make_module()
    module._rebind_w13_to_owned_bank()
    module.W_up.data = module.W_up.clone().contiguous()
    assert module._get_owned_w13() is None


@pytest.mark.cpu
def test_dtype_round_trip_drops_and_rebuilds_the_owned_bank(monkeypatch):
    monkeypatch.setattr(mh_moe, "_can_own_fused_w13", lambda weight: weight.dtype == torch.bfloat16)
    module = make_module()
    gate, up = module.W_gate.clone(), module.W_up.clone()
    module.prepare_bf16_weights()

    module.float()
    assert module._get_owned_w13() is None
    assert module.W_gate.dtype == torch.float32

    module.bfloat16()
    assert module._get_owned_w13() is not None
    assert torch.equal(module.W_gate, gate)
    assert torch.equal(module.W_up, up)


@pytest.mark.cpu
def test_owned_bank_forward_matches_the_unowned_module():
    module = make_module()
    reference = copy.deepcopy(module)
    module._rebind_w13_to_owned_bank()
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(9, module.local_num_heads, module.d_head, generator=generator).to(torch.bfloat16)
    with torch.inference_mode():
        torch.testing.assert_close(module._local_forward(x), reference._local_forward(x), rtol=0, atol=0)


@pytest.mark.parametrize(
    "device_type",
    [
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}, num_cards=1)),
        pytest.param("musa", marks=hardware_marks(res={"musa": "S5000"}, num_cards=1)),
    ],
)
@pytest.mark.parametrize("deterministic", ["0", "1"])
def test_owned_bank_forward_matches_the_unowned_module_on_device(monkeypatch, device_type, deterministic):
    """The fused path and the deterministic Triton kernel read the strided gate/up views exactly like the contiguous banks."""

    device = _gpu_device(device_type)
    monkeypatch.setenv("MAGI2_DETERMINISTIC", deterministic)
    with torch.device("cpu"):
        reference = Magi2MultiHeadMoE(
            # d_expert 96 is a multiple of the reference kernel's 32-wide tile, so
            # MAGI2_DETERMINISTIC=1 runs the Triton kernel rather than the Torch fallback.
            Magi2MultiHeadMoEConfig(128, 2, 4, 2, 96, torch.bfloat16),
            ep_group=Magi2ParallelGroup(None, 1, 0),
        )
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.5, 0.5)
    owned = copy.deepcopy(reference).to(device)
    reference = reference.to(device)
    owned.prepare_bf16_weights()
    assert owned._get_owned_w13() is not None and reference._get_owned_w13() is None
    # 4100 tokens also take the fused path on CUDA, which starts at 4096.
    generator = torch.Generator().manual_seed(7)
    x = (torch.randn(4100, 2, 64, generator=generator) * 0.5).to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = reference._local_forward(x)
        actual = owned._local_forward(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
