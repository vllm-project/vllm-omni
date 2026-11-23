# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ctypes

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.utils import rocm_h3_head_fmha as native

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
def test_unqualified_kernel_is_not_loaded():
    assert native._tile_first(b"unknown gfx950 kernel release") is None
    assert native.prepare_head_fmha(torch.device("cpu")) is None


@pytest.mark.cpu
@pytest.mark.parametrize("heads", [4, 10])
def test_split_packed_receive_abi(heads):
    args = native._layout_args(heads)
    assert ctypes.sizeof(args) == 656
    assert native._Args.q_bounds.offset == 27 * 16
    assert native._Args.padded_k_bounds.offset == 31 * 16
    assert args.heads.u32 == heads
    assert args.seq_len.u32 == args.kv_seq_len.u32 == 37760
    assert args.seq_stride.u32 == args.k_seq_stride.u32 == args.v_seq_stride.u32 == 3 * heads * 128 * 2
    assert args.tile_stride.u32 == 256 * args.seq_stride.u32
    assert args.o_seq_stride.u32 == heads * 128 * 2
    assert args.o_head_stride.u32 == args.head_stride.u32 == args.k_head_stride.u32 == args.v_head_stride.u32 == 256


@pytest.mark.cpu
def test_unqualified_head_group():
    with pytest.raises(ValueError, match="head group size"):
        native._layout_args(7)


@hardware_test(res={"rocm": "MI350"}, num_cards=1)
@pytest.mark.parametrize("heads", [4, 10])
@torch.inference_mode()
def test_grouped_head_output_matches_aiter(heads):
    if torch.version.hip is None or not torch.accelerator.is_available():
        pytest.skip("requires ROCm")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    run = native.prepare_head_fmha(device)
    if run is None:
        pytest.skip("requires gfx950 and the qualified AITER code object")
    from aiter import flash_attn_varlen_func

    generator = torch.Generator(device=device).manual_seed(15)
    wire = torch.randn((4, 9440, 3, heads, 128), device=device, dtype=torch.bfloat16, generator=generator)
    output = wire.new_empty((4, 9440, heads, 128))
    cut = 37749
    bounds = torch.tensor([0, cut, 37760], device=device, dtype=torch.int32)
    run(wire, bounds, output)
    q, k, v = (t.contiguous() for t in wire.view(37760, 3, heads, 128).unbind(1))
    reference = flash_attn_varlen_func(q, k, v, bounds, bounds, cut, cut, softmax_scale=128**-0.5, causal=False)
    assert torch.equal(output.view(37760, heads, 128).view(torch.int16), reference.view(torch.int16))
