# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import ctypes
import hashlib
from importlib import resources

import pytest

from vllm_omni.diffusion.attention.backends.utils import rocm_h3_fmha as native

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


def test_unknown_code_object_falls_back():
    assert native._with_tile_first_grid(b"unknown AITER release") is None


def test_schedule_only_changes_stock_register_moves():
    pytest.importorskip("aiter_meta")
    kernel = resources.files("aiter_meta").joinpath("hsa/gfx950/fmha_v3_fwd/fwd_hd128_bf16_group.co")
    if not kernel.is_file():
        pytest.skip("gfx950 ASM object is not installed")
    original = kernel.read_bytes()
    candidate = native._with_tile_first_grid(original)
    if hashlib.sha256(original).hexdigest() != native._STOCK_SHA:
        assert candidate is None
        return
    assert candidate is not None
    assert hashlib.sha256(candidate).hexdigest() == native._SCHEDULED_SHA
    assert sum(x != y for x, y in zip(original, candidate, strict=True)) == 4


@pytest.mark.parametrize(("row_stride", "head_stride"), [(1792, 128), (5376, 128), (5376, 384)])
def test_input_strides_and_contiguous_output_abi(row_stride, head_stride):
    # All native strides are in bytes, including the stride between 256-row query tiles.
    args = native._layout_args(row_stride, head_stride)
    assert ctypes.sizeof(args) == 656
    assert args.seq_stride.u32 == args.k_seq_stride.u32 == args.v_seq_stride.u32 == row_stride * 2
    assert args.head_stride.u32 == args.k_head_stride.u32 == args.v_head_stride.u32 == head_stride * 2
    assert args.tile_stride.u32 == 256 * row_stride * 2
    assert (args.o_seq_stride.u32, args.o_head_stride.u32) == (1792 * 2, 128 * 2)
