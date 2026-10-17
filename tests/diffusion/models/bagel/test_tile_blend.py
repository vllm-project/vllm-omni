# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.platforms import current_platform

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import GridSpec
from vllm_omni.diffusion.models.bagel import autoencoder
from vllm_omni.diffusion.models.bagel.tile_blend import try_blend

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cuda
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("axis", [2, 3])
def test_blend_preserves_rounding_for_strided_tiles_in_shared_storage(monkeypatch, dtype, axis):
    packed = torch.randn(2, 2, 3, 11, 26, device="cuda", dtype=dtype)
    expected = packed.clone()
    blend = autoencoder.DistributedAutoEncoder.blend_v if axis == 2 else autoencoder.DistributedAutoEncoder.blend_h
    with monkeypatch.context() as patch:
        patch.setattr(autoencoder, "try_blend", lambda *args, **kwargs: False)
        blend(None, expected[0, ..., ::2], expected[1, ..., ::2], 7)
    assert try_blend(packed[0, ..., ::2], packed[1, ..., ::2], 7, axis)
    int_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    # Compare the full backing tensor to include source and unwritten elements.
    assert torch.equal(packed.view(int_dtype), expected.view(int_dtype))


@pytest.mark.cuda
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@torch.inference_mode()
def test_overlapping_tiles_use_the_original_loop():
    packed = torch.randn(1, 3, 11, 14, device="cuda")
    before = packed.clone()
    assert not try_blend(packed[..., :-1], packed[..., 1:], 7, 3)
    assert torch.equal(packed, before)


@pytest.mark.cuda
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_grad_enabled_blend_uses_the_original_loop():
    source = torch.randn(1, 3, 11, 13, device="cuda", requires_grad=True)
    current = torch.randn_like(source)
    with torch.enable_grad():
        assert not try_blend(source, current, 7, 2)


@pytest.mark.cpu
def test_cpu_blend_clips_extent_and_keeps_noop():
    source = torch.ones(1, 1, 2, 3)
    current = torch.zeros(1, 1, 2, 3)
    blend = autoencoder.DistributedAutoEncoder.blend_v
    assert blend(None, source, current, 0) is current
    assert torch.count_nonzero(current) == 0
    assert blend(None, source, current, 7) is current
    assert torch.equal(current, torch.tensor([[[[1.0] * 3, [0.5] * 3]]]))


@pytest.mark.cuda
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@torch.inference_mode()
def test_tile_merge_preserves_corners_and_clipped_edges(monkeypatch):
    # The merge stage needs no weights or process group. Use its real methods.
    model = autoencoder.DistributedAutoEncoder.__new__(autoencoder.DistributedAutoEncoder)
    torch.nn.Module.__init__(model)
    model.tile_sample_stride_height = 7
    model.tile_sample_stride_width = 7
    packed = torch.randn(4, 1, 3, 11, 11, device="cuda", dtype=torch.bfloat16)
    expected_tiles = packed.clone()
    spec = GridSpec((2, 3), (2, 2), {"sample_height": 9, "sample_width": 10, "blend_height": 4, "blend_width": 4})

    def tiles(tensor):
        return {
            (0, 0): tensor[0],
            (0, 1): tensor[1, ..., :3],
            (1, 0): tensor[2, ..., :2, :],
            (1, 1): tensor[3, ..., :2, :3],
        }

    calls = []

    def record_blend(source, current, extent, axis):
        used = try_blend(source, current, extent, axis)
        calls.append(used)
        return used

    monkeypatch.setattr(autoencoder, "try_blend", record_blend)
    actual = model.decode_tile_merge(tiles(packed), spec)
    assert calls == [True] * 4
    monkeypatch.setattr(autoencoder, "try_blend", lambda *args, **kwargs: False)
    expected = model.decode_tile_merge(tiles(expected_tiles), spec)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(packed.view(torch.int16), expected_tiles.view(torch.int16))
