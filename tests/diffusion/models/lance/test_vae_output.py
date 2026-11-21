# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.platforms import current_platform

from vllm_omni.diffusion.models.lance import wan_vae
from vllm_omni.diffusion.models.lance.vae_output import write_unpatchified

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.diffusion,
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required"),
]


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_chunk_writes_match_unpatchify_and_clamp(dtype):
    # Batch, odd dimensions, a strided source, and the 1+4 frame transition.
    chunks = [torch.randn(2, 12, t, 7, 22, device="cuda", dtype=dtype)[..., ::2] for t in (1, 4)]
    chunks[0][0, 0, 0, 0, :5] = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -2.0], device="cuda")
    expected = wan_vae._unpatchify(torch.cat(chunks, dim=2), patch_size=2).clamp_(-1, 1)
    storage = torch.full((expected.numel() + 32,), 17, device="cuda", dtype=dtype)
    actual = storage[16:-16].view_as(expected)
    write_unpatchified(chunks[0], actual, 0, clamp=True)
    assert torch.all(actual[:, :, 1:] == 17)
    write_unpatchified(chunks[1], actual, 1, clamp=True)
    int_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(int_dtype), expected.view(int_dtype))
    assert torch.all(storage[:16] == 17) and torch.all(storage[-16:] == 17)


@torch.inference_mode()
@pytest.mark.parametrize(("latent_frames", "clamp"), [(1, False), (3, True)])
def test_decode_writes_all_frames_and_clears_cache(monkeypatch, latent_frames, clamp):
    # Use the real decoder with small channels. No weights or model download is needed.
    model = (
        wan_vae.WanVAE_(
            dim=4, dec_dim=4, z_dim=2, dim_mult=(1, 1, 1), num_res_blocks=0, temperal_downsample=(True, True)
        )
        .to("cuda")
        .eval()
    )
    z = torch.randn(1, 2, latent_frames, 2, 3, device="cuda")
    calls = []

    def record_write(chunk, output, offset, *, clamp):
        calls.append((offset, chunk.shape[2]))
        write_unpatchified(chunk, output, offset, clamp=clamp)

    monkeypatch.setattr(wan_vae, "write_unpatchified", record_write)
    actual = model.decode(z, [0.0, 1.0], clamp_output=clamp)
    assert calls == ([(0, 1)] if latent_frames == 1 else [(0, 1), (1, 4), (5, 4)])
    assert all(item is None for item in model._feat_map)
    monkeypatch.setattr(wan_vae, "can_use_fused_output", lambda _: False)
    expected = model.decode(z, [0.0, 1.0], clamp_output=clamp)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_grad_enabled_decode_uses_native_output(monkeypatch):
    model = (
        wan_vae.WanVAE_(
            dim=4, dec_dim=4, z_dim=2, dim_mult=(1, 1, 1), num_res_blocks=0, temperal_downsample=(True, True)
        )
        .to("cuda")
        .eval()
    )
    z = torch.randn(1, 2, 1, 2, 3, device="cuda", requires_grad=True)

    def unexpected_write(*args, **kwargs):
        pytest.fail("The fused output path does not support autograd")

    monkeypatch.setattr(wan_vae, "write_unpatchified", unexpected_write)
    with torch.enable_grad():
        out = model.decode(z, [0.0, 1.0])
        out.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
