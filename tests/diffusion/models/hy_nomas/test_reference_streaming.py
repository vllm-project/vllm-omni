# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

pytestmark = [pytest.mark.local_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("lengths,pixels", [([13, 13], [49, 49]), ([1, 3, 16], [1, 12, 61]), ([5], [17])])
def test_reference_pixelumm_stable_prefix_and_final_tail(lengths, pixels):
    reference = pytest.importorskip("hymm.models.vae.pixelumm_decoder.autoencoder")

    def predict(latent, noise, *, t_starts, **kwargs):
        # Only the learned decoder is substituted. The reference streaming
        # session performs its actual window scheduling and pixel bookkeeping.
        start, end = t_starts[0] * 4, min(t_starts[-1] + 10, latent.shape[2]) * 4
        return torch.arange(start, end, dtype=torch.float32).reshape(1, 1, -1, 1, 1)

    model = SimpleNamespace(
        temporal_window_latents=10,
        temporal_overlap_latents=5,
        time_compression_ratio=4,
        _distributed_decode_mode="none",
        tail_hold_latents=1,
        _predict_x0=predict,
        _noise_tile=torch.zeros_like,
    )
    session = reference.PixelUMMStreamingDecodeSession(model, to_cpu=True, retain_history=False)
    chunks, snapshots, expected = [], [], []
    offset = 0
    for length, pixel_count in zip(lengths, pixels):
        session.final_segment = False
        chunk = session.push(torch.zeros(1, 1, length, 1, 1), pixel_frames=pixel_count)
        if chunk.numel():
            chunks.append(chunk)
            snapshots.append(chunk.clone())
        expected.append(torch.arange(offset, offset + pixel_count, dtype=torch.float32))
        offset += length * 4
    session.final_segment = True
    tail = session.finish()
    if tail.numel():
        chunks.append(tail)
    for original, saved in zip(chunks, snapshots):
        torch.testing.assert_close(original, saved, rtol=0, atol=0)
    # Float32 cosine blending can round an otherwise identical overlap by one relative rounding unit.
    torch.testing.assert_close(
        torch.cat(chunks, 2).flatten(), torch.cat(expected), rtol=torch.finfo(torch.float32).eps, atol=0
    )
    assert session.finished and session.stable_pixel_frames == sum(pixels)
    assert not session.committed


def test_reference_component_sdpa_policy_is_scoped_after_failure():
    from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline

    runtime = HyNomasPipeline.__new__(HyNomasPipeline)
    nn.Module.__init__(runtime)
    runtime._cpu_offload = False
    runtime.initialize_ar({})
    with sdpa_kernel([SDPBackend.MATH]):
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        with pytest.raises(RuntimeError, match="component failed"), runtime._component_context(nn.Linear(2, 2)):
            assert torch.backends.cuda.cudnn_sdp_enabled()
            raise RuntimeError("component failed")
        assert not torch.backends.cuda.cudnn_sdp_enabled()
