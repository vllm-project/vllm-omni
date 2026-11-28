# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE, _VideoVAEPartProxy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _stub_pipeline(monkeypatch, events):
    class Stager:
        def __init__(self, name):
            self.name = name
            self.active = False

        def load(self):
            events.append(("load", self.name))
            self.active = True

        def offload(self):
            events.append(("offload", self.name))
            self.active = False

    class Audio(torch.nn.Module):
        def load_to_device(self):
            events.append(("load", "audio"))

        def offload_to_cpu(self):
            events.append(("offload", "audio"))

        def decode_latent(self, latent):
            return torch.zeros(1, 1, 8)

    class Session:
        def __init__(self, **kwargs):
            pass

        def start(self, **kwargs):
            pass

        def abort(self):
            events.append(("abort", "session"))

    vae = object.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(vae)
    vae._encoder_stager = Stager("encoder")
    vae._decoder_stager = Stager("decoder")
    vae._stager = None
    vae.decoder_component = _VideoVAEPartProxy(vae, "decoder")
    monkeypatch.setattr(MiniMaxH3VideoVAE, "is_distributed_enabled", lambda self: False)

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.video_vae = vae
    pipeline.audio_vae = Audio()
    pipeline.device = torch.device("cpu")
    pipeline.od_config = OmniDiffusionConfig(enable_distributed_layerwise_offload=True)
    monkeypatch.setattr(MiniMaxH3Pipeline, "_release_stage_cache", lambda self: events.append(("release", "stage")))
    return pipeline, vae, Session


@pytest.mark.parametrize("fail_decode", [False, True])
def test_streaming_decode_stages_only_decoder(monkeypatch, fail_decode):
    events = []
    pipeline, vae, Session = _stub_pipeline(monkeypatch, events)
    assert pipeline._uses_manual_component_offload(vae)

    def decode(video_vae, latent, **kwargs):
        assert video_vae is vae
        assert vae._decoder_stager.active
        if fail_decode:
            raise RuntimeError("decode failed")
        return [b"mp4"]

    monkeypatch.setattr("vllm_omni.diffusion.utils.chunked_video.ChunkedVideoMP4Session", Session)
    monkeypatch.setattr("vllm_omni.diffusion.utils.chunked_video.decode_to_mp4", decode)
    if fail_decode:
        with pytest.raises(RuntimeError, match="decode failed"):
            pipeline.decode_to_mp4(torch.zeros(1), torch.zeros(1), height=2, width=2)
        assert ("abort", "session") in events
    else:
        assert pipeline.decode_to_mp4(torch.zeros(1), torch.zeros(1), height=2, width=2) == b"mp4"
    assert ("load", "decoder") in events
    assert ("offload", "decoder") in events
    assert not vae._decoder_stager.active
    assert ("load", "encoder") not in events
    assert ("offload", "encoder") not in events


def test_streaming_decode_releases_stage_cache_after_the_chunks(monkeypatch):
    # The per-request peak counter resets but keeps the reserved pool, so the
    # pages this stage frees must go back to the allocator before the method
    # returns; otherwise the next request starts its peak on top of them.
    events = []
    pipeline, _vae, Session = _stub_pipeline(monkeypatch, events)

    def decode(video_vae, latent, **kwargs):
        events.append(("decoded", "chunks"))
        return [b"mp4"]

    monkeypatch.setattr("vllm_omni.diffusion.utils.chunked_video.ChunkedVideoMP4Session", Session)
    monkeypatch.setattr("vllm_omni.diffusion.utils.chunked_video.decode_to_mp4", decode)

    assert pipeline.decode_to_mp4(torch.zeros(1), torch.zeros(1), height=2, width=2) == b"mp4"
    order = [name for name, _ in events]
    assert "release" in order
    assert order.index("decoded") < order.index("release")
