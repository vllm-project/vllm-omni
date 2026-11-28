# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch
from transformers import MimiConfig, MimiModel

from vllm_omni.diffusion.models.kyutai_aura.audio import MimiSemanticFeatures

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("tail_samples", [0, 1, 3, 8, 15])
def test_mimi_stream_matches_frame_aligned_full_encode(monkeypatch, tail_samples):
    config = MimiConfig(
        hidden_size=16,
        num_filters=4,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        upsampling_ratios=[4, 2],
        upsample_groups=16,
        codebook_size=16,
        codebook_dim=8,
        vector_quantization_hidden_dimension=8,
        num_quantizers=2,
        sliding_window=5,
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = MimiModel(config).eval()
        for module in model.quantizer.modules():
            if (embed_sum := getattr(module, "embed_sum", None)) is not None:
                embed_sum.normal_(std=0.01)
    monkeypatch.setattr(MimiModel, "from_pretrained", lambda *args, **kwargs: model)
    features = MimiSemanticFeatures("unused", torch.device("cpu"))
    frame_size = config.frame_size
    pcm = np.random.default_rng(42).uniform(-1, 1, 17 * frame_size + tail_samples).astype(np.float32)
    padded = np.pad(pcm, (0, (-pcm.size) % frame_size))

    with torch.no_grad():
        codes = model.encode(torch.from_numpy(padded)[None, None], num_quantizers=1).audio_codes
        expected = model.quantizer.decode(codes).transpose(1, 2).float()
    assert codes.unique().numel() > 1

    stream = features.start_stream()
    assert stream.push(pcm[:3]) is None
    outputs = [
        stream.push(pcm[3 : 3 * frame_size + 1]),
        stream.push(pcm[3 * frame_size + 1 : 12 * frame_size + 5]),
        stream.push(pcm[12 * frame_size + 5 :]),
        stream.push(pcm[:0], final=True),
    ]
    actual = torch.cat([output for output in outputs if output is not None], dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    # The offline AURA path gives Mimi the original PCM and lets its
    # convolutions manage the tail padding themselves.
    with torch.no_grad():
        full_codes = model.encode(torch.from_numpy(pcm)[None, None], num_quantizers=1).audio_codes
        full_features = model.quantizer.decode(full_codes).transpose(1, 2).float()
    torch.testing.assert_close(actual, full_features, rtol=1e-6, atol=1e-7)
    with pytest.raises(RuntimeError, match="already finished"):
        stream.push(pcm[:0])


def test_live_video_session_trims_tail_without_dropping_audio(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.kyutai_aura.config import KYUTAI_AURA_1_3B
    from vllm_omni.diffusion.models.kyutai_aura.live_audio import AuraLiveAudio

    class FakeMimiStream:
        samples = 0

        def push(self, pcm, *, final):
            previous = self.samples // 1920
            self.samples += pcm.size
            frames = (self.samples + (1919 if final else 0)) // 1920 - previous
            return torch.ones(1, frames, 512)

    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: torch.no_grad())
    conditions = []

    def encode_condition(condition):
        conditions.append(condition)
        return None

    pipeline = SimpleNamespace(
        audio_features=SimpleNamespace(start_stream=FakeMimiStream, sample_rate=24000),
        load_portrait=lambda image: image,
        encode_reference=lambda _image: torch.zeros(1, 16, 1, 4, 4),
        encode_prompt=lambda _prompt: torch.zeros(1, 1),
        cfg=KYUTAI_AURA_1_3B,
        device=torch.device("cpu"),
        vae=SimpleNamespace(_cached_conv_counts={"decoder": 1}),
        transformer=SimpleNamespace(encode_audio=encode_condition),
        _sample_chunk=lambda *_args: None,
        chunk_generator=lambda *_args: None,
        decode_chunk=lambda _value, _cache, first: torch.zeros(3, 13 if first else 16, 4, 4),
    )
    session = AuraLiveAudio(pipeline, object(), "Speak.", 42)
    pcm = np.arange(48000, dtype="<i2")
    assert session.push(pcm[:24000].tobytes(), final=False) == []
    outputs = session.push(pcm[24000:].tobytes(), final=False)
    assert [chunk[0].shape[2] for chunk in outputs] == [13, 16]
    outputs += session.push(b"", final=True)
    assert [chunk[0].shape[2] for chunk in outputs] == [13, 16, 3]
    np.testing.assert_array_equal(np.concatenate([chunk[1] for chunk in outputs]), pcm.astype(np.float32) / 32768)
    assert all(condition.shape == (1, 1, 512, 16) for condition in conditions)
    assert not conditions[-1].any()
