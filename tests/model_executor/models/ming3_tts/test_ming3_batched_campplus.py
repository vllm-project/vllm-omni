# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from vllm_omni.model_executor.models.indextts2.utils.campplus.dtdnn import CAMPPlus
from vllm_omni.model_executor.models.ming3_tts.batched_campplus import BatchedCampplus
from vllm_omni.model_executor.models.ming3_tts.constants import (
    KEY_SPEAKER_SAMPLE_RATES,
    KEY_SPEAKER_WAVEFORM,
    KEY_SPEAKER_WAVEFORM_LENGTHS,
)
from vllm_omni.model_executor.models.ming3_tts.ming3_tts import Ming3TTSForConditionalGeneration
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.tts, pytest.mark.cpu]


def _campplus() -> CAMPPlus:
    torch.manual_seed(0)
    model = CAMPPlus(feat_dim=80, embedding_size=192).eval()
    # Non-trivial batch-norm statistics, so padded positions would leak if unmasked.
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.running_mean.uniform_(-0.5, 0.5)
            module.running_var.uniform_(0.5, 1.5)
    return model


def _reference_embedding(model: CAMPPlus, waveform: torch.Tensor, rate: int) -> torch.Tensor:
    if rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)(waveform)
    feat = kaldi.fbank(waveform, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    return model(feat.unsqueeze(0))[0]


@torch.inference_mode()
def test_batched_campplus_matches_per_clip_pipeline():
    model = _campplus()
    encoder = BatchedCampplus(model, torch.device("cpu"))
    # Covers resampling, fbank and masked CAM++, including partial 100-frame
    # segments. Two 24 kHz clips resample together; the others use their own paths.
    clips = [
        (torch.randn(1, 24000 * 2) * 0.1, 24000),
        (torch.randn(1, 16000 * 3) * 0.1, 16000),
        (torch.randn(1, 24000 * 3 + 517) * 0.1, 24000),
        (torch.randn(1, 44100 * 2) * 0.1, 44100),
    ]
    embeddings = encoder.embed(clips)
    for (waveform, rate), embedding in zip(clips, embeddings, strict=True):
        torch.testing.assert_close(embedding, _reference_embedding(model, waveform, rate), rtol=1e-4, atol=1e-4)


class _Encoder:
    def __init__(self):
        self.calls = []

    def embed(self, clips):
        self.calls.append([waveform.shape[-1] for waveform, _ in clips])
        return [torch.full((192,), float(waveform.shape[-1])) for waveform, _ in clips]


def _info(request_id, samples, key=None, **extra):
    info = {
        "request_id": request_id,
        KEY_SPEAKER_WAVEFORM: torch.zeros(1, samples),
        KEY_SPEAKER_WAVEFORM_LENGTHS: torch.tensor([samples]),
        KEY_SPEAKER_SAMPLE_RATES: torch.tensor([24000]),
        **extra,
    }
    if key is not None:
        info["speaker_audio_keys"] = [key]
    return info


def test_step_prefetch_embeds_new_references_once():
    wrapper = SimpleNamespace(
        _speaker_encoder=_Encoder(),
        _speaker_cache=SpeakerEmbeddingCache(),
        _prefetched_speakers={},
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(model="/model")),
        ming_config=SimpleNamespace(sample_rate=24000),
    )
    prefetch = Ming3TTSForConditionalGeneration._prefetch_speaker_embeddings
    infos = [
        _info("a", 100, key="voice"),
        _info("b", 100, key="voice"),
        _info("c", 300),
        _info("d", 500, mute_speaker=torch.tensor([True])),
    ]
    prefetch(wrapper, infos)
    # Requests sharing a reference clip embed it once; a muted request embeds nothing.
    assert wrapper._speaker_encoder.calls == [[100, 300]]
    assert sorted(wrapper._prefetched_speakers) == ["a", "b", "c"]
    assert wrapper._prefetched_speakers["b"][0] is wrapper._prefetched_speakers["a"][0]
    assert torch.equal(wrapper._prefetched_speakers["c"][0], torch.full((192,), 300.0))

    # A later step reuses the cached embedding of a keyed reference.
    wrapper._prefetched_speakers.clear()
    prefetch(wrapper, [_info("e", 100, key="voice")])
    assert wrapper._speaker_encoder.calls == [[100, 300]]
    assert torch.equal(wrapper._prefetched_speakers["e"][0], torch.full((192,), 100.0))
