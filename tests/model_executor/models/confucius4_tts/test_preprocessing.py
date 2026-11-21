# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
import torch

from vllm_omni.model_executor.models.confucius4_tts.preprocessing import (
    _wav2vec_features,
    extract_reference_features,
    normalize_wav2vec_stats,
)
from vllm_omni.model_executor.models.indextts2 import preprocess_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_normalize_wav2vec_stats_accepts_official_dict() -> None:
    stats = normalize_wav2vec_stats(
        {
            "mean": torch.tensor([[1.0, 2.0, 3.0]]),
            "var": torch.tensor([4.0, 9.0, 16.0]),
        }
    )

    assert stats.shape == (2, 3)
    assert stats.dtype == torch.float32
    assert stats.tolist() == [[1.0, 2.0, 3.0], [4.0, 9.0, 16.0]]


def test_normalize_wav2vec_stats_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="mean/var shape mismatch"):
        normalize_wav2vec_stats(
            {
                "mean": torch.ones(3),
                "var": torch.ones(4),
            }
        )
    with pytest.raises(ValueError, match="non-negative"):
        normalize_wav2vec_stats(
            {
                "mean": torch.ones(3),
                "var": torch.tensor([1.0, -1.0, 1.0]),
            }
        )


def test_wav2vec_features_use_layer_17_float32_and_normalize() -> None:
    hidden = [torch.full((1, 3, 4), float(i)) for i in range(20)]
    seen: dict[str, object] = {}

    class FakeProcessor:
        def __call__(self, wav, *, sampling_rate, return_tensors):
            seen["rate"] = sampling_rate
            seen["wav_len"] = len(wav)
            return {"input_features": torch.zeros(1, 5, 6, dtype=torch.float64)}

    class FakeModel:
        def __call__(self, *, input_features, attention_mask, output_hidden_states):
            seen["dtype"] = input_features.dtype
            seen["hidden_states_requested"] = output_hidden_states
            seen["autocast"] = torch.is_autocast_enabled()
            return type("Out", (), {"hidden_states": hidden})()

    stats = torch.stack([torch.full((4,), 5.0), torch.full((4,), 4.0)])  # mean 5, var 4 -> std 2

    features = _wav2vec_features(torch.zeros(160), FakeModel(), FakeProcessor(), torch.device("cpu"), stats)

    assert seen == {
        "rate": 16000,
        "wav_len": 160,
        "dtype": torch.float32,
        "hidden_states_requested": True,
        "autocast": False,
    }
    # layer 17 holds 17.0; (17 - 5) / 2 = 6.0
    assert torch.equal(features, torch.full((1, 3, 4), 6.0))


def test_extract_reference_features_truncates_and_resamples_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The canonical front-end trims the reference and derives the 16 kHz and 22.05 kHz signals
    # straight from it; chaining 22.05k -> 16k adds resampling error, so assert both come from source.
    source = torch.arange(48000, dtype=torch.float32)  # 3 s at 16 kHz
    resampled: list[tuple[int, int]] = []

    def fake_resample(wav, source_rate, target_rate):
        resampled.append((source_rate, target_rate))
        return torch.zeros(10)

    class FakeCAMPPlus:
        def __call__(self, _fbank):
            return torch.ones(1, 192)

    for name, value in (
        ("_load_audio_1d", lambda _audio: (source, 16000)),
        ("_resample", fake_resample),
        ("_wav2vec_features", lambda wav, *_args: torch.ones(1, 2, 1024)),
        ("load_wav2vec2", lambda *_args: (object(), object())),
        ("load_campplus", lambda *_args, **_kwargs: FakeCAMPPlus()),
        ("compute_fbank", lambda *_args: torch.ones(1, 2, 80)),
        ("mel_spectrogram", lambda *_args: torch.ones(1, 80, 2)),
    ):
        monkeypatch.setattr(f"vllm_omni.model_executor.models.confucius4_tts.preprocessing.{name}", value)

    features = extract_reference_features(
        "reference.wav",
        model_path="model",
        device=torch.device("cpu"),
        w2v_stats=torch.ones(2, 1024),
        max_audio_length_seconds=1.0,
    )

    assert resampled == [(16000, 16000), (16000, 22050)]
    assert set(features) == {"condition_vector", "style", "ref_mel"}
    assert features["condition_vector"].shape == (1, 2, 1024)


def test_extract_reference_features_trims_before_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []
    source = torch.arange(16000 * 10, dtype=torch.float32)  # 10 s at 16 kHz

    def fake_resample(wav, _source_rate, _target_rate):
        seen.append(int(wav.numel()))
        return torch.zeros(4)

    for name, value in (
        ("_load_audio_1d", lambda _audio: (source, 16000)),
        ("_resample", fake_resample),
        ("_wav2vec_features", lambda *_a: torch.ones(1, 1, 4)),
        ("load_wav2vec2", lambda *_a: (object(), object())),
        ("load_campplus", lambda *_a, **_k: (lambda _f: torch.ones(1, 192))),
        ("compute_fbank", lambda *_a: torch.ones(1, 1, 80)),
        ("mel_spectrogram", lambda *_a: torch.ones(1, 80, 1)),
    ):
        monkeypatch.setattr(f"vllm_omni.model_executor.models.confucius4_tts.preprocessing.{name}", value)

    extract_reference_features(
        "reference.wav",
        model_path="model",
        device=torch.device("cpu"),
        w2v_stats=torch.ones(2, 4),
        max_audio_length_seconds=30.0,
    )

    # 30 s cap applied at 16 kHz on the source signal.
    assert seen == [16000 * 10, 16000 * 10]


def test_load_wav2vec2_supports_official_pretrained_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    wav2vec_dir = tmp_path / "pretrained" / "w2v-bert-2.0"
    wav2vec_dir.mkdir(parents=True)
    calls: list[tuple[str, dict]] = []

    class FakeFeatureExtractor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((str(path), kwargs))
            return cls()

    class FakeWav2Vec2(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((str(path), kwargs))
            return cls()

    monkeypatch.setattr(
        transformers,
        "AutoFeatureExtractor",
        FakeFeatureExtractor,
    )
    monkeypatch.setattr(
        transformers,
        "Wav2Vec2BertModel",
        FakeWav2Vec2,
    )
    preprocess_utils._wav2vec2_cache.clear()

    model, processor = preprocess_utils.load_wav2vec2(
        str(tmp_path),
        torch.device("cpu"),
    )

    assert isinstance(model, FakeWav2Vec2)
    assert isinstance(processor, FakeFeatureExtractor)
    assert calls == [
        (str(wav2vec_dir), {"local_files_only": True}),
        (str(wav2vec_dir), {"local_files_only": True}),
    ]


def test_load_campplus_supports_official_pretrained_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.model_executor.models.indextts2.utils.campplus import dtdnn

    checkpoint = tmp_path / "pretrained" / "campplus" / "campplus_cn_common.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    loaded: dict[str, object] = {}

    class FakeCAMPPlus(torch.nn.Module):
        def load_state_dict(self, state, strict=True):
            loaded["state"] = state
            loaded["strict"] = strict

    monkeypatch.setattr(
        dtdnn,
        "CAMPPlus",
        lambda **_kwargs: FakeCAMPPlus(),
    )
    monkeypatch.setattr(
        preprocess_utils.torch,
        "load",
        lambda path, **_kwargs: (
            loaded.setdefault("path", path),
            {"weight": torch.ones(1)},
        )[1],
    )
    preprocess_utils._campplus_cache.clear()

    model = preprocess_utils.load_campplus(
        str(tmp_path),
        torch.device("cpu"),
    )

    assert isinstance(model, FakeCAMPPlus)
    assert loaded["path"] == str(checkpoint)
    assert loaded["strict"] is False


def test_load_campplus_can_complete_local_confucius_bundle_from_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.indextts2.utils.campplus import dtdnn
    from vllm_omni.transformers_utils import repo_utils

    checkpoint = tmp_path / "downloaded-campplus.bin"
    checkpoint.touch()
    downloads: list[tuple[str, str]] = []

    class FakeCAMPPlus(torch.nn.Module):
        def load_state_dict(self, state, strict=True):
            self.loaded_state = (state, strict)

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        downloads.append((repo_id, filename))
        return str(checkpoint)

    monkeypatch.setattr(
        dtdnn,
        "CAMPPlus",
        lambda **_kwargs: FakeCAMPPlus(),
    )
    monkeypatch.setattr(
        repo_utils,
        "hf_api",
        lambda: SimpleNamespace(
            hf_hub_download=fake_hf_hub_download,
        ),
    )
    monkeypatch.setattr(
        preprocess_utils.torch,
        "load",
        lambda *_args, **_kwargs: {"weight": torch.ones(1)},
    )
    preprocess_utils._campplus_cache.clear()

    model = preprocess_utils.load_campplus(
        str(tmp_path),
        torch.device("cpu"),
        allow_remote_fallback=True,
    )

    assert isinstance(model, FakeCAMPPlus)
    assert downloads == [("funasr/campplus", "campplus_cn_common.bin")]
