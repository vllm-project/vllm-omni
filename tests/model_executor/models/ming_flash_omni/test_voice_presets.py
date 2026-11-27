# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for Ming voice-preset registration."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.model_executor.models.ming_flash_omni.voice_presets import VoicePresetRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_register_accepts_inference_mode_speaker_embedding(tmp_path: Path, mocker: MockerFixture) -> None:
    registry = VoicePresetRegistry(
        talker_dir=str(tmp_path),
        model_path=str(tmp_path),
        download_dir=None,
        audio_vae=None,
        aggregator=mocker.Mock(),
        spk_head=torch.nn.Linear(2, 4),
        patch_size=1,
    )
    with torch.inference_mode():
        raw_embedding = torch.ones(1, 2)
    mocker.patch.object(
        registry,
        "_load_single_wav",
        return_value=(torch.zeros(1, 4), raw_embedding),
    )

    registry.register("DB30", "prompt.wav", device=torch.device("cpu"), dtype=torch.float32)

    [projected] = registry.registered["DB30"]["spk_emb"]
    assert projected.shape == (1, 4)
    assert not projected.requires_grad


@pytest.mark.parametrize("strict", [False, True], ids=["ming2", "converted-ming3"])
def test_weight_loading_registers_only_legacy_voice_presets(strict: bool, mocker: MockerFixture) -> None:
    from vllm_omni.model_executor.models.ming_flash_omni import ming_flash_omni_talker as talker_module

    talker = object.__new__(talker_module.MingFlashOmniTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.model = torch.nn.Linear(2, 2)
    talker.audio_vae = torch.nn.Identity()
    talker.config = SimpleNamespace(strict_weight_loading=strict)
    talker._standalone = False
    talker._vae_weight_source = None
    talker.voice_presets = mocker.Mock()
    talker._batch_rows = 2
    prepare = mocker.patch.object(talker, "_ensure_batch_engine")
    mocker.patch.object(type(talker), "device", new_callable=mocker.PropertyMock, return_value=torch.device("cuda"))
    loaded = set(dict(talker.named_parameters()))
    mocker.patch.object(talker_module, "AutoWeightsLoader").return_value.load_weights.return_value = loaded

    assert talker.load_weights([]) == loaded
    assert talker.training
    prepare.assert_not_called()

    if strict:
        talker.voice_presets.load_presets_from_manifest.assert_not_called()
    else:
        talker.voice_presets.load_presets_from_manifest.assert_called_once_with(
            device=talker.device, dtype=talker.dtype
        )


def test_inline_reference_audio_reuses_features_by_content(tmp_path: Path, mocker: MockerFixture) -> None:
    registry = VoicePresetRegistry(
        talker_dir=str(tmp_path),
        model_path=str(tmp_path),
        download_dir=None,
        audio_vae=None,
        aggregator=mocker.Mock(),
        spk_head=torch.nn.Linear(2, 4),
        patch_size=1,
    )
    extractor = mocker.Mock(return_value=torch.ones(1, 2))
    registry.__dict__["_spkemb_extractor"] = extractor
    samples = [0.1 * index for index in range(1, 1601)]

    def resolve(audio):
        return registry.resolve_reference(
            None,
            device=torch.device("cpu"),
            dtype=torch.float32,
            need_speaker=True,
            need_latents=False,
            reference_audio=[{"samples": audio, "sample_rate": 16000}],
        )

    first = resolve(samples)
    # Each streamed text segment resends the same voice samples.
    assert resolve(list(samples)) is first
    assert extractor.call_count == 1

    changed = resolve([*samples[:-1], 0.0])
    assert changed is not first
    assert extractor.call_count == 2


def test_reference_cache_evicts_least_recently_used_voice(tmp_path: Path, mocker: MockerFixture) -> None:
    registry = VoicePresetRegistry(
        talker_dir=str(tmp_path),
        model_path=str(tmp_path),
        download_dir=None,
        audio_vae=None,
        aggregator=mocker.Mock(),
        spk_head=torch.nn.Linear(2, 4),
        patch_size=1,
    )
    registry.REFERENCE_CACHE_SIZE = 2
    extractor = mocker.Mock(return_value=torch.ones(1, 2))
    registry.__dict__["_spkemb_extractor"] = extractor

    def resolve(value):
        return registry.resolve_reference(
            None,
            device=torch.device("cpu"),
            dtype=torch.float32,
            need_speaker=True,
            need_latents=False,
            reference_audio=[{"samples": [value] * 1600, "sample_rate": 16000}],
        )

    first = resolve(0.1)
    resolve(0.2)
    assert resolve(0.1) is first  # refreshes 0.1, so 0.2 is now the oldest
    resolve(0.3)
    assert extractor.call_count == 3
    assert resolve(0.1) is first
    resolve(0.2)
    assert extractor.call_count == 4
