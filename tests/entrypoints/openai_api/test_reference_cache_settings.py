# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest

from vllm_omni.config.speech_cache import SpeechCacheConfig
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def server(monkeypatch, tmp_path):
    import vllm_omni.utils.speaker_cache as module

    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path))
    monkeypatch.setattr(module, "_SINGLETON", None)
    return OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)


def test_defaults(server):
    server._init_speaker_storage()
    assert server._ref_audio_resolve_cache_max_bytes == 4 * 1024**3
    assert server._ref_audio_resolve_cache_max_entries == 2048
    assert server._speaker_cache.stats()["max_bytes"] == 512 * 1024**2


@pytest.mark.parametrize("entries,budget", [(1, 1000), (10, 8)])
def test_reference_cache_enforces_configured_limits(server, entries, budget):
    server.speech_cache_config = SpeechCacheConfig(
        resolve_max_entries=entries, resolve_max_bytes=budget, speaker_max_bytes=16
    )
    server._init_speaker_storage()
    server._put_resolved_ref_audio("a", np.asarray([0.0, 0.1], dtype=np.float32), 24000, "artifact-a")
    server._put_resolved_ref_audio("b", np.asarray([0.2, 0.3], dtype=np.float32), 24000, "artifact-b")
    assert list(server._ref_audio_resolve_cache) == ["b"]
    assert server._ref_audio_resolve_cache_bytes == 8
    assert server._speaker_cache.stats()["max_bytes"] == 16


@pytest.mark.parametrize("field", ["resolve_max_entries", "resolve_max_bytes"])
def test_zero_disables_storage(server, field):
    server.speech_cache_config = SpeechCacheConfig(**{field: 0})
    server._init_speaker_storage()
    server._put_resolved_ref_audio("a", np.asarray([0.0], dtype=np.float32), 24000, "artifact-a")
    assert not server._ref_audio_resolve_cache
    assert server._ref_audio_resolve_cache_bytes == 0


def test_diffusion_receives_config(server):
    config = SpeechCacheConfig(resolve_max_bytes=80, resolve_max_entries=1, speaker_max_bytes=0)
    speech = OmniOpenAIServingSpeech.for_diffusion(object(), "test", speech_cache_config=config)
    assert speech.speech_cache_config is config
    assert speech._ref_audio_resolve_cache_max_bytes == 80
    assert speech._speaker_cache.stats()["max_bytes"] == 0
