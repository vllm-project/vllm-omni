# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path))
    return OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)


@pytest.mark.parametrize("stereo", [False, True])
def test_finalized_waveform_owns_contiguous_numeric_storage(server, stereo):
    import gc

    source = np.linspace(-1, 1, 96000, dtype=np.float64).reshape(48000, 2)
    decoded = source if stereo else source[:, 0]
    expected = np.asarray(decoded, dtype=np.float32)
    if stereo:
        expected = expected.mean(axis=-1)
    waveform, sr, artifact, duration = server._finalize_fetched_ref_audio(decoded, 24000)
    assert waveform.dtype == np.float32
    assert waveform.flags.c_contiguous and waveform.flags.owndata
    assert not np.shares_memory(waveform, decoded)
    assert not gc.is_tracked(waveform)
    np.testing.assert_array_equal(waveform, expected)
    assert artifact == server._make_ref_audio_artifact_cache_key(expected, sr)
    assert duration == 2.0


@pytest.mark.asyncio
async def test_array_resolver_reuses_buffer_and_legacy_lists_are_temporary(server, mocker):
    server._init_speaker_storage()
    server._diffusion_mode = True
    server._allowed_local_media_path = ""
    server._ref_audio_cache_key = mocker.AsyncMock(return_value="locator")
    server._media_connector = mocker.Mock()
    server._media_connector.fetch_audio_async = mocker.AsyncMock(
        return_value=(np.linspace(-1, 1, 24000, dtype=np.float32), 24000)
    )
    first, sr, key = await server._resolve_ref_audio_array("reference")
    second, _, _ = await server._resolve_ref_audio_array("reference")
    legacy, _, _ = await server._resolve_ref_audio("reference")
    assert first is second
    assert isinstance(legacy, list)
    np.testing.assert_array_equal(first, legacy)
    legacy[0] = 100
    assert first[0] == -1
    assert server._ref_audio_resolve_cache[key][0] is first
    assert server._ref_audio_resolve_cache_bytes == first.nbytes == 96000
    server._media_connector.fetch_audio_async.assert_awaited_once()


@pytest.mark.parametrize("variant", [None, "local", "delay", "realtime"])
@pytest.mark.asyncio
async def test_moss_array_resolution_keeps_nano_list_transport(mocker, variant):
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.moss_tts import _MossTTSAdapterBase

    server = mocker.Mock()
    server._resolve_ref_audio = mocker.AsyncMock(return_value=([0.0], 24000, "key"))
    waveform = np.zeros(1, dtype=np.float32)
    server._resolve_ref_audio_array = mocker.AsyncMock(return_value=(waveform, 24000, "key"))
    adapter = _MossTTSAdapterBase.__new__(_MossTTSAdapterBase)
    adapter.ctx = SpeechServingContext(server=server)
    adapter._moss_variant = variant
    result, _, _ = await adapter._resolve_ref_audio("reference")
    if variant is None:
        assert isinstance(result, list)
        server._resolve_ref_audio_array.assert_not_awaited()
    else:
        assert result is waveform
        server._resolve_ref_audio.assert_not_awaited()


def test_waveform_cache_eviction_releases_owned_buffer(server):
    import weakref

    server._init_speaker_storage()
    server._ref_audio_resolve_cache_max_entries = 1
    waveform = np.zeros(24000, dtype=np.float32)
    ref = weakref.ref(waveform)
    server._put_resolved_ref_audio("first", waveform, 24000, "artifact-first")
    del waveform
    assert ref() is not None
    server._put_resolved_ref_audio("second", np.zeros(24000, dtype=np.float32), 24000, "artifact-second")
    assert ref() is None
    assert server._ref_audio_resolve_cache_bytes == 96000


@pytest.mark.parametrize("entries, budget", [(1, 1000), (10, 8)])
def test_numeric_waveform_cache_limits(server, entries, budget):
    server._init_speaker_storage()
    server._ref_audio_resolve_cache_max_entries = entries
    server._ref_audio_resolve_cache_max_bytes = budget
    server._put_resolved_ref_audio("a", np.zeros(2, dtype=np.float32), 24000, "artifact-a")
    server._put_resolved_ref_audio("b", np.ones(2, dtype=np.float32), 24000, "artifact-b")
    assert list(server._ref_audio_resolve_cache) == ["b"]
    assert server._ref_audio_resolve_cache_bytes == 8
