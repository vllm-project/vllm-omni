# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Video serving resolves audio rates via the shared reader, with no defaults."""

from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.sample_rate import MissingSampleRateError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


class StubVideoEngine:
    """Minimal AsyncOmni stand-in (the FakeAsyncOmni shape)."""

    def __init__(self, results, model_config=None):
        self.stage_configs = [
            SimpleNamespace(
                stage_type="diffusion",
                final_output=True,
                final_output_type="video",
            )
        ]
        self.default_sampling_params_list = [OmniDiffusionSamplingParams()]
        self.model_class_name = "WanPipeline"
        if model_config is not None:
            self.model_config = model_config
        self._results = results

    def get_diffusion_od_config(self):
        return SimpleNamespace(model_class_name=self.model_class_name, controlnet_model_path=None)

    async def generate(self, prompt, request_id, sampling_params_list):
        for result in self._results:
            yield result

    async def abort(self, request_id, *, timeout=None):
        del request_id, timeout


class StubVideoResult:
    def __init__(self, videos, audios=None, multimodal_output=None):
        self.multimodal_output = dict(multimodal_output or {"video": videos})
        if audios is not None:
            self.multimodal_output["audio"] = audios
        self.stage_durations = {}
        self.peak_memory_mb = 0.0
        self.custom_output = {}
        self.images = None


def _build_handler(engine) -> OmniOpenAIServingVideo:
    return OmniOpenAIServingVideo.for_diffusion(engine, model_name="test-model")


def test_result_rate_resolves_via_shared_reader() -> None:
    handler = _build_handler(StubVideoEngine([]))
    result = StubVideoResult([object()], multimodal_output={"video": [object()], "sr": UNUSUAL_RATE})
    assert handler._resolve_audio_sample_rate(result) == UNUSUAL_RATE


def test_canonical_sr_wins_over_legacy_spelling() -> None:
    # The old hand-rolled chain preferred the legacy audio_sample_rate key;
    # the shared reader resolves the canonical key first.
    handler = _build_handler(StubVideoEngine([]))
    result = StubVideoResult(
        [object()],
        multimodal_output={"video": [object()], "sr": UNUSUAL_RATE, "audio_sample_rate": 48000},
    )
    assert handler._resolve_audio_sample_rate(result) == UNUSUAL_RATE


def test_dict_shaped_result_resolves() -> None:
    # _extract_multimodal_output normalizes dict results too.
    handler = _build_handler(StubVideoEngine([]))
    assert handler._resolve_audio_sample_rate({"multimodal_output": {"sr": 16000}}) == 16000


def test_config_chain_resolves_wan_vocoder_spelling() -> None:
    # output_sampling_rate (Wan vocoder) is not in the shared key set and
    # keeps an explicit probe ahead of find_sample_rate.
    handler = _build_handler(
        StubVideoEngine(
            [],
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=16000)))
            ),
        )
    )
    assert handler._resolve_audio_sample_rate(StubVideoResult([object()])) == 16000


def test_config_chain_resolves_shared_spellings() -> None:
    handler = _build_handler(
        StubVideoEngine(
            [],
            model_config=SimpleNamespace(hf_config=SimpleNamespace(audio_vae=SimpleNamespace(sample_rate=44100))),
        )
    )
    assert handler._resolve_audio_sample_rate(StubVideoResult([object()])) == 44100


@pytest.mark.asyncio
async def test_audio_without_rate_fails_loudly() -> None:
    # Post-normalization every audio-emitting model advertises its rate;
    # audio without one is a producer bug that must surface, not fall back
    # to 24000 and mislabel the encoded response.
    engine = StubVideoEngine([StubVideoResult([object()], audios=[object()])])
    handler = _build_handler(engine)
    with pytest.raises(MissingSampleRateError):
        await handler._run_and_extract(VideoGenerationRequest(prompt="video"), "req-no-rate")


@pytest.mark.asyncio
async def test_video_only_result_reports_none_rate() -> None:
    # Video-only results carry no audio; the artifacts report None instead
    # of the old fabricated 24000 (consumers only read the rate when audio
    # is present).
    engine = StubVideoEngine([StubVideoResult([object()])])
    handler = _build_handler(engine)
    artifacts = await handler._run_and_extract(VideoGenerationRequest(prompt="video"), "req-video-only")
    assert artifacts.audios == [None]
    assert artifacts.audio_sample_rate is None


@pytest.mark.asyncio
async def test_payload_rate_reaches_artifacts() -> None:
    result = StubVideoResult(
        [object()],
        multimodal_output={"video": [object()], "audio": [object()], "sr": UNUSUAL_RATE},
    )
    handler = _build_handler(StubVideoEngine([result]))
    artifacts = await handler._run_and_extract(VideoGenerationRequest(prompt="video"), "req-rate")
    assert artifacts.audio_sample_rate == UNUSUAL_RATE
