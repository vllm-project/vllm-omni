# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""UTs for VoxCPM2 OpenAI speech serving behavior (`instructions` + `cfg_value`)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]


@pytest.fixture
def voxcpm2_server(mocker: MockerFixture):
    """Mock an OmniOpenAIServingSpeech instance detected as `voxcpm2`.

    Single `latent_generator` stage with `model_arch=VoxCPM2...` so
    `_detect_tts_model_type` returns ``voxcpm2``.
    """
    mocker.patch.object(OmniOpenAIServingSpeech, "_load_supported_speakers", return_value=set())
    mocker.patch.object(OmniOpenAIServingSpeech, "_load_codec_frame_rate", return_value=None)

    mock_engine_client = mocker.MagicMock()
    mock_engine_client.errored = False
    mock_engine_client.model_config = mocker.MagicMock(
        model="openbmb/VoxCPM2",
        hf_config=SimpleNamespace(
            audio_vae_config={"sample_rate": 16000, "encoder_rates": [2, 5, 8, 8]},
            patch_size=4,
        ),
    )
    mock_engine_client.default_sampling_params_list = [SimpleNamespace(max_tokens=4096)]
    mock_engine_client.tts_batch_max_items = 32
    # Explicit None so `_compute_max_instructions_length` skips the CLI-override
    # branch (MagicMock attributes are truthy by default) and falls through to
    # the `_TTS_MAX_INSTRUCTIONS_LENGTH = 500` default.
    mock_engine_client.tts_max_instructions_length = None
    mock_engine_client.generate = mocker.MagicMock(return_value="generator")
    mock_engine_client.stage_configs = [
        SimpleNamespace(
            engine_args=SimpleNamespace(
                model_stage="latent_generator",
                model_arch="VoxCPM2TalkerForConditionalGeneration",
            ),
            tts_args={},
        ),
    ]

    mock_models = mocker.MagicMock()
    mock_models.is_base_model.return_value = True

    server = OmniOpenAIServingSpeech(
        engine_client=mock_engine_client,
        models=mock_models,
        request_logger=mocker.MagicMock(),
    )
    # Lazy tokenizer path is not exercised by these tests; provide a no-op encoder
    # and sentinel tokenizer / split_map so `_build_voxcpm2_prompt` can run.
    server._voxcpm2_tokenizer = mocker.MagicMock(name="voxcpm2_tokenizer")
    server._voxcpm2_split_map = {}
    mocker.patch.object(server, "_voxcpm2_encode", return_value=[])
    return server


@pytest.fixture
def mock_build_voxcpm2_prompt(mocker: MockerFixture):
    """Patch `build_voxcpm2_prompt` so we can inspect its call args without
    touching the real VoxCPM2 tokenizer / audio VAE.

    Returns the prompt dict shape the real helper would produce; the test
    can then also assert `cfg_value` stashing into `additional_information`.
    """
    return mocker.patch(
        "vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker.build_voxcpm2_prompt",
        return_value={"prompt_token_ids": [1], "additional_information": {"text_token_ids": [[]]}},
    )


class TestVoxCPM2Serving:
    def test_voxcpm2_model_type_detection(self, voxcpm2_server):
        assert voxcpm2_server._tts_model_type == "voxcpm2"
        assert voxcpm2_server._is_tts is True

    def test_voxcpm2_accepts_any_text_input(self, voxcpm2_server):
        """VoxCPM2 skips the strict voxcpm validator — see `_validate_tts_request`."""
        request = OpenAICreateSpeechRequest(input="مرحباً", instructions="warm tone", cfg_value=2.7)
        assert voxcpm2_server._validate_tts_request(request) is None

    def test_build_prompt_text_only(self, voxcpm2_server, mock_build_voxcpm2_prompt):
        """No instructions, no cfg_value: text flows through unchanged, no cfg stash."""
        request = OpenAICreateSpeechRequest(input="hello voxcpm2")
        prompt = asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        assert mock_build_voxcpm2_prompt.call_args.kwargs["text"] == "hello voxcpm2"
        assert "cfg_value" not in prompt.get("additional_information", {})

    def test_build_prompt_prepends_instructions(self, voxcpm2_server, mock_build_voxcpm2_prompt):
        """`instructions` wraps in parens and prepends to text (native VoxCPM2 convention)."""
        request = OpenAICreateSpeechRequest(
            input="hello voxcpm2",
            instructions="A warm young woman",
        )
        asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        assert mock_build_voxcpm2_prompt.call_args.kwargs["text"] == "(A warm young woman)hello voxcpm2"

    def test_build_prompt_strips_instructions_whitespace(self, voxcpm2_server, mock_build_voxcpm2_prompt):
        """Leading/trailing whitespace in `instructions` is stripped before prepending."""
        request = OpenAICreateSpeechRequest(input="hello", instructions="  calm radio  ")
        asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        assert mock_build_voxcpm2_prompt.call_args.kwargs["text"] == "(calm radio)hello"

    def test_build_prompt_stashes_cfg_value(self, voxcpm2_server, mock_build_voxcpm2_prompt):
        """`cfg_value` lands in `additional_information` for the talker to lift."""
        request = OpenAICreateSpeechRequest(input="hello", cfg_value=2.7)
        prompt = asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        assert prompt["additional_information"]["cfg_value"] == 2.7

    def test_build_prompt_omits_cfg_value_when_none(self, voxcpm2_server, mock_build_voxcpm2_prompt):
        """Omitting `cfg_value` must not add the key (talker falls back to its default)."""
        request = OpenAICreateSpeechRequest(input="hello")
        prompt = asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        assert "cfg_value" not in prompt["additional_information"]

    def test_build_prompt_instructions_and_cfg_together(self, voxcpm2_server, mock_build_voxcpm2_prompt):
        """Both features compose: text prefixed AND cfg_value stashed."""
        request = OpenAICreateSpeechRequest(
            input="hello",
            instructions="excited",
            cfg_value=2.5,
        )
        prompt = asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        assert mock_build_voxcpm2_prompt.call_args.kwargs["text"] == "(excited)hello"
        assert prompt["additional_information"]["cfg_value"] == 2.5

    def test_build_prompt_hifi_cloning_ref_audio_ref_text_cfg(
        self, voxcpm2_server, mock_build_voxcpm2_prompt, mocker: MockerFixture
    ):
        """Hi-Fi Cloning (`ref_audio` + `ref_text`) resolves audio and threads cfg_value."""
        voxcpm2_server._resolve_ref_audio = AsyncMock(return_value=([0.1, -0.1, 0.2], 16000))

        request = OpenAICreateSpeechRequest(
            input="clone me",
            ref_audio="data:audio/wav;base64,QUJD",
            ref_text="reference transcript",
            cfg_value=2.7,
        )
        prompt = asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        kwargs = mock_build_voxcpm2_prompt.call_args.kwargs
        assert kwargs["text"] == "clone me"
        assert kwargs["ref_audio"] == [0.1, -0.1, 0.2]
        assert kwargs["ref_sr"] == 16000
        assert kwargs["ref_text"] == "reference transcript"
        assert prompt["additional_information"]["cfg_value"] == 2.7
        voxcpm2_server._resolve_ref_audio.assert_awaited_once_with("data:audio/wav;base64,QUJD")

    def test_build_prompt_hifi_mode_ignores_instructions(
        self, voxcpm2_server, mock_build_voxcpm2_prompt, mocker: MockerFixture
    ):
        """Hi-Fi Cloning (ref_audio + ref_text) strips `instructions` per canonical VoxCPM2 docs.

        Upstream doc: "When Hi-Fi mode is enabled, the control instruction is ignored."
        We drop the prepend server-side so Hi-Fi requests do not get a surprise text prefix.
        """
        voxcpm2_server._resolve_ref_audio = AsyncMock(return_value=([0.1, -0.1, 0.2], 16000))

        request = OpenAICreateSpeechRequest(
            input="hello",
            ref_audio="data:audio/wav;base64,QUJD",
            ref_text="reference transcript",
            instructions="this should be ignored in Hi-Fi mode",
        )
        asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        assert mock_build_voxcpm2_prompt.call_args.kwargs["text"] == "hello"

    def test_validate_rejects_overlong_instructions(self, voxcpm2_server):
        """Instructions longer than `_max_instructions_length` (500 default) are rejected."""
        oversize = "x" * (voxcpm2_server._max_instructions_length + 1)
        request = OpenAICreateSpeechRequest(input="hello", instructions=oversize)
        error = voxcpm2_server._validate_tts_request(request)
        assert error is not None
        assert str(voxcpm2_server._max_instructions_length) in error

    def test_validate_accepts_at_limit_instructions(self, voxcpm2_server):
        """Instructions exactly at `_max_instructions_length` are accepted."""
        at_limit = "x" * voxcpm2_server._max_instructions_length
        request = OpenAICreateSpeechRequest(input="hello", instructions=at_limit)
        assert voxcpm2_server._validate_tts_request(request) is None

    def test_prepare_speech_generation_runs_validator_for_voxcpm2(
        self, voxcpm2_server, mock_build_voxcpm2_prompt, mocker: MockerFixture
    ):
        """Single-request `/v1/audio/speech` path must invoke `_validate_tts_request`
        before building the prompt, so the instructions-length cap actually fires
        for normal (non-batch) requests too. Regression guard for #3118 P2.
        """
        oversize = "x" * (voxcpm2_server._max_instructions_length + 1)
        request = OpenAICreateSpeechRequest(input="hello", instructions=oversize)
        with pytest.raises(ValueError, match=str(voxcpm2_server._max_instructions_length)):
            asyncio.run(voxcpm2_server._prepare_speech_generation(request))
        # The prompt builder must NOT be reached when validation fails.
        mock_build_voxcpm2_prompt.assert_not_called()

    @pytest.mark.parametrize("cfg", [0.5, 1.5, 2.0, 2.7, 5.0, 10.0])
    def test_cfg_value_accepts_range(self, voxcpm2_server, mock_build_voxcpm2_prompt, cfg):
        """Protocol validates 0.1 ≤ cfg_value ≤ 10.0; these should all pass."""
        request = OpenAICreateSpeechRequest(input="hello", cfg_value=cfg)
        prompt = asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))
        assert prompt["additional_information"]["cfg_value"] == cfg

    @pytest.mark.parametrize("cfg", [0.0, -1.0, 10.5, 100.0])
    def test_cfg_value_rejects_out_of_range(self, cfg):
        """Out-of-range `cfg_value` raises at schema construction time."""
        with pytest.raises(ValueError):
            OpenAICreateSpeechRequest(input="hello", cfg_value=cfg)
