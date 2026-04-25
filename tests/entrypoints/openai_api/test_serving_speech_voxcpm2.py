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
        """VoxCPM2 skips the strict voxcpm validator (see `_validate_tts_request`)."""
        request = OpenAICreateSpeechRequest(input="مرحباً", instructions="warm tone")
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
        """`extra_params['cfg_value']` lands in `additional_information` for the talker to lift."""
        request = OpenAICreateSpeechRequest(input="hello", extra_params={"cfg_value": 2.7})
        prompt = asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        assert prompt["additional_information"]["cfg_value"] == 2.7

    def test_build_prompt_omits_cfg_value_when_extra_params_missing(self, voxcpm2_server, mock_build_voxcpm2_prompt):
        """Omitting `cfg_value` must not add the key (talker falls back to its default)."""
        request = OpenAICreateSpeechRequest(input="hello")
        prompt = asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        assert "cfg_value" not in prompt["additional_information"]

    def test_build_prompt_omits_cfg_value_when_extra_params_has_other_keys(
        self, voxcpm2_server, mock_build_voxcpm2_prompt
    ):
        """`extra_params` set but without `cfg_value` must not affect the prompt."""
        request = OpenAICreateSpeechRequest(input="hello", extra_params={"some_other_knob": 1})
        prompt = asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        assert "cfg_value" not in prompt["additional_information"]

    def test_build_prompt_instructions_and_cfg_together(self, voxcpm2_server, mock_build_voxcpm2_prompt):
        """Both features compose: text prefixed AND cfg_value stashed."""
        request = OpenAICreateSpeechRequest(
            input="hello",
            instructions="excited",
            extra_params={"cfg_value": 2.5},
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
            extra_params={"cfg_value": 2.7},
        )
        prompt = asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

        kwargs = mock_build_voxcpm2_prompt.call_args.kwargs
        assert kwargs["text"] == "clone me"
        assert kwargs["ref_audio"] == [0.1, -0.1, 0.2]
        assert kwargs["ref_sr"] == 16000
        assert kwargs["ref_text"] == "reference transcript"
        assert prompt["additional_information"]["cfg_value"] == 2.7
        voxcpm2_server._resolve_ref_audio.assert_awaited_once_with("data:audio/wav;base64,QUJD")

    def test_validate_rejects_instructions_with_ref_text(self, voxcpm2_server):
        """Hi-Fi Cloning (ref_audio + ref_text) + instructions returns a 400-level
        error string from the validator instead of silently dropping `instructions`.

        Matches upstream voxcpm CLI's hard error on the same combo
        (src/voxcpm/cli.py:128-131: --control + --prompt-text).
        """
        request = OpenAICreateSpeechRequest(
            input="hello",
            ref_audio="data:audio/wav;base64,QUJD",
            ref_text="reference transcript",
            instructions="this is incompatible with Hi-Fi mode",
        )
        error = voxcpm2_server._validate_tts_request(request)
        assert error is not None
        assert "Hi-Fi" in error
        assert "ref_text" in error

    def test_validate_accepts_instructions_without_ref_text(self, voxcpm2_server):
        """`instructions` is fine in Voice Design and Controllable Cloning modes
        (i.e. whenever `ref_text` is not also set)."""
        request = OpenAICreateSpeechRequest(input="hello", instructions="warm tone")
        assert voxcpm2_server._validate_tts_request(request) is None

    def test_validate_accepts_ref_text_without_instructions(self, voxcpm2_server):
        """Hi-Fi Cloning (ref_audio + ref_text, no instructions) is fine."""
        request = OpenAICreateSpeechRequest(
            input="hello",
            ref_audio="data:audio/wav;base64,QUJD",
            ref_text="reference transcript",
        )
        assert voxcpm2_server._validate_tts_request(request) is None

    def test_validate_rejects_overlong_instructions(self, voxcpm2_server):
        """Instructions longer than `_max_instructions_length` (500 default) are rejected
        with an error message that includes the actual length, the cap, and an
        upstream-context hint so users coming from the voxcpm CLI understand why
        the limit exists.
        """
        oversize = "x" * (voxcpm2_server._max_instructions_length + 1)
        request = OpenAICreateSpeechRequest(input="hello", instructions=oversize)
        error = voxcpm2_server._validate_tts_request(request)
        assert error is not None
        assert str(len(oversize)) in error
        assert str(voxcpm2_server._max_instructions_length) in error
        assert "Upstream voxcpm has no cap" in error

    def test_validate_accepts_at_limit_instructions(self, voxcpm2_server):
        """Instructions exactly at `_max_instructions_length` are accepted."""
        at_limit = "x" * voxcpm2_server._max_instructions_length
        request = OpenAICreateSpeechRequest(input="hello", instructions=at_limit)
        assert voxcpm2_server._validate_tts_request(request) is None

    def test_prepare_speech_generation_runs_validator_for_voxcpm2_length(
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

    def test_prepare_speech_generation_runs_validator_for_voxcpm2_hifi(
        self, voxcpm2_server, mock_build_voxcpm2_prompt, mocker: MockerFixture
    ):
        """Single-request path must also surface the Hi-Fi mode rejection
        (instructions + ref_text), not silently strip `instructions`.
        """
        request = OpenAICreateSpeechRequest(
            input="hello",
            ref_audio="data:audio/wav;base64,QUJD",
            ref_text="reference transcript",
            instructions="incompatible with Hi-Fi mode",
        )
        with pytest.raises(ValueError, match="Hi-Fi"):
            asyncio.run(voxcpm2_server._prepare_speech_generation(request))
        mock_build_voxcpm2_prompt.assert_not_called()

    @pytest.mark.parametrize("cfg", [0.1, 0.5, 1.5, 2.0, 2.7, 5.0, 10.0])
    def test_cfg_value_accepts_range(self, voxcpm2_server, mock_build_voxcpm2_prompt, cfg):
        """`_build_voxcpm2_prompt` accepts cfg_value within 0.1-10.0 inclusive."""
        request = OpenAICreateSpeechRequest(input="hello", extra_params={"cfg_value": cfg})
        prompt = asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))
        assert prompt["additional_information"]["cfg_value"] == cfg

    @pytest.mark.parametrize("cfg", [0.0, -1.0, 10.5, 100.0])
    def test_cfg_value_rejects_out_of_range(self, voxcpm2_server, mock_build_voxcpm2_prompt, cfg):
        """Out-of-range `cfg_value` (validated in `_build_voxcpm2_prompt`) raises ValueError."""
        request = OpenAICreateSpeechRequest(input="hello", extra_params={"cfg_value": cfg})
        with pytest.raises(ValueError, match="out of range"):
            asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))

    @pytest.mark.parametrize("bad", ["abc", None, [1, 2], {"x": 1}])
    def test_cfg_value_rejects_non_numeric(self, voxcpm2_server, mock_build_voxcpm2_prompt, bad):
        """Non-numeric `extra_params['cfg_value']` raises ValueError instead of crashing the talker."""
        request = OpenAICreateSpeechRequest(input="hello", extra_params={"cfg_value": bad})
        with pytest.raises(ValueError, match="must be a number|out of range"):
            asyncio.run(voxcpm2_server._build_voxcpm2_prompt(request))


class TestVoxCPM2DecodeStepCfgPropagation:
    """Static AST regression guard for #3118 P1: ensure both `_finish_prefill`
    and `_finish_decode` invoke `_run_cfm` with the per-request `cfg_value`
    kwarg. Originally only `_finish_prefill` threaded the kwarg, so requests
    longer than the prefill patch fell back to `self._cfg_value = 2.0` for
    every decode step. This test parses the talker source and asserts the
    kwarg is present at every `_run_cfm` call site inside the two methods,
    so a future refactor that drops the kwarg from either method fails CI
    even before the behavioral cfg-sweep tests run.
    """

    @pytest.fixture(scope="class")
    def talker_class_ast(self):
        import ast
        import inspect

        from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_talker

        src = inspect.getsource(voxcpm2_talker)
        tree = ast.parse(src)
        cls = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "VoxCPM2TalkerForConditionalGeneration"
        )
        return cls

    def _run_cfm_calls_in(self, cls_ast, method_name):
        import ast

        method = next(n for n in cls_ast.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
        return [
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_run_cfm"
        ]

    @pytest.mark.parametrize("method_name", ["_finish_prefill", "_finish_decode"])
    def test_run_cfm_called_with_cfg_value_kwarg(self, talker_class_ast, method_name):
        calls = self._run_cfm_calls_in(talker_class_ast, method_name)
        assert calls, f"{method_name} must call self._run_cfm at least once"
        for call in calls:
            kwarg_names = {kw.arg for kw in call.keywords if kw.arg}
            assert "cfg_value" in kwarg_names, (
                f"{method_name}: every self._run_cfm call must thread the "
                f"per-request `cfg_value` kwarg (the bug fixed in #3118 P1 was "
                f"that `_finish_decode` dropped this kwarg, so all decode steps "
                f"silently fell back to the talker default `self._cfg_value`)."
            )
