# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the TTS serving adapter registry (RFC #4327).

Pure-Python registry/resolution logic; no model or GPU resources are loaded.
"""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import (
    TTS_ADAPTER_REGISTRY,
    ARTTSAdapter,
    DiffusionTTSAdapter,
    SpeechServingContext,
    all_tts_model_types,
    detect_tts_model_type,
    resolve_adapter,
)
from vllm_omni.entrypoints.openai.tts_adapters.auk import AuKAdapter
from vllm_omni.entrypoints.openai.tts_adapters.base import resolve_stage_model_path
from vllm_omni.entrypoints.openai.tts_adapters.covo_audio import CovoAudioAdapter
from vllm_omni.entrypoints.openai.tts_adapters.higgs_audio_v2 import HiggsAudioV2Adapter
from vllm_omni.entrypoints.openai.tts_adapters.indextts2 import (
    IndexTTS2Adapter,
    IndexTTS25Adapter,
    indextts2_conditioning_cache_salt,
)
from vllm_omni.entrypoints.openai.tts_adapters.ming_flash_omni_tts import MingFlashOmniTTSAdapter
from vllm_omni.entrypoints.openai.tts_adapters.moss_tts import (
    MossTTSAdapter,
    MossTTSNanoAdapter,
)
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import (
    QWEN3_TTS_EFFECTIVE_MAX_TOKENS_KEY,
    Qwen3TTSAdapter,
    Qwen3TTSCodecLimitError,
)
from vllm_omni.entrypoints.openai.tts_adapters.step_audio2 import StepAudio2Adapter
from vllm_omni.entrypoints.openai.tts_adapters.voxtral import VoxtralTTSAdapter
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.indextts2 import prompt_utils
from vllm_omni.model_executor.models.indextts2.tokenizer_v2_5 import (
    INDEXTTS25_TOKENIZER_FILE,
)
from vllm_omni.model_executor.stage_input_processors.auk import encoder2dit

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# Every dedicated TTS model-type must have an adapter so the orchestrator's
# uniform ``self._adapter.build(...)`` dispatch covers it.
EXPECTED_MODEL_TYPES = {
    "qwen3_tts",
    "voxcpm2",
    "voxtral_tts",
    "fish_tts",
    "cosyvoice3",
    "omnivoice",
    "covo_audio",
    "ming_tts",
    "moss_tts_nano",
    "moss_tts",
    "higgs_audio_v2",
    "higgs_audio_v3",
    "glm_tts",
    "breeze_tts_2",
    "step_audio2",
    "indextts2",
    "indextts2_5",
    "auk",
    "gepard",
}


def test_all_model_types_registered():
    assert EXPECTED_MODEL_TYPES <= all_tts_model_types()


def test_registry_keyed_by_name():
    for name, cls in TTS_ADAPTER_REGISTRY.items():
        assert cls.name == name


def test_resolve_each_model_type():
    for model_type in EXPECTED_MODEL_TYPES:
        cls = resolve_adapter(model_type)
        assert cls is not None, model_type
        assert cls.name == model_type


def test_resolve_qwen3_tts_class():
    assert resolve_adapter("qwen3_tts") is Qwen3TTSAdapter


def test_resolve_unknown_returns_none():
    assert resolve_adapter("not_a_real_model") is None
    assert resolve_adapter(None) is None


@pytest.fixture
def auk_adapter(mocker):
    server = mocker.Mock(spec=OmniOpenAIServingSpeech)
    server._max_instructions_length = 4096
    server._validate_ref_audio_format.return_value = None
    server._resolve_ref_audio = AsyncMock(return_value=([0.1] * 480, 24000, "key"))
    return AuKAdapter(SpeechServingContext(server=server))


def test_auk_adapter_detection(mocker):
    assert detect_tts_model_type("encoder", "AuKForConditionalGeneration") == "auk"
    assert AuKAdapter.stage_keys == frozenset({"encoder"})

    server = mocker.Mock(spec=OmniOpenAIServingSpeech)
    server._diffusion_mode = False
    server._tts_model_type = detect_tts_model_type("encoder", "AuKForConditionalGeneration")
    server._adapter = None
    server.engine_client = SimpleNamespace()

    adapter = OmniOpenAIServingSpeech._get_tts_adapter(server)
    assert isinstance(adapter, AuKAdapter)
    server._drop_shadowing_uploads.assert_called_once_with()


def test_auk_source_length_default_and_complete_instruction(auk_adapter):
    request = OpenAICreateSpeechRequest(
        input="",
        ref_audio="reference.wav",
        ref_text="Accepted transcript",
        instructions="Change the pitch by one semitone.",
        extra_params={"sway": -0.5, "t_grid": [0, 0.25, 1], "vae_sample": True},
    )
    assert auk_adapter.validate(request) is None
    prepared = asyncio.run(auk_adapter.build(request, [], True))
    assert "Change the pitch by one semitone." in prepared.prompt["prompt"]
    assert "content to speak" not in prepared.prompt["prompt"]
    knobs = prepared.prompt["additional_information"]["auk"]
    assert knobs["gen_seconds"] is None
    assert knobs["vae_sample"] is True
    assert knobs["t_grid"] == [0, 0.25, 1]
    auk_adapter.ctx.server._resolve_ref_audio.assert_awaited_once_with("reference.wav")


@pytest.mark.parametrize(
    "t_grid",
    [
        [],
        [0.0],
        [0.0, 0.0],
        [0.5, 0.25],
        [0.0, float("nan")],
        [0.0, float("inf")],
        [0.0, "invalid"],
        "0,1",
    ],
)
def test_auk_rejects_invalid_t_grid_before_dispatch(auk_adapter, t_grid):
    request = OpenAICreateSpeechRequest(
        input="Say the following: 'target text'",
        duration_seconds=2,
        extra_params={"t_grid": t_grid},
    )

    error = auk_adapter.validate(request)

    assert error == "AuK extra_params.t_grid must contain at least two finite, strictly increasing values"
    auk_adapter.ctx.server._resolve_ref_audio.assert_not_awaited()


@pytest.mark.parametrize(
    ("task_type", "expected_instruction"),
    [
        ("CustomVoice", 'Say the following: "target text"'),
        ("Base", 'Say the following with the same voice: "target text"'),
    ],
)
def test_auk_task_type_normalizes_benchmark_text(auk_adapter, task_type, expected_instruction, mocker):
    warning_once = mocker.patch("vllm_omni.entrypoints.openai.tts_adapters.auk.logger.warning_once")
    request = OpenAICreateSpeechRequest(
        input="target text",
        task_type=task_type,
        duration_seconds=2,
        ref_audio="reference.wav" if task_type == "Base" else None,
    )

    server = mocker.Mock(spec=OmniOpenAIServingSpeech)
    server._validate_speech_sample_rate.return_value = None
    server._get_tts_adapter.return_value = auk_adapter
    assert OmniOpenAIServingSpeech._validate_tts_request(server, request) is None

    assert request.input == ""
    assert request.instructions == expected_instruction
    assert request.task_type is None
    warning_once.assert_called_once()
    assert "prefer a complete `instructions` prompt" in warning_once.call_args.args[0]

    auk_adapter.normalize(request)

    assert request.instructions == expected_instruction
    warning_once.assert_called_once()


def test_auk_task_type_quotes_embedded_delimiters(auk_adapter):
    request = OpenAICreateSpeechRequest(
        input='It\'s called "AuK"\\Flash',
        task_type="CustomVoice",
        duration_seconds=2,
    )

    auk_adapter.normalize(request)

    assert request.instructions == 'Say the following: "It\'s called \\"AuK\\"\\\\Flash"'
    assert request.task_type is None


def test_auk_task_type_preserves_explicit_instructions(auk_adapter, mocker):
    warning_once = mocker.patch("vllm_omni.entrypoints.openai.tts_adapters.auk.logger.warning_once")
    warning = mocker.patch("vllm_omni.entrypoints.openai.tts_adapters.auk.logger.warning")
    request = OpenAICreateSpeechRequest(
        input="target text",
        task_type="Base",
        instructions="Use this complete AuK instruction.",
        duration_seconds=2,
    )

    server = mocker.Mock(spec=OmniOpenAIServingSpeech)
    server._validate_speech_sample_rate.return_value = None
    server._get_tts_adapter.return_value = auk_adapter
    assert OmniOpenAIServingSpeech._validate_tts_request(server, request) is None

    assert request.input == ""
    assert request.instructions == "Use this complete AuK instruction."
    assert request.task_type is None
    warning_once.assert_called_once()
    warning.assert_called_once()
    assert "without applying another task template" in warning.call_args.args[0]


def test_auk_task_type_does_not_double_wrap_preformatted_input(auk_adapter, mocker):
    warning_once = mocker.patch("vllm_omni.entrypoints.openai.tts_adapters.auk.logger.warning_once")
    warning = mocker.patch("vllm_omni.entrypoints.openai.tts_adapters.auk.logger.warning")
    instruction = "Say the following with the same voice: 'target text'"
    request = OpenAICreateSpeechRequest(
        input=instruction,
        task_type="Base",
        duration_seconds=2,
        ref_audio="reference.wav",
    )
    server = mocker.Mock(spec=OmniOpenAIServingSpeech)
    server._validate_speech_sample_rate.return_value = None
    server._get_tts_adapter.return_value = auk_adapter

    assert OmniOpenAIServingSpeech._validate_tts_request(server, request) is None

    assert request.input == ""
    assert request.instructions == instruction
    assert request.task_type is None
    warning_once.assert_called_once()
    warning.assert_called_once()
    assert "already contains a complete task instruction" in warning.call_args.args[0]


def test_auk_sampling_overrides_reach_stage1_pipeline(auk_adapter, mocker):
    """Exercise one request from the Speech adapter through stage-1 admission."""

    class RecordingDiffusionStage:
        stage_type = "diffusion"
        final_output = True
        engine_input_source = [0]
        requires_multimodal_data = False
        custom_process_input_func = staticmethod(encoder2dit)

        def __init__(self) -> None:
            self.request: OmniDiffusionRequest | None = None

        async def add_request_async(self, request_id, prompt, sampling_params, **_kwargs) -> None:
            # Match the stage-client wire boundary and the receiving diffusion
            # process, where params are serialized then reconstructed before
            # creating the request consumed by the pipeline.
            wire_params = StageDiffusionClient._sampling_params_to_dict(sampling_params)
            self.request = OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=StageDiffusionProc._reconstruct_sampling_params(
                    object.__new__(StageDiffusionProc), wire_params
                ),
                request_id=request_id,
            )

    defaults = [SamplingParams(max_tokens=1), OmniDiffusionSamplingParams(seed=0, num_inference_steps=32)]
    request = OpenAICreateSpeechRequest(
        input="",
        instructions='Generate speech based on the following description: "A calm voice". The content to speak is: "Hello".',
        duration_seconds=2,
        seed=9,
        extra_params={
            "num_inference_steps": 4,
            "guidance_scale": 1.5,
            "sway": -0.5,
            "t_grid": [0.0, 0.25, 1.0],
            "vae_sample": True,
        },
    )
    prepared = asyncio.run(auk_adapter.build(request, defaults, False))
    updated = auk_adapter.apply_sampling_overrides(defaults, request)
    assert defaults[1].num_inference_steps == 32
    assert defaults[1].seed == 0
    assert updated[0].max_tokens == 1

    source_stage = SimpleNamespace(
        stage_type="llm",
        final_output=False,
        get_kv_sender_info=lambda: None,
    )
    diffusion_stage = RecordingDiffusionStage()
    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [StagePool(0, source_stage), StagePool(1, diffusion_stage)]
    orchestrator._cfg_tracker = CfgCompanionTracker()
    orchestrator.duplex_control_plane = None
    orchestrator._running_counter = None

    req_state = OrchestratorRequestState(
        request_id="auk-sampling-override",
        prompt=prepared.prompt,
        sampling_params_list=updated,
        final_stage_id=1,
    )
    encoder_output = SimpleNamespace(
        request_id=req_state.request_id,
        finished=True,
        multimodal_output={"hidden_states": {"output": torch.ones(3, 4)}},
        outputs=[],
    )
    asyncio.run(Orchestrator._forward_to_next_stage(orchestrator, req_state.request_id, 0, encoder_output, req_state))

    received = diffusion_stage.request
    assert received is not None
    assert received.prompt["prompt_embeds"].shape == (3, 4)
    assert received.prompt["additional_information"]["auk"] == {
        "gen_seconds": 2,
        "sway": -0.5,
        "t_grid": [0.0, 0.25, 1.0],
        "vae_sample": True,
        "has_audio": False,
    }
    assert received.sampling_params.num_inference_steps == 4
    assert received.sampling_params.guidance_scale == 1.5
    assert received.sampling_params.guidance_scale_provided is True
    assert received.sampling_params.seed == 9

    runner = mocker.Mock(spec=DiffusionModelRunner)
    runner.device = torch.device("cpu")
    DiffusionModelRunner._initialize_generator(runner, received.sampling_params)
    assert received.sampling_params.generator.initial_seed() == 9

    pipeline_params = DiffusionRequestBatch(requests=[received]).sampling_params
    assert pipeline_params is received.sampling_params
    assert pipeline_params.generator.initial_seed() == 9


def test_auk_instructions_take_precedence_over_input(auk_adapter, mocker):
    warning = mocker.patch("vllm_omni.entrypoints.openai.tts_adapters.auk.logger.warning")
    request = OpenAICreateSpeechRequest(
        input="ignored input",
        instructions='Generate speech based on the following description: "Speak warmly". The content to speak is: "Hello".',
        duration_seconds=2,
    )
    assert auk_adapter.validate(request) is None
    prepared = asyncio.run(auk_adapter.build(request, [], False))
    prompt = prepared.prompt
    assert 'Generate speech based on the following description: "Speak warmly".' in prompt["prompt"]
    assert 'The content to speak is: "Hello".' in prompt["prompt"]
    assert "|<no_prompt_audio>|" in prompt["prompt"]
    assert prompt["additional_information"]["auk"]["gen_seconds"] == 2
    warning.assert_called_once()
    assert "using the complete `instructions`" in warning.call_args.args[0]


def test_auk_input_only_is_treated_as_complete_instruction(auk_adapter, mocker):
    warning = mocker.patch("vllm_omni.entrypoints.openai.tts_adapters.auk.logger.warning")
    request = OpenAICreateSpeechRequest(
        input="Say the following with the same voice: 'Hello world.'",
        ref_audio="reference.wav",
        duration_seconds=2,
    )
    assert auk_adapter.validate(request) is None
    prepared = asyncio.run(auk_adapter.build(request, [], True))
    assert "Say the following with the same voice: 'Hello world.'" in prepared.prompt["prompt"]
    assert warning.call_count == 1
    assert 'Prefer `input=""`' in warning.call_args.args[0]


def test_auk_complete_instruction_does_not_require_spoken_text(auk_adapter):
    request = OpenAICreateSpeechRequest(
        input="",
        instructions="Replace 'morning' with 'evening' in the source recording.",
        ref_audio="reference.wav",
    )
    assert auk_adapter.validate(request) is None
    prepared = asyncio.run(auk_adapter.build(request, [], True))
    assert prepared.prompt["prompt"].count("Replace 'morning'") == 1
    assert "Generate speech based on" not in prepared.prompt["prompt"]
    assert prepared.prompt["additional_information"]["auk"]["gen_seconds"] is None


def test_stage_model_path_prefers_typed_override():
    engine_client = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(engine_args=SimpleNamespace(model="legacy-stage-model")),
            SimpleNamespace(
                model_config=SimpleNamespace(model="typed-stage-model"),
            ),
        ],
        model="served-model",
    )

    assert resolve_stage_model_path(engine_client) == "typed-stage-model"


def test_stage_model_path_falls_back_to_legacy_then_served_model():
    legacy_client = SimpleNamespace(
        stage_configs=[SimpleNamespace(engine_args=SimpleNamespace(model="legacy-stage-model"))],
        model="served-model",
    )
    served_client = SimpleNamespace(stage_configs=[SimpleNamespace()], model="served-model")

    assert resolve_stage_model_path(legacy_client) == "legacy-stage-model"
    assert resolve_stage_model_path(served_client) == "served-model"


def test_voxcpm2_resolves():
    """VoxCPM2 (the served ``latent_generator`` model) resolves cleanly.

    Detection never returns the legacy ``voxcpm`` type, so there is no shared
    stage-key ambiguity to resolve.
    """
    assert resolve_adapter("voxcpm2") is not None
    assert resolve_adapter("voxcpm") is None


def test_all_adapters_are_ar_or_diffusion():
    for cls in TTS_ADAPTER_REGISTRY.values():
        assert issubclass(cls, (ARTTSAdapter, DiffusionTTSAdapter))
        assert cls.backend in ("ar", "diffusion")


@pytest.mark.parametrize("adapter_cls", [MossTTSAdapter, MossTTSNanoAdapter])
def test_moss_tts_applies_request_max_new_tokens(adapter_cls):
    adapter = adapter_cls(SimpleNamespace(server=object()))
    stage_defaults = [SimpleNamespace(max_tokens=4096)]

    overridden = adapter.apply_sampling_overrides(
        stage_defaults,
        SimpleNamespace(max_new_tokens=512),
    )

    assert overridden[0].max_tokens == 512
    assert stage_defaults[0].max_tokens == 4096


def _build_moss_tts_request(adapter_cls, mocker, *, request_seed):
    server = mocker.Mock()
    adapter = adapter_cls(SpeechServingContext(server=server))
    adapter._build_moss_tts_params = mocker.AsyncMock(return_value={})
    request = OpenAICreateSpeechRequest(input="hello", seed=request_seed)

    return asyncio.run(
        adapter.build(
            request,
            [SamplingParams(seed=42)],
            has_inline_ref_audio=False,
        )
    )


@pytest.mark.parametrize(
    ("adapter_cls", "expect_accumulate"),
    [(MossTTSAdapter, False), (MossTTSNanoAdapter, True)],
)
def test_moss_tts_accumulate_nonstreaming_follows_adapter_flag(adapter_cls, expect_accumulate, mocker):
    prepared = _build_moss_tts_request(adapter_cls, mocker, request_seed=7)

    assert adapter_cls.accumulate_nonstreaming is expect_accumulate
    assert prepared.output_policy.accumulate_nonstreaming is expect_accumulate


# Full-family coverage pins the adapter contract; only Nano consumes this seed end to end today.
@pytest.mark.parametrize("adapter_cls", [MossTTSAdapter, MossTTSNanoAdapter])
@pytest.mark.parametrize("request_seed", [0, 1234])
def test_moss_tts_request_seed_overrides_stage_default(adapter_cls, request_seed, mocker):
    prepared = _build_moss_tts_request(
        adapter_cls,
        mocker,
        request_seed=request_seed,
    )

    assert prepared.tts_params["seed"] == [request_seed]


@pytest.mark.parametrize("adapter_cls", [MossTTSAdapter, MossTTSNanoAdapter])
def test_moss_tts_seed_falls_back_to_stage_default(adapter_cls, mocker):
    prepared = _build_moss_tts_request(
        adapter_cls,
        mocker,
        request_seed=None,
    )

    assert prepared.tts_params["seed"] == [42]


@pytest.mark.parametrize(
    "variant,ref_text,mode",
    [
        ("local", " Reference. ", "continuation"),
        ("local", None, "generation"),
        ("local", "  ", "generation"),
        ("tts", "Reference.", "generation"),
    ],
)
def test_moss_reference_transcript_mode(variant, ref_text, mode, mocker):
    import torch

    reference = [torch.ones((3, 12), dtype=torch.int64)]
    unified = torch.arange(52, dtype=torch.int64).reshape(1, 4, 13)
    processor = mocker.Mock(return_value={"input_ids": unified})
    server = mocker.Mock(_moss_variant="local", uploaded_speakers={"speaker": {}})
    server._voice_created_at.return_value = 123
    request = OpenAICreateSpeechRequest(
        input="Target.", ref_text=ref_text, language="English", voice="Speaker", seed=0, max_new_tokens=2048
    )
    adapter = MossTTSAdapter(SpeechServingContext(server=server))

    adapter._moss_variant = variant
    mocker.patch.object(adapter, "_get_moss_processor", return_value=processor)
    mocker.patch.object(
        adapter, "_encode_moss_references", new=mocker.AsyncMock(return_value=(reference, {0: "reference-key"}))
    )

    prepared = asyncio.run(adapter.build(request, [], has_inline_ref_audio=False))

    if mode == "continuation":
        processor.build_user_message.assert_called_once_with(text="Reference. Target.", language="English")
        processor.build_assistant_message.assert_called_once_with(audio_codes_list=reference)
        conversation = [processor.build_user_message.return_value, processor.build_assistant_message.return_value]
    else:
        processor.build_user_message.assert_called_once_with(text="Target.", language="English", reference=reference)
        processor.build_assistant_message.assert_not_called()
        conversation = [processor.build_user_message.return_value]
    processor.assert_called_once_with(conversations=[conversation], mode=mode)
    assert prepared.prompt["prompt_token_ids"] == unified[0, :, 0].tolist()
    assert torch.equal(prepared.tts_params["codes"]["ref"], unified[0, :, 1:])
    assert prepared.tts_params["max_new_frames"] == [2048]
    assert prepared.tts_params["ref_audio_cache_key"] == "reference-key"
    assert prepared.tts_params["voice_name"] == ["speaker"]
    assert prepared.tts_params["voice_created_at"] == [123]
    assert prepared.tts_params["seed"] == [0]
    assert prepared.prompt["cache_salt"]
    assert request.input == "Target."


def _moss_adapter_with_stage_configs(stage_configs, mocker):
    engine_client = SimpleNamespace(
        model_config=SimpleNamespace(model="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"),
        stage_configs=stage_configs,
    )
    return MossTTSAdapter(SpeechServingContext(server=mocker.Mock(), engine_client=engine_client))


def _moss_cuda_available(mocker, device_count=8):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.accelerator.device_count", return_value=device_count)


@pytest.mark.parametrize(
    "stage_configs,expected",
    [
        # Follows the code2wav stage's first device.
        (
            [
                SimpleNamespace(model_stage="moss_tts_local", runtime_config=SimpleNamespace(devices="0")),
                SimpleNamespace(model_stage="moss_tts_local_codec", runtime_config=SimpleNamespace(devices="3")),
            ],
            "cuda:3",
        ),
        # No codec-named stage: anchors on the last stage of the pipeline.
        (
            [
                SimpleNamespace(model_stage="talker", runtime_config=SimpleNamespace(devices="1")),
                SimpleNamespace(model_stage="decoder", runtime_config=SimpleNamespace(devices="2")),
            ],
            "cuda:2",
        ),
        # Multi-device codec stage pins replica 0 on the first device.
        (
            [
                SimpleNamespace(model_stage="moss_tts_local_codec", runtime_config=SimpleNamespace(devices="5,6")),
            ],
            "cuda:5",
        ),
        # No explicit pinning anywhere: stage worker defaults to cuda:0.
        (
            [SimpleNamespace(model_stage="moss_tts_local_codec", runtime_config=SimpleNamespace(devices=None))],
            "cuda:0",
        ),
    ],
)
def test_moss_ref_encoder_device_follows_codec_stage(stage_configs, expected, mocker):
    import torch

    _moss_cuda_available(mocker)
    adapter = _moss_adapter_with_stage_configs(stage_configs, mocker)
    assert adapter._resolve_ref_encoder_device() == torch.device(expected)


def test_moss_ref_encoder_device_accepts_dict_runtime(mocker):
    import torch

    _moss_cuda_available(mocker)
    adapter = _moss_adapter_with_stage_configs(
        [SimpleNamespace(model_stage="moss_tts_local_codec", runtime={"devices": "2"})],
        mocker,
    )
    assert adapter._resolve_ref_encoder_device() == torch.device("cuda:2")


def test_moss_ref_encoder_device_cpu_when_cuda_unavailable(mocker):
    import torch

    mocker.patch("torch.cuda.is_available", return_value=False)
    adapter = _moss_adapter_with_stage_configs(
        [SimpleNamespace(model_stage="moss_tts_local_codec", runtime_config=SimpleNamespace(devices="3"))],
        mocker,
    )
    assert adapter._resolve_ref_encoder_device() == torch.device("cpu")


def test_moss_ref_encoder_device_cpu_when_index_out_of_range(mocker):
    import torch

    _moss_cuda_available(mocker, device_count=2)
    adapter = _moss_adapter_with_stage_configs(
        [SimpleNamespace(model_stage="moss_tts_local_codec", runtime_config=SimpleNamespace(devices="5"))],
        mocker,
    )
    assert adapter._resolve_ref_encoder_device() == torch.device("cpu")


def test_moss_get_processor_places_audio_tokenizer_on_codec_stage_device(mocker):
    import torch

    _moss_cuda_available(mocker)
    adapter = _moss_adapter_with_stage_configs(
        [SimpleNamespace(model_stage="moss_tts_local_codec", runtime_config=SimpleNamespace(devices="3"))],
        mocker,
    )
    audio_tokenizer = mocker.Mock()
    audio_tokenizer.to.return_value = audio_tokenizer
    processor = SimpleNamespace(audio_tokenizer=audio_tokenizer)
    from_pretrained = mocker.patch("transformers.AutoProcessor.from_pretrained", return_value=processor)

    assert adapter._get_moss_processor() is processor
    audio_tokenizer.to.assert_called_once_with(torch.device("cuda", 3))
    audio_tokenizer.eval.assert_called_once_with()
    # Lazy cache: no re-load / re-placement on the second call.
    assert adapter._get_moss_processor() is processor
    from_pretrained.assert_called_once()
    audio_tokenizer.to.assert_called_once()


def test_qwen3_tts_metadata():
    assert Qwen3TTSAdapter.backend == "ar"
    assert issubclass(Qwen3TTSAdapter, ARTTSAdapter)


def test_qwen3_tts_build_constructs_prepared_request():
    server = SimpleNamespace(_tts_executor=None, _tts_tokenizer=None, uploaded_speakers={})
    engine_client = SimpleNamespace(model_config=SimpleNamespace())
    adapter = Qwen3TTSAdapter(SimpleNamespace(server=server, engine_client=engine_client))

    async def estimate_prompt_len(_tts_params):
        return 3

    adapter._estimate_prompt_len_async = estimate_prompt_len
    request = OpenAICreateSpeechRequest(input="hello")

    prepared = asyncio.run(adapter.build(request, [], False))

    assert prepared.prompt["prompt_token_ids"] == [1, 1, 1]
    assert prepared.prompt["additional_information"] is prepared.tts_params
    assert prepared.tts_params["text"] == ["hello"]
    assert prepared.tts_params["speaker"] == ["Vivian"]
    assert prepared.model_type == "CustomVoice"


def test_covo_audio_build_constructs_tokenized_prompt():
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages[-1] == {"role": "user", "content": "hello"}
            assert tokenize is True
            assert add_generation_prompt is True
            return [11, 12, 13]

    adapter = CovoAudioAdapter(SimpleNamespace(server=SimpleNamespace(), engine_client=SimpleNamespace()))
    adapter._tokenizer = FakeTokenizer()

    prepared = asyncio.run(adapter.build(OpenAICreateSpeechRequest(input="hello"), [], False))

    assert prepared.prompt == {"prompt_token_ids": [11, 12, 13]}
    assert prepared.tts_params == {}
    assert prepared.model_type == "covo_audio"


def test_ming_flash_omni_build_constructs_prepared_request():
    server = SimpleNamespace(uploaded_speakers={})
    adapter = MingFlashOmniTTSAdapter(SimpleNamespace(server=server, engine_client=SimpleNamespace()))
    request = OpenAICreateSpeechRequest(input="hello", voice="test")

    prepared = asyncio.run(adapter.build(request, [], False))

    assert prepared.prompt["prompt_token_ids"] == [0]
    info = prepared.prompt["additional_information"]
    assert info["ming_task"] == "instruct"
    assert info["text"] == "hello"
    assert info["voice_name"] == "test"
    assert prepared.model_type == "ming_flash_omni_tts"


def test_step_audio2_build_constructs_open_assistant_turn():
    adapter = StepAudio2Adapter(SimpleNamespace(server=SimpleNamespace(), engine_client=SimpleNamespace()))
    request = OpenAICreateSpeechRequest(input="hello", instructions="Read warmly.")

    prepared = asyncio.run(adapter.build(request, [], False))

    assert "<|im_start|>system\nRead warmly.<|im_end|>" in prepared.prompt["prompt"]
    assert prepared.prompt["prompt"].endswith("<|im_start|>assistant\n<tts_start>")
    assert prepared.tts_params == {}
    assert prepared.model_type == "step_audio2"


def test_voxtral_prompt_builder_is_sync():
    ctx = SimpleNamespace(
        server=SimpleNamespace(_tts_executor=None),
        engine_client=SimpleNamespace(),
    )
    adapter = VoxtralTTSAdapter(ctx)

    assert not asyncio.iscoroutinefunction(adapter._build_prompt)


@pytest.mark.parametrize(
    ("task_type", "text_tokens", "request_cap", "expected_cap"),
    [
        ("Base", 0, None, 4096),
        ("Base", 10, None, 192),
        ("Base", 23, None, 276),
        ("Base", 23, 128, 128),
        ("Base", 23, 512, 512),
        ("Base", 400, None, 4096),
        ("CustomVoice", 10, None, 4096),
        ("CustomVoice", 10, 256, 256),
    ],
)
def test_qwen3_tts_applies_text_scaled_codec_safety_limit(task_type, text_tokens, request_cap, expected_cap, mocker):
    server = mocker.Mock()
    server._count_usage_text_tokens.return_value = text_tokens
    adapter = Qwen3TTSAdapter(SpeechServingContext(server=server))
    stage_defaults = [SamplingParams(max_tokens=4096, min_tokens=2)]
    request = OpenAICreateSpeechRequest(
        input="test text",
        task_type=task_type,
        max_new_tokens=request_cap,
    )
    prompt: dict[str, Any] = {"additional_information": {}}

    overridden = adapter.apply_sampling_overrides(stage_defaults, request, prompt)

    assert overridden[0].max_tokens == expected_cap
    assert prompt["additional_information"][QWEN3_TTS_EFFECTIVE_MAX_TOKENS_KEY] == [expected_cap]
    assert stage_defaults[0].max_tokens == 4096


def test_qwen3_tts_rejects_only_length_finished_base_audio(mocker):
    adapter = Qwen3TTSAdapter(SpeechServingContext(server=mocker.Mock()))
    params = {
        "task_type": ["Base"],
        QWEN3_TTS_EFFECTIVE_MAX_TOKENS_KEY: [192],
    }

    # EOS at the exact budget is valid; the token count alone is not a
    # sufficient failure signal.
    adapter.validate_generation(params, stage0_finish_reason="stop", output_tokens=192)
    adapter.validate_generation(params, stage0_finish_reason=None, output_tokens=192)

    # Some frontends expose 191 decoded frames for max_new_tokens=192. The
    # engine terminal reason still makes this an unambiguous limit failure.
    with pytest.raises(Qwen3TTSCodecLimitError, match="191/192"):
        adapter.validate_generation(params, stage0_finish_reason="length", output_tokens=191)


def test_indextts_adapters_are_versioned():
    assert resolve_adapter("indextts2") is IndexTTS2Adapter
    assert resolve_adapter("indextts2_5") is IndexTTS25Adapter
    assert IndexTTS25Adapter.stage_keys == frozenset({"indextts2_5_talker"})
    assert detect_tts_model_type("indextts2_5_talker", None) == "indextts2_5"


def test_indextts25_validates_explicit_language():
    adapter = IndexTTS25Adapter(type("Context", (), {"server": object()})())

    assert adapter._validate_extra_params({"lang": "ja"}) is None
    assert "Unsupported IndexTTS 2.5 language" in adapter._validate_extra_params({"lang": "xx-invalid"})


def _indextts25_adapter_and_request(*, speed: float):
    server = SimpleNamespace(
        uploaded_speakers={},
        _validate_ref_audio_format=lambda ref_audio: None,
    )
    adapter = IndexTTS25Adapter(SimpleNamespace(server=server))
    request = SimpleNamespace(
        input="hello",
        voice="alloy",
        ref_audio=object(),
        max_new_tokens=None,
        extra_params=None,
        speed=speed,
    )
    return adapter, request


def test_indextts25_uses_native_speed_control_duration_factor():
    adapter, fast_request = _indextts25_adapter_and_request(speed=2.0)
    _, slow_request = _indextts25_adapter_and_request(speed=0.5)

    assert adapter.native_speed_control is True
    assert adapter.validate(fast_request) is None
    assert adapter.validate(slow_request) is None
    assert asyncio.run(adapter._build_params(fast_request))["duration_factor"] == [0.5]
    assert asyncio.run(adapter._build_params(slow_request))["duration_factor"] == [2.0]


@pytest.mark.parametrize("speed", [0.49, 2.01])
def test_indextts25_rejects_out_of_range_native_speed(speed):
    adapter, request = _indextts25_adapter_and_request(speed=speed)

    assert adapter.validate(request) == "IndexTTS 2.5 speed must be between 0.5 and 2.0"


def test_indextts25_speed_does_not_change_conditioning_cache_salt():
    adapter, fast_request = _indextts25_adapter_and_request(speed=2.0)
    _, slow_request = _indextts25_adapter_and_request(speed=0.5)
    slow_request.ref_audio = fast_request.ref_audio

    fast_params = asyncio.run(adapter._build_params(fast_request))
    slow_params = asyncio.run(adapter._build_params(slow_request))

    assert indextts2_conditioning_cache_salt(
        fast_request,
        fast_params,
    ) == indextts2_conditioning_cache_salt(slow_request, slow_params)


def test_indextts2_conditioning_cache_salt_changes_with_ref_audio_cache_key():
    request = SimpleNamespace(input="hello", ref_audio="file:///data/spk.wav")
    salt_a = indextts2_conditioning_cache_salt(request, {"ref_audio_cache_key": ["key_aaa"]})
    salt_b = indextts2_conditioning_cache_salt(request, {"ref_audio_cache_key": ["key_bbb"]})
    assert salt_a != salt_b


@pytest.mark.parametrize(
    ("hf_config", "expected_tokenizer_file"),
    [
        (SimpleNamespace(tokenizer_file="custom-tokenizer.tiktoken"), "custom-tokenizer.tiktoken"),
        (SimpleNamespace(), INDEXTTS25_TOKENIZER_FILE),
    ],
)
def test_indextts25_build_uses_configured_tokenizer_file(
    monkeypatch,
    hf_config,
    expected_tokenizer_file,
):
    captured = {}

    def fake_estimate(*args, **kwargs):
        captured.update(kwargs)
        return 4

    async def fake_build_params(request):
        return {"lang": ["en"], "text_normalization": [True]}

    monkeypatch.setattr(
        prompt_utils,
        "estimate_indextts2_prefill_prompt_len",
        fake_estimate,
    )
    server = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(
                model="/model",
                hf_config=hf_config,
            )
        )
    )
    adapter = IndexTTS25Adapter(SimpleNamespace(server=server))
    monkeypatch.setattr(adapter, "_build_params", fake_build_params)
    request = SimpleNamespace(input="hello", ref_audio=None)

    prepared = asyncio.run(adapter.build(request, [], False))

    assert prepared.prompt["prompt_token_ids"] == [1] * 4
    assert captured["tokenizer_file"] == expected_tokenizer_file


def test_diffusion_adapter_extra_body_params_fallback():
    class _DiffAdapter(DiffusionTTSAdapter):
        name = "diff_probe"

        async def build(self, request, sampling_params_list):  # pragma: no cover
            raise NotImplementedError

    assert _DiffAdapter.extra_body_params() == frozenset()


def _higgs_v2_adapter() -> HiggsAudioV2Adapter:
    server = SimpleNamespace(
        _apply_uploaded_speaker=lambda request: None,
        uploaded_speakers={},
    )
    return HiggsAudioV2Adapter(SimpleNamespace(server=server))


def _higgs_v2_request(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "input": "Hello world.",
        "ref_audio": None,
        "ref_text": None,
        "voice": None,
        "x_vector_only_mode": None,
        "speaker_embedding": None,
        "instructions": None,
        "task_type": None,
        "language": None,
        "speed": None,
        "max_new_tokens": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "overrides, err_substr",
    [
        pytest.param({"ref_audio": "data:audio/wav;base64,AA=="}, "ref_text", id="ref_audio_without_ref_text"),
        pytest.param({"ref_text": "some transcript"}, "ref_audio", id="ref_text_without_ref_audio"),
        pytest.param({"task_type": "Base"}, "task_type", id="task_type"),
        pytest.param({"language": "Chinese"}, "language", id="language_override"),
        pytest.param({"input": "[SPEAKER0] hi"}, "multi-speaker", id="multi_speaker_tag"),
        pytest.param({"input": "   "}, "empty", id="input_whitespace_only"),
    ],
)
def test_higgs_audio_v2_validate_rejects_out_of_scope_fields(overrides: dict[str, object], err_substr: str) -> None:
    """Adapter-only policy checks formerly covered by invalid_param e2e on a live V2 server."""
    adapter = _higgs_v2_adapter()
    err = adapter.validate(_higgs_v2_request(**overrides))
    assert err is not None
    assert err_substr.lower() in err.lower()


def test_higgs_audio_v2_validate_accepts_plain_text_and_paired_clone() -> None:
    adapter = _higgs_v2_adapter()
    assert adapter.validate(_higgs_v2_request()) is None
    assert (
        adapter.validate(_higgs_v2_request(ref_audio="data:audio/wav;base64,AA==", ref_text="some transcript")) is None
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
