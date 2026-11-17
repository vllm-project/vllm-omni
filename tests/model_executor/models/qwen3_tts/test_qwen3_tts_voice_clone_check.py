# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frontend rejection of unusable Qwen3-TTS Base requests (#3707, #3708)."""

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine, StageRuntimeInfo
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import _ref_audio_wire_waveform
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_FUSED_PIPELINE, QWEN3_TTS_PIPELINE
from vllm_omni.model_executor.models.qwen3_tts.prompt_checks import check_voice_clone_prompt
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import PRECOMPUTED_REF_IDS_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NO_REF = "requires 'ref_audio' or 'speaker_embedding'"
NO_TEXT = "requires non-empty 'ref_text'"
NOT_INTACT = "does not reach the engine intact"
AUDIO = ["ref.wav"]
TEXT = ["Reference words."]
SPK = [0.1] * 8
CODE = torch.zeros(4, 16, dtype=torch.long)
IDS = torch.arange(100, 112).reshape(1, 12)


def _prompt(**info):
    return {"prompt_token_ids": [0] * 4, "additional_information": {"text": ["Hi there."], **info}}


def _base(**info):
    return _prompt(task_type=["Base"], **info)


@pytest.mark.parametrize(
    "prompt, error",
    [
        pytest.param(_base(), NO_REF, id="no-reference"),
        pytest.param(_base(ref_audio=[], ref_text=TEXT), NO_REF, id="empty-ref-audio"),
        pytest.param(_base(ref_audio=[None], ref_text=TEXT), NO_REF, id="none-ref-audio"),
        pytest.param(_base(ref_audio=["  "], ref_text=TEXT), NO_REF, id="blank-ref-audio"),
        pytest.param(_base(ref_audio="ref.wav", ref_text=TEXT), NO_REF, id="bare-string-ref-audio"),
        pytest.param(_base(x_vector_only_mode=[True]), NO_REF, id="xvec-no-reference"),
        pytest.param(
            _base(voice_clone_prompt=[{"ref_spk_embedding": SPK}], ref_text=TEXT), NO_REF, id="icl-embedding-only"
        ),
        pytest.param(
            _base(voice_clone_prompt=[{"ref_code": CODE.tolist()}], ref_text=TEXT), NO_REF, id="ref-code-only"
        ),
        pytest.param(
            # The Talker reads only a tensor/array ref_code from voice_clone_prompt.
            _base(voice_clone_prompt=[{"ref_code": CODE.tolist(), "ref_spk_embedding": SPK}], ref_text=TEXT),
            NO_REF,
            id="list-ref-code",
        ),
        pytest.param(_base(speaker=[""], ref_text=TEXT), NO_REF, id="blank-speaker"),
        pytest.param(_base(ref_audio=AUDIO), NO_TEXT, id="no-ref-text"),
        pytest.param(_base(ref_audio=AUDIO, ref_text=[""]), NO_TEXT, id="empty-ref-text"),
        pytest.param(_base(ref_audio=AUDIO, ref_text=["  \t"]), NO_TEXT, id="blank-ref-text"),
        pytest.param(_base(ref_audio=AUDIO, x_vector_only_mode=[False]), NO_TEXT, id="explicit-icl"),
        # The Talker unwraps one list level, so flat ids fall back to x-vector mode.
        pytest.param(_base(ref_audio=AUDIO, ref_ids=[1, 2, 3]), NO_TEXT, id="flat-ref-ids"),
        # The Talker reads the transcript as ids[:, 3:-2] of one request.
        *(
            pytest.param(_base(ref_audio=AUDIO, **{key: ids}), NO_TEXT, id=f"{name}-{key}")
            for key in (PRECOMPUTED_REF_IDS_KEY, "ref_ids")
            for name, ids in (
                ("empty", torch.empty((1, 0), dtype=torch.long)),
                ("four-ids", IDS[:, :4]),
                ("two-requests", torch.arange(24).reshape(2, 12)),
            )
        ),
        # Only a top-level tensor survives the engine-core msgpack round trip; the rest arrive as raw buffers.
        *(
            pytest.param(_base(ref_audio=AUDIO, **{key: ids}), NOT_INTACT, id=f"{name}-{key}")
            for key in (PRECOMPUTED_REF_IDS_KEY, "ref_ids")
            for name, ids in (("str-array", np.array(["invalid"])), ("array", IDS.numpy()), ("listed-tensor", [IDS]))
        ),
        pytest.param(_base(ref_audio=AUDIO, voice_clone_prompt={"ref_ids": [IDS]}), NOT_INTACT, id="vcp-ref-ids"),
        pytest.param(_base(ref_audio=AUDIO, voice_clone_prompt={"ref_id": IDS}), NOT_INTACT, id="vcp-ref-id"),
        pytest.param(
            _base(voice_clone_prompt=[{"ref_code": CODE, "ref_spk_embedding": SPK, "ref_text": "Words."}]),
            NOT_INTACT,
            id="vcp-tensor-ref-code",
        ),
        pytest.param(
            _base(voice_clone_prompt=[{"ref_spk_embedding": torch.ones(8)}], x_vector_only_mode=[True]),
            NOT_INTACT,
            id="vcp-tensor-embedding",
        ),
        pytest.param(
            _base(ref_audio=[(np.zeros(16000, dtype=np.float32), 16000)], ref_text=TEXT),
            NOT_INTACT,
            id="array-ref-audio",
        ),
        pytest.param(_base(ref_audio=AUDIO, speaker=["clone"]), NO_TEXT, id="issue-3708-shape"),
        pytest.param(
            _base(
                ref_audio=AUDIO,
                x_vector_only_mode=[True],
                voice_clone_prompt=[{"ref_spk_embedding": SPK, "icl_mode": True}],
            ),
            NO_TEXT,
            id="icl-mode-overrides-xvec",
        ),
        pytest.param([_base(ref_audio=AUDIO, ref_text=TEXT), _base()], NO_REF, id="one-bad-in-list"),
    ],
)
def test_rejects_unusable_base_request(prompt, error):
    with pytest.raises(ValueError, match=error):
        check_voice_clone_prompt(prompt, [])


_USABLE_BASE = [
    pytest.param(_base(ref_audio=AUDIO, ref_text=TEXT), id="icl-list-ref-text"),
    pytest.param(_base(ref_audio=AUDIO, ref_text="Reference words."), id="icl-scalar-ref-text"),
    pytest.param(_base(ref_audio=AUDIO, x_vector_only_mode=[True]), id="xvec-ref-audio"),
    pytest.param(_base(ref_audio=AUDIO, ref_ids=[list(range(100, 106))]), id="ref-ids"),
    pytest.param(_base(ref_audio=AUDIO, ref_ids=IDS), id="tensor-ref-ids"),
    pytest.param(_base(ref_audio=AUDIO, **{PRECOMPUTED_REF_IDS_KEY: IDS}), id="precomputed-ref-ids"),
    pytest.param(_base(ref_audio=AUDIO, voice_clone_prompt=[{"ref_ids": IDS.tolist()}]), id="vcp-ref-ids"),
    # Intentionally passed to the Talker: it resolves a stored voice from a cache this check cannot see.
    pytest.param(_base(speaker=["stored"]), id="stored-speaker-no-ref-left-to-talker"),
    # Shapes the speech API's _build_tts_params sends.
    pytest.param(_base(ref_audio=[[[0.0, 0.1], 24000]], ref_text=TEXT), id="api-inline-audio"),
    pytest.param(
        _base(ref_audio=[[_ref_audio_wire_waveform(np.zeros(4, dtype=np.float32)), 24000]], ref_text=TEXT),
        id="api-wire-audio",
    ),
    pytest.param(_base(voice_clone_prompt=[{"ref_spk_embedding": SPK}], x_vector_only_mode=[True]), id="api-embedding"),
    pytest.param(
        _base(speaker=["up"], voice_clone_prompt=[{"ref_spk_embedding": SPK}], x_vector_only_mode=[True]),
        id="api-uploaded-embedding",
    ),
    pytest.param(
        _base(speaker=["up"], ref_audio=["data:audio/wav;base64,AA"], ref_text=TEXT, x_vector_only_mode=[False]),
        id="api-uploaded-audio",
    ),
    pytest.param(_base(speaker=["pre"], x_vector_only_mode=[False], ref_text=TEXT), id="api-precomputed"),
]
# Not Base, or not a request with additional_information: never checked.
_UNCHECKED = [
    pytest.param(_prompt(task_type=["CustomVoice"], speaker=["vivian"]), id="custom-voice"),
    pytest.param(_prompt(task_type=["VoiceDesign"], instruct=["calm"]), id="voice-design"),
    pytest.param(_prompt(ref_audio=AUDIO), id="no-task-type"),
    pytest.param({"prompt_token_ids": [0]}, id="streaming-close"),
    pytest.param("plain text", id="string"),
    pytest.param({"prompt_token_ids": [0, 0], "additional_information": {"text": ["more"]}}, id="text-chunk"),
]


@pytest.mark.parametrize("prompt", _USABLE_BASE + _UNCHECKED)
def test_passes_usable_or_unchecked_prompt(prompt):
    assert check_voice_clone_prompt(prompt, []) is prompt


def _ipc(info):
    # The engine-core path: serialize, then the msgpack round trip of the core request.
    request = OmniEngineCoreRequest(
        request_id="r",
        prompt_token_ids=[0],
        mm_features=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        additional_information=serialize_additional_information(info),
    )
    decoded = MsgpackDecoder(OmniEngineCoreRequest).decode(MsgpackEncoder().encode(request))
    return deserialize_additional_information(decoded.additional_information)


@pytest.mark.parametrize("prompt", _USABLE_BASE)
def test_accepted_base_prompts_reach_the_engine_intact(prompt):
    info = prompt["additional_information"]
    received = _ipc(info)
    assert received.keys() == info.keys()
    for key, value in info.items():
        assert torch.equal(received[key], value) if isinstance(value, torch.Tensor) else received[key] == value, key
    check_voice_clone_prompt({**prompt, "additional_information": received}, [])


@pytest.mark.parametrize("pipeline", [QWEN3_TTS_PIPELINE, QWEN3_TTS_FUSED_PIPELINE], ids=lambda p: p.model_type)
def test_pipelines_check_voice_clone_at_stage0(pipeline):
    hook = resolve_obj_by_qualname(pipeline.stages[0].prompt_transform_func)
    with pytest.raises(ValueError, match=NO_REF):
        hook(_base(), [])
    # The text check still runs first.
    with pytest.raises(ValueError, match="first item is non-blank"):
        hook(_base(text=["  "]), [])


@pytest.mark.parametrize(
    "prompt, message_type, error",
    [
        (_base(), "add_request", ValueError),
        (_base(ref_audio=AUDIO, ref_text=TEXT), "add_request", RuntimeError),
        ({"prompt_token_ids": [0]}, "streaming_update", RuntimeError),
        # Updates carry later chunks of a request already checked on submission.
        (_base(), "streaming_update", RuntimeError),
    ],
)
def test_engine_rejects_before_input_processing(mocker: MockerFixture, prompt, message_type, error):
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("speech",)
    engine.input_processor = mocker.Mock()
    engine.input_processor.process_inputs.side_effect = RuntimeError("reached")
    engine.prompt_transform_func = resolve_obj_by_qualname(QWEN3_TTS_PIPELINE.stages[0].prompt_transform_func)

    with pytest.raises(error):
        engine._build_add_request_message(
            request_id="req-1",
            prompt=prompt,
            sampling_params_list=[params],
            resumable=True,
            message_type=message_type,
        )
    assert engine.input_processor.process_inputs.called is (error is RuntimeError)
