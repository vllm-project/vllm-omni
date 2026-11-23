# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frontend rejection of unusable Qwen3-TTS Base requests (#3707, #3708)."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.engine.core_client import _with_packed_prompt
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine, StageRuntimeInfo
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import _ref_audio_wire_waveform
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_FUSED_PIPELINE, QWEN3_TTS_PIPELINE
from vllm_omni.model_executor.models.qwen3_tts.prompt_checks import _overlay, check_tts_update, check_voice_clone_prompt
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_REF_CODE_KEY,
    PRECOMPUTED_REF_IDS_KEY,
    REF_AUDIO_CACHE_KEY,
    Qwen3TTSPromptEmbedsBuilder,
)
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NO_REF = "requires 'ref_audio' or 'speaker_embedding'"
NO_TEXT = "requires non-empty 'ref_text'"
NOT_INTACT = "does not reach the engine intact"
AUDIO = ["ref.wav"]
TEXT = ["Reference words."]
SPK = [0.1] * 8
CODE = torch.zeros(4, 16, dtype=torch.long)
WAV = [0.0] * 1024  # The Talker's shortest usable reference waveform.
IDS = torch.arange(100, 112).reshape(1, 12)


def _prompt(**info):
    return {"prompt_token_ids": [0] * 4, "additional_information": {"text": ["Hi there."], **info}}


def _base(**info):
    return _prompt(task_type=["Base"], **info)


def _buffered(prompt, **buffer):
    return {**prompt, "model_intermediate_buffer": buffer}


VCP_SPK = [{"ref_spk_embedding": SPK}]
CACHED = {REF_AUDIO_CACHE_KEY: ["ref-artifacts"]}


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
        # The Talker takes the speaker embedding from ref_audio when the prompt has none.
        pytest.param(
            _base(codes={PRECOMPUTED_REF_CODE_KEY: CODE}, ref_text=TEXT), NO_REF, id="precomputed-ref-code-only"
        ),
        pytest.param(
            _base(codes={PRECOMPUTED_REF_CODE_KEY: torch.empty((0, 16))}, voice_clone_prompt=VCP_SPK, ref_text=TEXT),
            NO_REF,
            id="empty-precomputed-ref-code",
        ),
        pytest.param(
            _base(codes={PRECOMPUTED_REF_CODE_KEY: CODE.tolist()}, voice_clone_prompt=VCP_SPK, ref_text=TEXT),
            NO_REF,
            id="flat-list-precomputed-ref-code",
        ),
        pytest.param(_base(**{REF_AUDIO_CACHE_KEY: [""]}, ref_text=TEXT), NO_REF, id="empty-cache-key"),
        # The Talker's cached ref-audio artifacts hold no transcript.
        pytest.param(_base(**CACHED), NO_TEXT, id="cache-key-no-ref-text"),
        pytest.param(
            _base(codes={PRECOMPUTED_REF_CODE_KEY: CODE}, voice_clone_prompt=VCP_SPK),
            NO_TEXT,
            id="precomputed-ref-code-no-ref-text",
        ),
        # Model Runner V2 serves the request from the buffer laid over additional_information.
        pytest.param(
            _buffered(_base(ref_audio=AUDIO, ref_text=TEXT), ref_audio=None), NO_REF, id="buffer-drops-ref-audio"
        ),
        pytest.param(
            _buffered(_base(ref_audio=AUDIO, ref_text=TEXT), ref_text=["  "]), NO_TEXT, id="buffer-blanks-ref-text"
        ),
        pytest.param(
            _buffered(_base(ref_audio=AUDIO, x_vector_only_mode=[True]), x_vector_only_mode=[False]),
            NO_TEXT,
            id="buffer-turns-off-xvec",
        ),
        # A plain streaming update replaces the buffer, so the reference must not live there alone.
        pytest.param(_buffered(_base(ref_text=TEXT), ref_audio=AUDIO), NO_REF, id="buffer-only-ref-audio"),
        pytest.param(
            _buffered(_base(ref_audio=AUDIO, ref_text=TEXT), ref_audio=[(np.zeros(1024, dtype=np.float32), 24000)]),
            NOT_INTACT,
            id="buffer-array-ref-audio",
        ),
        # The buffer travels as a plain dict: even a top-level tensor arrives as a raw buffer list.
        pytest.param(
            _buffered(_base(ref_audio=AUDIO, ref_text=TEXT), codes={PRECOMPUTED_REF_CODE_KEY: CODE}),
            NOT_INTACT,
            id="buffer-tensor-ref-code",
        ),
        pytest.param(
            _buffered(_base(ref_audio=AUDIO, ref_text=TEXT), ref_ids=IDS), NOT_INTACT, id="buffer-tensor-ref-ids"
        ),
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
        pytest.param(
            _base(codes={PRECOMPUTED_REF_CODE_KEY: CODE.numpy()}, voice_clone_prompt=VCP_SPK, ref_text=TEXT),
            NOT_INTACT,
            id="array-precomputed-ref-code",
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
    pytest.param(_base(ref_audio=[[WAV, 24000]], ref_text=TEXT), id="plain-list-audio"),
    # Shapes the speech API's _build_tts_params sends.
    pytest.param(
        _base(ref_audio=[[_ref_audio_wire_waveform(np.asarray(WAV, dtype=np.float32)), 24000]], ref_text=TEXT),
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
    # Offline shapes the Talker serves without ref_audio.
    pytest.param(
        _base(codes={PRECOMPUTED_REF_CODE_KEY: CODE}, voice_clone_prompt=VCP_SPK, ref_text=TEXT),
        id="icl-precomputed-ref-code",
    ),
    pytest.param(
        # The Talker unwraps one list level, as it does for ref ids.
        _base(codes={PRECOMPUTED_REF_CODE_KEY: [CODE.tolist()]}, voice_clone_prompt=VCP_SPK, ref_text=TEXT),
        id="icl-list-precomputed-ref-code",
    ),
    pytest.param(_base(**CACHED, ref_text=TEXT), id="icl-cached-artifacts"),
    pytest.param(_base(**CACHED, x_vector_only_mode=[True]), id="xvec-cached-artifacts"),
    pytest.param(
        _buffered(_base(ref_audio=AUDIO, ref_text=TEXT), ref_text=["Other words."], codes={"audio": CODE}),
        id="buffer-keeps-reference",
    ),
    # Both runners serve these as their merged task; a plain streaming update falls back to the info's.
    pytest.param(_buffered(_prompt(), task_type=["Base"], ref_audio=AUDIO, ref_text=TEXT), id="buffer-sets-base-task"),
    pytest.param(
        _buffered(
            _base(ref_audio=AUDIO, x_vector_only_mode=[True]),
            task_type=["CustomVoice"],
            speaker=["vivian"],
            x_vector_only_mode=[False],
        ),
        id="buffer-sets-custom-voice-task",
    ),
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


def _ipc_request(info, buffer=None):
    # The engine-core path: serialize, pack, then the msgpack round trip of the core request.
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
        model_intermediate_buffer=buffer,
    )
    # v0.34 sends prompt token ids as a packed int32 ndarray (the decoder cannot
    # materialize a plain list into it); the real clients pack via
    # `_with_packed_prompt` in `add_request(_async)`.
    packed = _with_packed_prompt(request)
    return MsgpackDecoder(OmniEngineCoreRequest).decode(MsgpackEncoder().encode(packed))


def _ipc(info):
    return deserialize_additional_information(_ipc_request(info).additional_information)


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


@pytest.mark.parametrize("prompt", _USABLE_BASE)
def test_accepted_base_prompts_reach_the_engine_intact(prompt):
    info = prompt["additional_information"]
    received = _ipc(info)
    assert received.keys() == info.keys()
    for key, value in info.items():
        assert _same(value, received[key]), key
    check_voice_clone_prompt({**prompt, "additional_information": received}, [])
    ref_audio = received.get("ref_audio")
    if ref_audio is not None and not isinstance(ref_audio[0], str):
        # The Talker decodes an inline waveform without loading anything.
        object.__new__(Qwen3TTSPromptEmbedsBuilder).normalize_ref_audio(ref_audio[0])


def _mrv2_view(info, buffer):
    # The Talker's view after the real engine-core round trip and Model Runner V2's add_request.
    received = _ipc_request(info, buffer)
    mrv2 = OmniIntermediateBuffer(1)
    new_req = SimpleNamespace(
        req_id="r",
        prompt_embeds=None,
        additional_information=received.additional_information,
        model_intermediate_buffer=received.model_intermediate_buffer,
        mm_features=None,
        sampling_params=None,
    )
    mrv2.add_request(0, new_req)
    return received, {k: v for k, v in mrv2.buffers[0].items() if k != "req_id"}


def _passes(info):
    try:
        check_voice_clone_prompt({"additional_information": info}, [])
        return True
    except ValueError:
        return False


_BUFFERS = {
    "drops-ref-audio": {"ref_audio": None},
    "blanks-ref-text": {"ref_text": [""]},
    "new-ref-text": {"ref_text": ["Other words."]},
    "turns-off-xvec": {"x_vector_only_mode": [False]},
    "merges-codes": {"codes": {"audio": CODE.tolist()}},
    "replaces-vcp": {"voice_clone_prompt": {"icl_mode": True}},
    "tensor-ref-code": {"codes": {PRECOMPUTED_REF_CODE_KEY: CODE}},
    "tensor-ref-ids": {PRECOMPUTED_REF_IDS_KEY: IDS},
}


@pytest.mark.parametrize("buffer", list(_BUFFERS.values()), ids=list(_BUFFERS))
@pytest.mark.parametrize(
    "info",
    [
        {"ref_audio": AUDIO, "ref_text": TEXT},
        {"ref_audio": AUDIO, "x_vector_only_mode": [True]},
        {"voice_clone_prompt": VCP_SPK, "x_vector_only_mode": [True]},
        {"codes": {PRECOMPUTED_REF_CODE_KEY: CODE}, "voice_clone_prompt": VCP_SPK, "ref_text": TEXT},
    ],
    ids=["icl", "xvec", "xvec-embedding", "icl-precomputed-ref-code"],
)
def test_buffer_check_matches_the_mrv2_talker_view(info, buffer):
    # After the real engine-core round trip, Model Runner V2 builds the overlaid view, and the check
    # accepts a buffered request only when that view still passes.
    info = _base(**info)["additional_information"]
    check_voice_clone_prompt({"additional_information": info}, [])
    try:
        check_voice_clone_prompt({"additional_information": info, "model_intermediate_buffer": buffer}, [])
        accepted = True
    except ValueError:
        accepted = False
    received, talker_info = _mrv2_view(info, buffer)
    received_info = deserialize_additional_information(received.additional_information)
    assert _same(_overlay(received_info, received.model_intermediate_buffer), talker_info)
    served = _passes(talker_info)
    # A buffer tensor arrives as a raw buffer list, so the value the caller meant is lost even when
    # the view still has another reference: always rejected.
    values = [v for section in buffer.values() for v in (section.values() if isinstance(section, dict) else [section])]
    assert accepted == (served and not any(isinstance(v, torch.Tensor) for v in values))


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
        # Plain updates carry later chunks of a request already checked on submission.
        (_base(), "streaming_update", RuntimeError),
        # ... but still may not blank its text,
        (_buffered(_base(), text=["  "]), "streaming_update", ValueError),
        # and an update that replaces the prompt is checked like a new request.
        (_base(meta={"replace_streaming_prompt": True}), "streaming_update", ValueError),
        (_buffered(_base(), meta={"replace_streaming_prompt": True}), "streaming_update", ValueError),
        (
            _base(ref_audio=AUDIO, ref_text=TEXT, meta={"replace_streaming_prompt": True}),
            "streaming_update",
            RuntimeError,
        ),
    ],
    ids=[
        "invalid-request",
        "valid-request",
        "end-of-stream-update",
        "plain-update",
        "plain-update-blanks-text",
        "replace-update-without-reference",
        "buffer-replace-update-without-reference",
        "replace-update-with-reference",
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


_FIRST_CHUNKS: dict[str, dict[str, Any]] = {
    "icl": {"ref_audio": AUDIO, "ref_text": TEXT},
    "xvec": {"ref_audio": AUDIO, "x_vector_only_mode": [True]},
    "xvec-embedding": {"voice_clone_prompt": VCP_SPK, "x_vector_only_mode": [True]},
    "icl-precomputed-ref-code": {
        "codes": {PRECOMPUTED_REF_CODE_KEY: CODE},
        "voice_clone_prompt": VCP_SPK,
        "ref_text": TEXT,
    },
    "icl-cached-artifacts": {**CACHED, "ref_text": TEXT},
    "icl-ref-ids": {"ref_audio": AUDIO, "ref_ids": [IDS.tolist()[0]]},
    "icl-precomputed-ref-ids": {"ref_audio": AUDIO, PRECOMPUTED_REF_IDS_KEY: IDS},
    "stored-speaker": {"speaker": ["stored"]},
}
_UPDATE_BUFFERS = {
    **_BUFFERS,
    "empty-ref-audio": {"ref_audio": []},
    "blank-ref-audio": {"ref_audio": ["  "]},
    "array-ref-audio": {"ref_audio": [np.zeros(1024, dtype=np.float32)]},
    "turns-off-xvec-with-ref-text": {"x_vector_only_mode": [False], "ref_text": TEXT},
    "resends-reference": {"ref_audio": AUDIO, "ref_text": TEXT},
    "turns-on-xvec": {"x_vector_only_mode": [True]},
    "drops-cached-artifacts": {REF_AUDIO_CACHE_KEY: None},
    "drops-ref-ids": {"ref_ids": None},
    "drops-precomputed-ref-ids": {PRECOMPUTED_REF_IDS_KEY: None},
    "drops-speaker": {"speaker": None},
    "empty-speaker": {"speaker": []},
    "blanks-speaker": {"speaker": [""]},
    "resends-embedding-reference": {"voice_clone_prompt": VCP_SPK, "x_vector_only_mode": [True]},
    "next-text": {"text": ["Next words."]},
    # Needs the first chunk's task, which the caller cannot see: left to a follow-up.
    "only-sets-base-task": {"task_type": ["Base"]},
}
_UPDATE_REJECTS = {
    "drops-ref-audio": NO_REF,
    "empty-ref-audio": NO_REF,
    "blank-ref-audio": NO_REF,
    "blanks-ref-text": NO_TEXT,
    "turns-off-xvec": NO_TEXT,
    "replaces-vcp": NO_TEXT,
    # Over an embedding-only voice, in-context mode also needs a reference code.
    "turns-off-xvec-with-ref-text": NO_REF,
    "drops-cached-artifacts": NO_REF,
    "drops-ref-ids": NO_TEXT,
    "drops-precomputed-ref-ids": NO_TEXT,
    "drops-speaker": NO_REF,
    "empty-speaker": NO_REF,
    "blanks-speaker": NO_REF,
    "array-ref-audio": NOT_INTACT,
    "tensor-ref-code": NOT_INTACT,
    "tensor-ref-ids": NOT_INTACT,
}


@pytest.mark.parametrize("name", list(_UPDATE_BUFFERS))
def test_plain_update_buffer_must_not_break_a_base_request(name):
    # A plain update keeps the first chunk's additional_information and Model Runner V2 lays the
    # buffer over it, so the update may re-send a usable reference but not clear one.
    update = {"prompt_token_ids": [0], "model_intermediate_buffer": _UPDATE_BUFFERS[name]}
    if name in _UPDATE_REJECTS:
        with pytest.raises(ValueError, match=_UPDATE_REJECTS[name]):
            check_tts_update(update, [])
    else:
        assert check_tts_update(update, []) is update


@pytest.mark.parametrize("name", [name for name in _UPDATE_BUFFERS if name not in _UPDATE_REJECTS])
@pytest.mark.parametrize("first", list(_FIRST_CHUNKS))
def test_accepted_plain_update_keeps_a_base_request_servable(first, name):
    # The Talker's view of a valid first chunk with an accepted update buffer still passes.
    info = _base(**_FIRST_CHUNKS[first])["additional_information"]
    buffer = _UPDATE_BUFFERS[name]
    assert _passes(_mrv2_view(info, buffer)[1])
